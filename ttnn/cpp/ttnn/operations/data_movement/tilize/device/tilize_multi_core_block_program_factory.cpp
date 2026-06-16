// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_multi_core_block_program_factory.hpp"

#include <filesystem>
#include <string>
#include <vector>

#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// One per-core-group bundle of DFBs + kernels + WorkUnitSpec. The legacy factory created a
// separate (c_1, c_0, c_16) CBDescriptor triple per core range — and crucially the entry counts
// differ between groups (single_sub_block_size vs single_block_size_cliff_row). A Metal 2.0
// DataflowBufferSpec's num_entries is a single per-node template value, so each group needs its
// OWN DFB triple, its own reader/writer/compute KernelSpecs (binding that group's DFBs), and its
// own WorkUnitSpec — the Local-DFB rule (producer+consumer co-located) is satisfied because all
// three kernels of a group share that group's WorkUnitSpec on the group's cores.
struct BlockGroup {
    std::string suffix;       // unique-id suffix (e.g. "full", "cliff_row")
    CoreRangeSet cores;       // this group's core ranges
    uint32_t num_tiles;       // CB/DFB entry count for c_0/c_16 (legacy push_cb_pair num_tiles)
    uint32_t block_size_col;  // compute CTA 0
    uint32_t block_size_row;  // compute CTA 1
};

}  // namespace

// Metal 2.0 program factory: builds the immutable ProgramSpec and its mutable ProgramRunArgs.
// Behavior-preserving port of the legacy ProgramDescriptor multi-core block (WH) tilize factory.
ttnn::device_operation::ProgramArtifacts TilizeMultiCoreBlockProgramFactory::create_program_spec(
    const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

    IDevice* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid = sub_core_grids.has_value() ? sub_core_grids.value() : default_grid;

    uint32_t max_l1_size = operations::data_movement::get_max_l1_space(a);
    uint32_t num_tiles_per_col = output.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t num_tiles_per_row = output.padded_shape()[-1] / TILE_WIDTH;

    uint32_t num_blocks = (output.padded_shape()[-1] * output.padded_shape()[-2]) / (TILE_HEIGHT * TILE_WIDTH);
    uint32_t cb_block_size_limit = max_l1_size / (input_single_tile_size + output_single_tile_size);

    auto
        [ncores,
         all_cores,
         core_range,
         cliff_row_core_range,
         cliff_col_core_range,
         cliff_col_row_core_range,
         nblocks_per_core,
         single_block_size,
         single_block_size_cliff_row,
         single_block_size_cliff_col,
         has_cliff_row,
         has_cliff_col,
         full_cores_per_row,
         full_cores_per_col,
         single_sub_block_size] =
            ttnn::split_blocks_for_tilize_wh(
                available_grid, num_blocks, num_tiles_per_row, num_tiles_per_col, cb_block_size_limit);

    if (single_sub_block_size > 0 && single_block_size % single_sub_block_size) {
        TT_FATAL(false, "single_block_size is not divided by single_sub_block_size");
    }

    uint32_t total_tiles_per_row =
        (full_cores_per_row * single_block_size) + (has_cliff_row * single_block_size_cliff_row);

    uint32_t row_size_bytes = a.padded_shape()[-1] * a.element_size();  // Assuming bfloat16 dataformat

    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();

    // reader CT derivations (identical to legacy)
    uint32_t num_tiles_2d = output.padded_shape()[-1] * output.padded_shape()[-2] / TILE_HW;

    auto log_shape = output.logical_shape();
    uint32_t third_dim = 1;
    if (log_shape.rank() == 3) {
        third_dim = log_shape[-3];
    } else if (log_shape.rank() >= 4) {
        third_dim = log_shape[-3] * log_shape[-4];
    }

    uint32_t tile_height = output.tensor_spec().tile().get_height();

    uint32_t total_num_rows = a.logical_shape()[-2];

    if (output.padded_shape()[-2] > tt::round_up(total_num_rows, tile_height)) {
        total_num_rows = output.padded_shape()[-2];
    }

    // compute CT derivations (identical to legacy)
    uint32_t single_sub_block_wh = single_block_size * single_block_size / single_sub_block_size;
    uint32_t single_sub_block_cliff_col_wh = single_block_size_cliff_col * single_block_size / single_sub_block_size;

    // ---- Per-core-group bundles ----
    // Mirrors the legacy push_cb_pair / make_compute_kernel calls one-for-one. Each group carries
    // the CB entry count (num_tiles) and the compute CTAs (block_size_col/row) it used legacy-side.
    std::vector<BlockGroup> groups;
    if (!core_range.empty()) {
        groups.push_back({"full", core_range, single_sub_block_size, single_sub_block_wh, single_sub_block_size});
    }
    if (has_cliff_col && has_cliff_row) {
        groups.push_back(
            {"cliff_col_row",
             cliff_col_row_core_range,
             single_block_size_cliff_row,
             single_block_size_cliff_col,
             single_block_size_cliff_row});
    }
    if (has_cliff_row) {
        groups.push_back(
            {"cliff_row",
             cliff_row_core_range,
             single_block_size_cliff_row,
             single_block_size,
             single_block_size_cliff_row});
    }
    if (has_cliff_col) {
        groups.push_back(
            {"cliff_col",
             cliff_col_core_range,
             single_sub_block_size,
             single_sub_block_cliff_col_wh,
             single_sub_block_size});
    }

    // ---- ProgramSpec (immutable) ----
    m2::ProgramSpec spec;
    spec.name = "tilize_multi_core_block";

    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = a.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output.tensor_spec()},
    };

    const std::filesystem::path reader_src{
        "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
        "reader_unary_pad_multicore_both_dims_m2.cpp"};
    const std::filesystem::path writer_src{
        "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id_wh_m2.cpp"};
    const std::filesystem::path compute_src{
        "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_wh_m2.cpp"};

    std::vector<m2::WorkUnitSpec> work_units;
    work_units.reserve(groups.size());

    for (const auto& g : groups) {
        const std::string src0_id = "src0_" + g.suffix;
        const std::string temp_id = "temp_" + g.suffix;
        const std::string output_id = "output_" + g.suffix;
        const std::string reader_id = "reader_" + g.suffix;
        const std::string writer_id = "writer_" + g.suffix;
        const std::string compute_id = "compute_" + g.suffix;

        // c_1 temp staging buffer (one entry of temp_cb_size bytes): the reader rounds the DRAM
        // source down to a dram_alignment boundary, reads (row_bytes + dram_alignment) here, then
        // copies the correctly-offset slice into src0. It is used purely as an address source
        // (get_write_ptr) — there is no real FIFO producer/consumer pair — so it is bound as a
        // self-loop DFB on the reader (PRODUCER + CONSUMER, same kernel) to satisfy the validator.
        uint32_t input_row_bytes = input_single_tile_size / TILE_HEIGHT;
        uint32_t temp_cb_size = input_row_bytes * g.num_tiles + 2 * dram_alignment;

        // c_0 src0: row-major sticks the reader fills, the compute tilizes.
        // c_16 output: tilized output the compute produces, the writer drains.
        spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{temp_id},
            .entry_size = temp_cb_size,
            .num_entries = 1,
            .data_format_metadata = input_cb_data_format,
        });
        spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{src0_id},
            .entry_size = input_single_tile_size,
            .num_entries = g.num_tiles,
            .data_format_metadata = input_cb_data_format,
        });
        spec.dataflow_buffers.push_back(m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{output_id},
            .entry_size = output_single_tile_size,
            .num_entries = g.num_tiles,
            .data_format_metadata = output_cb_data_format,
        });

        // Reader: produces src0 + self-loops temp. Reads the input tensor (binding).
        // Legacy CTs {total_num_rows, third_dim, tile_height, element_size, row_size_bytes(unpadded_X_size),
        // dram_alignment} + TensorAccessorArgs<6> → named CTAs + ta::input.
        m2::KernelSpec reader{
            .unique_id = m2::KernelSpecName{reader_id},
            .source = reader_src,
            .dfb_bindings =
                {
                    m2::DFBBinding{
                        .dfb_spec_name = m2::DFBSpecName{src0_id},
                        .accessor_name = "src0",
                        .endpoint_type = m2::DFBEndpointType::PRODUCER,
                    },
                    m2::DFBBinding{
                        .dfb_spec_name = m2::DFBSpecName{temp_id},
                        .accessor_name = "temp",
                        .endpoint_type = m2::DFBEndpointType::PRODUCER,
                    },
                    m2::DFBBinding{
                        .dfb_spec_name = m2::DFBSpecName{temp_id},
                        .accessor_name = "temp",
                        .endpoint_type = m2::DFBEndpointType::CONSUMER,
                    },
                },
            .tensor_bindings =
                {
                    m2::TensorBinding{
                        .tensor_parameter_name = m2::TensorParamName{"input"},
                        .accessor_name = "input",
                    },
                },
            .compile_time_args =
                {
                    {"total_num_rows", total_num_rows},
                    {"third_dim", third_dim},
                    {"tile_height", tile_height},
                    {"element_size", a.element_size()},
                    {"unpadded_X_size", row_size_bytes},
                    {"dram_alignment", dram_alignment},
                },
            .runtime_arg_schema =
                {
                    .runtime_arg_names =
                        {"pad_value",
                         "width_size",
                         "start_row_id",
                         "start_column_id",
                         "single_block_size_row_arg",
                         "single_block_size_col_arg",
                         "sub_block_width_size",
                         "single_sub_block_size_row_arg"},
                },
            .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
        };

        // Writer: consumes output. Drains the output DFB into the output tensor (binding).
        // Legacy CTs {output_cb_index, num_tiles_2d, third_dim, total_tiles_per_row} + TensorAccessorArgs<4>
        // → dfb::output binding + named CTAs + ta::output.
        m2::KernelSpec writer{
            .unique_id = m2::KernelSpecName{writer_id},
            .source = writer_src,
            .dfb_bindings =
                {
                    m2::DFBBinding{
                        .dfb_spec_name = m2::DFBSpecName{output_id},
                        .accessor_name = "output",
                        .endpoint_type = m2::DFBEndpointType::CONSUMER,
                    },
                },
            .tensor_bindings =
                {
                    m2::TensorBinding{
                        .tensor_parameter_name = m2::TensorParamName{"output"},
                        .accessor_name = "output",
                    },
                },
            .compile_time_args =
                {
                    {"num_tiles_per_2d", num_tiles_2d},
                    {"third_dim", third_dim},
                    {"total_tiles_per_row", total_tiles_per_row},
                },
            .runtime_arg_schema =
                {
                    .runtime_arg_names = {"start_id", "single_block_size_row_arg", "single_block_size_col_arg"},
                },
            .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
        };

        // Compute: consumes src0, produces output. Legacy CTAs {block_size_col, block_size_row, third_dim}.
        m2::KernelSpec compute{
            .unique_id = m2::KernelSpecName{compute_id},
            .source = compute_src,
            .dfb_bindings =
                {
                    m2::DFBBinding{
                        .dfb_spec_name = m2::DFBSpecName{src0_id},
                        .accessor_name = "src0",
                        .endpoint_type = m2::DFBEndpointType::CONSUMER,
                    },
                    m2::DFBBinding{
                        .dfb_spec_name = m2::DFBSpecName{output_id},
                        .accessor_name = "output",
                        .endpoint_type = m2::DFBEndpointType::PRODUCER,
                    },
                },
            .compile_time_args =
                {
                    {"block_size_col", g.block_size_col},
                    {"block_size_row", g.block_size_row},
                    {"third_dim", third_dim},
                },
            .hw_config =
                m2::ComputeHardwareConfig{
                    .fp32_dest_acc_en = fp32_llk_acc,
                },
        };
        // Legacy set unpack_to_dest_mode[c_0]=UnpackToDestFp32 when fp32_llk_acc; preserve that for
        // this group's src0 DFB (the compute consumer of the fp32 input).
        if (fp32_llk_acc) {
            std::get<m2::ComputeHardwareConfig>(compute.hw_config).unpack_to_dest_mode = {
                {m2::DFBSpecName{src0_id}, UnpackToDestMode::UnpackToDestFp32}};
        }

        spec.kernels.push_back(std::move(reader));
        spec.kernels.push_back(std::move(writer));
        spec.kernels.push_back(std::move(compute));

        work_units.push_back(m2::WorkUnitSpec{
            .name = "wu_" + g.suffix,
            .kernels = {m2::KernelSpecName{reader_id}, m2::KernelSpecName{writer_id}, m2::KernelSpecName{compute_id}},
            .target_nodes = g.cores,
        });
    }

    spec.work_units = std::move(work_units);

    // ---- ProgramRunArgs (mutable) ----
    // Per-core RTAs computed exactly as legacy. Each core is routed to its group's reader/writer
    // KernelRunArgs by checking which group's core set contains it (membership matches the legacy
    // branch that picked the per-core block dimensions).
    m2::ProgramRunArgs run;
    // One reader/writer KernelRunArgs per group, keyed by suffix.
    std::vector<m2::KernelRunArgs> reader_runs;
    std::vector<m2::KernelRunArgs> writer_runs;
    reader_runs.reserve(groups.size());
    writer_runs.reserve(groups.size());
    for (const auto& g : groups) {
        reader_runs.push_back(m2::KernelRunArgs{.kernel = m2::KernelSpecName{"reader_" + g.suffix}});
        writer_runs.push_back(m2::KernelRunArgs{.kernel = m2::KernelSpecName{"writer_" + g.suffix}});
    }
    auto group_index_for = [&](const CoreCoord& core) -> size_t {
        for (size_t gi = 0; gi < groups.size(); ++gi) {
            if (groups[gi].cores.contains(core)) {
                return gi;
            }
        }
        TT_FATAL(false, "tilize block: core not found in any work-split group");
        return 0;
    };

    const auto& cores = corerange_to_cores(available_grid);
    uint32_t start_row_id = 0;
    uint32_t start_column_id = 0;
    uint32_t tile_start_id = 0;
    uint32_t single_block_size_row_arg;
    uint32_t single_block_size_col_arg;
    uint32_t single_sub_block_size_row_arg;

    uint32_t total_row_cores = full_cores_per_row;
    if (has_cliff_row) {
        total_row_cores++;
    }
    uint32_t cores_col_count = 1;
    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];
        if (has_cliff_col && has_cliff_row && i == ncores - 1) {
            single_block_size_row_arg = single_block_size_cliff_row;
            single_block_size_col_arg = single_block_size_cliff_col;
            single_sub_block_size_row_arg = single_block_size_cliff_row;

        } else if (has_cliff_row && i != 0 && ((i + 1) % (full_cores_per_row + 1)) == 0) {
            single_block_size_row_arg = single_block_size_cliff_row;
            single_block_size_col_arg = single_block_size;
            single_sub_block_size_row_arg = single_block_size_cliff_row;

        } else if (i < total_row_cores * full_cores_per_col) {
            single_block_size_row_arg = single_block_size;
            single_block_size_col_arg = single_block_size;
            single_sub_block_size_row_arg = single_sub_block_size;

        } else {
            single_block_size_row_arg = single_block_size;
            single_block_size_col_arg = single_block_size_cliff_col;
            single_sub_block_size_row_arg = single_sub_block_size;
        }

        const size_t gi = group_index_for(core);

        // reader runtime args — src addr is now a TensorBinding; the unused legacy slot 1 was the
        // pad_value (0 for tilize, which has no padding value to apply).
        reader_runs[gi].runtime_arg_values.push_back(
            {core,
             {{"pad_value", std::uint32_t{0}},
              {"width_size", TILE_WIDTH * a.element_size() * single_block_size_row_arg},
              {"start_row_id", start_row_id},
              {"start_column_id", start_column_id},
              {"single_block_size_row_arg", single_block_size_row_arg},
              {"single_block_size_col_arg", single_block_size_col_arg},
              {"sub_block_width_size", TILE_WIDTH * a.element_size() * single_sub_block_size_row_arg},
              {"single_sub_block_size_row_arg", single_sub_block_size_row_arg}}});

        // writer runtime args — dst addr is now a TensorBinding.
        writer_runs[gi].runtime_arg_values.push_back(
            {core,
             {{"start_id", tile_start_id},
              {"single_block_size_row_arg", single_block_size_row_arg},
              {"single_block_size_col_arg", single_block_size_col_arg}}});

        uint32_t end_column_id = start_column_id + (single_block_size_row_arg * TILE_WIDTH * a.element_size());
        start_column_id = end_column_id % row_size_bytes;
        if (end_column_id % row_size_bytes == 0 && end_column_id != 0) {
            start_row_id += single_block_size_col_arg * TILE_HEIGHT;
        }

        if (start_column_id == 0) {
            tile_start_id = cores_col_count * single_block_size_col_arg * total_tiles_per_row;
            cores_col_count++;
        } else {
            tile_start_id += single_block_size_row_arg;
        }
    }

    run.kernel_run_args.reserve(reader_runs.size() + writer_runs.size());
    for (auto& r : reader_runs) {
        run.kernel_run_args.push_back(std::move(r));
    }
    for (auto& w : writer_runs) {
        run.kernel_run_args.push_back(std::move(w));
    }
    run.tensor_args = {
        {m2::TensorParamName{"input"}, a.mesh_tensor()},
        {m2::TensorParamName{"output"}, output.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}
}  // namespace ttnn::prim
