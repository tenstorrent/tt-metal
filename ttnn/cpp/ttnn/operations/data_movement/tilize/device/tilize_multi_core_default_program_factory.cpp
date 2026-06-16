// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_multi_core_default_program_factory.hpp"

#include <filesystem>

#include "ttnn/operations/core/work_split/work_split_tilize.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {

// Metal 2.0 program factory: builds the immutable ProgramSpec and its mutable ProgramRunArgs.
// Behavior-preserving port of the legacy ProgramDescriptor multi-core (interleaved, default-selected)
// tilize factory.
ttnn::device_operation::ProgramArtifacts TilizeMultiCoreDefaultProgramFactory::create_program_spec(
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

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    auto logical_shape = a.logical_shape();
    uint32_t logical_width = logical_shape[-1];
    uint32_t ntiles_per_block = tt::div_up(logical_width, TILE_WIDTH);
    uint32_t ntiles = dst_buffer->num_pages();
    uint32_t nblocks = tt::div_up(ntiles, ntiles_per_block);
    auto* device = a.device();
    auto grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid = sub_core_grids.has_value() ? sub_core_grids.value() : default_grid;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(available_grid, nblocks);

    // ---- reader CT/RTA derivations (identical to legacy) ----
    uint32_t page_size = src0_buffer->page_size();
    uint32_t num_pages_in_row = 1;
    uint32_t size_of_valid_data_in_last_page_in_row = page_size;
    if (a.is_sharded()) {
        uint32_t shard_width =
            a.shard_spec().has_value() ? a.shard_spec().value().shape[1] : a.nd_shard_spec().value().shard_shape[-1];
        num_pages_in_row = tt::div_up(logical_width,
                                      shard_width);  // Compute number of pages in one tensor row.
        uint32_t padding_size =
            (num_pages_in_row * page_size) -
            (a.logical_shape()[-1] * a.element_size());  // Compute padding size for the last page in the row.
        size_of_valid_data_in_last_page_in_row = page_size - padding_size;
    }

    // ---- ProgramSpec (immutable) ----
    m2::ProgramSpec spec;
    spec.name = "tilize_multi_core";

    // "src0" (legacy CB c_0): input row-major sticks the reader fills and the compute tilizes.
    // "output" (legacy CB c_16): tilized output the compute produces and the writer drains.
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"src0"},
            .entry_size = input_single_tile_size,
            .num_entries = ntiles_per_block,
            .data_format_metadata = input_cb_data_format,
        },
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"output"},
            .entry_size = output_single_tile_size,
            .num_entries = ntiles_per_block,
            .data_format_metadata = output_cb_data_format,
        },
    };

    // Reader (multicore stick-layout split-rows). Reads the input tensor (binding) into the src0 DFB.
    // Legacy positional CTs were {aligned_page_size, num_pages_in_row, size_of_valid_data_in_last_page_in_row}
    // followed by TensorAccessorArgs<3>; the kernel only reads CT slots 1 and 2 (aligned_page_size is
    // unused), and the accessor moves to the ta::input binding.
    // Legacy positional RTAs were slots 0(src addr -> binding),1,3,4,5,8; the unused slots 2/6/7
    // (stick_size, num_leftover_tiles, leftover_width_in_row) are dropped — the kernel never read them.
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
                                        "reader_unary_stick_layout_split_rows_multicore.cpp"},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"src0"},
                    .accessor_name = "src0",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER,
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
                {"num_pages_in_row", num_pages_in_row},
                {"size_of_valid_data_in_last_page_in_row", size_of_valid_data_in_last_page_in_row},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"num_rows", "num_tiles_per_block", "block_width_size", "num_full_blocks_in_row", "start_page_id"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };

    // Writer (forked from eltwise/unary writer_unary_interleaved_start_id.cpp).
    // Drains the output DFB into the output tensor (binding). Legacy CT was {output_cb_index} +
    // TensorAccessorArgs; both become bindings (dfb::output / ta::output).
    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
                                        "writer_unary_interleaved_start_id_m2.cpp"},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"output"},
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
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_pages", "start_id"},
            },
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };

    // Compute (forked from ttnn/cpp/ttnn/kernel/compute/tilize.cpp).
    // Consumes src0, produces output. per_core_block_cnt / per_core_block_tile_cnt stay CTAs; the full
    // and cliff core groups differ only in per_core_block_cnt (nblocks_per_core vs nblocks_per_core_cliff).
    auto make_compute = [&](const std::string& id, uint32_t per_core_block_cnt) {
        m2::KernelSpec cd{
            .unique_id = m2::KernelSpecName{id},
            .source =
                std::filesystem::path{
                    "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_m2.cpp"},
            .dfb_bindings =
                {
                    m2::DFBBinding{
                        .dfb_spec_name = m2::DFBSpecName{"src0"},
                        .accessor_name = "src0",
                        .endpoint_type = m2::DFBEndpointType::CONSUMER,
                    },
                    m2::DFBBinding{
                        .dfb_spec_name = m2::DFBSpecName{"output"},
                        .accessor_name = "output",
                        .endpoint_type = m2::DFBEndpointType::PRODUCER,
                    },
                },
            .compile_time_args =
                {
                    {"per_core_block_cnt", per_core_block_cnt},
                    {"per_core_block_tile_cnt", ntiles_per_block},
                },
            .hw_config =
                m2::ComputeHardwareConfig{
                    .fp32_dest_acc_en = fp32_llk_acc,
                },
        };
        // Legacy set unpack_to_dest_mode[c_0]=UnpackToDestFp32 when fp32_llk_acc; preserve that for
        // the src0 DFB (the compute consumer of the fp32 input).
        if (fp32_llk_acc) {
            std::get<m2::ComputeHardwareConfig>(cd.hw_config).unpack_to_dest_mode = {
                {m2::DFBSpecName{"src0"}, UnpackToDestMode::UnpackToDestFp32}};
        }
        return cd;
    };

    const bool has_full = !core_range.ranges().empty();
    const bool has_cliff = !core_range_cliff.ranges().empty();

    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = a.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output.tensor_spec()},
    };

    // Local DFBs (src0/output) require their producer and consumer KernelSpecs to share the SAME
    // WorkUnitSpec(s) — every node hosting the DFB must host both endpoints. So each core group gets
    // one WorkUnitSpec containing reader + compute_<group> + writer (reader/writer are shared across
    // both groups; the per-group compute differs only in its per_core_block_cnt CTA).
    spec.kernels = {reader, writer};
    std::vector<m2::WorkUnitSpec> work_units;
    if (has_full) {
        spec.kernels.push_back(make_compute("compute", nblocks_per_core));
        work_units.push_back(m2::WorkUnitSpec{
            .name = "full",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}, m2::KernelSpecName{"compute"}},
            .target_nodes = core_range,
        });
    }
    if (has_cliff) {
        spec.kernels.push_back(make_compute("compute_cliff", nblocks_per_core_cliff));
        work_units.push_back(m2::WorkUnitSpec{
            .name = "cliff",
            .kernels =
                {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}, m2::KernelSpecName{"compute_cliff"}},
            .target_nodes = core_range_cliff,
        });
    }
    spec.work_units = std::move(work_units);

    // ---- ProgramRunArgs (mutable) ----
    m2::ProgramRunArgs run;
    m2::KernelRunArgs reader_run{.kernel = m2::KernelSpecName{"reader"}};
    m2::KernelRunArgs writer_run{.kernel = m2::KernelSpecName{"writer"}};

    // 1D distribution of blocks across cores (identical to legacy): first ncores_full cores are full,
    // the last core (when present) is the cliff core.
    uint32_t ncores_full = ncores - (has_cliff ? 1 : 0);
    uint32_t tile_start_id = 0;
    uint32_t page_start_id = 0;
    const auto& cores = corerange_to_cores(available_grid);
    for (uint32_t i = 0; i < ncores_full; ++i) {
        const CoreCoord& core = cores[i];

        reader_run.runtime_arg_values.push_back(
            {core,
             {{"num_rows", nblocks_per_core * TILE_HEIGHT},
              {"num_tiles_per_block", ntiles_per_block},
              {"block_width_size", page_size},
              {"num_full_blocks_in_row", std::uint32_t{1}},
              {"start_page_id", page_start_id}}});

        writer_run.runtime_arg_values.push_back(
            {core, {{"num_pages", ntiles_per_block * nblocks_per_core}, {"start_id", tile_start_id}}});

        tile_start_id += ntiles_per_block * nblocks_per_core;
        page_start_id += TILE_HEIGHT * nblocks_per_core * num_pages_in_row;
    }
    if (has_cliff) {
        // the last core is a cliff core with nblocks_per_core_cliff blocks
        const CoreCoord& core = cores[ncores_full];

        reader_run.runtime_arg_values.push_back(
            {core,
             {{"num_rows", nblocks_per_core_cliff * TILE_HEIGHT},
              {"num_tiles_per_block", ntiles_per_block},
              {"block_width_size", page_size},
              {"num_full_blocks_in_row", std::uint32_t{1}},
              {"start_page_id", page_start_id}}});

        writer_run.runtime_arg_values.push_back(
            {core, {{"num_pages", ntiles_per_block * nblocks_per_core_cliff}, {"start_id", tile_start_id}}});
    }

    run.kernel_run_args = {reader_run, writer_run};
    run.tensor_args = {
        {m2::TensorParamName{"input"}, a.mesh_tensor()},
        {m2::TensorParamName{"output"}, output.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::prim
