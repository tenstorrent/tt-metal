// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_multi_core_block_program_factory.hpp"

#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

namespace {

const TensorParamName BLK_INPUT_TENSOR{"input"};
const TensorParamName BLK_OUTPUT_TENSOR{"output"};

constexpr const char* BLK_READER_SRC =
    "ttnn/cpp/ttnn/operations/experimental/quasar/tilize/device/kernels/dataflow/"
    "reader_unary_pad_multicore_both_dims_metal2.cpp";
constexpr const char* BLK_WRITER_SRC =
    "ttnn/cpp/ttnn/operations/experimental/quasar/tilize/device/kernels/dataflow/"
    "writer_unary_interleaved_start_id_wh.cpp";
constexpr const char* BLK_COMPUTE_SRC =
    "ttnn/cpp/ttnn/operations/experimental/quasar/tilize/device/kernels/compute/tilize_wh.cpp";

// One group per (non-empty) core range produced by split_blocks_for_tilize_wh.
// Each group gets its own correctly-sized DFB set (c_0/c_1/c_16) and its own
// reader/writer/compute KernelSpecs bound to those DFBs (the legacy factory placed
// differently-sized CBs on different core ranges; in Metal 2.0 a DFB's size is
// per-spec, so the per-range CB triples become per-group DFB triples).
struct GroupSpecs {
    std::string suffix;
    DFBSpecName in;
    DFBSpecName stage;
    DFBSpecName out;
    KernelSpecName reader;
    KernelSpecName writer;
    KernelSpecName compute;
    CoreRangeSet cores;
    uint32_t cb_num_tiles;  // tiles for c_0 / c_16
    uint32_t compute_col;   // compute CTA: block_size_col
    uint32_t compute_row;   // compute CTA: block_size_row
};

}  // namespace

ttnn::device_operation::ProgramArtifacts TilizeMultiCoreBlockProgramFactory::create_program_artifacts(
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

    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();

    // reader / writer scalars (shared across groups; from legacy)
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

    uint32_t single_sub_block_wh = single_block_size * single_block_size / single_sub_block_size;
    uint32_t single_sub_block_cliff_col_wh = single_block_size_cliff_col * single_block_size / single_sub_block_size;

    // Assemble the (up to four) groups, mirroring the legacy push_cb_pair + compute-kernel placement.
    std::vector<GroupSpecs> groups;
    auto add_group = [&](const std::string& suffix,
                         const CoreRangeSet& cores,
                         uint32_t cb_num_tiles,
                         uint32_t compute_col,
                         uint32_t compute_row) {
        groups.push_back(GroupSpecs{
            .suffix = suffix,
            .in = DFBSpecName{"in_" + suffix},
            .stage = DFBSpecName{"stage_" + suffix},
            .out = DFBSpecName{"out_" + suffix},
            .reader = KernelSpecName{"reader_" + suffix},
            .writer = KernelSpecName{"writer_" + suffix},
            .compute = KernelSpecName{"compute_" + suffix},
            .cores = cores,
            .cb_num_tiles = cb_num_tiles,
            .compute_col = compute_col,
            .compute_row = compute_row});
    };

    if (!core_range.empty()) {
        add_group("main", core_range, single_sub_block_size, single_sub_block_wh, single_sub_block_size);
    }
    if (has_cliff_col && has_cliff_row) {
        add_group(
            "cliff_col_row",
            cliff_col_row_core_range,
            single_block_size_cliff_row,
            single_block_size_cliff_col,
            single_block_size_cliff_row);
    }
    if (has_cliff_row) {
        add_group(
            "cliff_row",
            cliff_row_core_range,
            single_block_size_cliff_row,
            single_block_size,
            single_block_size_cliff_row);
    }
    if (has_cliff_col) {
        add_group(
            "cliff_col",
            cliff_col_core_range,
            single_sub_block_size,
            single_sub_block_cliff_col_wh,
            single_sub_block_size);
    }

    // -- Spec --
    ProgramSpec spec;
    spec.name = "tilize_multi_core_block";
    spec.tensor_parameters = {
        TensorParameter{.unique_id = BLK_INPUT_TENSOR, .spec = a.tensor_spec()},
        TensorParameter{.unique_id = BLK_OUTPUT_TENSOR, .spec = output.tensor_spec()},
    };

    ttnn::ComputeKernelConfig compute_config_template{
        .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = fp32_llk_acc};

    uint32_t input_row_bytes = input_single_tile_size / TILE_HEIGHT;
    for (const auto& g : groups) {
        // c_1 staging fake CB size (legacy push_cb_pair):
        uint32_t stage_size = input_row_bytes * g.cb_num_tiles + 2 * dram_alignment;

        // DFBs for this group.
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = g.stage,
            .entry_size = stage_size,
            .num_entries = 1,
            .data_format_metadata = input_cb_data_format,
        });
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = g.in,
            .entry_size = input_single_tile_size,
            .num_entries = g.cb_num_tiles,
            .data_format_metadata = input_cb_data_format,
        });
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = g.out,
            .entry_size = output_single_tile_size,
            .num_entries = g.cb_num_tiles,
            .data_format_metadata = output_cb_data_format,
        });

        // Reader: produces c_0 (in); self-loops c_1 (stage) fake CB; reads input tensor.
        spec.kernels.push_back(KernelSpec{
            .unique_id = g.reader,
            .source = BLK_READER_SRC,
            .dfb_bindings =
                {DFBBinding{.dfb_spec_name = g.in, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER},
                 DFBBinding{
                     .dfb_spec_name = g.stage, .accessor_name = "stage", .endpoint_type = DFBEndpointType::PRODUCER},
                 DFBBinding{
                     .dfb_spec_name = g.stage, .accessor_name = "stage", .endpoint_type = DFBEndpointType::CONSUMER}},
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = BLK_INPUT_TENSOR, .accessor_name = "src"}},
            .compile_time_args =
                {{"total_num_rows", total_num_rows},
                 {"third_dim", third_dim},
                 {"tile_height", tile_height},
                 {"element_size", a.element_size()},
                 {"unpadded_X_size", row_size_bytes},
                 {"dram_alignment", dram_alignment}},
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"pad_value",
                      "width_size",
                      "start_row_id",
                      "start_column_id",
                      "single_block_size_row_arg",
                      "single_block_size_col_arg",
                      "sub_block_width_size",
                      "single_sub_block_size_row_arg"}},
            .hw_config =
                ttnn::create_reader_datamovement_config(device->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
        });

        // Writer: consumes c_16 (out); writes output tensor.
        spec.kernels.push_back(KernelSpec{
            .unique_id = g.writer,
            .source = BLK_WRITER_SRC,
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = g.out, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = BLK_OUTPUT_TENSOR, .accessor_name = "dst"}},
            .compile_time_args =
                {{"num_tiles_per_2d", num_tiles_2d},
                 {"third_dim", third_dim},
                 {"total_tiles_per_row", total_tiles_per_row}},
            .runtime_arg_schema =
                {.runtime_arg_names = {"start_id", "single_block_size_row_arg", "single_block_size_col_arg"}},
            .hw_config =
                ttnn::create_writer_datamovement_config(device->arch(), /*disable_dfb_implicit_sync_for_all=*/true),
        });

        // Compute: consumes c_0 (in), produces c_16 (out).
        ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(device->arch(), compute_config_template);
        if (fp32_llk_acc) {
            std::visit([&](auto& c) { c.unpack_modes.emplace(g.in, UnpackMode::UnpackToDest); }, compute_hw);
        }
        spec.kernels.push_back(KernelSpec{
            .unique_id = g.compute,
            .source = BLK_COMPUTE_SRC,
            .dfb_bindings =
                {DFBBinding{.dfb_spec_name = g.in, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = g.out, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args =
                {{"block_size_col", g.compute_col}, {"block_size_row", g.compute_row}, {"third_dim", third_dim}},
            .hw_config = compute_hw,
        });

        spec.work_units.push_back(WorkUnitSpec{
            .name = "tilize_block_wu_" + g.suffix,
            .kernels = {g.reader, g.writer, g.compute},
            .target_nodes = g.cores,
        });
    }

    // -- Run args (per-core, routed to the owning group; values exactly as legacy) --
    ProgramRunArgs run_args;
    std::vector<KernelRunArgs> reader_runs(groups.size());
    std::vector<KernelRunArgs> writer_runs(groups.size());
    for (size_t gi = 0; gi < groups.size(); ++gi) {
        reader_runs[gi].kernel = groups[gi].reader;
        writer_runs[gi].kernel = groups[gi].writer;
    }
    auto group_index_of = [&](const CoreCoord& core) -> int {
        for (size_t gi = 0; gi < groups.size(); ++gi) {
            if (groups[gi].cores.contains(core)) {
                return static_cast<int>(gi);
            }
        }
        return -1;
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

        int gi = group_index_of(core);
        TT_FATAL(gi >= 0, "tilize block: core not covered by any work-unit group");

        KernelRunArgs::RuntimeArgValues& reader_rtas = reader_runs[gi].runtime_arg_values;
        KernelRunArgs::RuntimeArgValues& writer_rtas = writer_runs[gi].runtime_arg_values;

        AddRuntimeArgsForNode(
            reader_rtas,
            core,
            {
                {"pad_value", 0u},
                {"width_size", TILE_WIDTH * a.element_size() * single_block_size_row_arg},
                {"start_row_id", start_row_id},
                {"start_column_id", start_column_id},
                {"single_block_size_row_arg", single_block_size_row_arg},
                {"single_block_size_col_arg", single_block_size_col_arg},
                {"sub_block_width_size", TILE_WIDTH * a.element_size() * single_sub_block_size_row_arg},
                {"single_sub_block_size_row_arg", single_sub_block_size_row_arg},
            });

        AddRuntimeArgsForNode(
            writer_rtas,
            core,
            {
                {"start_id", tile_start_id},
                {"single_block_size_row_arg", single_block_size_row_arg},
                {"single_block_size_col_arg", single_block_size_col_arg},
            });

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

    for (size_t gi = 0; gi < groups.size(); ++gi) {
        run_args.kernel_run_args.push_back(std::move(reader_runs[gi]));
        run_args.kernel_run_args.push_back(std::move(writer_runs[gi]));
    }
    run_args.tensor_args = {
        {BLK_INPUT_TENSOR, TensorArgument{a.mesh_tensor()}},
        {BLK_OUTPUT_TENSOR, TensorArgument{output.mesh_tensor()}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}
}  // namespace ttnn::prim::qsr
