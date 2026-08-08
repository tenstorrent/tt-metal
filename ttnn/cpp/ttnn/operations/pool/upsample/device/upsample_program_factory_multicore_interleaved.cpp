// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstdint>
#include <string>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_device_operation.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_common.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

const KernelSpecName INTLV_READER{"upsample_intlv_reader"};
const KernelSpecName INTLV_WRITER{"upsample_intlv_writer"};
const KernelSpecName INTLV_COMPUTE_G1{"upsample_intlv_compute_g1"};
const KernelSpecName INTLV_COMPUTE_G2{"upsample_intlv_compute_g2"};
const DFBSpecName INTLV_SRC0{"upsample_intlv_src0"};
const DFBSpecName INTLV_OUT{"upsample_intlv_out"};
const TensorParamName INTLV_INPUT{"upsample_intlv_input"};
const TensorParamName INTLV_OUTPUT{"upsample_intlv_output"};

// Existing Metal 2.0 fork of the untilize op's compute kernel (created by the data_movement/fold
// port). Its binding names (dfb::src / dfb::out) and named args (per_core_block_cnt,
// per_core_block_tile_cnt) are the shared interface this factory must match; reused as-is.
constexpr const char* UNTILIZE_METAL2_KERNEL =
    "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_metal2.cpp";

}  // namespace

ttnn::device_operation::ProgramArtifacts UpsampleMultiCoreInterleavedProgramFactory::create_program_artifacts(
    const UpsampleParams& operation_attributes, const Tensor& input_tensor, Tensor& output_tensor) {
    const auto& input = input_tensor;
    auto& output = output_tensor;
    const auto& input_mesh = input.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();
    // This factory only supports integer scale factors
    TT_FATAL(
        operations::pool::upsample::is_integer_scale(operation_attributes.scale_factor_h) &&
            operations::pool::upsample::is_integer_scale(operation_attributes.scale_factor_w),
        "Interleaved upsample factory requires integer scale factors, got scale_h={}, scale_w={}",
        operation_attributes.scale_factor_h,
        operation_attributes.scale_factor_w);
    const uint32_t scale_factor_h = static_cast<uint32_t>(operation_attributes.scale_factor_h);
    const uint32_t scale_factor_w = static_cast<uint32_t>(operation_attributes.scale_factor_w);

    const bool is_tiled_layout = (input.layout() == Layout::TILE);

    const tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());

    const auto& output_shape = output.padded_shape();
    IDevice* const device = output.device();

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Declare variables that will be set based on layout
    uint32_t input_unit_size;
    uint32_t output_unit_size;
    uint32_t input_cb_required_pages;
    uint32_t work_units_to_split;
    uint32_t aligned_input_unit_size;  // Size used for CB creation

    if (is_tiled_layout) {
        // Tiled layout specific calculations
        input_unit_size = tt::tile_size(input_cb_data_format);
        output_unit_size = tt::tile_size(output_cb_data_format);
        aligned_input_unit_size = input_unit_size;

        const uint32_t input_tensor_width = input.padded_shape()[-1];
        const uint32_t input_tensor_height = input.physical_volume() / input_tensor_width;

        const auto& tile_shape = input.tensor_spec().tile().get_tile_shape();
        const uint32_t tile_height = tile_shape[0];
        const uint32_t tile_width = tile_shape[1];

        const uint32_t num_input_tiles_in_row = input_tensor_width / tile_width;
        const uint32_t num_input_tiles_in_col = input_tensor_height / tile_height;

        /*
        For tiled layout, a unit of work (input wise) is a row of tiles
        */

        input_cb_required_pages = num_input_tiles_in_row;  // for CB sizing
        work_units_to_split = num_input_tiles_in_col;      // for work splitting
    } else {
        // Row-major layout specific calculations
        input_unit_size = input.padded_shape()[-1] * input.element_size();
        output_unit_size = output.padded_shape()[-1] * output.element_size();
        aligned_input_unit_size = tt::round_up(input_unit_size, hal::get_dram_alignment());

        /*
        For Row-major layout, a unit of work is one row (stick) of the input tensor
        */

        input_cb_required_pages = 1;                                               // One input unit is required in CB
        work_units_to_split = input.physical_volume() / input.padded_shape()[-1];  // N*H*W unit split
    }

    const auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_group_1, work_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, work_units_to_split);

    // Create dataflow buffers
    uint32_t num_pages_in_input_cb = input_cb_required_pages;
    if (work_per_core_group_1 != 1) {
        // Double buffer if the core is processing 2+ blocks
        num_pages_in_input_cb *= 2;
    }

    DataflowBufferSpec src0_dfb{
        .unique_id = INTLV_SRC0,
        .entry_size = aligned_input_unit_size,
        .num_entries = num_pages_in_input_cb,
        .data_format_metadata = input_cb_data_format,
    };

    // On the row-major path, the writer consumes directly from src0 (no separate output DFB /
    // compute kernel). On the tiled path, a separate output DFB + untilize compute kernel sit
    // between the reader and the writer.
    const DFBSpecName& writer_input_dfb = is_tiled_layout ? INTLV_OUT : INTLV_SRC0;
    std::optional<DataflowBufferSpec> out_dfb;
    if (is_tiled_layout) {
        // Separate output CB for tiled
        const uint32_t num_pages_in_output_cb = num_pages_in_input_cb;
        out_dfb = DataflowBufferSpec{
            .unique_id = INTLV_OUT,
            .entry_size = output_unit_size,
            .num_entries = num_pages_in_output_cb,
            .data_format_metadata = output_cb_data_format,
        };
    }

    KernelSpec::CompileTimeArgs reader_cta{{"aligned_input_unit_size", aligned_input_unit_size}};

    KernelSpec reader{
        .unique_id = INTLV_READER,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/"
                                        "reader_upsample_unary_stick_layout_interleaved_start_id.cpp"},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = INTLV_SRC0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INTLV_INPUT, .accessor_name = "input"}},
        .compile_time_args = std::move(reader_cta),
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_page_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // Writer compile time arguments
    const int32_t writer_unit_size = output.padded_shape()[-1] * output.element_size();

    KernelSpec::CompileTimeArgs writer_cta{
        {"output_page_size", static_cast<uint32_t>(writer_unit_size)},
        {"scale_h", scale_factor_h},
        {"scale_w", scale_factor_w},
        {"height", output_shape[1]},
        {"width", output_shape[2]},
    };

    if (is_tiled_layout) {
        const auto& tile_shape = input.tensor_spec().tile().get_tile_shape();
        const uint32_t tile_height = tile_shape[0];
        const uint32_t num_input_tiles_in_row = input.padded_shape()[-1] / tile_shape[1];

        // tile_height rows need to be processed at a time
        writer_cta.insert({"block_height", tile_height});
        // whole row of tiles needs to be processed to get valid output sticks
        writer_cta.insert({"num_tiles_per_block_row", num_input_tiles_in_row});
    } else {
        constexpr uint32_t block_height = 1;                // since input is row major, blocks are just one row tall
        constexpr uint32_t num_units_per_output_stick = 1;  // 1 page in out_cb is needed to get a valid output stick
        writer_cta.insert({"block_height", block_height});
        writer_cta.insert({"num_tiles_per_block_row", num_units_per_output_stick});
    }

    KernelSpec writer{
        .unique_id = INTLV_WRITER,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/writer_upsample_interleaved.cpp"},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = writer_input_dfb, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INTLV_OUTPUT, .accessor_name = "output"}},
        .compile_time_args = std::move(writer_cta),
        .runtime_arg_schema = {.runtime_arg_names = {"num_blocks_to_read", "start_block_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    ProgramSpec spec{
        .name = "upsample_multicore_interleaved",
        .kernels = {reader, writer},
        .dataflow_buffers = {src0_dfb},
        .tensor_parameters =
            {
                {.unique_id = INTLV_INPUT, .spec = input_mesh.tensor_spec()},
                {.unique_id = INTLV_OUTPUT, .spec = output_mesh.tensor_spec()},
            },
    };
    if (out_dfb) {
        spec.dataflow_buffers.push_back(std::move(*out_dfb));
    }

    // WorkUnitSpecs must have disjoint target_nodes, so a shared READER/WRITER can't sit in one
    // WorkUnitSpec spanning all_cores while COMPUTE sits in another over a sub-range of the same
    // cores. Row-major (no compute) needs only one WU over all_cores; tiled lists READER/WRITER in
    // each per-group WU alongside that group's COMPUTE instance — their combined placement (the
    // union of the WUs they appear in) is still all_cores.
    if (!is_tiled_layout) {
        spec.work_units.push_back(WorkUnitSpec{
            .name = "main",
            .kernels = {INTLV_READER, INTLV_WRITER},
            .target_nodes = all_cores,
        });
    } else {
        const uint32_t num_input_tiles_in_row =
            input.padded_shape()[-1] / input.tensor_spec().tile().get_tile_shape()[1];

        if (core_group_1.num_cores() > 0) {
            KernelSpec compute_g1{
                .unique_id = INTLV_COMPUTE_G1,
                .source = std::filesystem::path{UNTILIZE_METAL2_KERNEL},
                .dfb_bindings =
                    {DFBBinding{
                         .dfb_spec_name = INTLV_SRC0,
                         .accessor_name = "src",
                         .endpoint_type = DFBEndpointType::CONSUMER},
                     DFBBinding{
                         .dfb_spec_name = INTLV_OUT,
                         .accessor_name = "out",
                         .endpoint_type = DFBEndpointType::PRODUCER}},
                .compile_time_args =
                    {
                        {"per_core_block_cnt", work_per_core_group_1},
                        {"per_core_block_tile_cnt", num_input_tiles_in_row},
                    },
                .hw_config = ComputeGen1Config{},
            };
            spec.kernels.push_back(compute_g1);
            spec.work_units.push_back(WorkUnitSpec{
                .name = "wu_g1",
                .kernels = {INTLV_READER, INTLV_WRITER, INTLV_COMPUTE_G1},
                .target_nodes = core_group_1,
            });
        }

        if (core_group_2.num_cores() > 0) {
            KernelSpec compute_g2{
                .unique_id = INTLV_COMPUTE_G2,
                .source = std::filesystem::path{UNTILIZE_METAL2_KERNEL},
                .dfb_bindings =
                    {DFBBinding{
                         .dfb_spec_name = INTLV_SRC0,
                         .accessor_name = "src",
                         .endpoint_type = DFBEndpointType::CONSUMER},
                     DFBBinding{
                         .dfb_spec_name = INTLV_OUT,
                         .accessor_name = "out",
                         .endpoint_type = DFBEndpointType::PRODUCER}},
                .compile_time_args =
                    {
                        {"per_core_block_cnt", work_per_core_group_2},
                        {"per_core_block_tile_cnt", num_input_tiles_in_row},
                    },
                .hw_config = ComputeGen1Config{},
            };
            spec.kernels.push_back(compute_g2);
            spec.work_units.push_back(WorkUnitSpec{
                .name = "wu_g2",
                .kernels = {INTLV_READER, INTLV_WRITER, INTLV_COMPUTE_G2},
                .target_nodes = core_group_2,
            });
        }
    }

    // Per-core runtime args
    /*
    For tiled input, a block refers to a row of input tiles
    For row-major input, a block refers to a single input row (stick)
    */
    ProgramRunArgs run_args;
    KernelRunArgs reader_run_args{.kernel = INTLV_READER};
    KernelRunArgs writer_run_args{.kernel = INTLV_WRITER};

    for (uint32_t i = 0, blocks_processed = 0; i < num_cores; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t blocks_per_core = 0;
        if (core_group_1.contains(core)) {
            blocks_per_core = work_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            blocks_per_core = work_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        const uint32_t reader_units = blocks_per_core * input_cb_required_pages;
        const uint32_t reader_start = blocks_processed * input_cb_required_pages;

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"num_pages", reader_units}, {"start_page_id", reader_start}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_blocks_to_read", blocks_per_core}, {"start_block_id", blocks_processed}});

        blocks_processed += blocks_per_core;
    }
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {INTLV_INPUT, TensorArgument{input_mesh}},
        {INTLV_OUTPUT, TensorArgument{output_mesh}},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
