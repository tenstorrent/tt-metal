// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_wh_program_factory.hpp"
#include "transpose_utils.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <vector>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

ttnn::device_operation::ProgramArtifacts TransposeWHProgramFactory::create_program_artifacts(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    // Metal 2.0 named resource handles (locals to avoid unity-build name collisions).
    const DFBSpecName CB_IN0{"cb_in0"};        // legacy c_0: input tile stream
    const DFBSpecName CB_OUT0{"cb_out0"};      // legacy c_16: transposed output stream
    const DFBSpecName CB_TILIZE{"cb_tilize"};  // legacy c_24: RM-only tilize intermediate

    const TensorParamName INPUT_TENSOR{"input"};
    const TensorParamName OUTPUT_TENSOR{"output"};

    const KernelSpecName READER_KERNEL{"reader"};
    const KernelSpecName WRITER_KERNEL{"writer"};
    const KernelSpecName COMPUTE_KERNEL{"compute"};

    const auto& input_tensor = tensor_args.input;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_wh needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_wh needs to be allocated in a buffer on device!");

    const uint32_t num_tensor_tiles = input_tensor.physical_volume() / TILE_HW;
    const uint32_t W = input_tensor.logical_shape()[3], H = input_tensor.logical_shape()[2];
    const uint32_t NC = input_tensor.logical_shape()[1] * input_tensor.logical_shape()[0];
    const bool row_major = input_tensor.layout() == Layout::ROW_MAJOR;

    const uint32_t ht = (H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t wt = (W + TILE_WIDTH - 1) / TILE_WIDTH;

    const tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    const tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    const uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    const bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32 ||
                                  src0_cb_data_format == tt::DataFormat::Int32 ||
                                  src0_cb_data_format == tt::DataFormat::UInt32;
    const bool src_is_float32 = src0_cb_data_format == tt::DataFormat::Float32;

    log_debug(tt::LogOp, "transpose_wh");
    log_debug(tt::LogOp, "row_major: {}", row_major);
    log_debug(tt::LogOp, "src0_cb_data_format: {}", src0_cb_data_format);

    IDevice* device = input_tensor.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const uint32_t num_cores_total = num_cores_x * num_cores_y;
    const CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, row_major ? NC : num_tensor_tiles);

    // ------------------------------------------------------------------------
    // DataflowBufferSpecs. cb_in0 (c_0) / cb_out0 (c_16) exist on both paths; the row-major path
    // also needs cb_tilize (c_24), the tilize intermediate. The legacy RM path also allocated an
    // unused c_25 ("TODO REMOVE") that no kernel references — it is dropped here.
    // ------------------------------------------------------------------------
    const uint32_t num_input_tiles = row_major ? wt * 2 : 2;
    const uint32_t num_output_tiles = row_major ? ht * 2 : 2;

    std::vector<DataflowBufferSpec> dfbs;
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = CB_IN0,
        .entry_size = src0_single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = src0_cb_data_format,
    });
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = CB_OUT0,
        .entry_size = dst_single_tile_size,
        .num_entries = num_output_tiles,
        .data_format_metadata = dst_cb_data_format,
    });
    if (row_major) {
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = CB_TILIZE,
            .entry_size = src0_single_tile_size,
            .num_entries = ht * wt,
            .data_format_metadata = src0_cb_data_format,
        });
    }

    // ------------------------------------------------------------------------
    // Tensor parameters. Both accessors carried RuntimeTensorShape → dynamic_tensor_shape = true.
    // ------------------------------------------------------------------------
    TensorParameter input_param{
        .unique_id = INPUT_TENSOR,
        .spec = input_tensor.tensor_spec(),
        .advanced_options = {.dynamic_tensor_shape = true},
    };
    TensorParameter output_param{
        .unique_id = OUTPUT_TENSOR,
        .spec = output_tensor.tensor_spec(),
        .advanced_options = {.dynamic_tensor_shape = true},
    };

    std::vector<KernelSpec> kernels;
    KernelRunArgs reader_run{.kernel = READER_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    KernelRunArgs compute_run{.kernel = COMPUTE_KERNEL};

    if (row_major) {
        // --------------------------------------------------------------------
        // Row-major path: reader tilizes-by-rows into cb_in0, compute tilizes + transposes via
        // the cb_tilize intermediate, writer untilizes-by-cols out of cb_out0.
        // --------------------------------------------------------------------
        const uint32_t H_per_tile = H > TILE_HEIGHT ? TILE_HEIGHT : H % TILE_HEIGHT;
        const uint32_t H_per_tile_last = H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT;
        const uint32_t W_per_tile = W > TILE_WIDTH ? TILE_WIDTH : W % TILE_WIDTH;
        const uint32_t W_per_tile_last = W % TILE_WIDTH == 0 ? TILE_WIDTH : W % TILE_WIDTH;

        KernelSpec reader_spec{
            .unique_id = READER_KERNEL,
            .source =
                std::filesystem::path{"ttnn/cpp/ttnn/operations/experimental/quasar/transpose/device/kernels/dataflow/"
                                      "reader_unary_transpose_wh_interleaved_start_id_rm.cpp"},
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = CB_IN0, .accessor_name = "cb_in0", .endpoint_type = DFBEndpointType::PRODUCER}},
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "src"}},
            .compile_time_args =
                {{"Ht", ht},
                 {"H_per_tile", H_per_tile},
                 {"H_per_tile_last", H_per_tile_last},
                 {"Wt", wt},
                 {"W", W},
                 {"HtWt", ht * wt},
                 {"W_size_bytes", W * input_tensor.element_size()},
                 {"l1_write_offset_bytes", wt * input_tensor.element_size() * TILE_WIDTH}},
            .runtime_arg_schema = {.runtime_arg_names = {"start_id", "num_hw_blocks"}},
            .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
        };

        KernelSpec writer_spec{
            .unique_id = WRITER_KERNEL,
            .source =
                std::filesystem::path{"ttnn/cpp/ttnn/operations/experimental/quasar/transpose/device/kernels/dataflow/"
                                      "writer_unary_transpose_wh_interleaved_start_id_rm.cpp"},
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = CB_OUT0, .accessor_name = "cb_out0", .endpoint_type = DFBEndpointType::CONSUMER}},
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "dst"}},
            .compile_time_args =
                {{"Ht", ht},
                 {"H", H},
                 {"Wt", wt},
                 {"W_per_tile", W_per_tile},
                 {"W_per_tile_last", W_per_tile_last},
                 {"HtWt", ht * wt},
                 {"H_size_bytes", H * output_tensor.element_size()},
                 {"l1_read_offset_bytes", ht * output_tensor.element_size() * TILE_HEIGHT}},
            .runtime_arg_schema = {.runtime_arg_names = {"start_id", "num_hw_blocks"}},
            .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
        };

        ttnn::ComputeKernelConfig compute_cfg{
            .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = fp32_dest_acc_en};
        ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(device->arch(), compute_cfg);
        if (src_is_float32) {
            // Keep the source CB and the tile-formatted intermediate (cb_tilize) in full Float32
            // on the unpack-to-dest path; both feed the transpose.
            std::visit(
                [&](auto& c) {
                    c.unpack_to_dest_mode.emplace(CB_IN0, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32);
                    c.unpack_to_dest_mode.emplace(CB_TILIZE, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32);
                },
                compute_hw);
        }

        KernelSpec compute_spec{
            .unique_id = COMPUTE_KERNEL,
            .source =
                std::filesystem::path{"ttnn/cpp/ttnn/operations/experimental/quasar/transpose/device/kernels/compute/"
                                      "transpose_wh_rm.cpp"},
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = CB_IN0, .accessor_name = "cb_in0", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = CB_TILIZE,
                     .accessor_name = "cb_tilize",
                     .endpoint_type = DFBEndpointType::PRODUCER},
                 DFBBinding{
                     .dfb_spec_name = CB_TILIZE,
                     .accessor_name = "cb_tilize",
                     .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = CB_OUT0, .accessor_name = "cb_out0", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args = {{"Ht", ht}, {"Wt", wt}, {"HtWt", ht * wt}},
            .runtime_arg_schema = {.runtime_arg_names = {"num_hw_blocks"}},
            .hw_config = compute_hw,
        };
        if (input_tensor.dtype() == DataType::UINT32 || input_tensor.dtype() == DataType::INT32) {
            compute_spec.compiler_options.defines = {{"DST_ACCUM_MODE", "1"}};
        }

        kernels = {std::move(reader_spec), std::move(writer_spec), std::move(compute_spec)};

        for (uint32_t i = 0, num_sticks_read = 0, num_sticks_write = 0; i < num_cores_total; i++) {
            const CoreCoord core = {i / num_cores_y, i % num_cores_y};
            uint32_t num_hw_blocks_per_core;
            if (core_group_1.contains(core)) {
                num_hw_blocks_per_core = num_tiles_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_hw_blocks_per_core = num_tiles_per_core_group_2;
            } else {
                num_hw_blocks_per_core = 0;
            }

            const NodeCoord node = core;
            KernelRunArgs::RuntimeArgValues& reader_rtas = reader_run.runtime_arg_values;
            KernelRunArgs::RuntimeArgValues& writer_rtas = writer_run.runtime_arg_values;
            KernelRunArgs::RuntimeArgValues& compute_rtas = compute_run.runtime_arg_values;
            AddRuntimeArgsForNode(
                reader_rtas,
                node,
                {
                    {"start_id", num_sticks_read},
                    {"num_hw_blocks", num_hw_blocks_per_core},
                });
            compute_rtas["num_hw_blocks"][node] = num_hw_blocks_per_core;
            AddRuntimeArgsForNode(
                writer_rtas,
                node,
                {
                    {"start_id", num_sticks_write},
                    {"num_hw_blocks", num_hw_blocks_per_core},
                });

            num_sticks_read += num_hw_blocks_per_core * H;
            num_sticks_write += num_hw_blocks_per_core * W;
        }
    } else {
        // --------------------------------------------------------------------
        // Tiled path: reader streams tiles in NWH order into cb_in0, compute transposes each tile
        // into cb_out0, writer streams the transposed tiles out.
        // --------------------------------------------------------------------
        KernelSpec reader_spec{
            .unique_id = READER_KERNEL,
            .source =
                std::filesystem::path{"ttnn/cpp/ttnn/operations/experimental/quasar/transpose/device/kernels/dataflow/"
                                      "reader_unary_transpose_wh_interleaved_start_id.cpp"},
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = CB_IN0, .accessor_name = "cb_in0", .endpoint_type = DFBEndpointType::PRODUCER}},
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_TENSOR, .accessor_name = "src"}},
            .runtime_arg_schema =
                {.runtime_arg_names = {"num_tiles", "start_id", "start_ht", "start_wt", "Ht", "Wt", "HtWt"}},
            .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
        };

        KernelSpec writer_spec{
            .unique_id = WRITER_KERNEL,
            .source =
                std::filesystem::path{"ttnn/cpp/ttnn/operations/experimental/quasar/transpose/device/kernels/dataflow/"
                                      "writer_unary_interleaved_start_id.cpp"},
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = CB_OUT0, .accessor_name = "cb_out0", .endpoint_type = DFBEndpointType::CONSUMER}},
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "dst"}},
            .compile_time_args = {{"page_size", dst_single_tile_size}},
            .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
            .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
        };

        ttnn::ComputeKernelConfig compute_cfg{
            .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = fp32_dest_acc_en};
        ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(device->arch(), compute_cfg);
        if (src_is_float32) {
            std::visit(
                [&](auto& c) {
                    c.unpack_to_dest_mode.emplace(CB_IN0, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32);
                },
                compute_hw);
        }

        KernelSpec compute_spec{
            .unique_id = COMPUTE_KERNEL,
            .source =
                std::filesystem::path{
                    "ttnn/cpp/ttnn/operations/experimental/quasar/transpose/device/kernels/compute/transpose_wh.cpp"},
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = CB_IN0, .accessor_name = "cb_in0", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = CB_OUT0, .accessor_name = "cb_out0", .endpoint_type = DFBEndpointType::PRODUCER}},
            .runtime_arg_schema = {.runtime_arg_names = {"NHtWt"}},
            .hw_config = compute_hw,
        };

        kernels = {std::move(reader_spec), std::move(writer_spec), std::move(compute_spec)};

        // Tiled work walk uses padded-shape tile counts (preserved from legacy).
        const auto input_shape = input_tensor.padded_shape();
        const uint32_t Wt_walk = input_shape[3] / TILE_WIDTH;
        const uint32_t Ht_walk = input_shape[2] / TILE_HEIGHT;
        const uint32_t HtWt_walk = Ht_walk * Wt_walk;

        for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; i++) {
            const CoreCoord core = {i / num_cores_y, i % num_cores_y};
            uint32_t num_tiles_per_core;
            if (core_group_1.contains(core)) {
                num_tiles_per_core = num_tiles_per_core_group_1;
            } else if (core_group_2.contains(core)) {
                num_tiles_per_core = num_tiles_per_core_group_2;
            } else {
                num_tiles_per_core = 0;
            }

            const uint32_t h = num_tiles_read % Ht_walk;
            const uint32_t w = num_tiles_read / Ht_walk % Wt_walk;

            const NodeCoord node = core;
            KernelRunArgs::RuntimeArgValues& reader_rtas = reader_run.runtime_arg_values;
            KernelRunArgs::RuntimeArgValues& writer_rtas = writer_run.runtime_arg_values;
            KernelRunArgs::RuntimeArgValues& compute_rtas = compute_run.runtime_arg_values;
            AddRuntimeArgsForNode(
                reader_rtas,
                node,
                {
                    {"num_tiles", num_tiles_per_core},
                    {"start_id", tt::round_down(num_tiles_read, HtWt_walk) + (h * Wt_walk) + w},
                    {"start_ht", h},
                    {"start_wt", w},
                    {"Ht", Ht_walk},
                    {"Wt", Wt_walk},
                    {"HtWt", HtWt_walk},
                });
            compute_rtas["NHtWt"][node] = num_tiles_per_core;
            AddRuntimeArgsForNode(
                writer_rtas,
                node,
                {
                    {"num_pages", num_tiles_per_core},
                    {"start_id", num_tiles_read},
                });

            num_tiles_read += num_tiles_per_core;
        }
    }

    WorkUnitSpec wu{
        .name = "transpose_wh",
        .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL},
        .target_nodes = total_cores,
    };

    ProgramSpec spec{
        .name = "transpose_wh",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters = {input_param, output_param},
        .work_units = {wu},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run), std::move(compute_run)};
    run_args.tensor_args = {
        {INPUT_TENSOR, TensorArgument{std::cref(input_tensor.mesh_tensor())}},
        {OUTPUT_TENSOR, TensorArgument{std::cref(output_tensor.mesh_tensor())}}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
