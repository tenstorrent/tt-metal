// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/permute/device/permute_device_operation.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::data_movement {

namespace detail {
uint32_t num_pages(const ttnn::Tensor& input_tensor) {
    const auto& shape = input_tensor.logical_shape();
    return shape.volume() / shape[-1];
}

uint32_t page_size(const ttnn::Tensor& input_tensor) {
    auto BUFFER_ALIGNMENT = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                ? tt::tt_metal::hal::get_dram_alignment()
                                : tt::tt_metal::hal::get_l1_alignment();
    const auto& shape = input_tensor.logical_shape();  // in anticipation of RM padding
    return tt::round_up(shape[-1] * input_tensor.element_size(), BUFFER_ALIGNMENT);
}

std::vector<uint32_t> get_row_strides(const ttnn::Shape& shape) {
    std::vector<uint32_t> strides(shape.rank());
    strides[shape.rank() - 1] = 1;
    strides[shape.rank() - 2] = 1;
    for (int i = shape.rank() - 3; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

}  // namespace detail

ttnn::device_operation::ProgramArtifacts PermuteDeviceOperation::MultiCoreRowInvariant::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;
    const auto& input_mesh_tensor = input_tensor.mesh_tensor();
    const auto& output_mesh_tensor = output_tensor.mesh_tensor();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_rm_page_size = detail::page_size(input_tensor);
    uint32_t output_rm_page_size = detail::page_size(tensor_return_value);

    uint32_t num_input_pages_to_read = 2;
    uint32_t num_rows = detail::num_pages(input_tensor);
    uint32_t N = operation_attributes.dims.size();

    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows);

    // ---- Resource names ----
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const DFBSpecName SRC_CB{"cb_src"};  // legacy c_0
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};

    // ---- Dataflow buffer (legacy c_0): reader produces, writer consumes ----
    DataflowBufferSpec src_cb{
        .unique_id = SRC_CB,
        .entry_size = input_rm_page_size,
        .num_entries = num_input_pages_to_read,
        .data_format_metadata = cb_data_format,
    };

    // ---- Tensor parameters (both Case 1, via TensorAccessor) ----
    TensorParameter input_param{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output_mesh_tensor.tensor_spec()};

    // ---- Kernels ----
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
            "reader_permute_interleaved_rm_row_invariant.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SRC_CB, .accessor_name = "cb_src", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args = {{"N", N}, {"page_size", input_rm_page_size}, {"num_rows", num_rows}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_row", "end_row"}},
        .hw_config = ttnn::create_reader_datamovement_config(input_tensor.device()->arch()),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
            "writer_permute_interleaved_rm_row_invariant.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SRC_CB, .accessor_name = "cb_src", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args = {{"N", N}, {"page_size", output_rm_page_size}, {"num_rows", num_rows}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_row", "end_row"}},
        // rank-length shape/perm/stride arrays (count = N, a CTA) → runtime varargs.
        .hw_config = ttnn::create_writer_datamovement_config(input_tensor.device()->arch()),
        .advanced_options = {.num_runtime_varargs = 3 * N},
    };

    // ---- Writer varargs (core-invariant): input_shape ++ perm ++ output_strides ----
    auto input_shape_view = input_tensor.logical_shape().view();
    auto output_strides = detail::get_row_strides(output_tensor.logical_shape());  // in anticipation of RM padding
    std::vector<uint32_t> writer_varargs;
    writer_varargs.reserve(input_shape_view.size() + operation_attributes.dims.size() + output_strides.size());
    writer_varargs.insert(writer_varargs.end(), input_shape_view.begin(), input_shape_view.end());
    writer_varargs.insert(writer_varargs.end(), operation_attributes.dims.begin(), operation_attributes.dims.end());
    writer_varargs.insert(writer_varargs.end(), output_strides.begin(), output_strides.end());

    // ---- Per-node runtime args (legacy node-first loop, bridged name-first) ----
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    uint32_t start_row = 0;
    uint32_t num_rows_per_core = 0;
    for (const auto& core : cores) {
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op
            num_rows_per_core = 0;
        }
        uint32_t end_row = start_row + num_rows_per_core;
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"start_row", start_row}, {"end_row", end_row}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, core, {{"start_row", start_row}, {"end_row", end_row}});
        writer_run_args.advanced_options.runtime_varargs.emplace(core, writer_varargs);
        start_row = end_row;
    }

    // ---- Assemble ----
    ProgramSpec spec;
    spec.name = "permute_rm_row_invariant";
    spec.kernels = {std::move(reader), std::move(writer)};
    spec.dataflow_buffers = {std::move(src_cb)};
    spec.tensor_parameters = {std::move(input_param), std::move(output_param)};
    spec.work_units = {WorkUnitSpec{.name = "main", .kernels = {READER, WRITER}, .target_nodes = all_cores}};

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args.insert({INPUT, input_mesh_tensor});
    run_args.tensor_args.insert({OUTPUT, output_mesh_tensor});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ttnn::device_operation::ProgramArtifacts PermuteDeviceOperation::MultiCoreBlockedGeneric::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;
    const auto& input_mesh_tensor = input_tensor.mesh_tensor();
    const auto& output_mesh_tensor = output_tensor.mesh_tensor();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t w_block_size = constants::TILE_WIDTH;
    uint32_t input_cb_page_size = w_block_size * input_tensor.element_size();

    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t x_block_size = constants::TILE_HEIGHT;
    uint32_t output_cb_page_size = x_block_size * input_tensor.element_size();

    uint32_t num_input_pages_to_read = 2;

    // we are focused on reading one row at a time, in a pattern that allows us to write an entire output row at a time
    // if W is being swapped with another dim X (e.g. H), then we need to read X rows at a time (X is the new row
    // dimension) CB is thus X pages in size (X*W*element_size) we read in X input rows of size W, and write out W
    // output rows of size X find the new row dimension (X)

    uint32_t x_dim = operation_attributes.dims.back();
    uint32_t X = input_tensor.logical_shape()[x_dim];
    // stride from one row to the next for each dim in the input tensor
    auto input_strides = detail::get_row_strides(input_tensor.logical_shape());
    uint32_t X_stride = input_strides[x_dim];

    auto output_strides = detail::get_row_strides(output_tensor.logical_shape());
    // after we transpose X and W, we need to stride from one row to the next for each dim in the output tensor
    uint32_t W = input_tensor.logical_shape()[-1];
    uint32_t W_stride = output_strides[x_dim];

    uint32_t N = operation_attributes.dims.size();
    uint32_t num_rows = detail::num_pages(input_tensor);

    // treat the input tensor as 3D with rows * x_blocks * w_blocks
    uint32_t x_blocks = tt::div_up(X, x_block_size);
    uint32_t w_blocks = tt::div_up(W, w_block_size);
    uint32_t num_blocks_total = (num_rows / X) * x_blocks * w_blocks;

    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks_total);

    // ---- Resource names ----
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const DFBSpecName SRC_CB{"cb_in"};         // legacy c_0 (reader → compute)
    const DFBSpecName TILIZE_CB{"cb_tilize"};  // legacy c_1 (compute self-loop)
    const DFBSpecName OUT_CB{"cb_out"};        // legacy c_2 (compute → writer)
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    // ---- Dataflow buffers (entry_size/num_entries carried from legacy CB total_size/page_size) ----
    DataflowBufferSpec src_cb{
        .unique_id = SRC_CB,
        .entry_size = input_cb_page_size,
        .num_entries = num_input_pages_to_read * x_block_size,
        .data_format_metadata = cb_data_format,
    };
    DataflowBufferSpec out_cb{
        .unique_id = OUT_CB,
        .entry_size = output_cb_page_size,
        .num_entries = num_input_pages_to_read * w_block_size,
        .data_format_metadata = cb_data_format,
    };
    DataflowBufferSpec tilize_cb{
        .unique_id = TILIZE_CB,
        .entry_size = x_block_size * w_block_size * input_tensor.element_size(),
        .num_entries = num_input_pages_to_read,
        .data_format_metadata = cb_data_format,
    };

    // ---- Tensor parameters ----
    TensorParameter input_param{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output_mesh_tensor.tensor_spec()};

    // ---- Kernels ----
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
            "reader_permute_interleaved_rm_blocked_generic.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SRC_CB, .accessor_name = "cb_in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args =
            {{"N", N},
             {"page_size", input_cb_page_size},
             {"num_rows", num_rows},
             {"x_dim", x_dim},
             {"num_blocks_total", num_blocks_total},
             {"x_blocks", x_blocks},
             {"w_blocks", w_blocks},
             {"x_block_size", x_block_size},
             {"w_block_size", w_block_size},
             {"element_size", input_tensor.element_size()}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_block", "end_block"}},
        .hw_config = ttnn::create_reader_datamovement_config(input_tensor.device()->arch()),
        .advanced_options = {.num_runtime_varargs = 2 * N},
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
            "writer_permute_interleaved_rm_blocked_generic.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_CB, .accessor_name = "cb_out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args =
            {{"N", N},
             {"output_page_size", output_cb_page_size},
             {"num_rows", num_rows},
             {"X", X},
             {"X_stride", X_stride},
             {"x_dim", x_dim},
             {"W_stride", W_stride},
             {"input_page_size", input_cb_page_size},
             {"element_size", input_tensor.element_size()},
             {"num_blocks_total", num_blocks_total},
             {"x_blocks", x_blocks},
             {"w_blocks", w_blocks},
             {"x_block_size", x_block_size},
             {"w_block_size", w_block_size},
             {"W", W}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_block", "end_block"}},
        .hw_config = ttnn::create_writer_datamovement_config(input_tensor.device()->arch()),
        .advanced_options = {.num_runtime_varargs = 3 * N},
    };

    bool fp32_dest_acc_en = cb_data_format_output == tt::DataFormat::Float32 ||
                            cb_data_format_output == tt::DataFormat::Int32 ||
                            cb_data_format_output == tt::DataFormat::UInt32;
    // Style B compute config: build ComputeGen1Config directly, matching the legacy
    // ComputeConfigDescriptor{.fp32_dest_acc_en=...} (all other fields at legacy defaults).
    ComputeGen1Config compute_cfg{.enable_32_bit_dest = fp32_dest_acc_en};
    // Metal 2.0 requires an explicit unpack_modes entry for every Float32 DFB the compute
    // kernel consumes when enable_32_bit_dest = true. The compute kernel consumes SRC_CB
    // (via tilize) and TILIZE_CB (self-loop). UnpackToDest, not UnpackToSrc: SrcA/SrcB are
    // only 19 bits wide, so unpacking Float32 there truncates it to tf32. MultiCoreTiledGeneric
    // already sets UnpackToDest on the same two CBs.
    // (Float32-only; Int32/UInt32 not required yet — issue #49936.)
    if (cb_data_format == tt::DataFormat::Float32) {
        compute_cfg.unpack_modes = {{SRC_CB, UnpackMode::UnpackToDest}, {TILIZE_CB, UnpackMode::UnpackToDest}};
    }

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/compute/"
            "transpose_xw_rm_single_tile_size.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = SRC_CB, .accessor_name = "cb_in", .endpoint_type = DFBEndpointType::CONSUMER},
             // TILIZE_CB self-loop: produced (tilize) and consumed (transpose) by this kernel alone.
             DFBBinding{
                 .dfb_spec_name = TILIZE_CB, .accessor_name = "cb_tilize", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = TILIZE_CB, .accessor_name = "cb_tilize", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = OUT_CB, .accessor_name = "cb_out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .compile_time_args = {{"x_block_size", x_block_size}, {"w_block_size", w_block_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_blocks"}},
        .hw_config = ComputeHardwareConfig{std::move(compute_cfg)},
    };

    // ---- Varargs (core-invariant) ----
    auto input_shape_view = input_tensor.logical_shape().view();
    std::vector<uint32_t> reader_varargs;  // input_shape ++ input_strides
    reader_varargs.reserve(input_shape_view.size() + input_strides.size());
    reader_varargs.insert(reader_varargs.end(), input_shape_view.begin(), input_shape_view.end());
    reader_varargs.insert(reader_varargs.end(), input_strides.begin(), input_strides.end());

    std::vector<uint32_t> writer_varargs;  // input_shape ++ perm ++ output_strides
    writer_varargs.reserve(input_shape_view.size() + operation_attributes.dims.size() + output_strides.size());
    writer_varargs.insert(writer_varargs.end(), input_shape_view.begin(), input_shape_view.end());
    writer_varargs.insert(writer_varargs.end(), operation_attributes.dims.begin(), operation_attributes.dims.end());
    writer_varargs.insert(writer_varargs.end(), output_strides.begin(), output_strides.end());

    // ---- Per-node runtime args ----
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    uint32_t start_block = 0;
    uint32_t num_blocks_per_core = 0;
    for (const auto& core : cores) {
        if (core_group_1.contains(core)) {
            num_blocks_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_blocks_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op
            num_blocks_per_core = 0;
        }
        uint32_t end_block = start_block + num_blocks_per_core;

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"start_block", start_block}, {"end_block", end_block}});
        reader_run_args.advanced_options.runtime_varargs.emplace(core, reader_varargs);

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, core, {{"start_block", start_block}, {"end_block", end_block}});
        writer_run_args.advanced_options.runtime_varargs.emplace(core, writer_varargs);

        AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"num_blocks", num_blocks_per_core}});
        start_block = end_block;
    }

    // ---- Assemble ----
    ProgramSpec spec;
    spec.name = "permute_rm_blocked_generic";
    spec.kernels = {std::move(reader), std::move(writer), std::move(compute)};
    spec.dataflow_buffers = {std::move(src_cb), std::move(out_cb), std::move(tilize_cb)};
    spec.tensor_parameters = {std::move(input_param), std::move(output_param)};
    spec.work_units = {WorkUnitSpec{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_cores}};

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args), std::move(compute_run_args)};
    run_args.tensor_args.insert({INPUT, input_mesh_tensor});
    run_args.tensor_args.insert({OUTPUT, output_mesh_tensor});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::data_movement
