// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/permute/device/permute_device_operation.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <vector>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement {

namespace detail {
uint32_t tile_volume(const ttnn::Tensor& input_tensor) {
    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    return tile_shape[0] * tile_shape[1];
}

uint32_t num_tiles(const ttnn::Tensor& input_tensor) {
    const auto& shape = input_tensor.padded_shape();
    auto tile_vol = tile_volume(input_tensor);
    return shape.volume() / tile_vol;
}

uint32_t tile_size(const ttnn::Tensor& input_tensor) {
    auto dataformat = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    return tt::tile_size(dataformat);
}

ttnn::Shape get_tiled_shape(const ttnn::Tensor& input_tensor) {
    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const auto& shape = input_tensor.padded_shape();
    ttsl::SmallVector<uint32_t> tiled_shape;
    tiled_shape.reserve(shape.rank());
    for (int i = 0; i < shape.rank(); i++) {
        uint32_t dim = 0;
        if (i == shape.rank() - 1) {
            dim = shape[i] / tile_shape[1];
        } else if (i == shape.rank() - 2) {
            dim = shape[i] / tile_shape[0];
        } else {
            dim = shape[i];
        }
        tiled_shape.push_back(dim);
    }
    auto res = ttnn::Shape(tiled_shape);
    return res;
}

ttsl::SmallVector<uint32_t> get_strides(const ttnn::Shape& shape) {
    ttsl::SmallVector<uint32_t> strides(shape.rank());
    strides[shape.rank() - 1] = 1;
    for (int i = shape.rank() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

// Function to compute the inverse of a permutation
ttsl::SmallVector<uint32_t> get_inverse_permutation(const ttsl::SmallVector<uint32_t>& perm) {
    // Get the size of the permutation
    size_t n = perm.size();

    // Create a vector for the inverse permutation
    ttsl::SmallVector<uint32_t> inverse_permutation(n);

    // Validate the input permutation
    ttsl::SmallVector<bool> seen(n, false);
    for (size_t i = 0; i < n; ++i) {
        if (perm[i] >= n || seen[perm[i]]) {
            TT_FATAL(false, "Invalid permutation: duplicate or out of range value");
        }
        seen[perm[i]] = true;
        inverse_permutation[perm[i]] = static_cast<uint32_t>(i);
    }

    return inverse_permutation;
}

uint32_t get_buffer_alignment(const ttnn::Tensor& tensor) {
    return (
        tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? tt::tt_metal::hal::get_dram_alignment()
                                                                         : tt::tt_metal::hal::get_l1_alignment());
}

}  // namespace detail

ttnn::device_operation::ProgramArtifacts PermuteDeviceOperation::MultiCoreTileInvariant::create_program_artifacts(
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
    uint32_t input_page_size = detail::tile_size(input_tensor);
    uint32_t num_tiles = detail::num_tiles(tensor_return_value);
    uint32_t num_input_pages_to_read = 2;

    uint32_t rank = operation_attributes.dims.size();
    bool swap_hw = operation_attributes.dims[rank - 2] == rank - 1 && operation_attributes.dims[rank - 1] == rank - 2;

    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    // ---- Resource names ----
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const DFBSpecName SRC0{"cb_src0"};    // legacy c_0
    const DFBSpecName OUT16{"cb_out16"};  // legacy c_16 (swap-hw only)
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    // The output DFB the writer drains: c_16 when swap-hw (compute produces it), else c_0.
    const DFBSpecName OUTPUT_CB = swap_hw ? OUT16 : SRC0;

    // ---- Dataflow buffers ----
    Group<DataflowBufferSpec> dataflow_buffers = {DataflowBufferSpec{
        .unique_id = SRC0,
        .entry_size = input_page_size,
        .num_entries = num_input_pages_to_read,
        .data_format_metadata = cb_data_format}};
    if (swap_hw) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = OUT16,
            .entry_size = input_page_size,
            .num_entries = num_input_pages_to_read,
            .data_format_metadata = cb_data_format});
    }

    // ---- Reader (own) ----
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
            "reader_permute_interleaved_tiled_invariant.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = SRC0, .accessor_name = "cb_in0", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args = {{"rank", rank}, {"page_size", input_page_size}, {"num_tiles", num_tiles}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_tile", "end_tile"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
        .advanced_options = {.num_runtime_varargs = 3 * rank},
    };

    // ---- Writer (donor fork: eltwise/unary interleaved writer, Metal 2.0 copy) ----
    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUTPUT_CB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    Group<KernelSpec> kernels = {std::move(reader), std::move(writer)};

    // ---- Compute (donor fork: transpose_wh, swap-hw only) ----
    if (swap_hw) {
        bool fp32_dest_acc_en = cb_data_format == tt::DataFormat::Float32 || cb_data_format == tt::DataFormat::Int32 ||
                                cb_data_format == tt::DataFormat::UInt32;
        ComputeHardwareConfig compute_cfg{.enable_32_bit_dest = fp32_dest_acc_en};
        // Legacy set unpack_to_dest_mode[c_0] = UnpackToDestFp32 for Float32 → UnpackMode::UnpackToDest.
        // Compute consumes SRC0 (c_0); the required-entry rule fires only for Float32.
        if (cb_data_format == tt::DataFormat::Float32) {
            compute_cfg.unpack_modes = {{SRC0, UnpackMode::UnpackToDest}};
        }
        kernels.push_back(KernelSpec{
            .unique_id = COMPUTE,
            .source = "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_metal2.cpp",
            .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = SRC0, .accessor_name = "cb_in", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = OUT16, .accessor_name = "cb_out", .endpoint_type = DFBEndpointType::PRODUCER}},
            .runtime_arg_schema = {.runtime_arg_names = {"NHtWt"}},
            .hw_config = std::move(compute_cfg),
        });
    }

    // ---- Reader varargs (core-invariant): output_tiled_shape ++ inv_perm ++ input_tile_strides ----
    // think of tensor as its tiled shape rather than its logical shape
    auto output_tiled_shape = detail::get_tiled_shape(tensor_return_value);
    auto input_tiled_shape = detail::get_tiled_shape(input_tensor);
    auto output_shape_view = output_tiled_shape.view();
    // read is less expensive than write, so read in order of output tensor, get relevant pre-permutation input tiles,
    // and then write it out to determine index in input tensor we need the input strides
    auto input_tile_strides = detail::get_strides(input_tiled_shape);
    // we also need the inverse permutation to map back to input tensor
    auto inv_perm = detail::get_inverse_permutation(operation_attributes.dims);
    std::vector<uint32_t> reader_varargs;
    reader_varargs.reserve(output_shape_view.size() + inv_perm.size() + input_tile_strides.size());
    reader_varargs.insert(reader_varargs.end(), output_shape_view.begin(), output_shape_view.end());
    reader_varargs.insert(reader_varargs.end(), inv_perm.begin(), inv_perm.end());
    reader_varargs.insert(reader_varargs.end(), input_tile_strides.begin(), input_tile_strides.end());

    // ---- Per-node run args ----
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    uint32_t start_tile = 0;
    uint32_t num_tiles_per_core = 0;
    for (const auto& core : cores) {
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op
            num_tiles_per_core = 0;
        }
        uint32_t end_tile = start_tile + num_tiles_per_core;

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"start_tile", start_tile}, {"end_tile", end_tile}});
        reader_run_args.advanced_options.runtime_varargs.emplace(core, reader_varargs);

        // writer unary: num_pages then start_id
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, core, {{"num_pages", num_tiles_per_core}, {"start_id", start_tile}});
        if (swap_hw) {
            AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"NHtWt", num_tiles_per_core}});
        }
        start_tile = end_tile;
    }

    // ---- Assemble ----
    ProgramSpec spec;
    spec.name = "permute_tiled_invariant";
    spec.kernels = std::move(kernels);
    spec.dataflow_buffers = std::move(dataflow_buffers);
    spec.tensor_parameters = {
        TensorParameter{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT, .spec = output_mesh_tensor.tensor_spec()}};
    Group<KernelSpecName> wu_kernels = {READER, WRITER};
    if (swap_hw) {
        wu_kernels.push_back(COMPUTE);
    }
    spec.work_units = {WorkUnitSpec{.name = "main", .kernels = wu_kernels, .target_nodes = all_cores}};

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    if (swap_hw) {
        run_args.kernel_run_args.push_back(std::move(compute_run_args));
    }
    run_args.tensor_args.insert({INPUT, input_mesh_tensor});
    run_args.tensor_args.insert({OUTPUT, output_mesh_tensor});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ttnn::device_operation::ProgramArtifacts PermuteDeviceOperation::MultiCoreTileRowInvariant::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;
    const float pad_value = operation_attributes.pad_value;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;
    auto dims = operation_attributes.dims;

    auto input_shape = input_tensor.logical_shape();

    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    auto face_shape = input_tensor.tensor_spec().tile().get_face_shape();

    auto padded_output_shape = output_tensor.padded_shape();
    uint32_t rank = operation_attributes.dims.size();
    bool swap_hw = dims[rank - 1] == rank - 2;

    if (swap_hw) {
        for (uint32_t i = 0; i < rank; i++) {
            if (dims[i] == rank - 2) {
                dims[i] = rank - 1;
            } else if (dims[i] == rank - 1) {
                dims[i] = rank - 2;
            }
        }
        std::swap(tile_shape[0], tile_shape[1]);
        std::swap(input_shape[rank - 2], input_shape[rank - 1]);
        std::swap(padded_output_shape[rank - 2], padded_output_shape[rank - 1]);
        std::swap(face_shape[0], face_shape[1]);
    }

    const auto& input_mesh_tensor = input_tensor.mesh_tensor();
    const auto& output_mesh_tensor = output_tensor.mesh_tensor();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_page_size = detail::tile_size(input_tensor);

    uint32_t num_tiles = detail::num_tiles(input_tensor);
    uint32_t num_output_tiles = detail::num_tiles(tensor_return_value);

    uint32_t num_input_pages_to_read = 2;

    uint32_t padded_num_tensor_tiles =
        num_output_tiles / (padded_output_shape[rank - 2] / tile_shape[0]);  // only last row of Xt should have padding

    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);
    auto
        [padded_num_cores,
         padded_all_cores,
         padded_core_group_1,
         padded_core_group_2,
         padded_num_tiles_per_core_group_1,
         padded_num_tiles_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, padded_num_tensor_tiles);

    all_cores = num_cores > padded_num_cores ? all_cores : padded_all_cores;

    uint32_t output_H = input_shape[dims[rank - 2]];
    uint32_t element_size = input_tensor.element_size();

    bool needs_padding = (output_H % tile_shape[1] != 0);

    uint32_t padding_val_packed = 0;
    uint32_t num_writes = 0;
    if (output_H % tile_shape[1] != 0) {
        uint32_t num_packed_values = sizeof(uint32_t) / element_size;
        num_writes = face_shape[1] / num_packed_values;
        switch (input_tensor.dtype()) {
            case DataType::INT32: padding_val_packed = std::bit_cast<uint32_t>(pad_value); break;
            case DataType::UINT32: padding_val_packed = pad_value; break;
            case DataType::BFLOAT16:
                padding_val_packed = pack_two_bfloat16_into_uint32({bfloat16(pad_value), bfloat16(pad_value)});
                break;
            case DataType::UINT16:
                padding_val_packed =
                    pack_two_uint16_into_uint32({float_to_uint16(pad_value), float_to_uint16(pad_value)});
                break;
            case DataType::FLOAT32: padding_val_packed = std::bit_cast<uint32_t>(pad_value); break;
            default:
                padding_val_packed = 0;
                TT_ASSERT(
                    false,
                    "Unsupported datatype for pad tile multicore, can only support INT32, UINT32, BFLOAT16, UINT16, "
                    "FLOAT32");
        }
    }

    uint32_t h_in_dest = 0;
    for (uint32_t i = 0; i < rank; i++) {
        if (dims[i] == rank - 2) {
            h_in_dest = i;
            break;
        }
    }

    uint32_t accumulated_outer_dims = 1;
    for (uint32_t i = 0; i < rank - 2; i++) {
        accumulated_outer_dims *= input_shape[i];
    }

    // ---- Resource names ----
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const DFBSpecName SRC0{"cb_src0"};    // legacy c_0
    const DFBSpecName PAD{"cb_pad"};      // legacy c_1 (padding, only when needs_padding)
    const DFBSpecName OUT16{"cb_out16"};  // legacy c_16 (swap-hw only)
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    // The output DFB the writer drains: c_16 when swap-hw (compute produces it), else c_0.
    const DFBSpecName OUTPUT_CB = swap_hw ? OUT16 : SRC0;

    // ---- Dataflow buffers ----
    Group<DataflowBufferSpec> dataflow_buffers = {DataflowBufferSpec{
        .unique_id = SRC0,
        .entry_size = input_page_size,
        .num_entries = num_input_pages_to_read,
        .data_format_metadata = cb_data_format}};
    if (needs_padding) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = PAD,
            .entry_size = face_shape[1] * element_size,
            .num_entries = 1,
            .data_format_metadata = cb_data_format});
    }
    if (swap_hw) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = OUT16,
            .entry_size = input_page_size,
            .num_entries = num_input_pages_to_read,
            .data_format_metadata = cb_data_format});
    }

    // ---- Conditional cb_pad (legacy c_1) bindings + gating define (needs_padding) ----
    KernelSpec::CompilerOptions::Defines pad_defines;
    Group<DFBBinding> reader_dfb = {
        DFBBinding{.dfb_spec_name = SRC0, .accessor_name = "cb_in0", .endpoint_type = DFBEndpointType::PRODUCER}};
    Group<DFBBinding> writer_dfb = {
        DFBBinding{.dfb_spec_name = OUTPUT_CB, .accessor_name = "cb_out", .endpoint_type = DFBEndpointType::CONSUMER}};
    Group<std::string> writer_rta_names = {"start_tile", "end_tile"};
    if (needs_padding) {
        pad_defines.insert({"NEEDS_PADDING", "1"});
        reader_dfb.push_back(
            DFBBinding{.dfb_spec_name = PAD, .accessor_name = "cb_pad", .endpoint_type = DFBEndpointType::PRODUCER});
        writer_dfb.push_back(
            DFBBinding{.dfb_spec_name = PAD, .accessor_name = "cb_pad", .endpoint_type = DFBEndpointType::CONSUMER});
        writer_rta_names.push_back("start_padding_tile_idx");
        writer_rta_names.push_back("end_padding_tile_idx");
    }

    // ---- Reader (donor fork: transpose padding-aware reader, Metal 2.0 copy) ----
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
            "reader_unary_transpose_hc_interleaved_tiled_padding_aware_metal2.cpp",
        .compiler_options = {.defines = pad_defines},
        .dfb_bindings = reader_dfb,
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args =
            {{"num_writes", num_writes},
             {"padding_val_packed", padding_val_packed},
             {"swap_hw", swap_hw},
             {"H", input_shape[rank - 1]},
             {"W", input_shape[rank - 2]},
             {"accumulated_outer_dims", accumulated_outer_dims},
             {"tile_height", tile_shape[1]},
             {"tile_width", tile_shape[0]}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    // ---- Writer (own) ----
    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
            "writer_permute_interleaved_tiled_row_invariant.cpp",
        .compiler_options = {.defines = pad_defines},
        .dfb_bindings = writer_dfb,
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args =
            {{"element_size", element_size},
             {"output_H", output_H},
             {"H", input_shape[rank - 2]},
             {"W", input_shape[rank - 1]},
             {"tile_height", tile_shape[0]},
             {"tile_width", tile_shape[1]},
             {"face_height", face_shape[0]},
             {"face_width", face_shape[1]},
             {"rank", rank},
             {"h_in_dest", h_in_dest}},
        .runtime_arg_schema = {.runtime_arg_names = writer_rta_names},
        .hw_config = ttnn::create_writer_datamovement_config(),
        .advanced_options = {.num_runtime_varargs = 2 * rank},
    };

    Group<KernelSpec> kernels = {std::move(reader), std::move(writer)};

    // ---- Compute (donor fork: transpose_wh, swap-hw only) ----
    if (swap_hw) {
        bool fp32_dest_acc_en = cb_data_format == tt::DataFormat::Float32 || cb_data_format == tt::DataFormat::Int32 ||
                                cb_data_format == tt::DataFormat::UInt32;
        ComputeHardwareConfig compute_cfg{.enable_32_bit_dest = fp32_dest_acc_en};
        // Legacy set unpack_to_dest_mode[c_0] = UnpackToDestFp32 for Float32 → UnpackMode::UnpackToDest.
        // Compute consumes SRC0 (c_0); the required-entry rule fires only for Float32.
        if (cb_data_format == tt::DataFormat::Float32) {
            compute_cfg.unpack_modes = {{SRC0, UnpackMode::UnpackToDest}};
        }
        kernels.push_back(KernelSpec{
            .unique_id = COMPUTE,
            .source = "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_metal2.cpp",
            .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = SRC0, .accessor_name = "cb_in", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = OUT16, .accessor_name = "cb_out", .endpoint_type = DFBEndpointType::PRODUCER}},
            .runtime_arg_schema = {.runtime_arg_names = {"NHtWt"}},
            .hw_config = std::move(compute_cfg),
        });
    }

    // ---- Writer varargs (core-invariant): input_shape ++ dims (count = 2*rank) ----
    auto input_shape_view = input_shape.view();
    std::vector<uint32_t> writer_varargs;
    writer_varargs.reserve(input_shape_view.size() + dims.size());
    writer_varargs.insert(writer_varargs.end(), input_shape_view.begin(), input_shape_view.end());
    writer_varargs.insert(writer_varargs.end(), dims.begin(), dims.end());

    // ---- Per-node run args ----
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    uint32_t start_tile = 0;
    uint32_t num_tiles_per_core = 0;
    uint32_t start_tile_padding = 0;
    uint32_t num_tiles_per_core_padding = 0;
    for (const auto& core : cores) {
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op
            num_tiles_per_core = 0;
        }
        if (needs_padding) {
            if (padded_core_group_1.contains(core)) {
                num_tiles_per_core_padding = padded_num_tiles_per_core_group_1;
            } else if (padded_core_group_2.contains(core)) {
                num_tiles_per_core_padding = padded_num_tiles_per_core_group_2;
            } else {
                // no-op
                num_tiles_per_core_padding = 0;
            }
        }
        uint32_t end_tile = start_tile + num_tiles_per_core;

        // reader (donor): num_tiles (num_pages) then start_id
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"num_tiles", num_tiles_per_core}, {"start_id", start_tile}});

        if (swap_hw) {
            AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"NHtWt", num_tiles_per_core}});
        }

        if (needs_padding) {
            uint32_t end_tile_padding = start_tile_padding + num_tiles_per_core_padding;
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"start_tile", start_tile},
                 {"end_tile", end_tile},
                 {"start_padding_tile_idx", start_tile_padding},
                 {"end_padding_tile_idx", end_tile_padding}});
            start_tile_padding = end_tile_padding;
        } else {
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values, core, {{"start_tile", start_tile}, {"end_tile", end_tile}});
        }
        writer_run_args.advanced_options.runtime_varargs.emplace(core, writer_varargs);

        start_tile = end_tile;
    }

    // ---- Assemble ----
    ProgramSpec spec;
    spec.name = "permute_tiled_row_invariant";
    spec.kernels = std::move(kernels);
    spec.dataflow_buffers = std::move(dataflow_buffers);
    spec.tensor_parameters = {
        TensorParameter{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT, .spec = output_mesh_tensor.tensor_spec()}};
    Group<KernelSpecName> wu_kernels = {READER, WRITER};
    if (swap_hw) {
        wu_kernels.push_back(COMPUTE);
    }
    spec.work_units = {WorkUnitSpec{.name = "main", .kernels = wu_kernels, .target_nodes = all_cores}};

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    if (swap_hw) {
        run_args.kernel_run_args.push_back(std::move(compute_run_args));
    }
    run_args.tensor_args.insert({INPUT, input_mesh_tensor});
    run_args.tensor_args.insert({OUTPUT, output_mesh_tensor});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ttnn::device_operation::ProgramArtifacts PermuteDeviceOperation::MultiCoreTiledGeneric::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // X = output width
    // Y = output height
    // input shape = (..., H, W)
    // output shape = (..., Y, X)

    /**
     * The algorithm is as follows:
     * 1. Read in blocks of data along the X and W dimensions (XW blocks, W is contiguous)
     *  a. TILE_HEIGHT rows along X with TILE_WIDTH elements across W
     * 2. Tilize, transpose, and untilize the data into a WX block
     * 3. Write out all the data in WX block to its correct position in the permuted output tensor buffer
     *  a. We write out on face/subtile line at a time
     *  a. X is the output width dimension, but it's tiled so we can only write out face/subtile line at a time
     * 4. Repeat until all XW blocks are processed
     * 5. If X is not a multiple of TILE_WIDTH, we pad the last face/subtile line with the pad value
     * 6. If Y is not a multiple of TILE_HEIGHT, we pad the last set of tiles on the Y dimension with the pad value
     *
     */

    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;
    const float pad_value = operation_attributes.pad_value;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_shape = input_tensor.logical_shape();
    const auto& dims = operation_attributes.dims;
    uint32_t rank = dims.size();
    auto& output_tensor = tensor_return_value;
    const auto& output_shape = output_tensor.logical_shape();
    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const auto& face_shape = input_tensor.tensor_spec().tile().get_face_shape();
    const auto& input_mesh_tensor = input_tensor.mesh_tensor();
    const auto& output_mesh_tensor = output_tensor.mesh_tensor();

    uint32_t logical_volume = input_shape.volume();
    uint32_t num_rows = logical_volume / input_shape[rank - 1];
    uint32_t y_dim_index_in_input = dims[rank - 2];

    uint32_t x_dim_index_in_input = dims[rank - 1];
    uint32_t x = input_shape[x_dim_index_in_input];
    uint32_t y = input_shape[y_dim_index_in_input];
    uint32_t w = input_shape[rank - 1];

    // X is the new width so we need to pad it to the target tile_shape[1]
    uint32_t x_block_size = tile_shape[1];
    uint32_t w_block_size = tile_shape[1];

    uint32_t element_size = input_tensor.element_size();
    uint32_t X_p = x_block_size * ((x + x_block_size - 1) / x_block_size);
    uint32_t W_p = tile_shape[1] * ((w + tile_shape[1] - 1) / tile_shape[1]);
    uint32_t H_p = tile_shape[0] * ((input_shape[rank - 2] + tile_shape[0] - 1) / tile_shape[0]);
    uint32_t H_t = H_p / tile_shape[0];
    uint32_t W_t = W_p / tile_shape[1];

    uint32_t subtile_line_bytes = face_shape[1] * element_size;
    uint32_t read_alignment = detail::get_buffer_alignment(input_tensor);
    uint32_t misalignment = read_alignment > subtile_line_bytes ? read_alignment - subtile_line_bytes : 0;

    uint32_t permuted_w_dim = 0;  // Will hold the position of w_dim in the permuted array
    for (uint32_t i = 0; i < rank; ++i) {
        if (dims[i] == rank - 1) {
            permuted_w_dim = i;
            break;
        }
    }

    uint32_t w_blocks = W_p / w_block_size;
    uint32_t x_blocks = X_p / x_block_size;

    uint32_t num_faces_w = tile_shape[1] / face_shape[1];

    uint32_t padded_xw_volume = X_p * W_p;
    for (uint32_t i = 0; i < rank - 1; i++) {
        if (i == x_dim_index_in_input) {
            continue;
        }
        padded_xw_volume *= input_shape[i];
    }

    uint32_t xw_blocks = padded_xw_volume / (tile_shape[0] * tile_shape[1]);

    bool needs_x_padding = (x % tile_shape[1] != 0);
    bool needs_y_padding =
        (y % tile_shape[0] != 0);  // if H is not moved, we could just keep existing implicit padding instead of
                                   // re-padding, but it complicates logic, may be worth investigating in the future
    bool needs_padding = needs_x_padding or needs_y_padding;

    uint32_t padding_val_packed = 0;
    uint32_t num_writes = 0;

    if (needs_padding) {
        uint32_t num_packed_values = sizeof(uint32_t) / element_size;
        num_writes = face_shape[1] / num_packed_values;

        switch (input_tensor.dtype()) {
            case DataType::INT32: padding_val_packed = std::bit_cast<uint32_t>(pad_value); break;
            case DataType::UINT32: padding_val_packed = pad_value; break;
            case DataType::BFLOAT16:
                padding_val_packed = pack_two_bfloat16_into_uint32({bfloat16(pad_value), bfloat16(pad_value)});
                break;
            case DataType::UINT16:
                padding_val_packed =
                    pack_two_uint16_into_uint32({float_to_uint16(pad_value), float_to_uint16(pad_value)});
                break;
            case DataType::FLOAT32: padding_val_packed = std::bit_cast<uint32_t>(pad_value); break;
            default:
                padding_val_packed = 0;
                TT_ASSERT(
                    false,
                    "Unsupported datatype for pad tile multicore, can only support INT32, UINT32, BFLOAT16, UINT16, "
                    "FLOAT32");
        }
    }

    // Faces with real data in the final tile along the width dimension, divided up
    uint32_t final_tile_real_w = w % tile_shape[1];
    uint32_t final_tile_real_faces_w =
        w % tile_shape[1] == 0 ? num_faces_w : ((final_tile_real_w + face_shape[1] - 1) / face_shape[1]);

    uint32_t final_tile_real_x = x % tile_shape[1];
    uint32_t final_tile_real_faces_x =
        needs_x_padding
            ? num_faces_w
            : (final_tile_real_x == 0 ? num_faces_w : ((final_tile_real_x + face_shape[1] - 1) / face_shape[1]));

    uint32_t num_output_tiles = detail::num_tiles(tensor_return_value);

    uint32_t num_input_pages_to_read = 2;

    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    // CoreCoord compute_with_storage_grid_size = {1u, 1u};
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, xw_blocks);

    uint32_t padded_num_tensor_tiles = num_output_tiles / (output_tensor.padded_shape()[rank - 2] /
                                                           tile_shape[0]);  // only last row of Xt should have padding
    auto
        [padded_num_cores,
         padded_all_cores,
         padded_core_group_1,
         padded_core_group_2,
         padded_num_tiles_per_core_group_1,
         padded_num_tiles_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, padded_num_tensor_tiles);

    all_cores = num_cores > padded_num_cores ? all_cores : padded_all_cores;

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_page_size = detail::tile_size(tensor_return_value) + misalignment;

    // ---- Resource names ----
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const DFBSpecName SRC_CB{"cb_in"};         // legacy c_0 (reader → compute)
    const DFBSpecName TILIZE_CB{"cb_tilize"};  // legacy c_1 (compute self-loop)
    const DFBSpecName OUT_CB{"cb_out"};        // legacy c_2 (compute → writer)
    const DFBSpecName PAD_CB{"cb_pad"};        // legacy c_3 (reader → writer, only when needs_y_padding)
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    // ---- Dataflow buffers ----
    Group<DataflowBufferSpec> dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = SRC_CB,
            .entry_size = input_page_size,
            .num_entries = num_input_pages_to_read,
            .data_format_metadata = cb_data_format},
        DataflowBufferSpec{
            .unique_id = TILIZE_CB,
            .entry_size = input_page_size,
            .num_entries = num_input_pages_to_read,
            .data_format_metadata = cb_data_format},
        DataflowBufferSpec{
            .unique_id = OUT_CB,
            .entry_size = input_page_size,
            .num_entries = num_input_pages_to_read,
            .data_format_metadata = cb_data_format},
    };
    if (needs_y_padding) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = PAD_CB,
            .entry_size = face_shape[1] * element_size,
            .num_entries = 1,
            .data_format_metadata = cb_data_format});
    }

    uint32_t non_x_rows = num_rows / x;

    // ---- Compute config (Style B: build ComputeGen1Config directly, mirroring legacy) ----
    bool fp32_dest_acc_en = cb_data_format == tt::DataFormat::Float32 || cb_data_format == tt::DataFormat::Int32 ||
                            cb_data_format == tt::DataFormat::UInt32;
    ComputeHardwareConfig compute_cfg{.enable_32_bit_dest = fp32_dest_acc_en};
    // Metal 2.0 requires an explicit unpack_modes entry for each Float32 DFB the compute kernel consumes
    // when enable_32_bit_dest = true. Compute consumes SRC_CB (via tilize) and TILIZE_CB (self-loop).
    // Keep both the tilize input (c_0 = SRC_CB) and its output (c_1 = TILIZE_CB, which feeds the transpose)
    // in full Float32 on the unpack-to-dest path; otherwise the unpacker falls back to tf32 and drops the
    // low mantissa bits. UnpackToDest is the Metal 2.0 equivalent of the legacy UnpackToDestFp32.
    // (Float32-only; Int32/UInt32 deferred, #49936.)
    if (cb_data_format == tt::DataFormat::Float32) {
        compute_cfg.unpack_modes = {{SRC_CB, UnpackMode::UnpackToDest}, {TILIZE_CB, UnpackMode::UnpackToDest}};
    }

    // ---- Conditional cb_pad (legacy c_3) bindings + gating define (needs_y_padding) ----
    KernelSpec::CompilerOptions::Defines pad_defines;
    Group<DFBBinding> reader_dfb = {
        DFBBinding{.dfb_spec_name = SRC_CB, .accessor_name = "cb_in", .endpoint_type = DFBEndpointType::PRODUCER}};
    Group<DFBBinding> writer_dfb = {
        DFBBinding{.dfb_spec_name = OUT_CB, .accessor_name = "cb_out", .endpoint_type = DFBEndpointType::CONSUMER}};
    Group<std::string> writer_rta_names = {"start_block", "end_block"};
    if (needs_y_padding) {
        pad_defines.insert({"NEEDS_Y_PADDING", "1"});
        reader_dfb.push_back(
            DFBBinding{.dfb_spec_name = PAD_CB, .accessor_name = "cb_pad", .endpoint_type = DFBEndpointType::PRODUCER});
        writer_dfb.push_back(
            DFBBinding{.dfb_spec_name = PAD_CB, .accessor_name = "cb_pad", .endpoint_type = DFBEndpointType::CONSUMER});
        writer_rta_names.push_back("start_padding_tile_idx");
        writer_rta_names.push_back("end_padding_tile_idx");
    }

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
            "reader_permute_interleaved_tiled_generic.cpp",
        .compiler_options = {.defines = pad_defines},
        .dfb_bindings = reader_dfb,
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args =
            {{"rank", rank},
             {"element_size", element_size},
             {"tile_height", tile_shape[0]},
             {"tile_width", tile_shape[1]},
             {"face_height", face_shape[0]},
             {"face_width", face_shape[1]},
             {"x_dim_index_in_input", x_dim_index_in_input},
             {"X", x},
             {"W", w},
             {"H", input_shape[rank - 2]},
             {"X_p", X_p},
             {"W_p", W_p},
             {"H_p", H_p},
             {"H_t", H_t},
             {"W_t", W_t},
             {"final_tile_real_w", final_tile_real_w},
             {"final_tile_real_faces_w", final_tile_real_faces_w},
             {"xw_blocks", xw_blocks},
             {"x_blocks", x_blocks},
             {"w_blocks", w_blocks},
             {"num_writes", num_writes},
             {"padding_val_packed", padding_val_packed},
             {"needs_x_padding", needs_x_padding},
             {"rows_per_x", non_x_rows},
             {"misalignment", misalignment},
             {"read_alignment", read_alignment}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_block", "end_block"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
        .advanced_options = {.num_runtime_varargs = 2 * rank},
    };

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/compute/transpose_xw_tiled.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = SRC_CB, .accessor_name = "cb_in", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = TILIZE_CB, .accessor_name = "cb_tilize", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = TILIZE_CB, .accessor_name = "cb_tilize", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = OUT_CB, .accessor_name = "cb_out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_block", "end_block"}},
        .hw_config = std::move(compute_cfg),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
            "writer_permute_interleaved_tiled_generic.cpp",
        .compiler_options = {.defines = pad_defines},
        .dfb_bindings = writer_dfb,
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args =
            {{"rank", rank},
             {"element_size", element_size},
             {"tile_height", tile_shape[0]},
             {"tile_width", tile_shape[1]},
             {"face_height", face_shape[0]},
             {"face_width", face_shape[1]},
             {"x_dim_index_in_input", x_dim_index_in_input},
             {"X", x},
             {"W", w},
             {"Y", output_shape[rank - 2]},
             {"X_p", X_p},
             {"W_p", W_p},
             {"rows_per_x", non_x_rows},
             {"W_t", W_t},
             {"final_tile_real_x", final_tile_real_x},
             {"final_tile_real_faces_x", final_tile_real_faces_x},
             {"xw_blocks", xw_blocks},
             {"x_blocks", x_blocks},
             {"w_blocks", w_blocks},
             {"permuted_w_dim", permuted_w_dim}},
        .runtime_arg_schema = {.runtime_arg_names = writer_rta_names},
        .hw_config = ttnn::create_writer_datamovement_config(),
        .advanced_options = {.num_runtime_varargs = 2 * rank},
    };

    auto input_shape_view = input_shape.view();
    // Reader/writer varargs (core-invariant): input_shape ++ dims (count = 2*rank; rank is a CTA).
    std::vector<uint32_t> common_varargs;
    common_varargs.reserve(input_shape_view.size() + dims.size());
    common_varargs.insert(common_varargs.end(), input_shape_view.begin(), input_shape_view.end());
    common_varargs.insert(common_varargs.end(), dims.begin(), dims.end());

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    uint32_t start_block = 0;
    uint32_t num_blocks_per_core = 0;
    uint32_t num_tiles_per_core_padding = 0;
    uint32_t start_tile_padding = 0;
    for (const auto& core : cores) {
        if (core_group_1.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_2;
        } else {
            // no-op
            num_blocks_per_core = 0;
        }
        // Y-padding tiles are the only padding the writer consumes; accumulate only then.
        if (needs_y_padding) {
            if (padded_core_group_1.contains(core)) {
                num_tiles_per_core_padding = padded_num_tiles_per_core_group_1;
            } else if (padded_core_group_2.contains(core)) {
                num_tiles_per_core_padding = padded_num_tiles_per_core_group_2;
            } else {
                // no-op
                num_tiles_per_core_padding = 0;
            }
        }

        uint32_t end_block = start_block + num_blocks_per_core;

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"start_block", start_block}, {"end_block", end_block}});
        reader_run_args.advanced_options.runtime_varargs.emplace(core, common_varargs);

        AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values, core, {{"start_block", start_block}, {"end_block", end_block}});

        if (needs_y_padding) {
            uint32_t end_tile_padding = start_tile_padding + num_tiles_per_core_padding;
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"start_block", start_block},
                 {"end_block", end_block},
                 {"start_padding_tile_idx", start_tile_padding},
                 {"end_padding_tile_idx", end_tile_padding}});
            start_tile_padding = end_tile_padding;
        } else {
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values, core, {{"start_block", start_block}, {"end_block", end_block}});
        }
        writer_run_args.advanced_options.runtime_varargs.emplace(core, common_varargs);

        start_block = end_block;
    }

    // ---- Assemble ----
    ProgramSpec spec;
    spec.name = "permute_tiled_generic";
    spec.kernels = {std::move(reader), std::move(writer), std::move(compute)};
    spec.dataflow_buffers = std::move(dataflow_buffers);
    spec.tensor_parameters = {
        TensorParameter{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT, .spec = output_mesh_tensor.tensor_spec()}};
    spec.work_units = {WorkUnitSpec{.name = "main", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_cores}};

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args), std::move(compute_run_args)};
    run_args.tensor_args.insert({INPUT, input_mesh_tensor});
    run_args.tensor_args.insert({OUTPUT, output_mesh_tensor});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::data_movement
