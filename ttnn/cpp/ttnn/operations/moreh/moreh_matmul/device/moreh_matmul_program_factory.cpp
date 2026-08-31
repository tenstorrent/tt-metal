// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "moreh_matmul_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_matmul {

void get_tensor_dim(ttsl::SmallVector<uint32_t>& dim, const ttnn::Shape& shape) {
    const auto rank = shape.rank();
    for (auto i = 0; i < rank; ++i) {
        auto idx = rank - 1 - i;

        // last 2-dim
        if (idx == rank - 1 || idx == rank - 2) {
            dim[i] = shape[idx] / tt::constants::TILE_HEIGHT;
        } else {
            dim[i] = shape[idx];
        }
    }

    log_debug(tt::LogOp, "rank {}", rank);
    for (auto i = 0; i < ttnn::MAX_NUM_DIMENSIONS; ++i) {
        log_debug(tt::LogOp, "dim[{}] = {}", i, dim[i]);
    }
}

ttsl::SmallVector<int64_t> find_reduce_dim(const ttnn::Shape& a_shape, const ttnn::Shape& b_shape) {
    ttsl::SmallVector<uint32_t> a_dim(ttnn::MAX_NUM_DIMENSIONS, 1);
    ttsl::SmallVector<uint32_t> b_dim(ttnn::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(a_dim, a_shape);
    get_tensor_dim(b_dim, b_shape);
    int32_t rank = std::max(a_shape.rank(), b_shape.rank());
    log_debug(tt::LogOp, "find_reduce_dim :{} rank {} a {} b {}", __LINE__, rank, a_shape.rank(), b_shape.rank());
    ttsl::SmallVector<int64_t> dims;
    // batch dims
    for (int i = 0; i < rank - 2; ++i) {
        int idx = rank - 1 - i;
        TT_FATAL(idx >= 0, "idx < 0");
        if (a_dim[idx] != b_dim[idx]) {
            dims.push_back(i);
            log_debug(tt::LogOp, "find_reduce_dim :{} push {} dim", __LINE__, i);
        }
    }
    return dims;
}

bool is_same_batch_dim(const Tensor& tensor_a, const Tensor& tensor_b) {
    // check batch dims
    const auto& a_shape = tensor_a.padded_shape();
    const auto& b_shape = tensor_b.padded_shape();
    ttsl::SmallVector<uint32_t> a_dim(ttnn::MAX_NUM_DIMENSIONS, 1);
    ttsl::SmallVector<uint32_t> b_dim(ttnn::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(a_dim, a_shape);
    get_tensor_dim(b_dim, b_shape);
    for (auto i = 2; i < ttnn::MAX_NUM_DIMENSIONS; ++i) {
        if (a_dim[i] != b_dim[i]) {
            log_debug(tt::LogOp, "{}:{} {} a_dim {} - b_dim {}", __func__, __LINE__, i, a_dim[i], b_dim[i]);
            return false;
        }
    }
    log_debug(tt::LogOp, "{}:{} batch dims are the same.", __func__, __LINE__);
    return true;
}

void get_tensor_stride(ttsl::SmallVector<uint32_t>& stride, ttsl::SmallVector<uint32_t>& dim) {
    stride[0] = 1;
    for (auto i = 1; i < ttnn::MAX_NUM_DIMENSIONS; ++i) {
        stride[i] = stride[i - 1] * dim[i - 1];
    }

    for (auto i = 0; i < ttnn::MAX_NUM_DIMENSIONS; ++i) {
        log_debug(tt::LogOp, "stride[{}] = {}", i, stride[i]);
    }
}

void get_not_bcast(
    ttsl::SmallVector<uint32_t>& input_not_bcast,
    ttsl::SmallVector<uint32_t>& input_dim,
    ttsl::SmallVector<uint32_t>& other_not_bcast,
    ttsl::SmallVector<uint32_t>& other_dim) {
    // first 2-dims are M,K and K,N
    // TODO: refaactoring
    for (auto i = 2; i < ttnn::MAX_NUM_DIMENSIONS; ++i) {
        if (input_dim[i] == other_dim[i]) {
            input_not_bcast[i] = 1;
            other_not_bcast[i] = 1;
        } else {
            if (input_dim[i] == 1) {
                input_not_bcast[i] = 0;
                other_not_bcast[i] = 1;
            } else {
                input_not_bcast[i] = 1;
                other_not_bcast[i] = 0;
            }
        }
    }

    for (auto i = 0; i < ttnn::MAX_NUM_DIMENSIONS; ++i) {
        log_debug(tt::LogOp, "not bcast [{}] input {} other {}", i, input_not_bcast[i], other_not_bcast[i]);
    }
}

ttnn::device_operation::ProgramArtifacts MorehMatmulOperation::MultiCoreProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const Tensor& input = tensor_args.input;
    const Tensor& other = tensor_args.other;
    const Tensor& output = tensor_return_value;

    const std::optional<const Tensor>& bias = tensor_args.bias;

    // Metal 2.0 tensor bindings operate on the underlying MeshTensor; the TensorArgument
    // is matched back to its input by MeshTensor identity, so bind the exact references.
    const auto& input_mt = tensor_args.input.mesh_tensor();
    const auto& other_mt = tensor_args.other.mesh_tensor();
    const auto& output_mt = tensor_return_value.mesh_tensor();

    bool transpose_input = operation_attributes.transpose_input;
    bool transpose_other = operation_attributes.transpose_other;

    const DeviceComputeKernelConfig& compute_kernel_config = init_device_compute_kernel_config(
        input.device()->arch(), operation_attributes.compute_kernel_config, tt::tt_metal::MathFidelity::HiFi4);

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    IDevice* device{input.device()};

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    tt::DataFormat cb_data_format{datatype_to_dataformat_converter(output.dtype())};
    const auto num_output_tiles{output.physical_volume() / tt::constants::TILE_HW};

    // input tensor
    const auto& input_shape = input.padded_shape();
    const auto& input_shape_wo_padding = input.logical_shape();
    log_debug(tt::LogOp, "input dim");
    ttsl::SmallVector<uint32_t> input_dim(ttnn::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(input_dim, input_shape);

    log_debug(tt::LogOp, "input stride");
    ttsl::SmallVector<uint32_t> input_stride(ttnn::MAX_NUM_DIMENSIONS);
    get_tensor_stride(input_stride, input_dim);

    // other tensor
    const auto& other_shape = other.padded_shape();
    const auto& other_shape_wo_padding = other.logical_shape();
    log_debug(tt::LogOp, "other dim");
    ttsl::SmallVector<uint32_t> other_dim(ttnn::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(other_dim, other_shape);

    log_debug(tt::LogOp, "other stride");
    ttsl::SmallVector<uint32_t> other_stride(ttnn::MAX_NUM_DIMENSIONS);
    get_tensor_stride(other_stride, other_dim);

    log_debug(tt::LogOp, "not bcast");
    ttsl::SmallVector<uint32_t> input_not_bcast(ttnn::MAX_NUM_DIMENSIONS, 1);
    ttsl::SmallVector<uint32_t> other_not_bcast(ttnn::MAX_NUM_DIMENSIONS, 1);
    get_not_bcast(input_not_bcast, input_dim, other_not_bcast, other_dim);

    // output tensor
    const auto& output_shape = output.padded_shape();
    log_debug(tt::LogOp, "output dim");
    ttsl::SmallVector<uint32_t> output_dim(ttnn::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(output_dim, output_shape);

    log_debug(tt::LogOp, "output stride");
    ttsl::SmallVector<uint32_t> output_stride(ttnn::MAX_NUM_DIMENSIONS);
    get_tensor_stride(output_stride, output_dim);

    // matrix shape
    uint32_t Kt = (transpose_input) ? (input_shape[-2] / tt::constants::TILE_HEIGHT)
                                    : (input_shape[-1] / tt::constants::TILE_WIDTH);
    uint32_t Mt = (transpose_input) ? (input_shape[-1] / tt::constants::TILE_WIDTH)
                                    : (input_shape[-2] / tt::constants::TILE_HEIGHT);
    uint32_t Nt = (transpose_other) ? (other_shape[-2] / tt::constants::TILE_HEIGHT)
                                    : (other_shape[-1] / tt::constants::TILE_WIDTH);
    log_debug(tt::LogOp, "{}:{} Mt {} Nt {} Kt {}", __func__, __LINE__, Mt, Nt, Kt);

    // bias tensor
    bool is_scalar_bias = false;
    if (bias.has_value()) {
        const auto& bias_tensor = bias.value();
        const auto& bias_shape_wo_padding = bias_tensor.logical_shape();
        is_scalar_bias = (bias_shape_wo_padding[-1] == 1) ? (true) : (false);
        log_debug(tt::LogOp, "{}:{} bias tensor. is_scalar_bias {}", __func__, __LINE__, is_scalar_bias);
    }

    // mask
    uint32_t input_mask_h = input_shape_wo_padding[-2] % tt::constants::TILE_HEIGHT;
    uint32_t input_mask_w = input_shape_wo_padding[-1] % tt::constants::TILE_WIDTH;
    uint32_t other_mask_h = other_shape_wo_padding[-2] % tt::constants::TILE_HEIGHT;
    uint32_t other_mask_w = other_shape_wo_padding[-1] % tt::constants::TILE_WIDTH;

    [[maybe_unused]] bool need_input_mask_h = (input_mask_h) ? (true) : (false);
    [[maybe_unused]] bool need_input_mask_w = (input_mask_w) ? (true) : (false);

    [[maybe_unused]] bool need_other_mask_h = (other_mask_h) ? (true) : (false);
    [[maybe_unused]] bool need_other_mask_w = (other_mask_w) ? (true) : (false);

    if (input_mask_h == 0) {
        input_mask_h = tt::constants::TILE_HEIGHT;
    }
    if (input_mask_w == 0) {
        input_mask_w = tt::constants::TILE_WIDTH;
    }
    if (other_mask_h == 0) {
        other_mask_h = tt::constants::TILE_HEIGHT;
    }
    if (other_mask_w == 0) {
        other_mask_w = tt::constants::TILE_WIDTH;
    }

    log_debug(
        tt::LogOp,
        "{}:{} {} {} mask_h {} mask_w {}",
        __func__,
        __LINE__,
        need_input_mask_h,
        need_input_mask_w,
        input_mask_h,
        input_mask_w);
    log_debug(
        tt::LogOp,
        "{}:{} {} {} mask_h {} mask_w {}",
        __func__,
        __LINE__,
        need_other_mask_h,
        need_other_mask_w,
        other_mask_h,
        other_mask_w);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);
    log_debug(
        tt::LogOp,
        "math_fidelity {} math_approx_mode {} fp32_dest_acc_en {} packer_l1_acc {}",
        math_fidelity,
        math_approx_mode,
        fp32_dest_acc_en,
        packer_l1_acc);
    ////////////////////////////////////////////////////////////////////////////
    //                         Core Grid Configuration For Workload
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_output_tiles_per_core_group_1,
         num_output_tiles_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid, num_output_tiles);

    log_debug(tt::LogOp, "{}:{} num_output_tiles: {}", __func__, __LINE__, num_output_tiles);
    log_debug(
        tt::LogOp,
        "{}:{} num_output_tiles_per_core_group1: {}, 2: {} ",
        __func__,
        __LINE__,
        num_output_tiles_per_core_group_1,
        num_output_tiles_per_core_group_2);

    ////////////////////////////////////////////////////////////////////////////
    //                         Resource names
    ////////////////////////////////////////////////////////////////////////////
    const DFBSpecName IN0{"in0"};    // legacy c_0  (input)
    const DFBSpecName IN1{"in1"};    // legacy c_1  (other)
    const DFBSpecName IN2{"in2"};    // legacy c_2  (mask for input)
    const DFBSpecName IN3{"in3"};    // legacy c_3  (mask for other)
    const DFBSpecName IN4{"in4"};    // legacy c_4  (bias)
    const DFBSpecName IM0{"im0"};    // legacy c_24 (matmul reload temp)
    const DFBSpecName IM1{"im1"};    // legacy c_25 (input transpose)
    const DFBSpecName IM2{"im2"};    // legacy c_26 (other transpose)
    const DFBSpecName IM3{"im3"};    // legacy c_27 (bias-add temp)
    const DFBSpecName OUT0{"out0"};  // legacy c_16 (output)

    const TensorParamName INPUT{"input"};
    const TensorParamName OTHER{"other"};
    const TensorParamName BIAS{"bias"};
    const TensorParamName OUTPUT{"output"};

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};

    ////////////////////////////////////////////////////////////////////////////
    //                         DataflowBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t{2};   // input
    const uint32_t in1_t{2};   // other
    const uint32_t in2_t{3};   // mask for input
    const uint32_t in3_t{3};   // mask for other
    const uint32_t in4_t{2};   // bias
    const uint32_t im0_t{1};   // temp
    const uint32_t im1_t{2};   // transpose for input
    const uint32_t im2_t{2};   // transpose for other
    const uint32_t im3_t{1};   // temp for bias add
    const uint32_t out0_t{2};  // output

    auto im0_data_format = (fp32_dest_acc_en) ? tt::DataFormat::Float32 : cb_data_format;
    auto im3_data_format = (fp32_dest_acc_en) ? tt::DataFormat::Float32 : cb_data_format;

    auto make_dfb = [](const DFBSpecName& id, uint32_t num_tiles, tt::DataFormat fmt) {
        return DataflowBufferSpec{
            .unique_id = id,
            .entry_size = tile_size(fmt),
            .num_entries = num_tiles,
            .data_format_metadata = fmt,
        };
    };

    Group<DataflowBufferSpec> dataflow_buffers = {
        make_dfb(IN0, in0_t, cb_data_format),
        make_dfb(IN1, in1_t, cb_data_format),
        make_dfb(IN2, in2_t, cb_data_format),
        make_dfb(IN3, in3_t, cb_data_format),
        make_dfb(IN4, in4_t, cb_data_format),
        make_dfb(IM0, im0_t, im0_data_format),
        make_dfb(IM1, im1_t, cb_data_format),
        make_dfb(IM2, im2_t, cb_data_format),
        make_dfb(IM3, im3_t, im3_data_format),
        make_dfb(OUT0, out0_t, cb_data_format),
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Tensor parameters
    ////////////////////////////////////////////////////////////////////////////
    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = INPUT, .spec = input_mt.tensor_spec()},
        TensorParameter{.unique_id = OTHER, .spec = other_mt.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT, .spec = output_mt.tensor_spec()},
    };
    if (bias.has_value()) {
        tensor_parameters.push_back(
            TensorParameter{.unique_id = BIAS, .spec = bias.value().mesh_tensor().tensor_spec()});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelSpec::CompilerOptions::Defines reader_defines;
    if (bias.has_value()) {
        reader_defines.emplace("FUSE_BIAS", "1");
        log_debug(tt::LogOp, "{}:{} bias tensor. is bias dram {}", __func__, __LINE__, is_dram(bias));
    }

    KernelSpec::CompileTimeArgs reader_compile_time_args = {
        {"Kt", Kt},
        {"transpose_input", static_cast<uint32_t>(transpose_input)},
        {"transpose_other", static_cast<uint32_t>(transpose_other)},
        {"input_mask_h", input_mask_h},
        {"input_mask_w", input_mask_w},
        {"other_mask_h", other_mask_h},
        {"other_mask_w", other_mask_w},
    };
    if (bias.has_value()) {
        reader_compile_time_args["is_scalar_bias"] = static_cast<uint32_t>(is_scalar_bias);
    }

    Group<DFBBinding> reader_dfb_bindings = {
        DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = IN1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = IN2, .accessor_name = "in2", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = IN3, .accessor_name = "in3", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = IN4, .accessor_name = "in4", .endpoint_type = DFBEndpointType::PRODUCER},
    };
    Group<TensorBinding> reader_tensor_bindings = {
        TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"},
        TensorBinding{.tensor_parameter_name = OTHER, .accessor_name = "other"},
    };
    if (bias.has_value()) {
        reader_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = BIAS, .accessor_name = "bias"});
    }

    KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/moreh/moreh_matmul/device/kernels/reader_moreh_matmul.cpp",
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings = reader_dfb_bindings,
        .tensor_bindings = reader_tensor_bindings,
        .compile_time_args = reader_compile_time_args,
        // input_stride[8], other_stride[8], output_stride[8], input_not_bcast[8], other_not_bcast[8]
        .runtime_arg_schema = {.runtime_arg_names = {"output_tile_start_idx", "num_output_tiles"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
        .advanced_options = {.num_runtime_varargs = 5 * static_cast<uint32_t>(ttnn::MAX_NUM_DIMENSIONS)},
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/moreh/moreh_matmul/device/kernels/writer_moreh_matmul.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT0, .accessor_name = "out0", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_id", "num_output_tiles"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    log_debug(
        tt::LogOp,
        "{}:{} DMVK is_dram(input): {}, is_dram(other): {}, is_dram(output): {}",
        __func__,
        __LINE__,
        is_dram(input),
        is_dram(other),
        is_dram(output));

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (bias.has_value()) {
        compute_defines.emplace("FUSE_BIAS", "1");
    }
    if (fp32_dest_acc_en) {
        compute_defines.emplace("FP32_DEST_ACC_EN", "1");
    }

    const char* const compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_matmul/device/kernels/moreh_matmul.cpp";

    auto make_compute = [&](const KernelSpecName& id, uint32_t num_output_tiles_group) {
        // Style A: translate the resolved TTNN ComputeKernelConfig; helper handles the four knobs
        // (math_fidelity, math_approx_mode->sfpu_precision_mode, fp32_dest_acc_en->enable_32_bit_dest,
        // dst_full_sync_en->double_buffer_dest inverted) and the arch selection.
        auto compute_hw = ttnn::to_compute_hardware_config(ttnn::ComputeKernelConfig{
            .math_fidelity = math_fidelity,
            .math_approx_mode = math_approx_mode,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en});
        if (fp32_dest_acc_en) {
            // Legacy set unpack_to_dest_mode[c_24] = UnpackToDestFp32 (rest Default). Metal 2.0 requires
            // an explicit unpack_mode for every Float32 DFB consumed under enable_32_bit_dest: IM0 (c_24)
            // and IM3 (c_27) are Float32 here. IM0 -> UnpackToDest (legacy UnpackToDestFp32), IM3 ->
            // UnpackToSrc (legacy Default).
            compute_hw.unpack_modes = {
                {IM0, UnpackMode::UnpackToDest},
                {IM3, UnpackMode::UnpackToSrc},
            };
        }

        KernelSpec::CompileTimeArgs compute_args = {
            {"num_output_tiles", num_output_tiles_group},
            {"Mt", Mt},
            {"Nt", Nt},
            {"Kt", Kt},
            {"transpose_input", static_cast<uint32_t>(transpose_input)},
            {"transpose_other", static_cast<uint32_t>(transpose_other)},
            {"input_mask_h", input_mask_h},
            {"input_mask_w", input_mask_w},
            {"other_mask_h", other_mask_h},
            {"other_mask_w", other_mask_w},
        };
        if (bias.has_value()) {
            compute_args["is_scalar_bias"] = static_cast<uint32_t>(is_scalar_bias);
        }

        Group<DFBBinding> compute_dfb_bindings = {
            DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = IN1, .accessor_name = "in1", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = IN2, .accessor_name = "in2", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = IN3, .accessor_name = "in3", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = IN4, .accessor_name = "in4", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = OUT0, .accessor_name = "out0", .endpoint_type = DFBEndpointType::PRODUCER},
            // Self-loop intermediates (compute both produces and consumes; shared accessor name).
            DFBBinding{.dfb_spec_name = IM0, .accessor_name = "im0", .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{.dfb_spec_name = IM0, .accessor_name = "im0", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = IM1, .accessor_name = "im1", .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{.dfb_spec_name = IM1, .accessor_name = "im1", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = IM2, .accessor_name = "im2", .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{.dfb_spec_name = IM2, .accessor_name = "im2", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = IM3, .accessor_name = "im3", .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{.dfb_spec_name = IM3, .accessor_name = "im3", .endpoint_type = DFBEndpointType::CONSUMER},
        };

        return KernelSpec{
            .unique_id = id,
            .source = compute_kernel_file,
            .compiler_options = {.defines = compute_defines},
            .dfb_bindings = compute_dfb_bindings,
            .compile_time_args = compute_args,
            // output_stride[8]
            .runtime_arg_schema = {.runtime_arg_names = {"output_tile_start_idx"}},
            .hw_config = std::move(compute_hw),
            .advanced_options = {.num_runtime_varargs = static_cast<uint32_t>(ttnn::MAX_NUM_DIMENSIONS)},
        };
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Kernels + WorkUnits
    ////////////////////////////////////////////////////////////////////////////
    Group<KernelSpec> kernels = {reader, writer};
    Group<WorkUnitSpec> work_units;

    kernels.push_back(make_compute(COMPUTE_G1, num_output_tiles_per_core_group_1));
    work_units.push_back(
        WorkUnitSpec{.name = "moreh_mm_g1", .kernels = {READER, WRITER, COMPUTE_G1}, .target_nodes = core_group_1});

    const bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        kernels.push_back(make_compute(COMPUTE_G2, num_output_tiles_per_core_group_2));
        work_units.push_back(
            WorkUnitSpec{.name = "moreh_mm_g2", .kernels = {READER, WRITER, COMPUTE_G2}, .target_nodes = core_group_2});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    // The five reader dimensional arrays and the compute output_stride array are homogeneous,
    // node-invariant, and read positionally in a loop -> runtime varargs (patterns catalog:
    // homogeneous literal-count array). Same values on every node; built once here.
    std::vector<uint32_t> reader_varargs;
    reader_varargs.reserve(5 * ttnn::MAX_NUM_DIMENSIONS);
    reader_varargs.insert(reader_varargs.end(), input_stride.begin(), input_stride.end());
    reader_varargs.insert(reader_varargs.end(), other_stride.begin(), other_stride.end());
    reader_varargs.insert(reader_varargs.end(), output_stride.begin(), output_stride.end());
    reader_varargs.insert(reader_varargs.end(), input_not_bcast.begin(), input_not_bcast.end());
    reader_varargs.insert(reader_varargs.end(), other_not_bcast.begin(), other_not_bcast.end());

    std::vector<uint32_t> compute_varargs(output_stride.begin(), output_stride.end());

    KernelRunArgs::RuntimeArgValues reader_rta;
    KernelRunArgs::RuntimeArgValues writer_rta;
    KernelRunArgs::RuntimeArgValues compute_g1_rta;
    KernelRunArgs::RuntimeArgValues compute_g2_rta;
    AdvancedKernelRunArgs reader_adv;
    AdvancedKernelRunArgs compute_g1_adv;
    AdvancedKernelRunArgs compute_g2_adv;

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_output_tiles_per_core;
        const bool in_group_1 = core_group_1.contains(core);
        if (in_group_1) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            TT_FATAL(has_core_group_2, "Core not in specified core ranges");
            num_output_tiles_per_core = num_output_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        AddRuntimeArgsForNode(
            reader_rta,
            core,
            {{"output_tile_start_idx", num_tiles_written}, {"num_output_tiles", num_output_tiles_per_core}});
        reader_adv.runtime_varargs[core] = reader_varargs;

        AddRuntimeArgsForNode(
            writer_rta, core, {{"start_id", num_tiles_written}, {"num_output_tiles", num_output_tiles_per_core}});

        if (in_group_1) {
            AddRuntimeArgsForNode(compute_g1_rta, core, {{"output_tile_start_idx", num_tiles_written}});
            compute_g1_adv.runtime_varargs[core] = compute_varargs;
        } else {
            AddRuntimeArgsForNode(compute_g2_rta, core, {{"output_tile_start_idx", num_tiles_written}});
            compute_g2_adv.runtime_varargs[core] = compute_varargs;
        }

        num_tiles_written += num_output_tiles_per_core;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Assemble spec + run args
    ////////////////////////////////////////////////////////////////////////////
    ProgramSpec spec{
        .name = "moreh_matmul",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{
            .kernel = READER, .runtime_arg_values = std::move(reader_rta), .advanced_options = std::move(reader_adv)},
        KernelRunArgs{.kernel = WRITER, .runtime_arg_values = std::move(writer_rta)},
        KernelRunArgs{
            .kernel = COMPUTE_G1,
            .runtime_arg_values = std::move(compute_g1_rta),
            .advanced_options = std::move(compute_g1_adv)},
    };
    if (has_core_group_2) {
        run_args.kernel_run_args.push_back(KernelRunArgs{
            .kernel = COMPUTE_G2,
            .runtime_arg_values = std::move(compute_g2_rta),
            .advanced_options = std::move(compute_g2_adv)});
    }

    run_args.tensor_args = {
        {INPUT, input_mt},
        {OTHER, other_mt},
        {OUTPUT, output_mt},
    };
    if (bias.has_value()) {
        run_args.tensor_args.emplace(BIAS, bias.value().mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::moreh::moreh_matmul
