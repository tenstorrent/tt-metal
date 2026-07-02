// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "moreh_matmul_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_matmul {

void get_tensor_dim(ttnn::SmallVector<uint32_t>& dim, const ttnn::Shape& shape) {
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
    for (auto i = 0; i < tt::tt_metal::MAX_NUM_DIMENSIONS; ++i) {
        log_debug(tt::LogOp, "dim[{}] = {}", i, dim[i]);
    }
}

ttnn::SmallVector<int64_t> find_reduce_dim(const ttnn::Shape& a_shape, const ttnn::Shape& b_shape) {
    ttnn::SmallVector<uint32_t> a_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    ttnn::SmallVector<uint32_t> b_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(a_dim, a_shape);
    get_tensor_dim(b_dim, b_shape);
    int32_t rank = std::max(a_shape.rank(), b_shape.rank());
    log_debug(tt::LogOp, "find_reduce_dim :{} rank {} a {} b {}", __LINE__, rank, a_shape.rank(), b_shape.rank());
    ttnn::SmallVector<int64_t> dims;
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
    ttnn::SmallVector<uint32_t> a_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    ttnn::SmallVector<uint32_t> b_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(a_dim, a_shape);
    get_tensor_dim(b_dim, b_shape);
    for (auto i = 2; i < tt::tt_metal::MAX_NUM_DIMENSIONS; ++i) {
        if (a_dim[i] != b_dim[i]) {
            log_debug(tt::LogOp, "{}:{} {} a_dim {} - b_dim {}", __func__, __LINE__, i, a_dim[i], b_dim[i]);
            return false;
        }
    }
    log_debug(tt::LogOp, "{}:{} batch dims are the same.", __func__, __LINE__);
    return true;
}

void get_tensor_stride(ttnn::SmallVector<uint32_t>& stride, ttnn::SmallVector<uint32_t>& dim) {
    stride[0] = 1;
    for (auto i = 1; i < tt::tt_metal::MAX_NUM_DIMENSIONS; ++i) {
        stride[i] = stride[i - 1] * dim[i - 1];
    }

    for (auto i = 0; i < tt::tt_metal::MAX_NUM_DIMENSIONS; ++i) {
        log_debug(tt::LogOp, "stride[{}] = {}", i, stride[i]);
    }
}

void get_not_bcast(
    ttnn::SmallVector<uint32_t>& input_not_bcast,
    ttnn::SmallVector<uint32_t>& input_dim,
    ttnn::SmallVector<uint32_t>& other_not_bcast,
    ttnn::SmallVector<uint32_t>& other_dim) {
    // first 2-dims are M,K and K,N
    // TODO: refaactoring
    for (auto i = 2; i < tt::tt_metal::MAX_NUM_DIMENSIONS; ++i) {
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

    for (auto i = 0; i < tt::tt_metal::MAX_NUM_DIMENSIONS; ++i) {
        log_debug(tt::LogOp, "not bcast [{}] input {} other {}", i, input_not_bcast[i], other_not_bcast[i]);
    }
}

tt::tt_metal::ProgramDescriptor MorehMatmulOperation::MultiCoreProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const Tensor& input = tensor_args.input;
    const Tensor& other = tensor_args.other;
    const Tensor& output = tensor_return_value;

    const std::optional<const Tensor>& bias = tensor_args.bias;

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
    ttnn::SmallVector<uint32_t> input_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(input_dim, input_shape);

    log_debug(tt::LogOp, "input stride");
    ttnn::SmallVector<uint32_t> input_stride(tt::tt_metal::MAX_NUM_DIMENSIONS);
    get_tensor_stride(input_stride, input_dim);

    // other tensor
    const auto& other_shape = other.padded_shape();
    const auto& other_shape_wo_padding = other.logical_shape();
    log_debug(tt::LogOp, "other dim");
    ttnn::SmallVector<uint32_t> other_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(other_dim, other_shape);

    log_debug(tt::LogOp, "other stride");
    ttnn::SmallVector<uint32_t> other_stride(tt::tt_metal::MAX_NUM_DIMENSIONS);
    get_tensor_stride(other_stride, other_dim);

    log_debug(tt::LogOp, "not bcast");
    ttnn::SmallVector<uint32_t> input_not_bcast(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    ttnn::SmallVector<uint32_t> other_not_bcast(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    get_not_bcast(input_not_bcast, input_dim, other_not_bcast, other_dim);

    // output tensor
    const auto& output_shape = output.padded_shape();
    log_debug(tt::LogOp, "output dim");
    ttnn::SmallVector<uint32_t> output_dim(tt::tt_metal::MAX_NUM_DIMENSIONS, 1);
    get_tensor_dim(output_dim, output_shape);

    log_debug(tt::LogOp, "output stride");
    ttnn::SmallVector<uint32_t> output_stride(tt::tt_metal::MAX_NUM_DIMENSIONS);
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
    //                         CircularBuffer Setup
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

    ProgramDescriptor desc;

    auto add_cb = [&](CBIndex idx, uint32_t num_tiles, tt::DataFormat fmt) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_tiles * tile_size(fmt),
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(idx),
                .data_format = fmt,
                .page_size = tile_size(fmt),
            }}},
        });
    };

    add_cb(tt::CBIndex::c_0, in0_t, cb_data_format);
    add_cb(tt::CBIndex::c_1, in1_t, cb_data_format);
    add_cb(tt::CBIndex::c_2, in2_t, cb_data_format);
    add_cb(tt::CBIndex::c_3, in3_t, cb_data_format);
    add_cb(tt::CBIndex::c_4, in4_t, cb_data_format);
    add_cb(tt::CBIndex::c_24, im0_t, im0_data_format);
    add_cb(tt::CBIndex::c_25, im1_t, cb_data_format);
    add_cb(tt::CBIndex::c_26, im2_t, cb_data_format);
    add_cb(tt::CBIndex::c_27, im3_t, im3_data_format);
    add_cb(tt::CBIndex::c_16, out0_t, cb_data_format);

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> reader_defines_map;
    KernelDescriptor::CompileTimeArgs reader_compile_time_args = {
        Kt,
        static_cast<uint32_t>(transpose_input),
        static_cast<uint32_t>(transpose_other),
        input_mask_h,
        input_mask_w,
        other_mask_h,
        other_mask_w,
    };
    TensorAccessorArgs(input.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(other.buffer()).append_to(reader_compile_time_args);

    if (bias.has_value()) {
        reader_defines_map["FUSE_BIAS"] = "1";
        reader_compile_time_args.push_back(static_cast<uint32_t>(is_scalar_bias));
        TensorAccessorArgs(bias->buffer()).append_to(reader_compile_time_args);
        log_debug(tt::LogOp, "{}:{} bias tensor. is bias dram {}", __func__, __LINE__, is_dram(bias));
    }

    KernelDescriptor::CompileTimeArgs writer_compile_time_args;
    TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor::Defines reader_defines(reader_defines_map.begin(), reader_defines_map.end());

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = "ttnn/cpp/ttnn/operations/moreh/moreh_matmul/device/kernels/reader_moreh_matmul.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = std::move(reader_defines);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = "ttnn/cpp/ttnn/operations/moreh/moreh_matmul/device/kernels/writer_moreh_matmul.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

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
    std::map<std::string, std::string> compute_defines_map;

    const auto* const compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_matmul/device/kernels/moreh_matmul.cpp";
    KernelDescriptor::CompileTimeArgs compute_args_group_1 = {
        num_output_tiles_per_core_group_1,  // num_output_tiles
        Mt,
        Nt,
        Kt,
        static_cast<uint32_t>(transpose_input),
        static_cast<uint32_t>(transpose_other),
        input_mask_h,
        input_mask_w,
        other_mask_h,
        other_mask_w};

    if (bias.has_value()) {
        compute_defines_map["FUSE_BIAS"] = "1";
        compute_args_group_1.push_back(static_cast<uint32_t>(is_scalar_bias));
    }

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        compute_defines_map["FP32_DEST_ACC_EN"] = "1";
        unpack_to_dest_mode[tt::CBIndex::c_24] = UnpackToDestMode::UnpackToDestFp32;
    }
    KernelDescriptor::Defines compute_defines(compute_defines_map.begin(), compute_defines_map.end());

    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = compute_kernel_file;
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = std::move(compute_args_group_1);
    compute_desc_1.defines = compute_defines;
    compute_desc_1.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .unpack_to_dest_mode = unpack_to_dest_mode,
        .math_approx_mode = math_approx_mode,
    };

    KernelDescriptor compute_desc_2;
    bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        KernelDescriptor::CompileTimeArgs compute_args_group_2 = {
            num_output_tiles_per_core_group_2,  // num_output_tiles
            Mt,
            Nt,
            Kt,
            static_cast<uint32_t>(transpose_input),
            static_cast<uint32_t>(transpose_other),
            input_mask_h,
            input_mask_w,
            other_mask_h,
            other_mask_w};

        if (bias.has_value()) {
            compute_args_group_2.push_back(static_cast<uint32_t>(is_scalar_bias));
        }

        compute_desc_2.kernel_source = compute_kernel_file;
        compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_2.core_ranges = core_group_2;
        compute_desc_2.compile_time_args = std::move(compute_args_group_2);
        compute_desc_2.defines = compute_defines;
        compute_desc_2.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = math_approx_mode,
        };
    }
    log_debug(
        tt::LogOp,
        "{}:{} Compute ",
        __func__,
        __LINE__,
        static_cast<uint32_t>(transpose_input),
        static_cast<uint32_t>(transpose_other));

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto* const input_buf = input.buffer();
    auto* const other_buf = other.buffer();
    auto* const output_buf = output.buffer();
    auto* const bias_buf = bias.has_value() ? bias.value().buffer() : nullptr;

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_output_tiles_per_core;
        std::vector<uint32_t> compute_rt_args;
        compute_rt_args.push_back(num_tiles_written);
        compute_rt_args.insert(compute_rt_args.end(), output_stride.begin(), output_stride.end());

        if (core_group_1.contains(core)) {
            num_output_tiles_per_core = num_output_tiles_per_core_group_1;
            compute_desc_1.runtime_args.emplace_back(
                core, KernelDescriptor::CoreRuntimeArgs(compute_rt_args.begin(), compute_rt_args.end()));
        } else if (core_group_2.contains(core)) {
            TT_FATAL(has_core_group_2, "Core not in specified core ranges");
            num_output_tiles_per_core = num_output_tiles_per_core_group_2;
            compute_desc_2.runtime_args.emplace_back(
                core, KernelDescriptor::CoreRuntimeArgs(compute_rt_args.begin(), compute_rt_args.end()));
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        // Pass buffers as Buffer* so the ProgramDescriptor framework registers them as
        // BufferBindings and patches their addresses on program-cache hits. Raw uint32_t
        // addresses would be baked in once and go stale when inputs are reallocated.
        KernelDescriptor::RTArgList reader_rt_args;
        reader_rt_args.push_back(input_buf);
        reader_rt_args.push_back(other_buf);
        reader_rt_args.push_back(num_tiles_written);
        reader_rt_args.push_back(num_output_tiles_per_core);

        // TODO: move some to compile args
        reader_rt_args.append(std::vector<uint32_t>(input_stride.begin(), input_stride.end()));
        reader_rt_args.append(std::vector<uint32_t>(other_stride.begin(), other_stride.end()));
        reader_rt_args.append(std::vector<uint32_t>(output_stride.begin(), output_stride.end()));
        reader_rt_args.append(std::vector<uint32_t>(input_not_bcast.begin(), input_not_bcast.end()));
        reader_rt_args.append(std::vector<uint32_t>(other_not_bcast.begin(), other_not_bcast.end()));

        if (bias.has_value()) {
            reader_rt_args.push_back(bias_buf);
        }

        reader_desc.emplace_runtime_args(core, reader_rt_args);

        writer_desc.emplace_runtime_args(core, {output_buf, num_tiles_written, num_output_tiles_per_core});
        num_tiles_written += num_output_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_1));
    if (has_core_group_2) {
        desc.kernels.push_back(std::move(compute_desc_2));
    }

    return desc;
}

}  // namespace ttnn::operations::moreh::moreh_matmul
