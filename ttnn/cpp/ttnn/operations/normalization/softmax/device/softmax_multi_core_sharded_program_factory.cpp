#include "softmax_device_operation.hpp"

#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"

namespace ttnn::operations::normalization {

SoftmaxDeviceOperation::SoftmaxMultiCoreShardedProgramFactory::cached_program_t softmax_multi_core_sharded(
    const Tensor &input_tensor,
    const Tensor &output_tensor,
    const std::optional<const Tensor> mask,
    std::optional<float> scale,
    bool causal_mask,
    bool hw_dims_only_causal_mask,
    CoreCoord grid_size,
    uint32_t subblock_wt,
    uint32_t block_ht,
    uint32_t block_wt,
    const DeviceComputeKernelConfig& compute_kernel_config)
{
    ////////////////////////////////////////////////////////////////////////////
    //                       Device Setup
    ////////////////////////////////////////////////////////////////////////////
    Device *device = input_tensor.device();

    // convert data format
    tt::DataFormat in0_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

    MathFidelity math_fidelity;
    bool math_approx_mode;
    bool fp32_dest_acc_en;
    std::cout << "softmax_multi_core_sharded" << std::endl;

    std::visit([&](auto&& compute_kernel_config) {
        using T = std::decay_t<decltype(compute_kernel_config)>;
        if constexpr (std::is_same_v<T, GrayskullComputeKernelConfig>) {
            TT_ASSERT(device->arch() == tt::ARCH::GRAYSKULL, "kernel config is not for graykull");
            math_fidelity = compute_kernel_config.math_fidelity;
            math_approx_mode = compute_kernel_config.math_approx_mode;
            fp32_dest_acc_en = false;
        } else if constexpr (std::is_same_v<T, WormholeComputeKernelConfig>) {
            TT_ASSERT(ttnn::device::is_wormhole_or_blackhole(device->arch()), "kernel config is not for wormhole_b0 or blackhole");
            math_fidelity = compute_kernel_config.math_fidelity;
            math_approx_mode = compute_kernel_config.math_approx_mode;
            fp32_dest_acc_en = in0_cb_data_format == tt::DataFormat::Float32 ? true : compute_kernel_config.fp32_dest_acc_en;
            if (fp32_dest_acc_en)
                TT_FATAL(subblock_wt <= 4, "in fp32 mode, subblock width must be smaller/equal than 4");
        } else {
            TT_FATAL("arch not supported");
        }

    }, compute_kernel_config);
    std::cout << "softmax_multi_core_sharded done" << std::endl;

    tt::DataFormat out0_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
    tt::DataFormat im_cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    tt::DataFormat mask_cb_data_format = mask.has_value() ? tt::tt_metal::datatype_to_dataformat_converter(mask->get_dtype()) : tt::DataFormat::Float16_b;
    tt::DataFormat scale_cb_data_format = tt::DataFormat::Float16_b;
    tt::DataFormat scalar_cb_data_format = tt::DataFormat::Float16_b;

    tt::log_debug("in0_cb_data_format: {}", in0_cb_data_format);
    tt::log_debug("out0_cb_data_format: {}", out0_cb_data_format);
    tt::log_debug("mask_cb_data_format: {}", mask_cb_data_format);
    tt::log_debug("im_cb_data_format: {}", im_cb_data_format);
    tt::log_debug("scale_cb_data_format: {}", im_cb_data_format);
    tt::log_debug("scalar_cb_data_format: {}", im_cb_data_format);
    tt::log_debug("math_fidelity: {}", math_fidelity);
    tt::log_debug("math_approx_mode: {}", math_approx_mode);
    tt::log_debug("fp32_dest_acc_en: {}", fp32_dest_acc_en);

    // tensor shape
    const auto shard_orient = input_tensor.shard_spec().value().orientation;
    const auto shape = input_tensor.get_legacy_shape();
    uint32_t M = shape[2] * shape[0];
    uint32_t K = shape[3] * shape[1];
    uint32_t Mt = M / TILE_WIDTH;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t num_cores_per_batch = (shape[1] * shape[2] * shape[3]) / (input_tensor.shard_spec().value().shape[0] * input_tensor.shard_spec().value().shape[1]);

    uint32_t mask_H = shape[2];
    if (mask.has_value()) {
        mask_H = mask->get_legacy_shape()[2];
    }
    uint32_t mask_Ht = mask_H/TILE_HEIGHT;
    // block
    uint32_t block_w = block_wt * TILE_WIDTH;
    uint32_t block_h = block_ht * TILE_WIDTH;
    uint32_t num_subblocks_w = block_wt / subblock_wt;

    // single tile sizes
    uint32_t im_tile_size = tt::tt_metal::detail::TileSize(im_cb_data_format);
    uint32_t in0_tile_size = tt::tt_metal::detail::TileSize(in0_cb_data_format);
    uint32_t out0_tile_size = tt::tt_metal::detail::TileSize(out0_cb_data_format);
    uint32_t mask_tile_size = tt::tt_metal::detail::TileSize(mask_cb_data_format);
    uint32_t scale_tile_size = tt::tt_metal::detail::TileSize(scale_cb_data_format);
    uint32_t scalar_tile_size = tt::tt_metal::detail::TileSize(scalar_cb_data_format);
    // in out buffer
    auto src0_buffer = input_tensor.buffer();
    auto out0_buffer = output_tensor.buffer();
    // num tiles
    uint32_t num_tiles = input_tensor.volume()/TILE_HW;


    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // block size for in0 (tensor a)
    uint32_t in0_CB_size = block_wt * block_ht * in0_tile_size;
    // scaler for reduce coming from reader
    uint32_t in1_CB_size = 1 * scalar_tile_size;
    // 1/sqrt() scaler tile cb for fused scale/mask/softmax variant
    uint32_t in2_CB_size = 1 * scale_tile_size;
    // attention mask
    uint32_t in3_CB_size;
    if (causal_mask) {
        if (mask.value().is_sharded()) {
            in3_CB_size = block_wt * block_ht * mask_tile_size;
        } else {
            in3_CB_size = block_wt * mask_tile_size;
            if (!hw_dims_only_causal_mask) {
                // For some reason, if we have hw_dims_causal_mask version, single buffering is up to ~20% faster
                // Then double buffering CB3.
                in3_CB_size *= 2;
            }
        }
    } else {
        in3_CB_size = block_wt * mask_tile_size;
    }
    // cb_exps - keeps exps in tt::CB in L1 to avoid recomputing
    uint32_t im0_CB_size = block_wt * im_tile_size;
    // 1/sum(exp(x))
    uint32_t im1_CB_size = 1 * im_tile_size;
    // attn mask im
    uint32_t im2_CB_size = block_wt * im_tile_size;
    // output buffer size
    uint32_t out_CB_size = block_wt * block_ht * out0_tile_size;

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program = CreateProgram();
    // define core ranges
    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    uint32_t num_cores_c = grid_size.x;
    uint32_t num_cores_r = grid_size.y;
    uint32_t num_cores = num_cores_c * num_cores_r;
    CoreRange all_device_cores(
        {(std::size_t) start_core_x, (std::size_t) start_core_y},
        {(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1});
    // reader compile arg
    bool is_dram_mask = 0;
    if (mask.has_value()) {
        is_dram_mask = mask->buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    }
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) block_wt,
        (std::uint32_t) is_dram_mask
    };
    std::map<string, string> softmax_defines;
    // hw_dims_only_causal_mask does not support RM Layout atm
    bool use_row_major_kernel = (mask.has_value() and mask->get_layout() == Layout::ROW_MAJOR);
    if (use_row_major_kernel) {
        auto mask_stick_size = mask->get_legacy_shape()[3] * mask->element_size();
        bool mask_stick_size_is_power_of_two = is_power_of_two_at_least_32(mask_stick_size);
        reader_compile_time_args.push_back((std::uint32_t) mask_stick_size_is_power_of_two);
        if (mask_stick_size_is_power_of_two) {
            uint32_t mask_log2_stick_size = (std::uint32_t)log2(mask_stick_size);
            reader_compile_time_args.push_back((std::uint32_t) mask_log2_stick_size);
        } else {
            reader_compile_time_args.push_back(mask_stick_size);
        }
    } else {
        reader_compile_time_args.push_back(0);
        reader_compile_time_args.push_back(0);
    }
    if (causal_mask) {
        if (!hw_dims_only_causal_mask) {
            reader_compile_time_args.push_back((std::uint32_t) block_ht / mask_Ht); // fused head
        } else {
            reader_compile_time_args.push_back((std::uint32_t) block_ht);
        }
    }
    reader_compile_time_args.push_back((std::uint32_t) (mask_cb_data_format == tt::DataFormat::Float32)); // mask float32
    reader_compile_time_args.push_back((std::uint32_t) mask_Ht);

    if (mask.has_value()) {
        softmax_defines["FUSED_SCALE_MASK"] = "1";
    }
    if (causal_mask) {
        softmax_defines["CAUSAL_MASK"] = "1";
        if (mask.value().is_sharded())
            softmax_defines["SHARDED_CAUSAL_MASK"] =  "1";
    }
    std::string reader_kernel_path;
    if (use_row_major_kernel) {
        reader_kernel_path = "ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/dataflow/reader_unary_sharded_sm_rm_mask.cpp";
    } else if (!hw_dims_only_causal_mask) {
        reader_kernel_path = "ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/dataflow/reader_unary_sharded_sm.cpp";
    } else {
        reader_kernel_path = "ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/dataflow/readed_unary_sharded_sm_causal_mask_hw_dims.cpp";
    }
    auto reader_kernels_id = CreateKernel(
        program,
        reader_kernel_path,
        all_device_cores,
        tt::tt_metal::ReaderDataMovementConfig(
            reader_compile_time_args,
            softmax_defines
    ));
    // compute kernel compile time args
    std::vector<uint32_t> compute_compile_time_args = {
        block_ht,
        block_wt,
        subblock_wt,
        num_subblocks_w,
    };
    softmax_defines["EXP_APPROX"] = math_approx_mode ? "1" : "0";
    auto softmax_kernels_id = CreateKernel(
        program, "ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/compute/softmax_sharded.cpp", all_device_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_time_args,
            .defines = softmax_defines
    });

    // Create circular buffers
    // in0 sharded
    auto c_in0_config = CircularBufferConfig(in0_CB_size, {{tt::CB::c_in0, in0_cb_data_format}})
        .set_page_size(tt::CB::c_in0, in0_tile_size).set_globally_allocated_address(*src0_buffer);
    auto cb_in0_id = CreateCircularBuffer(program, all_device_cores, c_in0_config);
    // in1 scalar
    auto c_in1_config = CircularBufferConfig(in1_CB_size, {{tt::CB::c_in1, scalar_cb_data_format}})
        .set_page_size(tt::CB::c_in1, scalar_tile_size);
    auto cb_in1_id = CreateCircularBuffer(program, all_device_cores, c_in1_config);
    // in2 in3 attn scale mask
    std::optional<CBHandle> cb_intermed2_id;
    std::optional<CBHandle> cb_in2_id;
    std::optional<CBHandle> cb_in3_id;
    if (mask.has_value()) {
        // im2
        auto c_intermed2_config = CircularBufferConfig(im2_CB_size, {{tt::CB::c_intermed2, im_cb_data_format}})
            .set_page_size(tt::CB::c_intermed2, im_tile_size);
        cb_intermed2_id = CreateCircularBuffer( program, all_device_cores, c_intermed2_config );
        // in2 scale
        auto c_in2_config = CircularBufferConfig(in2_CB_size, {{tt::CB::c_in2, scale_cb_data_format}})
            .set_page_size(tt::CB::c_in2, scale_tile_size);
        cb_in2_id = CreateCircularBuffer(program, all_device_cores, c_in2_config);
        // in3 attn mask
        if (mask->is_sharded()) {
            auto mask_buffer = mask->buffer();
            auto c_in3_config = CircularBufferConfig(in3_CB_size, {{tt::CB::c_in3, mask_cb_data_format}})
                .set_page_size(tt::CB::c_in3, mask_tile_size).set_globally_allocated_address(*mask_buffer);
            cb_in3_id = CreateCircularBuffer( program, all_device_cores, c_in3_config);
        } else {
            auto c_in3_config = CircularBufferConfig(in3_CB_size, {{tt::CB::c_in3, mask_cb_data_format}})
                .set_page_size(tt::CB::c_in3, mask_tile_size);
            cb_in3_id = CreateCircularBuffer( program, all_device_cores, c_in3_config);
        }
    }
    // out
    auto c_out0_config = CircularBufferConfig(out_CB_size, {{tt::CB::c_out0, out0_cb_data_format}})
        .set_page_size(tt::CB::c_out0, out0_tile_size).set_globally_allocated_address(*out0_buffer);;
    auto cb_out0_id = CreateCircularBuffer( program, all_device_cores, c_out0_config );
    // im0 for exp(x)
    auto c_intermed0_config = CircularBufferConfig(im0_CB_size, {{tt::CB::c_intermed0, im_cb_data_format}})
        .set_page_size(tt::CB::c_intermed0, im_tile_size);
    auto cb_intermed0_id = CreateCircularBuffer( program, all_device_cores, c_intermed0_config );
    // im1 for 1/sum(exp(x))
    auto c_intermed1_config = CircularBufferConfig(im1_CB_size, {{tt::CB::c_intermed1, im_cb_data_format}})
        .set_page_size(tt::CB::c_intermed1, im_tile_size);
    auto cb_intermed1_id = CreateCircularBuffer( program, all_device_cores, c_intermed1_config );

    // Runtime Args
    uint32_t mask_addr = mask.has_value() ? mask->buffer()->address() : 0;
    union { float f; uint32_t u; } s; s.f = scale.value_or(1.0f); // scale for fused scale-mask-softmax
    uint32_t mask_start_tile_id = 0;

    uint32_t num_tiles_in_attn_mask = 0;
    uint32_t num_tiles_of_attn_mask_needed_per_core = 0;
    if (hw_dims_only_causal_mask) {
        num_tiles_in_attn_mask = mask.value().get_legacy_shape()[-1] * mask.value().get_legacy_shape()[-2] / TILE_HW;
        num_tiles_of_attn_mask_needed_per_core = block_ht * block_wt;
    }
    uint32_t num_cores_per_batch_index = 0;

    if (shard_orient == ShardOrientation::COL_MAJOR) {
        for(int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            for(int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
                CoreCoord core = {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y};

                // reader args
                std::vector<uint32_t> reader_args;
                reader_args.push_back(0x3f803f80);
                reader_args.push_back(s.u);
                reader_args.push_back(mask_addr);
                reader_args.push_back(mask_start_tile_id);
                if (hw_dims_only_causal_mask) {
                    reader_args.push_back(num_tiles_in_attn_mask);
                }

                tt::tt_metal::SetRuntimeArgs(program, reader_kernels_id, core, reader_args);

                num_cores_per_batch_index ++;

                if (hw_dims_only_causal_mask) {
                    uint32_t mask_tile_id_end = (mask_start_tile_id + num_tiles_of_attn_mask_needed_per_core) % num_tiles_in_attn_mask;
                    mask_start_tile_id = mask_tile_id_end;
                } else {
                    if (num_cores_per_batch_index == num_cores_per_batch) {
                        num_cores_per_batch_index = 0;
                        if (mask.has_value()) {
                            if (causal_mask) {
                                mask_start_tile_id += mask->get_legacy_shape()[-1] * mask->get_legacy_shape()[-2] / TILE_WIDTH / TILE_HEIGHT;
                            } else {
                                mask_start_tile_id += use_row_major_kernel ? mask->get_legacy_shape()[-2] : mask->get_legacy_shape()[-1] / TILE_WIDTH;
                            }
                        }
                    }
                }
            }
        }
    } else {
        for(int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
            for(int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
                CoreCoord core = {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y};

                // reader args
                std::vector<uint32_t> reader_args;
                reader_args.push_back(0x3f803f80);
                reader_args.push_back(s.u);
                reader_args.push_back(mask_addr);
                reader_args.push_back(mask_start_tile_id);
                if (hw_dims_only_causal_mask) {
                    reader_args.push_back(num_tiles_in_attn_mask);
                }

                tt::tt_metal::SetRuntimeArgs(program, reader_kernels_id, core, reader_args);

                num_cores_per_batch_index ++;

                if (hw_dims_only_causal_mask) {
                    uint32_t mask_tile_id_end = (mask_start_tile_id + num_tiles_of_attn_mask_needed_per_core) % num_tiles_in_attn_mask;
                    mask_start_tile_id = mask_tile_id_end;
                } else {
                    if (num_cores_per_batch_index == num_cores_per_batch) {
                        num_cores_per_batch_index = 0;
                        if (mask.has_value()) {
                            if (causal_mask) {
                                mask_start_tile_id += mask->get_legacy_shape()[-1] * mask->get_legacy_shape()[-2] / TILE_WIDTH / TILE_HEIGHT;
                            } else {
                                mask_start_tile_id += use_row_major_kernel ? mask->get_legacy_shape()[-2] : mask->get_legacy_shape()[-1] / TILE_WIDTH;
                            }
                        }
                    }
                }
            }
        }
    }

    return {
        std::move(program),
        SoftmaxDeviceOperation::SoftmaxMultiCoreShardedProgramFactory::shared_variables_t{
            .reader_kernels_id = reader_kernels_id,
            .cb_in0_id = cb_in0_id,
            .cb_out0_id = cb_out0_id,
            .cb_in3_id = cb_in3_id,
            .num_cores = num_cores,
            .grid_size = grid_size
        }
    };
}


SoftmaxDeviceOperation::SoftmaxMultiCoreShardedProgramFactory::cached_program_t SoftmaxDeviceOperation::SoftmaxMultiCoreShardedProgramFactory::create(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    auto program_config = std::get<SoftmaxShardedMultiCoreProgramConfig>(attributes.program_config);
    return softmax_multi_core_sharded(
        tensor_args.input_tensor,
        output_tensor,
        tensor_args.mask,
        attributes.scale,
        attributes.is_causal_mask,
        attributes.is_scale_causal_mask_hw_dims_softmax,
        program_config.compute_with_storage_grid_size,
        program_config.subblock_w,
        program_config.block_h,
        program_config.block_w,
        attributes.compute_kernel_config.value());
}

void SoftmaxDeviceOperation::SoftmaxMultiCoreShardedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {

    auto& reader_kernels_id = cached_program.shared_variables.reader_kernels_id;
    auto& cb_in0_id = cached_program.shared_variables.cb_in0_id;
    auto& cb_out0_id = cached_program.shared_variables.cb_out0_id;
    auto& cb_in3_id = cached_program.shared_variables.cb_in3_id;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& grid_size = cached_program.shared_variables.grid_size;

    auto& program = cached_program.program;
    auto& mask_tensor = tensor_args.mask;
    auto in0_buffer = tensor_args.input_tensor.buffer();
    auto out_buffer = attributes.inplace ? in0_buffer : tensor_return_value.buffer();

    UpdateDynamicCircularBufferAddress(program, cb_in0_id, *in0_buffer);
    UpdateDynamicCircularBufferAddress(program, cb_out0_id, *out_buffer);
    if (mask_tensor.has_value() && mask_tensor->is_sharded()) {
        UpdateDynamicCircularBufferAddress(program, cb_in3_id.value(), *mask_tensor->buffer());
    }

    if (mask_tensor.has_value()) {
        for (uint32_t i = 0; i < num_cores; ++i) {
            CoreCoord core = {i % grid_size.x, i / grid_size.x};
            auto &runtime_args = GetRuntimeArgs(program, reader_kernels_id, core);
                runtime_args[2] = mask_tensor->buffer()->address();
        }
    }
}

}  // namespace ttnn::operations::normalization
