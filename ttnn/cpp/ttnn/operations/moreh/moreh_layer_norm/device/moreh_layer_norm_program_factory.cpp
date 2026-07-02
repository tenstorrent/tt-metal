// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <string>
#include <vector>

#include "moreh_layer_norm_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm {

inline uint32_t find_divisor_with_max_block_size(uint32_t val, uint32_t max_block_size) {
    uint32_t divisor{1};
    for (uint32_t current_divisor = max_block_size; current_divisor >= 1; current_divisor--) {
        if (val % current_divisor == 0) {
            divisor = current_divisor;
            break;
        }
    }
    return divisor;
}

tt::tt_metal::ProgramDescriptor MorehLayerNormOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;

    const auto& mean_inp = tensor_args.mean;
    const auto& rstd_inp = tensor_args.rstd;

    const std::optional<const Tensor>& output = output_tensor.at(0);

    std::optional<Tensor> mean = std::nullopt;
    if (mean_inp.has_value()) {
        mean = output_tensor.at(1);
    }
    const std::optional<const Tensor> mean_as_tensor = mean ? std::optional<const Tensor>(mean.value()) : std::nullopt;

    std::optional<Tensor> rstd = std::nullopt;
    if (rstd_inp.has_value()) {
        rstd = output_tensor.at(2);
    }
    const std::optional<const Tensor> rstd_as_tensor = rstd ? std::optional<const Tensor>(rstd.value()) : std::nullopt;

    auto normalized_dims = operation_attributes.normalized_dims;
    auto eps = operation_attributes.eps;

    auto compute_kernel_config =
        init_device_compute_kernel_config(input.device()->arch(), operation_attributes.compute_kernel_config);

    using namespace tt::constants;
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    IDevice* device = input.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_shape = input.padded_shape();
    const auto input_shape_without_padding = input.logical_shape();
    const auto input_rank = input_shape.rank();

    const bool is_lastdim_layer_norm = normalized_dims == 1;
    const bool is_groupnorm = false;

    auto num_inner = compute_inner(input_shape, normalized_dims);
    auto num_outer = compute_outer(input_shape, normalized_dims);

    const auto gamma_has_value = gamma.has_value();
    const auto beta_has_value = beta.has_value();
    const auto mean_has_value = mean.has_value();
    const auto rstd_has_value = rstd.has_value();

    const auto origin_H = input_shape_without_padding[-2];
    const auto origin_W = input_shape_without_padding[-1];

    uint32_t mean_rstd_height = 0;
    uint32_t mean_rstd_width = 0;

    if (mean_has_value || rstd_has_value) {
        const auto mean_rstd_shape_without_padding = mean_has_value ? mean->logical_shape() : rstd->logical_shape();
        mean_rstd_height = mean_rstd_shape_without_padding[-2];
        mean_rstd_width = mean_rstd_shape_without_padding[-1];
    }

    const bool do_mask_h = (origin_H % TILE_HEIGHT) != 0 && !is_lastdim_layer_norm;
    const auto mask_h = do_mask_h ? origin_H % TILE_HEIGHT : TILE_HEIGHT;

    const bool do_mask_w = (origin_W % TILE_WIDTH) != 0;
    const auto mask_w = do_mask_w ? origin_W % TILE_WIDTH : TILE_WIDTH;

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    // core_group_2 works more.
    // If number of working cores is 108 and num_outer is 110,
    // core_group_2[(x=0, y=0), (x=0, y=1)] works for 2 rows. Others work for 1 row.
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(grid, num_outer);

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    // This could be inefficient.
    // If Wt is 65, the block_size will be 5. Then, the number of iteration is 13.
    // It can be 8 * 8 + 1, so the number of iterations is 9. It's more efficient.
    uint32_t MAX_BLOCK_SIZE = 4;
    if (fp32_dest_acc_en) {
        MAX_BLOCK_SIZE = 2;
    }
    const uint32_t block_size = find_divisor_with_max_block_size(num_inner, MAX_BLOCK_SIZE);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t in0_t = num_inner;                                   // input
    const uint32_t in1_t = 1;                                     // scaler
    const uint32_t in2_t = 1;                                     // epsilon
    const uint32_t in3_t = gamma_has_value ? 2 * block_size : 0;  // gamma
    const uint32_t in4_t = beta_has_value ? 2 * block_size : 0;   // beta
    const uint32_t in5_t = do_mask_h ? 1 : 0;                     // mask_h
    const uint32_t in6_t = do_mask_w ? 1 : 0;                     // mask_w

    const uint32_t out0_t = 2 * block_size;          // output
    const uint32_t out1_t = mean_has_value ? 1 : 0;  // mean
    const uint32_t out2_t = rstd_has_value ? 1 : 0;  // rstd

    const uint32_t im0_t = 1;                                                         // E[x]
    uint32_t im1_t = num_inner;                                                       // x - E[x]
    uint32_t im2_t = 1;                                                               // (x - E[x])^2
    const uint32_t im3_t = 1;                                                         // Sum[(x - E[x])^2]
    const uint32_t im4_t = 1;                                                         // E[(x - E[x])^2] = Var[x]
    const uint32_t im5_t = 1;                                                         // 1.0/(sqrt(Var[x] + eps))
    const uint32_t im6_t = (gamma_has_value || beta_has_value) ? 2 * block_size : 0;  // x * gamm + beta
    const uint32_t im7_t = 2;                                                         // Sum[x]

    const auto cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const auto single_tile_size = tt::tile_size(cb_data_format);
    auto intermed_cb_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : cb_data_format;
    const auto intermed_single_tile_size = tt::tile_size(intermed_cb_format);

    const uint32_t cb_usage =
        ((in0_t + in1_t + in2_t + in3_t + in4_t + in5_t + in6_t + out0_t + out1_t + out2_t) * single_tile_size) +
        ((im0_t + im1_t + im2_t + im3_t + im4_t + im5_t + im6_t + im7_t) * intermed_single_tile_size);
    const uint32_t available_L1 =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const bool use_large_algorithm = cb_usage >= available_L1;

    if (use_large_algorithm) {
        log_info(tt::LogTest, "Large moreh_layer_norm algorithm is selected.");
        in0_t = 2 * block_size;
        im1_t = 2 * block_size;
        im2_t = 2 * block_size;
    } else {
        log_info(tt::LogTest, "Small moreh_layer_norm algorithm is selected.");
    }

    ProgramDescriptor desc;

    // ---- Circular buffers ----
    // Helper to push a CB with a given index, num_tiles, and data format.
    auto push_cb = [&](uint8_t cb_index, uint32_t num_tiles, tt::DataFormat fmt) {
        if (num_tiles == 0) {
            // Preserve original behavior: skip zero-size CBs.
            return;
        }
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_tiles * tile_size(fmt),
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = cb_index,
                .data_format = fmt,
                .page_size = tile_size(fmt),
            }}},
        });
    };

    push_cb(static_cast<uint8_t>(CBIndex::c_0), in0_t, cb_data_format);       // input
    push_cb(static_cast<uint8_t>(CBIndex::c_1), in1_t, cb_data_format);       // scaler
    push_cb(static_cast<uint8_t>(CBIndex::c_2), in2_t, cb_data_format);       // epsilon
    push_cb(static_cast<uint8_t>(CBIndex::c_3), in3_t, cb_data_format);       // gamma
    push_cb(static_cast<uint8_t>(CBIndex::c_4), in4_t, cb_data_format);       // beta
    push_cb(static_cast<uint8_t>(CBIndex::c_5), in5_t, cb_data_format);       // mask_h
    push_cb(static_cast<uint8_t>(CBIndex::c_6), in6_t, cb_data_format);       // mask_w
    push_cb(static_cast<uint8_t>(CBIndex::c_16), out0_t, cb_data_format);     // output
    push_cb(static_cast<uint8_t>(CBIndex::c_17), out1_t, cb_data_format);     // mean
    push_cb(static_cast<uint8_t>(CBIndex::c_18), out2_t, cb_data_format);     // rstd
    push_cb(static_cast<uint8_t>(CBIndex::c_24), im0_t, intermed_cb_format);  // E[x]
    push_cb(static_cast<uint8_t>(CBIndex::c_25), im1_t, intermed_cb_format);  // x - E[x]
    push_cb(static_cast<uint8_t>(CBIndex::c_26), im2_t, intermed_cb_format);  // (x - E[x])^2
    push_cb(static_cast<uint8_t>(CBIndex::c_27), im3_t, intermed_cb_format);  // Sum[(x - E[x])^2]
    push_cb(static_cast<uint8_t>(CBIndex::c_28), im4_t, intermed_cb_format);  // E[(x - E[x])^2] = Var[x]
    push_cb(static_cast<uint8_t>(CBIndex::c_29), im5_t, intermed_cb_format);  // 1.0/(sqrt(Var[x] + eps))
    push_cb(static_cast<uint8_t>(CBIndex::c_30), im6_t, intermed_cb_format);  // y * gamm + beta
    push_cb(static_cast<uint8_t>(CBIndex::c_31), im7_t, intermed_cb_format);  // Sum[x]

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelDescriptor::CompileTimeArgs reader_compile_time_args{block_size};
    tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(gamma ? gamma->buffer() : nullptr).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(beta ? beta->buffer() : nullptr).append_to(reader_compile_time_args);

    KernelDescriptor::CompileTimeArgs writer_compile_time_args{
        static_cast<uint32_t>(mean_has_value), static_cast<uint32_t>(rstd_has_value), block_size};
    tt::tt_metal::TensorAccessorArgs(output->buffer()).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(mean_as_tensor ? mean_as_tensor->buffer() : nullptr)
        .append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(rstd_as_tensor ? rstd_as_tensor->buffer() : nullptr)
        .append_to(writer_compile_time_args);

    std::map<std::string, std::string> reader_defines_map{};
    std::map<std::string, std::string> compute_defines_map{};
    if (gamma_has_value) {
        reader_defines_map["GAMMA_HAS_VALUE"] = "1";
    }
    if (beta_has_value) {
        reader_defines_map["BETA_HAS_VALUE"] = "1";
    }
    if (do_mask_h) {
        reader_defines_map["DO_MASK_H"] = "1";
    }
    if (do_mask_w) {
        reader_defines_map["DO_MASK_W"] = "1";
    }
    compute_defines_map["REDUCE_OP"] = "PoolType::AVG";
    if (is_lastdim_layer_norm) {
        compute_defines_map["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";
    } else {
        compute_defines_map["REDUCE_DIM"] = "ReduceDim::REDUCE_SCALAR";
    }
    if (fp32_dest_acc_en) {
        reader_defines_map["FP32_DEST_ACC_EN"] = "1";
        compute_defines_map["FP32_DEST_ACC_EN"] = "1";
    }

    KernelDescriptor::Defines reader_defines(reader_defines_map.begin(), reader_defines_map.end());
    KernelDescriptor::Defines compute_defines(compute_defines_map.begin(), compute_defines_map.end());

    const auto* const reader_kernel_file =
        use_large_algorithm
            ? "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/reader_moreh_layer_norm_large.cpp"
            : "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/reader_moreh_layer_norm_small.cpp";
    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/writer_moreh_layer_norm.cpp";

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel_file;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = reader_defines;
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = writer_kernel_file;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    const auto* const compute_kernel_file =
        use_large_algorithm
            ? "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_large_kernel.cpp"
            : "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_small_kernel.cpp";

    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = compute_kernel_file;
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = {
        num_rows_per_core_group_1,
        origin_H,
        origin_W,
        num_inner,
        block_size,
        static_cast<uint32_t>(gamma_has_value),
        static_cast<uint32_t>(beta_has_value),
        static_cast<uint32_t>(mean_has_value),
        static_cast<uint32_t>(rstd_has_value),
        static_cast<uint32_t>(is_lastdim_layer_norm),
        static_cast<uint32_t>(is_groupnorm)};
    compute_desc_1.defines = compute_defines;
    compute_desc_1.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    KernelDescriptor compute_desc_2;
    const bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        compute_desc_2.kernel_source = compute_kernel_file;
        compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_2.core_ranges = core_group_2;
        compute_desc_2.compile_time_args = {
            num_rows_per_core_group_2,
            origin_H,
            origin_W,
            num_inner,
            block_size,
            static_cast<uint32_t>(gamma_has_value),
            static_cast<uint32_t>(beta_has_value),
            static_cast<uint32_t>(mean_has_value),
            static_cast<uint32_t>(rstd_has_value),
            static_cast<uint32_t>(is_lastdim_layer_norm),
            static_cast<uint32_t>(is_groupnorm)};
        compute_desc_2.defines = compute_defines;
        compute_desc_2.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
        };
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    float scaler_f = 0.0f;
    if (normalized_dims == 1) {
        scaler_f = 1.0f / static_cast<float>(origin_W);
    } else {
        uint32_t reduce_size = 1;
        for (uint32_t i = input_rank - normalized_dims; i < input_rank; i++) {
            auto size = input_shape_without_padding[i];
            reduce_size *= size;
        }

        scaler_f = 1.0f / std::sqrt(static_cast<float>(reduce_size));
    }
    const uint32_t scaler_u = std::bit_cast<uint32_t>(scaler_f);

    const uint32_t e_u = std::bit_cast<uint32_t>(eps);  // epsilon

    auto* const input_buf = input.buffer();
    auto* const output_buf = output->buffer();

    // Pass tensor buffers as Buffer* (nullptr when the optional is absent) so the
    // ProgramDescriptor framework registers them as BufferBindings and patches their
    // addresses on program-cache hits. Raw uint32_t addresses here would be baked in on
    // the first dispatch and go stale once these tensors are reallocated on a later call,
    // because the presence of the input/output Buffer* bindings already puts this op on
    // the fast cache-hit path (which skips create_descriptor() and only patches bindings).
    auto* const gamma_buf = gamma_has_value ? gamma.value().buffer() : nullptr;
    auto* const beta_buf = beta_has_value ? beta.value().buffer() : nullptr;
    auto* const mean_buf = mean_has_value ? mean.value().buffer() : nullptr;
    auto* const rstd_buf = rstd_has_value ? rstd.value().buffer() : nullptr;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_rows_per_core;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        reader_desc.emplace_runtime_args(
            core,
            {input_buf, gamma_buf, beta_buf, num_rows_per_core, num_inner, tile_offset, scaler_u, e_u, mask_h, mask_w});

        writer_desc.emplace_runtime_args(
            core,
            {output_buf,
             mean_buf,
             rstd_buf,
             num_rows_per_core,
             num_inner,
             tile_offset,
             mean_rstd_height,
             mean_rstd_width,
             normalized_dims});

        tile_offset += num_rows_per_core * num_inner;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_1));
    if (has_core_group_2) {
        desc.kernels.push_back(std::move(compute_desc_2));
    }

    return desc;
}

}  // namespace ttnn::operations::moreh::moreh_layer_norm
