// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "moreh_sum_device_operation.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/reduction/generic/device/common.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

namespace ttnn::operations::moreh::moreh_sum {

tt::tt_metal::ProgramDescriptor MorehSumOperation::MorehSumHFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const DeviceComputeKernelConfig& compute_kernel_config = operation_attributes.compute_kernel_config;

    ReduceOpMath reduce_op = ReduceOpMath::SUM;
    ReduceOpDim reduce_dim = ReduceOpDim::H;
    const auto& shape = input.padded_shape();
    const auto [W, H, other_dims_product] = extract_spatial_dims(shape);

    uint32_t Wt = W / constants::TILE_WIDTH;
    uint32_t Ht = H / constants::TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;

    // check mask for h-dim
    const auto& input_shape_without_padding = input.logical_shape();
    const auto origin_H = input_shape_without_padding[-2];
    const bool do_mask_h = (origin_H % constants::TILE_HEIGHT) != 0;
    const auto mask_h = do_mask_h ? origin_H % constants::TILE_HEIGHT : constants::TILE_HEIGHT;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input.device()->arch(), compute_kernel_config);
    log_debug(
        tt::LogOp,
        "math_fidelity {} math_approx_mode {} fp32_dest_acc_en {} packer_l1_acc {}",
        math_fidelity,
        math_approx_mode,
        fp32_dest_acc_en,
        packer_l1_acc);

    DataFormat src0_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t src0_single_tile_size = tile_size(src0_cb_data_format);
    DataFormat scaler_cb_data_format = DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tile_size(scaler_cb_data_format);
    DataFormat mask_h_cb_data_format = DataFormat::Float16_b;
    uint32_t mask_h_single_tile_size = tile_size(mask_h_cb_data_format);
    DataFormat intermed_cb_data_format = (fp32_dest_acc_en) ? DataFormat::Float32 : DataFormat::Float16_b;
    DataFormat intermed1_cb_data_format = DataFormat::Float16_b;
    uint32_t intermed_single_tile_size = tile_size(intermed_cb_data_format);
    DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tile_size(dst_cb_data_format);

    IDevice* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto num_cols = other_dims_product * Wt;

    const CoreRange all_core_range(
        {0, 0}, {compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2] =
        split_work_to_cores_wt_core_range(all_core_range, num_cols);

    std::string compute_kernel_name =
        "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_impl_kernels/moreh_sum_h.cpp";

    ProgramDescriptor desc;

    // ---- Circular buffers ----
    constexpr uint32_t num_input_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * src0_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_0),
            .data_format = src0_cb_data_format,
            .page_size = src0_single_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = scaler_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_2),
            .data_format = scaler_cb_data_format,
            .page_size = scaler_single_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = mask_h_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_3),
            .data_format = mask_h_cb_data_format,
            .page_size = mask_h_single_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = intermed_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_24),
            .data_format = intermed_cb_data_format,
            .page_size = intermed_single_tile_size,
        }}},
    });
    uint32_t intermed1_single_tile_size = tile_size(intermed1_cb_data_format);
    desc.cbs.push_back(CBDescriptor{
        .total_size = intermed1_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_25),
            .data_format = intermed1_cb_data_format,
            .page_size = intermed1_single_tile_size,
        }}},
    });
    constexpr uint32_t num_output_tiles = 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * dst_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_16),
            .data_format = dst_cb_data_format,
            .page_size = dst_single_tile_size,
        }}},
    });

    // ---- Reader kernel ----
    KernelDescriptor::CompileTimeArgs reader_compile_time_args = {Ht, Wt, HtWt};
    TensorAccessorArgs(*input.buffer()).append_to(reader_compile_time_args);

    KernelDescriptor::Defines reader_defines = {{"REDUCE_SCALER", "1"}};
    if (do_mask_h) {
        reader_defines.emplace_back("DO_MASK_H", "1");
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_impl_kernels/reader_moreh_sum_h.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = std::move(reader_defines);
    reader_desc.config = ReaderConfigDescriptor{};

    // ---- Writer kernel ----
    KernelDescriptor::CompileTimeArgs writer_compile_time_args = {static_cast<uint32_t>(CBIndex::c_16)};
    TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_h_impl_kernels/writer_moreh_sum_h.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    // ---- Compute kernels (two groups) ----
    auto reduce_defines_map = reduce_op_utils::get_defines(reduce_op, reduce_dim);
    if (fp32_dest_acc_en) {
        reduce_defines_map["FP32_DEST_ACC_EN"] = "1";
    }
    KernelDescriptor::Defines reduce_defines(reduce_defines_map.begin(), reduce_defines_map.end());

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        unpack_to_dest_mode[CBIndex::c_24] = UnpackToDestMode::UnpackToDestFp32;
    }

    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = compute_kernel_name;
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = {
        Ht,                         // Ht
        num_cols_per_core_group_1,  // Wt
        1,                          // NC
        origin_H,
    };
    compute_desc_1.defines = reduce_defines;
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
        compute_desc_2.kernel_source = compute_kernel_name;
        compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_2.core_ranges = core_group_2;
        compute_desc_2.compile_time_args = {
            Ht,                         // Ht
            num_cols_per_core_group_2,  // Wt
            1,                          // NC
            origin_H,
        };
        compute_desc_2.defines = reduce_defines;
        compute_desc_2.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = math_approx_mode,
        };
    }

    // ---- Runtime args per core ----
    auto* const input_buf = input.buffer();
    auto* const output_buf = output.buffer();
    for (uint32_t i = 0, num_cols_read = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_cols_per_core = 0;
        if (core_group_1.contains(core)) {
            num_cols_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_cols_per_core = num_cols_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }
        reader_desc.emplace_runtime_args(
            core,
            {input_buf,
             (num_cols_read / Wt * HtWt) + (num_cols_read % Wt),
             num_cols_read % Wt,
             num_cols_per_core,
             mask_h});

        writer_desc.emplace_runtime_args(
            core,
            {
                output_buf,
                num_cols_per_core,  // number of tiles to write
                num_cols_read       // output tile start index
            });
        num_cols_read += num_cols_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_1));
    if (has_core_group_2) {
        desc.kernels.push_back(std::move(compute_desc_2));
    }

    return desc;
}

}  // namespace ttnn::operations::moreh::moreh_sum
