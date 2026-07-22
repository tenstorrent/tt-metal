// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include "moreh_sum_device_operation.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/reduction/generic/device/common.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op.hpp"

namespace ttnn::operations::moreh::moreh_sum {

tt::tt_metal::ProgramDescriptor MorehSumOperation::MorehSumWFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const DeviceComputeKernelConfig& compute_kernel_config = operation_attributes.compute_kernel_config;

    ReduceOpMath reduce_op = ReduceOpMath::SUM;
    ReduceOpDim reduce_dim = ReduceOpDim::W;
    float scaler = 1.0f;

    const auto& shape = input.padded_shape();
    const auto [W, H, other_dims_product] = extract_spatial_dims(shape);

    uint32_t Wt = W / constants::TILE_WIDTH;
    uint32_t Ht = H / constants::TILE_HEIGHT;

    // check mask for w-dim
    const auto& input_shape_without_padding = input.logical_shape();
    const auto origin_W = input_shape_without_padding[-1];
    const bool do_mask_w = (origin_W % constants::TILE_WIDTH) != 0;
    const auto mask_w = do_mask_w ? origin_W % constants::TILE_WIDTH : constants::TILE_WIDTH;

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
    // Scaler datatype is hardcoded bfloat16 due to tile creation in reader
    DataFormat scaler_cb_data_format = DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tile_size(scaler_cb_data_format);
    DataFormat mask_w_cb_data_format = DataFormat::Float16_b;
    uint32_t mask_w_single_tile_size = tile_size(mask_w_cb_data_format);
    DataFormat intermed_cb_data_format = (fp32_dest_acc_en) ? DataFormat::Float32 : DataFormat::Float16_b;
    DataFormat intermed1_cb_data_format = DataFormat::Float16_b;
    uint32_t intermed_single_tile_size = tile_size(intermed_cb_data_format);
    DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tile_size(dst_cb_data_format);

    IDevice* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto num_rows = other_dims_product * Ht;

    const CoreRange all_core_range(
        {0, 0}, {compute_with_storage_grid_size.x - 1, compute_with_storage_grid_size.y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        split_work_to_cores_wt_core_range(all_core_range, num_rows);

    std::string compute_kernel_name =
        "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_w_impl_kernels/moreh_sum_w.cpp";

    ProgramDescriptor desc;

    // ---- Circular buffers ----
    // Reader batches read_batch async reads per NoC barrier (see
    // reader_moreh_sum_w.cpp). Size cb_in0 to 2*read_batch so the reader can
    // double-buffer (run a batch ahead of compute), overlapping NoC round-trips.
    // The op was reader(BRISC)-bound; read_batch=4 measured optimal on Blackhole
    // P100 (-16% to -41% DEVICE KERNEL DURATION across Wt={4,8,16}; tiny Wt=2
    // gets -4.5%). Larger batches revert the gain once NoC-read overlap
    // saturates (coarser reserve/push granularity stalls the handoff). Cap by Wt
    // so short per-row streams don't over-reserve.
    const uint32_t read_batch = std::min<uint32_t>(4u, std::max<uint32_t>(Wt, 1u));
    const uint32_t num_input_tiles = std::max<uint32_t>(2u * read_batch, 2u);
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
        .total_size = num_input_tiles * scaler_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_2),
            .data_format = scaler_cb_data_format,
            .page_size = scaler_single_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = mask_w_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_3),
            .data_format = mask_w_cb_data_format,
            .page_size = mask_w_single_tile_size,
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
    bfloat16 bfloat_scaler_value(scaler);
    uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});

    KernelDescriptor::CompileTimeArgs reader_compile_time_args;
    TensorAccessorArgs(*input.buffer()).append_to(reader_compile_time_args);
    reader_compile_time_args.push_back(packed_scaler_value);
    reader_compile_time_args.push_back(read_batch);

    KernelDescriptor::Defines reader_defines;
    if (do_mask_w) {
        reader_defines.emplace_back("DO_MASK_W", "1");
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_w_impl_kernels/reader_moreh_sum_w.cpp";
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
        "ttnn/cpp/ttnn/operations/moreh/moreh_sum/device/moreh_sum_w_impl_kernels/writer_moreh_sum_w.cpp";
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
        num_rows_per_core_group_1,  // Ht
        Wt,                         // Wt
        1,                          // NC
        origin_W,
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
            num_rows_per_core_group_2,  // Ht
            Wt,                         // Wt
            1,                          // NC
            origin_W,
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
    uint32_t out_dim_divider = Wt;
    auto* const input_buf = input.buffer();
    auto* const output_buf = output.buffer();
    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }
        uint32_t num_tensor_tiles_per_core = num_rows_per_core * Wt;

        reader_desc.emplace_runtime_args(
            core,
            {input_buf,
             num_tensor_tiles_per_core,
             num_tiles_read,  // tile index of row to start reading from
             mask_w});

        writer_desc.emplace_runtime_args(
            core,
            {
                output_buf,
                num_tensor_tiles_per_core / out_dim_divider,  // number of tiles to write
                num_tiles_read / out_dim_divider              // output tile start index
            });
        num_tiles_read += num_tensor_tiles_per_core;
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
