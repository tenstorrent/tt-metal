// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <string>

#include <tt-metalium/constants.hpp>
#include "moreh_nll_loss_step2_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace ttnn::operations::moreh::moreh_nll_loss_step2 {

namespace {

// Helper: append a CB with a given tile-count (skips creation when num_tiles == 0).
void push_cb(
    ProgramDescriptor& desc,
    const CoreRangeSet& core_ranges,
    uint8_t buffer_index,
    uint32_t num_tiles,
    tt::DataFormat data_format) {
    if (num_tiles == 0) {
        return;
    }
    const auto tile_sz = tt::tile_size(data_format);
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles * tile_sz,
        .core_ranges = core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = buffer_index,
            .data_format = data_format,
            .page_size = tile_sz,
        }}},
    });
}

}  // namespace

tt::tt_metal::ProgramDescriptor moreh_nll_loss_step2_impl_2d(
    const Tensor& input,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& divisor,
    const Tensor& output,
    const std::string& /*reduction*/,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig compute_kernel_config) {
    // split work
    auto input_shape = input.padded_shape();

    auto N = input_shape[0];

    // copy 32 Bytes per core
    uint32_t units_to_divide = N / tt::constants::TILE_HEIGHT;
    const auto& input_shape_without_padding = input.logical_shape();
    const auto origin_N = input_shape_without_padding[0];
    const auto origin_C = input_shape_without_padding[1];

    const bool weight_has_value = weight.has_value();
    const bool divisor_has_value = divisor.has_value();

    tt::tt_metal::IDevice* device = input.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    ProgramDescriptor desc;

    // create circular buffers
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());

    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_0), 1, data_format);                          // input
    push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_1), 1, tt::DataFormat::Int32);                // target
    push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_2), weight_has_value ? 1 : 0, data_format);   // weight
    push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_3), divisor_has_value ? 1 : 0, data_format);  // divisor
    push_cb(
        desc, all_cores, static_cast<uint8_t>(CBIndex::c_24), 1, fp32_dest_acc_en_data_format);  // tmp_weight to reduce
    push_cb(
        desc, all_cores, static_cast<uint8_t>(CBIndex::c_25), 1, fp32_dest_acc_en_data_format);  // tmp_input to reduce
    push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_26), 1, fp32_dest_acc_en_data_format);  // tmp1
    push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_27), 1, fp32_dest_acc_en_data_format);  // tmp2
    push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_28), 1, fp32_dest_acc_en_data_format);  // tmp3
    push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_16), 1, data_format);                   // output

    if (weight_has_value) {
        // This CB will be used as scratch storage when reading data from DRAM into L1,
        // since the two have different alignment requirements on some architectures.
        // Need space for only a single tile in scratch CB, because content is read immediately after writing.
        push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_7), 1, data_format);
    }

    // create read/write kernel
    KernelDescriptor::CompileTimeArgs reader_compile_time_args{};
    TensorAccessorArgs(*input.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*target.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(weight.has_value() ? weight.value().buffer() : nullptr).append_to(reader_compile_time_args);
    TensorAccessorArgs(divisor.has_value() ? divisor.value().buffer() : nullptr).append_to(reader_compile_time_args);

    KernelDescriptor::CompileTimeArgs writer_compile_time_args{};
    TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor::Defines reader_defines;
    KernelDescriptor::Defines writer_defines;
    KernelDescriptor::Defines compute_defines;

    if (weight_has_value) {
        reader_defines.emplace_back("WEIGHT", "1");
        compute_defines.emplace_back("WEIGHT", "1");
    }
    if (divisor_has_value) {
        reader_defines.emplace_back("DIVISOR", "1");
        compute_defines.emplace_back("DIVISOR", "1");
    }

    if (fp32_dest_acc_en) {
        reader_defines.emplace_back("FP32_DEST_ACC_EN", "1");
        compute_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
        "reader_moreh_nll_loss_step2_2d.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = std::move(reader_defines);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
        "writer_moreh_nll_loss_step2_2d.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.defines = std::move(writer_defines);
    writer_desc.config = WriterConfigDescriptor{};

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);

    const auto* const compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
        "moreh_nll_loss_step2_kernel.cpp";

    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = compute_kernel_file;
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = {units_per_core_group_1};
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
        compute_desc_2.kernel_source = compute_kernel_file;
        compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_2.core_ranges = core_group_2;
        compute_desc_2.compile_time_args = {units_per_core_group_2};
        compute_desc_2.defines = compute_defines;
        compute_desc_2.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = math_approx_mode,
        };
    }

    auto* const input_buf = input.buffer();
    auto* const target_buf = target.buffer();
    // Pass Buffer* (not a raw address) so the program-cache fast hit path re-patches the binding
    // when the tensor is reallocated; nullptr is fine for an absent optional (framework emits 0u).
    auto* const weight_buf = weight_has_value ? weight.value().buffer() : nullptr;
    auto* const divisor_buf = divisor_has_value ? divisor.value().buffer() : nullptr;
    auto* const output_buf = output.buffer();

    // Set Runtime Args
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t units_per_core;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        reader_desc.emplace_runtime_args(
            core,
            {
                input_buf,
                target_buf,
                weight_buf,
                divisor_buf,
                static_cast<uint32_t>(ignore_index),
                units_per_core,
                tile_offset,
                origin_N,
                origin_C,
                input.element_size(),
            });

        writer_desc.emplace_runtime_args(
            core,
            {
                output_buf,
                units_per_core,
                tile_offset,
                origin_N,
            });

        // compute
        const KernelDescriptor::CoreRuntimeArgs compute_runtime_args{units_per_core};

        if (core_group_1.contains(core)) {
            compute_desc_1.runtime_args.emplace_back(core, compute_runtime_args);
        } else if (core_group_2.contains(core)) {
            compute_desc_2.runtime_args.emplace_back(core, compute_runtime_args);
        } else {
            TT_FATAL(false, "Core not in specified core ranges.");
        }

        tile_offset += units_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_1));
    if (has_core_group_2) {
        desc.kernels.push_back(std::move(compute_desc_2));
    }

    return desc;
}

tt::tt_metal::ProgramDescriptor moreh_nll_loss_step2_impl_3d(
    const Tensor& input,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& divisor,
    const Tensor& output,
    const std::string& /*reduction*/,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    // split work
    const auto& input_shape_without_padding = input.logical_shape();
    const auto origin_N = input_shape_without_padding[0];
    const auto origin_C = input_shape_without_padding[1];
    const auto origin_W = input_shape_without_padding[2];

    const bool weight_has_value = weight.has_value();
    const bool divisor_has_value = divisor.has_value();

    tt::tt_metal::IDevice* device = input.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    // copy FACE_WIDTH per core
    uint32_t units_to_divide = origin_N * div_up(origin_W, tt::constants::FACE_WIDTH);

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    ProgramDescriptor desc;

    // create circular buffers
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());

    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_0), 1, data_format);                          // input
    push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_1), 1, tt::DataFormat::Int32);                // target
    push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_2), weight_has_value ? 1 : 0, data_format);   // weight
    push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_3), divisor_has_value ? 1 : 0, data_format);  // divisor
    push_cb(
        desc, all_cores, static_cast<uint8_t>(CBIndex::c_24), 1, fp32_dest_acc_en_data_format);  // tmp_weight to reduce
    push_cb(
        desc, all_cores, static_cast<uint8_t>(CBIndex::c_25), 1, fp32_dest_acc_en_data_format);  // tmp_input to reduce
    push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_26), 1, fp32_dest_acc_en_data_format);  // tmp1
    push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_27), 1, fp32_dest_acc_en_data_format);  // tmp2
    push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_28), 1, fp32_dest_acc_en_data_format);  // tmp3
    push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_16), 1, data_format);                   // output

    if (weight_has_value) {
        // This CB will be used as scratch storage when reading data from DRAM into L1,
        // since the two have different alignment requirements on some architectures.
        // Need space for only a single tile in scratch CB, because content is read immediately after writing.
        push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_7), 1, data_format);
    }

    // create read/write kernel
    KernelDescriptor::CompileTimeArgs reader_compile_time_args{};
    TensorAccessorArgs(*input.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*target.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(weight.has_value() ? weight.value().buffer() : nullptr).append_to(reader_compile_time_args);
    TensorAccessorArgs(divisor.has_value() ? divisor.value().buffer() : nullptr).append_to(reader_compile_time_args);

    KernelDescriptor::CompileTimeArgs writer_compile_time_args{};
    TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor::Defines reader_defines;
    KernelDescriptor::Defines writer_defines;
    KernelDescriptor::Defines compute_defines;

    if (weight_has_value) {
        reader_defines.emplace_back("WEIGHT", "1");
        compute_defines.emplace_back("WEIGHT", "1");
    }
    if (divisor_has_value) {
        reader_defines.emplace_back("DIVISOR", "1");
        compute_defines.emplace_back("DIVISOR", "1");
    }

    if (fp32_dest_acc_en) {
        reader_defines.emplace_back("FP32_DEST_ACC_EN", "1");
        compute_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
        "reader_moreh_nll_loss_step2_3d.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = std::move(reader_defines);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
        "writer_moreh_nll_loss_step2_3d.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.defines = std::move(writer_defines);
    writer_desc.config = WriterConfigDescriptor{};

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);

    const auto* const compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
        "moreh_nll_loss_step2_kernel.cpp";

    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = compute_kernel_file;
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = {units_per_core_group_1};
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
        compute_desc_2.kernel_source = compute_kernel_file;
        compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_2.core_ranges = core_group_2;
        compute_desc_2.compile_time_args = {units_per_core_group_2};
        compute_desc_2.defines = compute_defines;
        compute_desc_2.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = math_approx_mode,
        };
    }

    auto* const input_buf = input.buffer();
    auto* const target_buf = target.buffer();
    // Pass Buffer* (not a raw address) so the program-cache fast hit path re-patches the binding
    // when the tensor is reallocated; nullptr is fine for an absent optional (framework emits 0u).
    auto* const weight_buf = weight_has_value ? weight.value().buffer() : nullptr;
    auto* const divisor_buf = divisor_has_value ? divisor.value().buffer() : nullptr;
    auto* const output_buf = output.buffer();

    // Set Runtime Args
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t units_per_core;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        reader_desc.emplace_runtime_args(
            core,
            {
                input_buf,
                target_buf,
                weight_buf,
                divisor_buf,
                static_cast<uint32_t>(ignore_index),
                units_per_core,
                tile_offset,
                origin_N,
                origin_C,
                origin_W,
                input.element_size(),
            });

        writer_desc.emplace_runtime_args(
            core,
            {
                output_buf,
                units_per_core,
                tile_offset,
                origin_W,
                output.element_size(),
            });

        // compute
        const KernelDescriptor::CoreRuntimeArgs compute_runtime_args{units_per_core};

        if (core_group_1.contains(core)) {
            compute_desc_1.runtime_args.emplace_back(core, compute_runtime_args);
        } else if (core_group_2.contains(core)) {
            compute_desc_2.runtime_args.emplace_back(core, compute_runtime_args);
        } else {
            TT_FATAL(false, "Core not in specified core ranges.");
        }

        tile_offset += units_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_1));
    if (has_core_group_2) {
        desc.kernels.push_back(std::move(compute_desc_2));
    }

    return desc;
}

tt::tt_metal::ProgramDescriptor moreh_nll_loss_step2_impl_4d(
    const Tensor& input,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& divisor,
    const Tensor& output,
    const std::string& /*reduction*/,
    const uint32_t ignore_index,
    const DeviceComputeKernelConfig compute_kernel_config) {
    // split work
    auto input_shape = input.padded_shape();
    auto target_shape = target.padded_shape();
    auto N = input_shape[0];
    uint32_t channel_size = input_shape[1];

    auto H = target_shape[-2];
    auto W = target_shape[-1];
    auto Ht = H / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;
    uint32_t num_inner_tile = target.physical_volume() / N / tt::constants::TILE_HEIGHT / tt::constants::TILE_WIDTH;

    const auto& input_shape_without_padding = input.logical_shape();
    const auto origin_N = input_shape_without_padding[0];
    const auto origin_C = input_shape_without_padding[1];

    const bool weight_has_value = weight.has_value();
    const bool divisor_has_value = divisor.has_value();

    tt::tt_metal::IDevice* device = input.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    // copy TILE per loop
    uint32_t units_to_divide = target.physical_volume() / H / W * Ht * Wt;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    ProgramDescriptor desc;

    // create circular buffers
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());

    auto fp32_dest_acc_en_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    uint32_t weight_num_tile = div_up(channel_size, tt::constants::TILE_WIDTH);

    push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_0), 1, data_format);            // input
    push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_1), 1, tt::DataFormat::Int32);  // target
    push_cb(
        desc,
        all_cores,
        static_cast<uint8_t>(CBIndex::c_2),
        weight_has_value ? weight_num_tile : 0u,
        data_format);                                                                                        // weight
    push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_3), divisor_has_value ? 1u : 0u, data_format);  // divisor
    push_cb(
        desc, all_cores, static_cast<uint8_t>(CBIndex::c_24), 1, fp32_dest_acc_en_data_format);  // tmp_weight to reduce
    push_cb(
        desc, all_cores, static_cast<uint8_t>(CBIndex::c_25), 1, fp32_dest_acc_en_data_format);  // tmp_input to reduce
    push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_26), 1, fp32_dest_acc_en_data_format);  // tmp1
    push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_27), 1, fp32_dest_acc_en_data_format);  // tmp2
    push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_28), 1, fp32_dest_acc_en_data_format);  // tmp3
    push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_16), 1, data_format);                   // output

    if (weight_has_value) {
        // This CB will be used as scratch storage when reading data from DRAM into L1,
        // since the two have different alignment requirements on some architectures.
        // Need space for only a single tile in scratch CB, because content is read immediately after writing.
        push_cb(desc, all_cores, static_cast<uint8_t>(CBIndex::c_7), 1, data_format);
    }

    // create read/write kernel
    KernelDescriptor::CompileTimeArgs reader_compile_time_args{};
    TensorAccessorArgs(*input.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*target.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(weight.has_value() ? weight.value().buffer() : nullptr).append_to(reader_compile_time_args);
    TensorAccessorArgs(divisor.has_value() ? divisor.value().buffer() : nullptr).append_to(reader_compile_time_args);

    KernelDescriptor::CompileTimeArgs writer_compile_time_args{};
    TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor::Defines reader_defines;
    KernelDescriptor::Defines writer_defines;
    KernelDescriptor::Defines compute_defines;

    if (weight_has_value) {
        reader_defines.emplace_back("WEIGHT", "1");
        compute_defines.emplace_back("WEIGHT", "1");
    }
    if (divisor_has_value) {
        reader_defines.emplace_back("DIVISOR", "1");
        compute_defines.emplace_back("DIVISOR", "1");
    }

    if (fp32_dest_acc_en) {
        reader_defines.emplace_back("FP32_DEST_ACC_EN", "1");
        compute_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
        "reader_moreh_nll_loss_step2_4d.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = std::move(reader_defines);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
        "writer_moreh_nll_loss_step2_4d.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.defines = std::move(writer_defines);
    writer_desc.config = WriterConfigDescriptor{};

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);

    const auto* const compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step2/device/kernels/"
        "moreh_nll_loss_step2_kernel.cpp";

    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = compute_kernel_file;
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = {units_per_core_group_1};
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
        compute_desc_2.kernel_source = compute_kernel_file;
        compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_2.core_ranges = core_group_2;
        compute_desc_2.compile_time_args = {units_per_core_group_2};
        compute_desc_2.defines = compute_defines;
        compute_desc_2.config = ComputeConfigDescriptor{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = math_approx_mode,
        };
    }

    auto* const input_buf = input.buffer();
    auto* const target_buf = target.buffer();
    // Pass Buffer* (not a raw address) so the program-cache fast hit path re-patches the binding
    // when the tensor is reallocated; nullptr is fine for an absent optional (framework emits 0u).
    auto* const weight_buf = weight_has_value ? weight.value().buffer() : nullptr;
    auto* const divisor_buf = divisor_has_value ? divisor.value().buffer() : nullptr;
    auto* const output_buf = output.buffer();

    // Set Runtime Args
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t units_per_core;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        reader_desc.emplace_runtime_args(
            core,
            {
                input_buf,
                target_buf,
                weight_buf,
                divisor_buf,
                static_cast<uint32_t>(ignore_index),
                units_per_core,
                tile_offset,
                origin_N,
                origin_C,
                Wt,
                num_inner_tile,
                weight_num_tile,
                input.element_size(),
            });

        writer_desc.emplace_runtime_args(
            core,
            {
                output_buf,
                units_per_core,
                tile_offset,
            });

        // compute
        const KernelDescriptor::CoreRuntimeArgs compute_runtime_args{units_per_core};

        if (core_group_1.contains(core)) {
            compute_desc_1.runtime_args.emplace_back(core, compute_runtime_args);
        } else if (core_group_2.contains(core)) {
            compute_desc_2.runtime_args.emplace_back(core, compute_runtime_args);
        } else {
            TT_FATAL(false, "Core not in specified core ranges.");
        }

        tile_offset += units_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_1));
    if (has_core_group_2) {
        desc.kernels.push_back(std::move(compute_desc_2));
    }

    return desc;
}

tt::tt_metal::ProgramDescriptor MorehNllLossStep2DeviceOperation::Factory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const Tensor& input = tensor_args.input_tensor;
    const Tensor& target = tensor_args.target_tensor;
    const std::optional<Tensor>& weight = tensor_args.weight_tensor;
    const std::optional<Tensor>& divisor = tensor_args.divisor_tensor;
    const Tensor& output = tensor_return_value;
    const std::string reduction = operation_attributes.reduction;
    const uint32_t ignore_index = operation_attributes.ignore_index;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    // split work
    auto input_shape = input.padded_shape();
    auto rank = input_shape.rank();

    if (rank == 2) {
        return moreh_nll_loss_step2_impl_2d(
            input, target, weight, divisor, output, reduction, ignore_index, compute_kernel_config);
    }
    if (rank == 3) {
        return moreh_nll_loss_step2_impl_3d(
            input, target, weight, divisor, output, reduction, ignore_index, compute_kernel_config);
    }

    return moreh_nll_loss_step2_impl_4d(
        input, target, weight, divisor, output, reduction, ignore_index, compute_kernel_config);
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss_step2
