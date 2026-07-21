// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "moreh_nll_loss_step1_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace ttnn::operations::moreh::moreh_nll_loss_step1 {

tt::tt_metal::ProgramDescriptor MorehNllLossStep1DeviceOperation::Factory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const Tensor& target = tensor_args.target_tensor;
    const std::optional<Tensor>& weight = tensor_args.weight_tensor;
    const Tensor& output = tensor_return_value;
    const uint32_t ignore_index = operation_attributes.ignore_index;
    const uint32_t channel_size = operation_attributes.channel_size;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    auto target_shape = target.padded_shape();
    const bool weight_has_value = weight.has_value();
    auto H = target_shape[-2];
    auto W = target_shape[-1];
    auto Ht = H / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;

    // copy TILE per core
    uint32_t units_to_divide = target.physical_volume() / H / W * (Ht * Wt);

    tt::tt_metal::IDevice* device = target.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t core_h = grid.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    // create circular buffers
    const auto target_data_format = tt_metal::datatype_to_dataformat_converter(target.dtype());
    const auto data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    const auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    const auto target_tile_size = tt::tile_size(target_data_format);
    const auto data_tile_size = tt::tile_size(data_format);
    const auto intermed_tile_size = tt::tile_size(intermed_data_format);

    const uint32_t available_L1 =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);

    uint32_t target_num_tile = 1;
    uint32_t weight_num_tile = weight_has_value ? div_up(channel_size, tt::constants::TILE_WIDTH) : 0;
    uint32_t intermed_num_tile = 1;
    uint32_t output_num_tile = 1;
    uint32_t cb_usage = (target_num_tile * target_tile_size) + (weight_num_tile * data_tile_size) +
                        (intermed_num_tile * intermed_tile_size) + (output_num_tile * data_tile_size);

    const bool use_large_algorithm = cb_usage >= available_L1;

    ProgramDescriptor desc;

    // target CB (always Int32, single tile)
    desc.cbs.push_back(CBDescriptor{
        .total_size = target_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_0),
            .data_format = tt::DataFormat::Int32,
            .page_size = target_tile_size,
        }}},
    });

    // weight CB:
    //   - large algorithm: always allocate 1 tile (single-tile streaming through CB)
    //   - small algorithm: allocate weight_num_tile tiles (skip CB when weight is absent and weight_num_tile == 0)
    {
        const uint32_t weight_cb_tiles = use_large_algorithm ? 1u : weight_num_tile;
        if (weight_cb_tiles > 0) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = weight_cb_tiles * data_tile_size,
                .core_ranges = all_cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(CBIndex::c_1),
                    .data_format = data_format,
                    .page_size = data_tile_size,
                }}},
            });
        }
    }

    // tmp_weight (intermed) CB
    desc.cbs.push_back(CBDescriptor{
        .total_size = intermed_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_24),
            .data_format = intermed_data_format,
            .page_size = intermed_tile_size,
        }}},
    });

    // output CB
    desc.cbs.push_back(CBDescriptor{
        .total_size = data_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(CBIndex::c_16),
            .data_format = data_format,
            .page_size = data_tile_size,
        }}},
    });

    if (weight_has_value) {
        // This CB will be used as scratch storage when reading data from DRAM into L1,
        // since the two have different alignment requirements on some architectures.
        // Need space for only a single tile in scratch CB, because content is read immediately after writing.
        desc.cbs.push_back(CBDescriptor{
            .total_size = data_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(CBIndex::c_7),
                .data_format = data_format,
                .page_size = data_tile_size,
            }}},
        });
    }

    // create read/write kernel
    KernelDescriptor::CompileTimeArgs reader_compile_time_args{static_cast<uint32_t>(weight_has_value)};
    TensorAccessorArgs(*target.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(weight.has_value() ? weight.value().buffer() : nullptr).append_to(reader_compile_time_args);

    KernelDescriptor::CompileTimeArgs writer_compile_time_args{};
    TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor::Defines reader_defines;
    KernelDescriptor::Defines writer_defines;

    if (weight_has_value) {
        reader_defines.emplace_back("WEIGHT", "1");
    }

    if (fp32_dest_acc_en) {
        reader_defines.emplace_back("FP32_DEST_ACC_EN", "1");
    }
    const auto* const reader_kernel_file =
        use_large_algorithm ? "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step1/device/"
                              "kernels/reader_moreh_nll_loss_step1_large.cpp"
                            : "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step1/device/"
                              "kernels/reader_moreh_nll_loss_step1.cpp";
    const auto* const writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_nll_loss/moreh_nll_loss_step1/device/kernels/"
        "writer_moreh_nll_loss_step1.cpp";

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel_file;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = std::move(reader_defines);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = writer_kernel_file;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.defines = std::move(writer_defines);
    writer_desc.config = WriterConfigDescriptor{};

    auto* const target_buf = target.buffer();
    // Pass Buffer* (not a raw address) so the program-cache fast hit path re-patches the binding
    // when the tensor is reallocated; nullptr is fine for an absent optional (framework emits 0u).
    auto* const weight_buf = weight_has_value ? weight.value().buffer() : nullptr;
    auto* const output_buf = output.buffer();

    // Set Runtime Args
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h, i % core_h};
        uint32_t num_units_per_core;
        if (core_group_1.contains(core)) {
            num_units_per_core = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        uint32_t element_size = weight_has_value ? weight.value().element_size() : 0;

        reader_desc.emplace_runtime_args(
            core,
            {
                target_buf,
                weight_buf,
                static_cast<uint32_t>(ignore_index),
                num_units_per_core,
                tile_offset,
                channel_size,
                weight_num_tile,
                element_size,
                target.element_size(),
            });

        writer_desc.emplace_runtime_args(core, {output_buf, num_units_per_core, tile_offset});

        tile_offset += num_units_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss_step1
