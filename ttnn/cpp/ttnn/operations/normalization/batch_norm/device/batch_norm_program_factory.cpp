// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "batch_norm_device_operation.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include <bit>
#include <cmath>

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using namespace ttnn::operations::normalization;

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> extract_shape_dims(const ttnn::Tensor& x) {
    const auto& shape = x.padded_shape();
    const auto& tile = x.tensor_spec().tile();
    return {shape[-4], shape[-3], shape[-2] / tile.get_height(), shape[-1] / tile.get_width()};
}

void populate_runtime_arguments(
    tt::tt_metal::KernelDescriptor& reader_desc,
    tt::tt_metal::KernelDescriptor& writer_desc,
    tt::tt_metal::KernelDescriptor& compute_desc,
    tt::tt_metal::CoreCoord compute_with_storage_grid_size,
    bool any_float32,
    const BatchNormOperation::operation_attributes_t& operation_attributes,
    const BatchNormOperation::tensor_args_t& tensor_args,
    BatchNormOperation::tensor_return_value_t& c) {
    const auto& [input_tensor, batch_mean_tensor, batch_var_tensor, weight_tensor, bias_tensor, _] = tensor_args;
    const auto eps = operation_attributes.eps;

    const bool weight_has_value = weight_tensor.has_value();
    const bool bias_has_value = bias_tensor.has_value();

    const auto [aN, aC, aHt, aWt] = extract_shape_dims(input_tensor);
    const auto [bN, bC, bHt, bWt] = extract_shape_dims(batch_mean_tensor);
    const auto [cN, cC, cHt, cWt] = extract_shape_dims(c);

    uint32_t num_output_tiles = c.physical_volume() / c.tensor_spec().tile().get_tile_hw();

    constexpr bool row_major = true;
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    auto
        [_unused_num_cores,
         _unused_all_cores,
         core_group_1,
         core_group_2,
         num_tiles_per_core_group_1,
         num_tiles_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, row_major);

    auto cores = grid_to_cores(num_cores_total, num_cores_x, num_cores_y, row_major);

    constexpr size_t num_reader_args = 11;
    constexpr size_t num_writer_args = 14;
    constexpr size_t num_kernel_args = 3;
    for (uint32_t i = 0, start_tile_id = 0; i < num_cores_total; i++) {
        const auto& core = cores[i];

        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            reader_desc.runtime_args.emplace_back(
                core, tt::tt_metal::KernelDescriptor::CoreRuntimeArgs(num_reader_args, 0));
            writer_desc.runtime_args.emplace_back(
                core, tt::tt_metal::KernelDescriptor::CoreRuntimeArgs(num_writer_args, 0));
            compute_desc.runtime_args.emplace_back(
                core, tt::tt_metal::KernelDescriptor::CoreRuntimeArgs(num_kernel_args, 0));
            continue;
        }

        uint32_t cHtWt = cHt * cWt;
        const auto scalar = eps;
        const auto packed_scalar_eps =
            any_float32 ? std::bit_cast<uint32_t>(scalar) : pack_two_bfloat16_into_uint32({scalar, scalar});

        reader_desc.emplace_runtime_args(
            core,
            {packed_scalar_eps,
             input_tensor.buffer(),
             start_tile_id,
             num_tiles_per_core,
             cHtWt,
             aHt * aWt * aC * static_cast<uint32_t>(aN > 1),
             aHt * aWt * static_cast<uint32_t>(aC > 1),
             cN,
             cC,
             cHt,
             cWt});

        std::variant<uint32_t, tt::tt_metal::Buffer*> weight_arg = 0u;
        if (weight_has_value) {
            weight_arg = weight_tensor->buffer();
        }
        std::variant<uint32_t, tt::tt_metal::Buffer*> bias_arg = 0u;
        if (bias_has_value) {
            bias_arg = bias_tensor->buffer();
        }
        writer_desc.emplace_runtime_args(
            core,
            {batch_mean_tensor.buffer(),  //  batch mean
             batch_var_tensor.buffer(),   //  batch var
             weight_arg,                  // weight
             bias_arg,                    // bias
             c.buffer(),                  // output
             start_tile_id,
             num_tiles_per_core,
             cHtWt,
             bHt * bWt * bC * static_cast<uint32_t>(bN > 1),
             bHt * bWt * static_cast<uint32_t>(bC > 1),
             cN,
             cC,
             cHt,
             cWt});

        auto counter = start_tile_id % cHtWt;
        auto freq = cHtWt;

        tt::tt_metal::KernelDescriptor::CoreRuntimeArgs compute_runtime_args = {num_tiles_per_core, freq, counter};
        compute_desc.runtime_args.emplace_back(core, std::move(compute_runtime_args));

        start_tile_id += num_tiles_per_core;
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace ttnn::operations::normalization {
tt::tt_metal::ProgramDescriptor BatchNormOperation::BatchNormFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& [input_tensor, batch_mean_tensor, batch_var_tensor, weight_tensor, bias_tensor, _] = tensor_args;

    ProgramDescriptor desc;

    auto* device = input_tensor.device();

    const bool weight_has_value = weight_tensor.has_value();
    const bool bias_has_value = bias_tensor.has_value();

    auto a_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    auto b_data_format = datatype_to_dataformat_converter(batch_mean_tensor.dtype());
    auto c_data_format = datatype_to_dataformat_converter(output.dtype());
    auto d_data_format = datatype_to_dataformat_converter(batch_var_tensor.dtype());
    auto e_data_format =
        weight_has_value ? datatype_to_dataformat_converter(weight_tensor->dtype()) : DataFormat::Float16_b;
    auto f_data_format =
        bias_has_value ? datatype_to_dataformat_converter(bias_tensor->dtype()) : DataFormat::Float16_b;

    const bool any_float32 =
        (a_data_format == DataFormat::Float32 || b_data_format == DataFormat::Float32 ||
         c_data_format == DataFormat::Float32 || d_data_format == DataFormat::Float32 ||
         e_data_format == DataFormat::Float32 || f_data_format == DataFormat::Float32);
    auto interm_data_format = any_float32 ? DataFormat::Float32 : a_data_format;

    uint32_t a_single_tile_size = tt::tile_size(a_data_format);
    uint32_t b_single_tile_size = tt::tile_size(b_data_format);
    uint32_t c_single_tile_size = tt::tile_size(c_data_format);
    uint32_t d_single_tile_size = tt::tile_size(d_data_format);
    uint32_t e_single_tile_size = tt::tile_size(e_data_format);
    uint32_t f_single_tile_size = tt::tile_size(f_data_format);
    uint32_t interm_single_tile_size = tt::tile_size(interm_data_format);

    // If accumulation occurs in float32 but output dtype is different, the compute kernel must typecast from
    // float32 to output dtype
    const bool needs_output_typecast =
        (interm_data_format == DataFormat::Float32 && c_data_format != DataFormat::Float32);

    // we parallelize the computation across the output tiles
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRangeSet(CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1}));

    // Number of tiles to store per input CB (double buffer)
    constexpr uint32_t num_tiles_per_cb = 2;
    uint32_t b_num_tiles_per_cb = num_tiles_per_cb;

    // Input buffers
    uint32_t input_tensor_cb = static_cast<uint32_t>(tt::CBIndex::c_0);
    desc.cbs.push_back(CBDescriptor{
        .total_size = a_single_tile_size * num_tiles_per_cb,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_tensor_cb),
            .data_format = a_data_format,
            .page_size = a_single_tile_size,
        }}},
    });  // input
    uint32_t batch_mean_tensor_cb = static_cast<uint32_t>(tt::CBIndex::c_1);
    desc.cbs.push_back(CBDescriptor{
        .total_size = b_single_tile_size * b_num_tiles_per_cb,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(batch_mean_tensor_cb),
            .data_format = b_data_format,
            .page_size = b_single_tile_size,
        }}},
    });  // batch_mean
    uint32_t output_tensor_cb = static_cast<uint32_t>(tt::CBIndex::c_2);
    desc.cbs.push_back(CBDescriptor{
        .total_size = (needs_output_typecast ? interm_single_tile_size : c_single_tile_size) * num_tiles_per_cb,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_tensor_cb),
            .data_format = needs_output_typecast ? interm_data_format : c_data_format,
            .page_size = needs_output_typecast ? interm_single_tile_size : c_single_tile_size,
        }}},
    });  // compute output (staging when typecast)

    uint32_t writer_output_cb = output_tensor_cb;
    if (needs_output_typecast) {
        uint32_t writer_cb = static_cast<uint32_t>(tt::CBIndex::c_9);
        desc.cbs.push_back(CBDescriptor{
            .total_size = c_single_tile_size * num_tiles_per_cb,
            .core_ranges = all_device_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(writer_cb),
                .data_format = c_data_format,
                .page_size = c_single_tile_size,
            }}},
        });  // writer-facing output (BF16)
        writer_output_cb = writer_cb;
    }
    uint32_t batch_var_tensor_cb = static_cast<uint32_t>(tt::CBIndex::c_3);
    desc.cbs.push_back(CBDescriptor{
        .total_size = d_single_tile_size * b_num_tiles_per_cb,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(batch_var_tensor_cb),
            .data_format = d_data_format,
            .page_size = d_single_tile_size,
        }}},
    });  // batch_var
    uint32_t eps_cb = static_cast<uint32_t>(tt::CBIndex::c_4);
    desc.cbs.push_back(CBDescriptor{
        .total_size = interm_single_tile_size * b_num_tiles_per_cb,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(eps_cb),
            .data_format = interm_data_format,
            .page_size = interm_single_tile_size,
        }}},
    });  // eps
    uint32_t weight_tensor_cb = static_cast<uint32_t>(tt::CBIndex::c_5);
    desc.cbs.push_back(CBDescriptor{
        .total_size = e_single_tile_size * b_num_tiles_per_cb,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(weight_tensor_cb),
            .data_format = e_data_format,
            .page_size = e_single_tile_size,
        }}},
    });  // weight
    uint32_t bias_tensor_cb = static_cast<uint32_t>(tt::CBIndex::c_6);
    desc.cbs.push_back(CBDescriptor{
        .total_size = f_single_tile_size * b_num_tiles_per_cb,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(bias_tensor_cb),
            .data_format = f_data_format,
            .page_size = f_single_tile_size,
        }}},
    });  // bias

    // Temporary buffers to store intermediate results
    uint32_t den_cb = static_cast<uint32_t>(tt::CBIndex::c_7);
    desc.cbs.push_back(CBDescriptor{
        .total_size = interm_single_tile_size * num_tiles_per_cb,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(den_cb),
            .data_format = interm_data_format,
            .page_size = interm_single_tile_size,
        }}},
    });  // to store 1/(sqrt(batch_var + eps))
    uint32_t temp_1_cb = static_cast<uint32_t>(tt::CBIndex::c_8);
    desc.cbs.push_back(CBDescriptor{
        .total_size = interm_single_tile_size * num_tiles_per_cb,
        .core_ranges = all_device_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(temp_1_cb),
            .data_format = interm_data_format,
            .page_size = interm_single_tile_size,
        }}},
    });

    std::vector<uint32_t> reader_compile_time_args = {
        input_tensor_cb,
        eps_cb,
    };
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_time_args);
    reader_compile_time_args.push_back(static_cast<uint32_t>(any_float32));

    std::vector<uint32_t> writer_compile_time_args = {
        static_cast<uint32_t>(weight_has_value),
        static_cast<uint32_t>(bias_has_value),
        batch_mean_tensor_cb,
        writer_output_cb,
        batch_var_tensor_cb,
        weight_tensor_cb,
        bias_tensor_cb,
    };
    tt::tt_metal::TensorAccessorArgs(batch_mean_tensor.buffer()).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(batch_var_tensor.buffer()).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(weight_tensor ? weight_tensor->buffer() : nullptr)
        .append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(bias_tensor ? bias_tensor->buffer() : nullptr).append_to(writer_compile_time_args);
    writer_compile_time_args.push_back(static_cast<uint32_t>(b_data_format == DataFormat::Float32));
    auto param_data_format =
        weight_has_value ? e_data_format : (bias_has_value ? f_data_format : DataFormat::Float16_b);
    writer_compile_time_args.push_back(static_cast<uint32_t>(param_data_format == DataFormat::Float32));

    // READER KERNEL
    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/reader_batch_norm.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_device_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    // WRITER KERNEL
    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/dataflow/writer_batch_norm.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_device_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.config = WriterConfigDescriptor{};

    // COMPUTE KERNEL
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        for (const auto cb_index :
             {input_tensor_cb,
              batch_mean_tensor_cb,
              batch_var_tensor_cb,
              eps_cb,
              den_cb,
              weight_tensor_cb,
              temp_1_cb,
              bias_tensor_cb}) {
            unpack_to_dest_mode[cb_index] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        }
    }

    std::vector<uint32_t> compute_kernel_args = {
        static_cast<uint32_t>(weight_has_value),
        static_cast<uint32_t>(bias_has_value),
        input_tensor_cb,
        batch_mean_tensor_cb,
        output_tensor_cb,
        batch_var_tensor_cb,
        eps_cb,
        den_cb,
        weight_tensor_cb,
        temp_1_cb,
        bias_tensor_cb,
        writer_output_cb,
        static_cast<uint32_t>(needs_output_typecast),
        static_cast<uint32_t>(DataFormat::Float32),
        needs_output_typecast ? static_cast<uint32_t>(c_data_format) : static_cast<uint32_t>(DataFormat::Float32)};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = fmt::format(
        "ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_{}.cpp",
        (fp32_dest_acc_en || any_float32) ? "sfpu_kernel" : "kernel");
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_device_cores;
    compute_desc.compile_time_args = compute_kernel_args;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
        .math_approx_mode = math_approx_mode,
    };

    CMAKE_UNIQUE_NAMESPACE::populate_runtime_arguments(
        reader_desc,
        writer_desc,
        compute_desc,
        compute_with_storage_grid_size,
        any_float32,
        operation_attributes,
        tensor_args,
        output);

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));
    return desc;
}

}  // namespace ttnn::operations::normalization
