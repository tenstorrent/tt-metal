// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/topk/device/topk_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/work_split.hpp"

#include <map>
#include <string>

using namespace tt::tt_metal;

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor TopKDeviceOperation::TopKSingleCoreProgramFactory::create_descriptor(
    const TopkParams& operation_attributes,
    const TopkInputs& tensor_args,
    std::tuple<Tensor, Tensor>& tensor_return_value) {
    const auto& args = operation_attributes;
    auto& output_tensors = tensor_return_value;
    // Tensor references
    const auto& input_tensor = tensor_args.input;
    const auto& value_tensor = std::get<0>(output_tensors);
    const auto& index_tensor = std::get<1>(output_tensors);

    // Determine index output format based on dimension size constraints
    const ttnn::Shape input_shape = input_tensor.padded_shape();
    const bool uint16_output = (input_shape[args.dim] < std::numeric_limits<uint16_t>::max());

    ProgramDescriptor desc;

    // Data format conversions for circular buffer configurations
    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const tt::DataFormat output_val_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(value_tensor.dtype());
    const tt::DataFormat output_ind_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(index_tensor.dtype());

    // Use bf16 for compute intermediate buffers to avoid precision loss from bfp8/bfp4
    // shared-exponent grouping during sort (e.g. a single inf in a block makes all other
    // elements in that block encode to 0, corrupting the sort result).
    const tt::DataFormat compute_cb_data_format =
        (input_cb_data_format == tt::DataFormat::Bfp8_b || input_cb_data_format == tt::DataFormat::Bfp4_b)
            ? tt::DataFormat::Float16_b
            : input_cb_data_format;

    // Calculate tile sizes for memory allocation
    const uint32_t input_tile_size = tile_size(input_cb_data_format);
    const uint32_t value_tile_size = tile_size(output_val_cb_data_format);
    const uint32_t index_tile_size = tile_size(output_ind_cb_data_format);
    const uint32_t compute_tile_size = tile_size(compute_cb_data_format);

    // Device memory buffer pointers for kernel runtime arguments
    auto* const input_buffer = input_tensor.buffer();
    auto* const values_buffer = value_tensor.buffer();
    auto* const index_buffer = index_tensor.buffer();

    // Tensor shape and dimension calculations
    const uint32_t tile_height = input_tensor.tensor_spec().tile().get_height();
    const uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();
    const uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / tile_height;
    const uint32_t Wt = input_shape[3] / tile_width;

    // Single core selection from the provided core grid
    const auto
        [total_number_of_cores,       // number of cores utilized
         core_range,                  // set of all cores used
         core_group_1,                // Primary core group
         core_group_2,                // Secondary core group
         num_tiles_per_core_group_1,  // Number of tiles each core in the primary group processes
         num_tiles_per_core_group_2   // Number of tiles each core in the secondary group processes
    ] = tt::tt_metal::split_work_to_cores(args.sub_core_grids, Ht, true);
    const auto work_groups = {
        std::make_pair(core_group_1, num_tiles_per_core_group_1),
        std::make_pair(core_group_2, num_tiles_per_core_group_2)};
    const std::vector<CoreCoord>& cores = corerange_to_cores(core_range, total_number_of_cores, true);

    // Number of tiles needed to store K top elements
    const uint32_t Ktiles = tt::div_up(args.k, tile_width);

    // Pipeline Flow:
    // Input CB -> Reader Kernel -> Transposed CBs -> Compute Kernel -> Result Prep CBs -> Output CBs -> Writer Kernel
    const uint32_t num_cb_unit = 2;                         // Base unit for double buffering
    const uint32_t cb_in_units = num_cb_unit;               // 2 units total for input double buffering
    const uint32_t input_cb_tile_count = cb_in_units;       // Input stream buffer size
    const uint32_t transposed_cb_tile_count = 4;            // Transposed data staging
    const uint32_t result_prep_cb_tile_count = 2 * Ktiles;  // Intermediate TopK results (double-buffered)
    const uint32_t output_cb_tile_count = Ktiles;           // Final output buffer

    // Circular Buffer Creations:
    constexpr uint32_t input_cb_index = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = input_cb_tile_count * input_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_tile_size,
        }}},
    });

    constexpr uint32_t index_cb_index = tt::CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = input_cb_tile_count * index_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index_cb_index),
            .data_format = output_ind_cb_data_format,
            .page_size = index_tile_size,
        }}},
    });

    // Uses bf16 when input is bfp8/bfp4 so that the insertion sort operates at higher
    // precision and avoids shared-exponent corruption of tiles adjacent to inf values.
    constexpr uint32_t transposed_val_cb_index = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = transposed_cb_tile_count * compute_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(transposed_val_cb_index),
            .data_format = compute_cb_data_format,
            .page_size = compute_tile_size,
        }}},
    });

    constexpr uint32_t transposed_ind_cb_index = tt::CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = transposed_cb_tile_count * index_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(transposed_ind_cb_index),
            .data_format = output_ind_cb_data_format,
            .page_size = index_tile_size,
        }}},
    });

    // Uses bf16 when input is bfp8/bfp4 (same rationale as transposed_val_cb_index).
    constexpr uint32_t result_prep_val_cb_index = tt::CBIndex::c_4;
    desc.cbs.push_back(CBDescriptor{
        .total_size = result_prep_cb_tile_count * compute_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(result_prep_val_cb_index),
            .data_format = compute_cb_data_format,
            .page_size = compute_tile_size,
        }}},
    });

    constexpr uint32_t result_prep_ind_cb_index = tt::CBIndex::c_5;
    desc.cbs.push_back(CBDescriptor{
        .total_size = result_prep_cb_tile_count * index_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(result_prep_ind_cb_index),
            .data_format = output_ind_cb_data_format,
            .page_size = index_tile_size,
        }}},
    });

    constexpr uint32_t output_val_cb_index = tt::CBIndex::c_6;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_cb_tile_count * value_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_val_cb_index),
            .data_format = output_val_cb_data_format,
            .page_size = value_tile_size,
        }}},
    });

    constexpr uint32_t output_ind_cb_index = tt::CBIndex::c_7;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_cb_tile_count * index_tile_size,
        .core_ranges = core_range,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_ind_cb_index),
            .data_format = output_ind_cb_data_format,
            .page_size = index_tile_size,
        }}},
    });

    // Kernel Creations:
    std::vector<uint32_t> reader_compile_time_args = {
        input_cb_index,                       // Input values
        index_cb_index,                       // Generated indices
        Ht,                                   // Height in tiles
        Wt,                                   // Width in tiles
        total_number_of_cores,                // Total number of cores
        static_cast<uint32_t>(uint16_output)  // Index format flag
    };
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);
    if (tensor_args.indices.has_value()) {
        tt::tt_metal::TensorAccessorArgs(tensor_args.indices->buffer()).append_to(reader_compile_time_args);
    }
    const std::map<std::string, std::string> reader_defines_map = {
        {"GENERATE_INDICES", "1"},  // tensor_args.indices.has_value() ? "0" : "1" - GH issue: #36329
    };
    KernelDescriptor::Defines reader_defines(reader_defines_map.begin(), reader_defines_map.end());

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/reader_create_index_tensor.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_range;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = std::move(reader_defines);
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_compile_time_args = {
        output_val_cb_index,   // CB6: Output values
        output_ind_cb_index,   // CB7: Output indices
        Ht,                    // Height in tiles
        Ktiles,                // K value in tiles
        total_number_of_cores  // Total number of cores
    };
    tt::tt_metal::TensorAccessorArgs(values_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(index_buffer).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/dataflow/writer_binary_interleaved.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = core_range;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    const std::vector<uint32_t> compute_args = {
        input_cb_index,                            // Input values
        index_cb_index,                            // Input indices
        transposed_val_cb_index,                   // Transposed values
        transposed_ind_cb_index,                   // Transposed indices
        result_prep_val_cb_index,                  // Result prep values
        result_prep_ind_cb_index,                  // Result prep indices
        output_val_cb_index,                       // Output values
        output_ind_cb_index,                       // Output indices
        Ht,                                        // Height in tiles
        Wt,                                        // Width in tiles
        Ktiles,                                    // K value in tiles
        static_cast<std::uint32_t>(args.largest),  // Sort order: largest (true) or smallest (false)
    };

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_range;
    compute_desc.compile_time_args = compute_args;
    compute_desc.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = !uint16_output,
        .dst_full_sync_en = false,
    };

    uint32_t id = 0;  // Offset for the next core in the group
    for (const auto& [group, work_per_core] : work_groups) {
        for (const auto& range : group.ranges()) {
            for (const auto& core : range) {
                reader_desc.emplace_runtime_args(
                    core,
                    {
                        input_buffer,
                        id,
                        work_per_core,
                        tensor_args.indices.has_value() ? tensor_args.indices->buffer()->address()
                                                        : 0u,  // Optional indices tensor
                    });
                writer_desc.emplace_runtime_args(
                    core,
                    {
                        values_buffer,
                        index_buffer,
                        id,
                        work_per_core,
                    });
                compute_desc.runtime_args.emplace_back(
                    core,
                    KernelDescriptor::CoreRuntimeArgs{
                        work_per_core,
                    });
                id++;
            }
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
