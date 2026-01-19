// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_tile_multicore_program_factory.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::operations::data_movement::pad::program {

static inline int advance_tensor_index(std::vector<uint32_t>& idx, const ttnn::Shape& dims, uint32_t ndims) {
    // increment least-significant dim first
    for (int32_t d = ndims - 1; d >= 0; d--) {
        uint32_t v = idx[d] + 1;
        if (v < dims[d]) {
            idx[d] = v;
            return 1;
        }
        idx[d] = 0;  // wrap and carry
    }
    return 0;  // overflowed most-significant dim
}

PadTileMulticoreProgramFactory::cached_program_t PadTileMulticoreProgramFactory::create(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& pad_value = operation_attributes.pad_value;
    const auto& output_padded_shape = operation_attributes.output_padded_shape;
    Program program{};

    const auto& a_shape = a.logical_shape();
    uint32_t num_pages = get_num_pages(output);

    IDevice* device = a.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_pages);

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t page_size = output.buffer()->page_size();
    uint32_t multi_buffering_size = 2;
    uint32_t input_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig input_cb_config =
        tt::tt_metal::CircularBufferConfig(page_size * multi_buffering_size, {{input_cb_index, cb_data_format}})
            .set_page_size(input_cb_index, page_size);
    tt::tt_metal::CreateCircularBuffer(program, total_cores, input_cb_config);

    uint32_t output_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig output_cb_config =
        tt::tt_metal::CircularBufferConfig(page_size * multi_buffering_size, {{output_cb_index, cb_data_format}})
            .set_page_size(output_cb_index, page_size);
    tt::tt_metal::CreateCircularBuffer(program, total_cores, output_cb_config);

    uint32_t pad_val_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig pad_val_cb_config =
        tt::tt_metal::CircularBufferConfig(page_size, {{pad_val_cb_index, cb_data_format}})
            .set_page_size(pad_val_cb_index, page_size);
    tt::tt_metal::CreateCircularBuffer(program, total_cores, pad_val_cb_config);

    Buffer* input_buffer = a.buffer();
    Buffer* output_buffer = output.buffer();
    TT_ASSERT(output_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t packed_pad_value;
    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    switch (a.dtype()) {
        case DataType::INT32:
        case DataType::UINT32: packed_pad_value = pad_value; break;
        case DataType::BFLOAT16:
            packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_pad_value, bfloat_pad_value});
            break;
        case DataType::UINT16:
            packed_pad_value = pack_two_uint16_into_uint32({float_to_uint16(pad_value), float_to_uint16(pad_value)});
            break;
        case DataType::FLOAT32: packed_pad_value = std::bit_cast<uint32_t>(pad_value); break;
        default:
            packed_pad_value = 0;
            TT_ASSERT(
                false,
                "Unsupported datatype for pad tile multicore, can only support INT32, UINT32, BFLOAT16, UINT16, "
                "FLOAT32");
    }

    std::vector<uint32_t> reader_ct_args = {
        (std::uint32_t)input_cb_index,
        (std::uint32_t)page_size,
        (std::uint32_t)output_padded_shape.rank(),
    };
    TensorAccessorArgs(*input_buffer).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args = {
        (std::uint32_t)input_cb_index,
        (std::uint32_t)output_cb_index,
        (std::uint32_t)pad_val_cb_index,
        (std::uint32_t)page_size,
        (std::uint32_t)output_padded_shape.rank(),
        (std::uint32_t)packed_pad_value,
        (std::uint32_t)output.element_size(),
    };
    TensorAccessorArgs(*output_buffer).append_to(writer_ct_args);

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/reader_pad_tiled.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_pad_tiled.cpp",
        total_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    /*
    As an example, lets say we want to pad a [2, 1, 32, 32] tensor to [2, 3, 64, 64]
    The input tensor exists as [2, 2, 1, 1] if we reduce by tile (page) size, and the output as [2, 3, 2, 2]
    we increment through these shapes, and will write a total of 2 * 3 * 2 * 2 = 24 tiles, so we will utilize 24 cores
    for each core, we calculate if we are within the "input region" of the output. this does a check of
    if any element in the incremented input_id_per_dim is less than the output_id_per_dim, if so, we are outside
    the input region, and we will write a padding tile, and we will not increment the input_id_per_dim for that tile.
    if we are within the input region, we will write the tile from input to the output, and increment the
    input_id_per_dim. This works because we increment the least-significant dim first, and the input region correctly
    matches the output after the output wraps around. In this example:
    Core 0: input_id_per_dim: [0,0,0,0] ; output_id_per_dim: [0,0,0,0], we copy the tile and increment both input and
    output dims, next ->
    Core 1: input_id_per_dim: [0,1,0,0] ; output_id_per_dim: [0,0,0,1], the last output dim is
    greater than input, so we write the pad tile, and increment only output dim, next ->
    Core 2: input_id_per_dim: [0,1,0,0] ; output_id_per_dim: [0,0,1,0], the second last output dim is greater than
    input, so we write the pad tile, and increment only output dim, next ->
    Core 3: input_id_per_dim: [0,1,0,0] ; output_id_per_dim: [0,0,1,1], the last 2 output dims is greater than input,
    so we write the pad tile, and increment only output dim, next ->
    Core 4: input_id_per_dim: [0,1,0,0] ; output_id_per_dim: [0,1,0,0], we copy the tile and increment, next ->
    Core 5: input_id_per_dim: [1,0,0,0] ; output_id_per_dim: [0,1,0,1], Some output dims are greater
    than input, so we write the pad tile, and increment only output dim. next ->
    Core 6: input_id_per_dim: [1,0,0,0] ; output_id_per_dim: [0,1,1,0],
    Core 7, 8, 9, 10, 11, we write pad tiles, incrementing only output dim each time, next ->
    Core 12: input_id_per_dim: [1,0,0,0] ; output_id_per_dim: [1,0,0,0], we copy the tile and increment, next ->
    Core 13: input_id_per_dim: [1,1,0,0] ; output_id_per_dim: [1,0,0,1], Core 13, 14, 15, we write pad tiles,
    incrementing only output dim each time, next -> Core 16: input_id_per_dim: [1,1,0,0] ; output_id_per_dim: [1,1,0,0],
    we copy the tile and increment, next ->
    From now on, input_id_per_dim wraps around it's most significant dim, resulting in [0,0,0,0].
    This means for every output_id_per_dim, an element will always be greater than
    input_id_per_dim, so every core after core 16 will only write pad tiles, which is correct as we have filled all of
    the input region, and will always be outside of it from now on.

    As you can see, the input_id_per_dim only increments when we are within the input region of the output,
    and the output_id_per_dim increments every time, this means that when the output wraps around, the input
    will be correctly positioned for the next set of output tiles.
    */

    std::vector<uint32_t> input_id_per_dim, output_id_per_dim;  // input and output id_per_dims
    // initialize id_per_dims to vectors of length num_dims filled with 0
    input_id_per_dim.resize(a_shape.rank(), 0);
    output_id_per_dim.resize(output_padded_shape.rank(), 0);
    // instantiate the input and output tensor padded shapes
    auto input_page_shape = a.padded_shape();
    auto output_page_shape = output_padded_shape;
    input_page_shape[-1] /= tt::constants::TILE_HEIGHT;
    input_page_shape[-2] /= tt::constants::TILE_HEIGHT;
    output_page_shape[-1] /= tt::constants::TILE_HEIGHT;
    output_page_shape[-2] /= tt::constants::TILE_HEIGHT;
    bool within_input_region;
    uint32_t input_page_offset = 0;
    uint32_t output_page_offset = 0;

    std::vector<uint32_t> all_runtime_args;

    for (uint32_t i = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_pages_per_core;
        if (core_group_1.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_2;
        } else {
            num_pages_per_core = 0;  // no-op
        }

        all_runtime_args = {
            a.buffer()->address(),
            num_pages_per_core,
            input_page_offset,
        };

        // Every core should get the same input and output tile shapes
        all_runtime_args.insert(all_runtime_args.end(), input_page_shape.cbegin(), input_page_shape.cend());
        all_runtime_args.insert(all_runtime_args.end(), output_page_shape.cbegin(), output_page_shape.cend());

        // As well as where the core should start writing in the output tensor
        all_runtime_args.insert(all_runtime_args.end(), input_id_per_dim.begin(), input_id_per_dim.end());
        all_runtime_args.insert(all_runtime_args.end(), output_id_per_dim.begin(), output_id_per_dim.end());

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, all_runtime_args);
        all_runtime_args[0] = output.buffer()->address();  // change input addr to output addr before setting writer
                                                           // args
        all_runtime_args[2] =
            output_page_offset;  // change input page offset to output page offset before setting writer args
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, all_runtime_args);

        // We now need to increment the input and output id_per_dims by the number of pages this core is processing
        // Similarly to in the kernel, we only increment the input id_per_dim if we are within the input region
        for (uint32_t p = 0; p < num_pages_per_core; p++) {
            within_input_region = true;
            for (uint32_t d = 0; d < input_id_per_dim.size(); d++) {
                if (input_id_per_dim[d] < output_id_per_dim[d]) {
                    within_input_region = false;
                    break;
                }
            }
            if (within_input_region) {
                advance_tensor_index(input_id_per_dim, input_page_shape, input_id_per_dim.size());
                input_page_offset++;
            }
            advance_tensor_index(output_id_per_dim, output_page_shape, output_id_per_dim.size());
            output_page_offset++;
        }
        // The input and output id_per_dim should now be set correctly for the next core
    }

    return cached_program_t{std::move(program), {reader_kernel_id, writer_kernel_id, compute_with_storage_grid_size}};
}

void PadTileMulticoreProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const PadParams& /*operation_attributes*/,
    const PadInputs& tensor_args,
    Tensor& output) {
    auto* src_buffer = tensor_args.input.buffer();
    auto* dst_buffer = output.buffer();

    uint32_t num_cores_x = cached_program.shared_variables.compute_with_storage_grid_size.x;
    uint32_t num_cores_y = cached_program.shared_variables.compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;

    for (uint32_t i = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        // Update reader kernel runtime args
        {
            auto& runtime_args =
                GetRuntimeArgs(cached_program.program, cached_program.shared_variables.reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        // Update writer kernel runtime args
        {
            auto& runtime_args =
                GetRuntimeArgs(cached_program.program, cached_program.shared_variables.writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::operations::data_movement::pad::program
