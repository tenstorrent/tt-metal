// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include <tt-metalium/constants.hpp>
#include "full_like_device_operation.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/tensor/types.hpp"
#include "full_like_program_factory_interleaved.hpp"
#include "full_like_program_factory_common.hpp"

namespace ttnn::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal::detail;

FullLikeInterleavedProgramFactory::cached_program_t FullLikeInterleavedProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto input = tensor_args.input;
    auto fill_value = operation_attributes.fill_value;
    DataType dtype{operation_attributes.dtype};
    IDevice* device = input.device();
    MemoryConfig memory_config{operation_attributes.memory_config};
    const auto& layout = operation_attributes.layout;

    auto num_pages = (layout == Layout::TILE) ? (input.physical_volume() / TILE_HW)
                                              : input.physical_volume() / input.logical_shape()[-1];

    Program program{};

    auto data_format = datatype_to_dataformat_converter(dtype);
    uint32_t single_page_size =
        (layout == Layout::TILE) ? tt::tile_size(data_format) : input.logical_shape()[-1] * tt::datum_size(data_format);

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
        tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_pages);

    constexpr CBIndex cb_fill_value_id = CBIndex::c_24;

    auto cb_value_config = tt::tt_metal::CircularBufferConfig(single_page_size, {{cb_fill_value_id, data_format}})
                               .set_page_size(cb_fill_value_id, single_page_size);
    CreateCircularBuffer(program, all_cores, cb_value_config);
    std::map<std::string, std::string> writer_defines;

    switch (dtype) {
        case DataType::BFLOAT16: writer_defines["OUTPUT_DTYPE_BFLOAT16"] = "1"; break;
        case DataType::INT32: writer_defines["OUTPUT_DTYPE_INT32"] = "1"; break;
        case DataType::FLOAT32: writer_defines["OUTPUT_DTYPE_FLOAT32"] = "1"; break;
        default: break;
    }

    if (std::holds_alternative<int>(fill_value)) {
        u.u32 = std::get<int>(fill_value);
    } else if (std::holds_alternative<float>(fill_value)) {
        auto float_fill_value = std::get<float>(fill_value);
        if (dtype == DataType::BFLOAT16) {
            u.u32 = get_bfloat16_rounded(float_fill_value);
        } else {
            u.f32 = float_fill_value;
        }
    }

    uint32_t elems_per_page = (layout == Layout::TILE) ? TILE_HW : input.logical_shape()[-1];
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)cb_fill_value_id, elems_per_page, single_page_size};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    auto writer_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/full/device/kernels/writer_full.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_defines));

    uint32_t pages_offset = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord core(i / num_cores_y, i % num_cores_y);

        uint32_t num_pages_per_core = 0;
        if (core_group_1.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }
        SetRuntimeArgs(program, writer_id, core, {output.buffer()->address(), u.u32, num_pages_per_core, pages_offset});

        pages_offset += num_pages_per_core;
    }

    return {std::move(program), {writer_id, num_cores, num_cores_y}};
}

void FullLikeInterleavedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    auto output_buffer_address = output.buffer()->address();
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = output_buffer_address;
        }
    }
}

}  // namespace ttnn::prim
