// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>
#include <cstdint>
#include <string>

#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/work_split.hpp"
#include "upsample3d_op.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::operations::upsample3d {

tt::tt_metal::operation::ProgramWithCallbacks upsample3d_multi_core_interleaved(
    const Tensor& input, Tensor& output, uint32_t scale_factor_d, uint32_t scale_factor_h, uint32_t scale_factor_w) {
    tt::tt_metal::Program program{};

    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    tt::tt_metal::IDevice* const device = output.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    // For 5D tensor [N, D, H, W, C] in row-major layout:
    // Total input pages = N * D * H * W (each page has C elements)
    const auto& input_shape = input.logical_shape();
    const uint32_t N = input_shape[0];
    const uint32_t D = input_shape[1];
    const uint32_t H = input_shape[2];
    const uint32_t W = input_shape[3];
    const uint32_t C = input_shape[4];

    const uint32_t total_input_pages = N * D * H * W;
    const uint32_t input_page_size = C * input.element_size();
    const uint32_t aligned_input_page_size = tt::round_up(input_page_size, tt::tt_metal::hal::get_dram_alignment());

    const uint32_t output_page_size = C * output.element_size();
    const uint32_t aligned_output_page_size = tt::round_up(output_page_size, tt::tt_metal::hal::get_dram_alignment());

    // Work distribution: each core processes multiple input pages
    const auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_group_1, work_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_input_pages);

    // Create circular buffers
    uint32_t next_cb_index = tt::CBIndex::c_0;

    // Input CB: holds input pages
    uint32_t num_pages_in_input_cb = 1;
    if (work_per_core_group_1 > 1) {
        num_pages_in_input_cb = 2;  // Double buffer for efficiency
    }

    const auto [src0_cb_index, cb_src0] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, aligned_input_page_size, num_pages_in_input_cb, input_cb_data_format);

    // Output CB: holds output pages
    const auto [output_cb_index, cb_output] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, aligned_output_page_size, 1, output_cb_data_format);

    // Create kernels with proper compile-time arguments
    std::string reader_kernel_file =
        "ttnn/cpp/ttnn/operations/pool/upsample3d/device/kernels/dataflow/reader_upsample3d_interleaved.cpp";
    std::string writer_kernel_file =
        "ttnn/cpp/ttnn/operations/pool/upsample3d/device/kernels/dataflow/writer_upsample3d_interleaved.cpp";

    // Reader compile-time args: CB index, is_dram, page_size_is_pow2, log2_page_size
    const bool src_is_dram = input.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool src_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(aligned_input_page_size);
    const uint32_t src_log2_size = src_size_is_power_of_two ? (uint32_t)log2(aligned_input_page_size) : 0;

    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)src0_cb_index, (uint32_t)src_is_dram, (uint32_t)src_size_is_power_of_two, (uint32_t)src_log2_size};

    // Writer compile-time args: CB index, is_dram, page_size_is_pow2, log2_page_size, scale factors, input dims
    const bool dst_is_dram = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool dst_size_is_power_of_two = tt::tt_metal::is_power_of_two_at_least_32(aligned_output_page_size);
    const uint32_t dst_log2_size = dst_size_is_power_of_two ? (uint32_t)log2(aligned_output_page_size) : 0;

    std::vector<uint32_t> writer_compile_time_args = {
        (uint32_t)output_cb_index,
        (uint32_t)dst_is_dram,
        (uint32_t)dst_size_is_power_of_two,
        (uint32_t)dst_log2_size,
        scale_factor_d,  // scale_d
        scale_factor_h,  // scale_h
        scale_factor_w,  // scale_w
        D,               // input_d
        H,               // input_h
        W                // input_w
    };

    // Reader kernel
    const auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program, reader_kernel_file, all_cores, tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Writer kernel
    const auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program, writer_kernel_file, all_cores, tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Set runtime arguments for cores using the same pattern as existing upsample
    std::vector<uint32_t> reader_rt_arguments = {
        0,  // input buffer address - set in callback
        0,  // number of pages to process per core
        aligned_input_page_size};

    std::vector<uint32_t> writer_rt_arguments = {
        0,  // output buffer address - set in callback
        0,  // number of pages to process per core
        aligned_output_page_size};

    for (uint32_t i = 0, pages_processed = 0; i < num_cores; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t pages_per_core = 0;
        if (core_group_1.contains(core)) {
            pages_per_core = work_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            pages_per_core = work_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        reader_rt_arguments[1] = pages_per_core;
        writer_rt_arguments[1] = pages_per_core * scale_factor_d * scale_factor_h * scale_factor_w;  // Output pages

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_arguments);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_rt_arguments);

        pages_processed += pages_per_core;
    }

    // Runtime callback to set buffer addresses
    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id, num_cores, num_cores_y](
                                              const void* operation,
                                              tt::tt_metal::Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        const auto src_buffer = input_tensors.at(0).buffer();
        const auto dst_buffer = output_tensors.at(0).buffer();

        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord core = {i / num_cores_y, i % num_cores_y};
            {
                auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
            }
            {
                auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::upsample3d
