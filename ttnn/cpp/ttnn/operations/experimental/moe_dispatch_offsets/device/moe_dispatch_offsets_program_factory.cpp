// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>

#include "moe_dispatch_offsets_program_factory.hpp"
#include "moe_dispatch_offsets_device_operation_types.hpp"

#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operation.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/math.hpp>

namespace ttnn::experimental::prim {

MoeDispatchOffsetsProgramFactory::cached_program_t MoeDispatchOffsetsProgramFactory::create(
    const MoeDispatchOffsetsParams& operation_attributes, const Tensor& input, Tensor& tensor_return_value) {
    tt::tt_metal::Program program{};

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::UINT32);
    uint32_t input_element_size = input.element_size();

    const auto& logical_shape = input.logical_shape();
    uint32_t W = logical_shape[-1];
    uint32_t H = logical_shape[-2];
    uint32_t n_routed_experts = operation_attributes.n_routed_experts;

    auto* src_buffer = input.buffer();
    auto* dst_buffer = tensor_return_value.buffer();
    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    uint32_t input_page_size = src_buffer->aligned_page_size();
    uint32_t output_page_size = dst_buffer->aligned_page_size();
    uint32_t histogram_page_size = tt::round_up(n_routed_experts * sizeof(uint32_t), 32);

    const tt::tt_metal::IDevice* device = tensor_return_value.device();
    auto core_grid = device->compute_with_storage_grid_size();

    auto [num_total_cores, all_cores, core_group_0, core_group_1, rows_per_core0, rows_per_core1] =
        tt::tt_metal::split_work_to_cores(core_grid, H);

    uint32_t num_cores0 = core_group_0.num_cores();
    uint32_t num_cores1 = core_group_1.num_cores();

    // --- Circular Buffers ---

    // CB 0: input pages (per-core sized for the group's rows)
    uint32_t cb_in0_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_in0_config0 =
        tt::tt_metal::CircularBufferConfig(rows_per_core0 * input_page_size, {{cb_in0_index, input_cb_data_format}})
            .set_page_size(cb_in0_index, input_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_group_0, cb_in0_config0);

    if (num_cores1 > 0) {
        tt::tt_metal::CircularBufferConfig cb_in0_config1 =
            tt::tt_metal::CircularBufferConfig(rows_per_core1 * input_page_size, {{cb_in0_index, input_cb_data_format}})
                .set_page_size(cb_in0_index, input_page_size);
        tt::tt_metal::CreateCircularBuffer(program, core_group_1, cb_in0_config1);
    }

    // CB 1: partial histograms from all cores (only reduce core uses all slots, but all cores need the same layout)
    uint32_t cb_partial_index = tt::CBIndex::c_1;
    uint32_t partial_cb_total_size = num_total_cores * histogram_page_size;
    tt::tt_metal::CircularBufferConfig cb_partial_config =
        tt::tt_metal::CircularBufferConfig(partial_cb_total_size, {{cb_partial_index, output_cb_data_format}})
            .set_page_size(cb_partial_index, partial_cb_total_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_partial_config);

    // CB 2: final output (used by reduce core to write to DRAM)
    uint32_t cb_out_index = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_out_config =
        tt::tt_metal::CircularBufferConfig(output_page_size, {{cb_out_index, output_cb_data_format}})
            .set_page_size(cb_out_index, output_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);

    // --- Semaphore ---
    auto done_sem_idx = tt::tt_metal::CreateSemaphore(program, all_cores, 0);

    // --- Physical coordinates of reduce core ---
    constexpr uint32_t reduce_core_id = 0;
    auto all_cores_vec = tt::tt_metal::corerange_to_cores(all_cores, num_total_cores, true);
    auto reduce_core_phys = device->worker_core_from_logical_core(all_cores_vec[reduce_core_id]);

    // --- Compile-time args ---
    std::vector<uint32_t> compile_time_args = {
        cb_in0_index,
        cb_partial_index,
        cb_out_index,
        (uint32_t)src_is_dram,
        (uint32_t)dst_is_dram,
        input_page_size,
        output_page_size,
        histogram_page_size,
        W,
        n_routed_experts,
        input_element_size,
        num_total_cores,
        reduce_core_id,
        (uint32_t)reduce_core_phys.x,
        (uint32_t)reduce_core_phys.y,
        done_sem_idx,
    };
    tt::tt_metal::TensorAccessorArgs(src_buffer).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(compile_time_args);

    // --- Create kernels ---
    std::map<std::string, std::string> kernel_defines;
    tt::tt_metal::KernelHandle kernel_id0 = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/moe_dispatch_offsets/device/kernels/"
        "reader_moe_dispatch_offsets_interleaved.cpp",
        core_group_0,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = compile_time_args});

    tt::tt_metal::KernelHandle kernel_id1 = 0;
    if (num_cores1 > 0) {
        kernel_id1 = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/moe_dispatch_offsets/device/kernels/"
            "reader_moe_dispatch_offsets_interleaved.cpp",
            core_group_1,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt::tt_metal::NOC::RISCV_1_default,
                .compile_args = compile_time_args});
    }

    // --- Per-core runtime args ---
    auto cores0_vec = tt::tt_metal::corerange_to_cores(core_group_0, num_cores0, true);
    auto cores1_vec = tt::tt_metal::corerange_to_cores(core_group_1, num_cores1, true);

    uint32_t h_offset = 0;
    for (uint32_t i = 0; i < num_cores0; ++i) {
        uint32_t h_count = (i == num_cores0 - 1 && num_cores1 == 0) ? (H - h_offset) : rows_per_core0;
        tt::tt_metal::SetRuntimeArgs(
            program, kernel_id0, cores0_vec[i], {src_buffer->address(), dst_buffer->address(), i, h_offset, h_count});
        h_offset += h_count;
    }

    for (uint32_t i = 0; i < num_cores1; ++i) {
        uint32_t core_id = num_cores0 + i;
        uint32_t h_count = (i == num_cores1 - 1) ? (H - h_offset) : rows_per_core1;
        tt::tt_metal::SetRuntimeArgs(
            program,
            kernel_id1,
            cores1_vec[i],
            {src_buffer->address(), dst_buffer->address(), core_id, h_offset, h_count});
        h_offset += h_count;
    }

    return cached_program_t{std::move(program), {kernel_id0, kernel_id1, cores0_vec, cores1_vec}};
}

void MoeDispatchOffsetsProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const MoeDispatchOffsetsParams&,
    const Tensor& input,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    auto& kernel_id0 = cached_program.shared_variables.kernel_id0;
    auto& kernel_id1 = cached_program.shared_variables.kernel_id1;
    auto& cores0 = cached_program.shared_variables.cores0;
    auto& cores1 = cached_program.shared_variables.cores1;

    for (const auto& core : cores0) {
        auto& rt = GetRuntimeArgs(program, kernel_id0, core);
        rt[0] = input.buffer()->address();
        rt[1] = tensor_return_value.buffer()->address();
    }
    for (const auto& core : cores1) {
        auto& rt = GetRuntimeArgs(program, kernel_id1, core);
        rt[0] = input.buffer()->address();
        rt[1] = tensor_return_value.buffer()->address();
    }
}

}  // namespace ttnn::experimental::prim
