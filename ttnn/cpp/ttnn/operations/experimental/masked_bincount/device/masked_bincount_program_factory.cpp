// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "masked_bincount_program_factory.hpp"
#include "masked_bincount_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/operation.hpp"

namespace ttnn::experimental::prim {

MaskedBincountProgramFactory::cached_program_t MaskedBincountProgramFactory::create(
    const MaskedBincountParams& operation_attributes, const Tensor& input, Tensor& tensor_return_value) {
    tt::tt_metal::Program program{};

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::UINT32);
    uint32_t n_routed_experts = operation_attributes.n_routed_experts;

    const auto& logical_shape = input.logical_shape();
    uint32_t H = logical_shape[-2];
    uint32_t W = logical_shape[-1];

    uint32_t h_brisc = H / 2;
    uint32_t h_ncrisc = H - h_brisc;

    auto* src_buffer = input.buffer();
    auto* dst_buffer = tensor_return_value.buffer();

    uint32_t input_page_size = src_buffer->aligned_page_size();
    uint32_t output_page_size = dst_buffer->aligned_page_size();

    CoreCoord core = {0, 0};
    CoreRange core_range(core, core);

    // --- Circular Buffers ---

    // CB 0: BRISC input pages
    uint32_t cb_in_brisc = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_in_brisc_config =
        tt::tt_metal::CircularBufferConfig(h_brisc * input_page_size, {{cb_in_brisc, input_cb_data_format}})
            .set_page_size(cb_in_brisc, input_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, cb_in_brisc_config);

    // CB 1: shared output histogram
    uint32_t cb_out_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_out_config =
        tt::tt_metal::CircularBufferConfig(output_page_size, {{cb_out_index, output_cb_data_format}})
            .set_page_size(cb_out_index, output_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, cb_out_config);

    // CB 2: NCRISC input pages
    uint32_t cb_in_ncrisc = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_in_ncrisc_config =
        tt::tt_metal::CircularBufferConfig(h_ncrisc * input_page_size, {{cb_in_ncrisc, input_cb_data_format}})
            .set_page_size(cb_in_ncrisc, input_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range, cb_in_ncrisc_config);

    // --- Semaphores ---
    auto init_sem_idx = tt::tt_metal::CreateSemaphore(program, core_range, 0);
    auto done_sem_idx = tt::tt_metal::CreateSemaphore(program, core_range, 0);

    // --- TensorAccessor args (shared by both kernels) ---
    std::vector<uint32_t> accessor_args;
    tt::tt_metal::TensorAccessorArgs(src_buffer).append_to(accessor_args);
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(accessor_args);

    // --- BRISC compile-time args ---
    std::vector<uint32_t> ct_args_brisc = {
        cb_in_brisc,
        cb_out_index,
        input_page_size,
        output_page_size,
        h_brisc,
        W,
        n_routed_experts,
        1,  // is_initializer
        init_sem_idx,
        done_sem_idx,
    };
    ct_args_brisc.insert(ct_args_brisc.end(), accessor_args.begin(), accessor_args.end());

    // --- NCRISC compile-time args ---
    std::vector<uint32_t> ct_args_ncrisc = {
        cb_in_ncrisc,
        cb_out_index,
        input_page_size,
        output_page_size,
        h_ncrisc,
        W,
        n_routed_experts,
        0,  // is_initializer
        init_sem_idx,
        done_sem_idx,
    };
    ct_args_ncrisc.insert(ct_args_ncrisc.end(), accessor_args.begin(), accessor_args.end());

    // --- Create BRISC kernel (RISCV_0, NOC 0) ---
    tt::tt_metal::KernelHandle kernel_id_brisc = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/masked_bincount/device/kernels/reader_masked_bincount.cpp",
        core_range,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = ct_args_brisc});

    // --- Create NCRISC kernel (RISCV_1, NOC 1) ---
    tt::tt_metal::KernelHandle kernel_id_ncrisc = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/masked_bincount/device/kernels/reader_masked_bincount.cpp",
        core_range,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = ct_args_ncrisc});

    // --- Runtime args: src_addr, dst_addr, h_start ---
    tt::tt_metal::SetRuntimeArgs(program, kernel_id_brisc, core, {src_buffer->address(), dst_buffer->address(), 0u});
    tt::tt_metal::SetRuntimeArgs(
        program, kernel_id_ncrisc, core, {src_buffer->address(), dst_buffer->address(), h_brisc});

    return cached_program_t{std::move(program), {kernel_id_brisc, kernel_id_ncrisc, core}};
}

void MaskedBincountProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program, const MaskedBincountParams&, const Tensor& input, Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    auto& core = cached_program.shared_variables.core;

    auto& rt_brisc = GetRuntimeArgs(program, cached_program.shared_variables.kernel_id_brisc, core);
    rt_brisc[0] = input.buffer()->address();
    rt_brisc[1] = tensor_return_value.buffer()->address();

    auto& rt_ncrisc = GetRuntimeArgs(program, cached_program.shared_variables.kernel_id_ncrisc, core);
    rt_ncrisc[0] = input.buffer()->address();
    rt_ncrisc[1] = tensor_return_value.buffer()->address();
}

}  // namespace ttnn::experimental::prim
