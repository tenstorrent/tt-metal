// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "masked_bincount_program_factory.hpp"
#include "masked_bincount_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/operation.hpp"

namespace ttnn::experimental::prim {

MaskedBincountProgramFactory::cached_program_t MaskedBincountProgramFactory::create(
    const MaskedBincountParams& operation_attributes,
    const MaskedBincountInputs& tensor_args,
    Tensor& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const auto& expert_mask = tensor_args.expert_mask;

    tt::tt_metal::Program program{};

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::UINT32);
    uint32_t n_routed_experts = operation_attributes.n_routed_experts;

    const auto& shard_spec = input.shard_spec().value();
    auto all_cores = shard_spec.grid;
    uint32_t num_cores = all_cores.num_cores();
    uint32_t shard_height = shard_spec.shape[0];
    uint32_t W = shard_spec.shape[1];

    uint32_t h_brisc = shard_height / 2;
    uint32_t h_ncrisc = shard_height - h_brisc;

    auto* src_buffer = input.buffer();
    auto* dst_buffer = tensor_return_value.buffer();
    auto* mask_buffer = expert_mask.buffer();

    uint32_t input_page_size = src_buffer->aligned_page_size();
    uint32_t output_page_size = dst_buffer->aligned_page_size();
    uint32_t mask_page_size = mask_buffer->aligned_page_size();

    auto all_cores_vec = tt::tt_metal::corerange_to_cores(all_cores, num_cores, true);
    CoreCoord collector_core = all_cores_vec[0];
    const tt::tt_metal::IDevice* device = input.device();
    auto collector_noc = device->worker_core_from_logical_core(collector_core);

    // --- Circular Buffers ---

    // CB 0: BRISC input pages (per-shard)
    uint32_t cb_in_brisc = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_in_brisc_config =
        tt::tt_metal::CircularBufferConfig(h_brisc * input_page_size, {{cb_in_brisc, input_cb_data_format}})
            .set_page_size(cb_in_brisc, input_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_in_brisc_config);

    // CB 1: local output histogram
    uint32_t cb_out_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_out_config =
        tt::tt_metal::CircularBufferConfig(output_page_size, {{cb_out_index, output_cb_data_format}})
            .set_page_size(cb_out_index, output_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);

    // CB 2: NCRISC input pages (per-shard)
    uint32_t cb_in_ncrisc = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_in_ncrisc_config =
        tt::tt_metal::CircularBufferConfig(h_ncrisc * input_page_size, {{cb_in_ncrisc, input_cb_data_format}})
            .set_page_size(cb_in_ncrisc, input_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_in_ncrisc_config);

    // CB 3: gather temp buffer (collector reads remote histograms here)
    uint32_t cb_gather_tmp = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig cb_gather_config =
        tt::tt_metal::CircularBufferConfig(output_page_size, {{cb_gather_tmp, output_cb_data_format}})
            .set_page_size(cb_gather_tmp, output_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_gather_config);

    // CB 4: expert mask (UINT32, one value per expert)
    uint32_t cb_mask = tt::CBIndex::c_4;
    tt::tt_metal::CircularBufferConfig cb_mask_config =
        tt::tt_metal::CircularBufferConfig(mask_page_size, {{cb_mask, output_cb_data_format}})
            .set_page_size(cb_mask, mask_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_mask_config);

    // --- Semaphores ---
    auto init_sem_idx = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
    auto done_sem_idx = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
    auto gather_sem_idx = tt::tt_metal::CreateSemaphore(program, all_cores, 0);

    // --- TensorAccessor args (shared by both kernels) ---
    std::vector<uint32_t> accessor_args;
    tt::tt_metal::TensorAccessorArgs(src_buffer).append_to(accessor_args);
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(accessor_args);
    tt::tt_metal::TensorAccessorArgs(mask_buffer).append_to(accessor_args);

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
        gather_sem_idx,
        cb_gather_tmp,
        (uint32_t)collector_noc.x,
        (uint32_t)collector_noc.y,
        num_cores,
        cb_mask,
        mask_page_size,
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
        gather_sem_idx,
        cb_gather_tmp,
        (uint32_t)collector_noc.x,
        (uint32_t)collector_noc.y,
        num_cores,
        cb_mask,
        mask_page_size,
    };
    ct_args_ncrisc.insert(ct_args_ncrisc.end(), accessor_args.begin(), accessor_args.end());

    // --- Create BRISC kernel (RISCV_0, NOC 0) ---
    tt::tt_metal::KernelHandle kernel_id_brisc = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/masked_bincount/device/kernels/reader_masked_bincount.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = ct_args_brisc});

    // --- Create NCRISC kernel (RISCV_1, NOC 1) ---
    tt::tt_metal::KernelHandle kernel_id_ncrisc = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/masked_bincount/device/kernels/reader_masked_bincount.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = ct_args_ncrisc});

    // --- Per-core runtime args (tree reduction) ---
    for (uint32_t i = 0; i < all_cores_vec.size(); i++) {
        uint32_t page_offset = i * shard_height;

        // Tree structure: core i receives from children at consecutive levels,
        // then signals its parent. Children at level L: core i + 2^L.
        std::vector<uint32_t> children_noc;
        uint32_t num_receive = 0;
        for (uint32_t L = 0; (1u << L) < num_cores; L++) {
            uint32_t stride = 1u << L;
            uint32_t group = stride << 1;
            if ((i % group) == 0 && (i + stride) < num_cores) {
                auto child_noc = device->worker_core_from_logical_core(all_cores_vec[i + stride]);
                children_noc.push_back(child_noc.x);
                children_noc.push_back(child_noc.y);
                num_receive++;
            } else {
                break;
            }
        }

        // Parent: the core that reads from us. For i > 0, clear lowest set bit.
        uint32_t parent_noc_x = 0xFFFFFFFF;
        uint32_t parent_noc_y = 0xFFFFFFFF;
        if (i > 0) {
            uint32_t lowest_bit = i & (~i + 1);
            uint32_t parent_idx = i ^ lowest_bit;
            auto p_noc = device->worker_core_from_logical_core(all_cores_vec[parent_idx]);
            parent_noc_x = p_noc.x;
            parent_noc_y = p_noc.y;
        }

        // rt_args: [src, dst, mask, page_offset, num_receive, parent_noc_x, parent_noc_y, child0_x, child0_y, ...]
        std::vector<uint32_t> rt_brisc = {
            src_buffer->address(),
            dst_buffer->address(),
            mask_buffer->address(),
            page_offset,
            num_receive,
            parent_noc_x,
            parent_noc_y};
        rt_brisc.insert(rt_brisc.end(), children_noc.begin(), children_noc.end());

        tt::tt_metal::SetRuntimeArgs(program, kernel_id_brisc, all_cores_vec[i], rt_brisc);
        tt::tt_metal::SetRuntimeArgs(
            program,
            kernel_id_ncrisc,
            all_cores_vec[i],
            {src_buffer->address(), dst_buffer->address(), mask_buffer->address(), page_offset + h_brisc, 0u});
    }

    return cached_program_t{
        std::move(program), {kernel_id_brisc, kernel_id_ncrisc, all_cores_vec, collector_core, num_cores}};
}

void MaskedBincountProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const MaskedBincountParams&,
    const MaskedBincountInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    auto& all_cores_vec = cached_program.shared_variables.all_cores_vec;
    auto kernel_id_brisc = cached_program.shared_variables.kernel_id_brisc;
    auto kernel_id_ncrisc = cached_program.shared_variables.kernel_id_ncrisc;

    for (const auto& core : all_cores_vec) {
        auto& rt_brisc = GetRuntimeArgs(program, kernel_id_brisc, core);
        rt_brisc[0] = tensor_args.input_tensor.buffer()->address();
        rt_brisc[1] = tensor_return_value.buffer()->address();
        rt_brisc[2] = tensor_args.expert_mask.buffer()->address();

        auto& rt_ncrisc = GetRuntimeArgs(program, kernel_id_ncrisc, core);
        rt_ncrisc[0] = tensor_args.input_tensor.buffer()->address();
        rt_ncrisc[1] = tensor_return_value.buffer()->address();
        rt_ncrisc[2] = tensor_args.expert_mask.buffer()->address();
    }
}

}  // namespace ttnn::experimental::prim
