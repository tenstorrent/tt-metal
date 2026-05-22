// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "masked_bincount_program_factory.hpp"
#include "masked_bincount_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/operation.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

tt::tt_metal::ProgramDescriptor MaskedBincountProgramFactory::create_descriptor(
    const MaskedBincountParams& operation_attributes,
    const MaskedBincountInputs& tensor_args,
    Tensor& tensor_return_value) {
    ProgramDescriptor desc;

    const auto& input = tensor_args.input_tensor;
    const auto& expert_mask = tensor_args.expert_mask;

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::UINT32);
    uint32_t n_routed_experts = operation_attributes.n_routed_experts;

    const auto& shard_spec = input.shard_spec().value();
    auto all_cores = shard_spec.grid;
    uint32_t num_cores = all_cores.num_cores();
    uint32_t shard_height = shard_spec.shape[0];

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
    desc.cbs.push_back(CBDescriptor{
        .total_size = h_brisc * input_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_in_brisc),
            .data_format = input_cb_data_format,
            .page_size = input_page_size,
        }}},
    });

    // CB 1: local output histogram
    uint32_t cb_out_index = tt::CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_out_index),
            .data_format = output_cb_data_format,
            .page_size = output_page_size,
        }}},
    });

    // CB 2: NCRISC input pages (per-shard)
    uint32_t cb_in_ncrisc = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = h_ncrisc * input_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_in_ncrisc),
            .data_format = input_cb_data_format,
            .page_size = input_page_size,
        }}},
    });

    // CB 3: gather temp buffer (collector reads remote histograms here)
    uint32_t cb_gather_tmp = tt::CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = output_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_gather_tmp),
            .data_format = output_cb_data_format,
            .page_size = output_page_size,
        }}},
    });

    // CB 4: expert dispatch table (INT32, one value per expert; negative = absent, non-negative = present)
    tt::DataFormat mask_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::INT32);
    uint32_t cb_mask = tt::CBIndex::c_4;
    desc.cbs.push_back(CBDescriptor{
        .total_size = mask_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_mask),
            .data_format = mask_cb_data_format,
            .page_size = mask_page_size,
        }}},
    });

    // --- Semaphores ---
    // Semaphore IDs are 0-based, in declaration order.
    const uint32_t init_sem_idx = 0;
    const uint32_t done_sem_idx = 1;
    const uint32_t gather_sem_idx = 2;
    desc.semaphores.push_back(SemaphoreDescriptor{.core_ranges = all_cores, .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{.core_ranges = all_cores, .initial_value = 0});
    desc.semaphores.push_back(SemaphoreDescriptor{.core_ranges = all_cores, .initial_value = 0});

    // --- TensorAccessor args (shared by both kernels) ---
    std::vector<uint32_t> accessor_args;
    tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(accessor_args);
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(accessor_args);
    tt::tt_metal::TensorAccessorArgs(*mask_buffer).append_to(accessor_args);

    // --- BRISC compile-time args ---
    std::vector<uint32_t> ct_args_brisc = {
        cb_in_brisc,
        cb_out_index,
        input_page_size,
        output_page_size,
        h_brisc,
        operation_attributes.num_experts_per_token,
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
        operation_attributes.num_experts_per_token,
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

    // --- BRISC kernel (RISCV_0, NOC 0) ---
    KernelDescriptor brisc_desc;
    brisc_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/masked_bincount/device/kernels/"
        "reader_masked_bincount.cpp";
    brisc_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    brisc_desc.core_ranges = all_cores;
    brisc_desc.compile_time_args = std::move(ct_args_brisc);
    brisc_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
    };

    // --- NCRISC kernel (RISCV_1, NOC 1) ---
    KernelDescriptor ncrisc_desc;
    ncrisc_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/masked_bincount/device/kernels/"
        "reader_masked_bincount.cpp";
    ncrisc_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    ncrisc_desc.core_ranges = all_cores;
    ncrisc_desc.compile_time_args = std::move(ct_args_ncrisc);
    ncrisc_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
    };

    // --- Per-core runtime args (tree reduction) ---
    brisc_desc.runtime_args.reserve(all_cores_vec.size());
    ncrisc_desc.runtime_args.reserve(all_cores_vec.size());
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

        brisc_desc.runtime_args.emplace_back(all_cores_vec[i], std::move(rt_brisc));
        ncrisc_desc.runtime_args.emplace_back(
            all_cores_vec[i],
            std::vector<uint32_t>{
                src_buffer->address(), dst_buffer->address(), mask_buffer->address(), page_offset + h_brisc, 0u});
    }

    desc.kernels.push_back(std::move(brisc_desc));
    desc.kernels.push_back(std::move(ncrisc_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
