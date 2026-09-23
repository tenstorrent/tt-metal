// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "masked_bincount_program_factory.hpp"
#include "masked_bincount_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::experimental::prim {

tt::tt_metal::ProgramDescriptor MaskedBincountProgramFactory::create_descriptor(
    const MaskedBincountParams& operation_attributes,
    const MaskedBincountInputs& tensor_args,
    Tensor& tensor_return_value) {
    const auto& input = tensor_args.input_tensor;
    const auto& expert_mask = tensor_args.expert_mask;

    tt::tt_metal::ProgramDescriptor desc;

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::UINT32);
    uint32_t n_routed_experts = operation_attributes.n_routed_experts;

    // Input is TILE + interleaved (no shard spec). Derive the work split from the shape: keep the
    // fixed 8x8 (64-core) grid and the binary-tree reduction, giving each core a contiguous range of
    // token rows. Each core reads the TILE pages covering its rows from interleaved memory and untiles
    // in-kernel. tile_h (32) is the tile height; the token count is tile-aligned by construction.
    const uint32_t tile_h = input.tensor_spec().page_config().get_tile().get_height();
    const uint32_t tokens = input.padded_shape()[0];

    CoreRangeSet all_cores(CoreRange(CoreCoord(0, 0), CoreCoord(7, 7)));
    uint32_t num_cores = all_cores.num_cores();
    TT_FATAL(tokens % num_cores == 0, "Token count ({}) must be divisible by the {}-core grid", tokens, num_cores);
    uint32_t shard_height = tokens / num_cores;  // rows per core

    uint32_t h_brisc = shard_height / 2;
    uint32_t h_ncrisc = shard_height - h_brisc;

    // Max TILE pages a RISC's row range can span (misaligned ranges straddle one extra tile).
    uint32_t max_tiles_brisc = (h_brisc + tile_h - 1) / tile_h + 1;
    uint32_t max_tiles_ncrisc = (h_ncrisc + tile_h - 1) / tile_h + 1;

    auto* src_buffer = input.buffer();
    auto* dst_buffer = tensor_return_value.buffer();
    auto* mask_buffer = expert_mask.buffer();
    TT_FATAL(src_buffer != nullptr, "input buffer must be allocated on device");
    TT_FATAL(dst_buffer != nullptr, "output buffer must be allocated on device");
    TT_FATAL(mask_buffer != nullptr, "expert_mask buffer must be allocated on device");

    uint32_t input_page_size = src_buffer->aligned_page_size();  // one TILE (32x32 uint16)
    uint32_t output_page_size = dst_buffer->aligned_page_size();
    uint32_t mask_page_size = mask_buffer->aligned_page_size();

    auto all_cores_vec = tt::tt_metal::corerange_to_cores(all_cores, num_cores, true);
    CoreCoord collector_core = all_cores_vec[0];
    const tt::tt_metal::IDevice* device = input.device();
    auto collector_noc = device->worker_core_from_logical_core(collector_core);

    auto add_cb = [&](uint32_t cb_idx, uint32_t total_size, uint32_t page_size, tt::DataFormat data_format) {
        desc.cbs.push_back(tt::tt_metal::CBDescriptor{
            .total_size = total_size,
            .core_ranges = all_cores,
            .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_idx),
                .data_format = data_format,
                .page_size = page_size,
            }}},
        });
    };

    // --- Circular Buffers ---

    // CB 0: BRISC input pages (per-shard)
    uint32_t cb_in_brisc = tt::CBIndex::c_0;
    add_cb(cb_in_brisc, max_tiles_brisc * input_page_size, input_page_size, input_cb_data_format);

    // CB 1: local output histogram
    uint32_t cb_out_index = tt::CBIndex::c_1;
    add_cb(cb_out_index, output_page_size, output_page_size, output_cb_data_format);

    // CB 2: NCRISC input pages (per-shard)
    uint32_t cb_in_ncrisc = tt::CBIndex::c_2;
    add_cb(cb_in_ncrisc, max_tiles_ncrisc * input_page_size, input_page_size, input_cb_data_format);

    // CB 3: gather temp buffer (collector reads remote histograms here)
    uint32_t cb_gather_tmp = tt::CBIndex::c_3;
    add_cb(cb_gather_tmp, output_page_size, output_page_size, output_cb_data_format);

    // CB 4: expert dispatch table (INT32, one value per expert; negative = absent, non-negative = present)
    tt::DataFormat mask_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::INT32);
    uint32_t cb_mask = tt::CBIndex::c_4;
    add_cb(cb_mask, mask_page_size, mask_page_size, mask_cb_data_format);

    // --- Semaphores ---
    auto add_semaphore = [&]() {
        const uint32_t semaphore_id = static_cast<uint32_t>(desc.semaphores.size());
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = semaphore_id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = all_cores,
            .initial_value = 0,
        });
        return semaphore_id;
    };
    auto init_sem_idx = add_semaphore();
    auto done_sem_idx = add_semaphore();
    auto gather_sem_idx = add_semaphore();

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
        tile_h,
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
        tile_h,
    };
    ct_args_ncrisc.insert(ct_args_ncrisc.end(), accessor_args.begin(), accessor_args.end());

    // --- Create BRISC kernel (RISCV_0, NOC 0) ---
    tt::tt_metal::KernelDescriptor brisc_kernel_desc;
    brisc_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/masked_bincount/device/kernels/"
        "reader_masked_bincount.cpp";
    brisc_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    brisc_kernel_desc.core_ranges = all_cores;
    brisc_kernel_desc.compile_time_args = std::move(ct_args_brisc);
    brisc_kernel_desc.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::NOC::RISCV_0_default,
    };

    // --- Create NCRISC kernel (RISCV_1, NOC 1) ---
    tt::tt_metal::KernelDescriptor ncrisc_kernel_desc;
    ncrisc_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/masked_bincount/device/kernels/"
        "reader_masked_bincount.cpp";
    ncrisc_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    ncrisc_kernel_desc.core_ranges = all_cores;
    ncrisc_kernel_desc.compile_time_args = std::move(ct_args_ncrisc);
    ncrisc_kernel_desc.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = tt::tt_metal::NOC::RISCV_1_default,
    };

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
        tt::tt_metal::KernelDescriptor::RTArgList rt_brisc;
        rt_brisc.push_back(src_buffer);
        rt_brisc.push_back(dst_buffer);
        rt_brisc.push_back(mask_buffer);
        rt_brisc.push_back(page_offset);
        rt_brisc.push_back(num_receive);
        rt_brisc.push_back(parent_noc_x);
        rt_brisc.push_back(parent_noc_y);
        rt_brisc.append(children_noc);
        brisc_kernel_desc.emplace_runtime_args(all_cores_vec[i], rt_brisc);
        ncrisc_kernel_desc.emplace_runtime_args(
            all_cores_vec[i], {src_buffer, dst_buffer, mask_buffer, page_offset + h_brisc, 0u});
    }

    desc.kernels.push_back(std::move(brisc_kernel_desc));
    desc.kernels.push_back(std::move(ncrisc_kernel_desc));
    return desc;
}

}  // namespace ttnn::experimental::prim
