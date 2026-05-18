// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dram_prefetcher_dram_core_program_factory.hpp"

#include <algorithm>
#include <cstdint>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace ttnn::prim {

using namespace tt::tt_metal;

namespace {
constexpr uint32_t kRemoteCBId = 31;

inline uint32_t align_up(uint32_t a, uint32_t align) { return (a + align - 1) & ~(align - 1); }
}  // namespace

DramPrefetcherDramCoreProgramFactory::cached_program_t DramPrefetcherDramCoreProgramFactory::create(
    const DramPrefetcherParams& args, const DramPrefetcherInputs& tensor_args, Tensor& /*output_tensor*/) {
    TT_FATAL(args.run_on_dram_cores, "DramPrefetcherDramCoreProgramFactory requires run_on_dram_cores=true");
    TT_FATAL(args.dram_sender_global_cb.has_value(), "dram_sender_global_cb must be provided");

    const auto& gcb = *args.dram_sender_global_cb;
    const auto& input_tensors = tensor_args.input_tensors;
    TT_FATAL(input_tensors.size() >= 2, "Need at least one data tensor + the tensor_addrs tensor");

    // The last input tensor is the addrs tensor (matches the existing op contract). For this
    // prototype the addrs tensor is unused on the DRAM-core path; the kernel reads tensor base
    // addresses via runtime args instead.
    std::vector<const Tensor*> data_tensors;
    data_tensors.reserve(input_tensors.size() - 1);
    for (size_t i = 0; i + 1 < input_tensors.size(); ++i) {
        data_tensors.push_back(&input_tensors[i]);
    }
    const uint32_t num_tensors = static_cast<uint32_t>(data_tensors.size());

    const auto& sender_receiver_core_mapping = gcb.sender_receiver_core_mapping();
    const uint32_t num_senders = static_cast<uint32_t>(sender_receiver_core_mapping.size());
    TT_FATAL(num_senders > 0, "DramSenderGlobalCircularBuffer must have at least one sender");

    // Establish the receiver count by inspecting the first sender; all senders are expected to
    // have the same receiver count for the prototype.
    uint32_t num_receivers = sender_receiver_core_mapping.front().second.num_cores();
    for (const auto& [_, recv] : sender_receiver_core_mapping) {
        TT_FATAL(
            recv.num_cores() == num_receivers,
            "All senders must have the same receiver count for the DRAM-core prototype");
    }

    // Block sizing: for the prototype, treat each tensor as one block of size shard_bytes /
    // num_receivers. Per-receiver stripe is what the DRISC kernel pushes through remote_cb. The
    // worker-core path's block math is more elaborate; we'll match it in a follow-up.
    constexpr uint32_t kNumBlocks = 1;

    Program program{};

    // L1 layout for the DRISC kernel. The GCB has already agreed where pages_sent lives in DRISC
    // L1 (it placed each receiver's `aligned_pages_sent_addr` to point there); build the rest of
    // the layout immediately above that.
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    const uint32_t pages_sent_addr = static_cast<uint32_t>(gcb.pages_sent_drisc_l1_base());
    uint32_t cursor = pages_sent_addr + 2 * l1_alignment * num_receivers;
    cursor = align_up(cursor, l1_alignment);
    const uint32_t noc_xy_addr = cursor;
    cursor += 2 * sizeof(uint32_t) * num_receivers;
    cursor = align_up(cursor, l1_alignment);
    const uint32_t config_addr = cursor;
    cursor += 16;
    cursor = align_up(cursor, l1_alignment);
    const uint32_t stage_a_addr = cursor;
    // Stage buffer size: largest shard per-receiver bytes across tensors.
    uint32_t max_block_size = 0;
    for (const auto* t : data_tensors) {
        const auto shard_shape = t->buffer()->shard_spec().shape();
        const uint32_t bytes_per_tile = tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(t->dtype()));
        const uint32_t num_tiles_in_shard =
            (shard_shape[0] / tt::constants::TILE_HEIGHT) * (shard_shape[1] / tt::constants::TILE_WIDTH);
        const uint32_t shard_bytes = num_tiles_in_shard * bytes_per_tile;
        TT_FATAL(
            shard_bytes % num_receivers == 0,
            "shard bytes ({}) must divide num_receivers ({})",
            shard_bytes,
            num_receivers);
        max_block_size = std::max(max_block_size, shard_bytes / num_receivers);
    }
    max_block_size = align_up(max_block_size, l1_alignment);
    cursor += max_block_size;
    cursor = align_up(cursor, l1_alignment);
    const uint32_t stage_b_addr = cursor;

    // Build one kernel per sender DRAM core.
    for (uint32_t s = 0; s < num_senders; ++s) {
        const auto& [sender_logical, _receivers] = sender_receiver_core_mapping[s];
        std::vector<uint32_t> compile_args = {
            args.num_layers,
            num_tensors,
            kNumBlocks,
            num_receivers,
            max_block_size,
            stage_a_addr,
            stage_b_addr,
            kRemoteCBId,
            pages_sent_addr,
            noc_xy_addr,
            config_addr,
            gcb.size(),
            static_cast<uint32_t>(gcb.buffer_address()),
        };

        KernelHandle kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/dram_core_prefetcher.cpp",
            sender_logical,
            DramConfig{.noc = NOC::NOC_0, .compile_args = compile_args});

        // Runtime args:
        //   bank_id, [tensor_offset]*num_tensors, [block_size]*num_tensors,
        //   [recv_noc_x, recv_noc_y]*num_receivers
        std::vector<uint32_t> rt_args;
        const uint32_t bank_id = sender_logical.x;
        rt_args.push_back(bank_id);
        for (const auto* t : data_tensors) {
            rt_args.push_back(static_cast<uint32_t>(t->buffer()->address()));
        }
        for (const auto* t : data_tensors) {
            const auto shard_shape = t->buffer()->shard_spec().shape();
            const uint32_t bytes_per_tile = tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(t->dtype()));
            const uint32_t num_tiles_in_shard =
                (shard_shape[0] / tt::constants::TILE_HEIGHT) * (shard_shape[1] / tt::constants::TILE_WIDTH);
            const uint32_t shard_bytes = num_tiles_in_shard * bytes_per_tile;
            rt_args.push_back(shard_bytes / num_receivers);
        }
        const auto& receiver_phys = gcb.receiver_coords_per_sender().at(s);
        for (const auto& c : receiver_phys) {
            rt_args.push_back(c.x);
            rt_args.push_back(c.y);
        }
        SetRuntimeArgs(program, kernel_id, sender_logical, rt_args);
    }

    return cached_program_t{std::move(program), shared_variables_t{}};
}

void DramPrefetcherDramCoreProgramFactory::override_runtime_arguments(
    cached_program_t& /*cached_program*/,
    const DramPrefetcherParams& /*args*/,
    const DramPrefetcherInputs& /*tensor_args*/,
    Tensor& /*output_tensor*/) {
    // Tensor addresses are baked into compile-time-arg-style runtime args at create() time. The
    // prototype rebuilds the program on each launch by re-running create(), so there is nothing
    // to override here.
}

}  // namespace ttnn::prim
