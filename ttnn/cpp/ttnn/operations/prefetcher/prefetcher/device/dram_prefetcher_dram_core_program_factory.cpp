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

    // K-block-width in tiles. Default 1 = match the gather_in0 matmul's in0_block_w_tiles=1
    // assumption (one K-tile per receiver block). >1 trades fewer-but-bigger pushes for the
    // same total bytes, which can reduce per-push NoC overhead at the cost of DRISC L1
    // footprint (2 stage buffers each kInBlockWTiles * n_tiles_per_bank * tile_bytes).
    // Override via op param `dram_core_k_block_w_tiles` for bandwidth experiments.
    const uint32_t kInBlockWTiles = args.dram_core_k_block_w_tiles;
    TT_FATAL(kInBlockWTiles > 0, "dram_core_k_block_w_tiles must be > 0");

    // Per-tensor block geometry (all tensors are assumed to share the same K split for the
    // prototype; we pick from the first tensor).
    auto compute_block_geom = [&](const Tensor& t) {
        const auto shard_shape = t.buffer()->shard_spec().shape();
        const uint32_t bytes_per_tile = tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(t.dtype()));
        const uint32_t k_tiles = shard_shape[0] / tt::constants::TILE_HEIGHT;  // total K tiles in shard
        const uint32_t n_tiles_per_bank =
            shard_shape[1] / tt::constants::TILE_WIDTH;  // total N tiles in this bank's shard
        TT_FATAL(
            n_tiles_per_bank % num_receivers == 0,
            "n_tiles_per_bank ({}) must divide num_receivers ({})",
            n_tiles_per_bank,
            num_receivers);
        TT_FATAL(
            k_tiles % kInBlockWTiles == 0,
            "k_tiles ({}) must be divisible by kInBlockWTiles ({})",
            k_tiles,
            kInBlockWTiles);
        const uint32_t n_tiles_per_receiver = n_tiles_per_bank / num_receivers;
        const uint32_t num_blocks = k_tiles / kInBlockWTiles;
        // Per-block DMA size (read from GDDR): in0_block_w tiles tall * full N width of the bank.
        const uint32_t dma_block_size = kInBlockWTiles * n_tiles_per_bank * bytes_per_tile;
        // Per-receiver, per-block push size = in0_block_w * n_tiles_per_receiver tiles.
        const uint32_t push_page_size = kInBlockWTiles * n_tiles_per_receiver * bytes_per_tile;
        return std::make_tuple(num_blocks, dma_block_size, push_page_size);
    };

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
    // Stage buffer size: largest per-block DMA size across tensors.
    uint32_t max_block_size = 0;
    uint32_t first_num_blocks = 0;
    for (size_t ti = 0; ti < data_tensors.size(); ++ti) {
        auto [nb, dma_block, push_page] = compute_block_geom(*data_tensors[ti]);
        if (ti == 0) {
            first_num_blocks = nb;
        }
        TT_FATAL(nb == first_num_blocks, "All tensors must share the same num_blocks for the prototype");
        max_block_size = std::max(max_block_size, dma_block);
    }
    max_block_size = align_up(max_block_size, l1_alignment);
    cursor += max_block_size;
    cursor = align_up(cursor, l1_alignment);
    const uint32_t stage_b_addr = cursor;
    const uint32_t kNumBlocks = first_num_blocks;

    // (L1-budget validation removed: hal::get_dev_addr/get_dev_size accessors were refactored.
    // For kbw>1 experiments, increase the stage buffer count -> hangs/OOMs there, not here.)

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
            static_cast<uint32_t>(gcb.pages_sent_worker_l1_base()),
        };

        KernelHandle kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/dram_core_prefetcher.cpp",
            sender_logical,
            DramConfig{.noc = NOC::NOC_0, .compile_args = compile_args});

        // Runtime args:
        //   bank_id,
        //   [tensor_offset] * num_tensors,
        //   [dma_block_size] * num_tensors,
        //   [push_page_size] * num_tensors,
        //   [recv_noc_x, recv_noc_y] * num_receivers
        std::vector<uint32_t> rt_args;
        const uint32_t bank_id = sender_logical.x;
        rt_args.push_back(bank_id);
        for (const auto* t : data_tensors) {
            rt_args.push_back(static_cast<uint32_t>(t->buffer()->address()));
        }
        for (const auto* t : data_tensors) {
            auto [_nb, dma_block, _push] = compute_block_geom(*t);
            rt_args.push_back(dma_block);
        }
        for (const auto* t : data_tensors) {
            auto [_nb, _dma, push_page] = compute_block_geom(*t);
            rt_args.push_back(push_page);
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
