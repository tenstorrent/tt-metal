// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/global_circular_buffer.hpp"

#include <tt-metalium/experimental/prefetcher_pipe.hpp>

#include <memory>
#include <unordered_set>
#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/experimental/global_circular_buffer.hpp>
#include <tt-metalium/experimental/allocation_context.hpp>
#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace ttnn::global_circular_buffer {

GlobalCircularBuffer create_global_circular_buffer(
    MeshDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type) {
    auto guard = tt::tt_metal::make_allocation_context_guard("ttnn.create_global_circular_buffer");
    return tt::tt_metal::experimental::GlobalCircularBuffer(*device, sender_receiver_core_mapping, size, buffer_type);
}

namespace {

// Classify a prefetcher weight's DRAM layout. This mirrors tt_metal's detect_layout_mode
// (impl/buffers/tensor_prefetcher_manager.cpp) and MUST stay in sync with it: the runtime
// prefetcher routes on that function, so if this factory classifies a weight differently the
// validation here and the kernel's actual consumption disagree (wrong output / silent hang).
//
// Key on how the weight was ALLOCATED, not on shard count. Some legacy WIDTH_SHARDED buffers also
// expose an NdShardSpec-like descriptor via BDS, so the explicit legacy shard spec must win; and a
// shard-count test is ambiguous when total_receivers == num_banks (num_shards == num_banks in both
// layouts). num_shards == ring_size for recv-contig is enforced separately by the validator. At one
// receiver per bank the prefetcher reads a WIDTH_SHARDED weight as receiver-contiguous while this
// checks it as K-row-major; the two layouts name the same bytes there, so the checks agree.
bool is_receiver_contiguous_weight(const ttnn::Tensor& weight) {
    TT_FATAL(weight.is_allocated() && weight.buffer()->is_dram(), "prefetcher weight must live in DRAM");
    if (weight.buffer()->has_shard_spec()) {
        return false;  // legacy K-row-major (WIDTH_SHARDED)
    }
    return weight.nd_shard_spec().has_value();
}

// The consumer -> block_count contract, shared by every weight layout: gather takes one K-block per
// ring position, mcast takes K_tiles / in0_block_w natural-order blocks. This is the silent-hang
// guard — an indivisible K makes the prefetcher round the K-block width up and over-read past the
// weight while the matmul waits on pages that never come — so it lives in one place for all layouts.
uint32_t block_count_for_consumer(
    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config,
    uint32_t weight_K_tiles,
    uint32_t receiver_count) {
    if (program_config.gather_in0) {
        TT_FATAL(
            weight_K_tiles % receiver_count == 0,
            "weight K ({} tiles) must be divisible by ring_size ({}) for gather_in0; remainder {}. The matmul "
            "activation grid would pad K beyond what the prefetcher pushes and the receivers would wait forever "
            "for in1 pages.",
            weight_K_tiles,
            receiver_count,
            weight_K_tiles % receiver_count);
        return receiver_count;
    }
    TT_FATAL(program_config.in0_block_w > 0, "mcast_in0 requires in0_block_w > 0");
    TT_FATAL(
        weight_K_tiles % program_config.in0_block_w == 0,
        "weight K ({} tiles) must be divisible by mcast_in0 in0_block_w ({}); remainder {}",
        weight_K_tiles,
        program_config.in0_block_w,
        weight_K_tiles % program_config.in0_block_w);
    return weight_K_tiles / static_cast<uint32_t>(program_config.in0_block_w);
}

// How deep in K a receiver's slab of the weight may be. A gather_in0 matmul over PrefetcherPipes takes
// K-blocks one in0 shard deep, and when K does not fill the ring its shards are padded, so the slab has
// to run past the weight's K for the prefetcher to cut it into whole shard-deep K-blocks (the matmul
// checks the depth against its shards). Every other consumer reads the weight's own K.
enum class SlabDepth : uint8_t {
    WeightK,
    PaddedForPipeGather,
};

// Shared receiver-contiguous weight ↔ matmul cross-checks. Returns the number of K-blocks the
// prefetcher must push per receiver: gather-in0 uses one block per ring position, while mcast-in0
// uses the configured inner-dimension block width.
uint32_t validate_recv_contig_weight_for_matmul_1d(
    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config,
    const ttnn::Tensor& weight,
    uint32_t receiver_count,
    SlabDepth slab_depth) {
    TT_FATAL(
        program_config.gather_in0 != program_config.mcast_in0,
        "receiver-contiguous Tensor prefetcher requires exactly one of gather_in0 or mcast_in0 to be true");
    TT_FATAL(
        !program_config.mcast_in0 || !program_config.stream_in1,
        "mcast_in0 consumes GCB blocks in natural FIFO order and requires stream_in1=false");
    TT_FATAL(receiver_count > 0, "receiver_count must be > 0");

    // The receiver-contiguous weight is an NdShardSpec DRAM tensor: num_shards == receiver_count,
    // each shard (full K, N/receiver_count). This is also exactly what the manager's
    // detect_layout_mode keys on.
    TT_FATAL(weight.is_allocated() && weight.buffer()->is_dram(), "weight must live in DRAM");
    const auto& nd_opt = weight.nd_shard_spec();
    TT_FATAL(
        nd_opt.has_value(),
        "weight must be allocated with an NdShardSpec (ttnn.MemoryConfig(BufferType.DRAM, NdShardSpec(...))) "
        "for the receiver-contiguous Tensor prefetcher path");
    // A rank-3 shard (G, K, n_per_recv) is a grouped weight (an expert stack): every receiver slab holds
    // all G groups, each a (K, n_per_recv) block the prefetcher streams like a weight of its own.
    const auto& shard_shape = nd_opt->shard_shape;
    TT_FATAL(
        shard_shape.rank() == 2 || shard_shape.rank() == 3,
        "receiver-contiguous NdShardSpec shard shape must be (K, n_per_recv) or, for a grouped weight, "
        "(num_groups, K, n_per_recv); got {}",
        shard_shape);
    if (shard_shape.rank() == 3) {
        TT_FATAL(
            weight.padded_shape().rank() >= 3 && shard_shape[0] == weight.padded_shape()[-3],
            "grouped receiver-contiguous weight: the shard's group dim ({}) must hold the weight's whole dim -3 "
            "(weight shape {})",
            shard_shape[0],
            weight.padded_shape());
    }

    const auto& tile = weight.tensor_spec().tile();
    const uint32_t tile_h = tile.get_height();
    const uint32_t tile_w = tile.get_width();
    const uint32_t shard_K = shard_shape[-2];
    const uint32_t shard_N = shard_shape[-1];
    TT_FATAL(
        shard_K % tile_h == 0 && shard_N % tile_w == 0,
        "receiver-contiguous shard shape ({}, {}) must be tile-aligned (tile {}x{})",
        shard_K,
        shard_N,
        tile_h,
        tile_w);

    const auto& wp = weight.padded_shape();
    TT_FATAL(wp.rank() >= 2, "weight must be at least 2D; got rank {}", wp.rank());
    const auto weight_K = static_cast<uint32_t>(wp[-2]);
    const bool slab_may_pad = slab_depth == SlabDepth::PaddedForPipeGather;
    TT_FATAL(
        slab_may_pad ? (shard_K >= weight_K) : (shard_K == weight_K),
        "receiver-contiguous shard K ({}) must {} the weight's K ({}); each shard spans the full K dimension",
        shard_K,
        slab_may_pad ? "cover" : "equal",
        weight_K);

    const auto& bds = weight.buffer()->buffer_distribution_spec();
    TT_FATAL(bds.has_value(), "receiver-contiguous weight buffer must have a BufferDistributionSpec");
    TT_FATAL(
        static_cast<uint32_t>(bds->num_shards()) == receiver_count,
        "receiver-contiguous weight has {} shards but global_cb has {} receivers; num_shards must equal "
        "receiver_count (one shard per receiver)",
        bds->num_shards(),
        receiver_count);
    TT_FATAL(
        static_cast<uint64_t>(shard_N) * receiver_count == static_cast<uint64_t>(wp[-1]),
        "receiver-contiguous shard N ({}) * receiver_count ({}) must equal full weight N ({})",
        shard_N,
        receiver_count,
        wp[-1]);

    const uint32_t weight_K_tiles = shard_K / tile_h;
    const uint32_t block_count = block_count_for_consumer(program_config, weight_K_tiles, receiver_count);

    // Page-size guard: the matmul sizes its in1 remote-CB page from per_core_N; the prefetcher pushes
    // pages of n_per_recv tiles. A mismatch desyncs the page-credit accounting (wrong output / hang).
    const uint32_t n_per_recv_tiles = shard_N / tile_w;
    TT_FATAL(
        n_per_recv_tiles == program_config.per_core_N,
        "program_config.per_core_N ({}) must equal the weight's per-receiver N ({} tiles = shard N {} / tile_w {})",
        program_config.per_core_N,
        n_per_recv_tiles,
        shard_N,
        tile_w);
    return block_count;
}

// Shared legacy K-row-major (WIDTH_SHARDED) weight ↔ matmul cross-checks, the counterpart to
// validate_recv_contig_weight_for_matmul_1d above. One shard per DRAM bank spanning the full K and
// N/num_banks columns, K-row-major within the bank so one read serves all of that bank's receivers.
// Returns the number of K-blocks the prefetcher must push per receiver: gather-in0 uses one block per
// ring position, mcast-in0 uses the configured inner-dimension block width.
//
// The layout carries no ring assumption of its own — the sender divides K into whatever block count the
// request names (compute_tensor_layout_krow_major in tensor_prefetcher_manager.cpp) and each receiver
// gets its own N-slice of every block — so the per-receiver page stream is the same shape either
// consumer sees on a receiver-contiguous weight.
//
// `recv_per_bank` is how many receivers each bank's shard is sliced across: what the delivery target
// gives every bank, which the caller has already checked is the same for each.
uint32_t validate_krow_major_weight_for_matmul_1d(
    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config,
    const ttnn::Tensor& weight,
    uint32_t receiver_count,
    uint32_t recv_per_bank) {
    TT_FATAL(
        program_config.gather_in0 != program_config.mcast_in0,
        "legacy K-row-major Tensor prefetcher requires exactly one of gather_in0 or mcast_in0 to be true");
    // Streaming needs a per-receiver rotation table, and the K-row-major sender has no rotation
    // parameter at all (compute_tensor_layout_krow_major in tensor_prefetcher_manager.cpp), so a
    // rotation would be silently dropped and the streaming matmul would wait on blocks delivered in
    // natural order. Reject it here, where every other layout rule lives.
    TT_FATAL(
        !program_config.stream_in1,
        "stream_in1 requires a receiver-contiguous weight: the legacy K-row-major sender cannot deliver a "
        "per-receiver ring rotation, so a streaming matmul would wait forever on natural-order blocks");
    TT_FATAL(receiver_count > 0, "receiver_count must be > 0");
    TT_FATAL(recv_per_bank > 0, "receivers per bank must be > 0");
    TT_FATAL(
        receiver_count % recv_per_bank == 0,
        "receiver count ({}) must be a whole number of banks at {} receivers per bank; the legacy K-row-major "
        "layout gives every bank the same receiver fan-out",
        receiver_count,
        recv_per_bank);
    const uint32_t num_senders = receiver_count / recv_per_bank;

    TT_FATAL(weight.is_allocated() && weight.buffer()->is_dram(), "weight must live in DRAM");
    TT_FATAL(
        weight.buffer()->has_shard_spec(),
        "weight must be WIDTH_SHARDED across the DRAM banks (ttnn.MemoryConfig(TensorMemoryLayout.WIDTH_SHARDED, "
        "BufferType.DRAM, ShardSpec(...))) for the legacy K-row-major Tensor prefetcher path");

    const auto& wp = weight.padded_shape();
    TT_FATAL(wp.rank() >= 2, "weight must be at least 2D; got rank {}", wp.rank());
    const auto& tile = weight.tensor_spec().tile();
    const uint32_t tile_h = tile.get_height();
    const uint32_t tile_w = tile.get_width();
    const uint32_t weight_K = wp[-2];
    const uint32_t weight_N = wp[-1];
    TT_FATAL(weight_K % tile_h == 0, "weight K ({}) must be tile-aligned (tile_h={})", weight_K, tile_h);
    TT_FATAL(weight_N % tile_w == 0, "weight N ({}) must be tile-aligned (tile_w={})", weight_N, tile_w);
    const uint32_t weight_K_tiles = weight_K / tile_h;

    const auto& shard_shape = weight.buffer()->shard_spec().shape();
    const uint32_t shard_K = shard_shape[0];
    const uint32_t shard_N = shard_shape[1];
    TT_FATAL(
        shard_K == weight_K,
        "DRAM shard K ({}) must equal full K ({}); the weight must be width-sharded across banks with each bank "
        "holding the full K dimension",
        shard_K,
        weight_K);
    TT_FATAL(
        shard_N * num_senders == weight_N,
        "DRAM shard N ({}) * num_senders ({}) must equal full N ({})",
        shard_N,
        num_senders,
        weight_N);
    const uint32_t shard_N_tiles = shard_N / tile_w;
    TT_FATAL(
        shard_N_tiles % recv_per_bank == 0,
        "per-bank N ({} tiles) must be divisible by the {} receivers per bank",
        shard_N_tiles,
        recv_per_bank);
    const uint32_t per_recv_N_tiles = shard_N_tiles / recv_per_bank;
    TT_FATAL(
        per_recv_N_tiles == program_config.per_core_N,
        "program_config.per_core_N ({}) must equal the weight's per-receiver N ({} tiles = per-bank N {} tiles / "
        "{} receivers per bank); otherwise the prefetcher's pushed page size and the matmul's in1 page size "
        "disagree and the page-credit accounting desyncs",
        program_config.per_core_N,
        per_recv_N_tiles,
        shard_N_tiles,
        recv_per_bank);

    const uint32_t block_count = block_count_for_consumer(program_config, weight_K_tiles, receiver_count);
    if (program_config.gather_in0) {
        // Legacy-only extra guard, predating this layout's mcast consumer: gather ignores
        // cfg.in0_block_w (it derives the K-block width from K/ring_size), but the K-row-major builder
        // has always required K to divide by it, so keep rejecting exactly the configs it always did.
        TT_FATAL(
            weight_K_tiles % program_config.in0_block_w == 0,
            "legacy K-row-major gather still requires weight K ({} tiles) divisible by config in0_block_w ({}); "
            "the matmul derives K-block width from K/ring_size, not this field",
            weight_K_tiles,
            program_config.in0_block_w);
    }
    return block_count;
}

// The K-row-major sender slices a bank's shard by a single receivers-per-bank number — the manager
// derives it as total_receivers / num_banks in serialize_request_pages
// (impl/buffers/tensor_prefetcher_manager.cpp) — so every sender must own the same receiver count.
// The manager already rejects split and partial-bank sender topologies and requires the total to
// divide evenly across banks, but a mapping whose per-bank counts are uneven yet still average out
// ((1, 3, 2, 2) over four banks) clears both of those and then gets sliced at the average, silently
// delivering the wrong pages.
//
// Only meaningful for K-row-major: a receiver-contiguous weight takes its geometry from the shard
// shape and tolerates non-uniform per-bank receivers (that's what the dual-sender split produces).
void validate_krow_major_gcb_topology(
    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config,
    const GlobalCircularBuffer& gcb) {
    const uint32_t recv_per_bank = static_cast<uint32_t>(program_config.num_global_cb_receivers);
    const auto& mapping = gcb.sender_receiver_core_mapping();
    for (size_t s = 0; s < mapping.size(); ++s) {
        const uint32_t sender_recv_count = mapping[s].second.num_cores();
        TT_FATAL(
            sender_recv_count == recv_per_bank,
            "global_cb sender {} (core {}) drives {} receivers, but the legacy K-row-major layout gives every bank "
            "the same fan-out of num_global_cb_receivers ({}); the sender would slice its shard at the average "
            "receiver count and deliver the wrong pages",
            s,
            mapping[s].first.str(),
            sender_recv_count,
            recv_per_bank);
    }
}

}  // namespace

uint32_t tensor_prefetcher_block_count_for_matmul_1d(
    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config,
    const ttnn::Tensor& weight,
    const GlobalCircularBuffer& gcb) {
    const uint32_t receiver_count = gcb.receiver_cores().num_cores();
    TT_FATAL(receiver_count > 0, "global_cb has no receivers");
    if (is_receiver_contiguous_weight(weight)) {
        return validate_recv_contig_weight_for_matmul_1d(program_config, weight, receiver_count, SlabDepth::WeightK);
    }
    // Weight checks first: they establish num_global_cb_receivers > 0 and that it divides the
    // receiver count, which is what the per-sender rule below is stated against.
    const uint32_t block_count = validate_krow_major_weight_for_matmul_1d(
        program_config, weight, receiver_count, static_cast<uint32_t>(program_config.num_global_cb_receivers));
    validate_krow_major_gcb_topology(program_config, gcb);
    return block_count;
}

uint32_t tensor_prefetcher_block_count_for_matmul_1d(
    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config,
    const ttnn::Tensor& weight,
    const ttnn::PrefetcherPipeList& prefetcher_pipes) {
    const uint32_t receiver_count =
        tt::tt_metal::experimental::GetPrefetcherPipeReceiverCores(ttnn::prefetcher_pipe_refs(prefetcher_pipes))
            .num_cores();
    TT_FATAL(receiver_count > 0, "prefetcher_pipes has no receivers");
    if (is_receiver_contiguous_weight(weight)) {
        return validate_recv_contig_weight_for_matmul_1d(
            program_config,
            weight,
            receiver_count,
            program_config.gather_in0 ? SlabDepth::PaddedForPipeGather : SlabDepth::WeightK);
    }

    // K-row-major: one read of a bank's shard serves every receiver on that bank, so each bank has one
    // sender, and the sender slices the shard by a single receivers-per-bank count, so each sender
    // drives the same number of receivers.
    const auto mapping = tt::tt_metal::experimental::GetPrefetcherPipeSenderReceiverMapping(
        ttnn::prefetcher_pipe_refs(prefetcher_pipes));
    const uint32_t recv_per_bank = mapping.front().second.num_cores();
    std::unordered_set<uint32_t> banks;
    for (const auto& [sender, receivers] : mapping) {
        const auto bank = static_cast<uint32_t>(sender.x);
        TT_FATAL(
            banks.insert(bank).second,
            "a legacy K-row-major (WIDTH_SHARDED) weight needs one sender per DRAM bank, but bank {} has two "
            "prefetcher_pipes; create them with support_multi_receiver_shards=True",
            bank);
        TT_FATAL(
            receivers.num_cores() == recv_per_bank,
            "a legacy K-row-major (WIDTH_SHARDED) weight is sliced by one receivers-per-bank count, but bank {}'s pipe "
            "drives {} receivers and bank {}'s drives {}; the sender would deliver the wrong pages",
            bank,
            receivers.num_cores(),
            mapping.front().first.x,
            recv_per_bank);
    }
    return validate_krow_major_weight_for_matmul_1d(program_config, weight, receiver_count, recv_per_bank);
}

}  // namespace ttnn::global_circular_buffer
