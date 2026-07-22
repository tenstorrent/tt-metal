// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/global_circular_buffer.hpp"

#include <algorithm>
#include <memory>
#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/experimental/global_circular_buffer.hpp>
#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace ttnn::global_circular_buffer {

GlobalCircularBuffer create_global_circular_buffer(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type) {
    return tt::tt_metal::experimental::CreateGlobalCircularBuffer(
        device, sender_receiver_core_mapping, size, buffer_type);
}

GlobalCircularBuffer create_global_circular_buffer(
    MeshDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type) {
    return tt::tt_metal::experimental::CreateGlobalCircularBuffer(
        device, sender_receiver_core_mapping, size, buffer_type);
}

GlobalCircularBuffer create_global_circular_buffer_for_tensor_prefetcher(
    MeshDevice* mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type,
    bool support_multi_receiver_shards) {
    return tt::tt_metal::experimental::CreateGlobalCircularBufferForTensorPrefetcher(
        *mesh_device, bank_to_receivers, size, buffer_type, support_multi_receiver_shards);
}

// Builds the GCB for a legacy K-row-major (WIDTH_SHARDED) weight: one shard per DRAM bank, the
// bank's shard interleaving all its receivers (one read serves every receiver on the bank). Always
// single-sender per bank. Extracted from the former public create_global_circular_buffer_for_matmul_1d;
// the public entry point now detects the layout and dispatches here or to build_matmul_1d_gcb_recv_contig.
static GlobalCircularBuffer build_matmul_1d_gcb_krow_major(
    MeshDevice* mesh_device,
    const std::vector<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>& program_configs,
    const std::vector<tt::tt_metal::Tensor>& weights,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type) {
    TT_FATAL(size > 0, "size must be > 0");
    TT_FATAL(!bank_to_receivers.empty(), "bank_to_receivers must be non-empty");

    // All matmuls share the same GCB receiver rectangle, so they must all agree on the
    // ring shape and per-bank receiver count.
    const auto& first = program_configs.front();
    TT_FATAL(
        first.gather_in0,
        "create_global_circular_buffer_for_matmul_1d requires gather_in0=true on every program "
        "config; config[0] has gather_in0=false");
    TT_FATAL(first.num_global_cb_receivers > 0, "config[0].num_global_cb_receivers must be > 0");

    const auto& grid = first.compute_with_storage_grid_size;
    const uint32_t ring_cols = grid.x;
    const uint32_t ring_rows = grid.y;
    const uint32_t ring_size = ring_cols * ring_rows;
    const uint32_t num_senders = static_cast<uint32_t>(bank_to_receivers.size());
    const uint32_t num_recv_per_bank = static_cast<uint32_t>(first.num_global_cb_receivers);

    // Validate bank_to_receivers shape against the program config:
    //   - num_senders * num_recv_per_bank must equal ring_size (the matmul's worker count).
    //   - Each bank must own exactly num_recv_per_bank receiver cores.
    // We don't check that the receivers row-major-walk matches the matmul's activation grid in
    // ring-position order — that's the matmul op's responsibility at op-construction time.
    TT_FATAL(
        num_senders * num_recv_per_bank == ring_size,
        "bank_to_receivers has {} senders * {} receivers/bank = {} cores, but program_config "
        "ring_size ({} x {} = {}) needs that many receivers",
        num_senders,
        num_recv_per_bank,
        num_senders * num_recv_per_bank,
        ring_cols,
        ring_rows,
        ring_size);
    for (size_t b = 0; b < bank_to_receivers.size(); ++b) {
        const uint32_t bank_recv_count = bank_to_receivers[b].second.num_cores();
        TT_FATAL(
            bank_recv_count == num_recv_per_bank,
            "bank_to_receivers[{}] (bank_id={}) has {} receiver cores; expected num_global_cb_receivers={}",
            b,
            bank_to_receivers[b].first,
            bank_recv_count,
            num_recv_per_bank);
    }

    // Validate every (config, weight) pair against the matmul invariants, and collect
    // the largest in1_block_size to size the buffer.
    uint32_t max_in1_block_size = 0;
    for (size_t i = 0; i < program_configs.size(); ++i) {
        const auto& cfg = program_configs[i];
        const auto& w = weights[i];

        TT_FATAL(cfg.gather_in0, "config[{}].gather_in0 must be true", i);
        TT_FATAL(cfg.num_global_cb_receivers > 0, "config[{}].num_global_cb_receivers must be > 0", i);
        TT_FATAL(
            cfg.compute_with_storage_grid_size.x == ring_cols && cfg.compute_with_storage_grid_size.y == ring_rows,
            "config[{}] has compute_with_storage_grid_size {{{}, {}}}; must match config[0] {{{}, {}}} "
            "(all matmuls sharing a GCB must use the same receiver rectangle)",
            i,
            cfg.compute_with_storage_grid_size.x,
            cfg.compute_with_storage_grid_size.y,
            ring_cols,
            ring_rows);
        TT_FATAL(
            static_cast<uint32_t>(cfg.num_global_cb_receivers) == num_recv_per_bank,
            "config[{}].num_global_cb_receivers ({}) must match config[0] ({}); the GCB has a single "
            "receiver-per-bank count shared across all matmuls",
            i,
            cfg.num_global_cb_receivers,
            num_recv_per_bank);

        // ---- Weight shape & dtype ----
        const auto& wp = w.padded_shape();
        TT_FATAL(wp.rank() >= 2, "weights[{}] must be at least 2D; got rank {}", i, wp.rank());
        const auto& tile = w.tensor_spec().tile();
        const uint32_t tile_h = tile.get_height();
        const uint32_t tile_w = tile.get_width();
        const uint32_t weight_K = wp[-2];
        const uint32_t weight_N = wp[-1];
        TT_FATAL(weight_K % tile_h == 0, "weights[{}] K ({}) must be tile-aligned (tile_h={})", i, weight_K, tile_h);
        TT_FATAL(weight_N % tile_w == 0, "weights[{}] N ({}) must be tile-aligned (tile_w={})", i, weight_N, tile_w);
        const uint32_t weight_K_tiles = weight_K / tile_h;
        const uint32_t weight_N_tiles = weight_N / tile_w;

        // ---- The silent-hang check ----
        TT_FATAL(
            weight_K_tiles % ring_size == 0,
            "weights[{}] K must be divisible by ring_size in tiles. Got weight_K_tiles={}, ring_size={} "
            "(remainder={}). The matmul activation grid would pad K beyond what the prefetcher pushes "
            "and the receivers would wait forever for in1 pages.",
            i,
            weight_K_tiles,
            ring_size,
            weight_K_tiles % ring_size);
        TT_FATAL(
            weight_K_tiles % cfg.in0_block_w == 0,
            "weights[{}] K ({} tiles) must be divisible by config[{}].in0_block_w ({})",
            i,
            weight_K_tiles,
            i,
            cfg.in0_block_w);
        TT_FATAL(
            weight_N_tiles % num_senders == 0,
            "weights[{}] N ({} tiles) must be divisible by num_senders ({})",
            i,
            weight_N_tiles,
            num_senders);

        // ---- Weight DRAM shard layout ----
        TT_FATAL(w.buffer() != nullptr && w.buffer()->is_dram(), "weights[{}] must live in DRAM", i);
        const auto& shard_shape = w.buffer()->shard_spec().shape();
        const uint32_t shard_K = shard_shape[0];
        const uint32_t shard_N = shard_shape[1];
        TT_FATAL(
            shard_K == weight_K,
            "weights[{}] DRAM shard K ({}) must equal full K ({}); weight must be width-sharded across "
            "banks with each bank holding the full K dimension",
            i,
            shard_K,
            weight_K);
        TT_FATAL(
            shard_N * num_senders == weight_N,
            "weights[{}] DRAM shard N ({}) * num_senders ({}) must equal full N ({})",
            i,
            shard_N,
            num_senders,
            weight_N);
        const uint32_t shard_N_tiles = shard_N / tile_w;
        TT_FATAL(
            shard_N_tiles % num_recv_per_bank == 0,
            "weights[{}] per-bank N ({} tiles) must be divisible by num_global_cb_receivers ({})",
            i,
            shard_N_tiles,
            num_recv_per_bank);

        const uint32_t per_recv_N_tiles = shard_N_tiles / num_recv_per_bank;
        TT_FATAL(
            per_recv_N_tiles == cfg.per_core_N,
            "config[{}].per_core_N ({}) must equal weights[{}] per-receiver N ({} = shard_N_tiles {} "
            "/ num_global_cb_receivers {})",
            i,
            cfg.per_core_N,
            i,
            per_recv_N_tiles,
            shard_N_tiles,
            num_recv_per_bank);

        // ---- This matmul's in1 block size ----
        // gather_in0 matmul derives its effective in0_block_w from weight_K_tiles / ring_size,
        // not from cfg.in0_block_w (which is typically left at 1 in the program config). Use the
        // same derivation here so the GCB page matches what the matmul will actually consume.
        const uint32_t actual_in0_block_w = weight_K_tiles / ring_size;
        const uint32_t bytes_per_tile = tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(w.dtype()));
        const uint32_t in1_block_size = static_cast<uint32_t>(actual_in0_block_w * cfg.per_core_N) * bytes_per_tile;
        TT_FATAL(in1_block_size > 0, "config[{}] in1_block_size computed as 0", i);
        max_in1_block_size = std::max(max_in1_block_size, in1_block_size);
    }

    // ---- Validate the caller-supplied size fits the remote-CB page-count cap and at
    // least one full layer's worth of pages (the matmul does wait_front(num_blocks)).
    //
    // No L1 budget check here — receivers may have very different L1 usage on top of the
    // GCB (matmul in0/in1/out/interm CBs etc.) and we don't have enough context at the
    // factory to compute a real cap. Callers must size the GCB to fit their own L1.
    //
    // kMaxCbPagesBytes is a cap on fifo_aligned_num_pages = fifo_size /
    // REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE. Two reasons it exists:
    //   1. The NoC stream overlay's STREAM_REMOTE_DEST_BUF_SIZE register holds the
    //      buffer size in 16-byte words and is 17 bits wide on BH/WH (see
    //      MEM_WORD_ADDR_WIDTH in noc_overlay_parameters.h), so the largest representable
    //      buffer is (2^17 - 1) * 16 ≈ 2 MB. Paths that wire the GCB through the overlay
    //      would silently truncate beyond that.
    //   2. The remote-CB receiver tracks pages with 32-bit counters wrapped at 2^31
    //      (noc_fast_atomic_increment wrap=31) and computes
    //      free_pages = fifo_aligned_num_pages - (pages_sent - pages_acked) in unsigned
    //      32-bit arithmetic. Keeping fifo_aligned_num_pages well under 2^30 leaves
    //      plenty of headroom between the counter range and any plausible in-flight count
    //      so signed/unsigned interpretation of the difference can never misfire.
    // 2 MB satisfies both — it's the hardware overlay-field max, and ~5 orders of magnitude
    // under the counter wrap.
    constexpr uint32_t kMaxCbPagesBytes = 131072u * 16u;
    const uint32_t num_blocks = ring_size;
    const uint32_t min_size = max_in1_block_size * num_blocks;
    TT_FATAL(
        size >= min_size,
        "GCB size ({} B) must be at least num_blocks * largest in1_block ({} * {} = {} B); the "
        "matmul does wait_front(num_blocks) so it needs that many pages buffered before it consumes.",
        size,
        num_blocks,
        max_in1_block_size,
        min_size);
    TT_FATAL(
        size <= kMaxCbPagesBytes,
        "GCB size ({} B) exceeds the remote-CB page-count cap ({} B). Reduce size.",
        size,
        kMaxCbPagesBytes);

    return tt::tt_metal::experimental::CreateGlobalCircularBufferForTensorPrefetcher(
        *mesh_device, bank_to_receivers, size, buffer_type, /*support_multi_receiver_shards=*/true);
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
// layouts). num_shards == ring_size for recv-contig is enforced separately by the validator.
bool is_receiver_contiguous_weight(const tt::tt_metal::Tensor& weight) {
    TT_FATAL(weight.buffer() != nullptr && weight.buffer()->is_dram(), "prefetcher weight must live in DRAM");
    if (weight.buffer()->has_shard_spec()) {
        return false;  // legacy K-row-major (WIDTH_SHARDED)
    }
    return weight.nd_shard_spec().has_value();
}

// Shared receiver-contiguous weight ↔ matmul cross-checks. Returns the number of K-blocks the
// prefetcher must push per receiver: gather-in0 uses one block per ring position, while mcast-in0
// uses the configured inner-dimension block width.
uint32_t validate_recv_contig_weight_for_matmul_1d(
    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config,
    const tt::tt_metal::Tensor& weight,
    uint32_t receiver_count) {
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
    TT_FATAL(weight.buffer() != nullptr && weight.buffer()->is_dram(), "weight must live in DRAM");
    const auto& nd_opt = weight.nd_shard_spec();
    TT_FATAL(
        nd_opt.has_value(),
        "weight must be allocated with an NdShardSpec (ttnn.MemoryConfig(BufferType.DRAM, NdShardSpec(...))) "
        "for the receiver-contiguous Tensor prefetcher path");
    const auto& shard_shape = nd_opt->shard_shape;
    TT_FATAL(
        shard_shape.rank() == 2,
        "receiver-contiguous NdShardSpec shard shape must be 2D (K, n_per_recv); got rank {}",
        shard_shape.rank());

    const auto& tile = weight.tensor_spec().tile();
    const uint32_t tile_h = tile.get_height();
    const uint32_t tile_w = tile.get_width();
    const uint32_t shard_K = shard_shape[0];
    const uint32_t shard_N = shard_shape[1];
    TT_FATAL(
        shard_K % tile_h == 0 && shard_N % tile_w == 0,
        "receiver-contiguous shard shape ({}, {}) must be tile-aligned (tile {}x{})",
        shard_K,
        shard_N,
        tile_h,
        tile_w);

    const auto& wp = weight.padded_shape();
    TT_FATAL(wp.rank() >= 2, "weight must be at least 2D; got rank {}", wp.rank());
    TT_FATAL(
        shard_K == static_cast<uint32_t>(wp[-2]),
        "receiver-contiguous shard K ({}) must equal full weight K ({}); each shard spans the full K dimension",
        shard_K,
        static_cast<uint32_t>(wp[-2]));

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
    uint32_t block_count = 0;
    if (program_config.gather_in0) {
        // Gather consumes one rotated K-block per ring position. Silent-hang / over-read guard: K must
        // divide evenly into receiver_count blocks. Otherwise the K-block width the prefetcher lays out
        // rounds up, the kernel reads past the receiver's slab, and the matmul (which pads K to a
        // multiple of receiver_count) waits on pages that never come.
        TT_FATAL(
            weight_K_tiles % receiver_count == 0,
            "weight K ({} tiles) must be divisible by receiver_count ({}) for gather_in0; remainder {}",
            weight_K_tiles,
            receiver_count,
            weight_K_tiles % receiver_count);
        block_count = receiver_count;
    } else {
        // Mcast consumes the same natural K-block sequence on every output worker. Same silent-hang /
        // over-read guard, keyed on in0_block_w: an indivisible K rounds the block width up and the
        // kernel over-reads while the matmul waits forever.
        TT_FATAL(program_config.in0_block_w > 0, "mcast_in0 requires in0_block_w > 0");
        TT_FATAL(
            weight_K_tiles % program_config.in0_block_w == 0,
            "weight K ({} tiles) must be divisible by mcast_in0 in0_block_w ({}); remainder {}",
            weight_K_tiles,
            program_config.in0_block_w,
            weight_K_tiles % program_config.in0_block_w);
        block_count = weight_K_tiles / static_cast<uint32_t>(program_config.in0_block_w);
    }

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

struct KRowMcastIn1Geometry {
    uint32_t block_count = 0;
    uint32_t page_bytes = 0;
};

KRowMcastIn1Geometry validate_krow_mcast_in1_weight_for_matmul_1d(
    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config,
    const tt::tt_metal::Tensor& weight,
    uint32_t num_banks) {
    TT_FATAL(
        !program_config.gather_in0 && !program_config.mcast_in0,
        "bank-striped Tensor prefetcher mcast-in1 requires gather_in0=false and mcast_in0=false");
    TT_FATAL(
        !program_config.stream_in1, "mcast-in1 consumes bank stripes in natural order and requires stream_in1=false");
    TT_FATAL(
        program_config.num_global_cb_receivers == 1,
        "bank-striped mcast-in1 requires num_global_cb_receivers=1 (one relay worker per DRAM bank), got {}",
        program_config.num_global_cb_receivers);
    TT_FATAL(num_banks > 0, "bank-striped mcast-in1 requires at least one DRAM bank");

    TT_FATAL(weight.buffer() != nullptr && weight.buffer()->is_dram(), "mcast-in1 weight must live in DRAM");
    TT_FATAL(
        weight.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
            weight.buffer()->has_shard_spec(),
        "bank-striped mcast-in1 requires a legacy WIDTH_SHARDED DRAM weight with one full-K shard per bank");

    const auto& wp = weight.padded_shape();
    TT_FATAL(wp.rank() >= 2, "mcast-in1 weight must be at least 2D; got rank {}", wp.rank());
    const auto& tile = weight.tensor_spec().tile();
    const uint32_t tile_h = tile.get_height();
    const uint32_t tile_w = tile.get_width();
    const uint32_t weight_K = wp[-2];
    const uint32_t weight_N = wp[-1];
    TT_FATAL(
        weight_K % tile_h == 0 && weight_N % tile_w == 0,
        "mcast-in1 weight shape ({}, {}) must be tile-aligned (tile {}x{})",
        weight_K,
        weight_N,
        tile_h,
        tile_w);
    const uint32_t weight_K_tiles = weight_K / tile_h;
    const uint32_t weight_N_tiles = weight_N / tile_w;

    const auto& shard_shape = weight.buffer()->shard_spec().shape();
    const uint32_t shard_K = shard_shape[0];
    const uint32_t shard_N = shard_shape[1];
    TT_FATAL(shard_K == weight_K, "mcast-in1 weight DRAM shard K ({}) must equal full K ({})", shard_K, weight_K);
    TT_FATAL(
        static_cast<uint64_t>(shard_N) * num_banks == weight_N,
        "mcast-in1 weight DRAM shard N ({}) * num_banks ({}) must equal full N ({})",
        shard_N,
        num_banks,
        weight_N);
    TT_FATAL(
        shard_N % tile_w == 0, "mcast-in1 weight per-bank N ({}) must be tile-aligned (tile_w={})", shard_N, tile_w);
    TT_FATAL(
        program_config.per_core_N == weight_N_tiles,
        "mcast-in1 program_config.per_core_N ({}) must equal full weight N ({} tiles); every matmul worker "
        "receives the assembled full-width in1 block",
        program_config.per_core_N,
        weight_N_tiles);
    TT_FATAL(program_config.in0_block_w > 0, "mcast-in1 requires in0_block_w > 0");
    TT_FATAL(
        weight_K_tiles % program_config.in0_block_w == 0,
        "mcast-in1 weight K ({} tiles) must be divisible by in0_block_w ({})",
        weight_K_tiles,
        program_config.in0_block_w);

    const uint32_t block_count = weight_K_tiles / static_cast<uint32_t>(program_config.in0_block_w);
    const uint32_t stripe_N_tiles = shard_N / tile_w;
    const uint32_t bytes_per_tile = tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(weight.dtype()));
    const uint32_t page_bytes = static_cast<uint32_t>(program_config.in0_block_w) * stripe_N_tiles * bytes_per_tile;
    TT_FATAL(block_count > 0 && page_bytes > 0, "mcast-in1 block_count and page_bytes must be non-zero");
    return {.block_count = block_count, .page_bytes = page_bytes};
}

}  // namespace

uint32_t tensor_prefetcher_block_count_for_matmul_1d(
    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config,
    const tt::tt_metal::Tensor& weight,
    const GlobalCircularBuffer& gcb) {
    if (!program_config.gather_in0 && !program_config.mcast_in0) {
        TT_FATAL(
            !is_receiver_contiguous_weight(weight),
            "bank-striped mcast-in1 requires a legacy WIDTH_SHARDED weight, not a receiver-contiguous NdShardSpec");
        const auto& mapping = gcb.sender_receiver_core_mapping();
        const uint32_t num_banks = static_cast<uint32_t>(mapping.size());
        TT_FATAL(num_banks > 0, "global_cb has no DRAM-bank senders");
        for (size_t i = 0; i < mapping.size(); ++i) {
            const auto& [sender, receivers] = mapping[i];
            TT_FATAL(
                receivers.num_cores() == 1,
                "mcast-in1 global_cb requires one relay receiver per bank; sender {} has {} receivers",
                sender,
                receivers.num_cores());
        }
        const auto geometry = validate_krow_mcast_in1_weight_for_matmul_1d(program_config, weight, num_banks);
        TT_FATAL(
            gcb.size() % geometry.page_bytes == 0,
            "mcast-in1 global_cb size {} must be a multiple of its per-bank stripe page size {}",
            gcb.size(),
            geometry.page_bytes);
        TT_FATAL(
            gcb.size() >= 2 * geometry.page_bytes,
            "mcast-in1 global_cb requires a two-page streaming window: size {} must be at least {}",
            gcb.size(),
            2 * geometry.page_bytes);
        return geometry.block_count;
    }

    const uint32_t receiver_count = gcb.receiver_cores().num_cores();
    TT_FATAL(receiver_count > 0, "global_cb has no receivers");
    return validate_recv_contig_weight_for_matmul_1d(program_config, weight, receiver_count);
}

// Builds the bank-striped mcast-in1 GCB. Each K-row-major DRAM bank owns one N stripe and unicasts
// it to exactly one relay worker. The relay workers assemble full in1 blocks on all matmul workers
// with worker-side multicast; the Tensor prefetcher itself remains unicast-only.
static GlobalCircularBuffer build_matmul_1d_gcb_krow_major_mcast_in1(
    MeshDevice* mesh_device,
    const std::vector<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>& program_configs,
    const std::vector<tt::tt_metal::Tensor>& weights,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type) {
    TT_FATAL(size > 0, "size must be > 0");
    TT_FATAL(!bank_to_receivers.empty(), "bank_to_receivers must be non-empty");

    const uint32_t num_banks = static_cast<uint32_t>(bank_to_receivers.size());
    std::vector<bool> seen_banks(num_banks, false);
    for (size_t i = 0; i < bank_to_receivers.size(); ++i) {
        const auto& [bank, receivers] = bank_to_receivers[i];
        TT_FATAL(
            bank < num_banks && !seen_banks[bank],
            "mcast-in1 bank ids must be dense and unique in [0, {}); bank_to_receivers[{}] has bank {}",
            num_banks,
            i,
            bank);
        seen_banks[bank] = true;
        TT_FATAL(
            receivers.num_cores() == 1,
            "mcast-in1 requires exactly one relay worker per DRAM bank; bank {} has {} receivers",
            bank,
            receivers.num_cores());
    }

    uint32_t max_page_bytes = 0;
    for (size_t i = 0; i < program_configs.size(); ++i) {
        const auto& cfg = program_configs[i];
        const uint32_t grid_capacity = cfg.compute_with_storage_grid_size.x * cfg.compute_with_storage_grid_size.y;
        TT_FATAL(
            grid_capacity >= num_banks,
            "mcast-in1 program_configs[{}] grid {}x{} has {} workers but needs at least one relay for each of {} "
            "DRAM banks",
            i,
            cfg.compute_with_storage_grid_size.x,
            cfg.compute_with_storage_grid_size.y,
            grid_capacity,
            num_banks);
        const auto geometry = validate_krow_mcast_in1_weight_for_matmul_1d(cfg, weights[i], num_banks);
        TT_FATAL(
            size % geometry.page_bytes == 0,
            "mcast-in1 GCB size {} must be a multiple of program_configs[{}] stripe page size {}",
            size,
            i,
            geometry.page_bytes);
        max_page_bytes = std::max(max_page_bytes, geometry.page_bytes);
    }

    constexpr uint32_t kMaxCbPagesBytes = 131072u * 16u;
    constexpr uint32_t kFifoMinWindowBlocks = 2;
    TT_FATAL(
        size >= kFifoMinWindowBlocks * max_page_bytes,
        "mcast-in1 GCB size ({} B) must hold at least two largest stripe pages (2 * {} B = {} B)",
        size,
        max_page_bytes,
        kFifoMinWindowBlocks * max_page_bytes);
    TT_FATAL(
        size <= kMaxCbPagesBytes,
        "GCB size ({} B) exceeds the remote-CB page-count cap ({} B). Reduce size.",
        size,
        kMaxCbPagesBytes);

    auto ordered_bank_to_receivers = bank_to_receivers;
    std::sort(ordered_bank_to_receivers.begin(), ordered_bank_to_receivers.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });
    return tt::tt_metal::experimental::CreateGlobalCircularBufferForTensorPrefetcher(
        *mesh_device, ordered_bank_to_receivers, size, buffer_type, /*support_multi_receiver_shards=*/true);
}

// Builds the GCB for a receiver-contiguous (NdShardSpec) weight: num_shards == ring_size, each shard
// (full K, N/ring_size) owned by exactly one receiver. Supports dual senders per bank. Extracted from
// the former public create_global_circular_buffer_for_matmul_1d_recv_contig.
static GlobalCircularBuffer build_matmul_1d_gcb_recv_contig(
    MeshDevice* mesh_device,
    const std::vector<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>& program_configs,
    const std::vector<tt::tt_metal::Tensor>& weights,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type,
    bool support_multi_receiver_shards) {
    TT_FATAL(size > 0, "size must be > 0");
    TT_FATAL(!bank_to_receivers.empty(), "bank_to_receivers must be non-empty");

    // receiver_count for the recv-contig layout is the total receiver count (= num_shards). Unlike the
    // K-row-major builder we do NOT require a uniform per-bank receiver count or a contiguous
    // bank->ring mapping — recv-contig uses a strided round-robin placement, and dual senders split
    // a bank's receivers across two DRISC cores.
    uint32_t receiver_count = 0;
    for (const auto& [_bank, receivers] : bank_to_receivers) {
        receiver_count += receivers.num_cores();
    }
    TT_FATAL(receiver_count > 0, "bank_to_receivers has no receivers");

    // All matmuls share the GCB receiver rectangle, so they must agree on the ring shape, and that
    // ring must match bank_to_receivers' total receiver count.
    uint32_t max_page_bytes = 0;
    uint32_t max_block_count = 0;
    bool all_configs_fifo = true;
    for (size_t i = 0; i < program_configs.size(); ++i) {
        const auto& cfg = program_configs[i];
        const auto& grid = cfg.compute_with_storage_grid_size;
        const uint32_t grid_capacity = grid.x * grid.y;
        if (cfg.gather_in0) {
            TT_FATAL(
                grid_capacity == receiver_count,
                "gather_in0 program_configs[{}] grid {}x{} = {} workers, but bank_to_receivers has {} total "
                "receivers; they must match",
                i,
                grid.x,
                grid.y,
                grid_capacity,
                receiver_count);
        } else {
            TT_FATAL(
                grid_capacity >= receiver_count,
                "mcast_in0 program_configs[{}] grid {}x{} has capacity for {} workers, but bank_to_receivers has "
                "{} receivers",
                i,
                grid.x,
                grid.y,
                grid_capacity,
                receiver_count);
        }

        // Per-(config, weight) recv-contig cross-checks and consumer-specific K-block count.
        const uint32_t block_count = validate_recv_contig_weight_for_matmul_1d(cfg, weights[i], receiver_count);
        max_block_count = std::max(max_block_count, block_count);
        all_configs_fifo = all_configs_fifo && (cfg.mcast_in0 || cfg.stream_in1);

        // One GCB page is one consumer K-block for one receiver.
        const auto& w = weights[i];
        const auto& tile = w.tensor_spec().tile();
        const uint32_t weight_K_tiles = static_cast<uint32_t>(w.padded_shape()[-2]) / tile.get_height();
        const uint32_t k_block_w_tiles = weight_K_tiles / block_count;
        const uint32_t bytes_per_tile = tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(w.dtype()));
        const uint32_t page_bytes = k_block_w_tiles * cfg.per_core_N * bytes_per_tile;
        TT_FATAL(page_bytes > 0, "program_configs[{}] page_bytes computed as 0", i);
        max_page_bytes = std::max(max_page_bytes, page_bytes);
    }

    // A batched gather matmul waits for all blocks before consuming. A stream_in1 gather or a
    // GCB-backed mcast consumes K-blocks FIFO as they land, so a shallow window is valid. Relax the
    // floor to a double-buffer when every matmul sharing this GCB is a FIFO consumer; otherwise keep
    // enough space for the largest full tensor. Same cap as the
    // K-row-major builder (see create_global_circular_buffer_for_matmul_1d for why kMaxCbPagesBytes
    // exists); no L1 budget check — callers size to fit their own receiver L1.
    constexpr uint32_t kMaxCbPagesBytes = 131072u * 16u;
    constexpr uint32_t kFifoMinWindowBlocks = 2;
    const uint32_t min_blocks = all_configs_fifo ? kFifoMinWindowBlocks : max_block_count;
    const uint32_t min_size = max_page_bytes * min_blocks;
    TT_FATAL(
        size >= min_size,
        "GCB size ({} B) must be at least {} * largest page ({} B) = {} B. {}",
        size,
        min_blocks,
        max_page_bytes,
        min_size,
        all_configs_fifo ? "FIFO matmuls consume K-blocks as they arrive but still need a double-buffered window."
                         : "A batched gather matmul needs a full layer buffered before it consumes.");
    TT_FATAL(
        size <= kMaxCbPagesBytes,
        "GCB size ({} B) exceeds the remote-CB page-count cap ({} B). Reduce size.",
        size,
        kMaxCbPagesBytes);

    return tt::tt_metal::experimental::CreateGlobalCircularBufferForTensorPrefetcher(
        *mesh_device, bank_to_receivers, size, buffer_type, support_multi_receiver_shards);
}

GlobalCircularBuffer create_global_circular_buffer_for_matmul_1d(
    MeshDevice* mesh_device,
    const std::vector<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>& program_configs,
    const std::vector<tt::tt_metal::Tensor>& weights,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type,
    std::optional<bool> support_multi_receiver_shards) {
    TT_FATAL(!program_configs.empty(), "Must provide at least one program config");
    TT_FATAL(
        program_configs.size() == weights.size(),
        "Expected one weight tensor per program config; got {} configs and {} weights",
        program_configs.size(),
        weights.size());

    // All weights share one GCB receiver rectangle, so they must all use the same DRAM layout.
    // Detect from the weight allocation (not the caller) so callers don't have to know which builder
    // to pick — the tensor's layout determines what the prefetcher does.
    const bool recv_contig = is_receiver_contiguous_weight(weights.front());
    for (size_t i = 1; i < weights.size(); ++i) {
        TT_FATAL(
            is_receiver_contiguous_weight(weights[i]) == recv_contig,
            "weights[{}] has a different DRAM layout than weights[0]; all weights sharing one GCB must be either "
            "all receiver-contiguous (NdShardSpec) or all legacy K-row-major (WIDTH_SHARDED)",
            i);
    }

    const bool mcast_in1 = !program_configs.front().gather_in0 && !program_configs.front().mcast_in0;
    for (size_t i = 0; i < program_configs.size(); ++i) {
        const bool config_mcast_in1 = !program_configs[i].gather_in0 && !program_configs[i].mcast_in0;
        TT_FATAL(
            config_mcast_in1 == mcast_in1,
            "All program configs sharing one GCB must use the same consumer mode; config[0] mcast-in1={} but "
            "config[{}] mcast-in1={}",
            mcast_in1,
            i,
            config_mcast_in1);
    }

    if (mcast_in1) {
        TT_FATAL(
            !recv_contig,
            "bank-striped mcast-in1 requires legacy K-row-major WIDTH_SHARDED weights, not receiver-contiguous "
            "NdShardSpec weights");
        TT_FATAL(
            support_multi_receiver_shards.value_or(true),
            "bank-striped mcast-in1 uses one K-row-major shard and one relay per bank, so dual DRAM senders are "
            "not supported");
        return build_matmul_1d_gcb_krow_major_mcast_in1(
            mesh_device, program_configs, weights, bank_to_receivers, size, buffer_type);
    }

    if (recv_contig) {
        // Dual senders are the production default for receiver-contiguous weights (highest per-bank
        // bandwidth); single-receiver banks fall back to one sender automatically. An explicit value
        // overrides (e.g. a benchmark forcing single-sender for an A/B comparison). Recall the flag's
        // sense: false => dual senders, true => single sender.
        const bool single_sender = support_multi_receiver_shards.value_or(false);
        return build_matmul_1d_gcb_recv_contig(
            mesh_device, program_configs, weights, bank_to_receivers, size, buffer_type, single_sender);
    }

    for (size_t i = 0; i < program_configs.size(); ++i) {
        TT_FATAL(
            program_configs[i].gather_in0,
            "program_configs[{}] uses mcast_in0, which requires a receiver-contiguous NdShardSpec weight",
            i);
    }

    // Legacy K-row-major is single-sender per bank by construction (a bank's shard feeds all its
    // receivers), so it cannot honor a dual-sender request. The layout-derived default is single; an
    // explicit request for dual (support_multi_receiver_shards=false) is an error here.
    TT_FATAL(
        support_multi_receiver_shards.value_or(true),
        "support_multi_receiver_shards=false (dual senders) requires a receiver-contiguous (NdShardSpec) weight; the "
        "supplied weight is legacy K-row-major (WIDTH_SHARDED), which is always single-sender per bank");
    return build_matmul_1d_gcb_krow_major(mesh_device, program_configs, weights, bank_to_receivers, size, buffer_type);
}

}  // namespace ttnn::global_circular_buffer
