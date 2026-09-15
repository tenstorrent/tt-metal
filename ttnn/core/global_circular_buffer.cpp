// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/global_circular_buffer.hpp"

#include <memory>
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

GlobalCircularBuffer create_global_circular_buffer_for_tensor_prefetcher(
    MeshDevice* mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type,
    bool support_multi_receiver_shards) {
    return tt::tt_metal::experimental::CreateGlobalCircularBufferForTensorPrefetcher(
        *mesh_device, bank_to_receivers, size, buffer_type, support_multi_receiver_shards);
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
bool is_receiver_contiguous_weight(const ttnn::Tensor& weight) {
    TT_FATAL(weight.buffer() != nullptr && weight.buffer()->is_dram(), "prefetcher weight must live in DRAM");
    if (weight.buffer()->has_shard_spec()) {
        return false;  // legacy K-row-major (WIDTH_SHARDED)
    }
    return weight.nd_shard_spec().has_value();
}

// A FIFO consumer takes K-blocks as they land, so it needs only a double-buffered window rather than
// the whole tensor resident.
constexpr uint32_t kFifoMinWindowBlocks = 2;

// True when the consumer drains the GCB block-at-a-time: mcast_in0 in natural FIFO order, or a
// gather_in0 matmul streaming in ring-rotated order. Only a batched gather waits for a whole layer.
// Phrased as a property of the config rather than a list of the consumers we know about today, so a
// new non-gather consumer gets the shallow floor rather than silently inheriting the full-layer one.
bool is_fifo_consumer(const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& config) {
    return !config.gather_in0 || config.stream_in1;
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

// One GCB page is one consumer K-block for one receiver. A gather_in0 matmul derives its effective
// in0_block_w from weight_K_tiles / ring_size rather than from cfg.in0_block_w (which is typically
// left at 1), so dividing K by the consumer's block count gives the width each consumer actually
// reads: that same derived width for gather, cfg.in0_block_w for mcast.
uint32_t gcb_page_bytes(
    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config,
    const ttnn::Tensor& weight,
    uint32_t block_count) {
    const auto& tile = weight.tensor_spec().tile();
    const uint32_t weight_K_tiles = static_cast<uint32_t>(weight.padded_shape()[-2]) / tile.get_height();
    const uint32_t k_block_w_tiles = weight_K_tiles / block_count;
    const uint32_t bytes_per_tile = tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(weight.dtype()));
    return static_cast<uint32_t>(k_block_w_tiles * program_config.per_core_N) * bytes_per_tile;
}

// gather_in0 consumes on a ring of exactly receiver_count workers, so its grid must hold that many;
// mcast_in0 only needs capacity for them (the matmul op pins receivers == cores-with-work).
void validate_grid_for_consumer(
    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config,
    size_t config_index,
    uint32_t receiver_count) {
    const auto& grid = program_config.compute_with_storage_grid_size;
    const uint32_t grid_capacity = grid.x * grid.y;
    if (program_config.gather_in0) {
        TT_FATAL(
            grid_capacity == receiver_count,
            "gather_in0 program_configs[{}] grid {}x{} = {} workers, but the GCB has {} receivers; they must match",
            config_index,
            grid.x,
            grid.y,
            grid_capacity,
            receiver_count);
    } else {
        TT_FATAL(
            grid_capacity >= receiver_count,
            "mcast_in0 program_configs[{}] grid {}x{} has capacity for {} workers, but the GCB has {} receivers",
            config_index,
            grid.x,
            grid.y,
            grid_capacity,
            receiver_count);
    }
}

// One consumer's own floor on the GCB window: a batched gather matmul does wait_front(block_count)
// and needs a whole layer resident, while a FIFO consumer takes K-blocks as they land and needs only
// a double-buffered window.
//
// Checked per (config, weight) as the builders walk them, rather than once against the largest page
// and the largest block count over all configs — those two maxima can come from different configs,
// and their product demands a buffer neither consumer asked for. A GCB shared by a batched gather
// (small page, whole layer) and an mcast (large page, two blocks) is the case that bites: the
// product can exceed kMaxCbPagesBytes and make the mix unconstructible even though each consumer
// fits on its own.
//
// No L1 budget check here — receivers may have very different L1 usage on top of the GCB (matmul
// in0/in1/out/interm CBs etc.) and we don't have enough context at the factory to compute a real cap.
// Callers must size the GCB to fit their own L1.
void validate_gcb_size_for_config(
    uint32_t size,
    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& config,
    size_t config_index,
    uint32_t page_bytes,
    uint32_t block_count) {
    const bool fifo = is_fifo_consumer(config);
    const uint32_t num_blocks = fifo ? kFifoMinWindowBlocks : block_count;
    const uint32_t min_size = page_bytes * num_blocks;
    TT_FATAL(
        size >= min_size,
        "GCB size ({} B) must be at least num_blocks * page ({} * {} = {} B) for program_configs[{}]. {}",
        size,
        num_blocks,
        page_bytes,
        min_size,
        config_index,
        fifo ? "A FIFO consumer takes K-blocks as they arrive but still needs a double-buffered window."
             : "A batched gather matmul does wait_front(num_blocks), so it needs that many pages "
               "buffered before it consumes.");
}

// kMaxCbPagesBytes is a cap on fifo_aligned_num_pages = fifo_size /
// REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE. Two reasons it exists:
//   1. The NoC stream overlay's STREAM_REMOTE_DEST_BUF_SIZE register holds the buffer size in
//      16-byte words and is 17 bits wide on BH/WH (see MEM_WORD_ADDR_WIDTH in
//      noc_overlay_parameters.h), so the largest representable buffer is (2^17 - 1) * 16 ~= 2 MB.
//      Paths that wire the GCB through the overlay would silently truncate beyond that.
//   2. The remote-CB receiver tracks pages with 32-bit counters wrapped at 2^31
//      (noc_fast_atomic_increment wrap=31) and computes
//      free_pages = fifo_aligned_num_pages - (pages_sent - pages_acked) in unsigned 32-bit
//      arithmetic. Keeping fifo_aligned_num_pages well under 2^30 leaves plenty of headroom between
//      the counter range and any plausible in-flight count so signed/unsigned interpretation of the
//      difference can never misfire.
// 2 MB satisfies both — it's the hardware overlay-field max, and ~5 orders of magnitude under the
// counter wrap.
void validate_gcb_size_cap(uint32_t size) {
    constexpr uint32_t kMaxCbPagesBytes = 131072u * 16u;
    TT_FATAL(
        size <= kMaxCbPagesBytes,
        "GCB size ({} B) exceeds the remote-CB page-count cap ({} B). Reduce size.",
        size,
        kMaxCbPagesBytes);
}

// Shared receiver-contiguous weight ↔ matmul cross-checks. Returns the number of K-blocks the
// prefetcher must push per receiver: gather-in0 uses one block per ring position, while mcast-in0
// uses the configured inner-dimension block width.
uint32_t validate_recv_contig_weight_for_matmul_1d(
    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config,
    const ttnn::Tensor& weight,
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
uint32_t validate_krow_major_weight_for_matmul_1d(
    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config,
    const ttnn::Tensor& weight,
    uint32_t receiver_count) {
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
    const uint32_t recv_per_bank = static_cast<uint32_t>(program_config.num_global_cb_receivers);
    TT_FATAL(recv_per_bank > 0, "num_global_cb_receivers must be > 0");
    TT_FATAL(
        receiver_count % recv_per_bank == 0,
        "global_cb receiver count ({}) must be a whole number of banks at num_global_cb_receivers ({}) per bank; "
        "the legacy K-row-major layout gives every bank the same receiver fan-out",
        receiver_count,
        recv_per_bank);
    const uint32_t num_senders = receiver_count / recv_per_bank;

    TT_FATAL(weight.buffer() != nullptr && weight.buffer()->is_dram(), "weight must live in DRAM");
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
        "per-bank N ({} tiles) must be divisible by num_global_cb_receivers ({})",
        shard_N_tiles,
        recv_per_bank);
    const uint32_t per_recv_N_tiles = shard_N_tiles / recv_per_bank;
    TT_FATAL(
        per_recv_N_tiles == program_config.per_core_N,
        "program_config.per_core_N ({}) must equal the weight's per-receiver N ({} tiles = per-bank N {} tiles / "
        "num_global_cb_receivers {}); otherwise the prefetcher's pushed page size and the matmul's in1 remote-CB "
        "page size disagree and the page-credit accounting desyncs",
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
// delivering the wrong pages. The GCB factory enforces uniformity where it builds the mapping
// (build_matmul_1d_gcb_krow_major below); this is the same rule for a caller-supplied GCB, which
// reaches the matmul through the entry point below rather than through the factory.
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
        return validate_recv_contig_weight_for_matmul_1d(program_config, weight, receiver_count);
    }
    // Weight checks first: they establish num_global_cb_receivers > 0 and that it divides the
    // receiver count, which is what the per-sender rule below is stated against.
    const uint32_t block_count = validate_krow_major_weight_for_matmul_1d(program_config, weight, receiver_count);
    validate_krow_major_gcb_topology(program_config, gcb);
    return block_count;
}

// Builds the GCB for a legacy K-row-major (WIDTH_SHARDED) weight: one shard per DRAM bank, the
// bank's shard interleaving all its receivers (one read serves every receiver on the bank). Always
// single-sender per bank. Extracted from the former public create_global_circular_buffer_for_matmul_1d;
// the public entry point now detects the layout and dispatches here or to build_matmul_1d_gcb_recv_contig.
static GlobalCircularBuffer build_matmul_1d_gcb_krow_major(
    MeshDevice* mesh_device,
    const std::vector<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>& program_configs,
    const std::vector<ttnn::Tensor>& weights,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t size,
    BufferType buffer_type) {
    TT_FATAL(size > 0, "size must be > 0");
    TT_FATAL(!bank_to_receivers.empty(), "bank_to_receivers must be non-empty");

    // All matmuls share the same GCB receiver rectangle, so they must all agree on the
    // ring shape and per-bank receiver count.
    const auto& first = program_configs.front();
    TT_FATAL(first.num_global_cb_receivers > 0, "config[0].num_global_cb_receivers must be > 0");

    const auto& grid = first.compute_with_storage_grid_size;
    const uint32_t ring_cols = grid.x;
    const uint32_t ring_rows = grid.y;
    const uint32_t num_senders = static_cast<uint32_t>(bank_to_receivers.size());
    const uint32_t num_recv_per_bank = static_cast<uint32_t>(first.num_global_cb_receivers);
    // A bank's single K-row-major read serves all of that bank's receivers, so the receiver count is
    // fixed by the bank fan-out rather than by the program config's grid.
    const uint32_t receiver_count = num_senders * num_recv_per_bank;

    // Validate bank_to_receivers shape against the program config: each bank must own exactly
    // num_recv_per_bank receiver cores (the grid rule is per-config, in the loop below). We don't check
    // that the receivers row-major-walk matches the matmul's activation grid in ring-position order —
    // that's the matmul op's responsibility at op-construction time.
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

    // Validate every (config, weight) pair against the matmul invariants, including the GCB window
    // that pair needs on its own.
    for (size_t i = 0; i < program_configs.size(); ++i) {
        const auto& cfg = program_configs[i];
        const auto& w = weights[i];

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

        validate_grid_for_consumer(cfg, i, receiver_count);

        // ---- Per-(config, weight) K-row-major cross-checks and consumer-specific K-block count ----
        const uint32_t block_count = validate_krow_major_weight_for_matmul_1d(cfg, w, receiver_count);

        const uint32_t in1_block_size = gcb_page_bytes(cfg, w, block_count);
        TT_FATAL(in1_block_size > 0, "config[{}] in1_block_size computed as 0", i);
        validate_gcb_size_for_config(size, cfg, i, in1_block_size, block_count);
    }

    validate_gcb_size_cap(size);

    return tt::tt_metal::experimental::CreateGlobalCircularBufferForTensorPrefetcher(
        *mesh_device, bank_to_receivers, size, buffer_type, /*support_multi_receiver_shards=*/true);
}

// Builds the GCB for a receiver-contiguous (NdShardSpec) weight: num_shards == ring_size, each shard
// (full K, N/ring_size) owned by exactly one receiver. Supports dual senders per bank. Extracted from
// the former public create_global_circular_buffer_for_matmul_1d_recv_contig.
static GlobalCircularBuffer build_matmul_1d_gcb_recv_contig(
    MeshDevice* mesh_device,
    const std::vector<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>& program_configs,
    const std::vector<ttnn::Tensor>& weights,
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
    for (size_t i = 0; i < program_configs.size(); ++i) {
        const auto& cfg = program_configs[i];
        validate_grid_for_consumer(cfg, i, receiver_count);

        // Per-(config, weight) recv-contig cross-checks and consumer-specific K-block count.
        const uint32_t block_count = validate_recv_contig_weight_for_matmul_1d(cfg, weights[i], receiver_count);

        const uint32_t page_bytes = gcb_page_bytes(cfg, weights[i], block_count);
        TT_FATAL(page_bytes > 0, "program_configs[{}] page_bytes computed as 0", i);
        validate_gcb_size_for_config(size, cfg, i, page_bytes, block_count);
    }

    validate_gcb_size_cap(size);

    return tt::tt_metal::experimental::CreateGlobalCircularBufferForTensorPrefetcher(
        *mesh_device, bank_to_receivers, size, buffer_type, support_multi_receiver_shards);
}

GlobalCircularBuffer create_global_circular_buffer_for_matmul_1d(
    MeshDevice* mesh_device,
    const std::vector<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>& program_configs,
    const std::vector<ttnn::Tensor>& weights,
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

    if (recv_contig) {
        // Dual senders are the production default for receiver-contiguous weights (highest per-bank
        // bandwidth); single-receiver banks fall back to one sender automatically. An explicit value
        // overrides (e.g. a benchmark forcing single-sender for an A/B comparison). Recall the flag's
        // sense: false => dual senders, true => single sender.
        const bool single_sender = support_multi_receiver_shards.value_or(false);
        return build_matmul_1d_gcb_recv_contig(
            mesh_device, program_configs, weights, bank_to_receivers, size, buffer_type, single_sender);
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
