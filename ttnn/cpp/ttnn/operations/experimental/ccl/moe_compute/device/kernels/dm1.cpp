// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "moe_ring_common.h"

namespace {
// LOCAL_OUTPUT: the e_t and final-output TensorAccessor args follow the three matmul tensors in the
// compile-time args. They exist only when the mode is on; the disabled form parses the (always
// present) first accessor so the never-taken branch stays well-formed.
template <bool Enable, uint32_t Base>
struct LocalOutputArgs {
    static constexpr auto e_t_args = TensorAccessorArgs<0>();
    static constexpr auto out_args = TensorAccessorArgs<0>();
};
template <uint32_t Base>
struct LocalOutputArgs<true, Base> {
    static constexpr auto e_t_args = TensorAccessorArgs<Base>();
    static constexpr auto out_args = TensorAccessorArgs<e_t_args.next_compile_time_args_offset()>();
};
}  // namespace

void kernel_main() {
    constexpr bool has_bias = get_named_compile_time_arg_val("has_bias") == 1;
    constexpr uint32_t Ht = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t Nt = get_named_compile_time_arg_val("intermediate_tiles");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");

    // Compile time arguments
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    [[maybe_unused]] constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");

    // For synchronization with tilize cores
    constexpr uint32_t metadata_ready_semaphore_id = get_named_compile_time_arg_val("metadata_ready_semaphore_id");
    constexpr uint32_t matmul_chunk_ready_semaphore_id =
        get_named_compile_time_arg_val("matmul_chunk_ready_semaphore_id");
    // The ring exchange's credit: a core's successor increments it once per OWNED chunk when its compute has read
    // every a2a buffer of that chunk; the core waits for it before its first a2a write of its next owned chunk.
    constexpr uint32_t a2a_free_semaphore_id = get_named_compile_time_arg_val("a2a_free_semaphore_id");
    // The chunk halves of the input buffer (2 today, R + 1 with R prefill rings) and the drain's credit semaphore per
    // half: chunk g sits in half g % chunk_halves and frees it once consumed.
    constexpr uint32_t chunk_halves = get_named_compile_time_arg_val("chunk_halves");
    static_assert(chunk_halves >= 2 && chunk_halves <= moe_ring::rings::MAX_CHUNK_HALVES, "chunk_halves out of range");
    constexpr uint32_t half_free_semaphore_ids[moe_ring::rings::MAX_CHUNK_HALVES] = {
        get_named_compile_time_arg_val("half_free_semaphore_id_0"),
        get_named_compile_time_arg_val("half_free_semaphore_id_1"),
        get_named_compile_time_arg_val("half_free_semaphore_id_2"),
        get_named_compile_time_arg_val("half_free_semaphore_id_3"),
        get_named_compile_time_arg_val("half_free_semaphore_id_4")};
    // Chunk ownership over the prefill rings: every role derives the same table from the per-expert counts
    // (moe_ring::rings::ChunkOwners); one ring today, so every chunk is this core's.
    constexpr uint32_t prefill_rings = get_named_compile_time_arg_val("prefill_rings");
    constexpr uint32_t num_rings = prefill_rings < 2 ? 1u : prefill_rings;
    constexpr uint32_t per_expert_total_tokens_cb_id = get_named_compile_time_arg_val("per_expert_total_tokens_cb_id");
    constexpr uint32_t tokens_per_chunk = get_named_compile_time_arg_val("tokens_per_chunk");
    constexpr uint32_t tilize_drain_core_noc_x = get_named_compile_time_arg_val("tilize_drain_core_noc_x");
    constexpr uint32_t tilize_drain_core_noc_y = get_named_compile_time_arg_val("tilize_drain_core_noc_y");

    // Compile time arguments for writing to sharded output for combine
    constexpr uint32_t tile_height = get_named_compile_time_arg_val("tile_height");
    constexpr uint32_t tile_width = get_named_compile_time_arg_val("tile_width");
    constexpr uint32_t tile_width_size_bytes = get_named_compile_time_arg_val("tile_width_size_bytes");

    constexpr uint32_t combine_shard_width_tiles = get_named_compile_time_arg_val("combine_shard_width_tiles");
    constexpr uint32_t token_expert_row_offset = get_named_compile_time_arg_val("token_expert_row_offset");
    constexpr uint32_t height_shard_dim = get_named_compile_time_arg_val("height_shard_dim");
    constexpr uint32_t width_shard_dim = get_named_compile_time_arg_val("width_shard_dim");
    constexpr uint32_t matmul_combine_sync_semaphore_id =
        get_named_compile_time_arg_val("matmul_combine_sync_semaphore_id");

    // When compute_only=1, the fused selective_reduce_combine path is bypassed: no combine kernels
    // run on the combine cores, but the combine cores' L1 IS still allocated (the matmul output
    // tensor is sharded across the entire compute grid, including combine cores). dm1 still issues
    // its NOC writes to combine-core L1 because the unit test reads slot 4 back via
    // prepare_output_tensor_from_combine_writer. What IS gated off in compute_only: the
    // matmul<->combine semaphore wait/inc (no consumer to coordinate with).
    constexpr bool compute_only = get_named_compile_time_arg_val("compute_only") == 1;

    // LOCAL_OUTPUT (a cluster axis of extent 1): no combine kernels. Each token row slice this core
    // produces goes straight into the final [k, T, H] row-major output, page k * T + t, where (t, k)
    // are the expert's e_t entries (word 0 token id, word 1 k slot) that the tilize drain publishes
    // before it releases the matmul cores; dm1 fetches one chunk's entries per chunk. Only the rows
    // of the experts this device holds are written by default; explicit zero_fill also clears other rows.
    constexpr bool local_output = get_named_compile_time_arg_val("local_output") == 1;
    // The matmul<->combine handshake exists only when combine kernels are built.
    constexpr bool has_combine = !compute_only && !local_output;

    // Posted writes for matmul->combine output: in production, the matmul<->combine semaphore
    // handshake (noc_async_write barrier of `combine_semaphore_inc`) provides receiver-side
    // ordering, so posted writes are safe + faster. In compute_only there is no consumer to
    // coordinate with, and on Blackhole the host can read matmul_output_tensor before posted
    // writes have committed in destination L1 -> uninitialized bf16 -> NaN/Inf in PCC.
    // Use non-posted writes + ACK barrier in compute_only to guarantee destination commit.
    constexpr bool kPostedWrite = has_combine;

    std::array<uint32_t, 2 * height_shard_dim * width_shard_dim> output_shard_core_map = OUTPUT_SHARD_CORE_MAP;

    constexpr auto w0_w1_args = TensorAccessorArgs<0>();
    constexpr auto w2_args = TensorAccessorArgs<w0_w1_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto out_args = TensorAccessorArgs<w2_args.next_compile_time_args_offset()>();
    using LoArgs = LocalOutputArgs<local_output, out_args.next_compile_time_args_offset()>;

    // Run-time arguments
    uint32_t argidx = 0;
    [[maybe_unused]] const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    [[maybe_unused]] const auto w0_w1_addr = get_arg_val<uint32_t>(argidx++);
    [[maybe_unused]] const auto w2_addr = get_arg_val<uint32_t>(argidx++);
    [[maybe_unused]] const auto out_addr = get_arg_val<uint32_t>(argidx++);
    const auto ring_semaphore_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_core_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_neighbor_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto ring_neighbor_physical_y = get_arg_val<uint32_t>(argidx++);
    const auto ring_index = get_arg_val<uint32_t>(argidx++);
    const auto ring_predecessor_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto ring_predecessor_physical_y = get_arg_val<uint32_t>(argidx++);
    // LOCAL_OUTPUT: the final output and e_t addresses trail dm0's num_banks-entry shard_to_bank table.
    uint32_t local_output_addr = 0;
    uint32_t e_t_addr = 0;
    if constexpr (local_output) {
        constexpr uint32_t num_banks = get_named_compile_time_arg_val("num_banks");
        argidx += num_banks;  // dm0's shard_to_bank table
        local_output_addr = get_arg_val<uint32_t>(argidx++);
        e_t_addr = get_arg_val<uint32_t>(argidx++);
    }

    Noc noc_obj(noc_index);
    Noc noc1_obj(1);

    // CBs
    constexpr auto cb_s2c_in_id = tt::CBIndex::c_0;     // tilize_output_cb_id
    constexpr auto cb_r2c_w0_w1_id = tt::CBIndex::c_3;  // cb_r2c_w0
    constexpr auto cb_c2w_rdy_id = tt::CBIndex::c_4;
    constexpr auto cb_w2c_rdy_id = tt::CBIndex::c_5;
    constexpr auto cb_s2c_in2_id = tt::CBIndex::c_6;
    constexpr auto cb_w2c_md_id = tt::CBIndex::c_7;

    // CB Aliases
    constexpr auto cb_c2s_out_id = tt::CBIndex::c_1;  // matmul_writer_cb_id
    constexpr auto cb_r2c_w2_id = tt::CBIndex::c_3;   // reuse cb_r2c_w0_w1

    // CircularBuffer typed wrappers
    CircularBuffer cb_s2c_in(cb_s2c_in_id);
    CircularBuffer cb_c2w_rdy(cb_c2w_rdy_id);
    CircularBuffer cb_w2c_rdy(cb_w2c_rdy_id);
    CircularBuffer cb_s2c_in2(cb_s2c_in2_id);
    CircularBuffer cb_w2c_md(cb_w2c_md_id);
    CircularBuffer cb_c2s_out(cb_c2s_out_id);
    CircularBuffer cb_per_expert_total_tokens(per_expert_total_tokens_cb_id);

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in_id);
    constexpr uint32_t w0_w1_tile_size = get_tile_size(cb_r2c_w0_w1_id);
    constexpr uint32_t w2_tile_size = get_tile_size(cb_r2c_w2_id);
    constexpr uint32_t in2_tile_size = get_tile_size(cb_s2c_in2_id);

    // Pre-computed shard lookup tables — same LUT definitions as compute.cpp.
    constexpr auto shard_tiles_lut = moe_ring::make_shard_lut<Nt, num_cores>();
    constexpr auto w2_shard_tiles_lut = moe_ring::make_w2_shard_lut<Ht, Nt, num_cores>();
    constexpr auto w2_offset_lut = moe_ring::make_w2_offset_lut<Ht, Nt, num_cores>();

    // Constants for MoE — derived from compile-time shape args
    constexpr uint32_t num_w0_w1_tiles_h = Ht;
    [[maybe_unused]] const uint32_t num_w0_w1_tiles_w = shard_tiles_lut[ring_core_id];
    [[maybe_unused]] const uint32_t num_w2_tiles_w = w2_shard_tiles_lut[ring_core_id];

    using Cfg = moe_ring::MoeRingConfig<Ht, Nt, num_cores, has_bias>;

    // constants needed for writing to combine sharded output
    constexpr uint32_t shard_offset_per_expert_bytes =
        token_expert_row_offset * combine_shard_width_tiles * tile_width_size_bytes;
    cb_s2c_in.reserve_back(1);
    const uint32_t output_base_l1_addr = cb_s2c_in.get_write_ptr();
    cb_s2c_in.push_back(1);
    constexpr uint32_t source_width_tiles = Cfg::w2_tiles_per_expert_w;
    const uint32_t output_width_tiles_core = w2_shard_tiles_lut[ring_core_id];
    const uint32_t width_tile_base = w2_offset_lut[ring_core_id];
    constexpr uint32_t RING_CORES_PER_COMBINE_COL = num_cores / width_shard_dim;
    const uint32_t combine_core_x = ring_core_id / RING_CORES_PER_COMBINE_COL;
    Semaphore<> combine_sem(matmul_combine_sync_semaphore_id);
    // Device 2.0 migration: legacy primitive retained: raw L1 semaphore address used as the
    // base for multicast destination addresses (safe_get_noc_addr below)
    const auto combine_semaphore_addr = get_semaphore(matmul_combine_sync_semaphore_id);

    // LOCAL_OUTPUT geometry: this core's slice of every token row is output_width_tiles_core tiles of
    // the hidden dim starting at tile width_tile_base; the map scratch holds one chunk of the expert's
    // packed (k slot, token id) entries (moe_ring::token_list: one DRAM page, `token_list_header_words`
    // of segment starts, then every expert's entries back to back; the segment starts follow from the
    // per-expert counts, so they are recomputed here instead of read).
    [[maybe_unused]] constexpr uint32_t total_tokens = get_named_compile_time_arg_val("total_tokens");
    [[maybe_unused]] constexpr uint32_t token_list_header_words = moe_ring::token_list::header_words(num_experts);
    [[maybe_unused]] constexpr uint32_t token_list_chunk_bytes = tile_height * moe_ring::token_list::ENTRY_BYTES;
    [[maybe_unused]] const uint32_t local_output_row_bytes = output_width_tiles_core * tile_width_size_bytes;
    [[maybe_unused]] const uint32_t local_output_col_offset_bytes = width_tile_base * tile_width_size_bytes;
    uint32_t local_output_map_addr = 0;
    if constexpr (local_output) {
        // The map is the L1 destination of a DRAM read: DRAM-aligned inside the CB (the CB has the slack).
        constexpr uint32_t dram_alignment = get_named_compile_time_arg_val("dram_alignment");
        CircularBuffer cb_local_output_map(get_named_compile_time_arg_val("local_output_map_cb_id"));
        cb_local_output_map.reserve_back(1);
        local_output_map_addr = (cb_local_output_map.get_write_ptr() + dram_alignment - 1) & ~(dram_alignment - 1);
    }

    // LOCAL_OUTPUT zero fill. The output contract is "every row of [k, T, H] is what this op
    // wrote": the rows of the experts this device holds carry their W2 results and every other row
    // is zero, so the per-device partials of a 1xN mesh sum directly, also into a reused caller
    // tensor. Before the tilize release (metadata_ready below) this core writes its column slice of
    // all k x T rows from a pre-zeroed L1 row slice: one non-posted write per row on the same NoC
    // as the later row writes, issued while dm1 would otherwise wait for the tilize phase. The
    // write barrier before the first owned-row write (local_output_fill_pending) orders the fill
    // ahead of the rows that overwrite it; both come from this core, so the barrier is the order.
    // zero_fill=0 (the caller's choice on this path) leaves the unowned rows as the buffer holds them.
    constexpr bool zero_fill = get_named_compile_time_arg_val("zero_fill") == 1;
    bool local_output_fill_pending = false;
    if constexpr (zero_fill) {
        constexpr uint32_t local_output_num_rows = get_named_compile_time_arg_val("local_output_num_rows");
        CircularBuffer cb_local_output_zero(get_named_compile_time_arg_val("local_output_zero_cb_id"));
        cb_local_output_zero.reserve_back(1);
        const uint32_t zero_addr = cb_local_output_zero.get_write_ptr();
        volatile tt_l1_ptr uint32_t* zero_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_addr);
        for (uint32_t i = 0; i < local_output_row_bytes / sizeof(uint32_t); ++i) {
            zero_ptr[i] = 0;
        }
        const auto output_accessor = TensorAccessor(LoArgs::out_args, local_output_addr);
        for (uint32_t row = 0; row < local_output_num_rows; ++row) {
            noc_async_write(
                zero_addr,
                output_accessor.get_noc_addr(row, local_output_col_offset_bytes, /*noc=*/1),
                local_output_row_bytes,
                /*noc=*/1);
        }
        local_output_fill_pending = local_output_num_rows > 0;
    }

    //-------------------------------------------------------------------------
    // Ring setup
    //-------------------------------------------------------------------------
    constexpr uint32_t num_a2a_steps_per_iter = num_cores;

    constexpr uint32_t tiles_per_step = Cfg::in2_tiles_per_step;

    //-------------------------------------------------------------------------
    // Ring NoC setup
    //-------------------------------------------------------------------------
    Semaphore<> ring_sem(ring_semaphore_id);
    uint32_t semaphore_addr = get_semaphore(ring_semaphore_id);
    // Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address
    const uint64_t neighbor_semaphore_noc_addr =
        get_noc_addr(ring_neighbor_physical_x, ring_neighbor_physical_y, semaphore_addr);

    // Size of each transfer in bytes
    constexpr uint32_t a2a_xfer_bytes_per_step = tiles_per_step * in2_tile_size;

    // Split each A2A transfer into max-burst packets plus a smaller remainder.
    constexpr uint32_t noc_max_burst_bytes = get_named_compile_time_arg_val("noc_max_burst_bytes");
    constexpr uint32_t max_tiles_per_burst = noc_max_burst_bytes / in2_tile_size;
    constexpr uint32_t a2a_full_packets = tiles_per_step / max_tiles_per_burst;
    constexpr uint32_t a2a_full_packet_size = max_tiles_per_burst * in2_tile_size;
    constexpr uint32_t a2a_remainder_tiles = tiles_per_step % max_tiles_per_burst;
    constexpr uint32_t a2a_remainder_size = a2a_remainder_tiles * in2_tile_size;

    // Source and destination addresses for the all2all
    const uint32_t local_base_addr = cb_s2c_in2.get_write_ptr();
    // Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address
    const uint64_t neighbor_base_addr =
        get_noc_addr(ring_neighbor_physical_x, ring_neighbor_physical_y, local_base_addr);

    // Precompute buffer offsets (parity slot 0; the a2a pipeline puts chunk c's buffers in slot c % 2, a whole set of
    // num_cores buffers further on, see moe_ring::a2a_pipeline)
    constexpr uint32_t a2a_pipeline = get_named_compile_time_arg_val("a2a_pipeline");
    constexpr uint32_t a2a_parity_bytes = num_a2a_steps_per_iter * a2a_xfer_bytes_per_step;
    uint32_t LOCAL_BUFFER_OFFSET[num_a2a_steps_per_iter];
    for (uint32_t i = 0; i < num_a2a_steps_per_iter; ++i) {
        LOCAL_BUFFER_OFFSET[i] = local_base_addr + i * a2a_xfer_bytes_per_step;
    }
    uint32_t semaphore_value = 0;

    // The exchange's backpressure. Buffer s of a core holds the partial of the core s hops back; the predecessor
    // writes buffers 1..num_cores-1 (its step s lands in buffer s + 1) and the core's own compute writes buffer 0.
    // Nothing in the ring semaphore tells the predecessor when this core has READ a buffer, so before the fix the
    // predecessor's writes of the next chunk (and its redundant last-step write of this core's own partial back into
    // buffer 0) could land while this core still read the previous chunk's buffers or had already packed the next
    // chunk's partial -- a race whose slack was the predecessor's inter-chunk gap (measured: one lagging core
    // corrupts every chunk once its lag exceeds that gap). Now: (1) the last step sends no data (buffer 0 is the
    // core's own, written by its compute alone; the step's semaphore increment stays, the successor's wait counts
    // it); (2) each core credits its PREDECESSOR once per owned chunk when its compute has read every a2a buffer of
    // that chunk (the W2 output is packed), and a core waits for its successor's credits before its first a2a write
    // of its next owned chunk. Both cores of a link skip the same chunks (one owner table per ring), so the counts
    // agree. The credit also certifies that this core's dm1 has finished READING its buffers as the sources of its
    // own forwards: every step ends in a posted-writes flush (the writes have left the core), so by the time the
    // credit is sent no forward of this chunk still reads a buffer -- that flush is load-bearing for the credit.
    Semaphore<> a2a_free_sem(a2a_free_semaphore_id);
    // Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address
    const uint64_t predecessor_a2a_free_noc_addr =
        get_noc_addr(ring_predecessor_physical_x, ring_predecessor_physical_y, get_semaphore(a2a_free_semaphore_id));
    uint32_t a2a_chunks_exchanged = 0;

    //-------------------------------------------------------------------------
    // Init synchronization with tilize cores
    //-------------------------------------------------------------------------

    // Receive number of tokens per expert from the tilize cores
    Semaphore<> metadata_ready_sem(metadata_ready_semaphore_id);
    metadata_ready_sem.wait_min(1);

    // Signal to the compute core that num_tokens_per_expert has arrived.
    // We also use this CB to transfer (from the writer to compute) 2 semaphore addresses:
    // - 0: address of L1 page (CB) used to send metadata (number of tokens per expert)
    // - 1: address of semaphore used to notify matmuls cores that tilized chunks have arrived

    // Read per-expert token counts from CB
    const auto num_tokens_per_expert_addr = cb_per_expert_total_tokens.get_read_ptr();
    volatile tt_l1_ptr uint32_t* num_tokens_per_expert_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(num_tokens_per_expert_addr);

    cb_w2c_md.reserve_back(2);
    volatile tt_l1_ptr uint32_t* cb_w2c_md_write_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_w2c_md.get_write_ptr());
    cb_w2c_md_write_ptr[0] = num_tokens_per_expert_addr;
    cb_w2c_md_write_ptr[1] = get_semaphore(matmul_chunk_ready_semaphore_id);
    cb_w2c_md.push_back(2);

    // Precompute NUM_CHUNKS_PER_EXPERT
    uint32_t NUM_TOKENS_PER_EXPERT[num_experts];
    uint32_t NUM_CHUNKS_PER_EXPERT[num_experts];
    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        uint32_t num_tokens = num_tokens_per_expert_ptr[expert_id];
        NUM_TOKENS_PER_EXPERT[expert_id] = num_tokens;
        NUM_CHUNKS_PER_EXPERT[expert_id] = moe_ring::detail::div_up(num_tokens, tokens_per_chunk);
    }

    // The drain's credit semaphore of each chunk half: consuming chunk g frees half g % chunk_halves.
    // Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC addresses
    uint64_t half_free_semaphore_noc_addr[moe_ring::rings::MAX_CHUNK_HALVES];
    for (uint32_t h = 0; h < chunk_halves; ++h) {
        half_free_semaphore_noc_addr[h] =
            get_noc_addr(tilize_drain_core_noc_x, tilize_drain_core_noc_y, get_semaphore(half_free_semaphore_ids[h]));
    }
    uint32_t chunk_index = 0;  // over all experts, in feed order (owned or not)
    moe_ring::rings::ChunkOwners<num_rings> owners;

    // Signal to combine cores that chunk is available
    auto combine_semaphore_inc = [&](const uint32_t inc = 1) {
        for (uint32_t y = 0; y < height_shard_dim; ++y) {
            const uint32_t idx = combine_core_x + y * width_shard_dim;
            const uint64_t dest_sem_noc_addr = safe_get_noc_addr(
                output_shard_core_map[2 * idx],
                output_shard_core_map[2 * idx + 1],
                combine_semaphore_addr,
                /*noc_id=*/1);
            noc_semaphore_inc</*posted=*/true>(dest_sem_noc_addr, inc, /*noc_id=*/1, vchannel);
        };
        noc1_obj.async_writes_flushed<NocOptions::POSTED>();
    };

    //-------------------------------------------------------------------------
    // Expert loop
    //-------------------------------------------------------------------------
    bool output_buffer_idx = 0;
    // both sections of the double buffer are initially free
    uint32_t combine_semaphore_val = 0;
    // LOCAL_OUTPUT: start of the current expert's segment in the packed token lists (entries).
    [[maybe_unused]] uint32_t token_list_segment_start = 0;
    // The a2a pipeline (a2a_pipeline): dm1 exchanges chunk c+1 as soon as compute has its partial (rdy) and
    // finishes chunk c afterwards -- waits for its W2 output, credits the predecessor, writes its rows, frees its
    // half. The serial form finishes each chunk right after its exchange. A foreign chunk's empty pop and the
    // end of the loop finish the pending chunk first, so the output CB stays in step with compute's pushes; an
    // expert's combine handshake (the per-expert increment) follows that expert's last rows in the same stream.
    struct PendingChunk {
        bool valid;
        uint32_t chunk_g;
        uint32_t chunk;
        uint32_t num_tokens_block;
        uint32_t output_buffer_offset_bytes;
        uint32_t tokens_per_height_shard_chunk;
        uint32_t tokens_per_height_shard_rem;
        uint32_t map_addr;
        bool zone_on;
        uint32_t epilogues;
    };
    PendingChunk pending{};
    // The e_t map scratch holds two chunks under the pipeline (chunk c+1's read lands while chunk c's rows read
    // theirs).
    [[maybe_unused]] constexpr uint32_t map_dram_alignment = get_named_compile_time_arg_val("dram_alignment");
    [[maybe_unused]] constexpr uint32_t map_slot_bytes =
        ((token_list_chunk_bytes + map_dram_alignment - 1) / map_dram_alignment) * map_dram_alignment;
    // The staging-ring walk's running position (per expert; every chunk of an expert is this ring's on that path).
    [[maybe_unused]] uint32_t dest_height_shard_start = 0;
    [[maybe_unused]] uint32_t shard_row_start = 0;
    auto finish_chunk = [&](const PendingChunk& p) {
        const uint32_t chunk_g = p.chunk_g;
        const uint32_t chunk = p.chunk;
        const uint32_t num_tokens_block = p.num_tokens_block;
        [[maybe_unused]] const uint32_t output_buffer_offset_bytes = p.output_buffer_offset_bytes;
        [[maybe_unused]] const uint32_t tokens_per_height_shard_chunk = p.tokens_per_height_shard_chunk;
        [[maybe_unused]] const uint32_t tokens_per_height_shard_rem = p.tokens_per_height_shard_rem;
        [[maybe_unused]] const bool zone_on = p.zone_on;
        MOE_ZONE_IF(zone_on, "mz_o_finish");
        if (chunk == 0) {
            dest_height_shard_start = 0;
            shard_row_start = 0;
        }
        MOE_STUDY_DELAY(O_BEFORE_ROWS);
        cb_c2s_out.wait_front(num_w0_w1_tiles_h);
        // Compute packed this chunk's W2 output: every a2a buffer of the chunk has been read (and this core's own
        // forwards of them have left the core, see the flush above). Credit the predecessor. Non-posted so that a
        // response exists (Blackhole makes every atomic non-posted anyway; Wormhole does not): the exit barriers
        // below wait for it, so no credit can land after the next launch has re-initialised the semaphore.
        // Device 2.0 migration: legacy primitive retained: a precomposed uint64_t NoC address cannot be wrapped by
        // Semaphore<>::inc
        noc_semaphore_inc</*posted=*/false>(predecessor_a2a_free_noc_addr, /*incr=*/1, /*noc_id=*/1, vchannel);

        const uint32_t source_base_l1_addr = cb_c2s_out.get_read_ptr();
        [[maybe_unused]] const uint32_t elts_per_page = source_width_tiles * tile_width;

        if constexpr (local_output) {
            // Final output rows: row bt of this chunk belongs to token (t, k) of the expert's e_t
            // page; this core's slice of it lands in page k * T + t of [k, T, H] at the core's
            // column offset. Non-posted writes; the flush before pop_front below frees the source.
            if (local_output_fill_pending) {
                // The zero fill of these same bytes must be committed before the rows overwrite it.
                noc1_obj.async_write_barrier();
                local_output_fill_pending = false;
            }
            noc_async_read_barrier(/*noc=*/1);
            volatile tt_l1_ptr uint32_t* map = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(p.map_addr);
            const auto output_accessor = TensorAccessor(LoArgs::out_args, local_output_addr);
            for (uint32_t bt = 0; bt < num_tokens_block; ++bt) {
                const uint32_t entry = map[bt];
                const uint32_t token_id = moe_ring::token_list::entry_token(entry);
                const uint32_t k_slot = moe_ring::token_list::entry_k_slot(entry);
                const uint64_t dst_noc_addr = output_accessor.get_noc_addr(
                    k_slot * total_tokens + token_id, local_output_col_offset_bytes, /*noc=*/1);
                noc_async_write(
                    source_base_l1_addr + bt * source_width_tiles * tile_width_size_bytes,
                    dst_noc_addr,
                    local_output_row_bytes,
                    /*noc=*/1);
            }
        } else {
            // Staging ring rows for the fused combine (or the kernel-less combine cores' L1 in
            // compute_only).
            uint32_t width_tiles_to_send = output_width_tiles_core;  // split width of hidden dim, maybe padded
            uint32_t width_tiles_sent = 0;

            while (width_tiles_to_send > 0) {
                const uint32_t width_tile_start = width_tile_base + width_tiles_sent;
                const uint32_t dest_width_shard = width_tile_start / combine_shard_width_tiles;
                const uint32_t dest_width_offset_tiles = width_tile_start % combine_shard_width_tiles;

                const uint32_t dest_width_offset_bytes = dest_width_offset_tiles * tile_width_size_bytes;

                const uint32_t width_transfer_tiles = std::min(
                    combine_shard_width_tiles - dest_width_offset_tiles, output_width_tiles_core - width_tiles_sent);
                const uint32_t width_transfer_bytes = width_transfer_tiles * tile_width_size_bytes;

                // In production: at each expert's first chunk, wait for combine to signal that the
                // buffer segment is available. The wait also acts as an implicit barrier between
                // experts -- `noc_async_write_one_packet_set_state` sets a global size state for
                // subsequent posted writes, and consecutive experts may use different
                // `width_transfer_bytes`. Without the inter-expert barrier, queued writes from a
                // prior expert could be issued with the next expert's state.
                // In compute_only there's no consumer to wait for, so we explicitly flush previous
                // chunk's writes before reissuing set_state for this chunk.
                if constexpr (compute_only) {
                    noc1_obj.async_writes_flushed();  // non-posted in compute_only; use NON-posted flush API
                } else if (chunk == 0) {
                    combine_sem.wait(combine_semaphore_val);
                }

                uint32_t dest_height_shard = dest_height_shard_start;
                uint32_t shard_row = shard_row_start;
                for (uint32_t bt = 0; bt < num_tokens_block; ++bt) {
                    const uint32_t shard_row_offset_bytes =
                        shard_row * combine_shard_width_tiles * tile_width_size_bytes;

                    const auto dest_noc_x =
                        output_shard_core_map[2 * (dest_height_shard * width_shard_dim + dest_width_shard)];
                    const auto dest_noc_y =
                        output_shard_core_map[2 * (dest_height_shard * width_shard_dim + dest_width_shard) + 1];

                    // Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address
                    // used as state-machine base
                    const uint64_t dest_noc_addr_base = get_noc_addr(dest_noc_x, dest_noc_y, output_base_l1_addr, 1);
                    // Device 2.0 migration: legacy primitive retained: state-machine setup
                    // (noc_async_write_one_packet_set_state) has no Device 2.0 wrapper
                    noc_async_write_one_packet_set_state</*posted=*/kPostedWrite>(
                        dest_noc_addr_base, width_transfer_bytes, /*noc=*/1, vchannel);

                    const uint32_t dest_l1_addr = output_base_l1_addr + output_buffer_offset_bytes +
                                                  dest_width_offset_bytes + shard_row_offset_bytes;

                    const uint32_t source_l1_addr =
                        source_base_l1_addr + (bt * source_width_tiles + width_tiles_sent) * tile_width_size_bytes;

                    // Device 2.0 migration: legacy primitive retained: paired with
                    // noc_async_write_one_packet_set_state above
                    noc_async_write_one_packet_with_state</*posted=*/kPostedWrite>(source_l1_addr, dest_l1_addr);

                    if (++shard_row == ((dest_height_shard < tokens_per_height_shard_rem)
                                            ? tokens_per_height_shard_chunk + 1
                                            : tokens_per_height_shard_chunk)) {
                        ++dest_height_shard;
                        shard_row = 0;
                    }
                }
                width_tiles_sent += width_transfer_tiles;
                width_tiles_to_send -= width_transfer_tiles;

                if (width_tiles_to_send == 0) {
                    dest_height_shard_start = dest_height_shard;
                    shard_row_start = shard_row;
                }
            }
        }  // staging ring rows

        // Source CB recycle barrier: must wait for NIU to finish READING source L1 before
        // cb_pop_front recycles those pages. compute_only and local_output use non-posted
        // writes (kPostedWrite=false), so the posted-write counter is 0 -> posted-flush is a
        // no-op and cb_pop_front would race with in-flight reads -> source clobber.
        if constexpr (!has_combine) {
            noc1_obj.async_writes_flushed();  // non-posted: flush issuer queue for non-posted writes
        } else {
            noc1_obj.async_writes_flushed<NocOptions::POSTED>();  // production: original posted flush
        }
        cb_c2s_out.pop_front(num_w0_w1_tiles_h);

        // Credit this chunk's half to the drain: it may send another chunk into it
        // Device 2.0 migration: legacy primitive retained: a precomposed uint64_t NoC address cannot be wrapped by
        // Semaphore<>::inc
        MOE_STUDY_DELAY(O_BEFORE_CREDIT);
        noc_semaphore_inc</*posted=*/true>(
            half_free_semaphore_noc_addr[chunk_g % chunk_halves], /*incr=*/1, /*noc_id=*/1, /*vc=*/vchannel);
        for (uint32_t k = 0; k < p.epilogues; ++k) {
            if constexpr (has_combine) {
                combine_semaphore_inc();
                combine_semaphore_val += height_shard_dim;
            }
        }
    };
    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        const uint32_t num_expert_chunks = NUM_CHUNKS_PER_EXPERT[expert_id];
        const uint32_t active_tokens = NUM_TOKENS_PER_EXPERT[expert_id];
        owners.begin_expert(num_expert_chunks);

        // Zero-token experts produce no chunks, so skip the whole per-expert
        // matmul<->combine handshake (buffer-free wait, combine_semaphore_inc, double-buffer
        // toggle). The combine writer skips these experts gated on the same tilize-produced
        // counts, so semaphore totals and double-buffer parity stay aligned. compute_only and
        // local_output have no combine writer: keep the toggle so host readback sees the same halves.
        if constexpr (has_combine) {
            if (active_tokens == 0) {
                continue;
            }
        }

        const uint32_t tokens_per_height_shard_chunk = active_tokens / height_shard_dim;
        const uint32_t tokens_per_height_shard_rem = active_tokens % height_shard_dim;
        const uint32_t output_buffer_offset_bytes = shard_offset_per_expert_bytes * output_buffer_idx;

        for (uint32_t chunk = 0; chunk < num_expert_chunks; ++chunk) {
            const uint32_t chunk_g = chunk_index++;
            if (owners.owner(chunk) != ring_index) {
                // Another ring's chunk: no a2a pass, no output rows, no credit from this core; the empty pop matches
                // compute's empty push so the output staging stays in step with the input halves (see compute.cpp).
                // (The combine and compute_only output walks below track rows per chunk; they are single-ring paths.)
                if constexpr (a2a_pipeline) {
                    if (pending.valid) {
                        finish_chunk(pending);
                        pending.valid = false;
                    }
                }
                cb_c2s_out.wait_front(num_w0_w1_tiles_h);
                cb_c2s_out.pop_front(num_w0_w1_tiles_h);
                continue;
            }
            const uint32_t num_tokens_block = std::min(tile_height, active_tokens - chunk * tile_height);
            // Study zones (MOE_ZONES): this owned chunk's a2a passes, output rows and credit
            const bool zone_on = moe_ring::zones::in_window(chunk_g);
            [[maybe_unused]] const uint32_t map_addr =
                local_output_map_addr + (a2a_pipeline ? (a2a_chunks_exchanged & 1u) * map_slot_bytes : 0u);
            MOE_ZONE_IF(zone_on, "mz_o_chunk");
            if constexpr (local_output) {
                // Fetch this chunk's packed (k slot, token id) entries: always a whole chunk (the page keeps
                // one chunk of tail past the last segment); the read completes under the ring A2A below
                // and is waited for right before the output writes.
                const auto e_t_accessor = TensorAccessor(LoArgs::e_t_args, e_t_addr);
                const uint32_t chunk_entry = token_list_header_words + token_list_segment_start + chunk * tile_height;
                noc_async_read(
                    e_t_accessor.get_noc_addr(0, chunk_entry * moe_ring::token_list::ENTRY_BYTES, /*noc=*/1),
                    map_addr,
                    token_list_chunk_bytes,
                    /*noc=*/1);
            }

            // Device 2.0 migration: legacy primitives retained: state-machine setup
            // (noc_async_write_one_packet_set_state, noc_inline_dw_write_set_state) has no
            // Device 2.0 wrappers
            if constexpr (a2a_full_packets == 0 || a2a_remainder_tiles == 0) {
                // Set only once here if there is only 1 type of packet: either all full with none partial, or none full
                // with one partial
                noc_async_write_one_packet_set_state</*posted=*/true>(
                    neighbor_base_addr,
                    a2a_full_packets > 0 ? a2a_full_packet_size : a2a_remainder_size,
                    /*noc=*/1,
                    vchannel);
            }
            // Set state for the semaphore write
#if !defined(ARCH_BLACKHOLE)
            // WH: keep original stateful path (BH does NOT support stateful inline-write to L1
            // per dataflow_api.h:2140,2181 -- handled in the per-iteration block below).
            noc_inline_dw_write_set_state</*posted=*/true, /*set_val=*/false>(
                neighbor_semaphore_noc_addr,
                /*val=*/0,
                /*be=*/0xF,
                /*cmd_buf=*/write_at_cmd_buf,
                /*noc=*/1,
                vchannel);
#endif

            // Wait for compute core to tell us that all mm01 data is ready
            {
                MOE_ZONE_IF(zone_on, "mz_o_wait_rdy");
                cb_c2w_rdy.wait_front(1);
            }
            cb_c2w_rdy.pop_front(1);

            // The successor has read every a2a buffer of the previous chunk (its credit): this chunk's writes may
            // land in its buffers. Idle in the steady state (the credit arrives during this core's W2 tail and W0/W1).
            // With the pipeline the chunk's buffers are the parity slot the successor last read two chunks ago: one
            // chunk's credit may still be outstanding.
            constexpr uint32_t credit_lag = a2a_pipeline ? 1u : 0u;
            if (a2a_chunks_exchanged > credit_lag) {
                MOE_ZONE_IF(zone_on, "mz_o_a2a_free");
                a2a_free_sem.wait_min(a2a_chunks_exchanged - credit_lag);
            }
            const uint32_t parity_offset = a2a_pipeline ? (a2a_chunks_exchanged & 1u) * a2a_parity_bytes : 0u;

            // Take the data in cb_s2c_in2 and send it to the next core in the ring
            // Ring synchronization: all cores participate regardless of whether they had CB work. The ring runs
            // moe_ring::a2a_handshake_iters of the compute's Cfg::num_a2a_iters W2 iterations: the partials travel
            // once per chunk (below), so the compute's later iterations read the resident buffers without it.
            for (uint32_t i = 0; i < moe_ring::a2a_handshake_iters; ++i) {
                for (uint32_t step = 0; step < num_a2a_steps_per_iter; ++step) {
                    if constexpr (MOE_STUDY_FAULT(O_A2A_LAG)) {
                        if (ring_core_id == MOE_STUDY_PARAM(O_A2A_LAG_POS)) {
                            MOE_STUDY_DELAY(O_A2A_LAG);
                        }
                    }
                    // Wait for current data to be ready in cb_s2c_in2
                    ring_sem.wait_min(semaphore_value);

                    // Signal to compute core that data is ready
                    cb_w2c_rdy.reserve_back(1);
                    cb_w2c_rdy.push_back(1);

                    // Write tiles from local cb_s2c_in2 to neighbor's cb_s2c_in2: buffer `step` (the partial of the
                    // core `step` hops back) into the neighbour's buffer `step + 1`. The last step would send the
                    // neighbour its OWN partial back into its buffer 0: that buffer is the neighbour's compute's
                    // alone (see the exchange's backpressure above), so the last step sends only its increment. The
                    // partials travel once per chunk: this first iteration fills buffers 1..num_cores-1 (buffer s at
                    // step s - 1 of the predecessor; buffer 0 is the core's own, its compute's), and the compute's
                    // later W2 iterations read the same buffers again. A further handshake iteration would send
                    // only its increments (the rdy cadence and the ring semaphore accounting of a dropped iteration
                    // disappear on every core alike). The per-step posted-writes flush below stays on every step:
                    // the credit rests on it (the forwards have left the core before the chunk's credit is sent).
                    if (i == 0 && step != num_cores - 1) {
                        if constexpr (a2a_full_packets > 0 && a2a_remainder_tiles > 0) {
                            // Resetting required as both full and partial packets exist (only the data-carrying
                            // steps use the state)
                            // Device 2.0 migration: legacy primitive retained: state-machine setup
                            // (noc_async_write_one_packet_set_state) has no Device 2.0 wrapper
                            noc_async_write_one_packet_set_state</*posted=*/true>(
                                neighbor_base_addr, a2a_full_packet_size, /*noc=*/1, vchannel);
                        }
                        const uint32_t local_src_addr = LOCAL_BUFFER_OFFSET[step] + parity_offset;
                        const uint64_t neighbor_dst_addr = LOCAL_BUFFER_OFFSET[step + 1] + parity_offset;

                        uint32_t pkt_offset = 0;
                        // Rely on compiler to remove loop if no full packet exists
                        for (uint32_t pkt = 0; pkt < a2a_full_packets; ++pkt) {
                            noc_async_write_one_packet_with_state</*posted=*/true>(
                                local_src_addr + pkt_offset, neighbor_dst_addr + pkt_offset);
                            pkt_offset += a2a_full_packet_size;
                        }
                        if constexpr (a2a_remainder_tiles > 0) {
                            if constexpr (a2a_full_packets > 0) {
                                // Reset here if full packets exist, otherwise, it was already set once at the top and
                                // no reset required
                                noc_async_write_one_packet_set_state</*posted=*/true>(
                                    neighbor_base_addr, a2a_remainder_size, /*noc=*/1, vchannel);
                            }
                            noc_async_write_one_packet_with_state</*posted=*/true>(
                                local_src_addr + pkt_offset, neighbor_dst_addr + pkt_offset);
                        }
                    }

                    // Signal neighbor that data is ready (increment their semaphore value).
                    // Receiver waits via Semaphore<>::wait_min(semaphore_value); both arches
                    // advance `semaphore_value` by 1 here, just by different mechanisms.
#if defined(ARCH_BLACKHOLE)
                    // BH-safe: noc_inline_dw_write_with_state to L1 hangs on BH
                    // (dataflow_api.h:2140,2181). Use atomic-increment pattern instead;
                    // receiver-side wait condition is value-equivalent.
                    // Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address
                    // (neighbor_semaphore_noc_addr) cannot be wrapped by Semaphore<>::inc
                    MOE_STUDY_DELAY(O_A2A_BEFORE_INC);
                    noc_semaphore_inc</*posted=*/true>(neighbor_semaphore_noc_addr, /*incr=*/1, /*noc_id=*/1, vchannel);
                    ++semaphore_value;
#else
                    // WH: original stateful path.
                    // Device 2.0 migration: legacy primitive retained: paired with
                    // noc_inline_dw_write_set_state above
                    noc_inline_dw_write_with_state<
                        /*update_addr_lo=*/false,
                        /*update_counter=*/true,
                        /*posted=*/true,
                        /*update_addr_hi=*/false,
                        /*update_val=*/true>(++semaphore_value);
#endif

                    // Ensure writes have left the core before continuing
                    noc1_obj.async_writes_flushed<NocOptions::POSTED>();
                }
            }
            ++a2a_chunks_exchanged;

            const PendingChunk this_chunk{
                true,
                chunk_g,
                chunk,
                num_tokens_block,
                output_buffer_offset_bytes,
                tokens_per_height_shard_chunk,
                tokens_per_height_shard_rem,
                map_addr,
                zone_on,
                0u};
            if constexpr (a2a_pipeline) {
                if (pending.valid) {
                    finish_chunk(pending);
                }
                pending = this_chunk;
            } else {
                finish_chunk(this_chunk);
            }
        }
        if constexpr (has_combine) {
            if (a2a_pipeline != 0 && pending.valid) {
                ++pending.epilogues;  // after that chunk's rows, in the rows stream
            } else {
                combine_semaphore_inc();
                combine_semaphore_val += height_shard_dim;
            }
        }
        // (compute_only branch: nothing to do -- the next expert's first chunk flushes any
        //  in-flight writes via the inter-chunk flush before its set_state. Output buffer
        //  toggle below picks the other half so there's no destination overlap either.
        //  local_output: every chunk's writes were flushed before pop_front; the commit of the
        //  output is waited for once at the end.)
        output_buffer_idx = !output_buffer_idx;
        if constexpr (local_output) {
            // Same segment rule as the tilize drain that packs the lists.
            token_list_segment_start =
                moe_ring::token_list::next_segment_start(token_list_segment_start, active_tokens);
        }
    }

    if constexpr (a2a_pipeline) {
        if (pending.valid) {
            finish_chunk(pending);
            pending.valid = false;
        }
    }

    if constexpr (!has_combine) {
        // Non-posted writes need the ACK barrier (not just the issuer-queue flush) so the
        // destination (compute_only: the matmul output L1; local_output: the final output tensor)
        // is committed before the host or the next op reads it.
        noc1_obj.async_full_barrier();
    } else {
        // wait for combine to do its final semaphore increment before resetting. Otherwise, leads to hang.
        combine_sem.wait(combine_semaphore_val);
        combine_sem.set(0);
        noc1_obj.async_writes_flushed<NocOptions::POSTED>();
        // the a2a credits are non-posted increments: their responses must be back before the exit
        noc1_obj.async_atomic_barrier();
    }
}
