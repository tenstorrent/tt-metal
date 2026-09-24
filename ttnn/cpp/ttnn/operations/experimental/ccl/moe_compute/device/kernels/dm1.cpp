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
    constexpr uint32_t matmul_chunk_available_semaphore_id =
        get_named_compile_time_arg_val("matmul_chunk_available_semaphore_id");
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
    // before it releases the matmul cores; dm1 fetches one chunk's entries per chunk. The rows of
    // experts this device does not hold are written as zero first (see the zero fill below), so
    // every row of the output is what this op wrote.
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
    // LOCAL_OUTPUT: the final output and e_t addresses trail dm0's num_banks-entry shard_to_bank table.
    uint32_t local_output_addr = 0;
    uint32_t e_t_addr = 0;
    if constexpr (local_output) {
        constexpr uint32_t num_banks = get_named_compile_time_arg_val("num_banks");
        argidx += num_banks;
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
    // the hidden dim starting at tile width_tile_base; the map scratch holds one e_t entry per token
    // of the current chunk.
    [[maybe_unused]] constexpr uint32_t total_tokens = get_named_compile_time_arg_val("total_tokens");
    [[maybe_unused]] constexpr uint32_t e_t_entry_size = get_named_compile_time_arg_val("e_t_entry_size");
    [[maybe_unused]] constexpr uint32_t e_t_entry_words = e_t_entry_size / sizeof(uint32_t);
    [[maybe_unused]] const uint32_t local_output_row_bytes = output_width_tiles_core * tile_width_size_bytes;
    [[maybe_unused]] const uint32_t local_output_col_offset_bytes = width_tile_base * tile_width_size_bytes;
    uint32_t local_output_map_addr = 0;
    if constexpr (local_output) {
        CircularBuffer cb_local_output_map(get_named_compile_time_arg_val("local_output_map_cb_id"));
        cb_local_output_map.reserve_back(1);
        local_output_map_addr = cb_local_output_map.get_write_ptr();
    }

    // LOCAL_OUTPUT zero fill. The output contract is "every row of [k, T, H] is what this op
    // wrote": the rows of the experts this device holds carry their W2 results and every other row
    // is zero, so the per-device partials of a 1xN mesh sum directly, also into a reused caller
    // tensor. Before the tilize release (metadata_ready below) this core writes its column slice of
    // all k x T rows from a pre-zeroed L1 row slice: one non-posted write per row on the same NoC
    // as the later row writes, issued while dm1 would otherwise wait for the tilize phase. The
    // write barrier before the first owned-row write (local_output_fill_pending) orders the fill
    // ahead of the rows that overwrite it; both come from this core, so the barrier is the order.
    bool local_output_fill_pending = false;
    if constexpr (local_output) {
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

    // Precompute buffer offsets
    uint32_t LOCAL_BUFFER_OFFSET[num_a2a_steps_per_iter];
    for (uint32_t i = 0; i < num_a2a_steps_per_iter; ++i) {
        LOCAL_BUFFER_OFFSET[i] = local_base_addr + i * a2a_xfer_bytes_per_step;
    }
    uint32_t semaphore_value = 0;

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

    // Tilize core we signal to that tilize cores can send another chunk of tiles
    // Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address
    uint64_t matmul_chunk_available_semaphore_noc_addr = get_noc_addr(
        tilize_drain_core_noc_x, tilize_drain_core_noc_y, get_semaphore(matmul_chunk_available_semaphore_id));

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
    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        const uint32_t num_expert_chunks = NUM_CHUNKS_PER_EXPERT[expert_id];
        const uint32_t active_tokens = NUM_TOKENS_PER_EXPERT[expert_id];
        const uint32_t tokens_per_height_shard_chunk = active_tokens / height_shard_dim;
        const uint32_t tokens_per_height_shard_rem = active_tokens % height_shard_dim;
        const uint32_t output_buffer_offset_bytes = shard_offset_per_expert_bytes * output_buffer_idx;

        uint32_t dest_height_shard_start = 0;
        uint32_t shard_row_start = 0;

        // required to prevent a race with combine writer (only relevant in production:
        // compute_only and local_output have no combine writer to coordinate with).
        if constexpr (has_combine) {
            if (num_expert_chunks == 0) {
                combine_sem.wait(combine_semaphore_val);
            }
        }

        for (uint32_t chunk = 0; chunk < num_expert_chunks; ++chunk) {
            const uint32_t num_tokens_block = std::min(tile_height, active_tokens - chunk * tile_height);
            if constexpr (local_output) {
                // Fetch this chunk's (token id, k slot) e_t entries; the read completes under the ring
                // A2A below and is waited for right before the output writes.
                const auto e_t_accessor = TensorAccessor(LoArgs::e_t_args, e_t_addr);
                noc_async_read(
                    e_t_accessor.get_noc_addr(expert_id, chunk * tile_height * e_t_entry_size, /*noc=*/1),
                    local_output_map_addr,
                    num_tokens_block * e_t_entry_size,
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
            cb_c2w_rdy.wait_front(1);
            cb_c2w_rdy.pop_front(1);

            // Take the data in cb_s2c_in2 and send it to the next core in the ring
            // Ring synchronization: all cores participate regardless of whether they had CB work
            for (uint32_t i = 0; i < Cfg::num_a2a_iters; ++i) {
                for (uint32_t step = 0; step < num_a2a_steps_per_iter; ++step) {
                    if constexpr (a2a_full_packets > 0 && a2a_remainder_tiles > 0) {
                        // Resetting required as both full and partial packets exist
                        // Device 2.0 migration: legacy primitive retained: state-machine setup
                        // (noc_async_write_one_packet_set_state) has no Device 2.0 wrapper
                        noc_async_write_one_packet_set_state</*posted=*/true>(
                            neighbor_base_addr, a2a_full_packet_size, /*noc=*/1, vchannel);
                    }

                    // Wait for current data to be ready in cb_s2c_in2
                    ring_sem.wait_min(semaphore_value);

                    // Signal to compute core that data is ready
                    cb_w2c_rdy.reserve_back(1);
                    cb_w2c_rdy.push_back(1);

                    // Write tiles from local cb_s2c_in2 to neighbor's cb_s2c_in2
                    // Double buffer offset: alternate between buffer 0 and buffer 1 based on step
                    const uint32_t local_src_addr = LOCAL_BUFFER_OFFSET[step];
                    const uint64_t neighbor_dst_addr = LOCAL_BUFFER_OFFSET[(step == num_cores - 1) ? 0 : (step + 1)];

                    uint32_t pkt_offset = 0;
                    // Rely on compiler to remove loop if no full packet exists
                    for (uint32_t pkt = 0; pkt < a2a_full_packets; ++pkt) {
                        noc_async_write_one_packet_with_state</*posted=*/true>(
                            local_src_addr + pkt_offset, neighbor_dst_addr + pkt_offset);
                        pkt_offset += a2a_full_packet_size;
                    }
                    if constexpr (a2a_remainder_tiles > 0) {
                        if constexpr (a2a_full_packets > 0) {
                            // Reset here if full packets exist, otherwise, it was already set once at the top and no
                            // reset required
                            noc_async_write_one_packet_set_state</*posted=*/true>(
                                neighbor_base_addr, a2a_remainder_size, /*noc=*/1, vchannel);
                        }
                        noc_async_write_one_packet_with_state</*posted=*/true>(
                            local_src_addr + pkt_offset, neighbor_dst_addr + pkt_offset);
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

            cb_c2s_out.wait_front(num_w0_w1_tiles_h);

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
                volatile tt_l1_ptr uint32_t* map =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_output_map_addr);
                const auto output_accessor = TensorAccessor(LoArgs::out_args, local_output_addr);
                for (uint32_t bt = 0; bt < num_tokens_block; ++bt) {
                    const uint32_t token_id = map[bt * e_t_entry_words];
                    const uint32_t k_slot = map[bt * e_t_entry_words + 1];
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
                        combine_shard_width_tiles - dest_width_offset_tiles,
                        output_width_tiles_core - width_tiles_sent);
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
                        const uint64_t dest_noc_addr_base =
                            get_noc_addr(dest_noc_x, dest_noc_y, output_base_l1_addr, 1);
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

            // Signal to tilize cores that they can send another chunk of tiles
            // Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address
            // (matmul_chunk_available_semaphore_noc_addr) cannot be wrapped by Semaphore<>::inc
            noc_semaphore_inc</*posted=*/true>(
                matmul_chunk_available_semaphore_noc_addr, /*incr=*/1, /*noc_id=*/1, /*vc=*/vchannel);
        }
        if constexpr (has_combine) {
            combine_semaphore_inc();
            combine_semaphore_val += height_shard_dim;
        }
        // (compute_only branch: nothing to do -- the next expert's first chunk flushes any
        //  in-flight writes via the inter-chunk flush before its set_state. Output buffer
        //  toggle below picks the other half so there's no destination overlap either.
        //  local_output: every chunk's writes were flushed before pop_front; the commit of the
        //  output is waited for once at the end.)
        output_buffer_idx = !output_buffer_idx;
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
    }
}
