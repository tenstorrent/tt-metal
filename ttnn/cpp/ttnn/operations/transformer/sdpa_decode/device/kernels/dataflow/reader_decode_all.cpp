// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"
#include <vector>

#include "ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"
#include "dataflow_common.hpp"

void kernel_main() {
    Noc noc;

    /*
    In DRAM, Q is (B, PNHt, DHt), K is (B, St, DHt), V is (B, St, DHt), mask is (B, PNHt, PSt)
    We want to read for a particular batch cur_batch, and sequence length up to padded layer length.
    We read Q: (cur_batch, PNHt, DHt), K: (cur_batch, PSt, DHt), V: (cur_batch, PSt, DHt), mask: (cur_batch, PNHt, PSt)
    */
    constexpr auto B = get_arg(args::B);                    // batch size
    constexpr auto PNHt = get_arg(args::PNHt);              // padded number of heads in tiles
    constexpr auto St = get_arg(args::St);                  // full sequence length of kv cache in tiles
    constexpr auto DHt = get_arg(args::DHt);                // head dim
    constexpr auto vDHt = get_arg(args::vDHt);              // head dim of V
    constexpr auto Sk_chunk_t = get_arg(args::Sk_chunk_t);  // number of tiles in seqlen of a k/v/mask chunk
    constexpr auto num_cores = get_arg(args::num_cores);
    constexpr bool is_q_sharded = get_arg(args::is_q_sharded);
    constexpr auto num_cores_per_batch = get_arg(args::num_cores_per_batch);
    constexpr auto k_chunk_size = get_arg(args::k_chunk_size);
    constexpr auto index_stick_size_B = get_arg(args::index_stick_size_B);
    constexpr auto num_kv_heads = get_arg(args::num_kv_heads);
    constexpr auto block_size_t = get_arg(args::block_size_t);
    constexpr auto Bkv = get_arg(args::Bkv);
    constexpr auto q_heads_parallel_factor = get_arg(args::q_heads_parallel_factor);
    constexpr auto num_cores_per_head = get_arg(args::num_cores_per_head);
    constexpr auto num_heads_per_core = get_arg(args::num_heads_per_core);
    constexpr auto num_output_cores = get_arg(args::num_output_cores);
    constexpr auto max_dynamic_chunk_size = get_arg(args::max_dynamic_chunk_size);
    constexpr bool reuse_k = get_arg(args::reuse_k) == 1;
    constexpr bool use_half_tile = get_arg(args::use_half_tile);
    constexpr auto q_chunk_size_bytes = get_arg(args::q_chunk_size_bytes);
    constexpr auto sliding_window_size = get_arg(args::sliding_window_size);
    constexpr auto original_block_size = get_arg(args::original_block_size);
    constexpr auto Bmask = get_arg(args::Bmask);
    // 0 = unbounded cache (legacy); nonzero = wrap virtual tile index mod this value
    // before page_table lookup. Value is in TILE rows (= cache_position_modulo /
    // TILE_HEIGHT). Validated to be a multiple of block_size_t at op level.
    constexpr auto capacity_t = get_arg(args::capacity_t);
    constexpr bool use_k_mcast = get_arg(args::use_k_mcast) == 1;

#ifdef IS_CAUSAL
    constexpr bool is_causal = true;
#else
    constexpr bool is_causal = false;
#endif
#ifdef IS_PAGED_ATTENTION
    constexpr bool is_paged_attention = true;
#else
    constexpr bool is_paged_attention = false;
#endif
#ifdef IS_PAGE_TABLE_SHARDED
    constexpr bool is_page_table_sharded = true;
#else
    constexpr bool is_page_table_sharded = false;
#endif
    constexpr bool has_block_padding = is_paged_attention && original_block_size > 0 && original_block_size < 32;

    // DFB accessors (each exists only where the host bound it — see #ifdef gates).
#ifndef TILIZE_Q
    constexpr auto dfb_q_in = dfb::q_in;
#endif
#ifdef TILIZE_Q
    constexpr auto dfb_q_rm = dfb::q_rm;
#endif
    constexpr auto dfb_k_in = dfb::k_in;
    constexpr auto dfb_v_in = dfb::v_in;
#ifdef USE_ATTENTION_MASK
    constexpr auto dfb_mask_in = dfb::mask_in;
#endif
#ifdef USE_ATTENTION_SINK
    constexpr auto dfb_attention_sink = dfb::attention_sink;
#endif
#ifdef USE_CUR_POS_TENSOR
    // #44366: cur_pos is consumed by both the writer (writer_cur_pos)
    // and compute (compute_cur_pos). Using one shared DFB races —
    // whichever consumer pops first drains the count and the other hangs waiting
    // for tiles. Each consumer gets its own DFB.
    constexpr auto dfb_writer_cur_pos = dfb::writer_cur_pos;
    constexpr auto dfb_compute_cur_pos = dfb::compute_cur_pos;
#endif
#ifdef IS_PAGED_ATTENTION
    constexpr auto dfb_id_page_table = dfb::page_table;
#endif

    auto do_reduce_arg = get_arg(args::do_reduce);
    // idle core: do_reduce==65 is out of the valid {0,1} range and marks a core with no assigned
    // work. Exit before touching any binding, mcast, or semaphore. (Legacy keyed this off q_addr==0;
    // Metal 2.0 supplies addresses via TensorBindings, so the compute kernel's ==65 marker is reused.)
    if (do_reduce_arg == 65) {
        return;
    }
    const bool is_worker = do_reduce_arg == 0;
    const bool is_output_core = get_arg(args::do_output) == 1;
    const uint32_t page_table_page_size = get_arg(args::page_table_page_size);
    const uint32_t cur_head_group = get_arg(args::cur_head_group);
    const uint32_t cur_batch = get_arg(args::cur_batch);
    const uint32_t core_num_in_reduce = get_arg(args::core_num_in_reduce);
    const uint32_t core_num_in_output = get_arg(args::core_num_in_output);
    const uint32_t cur_pos_arg = get_arg(args::cur_pos_arg);
    const bool do_k_mcast = get_arg(args::do_k_mcast);
    const uint32_t mcast_x = get_arg(args::mcast_x);
    const uint32_t mcast_y0 = get_arg(args::mcast_y0);
    const uint32_t mcast_y1 = get_arg(args::mcast_y1);
    const uint32_t num_dests = get_arg(args::num_dests);

    // Get cur_pos
    constexpr uint32_t cur_pos_base = St * 32 - 1;
    uint32_t cur_pos = cur_pos_base;  // default to non-causal, which we do attention on the entire kv cache. In this
                                      // case we set cur_pos to the last position
    if constexpr (is_causal) {
        // using UINT32_MAX as a flag to indicate that cur_pos is not provided as a list
        if (cur_pos_arg != UINT32_MAX) {
            cur_pos = cur_pos_arg;
        } else {
#ifdef USE_CUR_POS_TENSOR
            // Reader fills dfb_writer_cur_pos first (from DRAM, or via the
            // aliased sharded buffer) then copies the same stick into
            // dfb_compute_cur_pos via an L1->L1 read.
            DataflowBuffer dfb_writer(dfb_writer_cur_pos);
            dfb_writer.reserve_back(1);
            uint32_t index_dfb_wr_ptr = dfb_writer.get_write_ptr();
#ifndef IS_CUR_POS_TENSOR_SHARDED
            const auto addrg = TensorAccessor(tensor::cur_pos);
            // index_tensor has one page to read
            noc.async_read(addrg, CoreLocalMem<uint32_t>(index_dfb_wr_ptr), index_stick_size_B, {.page_id = 0}, {});
            noc.async_read_barrier();
#endif
            DataflowBuffer dfb_compute(dfb_compute_cur_pos);
            dfb_compute.reserve_back(1);
            uint32_t index_dfb_compute_wr_ptr = dfb_compute.get_write_ptr();
            const uint8_t noc_id = noc.get_noc_id();
            const uint32_t my_noc_x = my_x[noc_id];
            const uint32_t my_noc_y = my_y[noc_id];
            UnicastEndpoint pos_src;
            noc.async_read(
                pos_src,
                CoreLocalMem<uint32_t>(index_dfb_compute_wr_ptr),
                index_stick_size_B,
                {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = index_dfb_wr_ptr},
                {});
            noc.async_read_barrier();
            dfb_writer.push_back(1);
            dfb_compute.push_back(1);
            volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_dfb_wr_ptr);
            cur_pos = index_ptr[cur_batch / q_heads_parallel_factor];
#endif
        }
        if (cur_pos == UINT32_MAX) {
            // cur_pos of -1 indicates that the user should be skipped
            return;
        }
    }

    // When block_size < TILE_HEIGHT, each tile has zero-padded rows. Convert cur_pos from
    // the original sequence space to the padded tile space so get_runtime_args computes the
    // correct number of tiles to process. Only needed for causal mode where cur_pos comes
    // from user input; non-causal uses cur_pos_base which is already in the padded space.
    if constexpr (has_block_padding && is_causal) {
        cur_pos = (cur_pos / original_block_size) * 32 + (cur_pos % original_block_size);
    }

    auto Sk_chunk_t_dynamic = get_dynamic_Sk_chunk_t<Sk_chunk_t, max_dynamic_chunk_size>(cur_pos);
    auto k_chunk_size_dynamic = Sk_chunk_t_dynamic * tt::constants::TILE_HEIGHT;

    // Sequence length assignment
    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end, window_start_unaligned, window_start_chunk] =
        get_workload_for_core(
            cur_pos,
            cur_batch,
            core_num_in_reduce,
            num_cores_per_head,
            k_chunk_size_dynamic,
            sliding_window_size > 0 ? std::optional<uint32_t>(sliding_window_size) : std::nullopt);

    if (k_chunk_start == k_chunk_end) {
        return;  // early exit because no computes needs to be done
    }

    // all_output_noc_x[cur_batch] / all_output_noc_y[cur_batch] — vararg physical-core coordinate arrays.
    uint32_t output_core_noc_x = get_vararg(cur_batch);
    uint32_t output_core_noc_y = get_vararg(num_output_cores + cur_batch);

    constexpr uint32_t q_chunk_tiles = PNHt * DHt;
    uint32_t k_chunk_tiles = Sk_chunk_t_dynamic * DHt;
    uint32_t v_chunk_tiles = Sk_chunk_t_dynamic * vDHt;
    uint32_t mask_chunk_tiles = PNHt * Sk_chunk_t_dynamic;

    constexpr uint32_t onetile = 1;
#ifdef TILIZE_Q
    constexpr uint32_t q_tile_bytes = get_tile_size(dfb::q_rm);
#else
    constexpr uint32_t q_tile_bytes = get_tile_size(dfb::q_in);
#endif
    constexpr uint32_t k_tile_bytes = get_tile_size(dfb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(dfb_v_in);
#ifdef USE_ATTENTION_MASK
    constexpr uint32_t mask_tile_bytes = get_tile_size(dfb_mask_in);
#endif
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();
    uint32_t barrier_count = 0;

    // Read Q entirely - always read into dfb_q_in
    // When tilize_q is true, compute will tilize back to dfb_q_in
    // When tilize_q is false, Q is already tilized
    const uint32_t q_batch_offset = cur_batch * q_chunk_tiles;

    // Read Q
#ifdef Q_LOCALLY_AVAILABLE
    // Q pre-sharded / replicated to all cores: it is already resident in the borrowed DFB. Just reserve + push.
#ifdef TILIZE_Q
    {
        DataflowBuffer dfb_q_rm_buf(dfb_q_rm);
        dfb_q_rm_buf.reserve_back(q_chunk_tiles);
        dfb_q_rm_buf.push_back(q_chunk_tiles);
    }
#else
    {
        DataflowBuffer dfb_q(dfb_q_in);
        dfb_q.reserve_back(q_chunk_tiles);
        dfb_q.push_back(q_chunk_tiles);
    }
#endif
#else
    // Not locally available: read Q from the output core's L1 (sharded) or from DRAM (interleaved).
    {
        const auto q_reader = TensorAccessor(tensor::q);
#ifdef TILIZE_Q
        read_q<
            dfb::q_rm,
            dfb::q_rm,
            q_tile_bytes,
            q_chunk_tiles,
            is_q_sharded,
            /*tilize_q=*/true,
            use_half_tile,
            barrier_threshold>(
            is_output_core, q_reader, output_core_noc_x, output_core_noc_y, q_chunk_size_bytes, q_batch_offset);
#else
        read_q<
            dfb::q_in,
            dfb::q_in,
            q_tile_bytes,
            q_chunk_tiles,
            is_q_sharded,
            /*tilize_q=*/false,
            use_half_tile,
            barrier_threshold>(
            is_output_core, q_reader, output_core_noc_x, output_core_noc_y, q_chunk_size_bytes, q_batch_offset);
#endif
    }
#endif

    const auto k_reader = TensorAccessor(tensor::k);
#ifndef REUSE_K
    const auto v_reader = TensorAccessor(tensor::v);
#endif
#ifdef USE_ATTENTION_MASK
    const auto mask_reader = TensorAccessor(tensor::attn_mask);
#endif

    // Read attention sink
#ifdef USE_ATTENTION_SINK
    {
        const auto attention_sink_reader = TensorAccessor(tensor::attention_sink);
        constexpr uint32_t attention_sink_tile_bytes = get_tile_size(dfb_attention_sink);

        DataflowBuffer dfb_sink(dfb_attention_sink);
        dfb_sink.reserve_back(PNHt);
        uint32_t attention_sink_write_ptr = dfb_sink.get_write_ptr();

        for (uint32_t tile = 0; tile < PNHt; ++tile) {
            // Use noc.async_read with explicit size instead of noc.async_read_page because
            // the CB may use half tiles (16x32) while the DRAM buffer stores full tiles (32x32).
            // noc.async_read_page would read buffer->aligned_page_size() bytes, overflowing the CB.
            noc.async_read(
                attention_sink_reader,
                CoreLocalMem<uint32_t>(attention_sink_write_ptr),
                attention_sink_tile_bytes,
                {.page_id = tile},
                {});
            attention_sink_write_ptr += attention_sink_tile_bytes;
        }
        noc.async_read_barrier();
        dfb_sink.push_back(PNHt);
    }
#endif

    // Read page table
    volatile tt_l1_ptr uint32_t* page_table_ptr;
    volatile tt_l1_ptr uint16_t* page_table_ptr_u16 = nullptr;
    volatile tt_l1_ptr uint32_t* page_table_ptr_u32 = nullptr;
#ifdef IS_PAGED_ATTENTION
    {
        DataflowBuffer dfb_page_table(dfb_id_page_table);
        uint32_t num_pages_to_read = is_page_table_sharded ? B : 1;
        dfb_page_table.reserve_back(num_pages_to_read);
#ifndef IS_PAGE_TABLE_SHARDED
        // Read page table from DRAM via the sdpa donor's Metal 2.0 overload of read_page_table_for_batch,
        // building the accessor from the tensor::page_table binding (the legacy 3rd page-size argument is
        // redundant — the binding token supplies the aligned page size — and is dropped).
        page_table_ptr = read_page_table_for_batch(
            noc,
            dfb_page_table,
            cur_batch / q_heads_parallel_factor,
            TensorAccessor(tensor::page_table),
            page_table_page_size);
        page_table_ptr_u32 = page_table_ptr;
#else
        // Read page table from dynamically allocated L1 buffer (borrowed sharded buffer)
        uint32_t page_table_dfb_wr_ptr =
            dfb_page_table.get_write_ptr() + (cur_batch / q_heads_parallel_factor) * page_table_page_size;
        page_table_ptr_u16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(page_table_dfb_wr_ptr);
#endif
        dfb_page_table.push_back(num_pages_to_read);
    }
#endif

    for (uint32_t cur_head = cur_head_group * num_heads_per_core;
         cur_head < cur_head_group * num_heads_per_core + num_heads_per_core;
         ++cur_head) {
        const uint32_t mask_batch_offset = ((cur_batch / q_heads_parallel_factor) % Bmask) * PNHt * St;
        const uint32_t mask_chunk_offset = k_chunk_start * Sk_chunk_t_dynamic;
        uint32_t mask_start_tile_id = mask_batch_offset + mask_chunk_offset;
        // Setup multicast parameters for K streaming (vertical multicast)
        KMcastParams k_mcast_params = {
            .do_mcast = do_k_mcast,
            .mcast_x = mcast_x,
            .mcast_y0 = mcast_y0,
            .mcast_y1 = mcast_y1,
            .num_dests = num_dests,
            .mcast_sem_id = sem::k_mcast};

#ifdef IS_PAGED_ATTENTION
        for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
            const uint32_t k_chunk_start_row_num = k_chunk * Sk_chunk_t_dynamic;
            uint64_t k_base_read_ptr;

            // Read K chunk - supports both multicast and non-multicast paths
            k_base_read_ptr = read_k<
                dfb::k_in,
                DHt,
                num_kv_heads,
                block_size_t,
                k_tile_bytes,
                barrier_threshold,
                is_page_table_sharded,
                use_k_mcast,
                capacity_t>(
                k_chunk_tiles,
                cur_head,
                Sk_chunk_t_dynamic,
                k_chunk_start_row_num,
                k_reader,
                page_table_ptr_u16,
                page_table_ptr_u32,
                barrier_count,
                k_mcast_params);

#ifdef USE_ATTENTION_MASK
            mask_start_tile_id = read_mask_chunk<dfb::mask_in, mask_tile_bytes, barrier_threshold, PNHt>(
                PSt, Sk_chunk_t_dynamic, mask_chunk_tiles, mask_start_tile_id, mask_reader);
#endif
            // Read V chunk - either from DRAM or from K's L1 buffer (transpose) when reuse_k is true
            read_v<
                dfb::v_in,
                vDHt,
                num_kv_heads,
                block_size_t,
                v_tile_bytes,
                barrier_threshold,
                is_page_table_sharded,
                reuse_k,
                capacity_t>(
                v_chunk_tiles,
                cur_head,
                Sk_chunk_t_dynamic,
                k_chunk_start_row_num,
#ifdef REUSE_K
                k_reader,  // unused under reuse_k (V rides on K's L1); pass k_reader as a placeholder accessor
#else
                v_reader,
#endif
                page_table_ptr_u16,
                page_table_ptr_u32,
                barrier_count,
                k_base_read_ptr,
                k_tile_bytes);
        }
#else
        {
            // Offset for current batch
            const uint32_t k_batch_offset = ((cur_batch / q_heads_parallel_factor) % Bkv) * num_kv_heads * St * DHt;
            const uint32_t k_head_offset = cur_head * St * DHt;

            // Then, read K, V, Mask k_chunk_tiles at a time
            const uint32_t k_chunk_offset = k_chunk_start * Sk_chunk_t_dynamic * DHt;
            uint32_t k_start_tile_id = k_batch_offset + k_head_offset + k_chunk_offset;

            // V has its own layout when it's an independent tensor (width = vDHt, not DHt)
            const uint32_t v_batch_offset = ((cur_batch / q_heads_parallel_factor) % Bkv) * num_kv_heads * St * vDHt;
            const uint32_t v_head_offset = cur_head * St * vDHt;
            const uint32_t v_chunk_offset = k_chunk_start * Sk_chunk_t_dynamic * vDHt;
            uint32_t v_start_tile_id = v_batch_offset + v_head_offset + v_chunk_offset;

            read_kv_mask_chunks<
                DHt,
                vDHt,
                barrier_threshold,
#ifdef USE_ATTENTION_MASK
                mask_tile_bytes,
                PNHt,
                /*use_attention_mask=*/true,
                dfb::k_in,
                dfb::v_in,
                dfb::mask_in,
#else
                /*mask_tile_bytes=*/0,
                PNHt,
                /*use_attention_mask=*/false,
                dfb::k_in,
                dfb::v_in,
                dfb::k_in,  // dummy — mask path is disabled, this NTTP is unused
#endif
                reuse_k>(
                k_chunk_start,
                k_chunk_end,
                k_start_tile_id,
                v_start_tile_id,
                mask_start_tile_id,
                Sk_chunk_t_dynamic,
                k_chunk_tiles,
                v_chunk_tiles,
                mask_chunk_tiles,
                k_reader,
#ifdef REUSE_K
                k_reader,
#else
                v_reader,
#endif
#ifdef USE_ATTENTION_MASK
                mask_reader,
#else
                k_reader,  // dummy — mask path disabled
#endif
                k_tile_bytes,
                v_tile_bytes,
                PSt);
        }
#endif
    }
}
