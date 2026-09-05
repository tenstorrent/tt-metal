// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#ifdef FUSE_RMS_NORM
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/experimental/add_rsqrt.h"
#include "api/compute/experimental/mul_reduce_scalar.h"
#include "api/compute/experimental/pack_block.h"
#include "api/compute/experimental/rmsnorm.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#endif
#ifdef USE_CUSTOM_MM
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/experimental/pack_block.h"
#endif
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

using std::uint32_t;

// C = A @ B per core. full_in0 is sender-major. matmul_block does not reduce over kt_dim; K is accumulated in the loop.
//
// With ENABLE_GLOBAL_CB the in1 tiles arrive through a GCB-backed circular buffer instead of a
// globally-allocated one, so this kernel must actually wait on them and, when done, tell the
// reader (via the sync CB) that the GCB page can be released.
//
// num_k_blocks > 1 means a receiver's slab arrives as that many GCB pages rather than one, so K
// becomes the outer loop: a page can only be read while it is held, and it is popped before the
// next arrives. Every output column is therefore still incomplete when its page is released, and
// the running sums live in the output CB, where the packer accumulates into them
// (pack_reconfig_l1_acc). num_k_blocks == 1 leaves the single-page traversal below unchanged: the
// loop runs once, the packer never switches into accumulate mode, and the tile indexing collapses
// to what it was.
using namespace ckernel;
void kernel_main() {
    constexpr uint32_t out_block_w = 1;

    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t N_tiles_per_core = get_compile_time_arg_val(2);
    constexpr uint32_t inA_K_tiles_per_core = get_compile_time_arg_val(3);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(4);

    constexpr uint32_t out_block_h = M_tiles;
    constexpr uint32_t in0_block_w = inA_K_tiles_per_core;

    // Named so op fusion can remap them; the reader gathers A into cb_full_in0, which is what
    // compute reads as its in0.
    constexpr uint32_t in0_cb_id = get_named_compile_time_arg_val("cb_full_in0");
    constexpr uint32_t in1_cb_id = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t out_cb_id = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t sync_cb_id = get_named_compile_time_arg_val("cb_sync");
#ifdef FUSE_RMS_NORM
    constexpr uint32_t mm_out_cb_id = get_named_compile_time_arg_val("cb_mm_out");
    constexpr uint32_t rms_local_cb_id = get_named_compile_time_arg_val("cb_rms_local");
    constexpr uint32_t rms_gathered_cb_id = get_named_compile_time_arg_val("cb_rms_gathered");
    // Hub only: the scale as produced here, handed to the writer RISC for multicast. Distinct from
    // cb_rms_scale, which is the multicast destination and is consumed by compute on every producer.
    constexpr uint32_t rms_scale_src_cb_id = get_named_compile_time_arg_val("cb_rms_scale_src");
    constexpr uint32_t rms_scale_cb_id = get_named_compile_time_arg_val("cb_rms_scale");
    constexpr uint32_t rms_reduce_scaler_cb_id = get_named_compile_time_arg_val("cb_rms_reduce_scaler");
    constexpr uint32_t rms_reduced_cb_id = get_named_compile_time_arg_val("cb_rms_reduced");
    constexpr uint32_t rms_packed_tiles_per_row = get_named_compile_time_arg_val("rms_packed_tiles_per_row");

    const bool rms_is_hub = get_arg_val<uint32_t>(0) != 0;
    const uint32_t rms_inv_n_bits = get_arg_val<uint32_t>(1);
    const uint32_t rms_epsilon_bits = get_arg_val<uint32_t>(2);
    const uint32_t rms_gamma_bits = get_arg_val<uint32_t>(3);
#else
    constexpr uint32_t mm_out_cb_id = out_cb_id;
#endif

    constexpr uint32_t in0_num_tiles = M_tiles * K_tiles;
    constexpr uint32_t num_senders = K_tiles / inA_K_tiles_per_core;
    constexpr uint32_t sender_slice_tiles = M_tiles * inA_K_tiles_per_core;
    // One GCB page: k_block_tiles whole K-rows of this receiver's slab.
    constexpr uint32_t k_block_tiles = K_tiles / num_k_blocks;
    constexpr uint32_t in1_page_tiles = k_block_tiles * N_tiles_per_core;

    CircularBuffer in0_cb(in0_cb_id);
    CircularBuffer out_cb(mm_out_cb_id);
#ifdef ENABLE_GLOBAL_CB
    CircularBuffer in1_cb(in1_cb_id);
    CircularBuffer sync_cb(sync_cb_id);
#endif

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb_id, in1_cb_id, mm_out_cb_id);

    in0_cb.wait_front(in0_num_tiles);

#ifdef USE_CUSTOM_MM
    constexpr bool transpose = false;
    constexpr bool split_acc = true;
    constexpr bool dense_packing = true;
    constexpr bool finalize = true;

    static_assert(k_block_tiles >= 2 && k_block_tiles <= 256 && k_block_tiles % 2 == 0);
    static_assert(N_tiles_per_core >= 1 && N_tiles_per_core <= 16);
    static_assert(num_k_blocks == 1 || M_tiles == 1, "streamed custom_mm keeps a whole output row in DST");

    custom_mm_block_init_short<transpose, split_acc, dense_packing>(
        in0_cb_id, in1_cb_id, mm_out_cb_id, N_tiles_per_core);
    pack_block_contiguous_init(mm_out_cb_id);

    out_cb.reserve_back(M_tiles * N_tiles_per_core);
    if constexpr (num_k_blocks == 1) {
#ifdef ENABLE_GLOBAL_CB
        in1_cb.wait_front(in1_page_tiles);
#endif
        for (uint32_t mt = 0; mt < M_tiles; ++mt) {
            tile_regs_acquire();
            custom_mm_block<finalize, false>(in0_cb_id, in1_cb_id, mt * K_tiles, 0, 0, K_tiles, N_tiles_per_core);
            tile_regs_commit();
            tile_regs_wait();
            pack_block_contiguous(0, mm_out_cb_id, N_tiles_per_core);
            tile_regs_release();
        }
#ifdef ENABLE_GLOBAL_CB
        in1_cb.pop_front(in1_page_tiles);
        sync_cb.reserve_back(1);
        sync_cb.push_back(1);
#endif
    } else {
        // Each page covers k_block_tiles of the reduction, and MVMUL adds into DST without
        // clearing it, so chaining one call per page leaves the running sum in DST rather than in
        // the output CB. Only the last page finalizes, merging the split_acc partials once the
        // reduction is whole. This is why streaming needs no packer L1 accumulation here.
        tile_regs_acquire();
        for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
#ifdef ENABLE_GLOBAL_CB
            in1_cb.wait_front(in1_page_tiles);
#endif
            const uint32_t in0_tile = kb * k_block_tiles;
            if (kb == num_k_blocks - 1) {
                custom_mm_block<true, false>(in0_cb_id, in1_cb_id, in0_tile, 0, 0, k_block_tiles, N_tiles_per_core);
            } else {
                custom_mm_block<false, false>(in0_cb_id, in1_cb_id, in0_tile, 0, 0, k_block_tiles, N_tiles_per_core);
            }
#ifdef ENABLE_GLOBAL_CB
            // This page has been read in full; release the local alias and let the reader ack it.
            in1_cb.pop_front(in1_page_tiles);
            sync_cb.reserve_back(1);
            sync_cb.push_back(1);
#endif
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_block_contiguous(0, mm_out_cb_id, N_tiles_per_core);
        tile_regs_release();
    }
    custom_mm_block_uninit<dense_packing>();
    out_cb.push_back(M_tiles * N_tiles_per_core);
#else
    matmul_block_init(in0_cb_id, in1_cb_id, false, out_block_w, out_block_h, in0_block_w);

    out_cb.reserve_back(M_tiles * N_tiles_per_core);
    for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
#ifdef ENABLE_GLOBAL_CB
        in1_cb.wait_front(in1_page_tiles);
#endif
        const uint32_t k_block_base = kb * k_block_tiles;
        for (uint32_t bw = 0; bw < N_tiles_per_core; ++bw) {
            tile_regs_acquire();
            for (uint32_t kc = 0; kc < k_block_tiles; ++kc) {
                const uint32_t k_global = k_block_base + kc;
                const uint32_t sender = k_global / inA_K_tiles_per_core;
                const uint32_t kc_local = k_global - sender * inA_K_tiles_per_core;
                const uint32_t in0_tile = sender * sender_slice_tiles + kc_local;
                // in1 is indexed within the page currently at the front of the CB.
                const uint32_t in1_tile = kc * N_tiles_per_core + bw;
                matmul_block(in0_cb_id, in1_cb_id, in0_tile, in1_tile, 0, false, out_block_w, out_block_h, in0_block_w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t mt = 0; mt < out_block_h; ++mt) {
                pack_tile<true>(mt, mm_out_cb_id, mt * N_tiles_per_core + bw);
            }
            tile_regs_release();
        }
#ifdef ENABLE_GLOBAL_CB
        // This page has been read in full; release the local alias and let the reader ack it.
        in1_cb.pop_front(in1_page_tiles);
        sync_cb.reserve_back(1);
        sync_cb.push_back(1);
#endif
        if constexpr (num_k_blocks > 1) {
            // From here on the packs add to the partial sums the previous pages left behind.
            if (kb == 0) {
                pack_reconfig_l1_acc(1);
            }
        }
    }
    if constexpr (num_k_blocks > 1) {
        pack_reconfig_l1_acc(0);
    }
    out_cb.push_back(M_tiles * N_tiles_per_core);
#endif

    in0_cb.pop_front(in0_num_tiles);

#ifdef FUSE_RMS_NORM
    constexpr uint32_t local_out_tiles = M_tiles * N_tiles_per_core;
    // The chunked reduction reserves one DST slot as its cross-chunk accumulator.
    constexpr uint32_t rms_dst_capacity = get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>();

    CircularBuffer mm_out_cb(mm_out_cb_id);
    CircularBuffer rms_local_cb(rms_local_cb_id);
    CircularBuffer rms_scale_cb(rms_scale_cb_id);
    CircularBuffer final_out_cb(out_cb_id);

    // custom_mm_block_uninit restores the packer's tile stride from TILE_NUM_FACES/FACE_R_DIM/
    // FACE_C_DIM, which are fixed 32x32 constants, so it leaves a 2048-byte stride behind. The
    // epilogue packs 1x32 tiles whose stride is 64 bytes, and with a 2048-byte stride each tile's
    // second face lands a whole 32x32 tile away instead of next to the first. Reprogram the packer
    // from the geometry of the circular buffers this epilogue actually uses.
    compute_kernel_hw_startup(mm_out_cb_id, rms_scale_cb_id, out_cb_id);

    mm_out_cb.wait_front(local_out_tiles);

    // sum(x^2) over this core's shard of each output row, landing as a single value at [0, 0] of that
    // row's statistics tile. Each output tile is one logical row, so a scalar reduction over a row's
    // tiles is exactly that row's statistic. mul_reduce_scalar folds the square into the reduction,
    // so the narrow output tile never passes through an SFPU unary or a row-wise reduce -- neither of
    // which handles a 2-face tile. The reduction applies its scaler on both the column and the final
    // scalar pass, so it gets 1.0 and the hub divides by N once instead.
    reconfig_data_format(mm_out_cb_id, mm_out_cb_id);
    pack_reconfig_data_format(rms_local_cb_id);
    rms_local_cb.reserve_back(M_tiles);
    for (uint32_t mt = 0; mt < M_tiles; ++mt) {
        const uint32_t row_start = mt * N_tiles_per_core;
        mul_reduce_scalar_init(mm_out_cb_id, mm_out_cb_id);
        tile_regs_acquire();
        if constexpr (N_tiles_per_core <= rms_dst_capacity) {
            mul_reduce_scalar_tile<PoolType::SUM>(
                mm_out_cb_id, mm_out_cb_id, rms_local_cb_id, N_tiles_per_core, 1.0F, row_start);
            mul_reduce_scalar_uninit();
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, rms_local_cb_id, mt);
        } else {
            // Row wider than DST holds: chunk it and accumulate across chunks.
            add_binary_tile_init();
            mul_reduce_scalar_chunked_tile<N_tiles_per_core, rms_dst_capacity>(
                mm_out_cb_id, mm_out_cb_id, rms_local_cb_id, 1.0F, row_start);
            mul_reduce_scalar_uninit();
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(rms_dst_capacity - 1, rms_local_cb_id, mt);
        }
        tile_regs_release();
    }
    rms_local_cb.push_back(M_tiles);

    // Only the first row-major producer consumes the gathered statistics. It sums the producers'
    // partial mean-squares and forms gamma * rsqrt(mean + epsilon), then publishes it to BRISC for
    // multicast. The writer has packed the scalar-only producer pages into full tiles, so one
    // REDUCE_SCALAR pass replaces the old O(num_producers) copy/add chain.
    if (rms_is_hub) {
        compute_kernel_lib::reduce<
            PoolType::SUM,
            ReduceDim::REDUCE_SCALAR,
            rms_gathered_cb_id,
            rms_reduce_scaler_cb_id,
            rms_reduced_cb_id,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
            compute_kernel_lib::ReduceInputBlockShape::of(1, rms_packed_tiles_per_row, M_tiles));

        CircularBuffer rms_reduced_cb(rms_reduced_cb_id);
        CircularBuffer rms_scale_src_cb(rms_scale_src_cb_id);
        rms_reduced_cb.wait_front(M_tiles);
        rms_scale_src_cb.reserve_back(M_tiles);
        reconfig_data_format(rms_reduced_cb_id, rms_reduced_cb_id);
        pack_reconfig_data_format(rms_scale_src_cb_id);
        copy_init(rms_reduced_cb_id);
        for (uint32_t mt = 0; mt < M_tiles; ++mt) {
            tile_regs_acquire();
            copy_tile(rms_reduced_cb_id, mt, 0);
            // The producers contribute raw sums of squares, so the mean is formed here.
            binop_with_scalar_tile_init();
            mul_unary_tile(0, rms_inv_n_bits);
            add_rsqrt_tile_init();
            add_rsqrt_tile<false, VectorMode::RC_custom, 1>(0, rms_epsilon_bits);
            binop_with_scalar_tile_init();
            mul_unary_tile(0, rms_gamma_bits);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, rms_scale_src_cb_id, mt);
            tile_regs_release();
        }
        // Handed off to the writer RISC, which is this CB's only consumer and pops it after the mcast.
        rms_scale_src_cb.push_back(M_tiles);
        rms_reduced_cb.pop_front(M_tiles);
    }

    // The writer RISC pushes cb_rms_scale on every producer, hub included, once the multicast payload
    // has landed locally. Compute is its only consumer, so this wait cannot race the transport.
    rms_scale_cb.wait_front(M_tiles);
#if defined(RMS_DEBUG_DUMP_COPY)
    // TODO(debug): temporary. Plain copy of the matmul result through DST, to check whether an
    // ordinary unpack/pack round trip preserves all 32 datums of a narrow tile.
    final_out_cb.reserve_back(local_out_tiles);
    reconfig_data_format(mm_out_cb_id, mm_out_cb_id);
    pack_reconfig_data_format(out_cb_id);
    copy_init(mm_out_cb_id);
    for (uint32_t t = 0; t < local_out_tiles; ++t) {
        tile_regs_acquire();
        copy_tile(mm_out_cb_id, t, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, out_cb_id, t);
        tile_regs_release();
    }
    final_out_cb.push_back(local_out_tiles);
    rms_scale_cb.pop_front(M_tiles);
    mm_out_cb.pop_front(local_out_tiles);
#elif defined(RMS_DEBUG_DUMP_LOCAL) || defined(RMS_DEBUG_DUMP_SCALE)
    // TODO(debug): temporary. Copies an intermediate statistics tile verbatim into every output tile
    // so host can read element [0] of each tile per core.
#ifdef RMS_DEBUG_DUMP_LOCAL
    constexpr uint32_t rms_dbg_cb_id = rms_local_cb_id;
#else
    constexpr uint32_t rms_dbg_cb_id = rms_scale_cb_id;
#endif
    final_out_cb.reserve_back(local_out_tiles);
    reconfig_data_format(rms_dbg_cb_id, rms_dbg_cb_id);
    pack_reconfig_data_format(out_cb_id);
    copy_init(rms_dbg_cb_id);
    for (uint32_t mt = 0; mt < M_tiles; ++mt) {
        for (uint32_t bw = 0; bw < N_tiles_per_core; ++bw) {
            tile_regs_acquire();
            copy_tile(rms_dbg_cb_id, mt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, out_cb_id, mt * N_tiles_per_core + bw);
            tile_regs_release();
        }
    }
    final_out_cb.push_back(local_out_tiles);
    rms_scale_cb.pop_front(M_tiles);
    mm_out_cb.pop_front(local_out_tiles);
#else
    // The scale has to reach SrcB from DST, not from a circular buffer. The unpacker's SCALAR
    // broadcast replicates [0, 0] across one face only, which silently zeroes the second face of a
    // 1x32 tile; the rmsnorm dest-reuse LLK is the variant that covers the whole narrow tile. It
    // leaves the row packed densely in DST, so the row is packed as a block rather than tile by
    // tile, one tile row per reservation so the write pointer walks the output shard.
    reconfig_data_format(rms_scale_cb_id, rms_scale_cb_id);
    pack_reconfig_data_format(out_cb_id);
    pack_block_contiguous_init(out_cb_id);
    for (uint32_t mt = 0; mt < M_tiles; ++mt) {
        final_out_cb.reserve_back(N_tiles_per_core);
        tile_regs_acquire();
        copy_init(rms_scale_cb_id);
        copy_tile(rms_scale_cb_id, mt, 0);
        // Consumes DST[0] as the broadcast scalar and refills DST with the scaled row, so the whole
        // row must fit in DST.
        rmsnorm_mul_bcast_scalar_reuse_tiles_init<N_tiles_per_core>(mm_out_cb_id);
        rmsnorm_mul_bcast_scalar_reuse_tiles<N_tiles_per_core, true>(mm_out_cb_id, mt * N_tiles_per_core, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_block_contiguous(0, out_cb_id, N_tiles_per_core);
        tile_regs_release();
        final_out_cb.push_back(N_tiles_per_core);
    }
    rms_scale_cb.pop_front(M_tiles);
    mm_out_cb.pop_front(local_out_tiles);
#endif
#endif
}
