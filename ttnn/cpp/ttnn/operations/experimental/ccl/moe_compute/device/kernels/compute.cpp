// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_ring_common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"

// Need these headers for running SFPU on PACK thread
#ifdef TRISC_PACK
#include "ckernel_sfpu_exp.h"
#include "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/swiglu_sfpu.h"
#include "ckernel_sfpu_silu.h"
#include "ckernel_sfpu_binary.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#include "ckernel_sfpu_gelu.h"
#endif

namespace detail {

FORCE_INLINE
void noc_semaphore_wait_min(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
    WAYPOINT("NSMW");
    do {
        invalidate_l1_cache();
    } while ((*sem_addr) < val);
    WAYPOINT("NSMD");
}

// Note GELU gets init'd at each iteration in the pack_compute_activation specialization
template <ttnn::experimental::prim::detail::MoEActivationFunction activation>
inline void pack_init_activation() {};

template <>
inline void pack_init_activation<ttnn::experimental::prim::detail::MoEActivationFunction::SWIGLU>() {
    PACK((llk_math_eltwise_binary_sfpu_swiglu_init()));
};

template <>
inline void pack_init_activation<ttnn::experimental::prim::detail::MoEActivationFunction::SILU>() {
    PACK(SFPU_UNARY_INIT_FN(silu, sfpu::silu_init, (true /*APPROXIMATE*/)));
};

template <ttnn::experimental::prim::detail::MoEActivationFunction activation>
inline void pack_compute_activation() {};

template <>
inline void pack_compute_activation<ttnn::experimental::prim::detail::MoEActivationFunction::SILU>() {
    PACK(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_silu,
        (false /*is_fp32_dest_acc_en*/, 8 /*ITERATIONS*/),
        0 /*DST_IDX*/,
        ::ckernel::VectorMode::RC));
    PACK(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_silu,
        (false /*is_fp32_dest_acc_en*/, 8 /*ITERATIONS*/),
        2 /*DST_IDX*/,
        ::ckernel::VectorMode::RC));

    PACK((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (true /*APPROXIMATE*/, ckernel::BinaryOp::MUL, 8 /*ITERATIONS*/),
        0 /*DST_IN0*/,
        1 /*DST_IN1*/,
        0 /*DST_OUT*/,
        ::ckernel::VectorMode::RC)));
    PACK((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (true /*APPROXIMATE*/, ckernel::BinaryOp::MUL, 8 /*ITERATIONS*/),
        2 /*DST_IN0*/,
        3 /*DST_IN1*/,
        2 /*DST_OUT*/,
        ::ckernel::VectorMode::RC)));
};

template <>
inline void pack_compute_activation<ttnn::experimental::prim::detail::MoEActivationFunction::SWIGLU>() {
    PACK((llk_math_eltwise_binary_sfpu_swiglu<false>(0, 1, 0)));
    PACK((llk_math_eltwise_binary_sfpu_swiglu<false>(2, 3, 2)));
};

template <>
inline void pack_compute_activation<ttnn::experimental::prim::detail::MoEActivationFunction::GELU>() {
    // GELU programs an SFPU LUT (gelu_init). The trailing binary MUL below clobbers that LUT,
    // so when the activation loop runs >1 iteration per chunk (tiles_per_step > 2, which happens
    // for ring sizes where ceil(Nt/ring) is odd — e.g. gemma at ring=8) the next iteration's
    // gelu reads a stale LUT and produces garbage. Re-init the LUT here so every gelu is valid.
    // SILU/SWIGLU don't use this LUT, so they keep their cheaper once-per-chunk init.
    PACK((llk_math_eltwise_unary_sfpu_init<SfpuType::gelu>(ckernel::sfpu::gelu_init<true, false>)));
    PACK(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_gelu,
        (true /*APPROXIMATE*/, false /*is_fp32_dest_acc_en*/, 8 /*ITERATIONS*/),
        0 /*DST_IDX*/,
        ::ckernel::VectorMode::RC));
    PACK(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_gelu,
        (true /*APPROXIMATE*/, false /*is_fp32_dest_acc_en*/, 8 /*ITERATIONS*/),
        2 /*DST_IDX*/,
        ::ckernel::VectorMode::RC));

    PACK((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (true /*APPROXIMATE*/, ckernel::BinaryOp::MUL, 8 /*ITERATIONS*/),
        0 /*DST_IN0*/,
        1 /*DST_IN1*/,
        0 /*DST_OUT*/,
        ::ckernel::VectorMode::RC)));
    PACK((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_sfpu_binary,
        (true /*APPROXIMATE*/, ckernel::BinaryOp::MUL, 8 /*ITERATIONS*/),
        2 /*DST_IN0*/,
        3 /*DST_IN1*/,
        2 /*DST_OUT*/,
        ::ckernel::VectorMode::RC)));
};

}  // namespace detail
void kernel_main() {
    constexpr bool has_bias = get_named_compile_time_arg_val("has_bias") == 1;
    constexpr uint32_t Ht = get_named_compile_time_arg_val("hidden_tiles");
    constexpr uint32_t Nt = get_named_compile_time_arg_val("intermediate_tiles");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");

    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    [[maybe_unused]] constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_shared_experts = get_named_compile_time_arg_val("num_shared_experts");
    constexpr uint32_t shared_expert_tp_factor = get_named_compile_time_arg_val("shared_expert_tp_factor");

    constexpr auto activation_type =
        ttnn::experimental::prim::detail::MoEActivationFunction(get_named_compile_time_arg_val("activation_function"));

    // For synchronization with tilize cores
    constexpr uint32_t tokens_per_chunk = get_named_compile_time_arg_val("tokens_per_chunk");

    // Run-time arguments
    uint32_t argidx = 0;
    [[maybe_unused]] const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    [[maybe_unused]] const auto vchannel = get_arg_val<uint32_t>(argidx++);
    [[maybe_unused]] const auto w0_w1_addr = get_arg_val<uint32_t>(argidx++);
    [[maybe_unused]] const auto w2_addr = get_arg_val<uint32_t>(argidx++);
    [[maybe_unused]] const auto out_addr = get_arg_val<uint32_t>(argidx++);
    [[maybe_unused]] const auto ring_semaphore_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_core_id = get_arg_val<uint32_t>(argidx++);
    [[maybe_unused]] const auto ring_neighbor_physical_x = get_arg_val<uint32_t>(argidx++);
    [[maybe_unused]] const auto ring_neighbor_physical_y = get_arg_val<uint32_t>(argidx++);

    // CB ids
    constexpr auto cb_s2c_in_id = tt::CBIndex::c_0;     // tilize_output_cb_id
    constexpr auto cb_r2c_w0_w1_id = tt::CBIndex::c_3;  // cb_r2c_w0
    constexpr auto cb_c2w_rdy_id = tt::CBIndex::c_4;
    constexpr auto cb_w2c_rdy_id = tt::CBIndex::c_5;
    constexpr auto cb_s2c_in2_id = tt::CBIndex::c_6;
    constexpr auto cb_w2c_md_id = tt::CBIndex::c_7;

    constexpr auto cb_c2s_out_id = tt::CBIndex::c_1;  // matmul_writer_cb_id
    // c_8 is unused on matmul cores when bias is off (CB not allocated); when has_bias, tilize uses c_8 only on
    // tilize cores. Only referenced in if constexpr(has_bias) branches below.
    constexpr auto cb_c2c_ones_tile_id = tt::CBIndex::c_8;

    // CB Aliases
    constexpr auto cb_r2c_w2_id = tt::CBIndex::c_3;  // reuse cb_r2c_w0_w1

    // CircularBuffer typed wrappers
    CircularBuffer cb_s2c_in(cb_s2c_in_id);
    CircularBuffer cb_r2c_w0_w1(cb_r2c_w0_w1_id);
    CircularBuffer cb_c2w_rdy(cb_c2w_rdy_id);
    CircularBuffer cb_w2c_rdy(cb_w2c_rdy_id);
    CircularBuffer cb_s2c_in2(cb_s2c_in2_id);
    CircularBuffer cb_w2c_md(cb_w2c_md_id);
    CircularBuffer cb_c2s_out(cb_c2s_out_id);
    CircularBuffer cb_c2c_ones_tile(cb_c2c_ones_tile_id);
    CircularBuffer cb_r2c_w2(cb_r2c_w2_id);

    // Pre-computed shard lookup tables — avoids runtime div/mod in inner loops.
    // ring_core_id is a runtime arg, but Nt/Ht/num_cores are compile-time.
    constexpr auto shard_tiles_lut = moe_ring::make_shard_lut<Nt, num_cores>();
    constexpr auto w2_shard_tiles_lut = moe_ring::make_w2_shard_lut<Ht, Nt, num_cores>();

    // Constants for MoE — derived from compile-time shape args
    constexpr uint32_t num_w0_w1_tiles_h = Ht;
    constexpr uint32_t num_w2_tiles_h = Nt;

    [[maybe_unused]] const uint32_t num_w0_w1_tiles_w = shard_tiles_lut[ring_core_id];
    const uint32_t num_w2_tiles_w = w2_shard_tiles_lut[ring_core_id];

    [[maybe_unused]] const uint32_t num_in2_tiles = num_w2_tiles_w;
    [[maybe_unused]] const uint32_t num_mm2_tiles = num_w2_tiles_w;

    //-------------------------------------------------------------------------
    // W0 and W1 reading constants
    //-------------------------------------------------------------------------
    constexpr uint32_t w0_w1_txns_per_block = moe_ring::W0_W1_TXNS_PER_BLOCK;
    constexpr uint32_t w0_w1_tiles_per_txn = moe_ring::W0_W1_TILES_PER_TXN;
    constexpr uint32_t w0_w1_tiles_per_block = w0_w1_tiles_per_txn * w0_w1_txns_per_block;  // 14 * 2 = 28

    using Cfg = moe_ring::MoeRingConfig<Ht, Nt, num_cores, has_bias, shared_expert_tp_factor>;

    // W2 reading constants (base-constant aliases only; derived values come from Cfg)
    constexpr auto w2_tiles_per_iter_w = moe_ring::W2_TILES_PER_A2A_ITER_W;
    constexpr uint32_t w2_tiles_per_block = moe_ring::W2_TILES_PER_TXN * moe_ring::W2_TXNS_PER_BLOCK;  // 14 * 2 = 28
    [[maybe_unused]] constexpr uint32_t w2_tiles_per_iter_h = moe_ring::W2_TILES_PER_A2A_ITER_H;

    //-------------------------------------------------------------------------
    // Ring setup
    //-------------------------------------------------------------------------
    constexpr uint32_t w2_blocks_per_a2a_iter = Cfg::w2_blocks_per_expert / Cfg::num_a2a_iters;

    [[maybe_unused]] constexpr uint32_t num_a2a_steps_per_iter = num_cores;

    constexpr uint32_t tiles_per_step = Cfg::in2_tiles_per_step;
    constexpr uint32_t tiles_per_step_shared = Cfg::in2_tiles_per_step_shared;

    //-------------------------------------------------------------------------
    // Compute
    //-------------------------------------------------------------------------
    // compute_kernel_hw_startup must be the first compute API call; the has_bias block below
    // issues compute work, so the startup is hoisted above it (otherwise it would be mid-kernel).
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_s2c_in_id, cb_r2c_w0_w1_id, cb_s2c_in2_id);

    if constexpr (has_bias) {
        // Create a ones-tile for bias addition (matmul with ones × bias_row = bias).
        // Same sequence as moe_gpt compute.cpp for GPT-OSS compatibility.
        compute_kernel_hw_startup(cb_c2c_ones_tile_id, cb_c2c_ones_tile_id);
        copy_init(cb_c2c_ones_tile_id);
        tile_regs_acquire();
        fill_tile_init();
        constexpr uint32_t dst0 = 0;
        fill_tile(dst0, 1.f);
        tile_regs_commit();
        tile_regs_wait();
        cb_c2c_ones_tile.reserve_back(1);
        pack_tile(dst0, cb_c2c_ones_tile_id);
        tile_regs_release();
        cb_c2c_ones_tile.push_back(1);
        // Synchronize: UNPACK must wait for PACK to finish writing before the first
        // matmul_block read. The tile is never popped so one cb_wait_front suffices.
        cb_c2c_ones_tile.wait_front(1);
    }

    // Pack is always configured to Float16_b
    pack_reconfig_data_format(cb_s2c_in2_id);

    // Unpacker B is for input/activation and eltiwse inputs, so Float16_b
    reconfig_data_format_srcb(cb_s2c_in_id);

    // Unpacker A is for W0,W1 and W2, so Bf4_b
    reconfig_data_format_srca(cb_r2c_w0_w1_id);

    //-------------------------------------------------------------------------
    // Init synchronization with tilize cores
    //-------------------------------------------------------------------------

    // Receive num_tokens_per_expert from the tilize cores, wait for signal from writer that the data has arrived
    // We also use this CB to transfer (from the writer to compute) 2 semaphore addresses:
    // - 0: address of semaphore used to send metadata (number of tokens per expert)
    // - 1: address of semaphore used to notify matmuls cores that tilized chunks have arrived
    cb_w2c_md.wait_front(2);
    cb_w2c_md.pop_front(2);
    volatile tt_l1_ptr uint32_t* cb_w2c_md_read_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_tile_address(cb_w2c_md_id, 0));

    // Read per-expert token counts from CB
    volatile tt_l1_ptr uint32_t* num_tokens_per_expert_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_w2c_md_read_ptr[0]);

    // Precompute NUM_CHUNKS_PER_EXPERT
    uint32_t NUM_CHUNKS_PER_EXPERT[num_experts];
    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        uint32_t num_tokens = num_tokens_per_expert_ptr[expert_id];
        NUM_CHUNKS_PER_EXPERT[expert_id] = (num_tokens + tokens_per_chunk - 1) / tokens_per_chunk;
    }

    // Value we wait on that indicates the next chunk of tiles have arrived from the tilize cores
    uint32_t matmul_chunk_ready_semaphore_wait_value = 1;
    uint32_t matmul_chunk_ready_semaphore_addr = cb_w2c_md_read_ptr[1];

    // Zero out dest registers
    MATH((ckernel::zeroacc()));

    //-------------------------------------------------------------------------
    // Expert loop
    //-------------------------------------------------------------------------

    // This decides which half of the buffer will have the valid data sent by tilize cores
    bool use_second_half_buffer = false;
    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        uint32_t num_expert_chunks = NUM_CHUNKS_PER_EXPERT[expert_id];
        for (uint32_t chunk = 0; chunk < num_expert_chunks; ++chunk) {
            // GELU re-inits its SFPU LUT inside pack_compute_activation on every iteration (the
            // trailing MUL there clobbers it), so it's its own initializer — skip the per-chunk
            // init for GELU to avoid a redundant gelu_init. SILU/SWIGLU init once here.
            if constexpr (activation_type != ttnn::experimental::prim::detail::MoEActivationFunction::GELU) {
                ::detail::pack_init_activation<activation_type>();
            }

            // Initialize matmul for W0
            matmul_block_init(
                cb_s2c_in_id, cb_r2c_w0_w1_id, /*transpose=*/false, /*ct_dim=*/4, /*rt_dim=*/1, /*kt_dim=*/1);

            // Wait for next chunk of tiles to arrive from the tilize cores
            // Min to allow tilize cores to send increment for second expert
            // while first expert still being processed
            ::detail::noc_semaphore_wait_min(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(matmul_chunk_ready_semaphore_addr),
                matmul_chunk_ready_semaphore_wait_value++);

            //---------------------------------------------------------------------
            // Compute in @ {W0,W1}
            //---------------------------------------------------------------------
            // Shared experts are TP-split + front-packed: produce only the real TpNt prefix
            // (dm0 reads the matching shortened W0/W1), then zero-fill the rest of the full
            // tiles_per_step stride below. The unchanged full W2 walk then contracts real×real
            // in the prefix and (zero in2)×(front-packed zero W2) past it.
            const bool is_shared_expert = expert_id >= num_experts - num_shared_experts;
            const uint32_t prod_tiles_per_step = is_shared_expert ? tiles_per_step_shared : tiles_per_step;
            for (uint32_t tile_id = 0; tile_id < prod_tiles_per_step; tile_id += 2) {
                uint32_t in0_index = use_second_half_buffer ? num_w0_w1_tiles_h : 0;

                tile_regs_acquire();
                [[maybe_unused]] uint32_t k_tracker = 0;
                for (uint32_t block_id = 0; block_id < Cfg::w0_w1_blocks_per_col; ++block_id) {
                    cb_r2c_w0_w1.wait_front(w0_w1_tiles_per_block);

                    for (uint32_t k = 0; k < w0_w1_tiles_per_block; k += 4) {
                        if constexpr (has_bias) {
                            if (k_tracker == num_w0_w1_tiles_h) {
                                // Bias addition: matmul(ones_tile, bias_row)
                                matmul_block(
                                    cb_c2c_ones_tile_id,
                                    cb_r2c_w0_w1_id,
                                    0,
                                    /*in1_index=*/k,
                                    /*idst=*/0,
                                    /*transpose=*/false,
                                    /*ct_dim=*/4,
                                    /*rt_dim=*/1,
                                    /*kt_dim=*/1);
                                k_tracker++;
                                continue;
                            } else if (k_tracker > num_w0_w1_tiles_h) {
                                k_tracker++;
                                continue;  // skip padding K slots after bias
                            }
                        }
                        matmul_block(
                            cb_s2c_in_id,
                            cb_r2c_w0_w1_id,
                            in0_index++,
                            /*in1_index=*/k,
                            /*idst=*/0,
                            /*transpose=*/false,
                            /*ct_dim=*/4,
                            /*rt_dim=*/1,
                            /*kt_dim=*/1);
                        if constexpr (has_bias) {
                            k_tracker++;
                        }
                    }
                    cb_r2c_w0_w1.pop_front(w0_w1_tiles_per_block);
                }

                tile_regs_commit();

                // The below is equivalent to tile_regs_wait(), but we stall CFG as well, so that the succeeding
                // TT_SETC16 instruction is also stalled until math thread is done with these dest registers.
                PACK(TTI_SEMWAIT(
                    p_stall::STALL_TDMA | p_stall::STALL_CFG,
                    semaphore::t6_sem(semaphore::MATH_PACK),
                    p_stall::STALL_ON_ZERO));

                // Make SFPU access the appropriate half of the destination registers
                PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));

                //---------------------------------------------------------------------
                // Apply activation
                //---------------------------------------------------------------------
                ::detail::pack_compute_activation<activation_type>();

                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));

                pack_tile</*out_of_order_output=*/true>(0, cb_s2c_in2_id, /*output_tile_index=*/tile_id);
                pack_tile</*out_of_order_output=*/true>(2, cb_s2c_in2_id, /*output_tile_index=*/tile_id + 1);
                tile_regs_release();
            }

            // Zero-fill the unproduced tail [prod_tiles_per_step, tiles_per_step) of this core's in2
            // stride for shared experts, so the full W2 walk reads zeros there (annihilated by the
            // front-packed zero W2 rows) rather than stale data from a prior expert.
            if (is_shared_expert) {
                tile_regs_acquire();
                fill_tile_init();
                fill_tile(0, 0.0f);
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t tile_id = prod_tiles_per_step; tile_id < tiles_per_step; ++tile_id) {
                    pack_tile</*out_of_order_output=*/true>(0, cb_s2c_in2_id, /*output_tile_index=*/tile_id);
                }
                tile_regs_release();
            }

            // Signal to DM1 that the output from this core is ready
            cb_c2w_rdy.reserve_back(1);
            cb_c2w_rdy.push_back(1);

            //---------------------------------------------------------------------
            // Compute in2 @ W2 (in pairs of 4)
            //---------------------------------------------------------------------

            cb_c2s_out.reserve_back(num_w0_w1_tiles_h);

            // Init pack_untilize ONCE before the iter loop (hoisted, mirrors moe_gpt pattern).
            // Cycling init/uninit per-iter triggers BH's MATH reconfig_remap workaround
            // (pack_untilize.h:66-80, tt-metal#17132) which races with in-flight PACR/MOP
            // execution and produces garbage output (NaN/Inf) on BH silicon.
            pack_untilize_dest_init<
                /*block_ct_dim=*/w2_tiles_per_iter_w,
                /*full_ct_dim=*/Cfg::w2_tiles_per_expert_w>(cb_c2s_out_id);

            for (uint32_t iter = 0; iter < Cfg::num_a2a_iters; ++iter) {
                uint32_t src_core = ring_core_id;
                uint32_t dm1_tiles_remaining = shard_tiles_lut[ring_core_id];
                cb_w2c_rdy.wait_front(1);

                uint32_t in2_offset = 0, in2_index = 0;

                tile_regs_acquire();

                uint32_t w2_k_tracker = 0;
                for (uint32_t block_id = 0; block_id < w2_blocks_per_a2a_iter; ++block_id) {
                    cb_r2c_w2.wait_front(w2_tiles_per_block);
                    for (uint32_t k = 0; k < w2_tiles_per_block; k += w2_tiles_per_iter_w) {
                        if constexpr (has_bias) {
                            if (w2_k_tracker == num_w2_tiles_h) {
                                // Bias addition: matmul(ones_tile, bias_row); padding K slots do not consume in2/dm1.
                                matmul_block(
                                    cb_c2c_ones_tile_id,
                                    cb_r2c_w2_id,
                                    0,
                                    /*in1_index=*/k,
                                    /*idst=*/0,
                                    /*transpose=*/false,
                                    /*ct_dim=*/4,
                                    /*rt_dim=*/1,
                                    /*kt_dim=*/1);
                                w2_k_tracker++;
                                continue;
                            }
                        }
                        if (w2_k_tracker >= num_w2_tiles_h) {
                            continue;  // skip padding K slots (bias: after bias tile; no_bias: at/past logical K)
                        }
                        if (dm1_tiles_remaining == 0) {
                            cb_w2c_rdy.pop_front(1);
                            cb_w2c_rdy.wait_front(1);
                            src_core = (src_core == 0) ? num_cores - 1 : src_core - 1;
                            dm1_tiles_remaining = shard_tiles_lut[src_core];
                            in2_offset += tiles_per_step;
                            in2_index = in2_offset;
                        }
                        dm1_tiles_remaining--;

                        matmul_block(
                            cb_s2c_in2_id,
                            cb_r2c_w2_id,
                            in2_index++,
                            /*in1_index=*/k,
                            /*idst=*/0,
                            /*transpose=*/false,
                            /*ct_dim=*/4,
                            /*rt_dim=*/1,
                            /*kt_dim=*/1);
                        w2_k_tracker++;
                    }
                    cb_r2c_w2.pop_front(w2_tiles_per_block);
                }
                cb_w2c_rdy.pop_front(1);

                tile_regs_commit();

                tile_regs_wait();
                pack_untilize_dest</*block_ct_dim=*/w2_tiles_per_iter_w, /*full_ct_dim=*/Cfg::w2_tiles_per_expert_w>(
                    cb_c2s_out_id, /*block_rt_dim=*/1, /*block_c_index=*/iter);

                tile_regs_release();
            }

            // Uninit pack_untilize ONCE after the iter loop (hoisted, mirrors moe_gpt pattern).
            pack_untilize_uninit(cb_c2s_out_id);

            cb_c2s_out.push_back(num_w0_w1_tiles_h);

            // Toggle the buffer to use
            use_second_half_buffer = !use_second_half_buffer;
            // Restore packer data format for next chunk's activation pipeline (mirrors moe_gpt:342).
            pack_reconfig_data_format(cb_s2c_in2_id);

        }  // end for (chunk)
    }  // end for (expert_id)

    // Drain the pipeline - the last dummy push
    cb_r2c_w2.wait_front(w2_tiles_per_block);
    cb_r2c_w2.pop_front(w2_tiles_per_block);
}
