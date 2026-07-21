// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "internal/mod_div_lib.h"

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/compile_time_arg_tmp.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers_advanced.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reblock_untilize_helpers.hpp"

enum class CORE_TYPE : uint8_t { IDLE_CORE = 0, WORKER_CORE = 1, HOP_CORE = 2 };

// ── Local CB rd_ptr utilities (used by ENABLE_GLOBAL_CB ring management) ─────
// reload_from_cb_to_dst is gone — the matmul_block helper does its own reload
// (copy_tile_to_dst_init_short_with_dt + copy_block_matmul_partials +
// reconfig_data_format_srca + matmul_block_init) inline as part of its K-loop.

FORCE_INLINE uint32_t get_local_cb_rd_ptr(uint32_t cb_id) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    return local_cb.fifo_rd_ptr;
}

FORCE_INLINE void update_local_cb_rd_ptr(uint32_t cb_id, uint32_t val) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    local_cb.fifo_rd_ptr = val;
}

FORCE_INLINE uint32_t get_local_cb_start_addr(uint32_t cb_id) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    uint32_t fifo_size = local_cb.fifo_size;
    uint32_t fifo_limit = local_cb.fifo_limit;
    uint32_t fifo_start_addr = fifo_limit - fifo_size;
    return fifo_start_addr;
}

FORCE_INLINE bool is_tensor_split(uint32_t cb_id, uint32_t tensor_size_bytes) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    uint32_t fifo_rd_ptr = local_cb.fifo_rd_ptr;
    uint32_t fifo_limit = local_cb.fifo_limit;
    bool split = (fifo_limit - fifo_rd_ptr) < tensor_size_bytes / L1_ALIGNMENT;
    return split;
}

FORCE_INLINE void calculate_next_block_index_and_update_rd_ptr(
    uint32_t cb_id,
    uint32_t num_blocks,
    uint32_t block_size_bytes,
    uint32_t curr_block_index,
    uint32_t cb_start_addr,
    uint32_t rd_ptr_start_addr,
    bool tensor_split,
    uint32_t* updated_block_index,
    uint32_t* updated_rd_ptr) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    uint32_t next_block_index = curr_block_index + 1;
    uint32_t next_fifo_rd_ptr = local_cb.fifo_rd_ptr;
    uint32_t block_size_bytes_aligned = block_size_bytes / L1_ALIGNMENT;
    bool reach_limit = local_cb.fifo_rd_ptr == local_cb.fifo_limit;
    bool last_block = curr_block_index == (num_blocks - 1);
    if (tensor_split) {
        if (reach_limit) {
            local_cb.fifo_rd_ptr = cb_start_addr;
            if (last_block) {
                next_block_index = 0;
                next_fifo_rd_ptr = rd_ptr_start_addr;
            } else {
                next_fifo_rd_ptr = cb_start_addr + block_size_bytes_aligned;
            }
        } else {
            if (last_block) {
                next_block_index = 0;
                next_fifo_rd_ptr = rd_ptr_start_addr;
            } else {
                next_fifo_rd_ptr += block_size_bytes_aligned;
            }
        }
    } else {
        if (last_block) {
            next_block_index = 0;
            next_fifo_rd_ptr = rd_ptr_start_addr;
        } else {
            next_fifo_rd_ptr += block_size_bytes_aligned;
        }
    }
    *updated_block_index = next_block_index;
    *updated_rd_ptr = next_fifo_rd_ptr;
}

FORCE_INLINE void update_rd_ptr_to_ring_index(
    uint32_t cb_id, uint32_t block_size_bytes, uint32_t ring_index, bool tensor_split) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);

    if (tensor_split) {
        if ((local_cb.fifo_rd_ptr + ring_index * block_size_bytes / L1_ALIGNMENT) >= local_cb.fifo_limit) {
            uint32_t fifo_size = local_cb.fifo_size;
            uint32_t fifo_limit = local_cb.fifo_limit;
            uint32_t fifo_start_addr = fifo_limit - fifo_size;
            uint32_t fifo_size_skip_bytes = local_cb.fifo_rd_ptr - fifo_start_addr;
            local_cb.fifo_rd_ptr =
                fifo_start_addr +
                (fifo_size_skip_bytes + ring_index * block_size_bytes / L1_ALIGNMENT) % local_cb.fifo_size;

        } else {
            local_cb.fifo_rd_ptr = local_cb.fifo_rd_ptr + ring_index * block_size_bytes / L1_ALIGNMENT;
        }
    } else {
        local_cb.fifo_rd_ptr = local_cb.fifo_rd_ptr + ring_index * block_size_bytes / L1_ALIGNMENT;
    }
}

// ── Functors threaded into compute_kernel_lib::matmul_block ─────────────────

#ifdef SFPU_OP_INIT_ACTIVATION
// Per-output-subblock SFPU on the last K-block. Helper invokes this via
// PostComputeFn after the FMA accumulation lands in DST and before pack.
struct SFPUPostCompute {
    ALWI void operator()(uint32_t num_tiles) const {
        for (uint32_t i = 0; i < num_tiles; i++) {
            SFPU_OP_FUNC_ACTIVATION
        }
    }
};
#endif

// Per-K-block PreKBlockFn. Two responsibilities, both gated by compile-time switches:
//   ENABLE_GLOBAL_CB → compute the next in1 ring rd_ptr (UNPACK-only side effect).
//   PACK_RELU + untilize_out → enable ZERO_RELU on the last K-block (the untilize
//      path via the Interm target doesn't auto-enable relu the way OutWithRelu does).
template <bool EnableGlobalCb, bool EnableReluOnLast, uint32_t In1CbId, uint32_t NumBlocks, uint32_t In1BlockSizeBytes>
struct RingPreKBlock {
    uint32_t* curr_block_index_ptr;
    uint32_t cb_start_addr;
    uint32_t rd_ptr_start_addr;
    bool tensor_split;
    uint32_t* next_block_index_ptr;
    uint32_t* next_rd_ptr_addr_ptr;

    ALWI void operator()(uint32_t /*block*/, uint32_t /*num_k_blocks*/, bool last_out) const {
        if constexpr (EnableReluOnLast) {
            if (last_out) {
                PACK((llk_pack_relu_config(ReluConfig::zero())));
            }
        }
        if constexpr (EnableGlobalCb) {
            UNPACK((calculate_next_block_index_and_update_rd_ptr(
                In1CbId,
                NumBlocks,
                In1BlockSizeBytes,
                *curr_block_index_ptr,
                cb_start_addr,
                rd_ptr_start_addr,
                tensor_split,
                next_block_index_ptr,
                next_rd_ptr_addr_ptr)));
        }
    }
};

// Per-K-block PostKBlockFn. ENABLE_GLOBAL_CB only — commits the next ring rd_ptr
// computed by RingPreKBlock. Fires after the helper's in0/in1 pop_front and after
// the L1_ACC drain, so the rd_ptr advance observes a fully-consumed K-block.
template <bool EnableGlobalCb, uint32_t In1CbId>
struct RingPostKBlock {
    uint32_t* curr_block_index_ptr;
    const uint32_t* next_block_index_ptr;
    const uint32_t* next_rd_ptr_addr_ptr;

    ALWI void operator()(uint32_t /*block*/, uint32_t /*num_k_blocks*/, bool /*last_out*/) const {
        if constexpr (EnableGlobalCb) {
            *curr_block_index_ptr = *next_block_index_ptr;
            UNPACK((update_local_cb_rd_ptr(In1CbId, *next_rd_ptr_addr_ptr)));
        }
    }
};

// Per-K-block In1BaseOffsetFn. The pre-migration kernel selected:
//   ENABLE_GLOBAL_CB    → 0  (rd_ptr rotation handles ring positioning)
//   in1_is_dram         → 0  (per-K-block wait/pop fronts the right block)
//   else (Case 4)       → in1_block_num_tiles * curr_ring_idx
//                          (single fronted region; per-K-block offset shift)
// Compile-time NeedsOffset folds the first two arms to NoIn1BaseOffset's default.
template <bool NeedsOffset, uint32_t In1BlockNumTiles, uint32_t RingSize>
struct RingIn1BaseOffsetFn {
    uint32_t ring_idx;

    ALWI uint32_t operator()(uint32_t block) const {
        if constexpr (!NeedsOffset) {
            return 0;
        } else {
            const uint32_t curr_ring_idx = (ring_idx - block + RingSize) % RingSize;
            return In1BlockNumTiles * curr_ring_idx;
        }
    }
};

void kernel_main() {
    using namespace compute_kernel_lib;

    // Compile time args
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);        // inner block size in tiles
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);  // outer row block size (in inner row blocks)
    constexpr uint32_t in0_block_num_tiles =
        get_compile_time_arg_val(2);  // out_subblock_h*in0_block_w*in0_num_subblocks;
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
    constexpr uint32_t in1_num_subblocks =
        get_compile_time_arg_val(4);  // outer column block size (in inner column blocks)
    constexpr uint32_t in1_block_num_tiles =
        get_compile_time_arg_val(5);  // out_subblock_w*in0_block_w* in1_num_subblocks;
    constexpr uint32_t in1_block_size_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t in1_tensor_size_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t in1_per_core_w = get_compile_time_arg_val(8);           // out_subblock_w*in1_num_subblocks
    constexpr uint32_t num_blocks = get_compile_time_arg_val(9);               // outer inner dim (in inner dim blocks)
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(10);          // inner row block size in tiles
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(11);          // inner column block size in tiles
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(12);  // out_subblock_h * out_subblock_w;
    constexpr uint32_t batch = get_compile_time_arg_val(13);                   // batch dim
    constexpr uint32_t out_block_num_tiles = get_compile_time_arg_val(14);     // number of tiles in out_block
    constexpr bool untilize_out = get_compile_time_arg_val(15);                // untilize output
    constexpr bool in1_is_dram_interleaved = get_compile_time_arg_val(16);     // in1 is in dram
    constexpr bool in1_is_dram_sharded = get_compile_time_arg_val(17);
    constexpr uint32_t in0_cb_id = get_compile_time_arg_val(18);
    constexpr uint32_t in1_cb_id = get_compile_time_arg_val(19);
    constexpr uint32_t in2_cb_id = get_compile_time_arg_val(20);
    constexpr uint32_t sync_cb = get_compile_time_arg_val(21);
    constexpr uint32_t sync_cb2 = get_compile_time_arg_val(22);
    constexpr uint32_t OUTPUT_CB_ARRAY_IDX = get_compile_time_arg_val(23);
    constexpr std::array<uint32_t, batch> mm_out_cb_ids =
        fill_array_with_next_n_args<uint32_t, OUTPUT_CB_ARRAY_IDX, batch>();
    constexpr uint32_t INTERM_CB_ARRAY_IDX = OUTPUT_CB_ARRAY_IDX + batch;
    constexpr std::array<uint32_t, batch> mm_partials_cb_ids =
        fill_array_with_next_n_args<uint32_t, INTERM_CB_ARRAY_IDX, batch>();
    (void)in2_cb_id;  // unused — original kernel notes "no need to use in2_cb_id"

    constexpr uint32_t ring_size = num_blocks;
    constexpr bool in1_is_dram = in1_is_dram_interleaved || in1_is_dram_sharded;

    // Compile-time helper switches. ENABLE_GLOBAL_CB and PACK_RELU are #defines from
    // the host factory; lift them into constexpr bools so the helper-arg selection
    // below uses if constexpr cleanly.
#ifdef ENABLE_GLOBAL_CB
    constexpr bool enable_global_cb = true;
#else
    constexpr bool enable_global_cb = false;
#endif

#ifdef PACK_RELU
    constexpr bool pack_relu_defined = true;
#else
    constexpr bool pack_relu_defined = false;
#endif

#ifdef PACKER_L1_ACC
    constexpr bool l1_acc = true;
#else
    constexpr bool l1_acc = false;
#endif

#ifdef IN1_TRANSPOSE_TILE
    constexpr bool in1_transpose_tile = true;
#else
    constexpr bool in1_transpose_tile = false;
#endif

    // LastBlockTarget selection. FUSE_BIAS is host-side dead code in this kernel
    // (llama_1d_mm_fusion.cpp never sets it; the original kernel had `#if not
    // defined FUSE_BIAS` guards but no positive FUSE_BIAS branch), so the
    // FUSE_BIAS arm from the original migration plan is moot. Routes:
    //   untilize_out → Interm + downstream reblock_and_untilize (see after the matmul call)
    //   PACK_RELU    → OutWithRelu      (helper's pack_relu mechanism)
    //   else         → Out
    // untilize_out uses Interm (not the old fused OutWithUntilize): the matmul accumulates the
    // fully-tiled block into interm, then a downstream reblock_and_untilize untilizes the
    // FULLY-accumulated block into out. The fused pack_untilize untilized a partial before the
    // block finished accumulating, corrupting under packer_l1_acc.
    constexpr LastBlockTarget last_block_target =
        untilize_out ? LastBlockTarget::Interm
                     : (pack_relu_defined ? LastBlockTarget::OutWithRelu : LastBlockTarget::Out);

    // LastBlockTarget::Interm does NOT auto-enable relu, and the downstream reblock does not apply
    // activation, so keep the PreKBlockFn relu-on-last for the untilize path — relu still runs on
    // the interm block before untilize (matches the pre-migration behavior).
    constexpr bool enable_relu_on_last_via_pre_k = pack_relu_defined && untilize_out;

    // in1_policy mirrors the pre-migration kernel's `if constexpr (in1_is_dram)`
    // gates on the per-K-block cb_wait_front/cb_pop_front. The wait/pop is keyed on
    // in1_is_dram, not on ENABLE_GLOBAL_CB — global-CB receivers manage in1 lifecycle
    // via fabric pushes + manual rd_ptr advance from PostKBlockFn.
    constexpr InputPolicy in1_policy_const = in1_is_dram ? InputPolicy::WaitAndPopPerKBlock : InputPolicy::NoWaitNoPop;

    // The per-K-block in1 base offset is non-zero only on the
    // (`!ENABLE_GLOBAL_CB && !in1_is_dram`) path, where a single fronted region
    // holds all ring positions and the kernel rotates between them via offset
    // arithmetic instead of rd_ptr advance.
    constexpr bool needs_in1_base_offset = !enable_global_cb && !in1_is_dram;

    // Runtime args
    uint32_t rt_args_idx = 0;
    uint32_t core_type = get_arg_val<uint32_t>(rt_args_idx++);
    if (core_type == (uint32_t)CORE_TYPE::IDLE_CORE || core_type == (uint32_t)CORE_TYPE::HOP_CORE) {
        return;
    }
    uint32_t ring_idx = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t* unpadded_in0_shard_widths_in_tiles = (uint32_t*)get_arg_addr(rt_args_idx);
    rt_args_idx += ring_size;
    (void)unpadded_in0_shard_widths_in_tiles;  // unused — original kernel notes "no need to unpad in0"

#ifdef SFPU_OP_INIT_ACTIVATION
    SFPU_OP_INIT_ACTIVATION
#endif

    // Buf wrappers for the helper's wait_front / pop_front / LLK call hygiene.
    CircularBuffer in0_buf(in0_cb_id);
    CircularBuffer in1_buf(in1_cb_id);

    // Boot-time matmul init: compute_kernel_hw_startup does the one hw_configure MMIO, then
    // matmul_block_init sets up unpack/math matmul state (mm_block_init is deprecated). The helper
    // invocation below uses InitMode::None so it doesn't re-init each call; the per-batch
    // pack_reconfig_data_format below re-binds the packer to the current batch's partials CB without
    // disturbing the unpacker matmul state.
    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb_id, in1_cb_id, mm_partials_cb_ids[0]);
    matmul_block_init(in0_cb_id, in1_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);

    // Per-batch ring CB state (used by ENABLE_GLOBAL_CB; declared at the outer
    // scope so the functors can hold pointers into it across helper invocations).
    [[maybe_unused]] uint32_t in1_cb_start_addr = 0;
    [[maybe_unused]] uint32_t in1_rd_ptr_start_addr = 0;
    [[maybe_unused]] uint32_t curr_in1_block_index = 0;
    [[maybe_unused]] bool in1_tensor_split = false;
    [[maybe_unused]] uint32_t next_in1_block_index = 0;
    [[maybe_unused]] uint32_t next_in1_rd_ptr_addr = 0;

    using PreFn =
        RingPreKBlock<enable_global_cb, enable_relu_on_last_via_pre_k, in1_cb_id, num_blocks, in1_block_size_bytes>;
    using PostFnRing = RingPostKBlock<enable_global_cb, in1_cb_id>;
    using In1OffsetFn = RingIn1BaseOffsetFn<needs_in1_base_offset, in1_block_num_tiles, ring_size>;

#ifdef SFPU_OP_INIT_ACTIVATION
    using PostComputeFnT = SFPUPostCompute;
#else
    using PostComputeFnT = NoPostCompute;
#endif

    for (uint32_t b = 0; b < batch; b++) {
        if constexpr (enable_global_cb) {
            UNPACK((in1_cb_start_addr = get_local_cb_start_addr(in1_cb_id)));
            UNPACK((in1_rd_ptr_start_addr = get_local_cb_rd_ptr(in1_cb_id)));
            UNPACK((curr_in1_block_index = ring_idx));
            UNPACK((in1_tensor_split = is_tensor_split(in1_cb_id, in1_tensor_size_bytes)));
            UNPACK((update_rd_ptr_to_ring_index(in1_cb_id, in1_block_size_bytes, ring_idx, in1_tensor_split)));
        }

        const uint32_t mm_out_cb_id = mm_out_cb_ids[b];
        const uint32_t mm_partials_cb_id = mm_partials_cb_ids[b];

        CircularBuffer mm_out_buf(mm_out_cb_id);
        CircularBuffer mm_partials_buf(mm_partials_cb_id);

#ifdef PACK_RELU
        // for each batch we start with relu disabled so that intermediate results are not relu'd
        if constexpr (batch > 1) {
            PACK((llk_pack_relu_config(ReluConfig::none())));
        }
#endif

        if constexpr (batch > 1) {
            PACK((pack_reconfig_data_format(mm_partials_cb_id)));
        }

        // Wait to receive in1 (sync_cb2 gates the entire batch — in1 data is staged
        // upfront via the all-gather producer, separate from the per-K-block
        // wait/pop the helper would manage on the WaitAndPopPerKBlock branch).
        cb_wait_front(sync_cb2, 1);
        cb_pop_front(sync_cb2, 1);

        PreFn pre_fn{
            /*curr_block_index_ptr=*/&curr_in1_block_index,
            /*cb_start_addr=*/in1_cb_start_addr,
            /*rd_ptr_start_addr=*/in1_rd_ptr_start_addr,
            /*tensor_split=*/in1_tensor_split,
            /*next_block_index_ptr=*/&next_in1_block_index,
            /*next_rd_ptr_addr_ptr=*/&next_in1_rd_ptr_addr,
        };
        PostFnRing post_fn{
            /*curr_block_index_ptr=*/&curr_in1_block_index,
            /*next_block_index_ptr=*/&next_in1_block_index,
            /*next_rd_ptr_addr_ptr=*/&next_in1_rd_ptr_addr,
        };
        In1OffsetFn in1_offset_fn{/*ring_idx=*/ring_idx};

        matmul_block_gathered<
            in1_transpose_tile,
            l1_acc,
            last_block_target,
            OutputCBLayout::SubblockMajor,
            matmul_config::InitMode::None,
            InputPolicy::WaitAndPopPerKBlock,
            in1_policy_const,
            matmul_config::DataFormatReconfig::InputAndOutput,  // reconfig (was defaulted)
            NoneActivation,                                     // Activation (was defaulted)
            PostComputeFnT,
            PreFn,
            PostFnRing,
            NoKBlockInnerDimFn,
            NoIn0Source,
            In1OffsetFn>(
            in0_buf,
            in1_buf,
            mm_out_buf,
            mm_partials_buf,
            MatmulBlockShape::of(
                in0_num_subblocks,
                in1_num_subblocks,
                out_subblock_h,
                out_subblock_w,
                in0_block_w,
                num_blocks,
                /*batch=*/1,
                /*in1_per_core_w=*/in1_per_core_w),  // out_row_width defaults to in1_per_core_w (SubblockMajor)
            PostComputeFnT{},
            pre_fn,
            post_fn,
            NoKBlockInnerDimFn{},
            NoIn0Source{},
            in1_offset_fn);

        // ── Downstream untilize (Interm target) ─────────────────────────────
        // matmul_block accumulated the fully-tiled block into mm_partials_buf (SubblockMajor).
        // reblock_and_untilize gathers it into row-major and untilizes into mm_out_buf — untilizing
        // the FULLY-accumulated block (fixes the packer_l1_acc corruption the old fused
        // OutWithUntilize had, where a partial got untilized before accumulation finished). Mirrors
        // the plain-gather kernel bmm_large_block_zm_fused_bias_activation_gathered.cpp.
        if constexpr (untilize_out) {
            constexpr uint32_t out_block_w = in1_num_subblocks * out_subblock_w;
#ifdef PACK_RELU
            // The PreKBlockFn (enable_relu_on_last_via_pre_k) enabled relu for the last K-block's
            // interm pack; restore a clean packer state before the untilize reblock.
            PACK((llk_pack_relu_config(ReluConfig::none())));
#endif
            // Reconfigure srcA / pack DF / l1_acc for the reblock read, then invoke with
            // NoReconfigure so the helper adds no reconfig of its own.
            reconfig_data_format_srca(in1_cb_id, mm_partials_cb_id);
#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
            PACK((pack_reconfig_data_format(mm_out_cb_id)));
#endif
#ifdef PACKER_L1_ACC
            PACK((llk_pack_reconfig_l1_acc(0)));
#endif
            reblock_and_untilize<
                out_subblock_w,
                out_block_w,
                reblock_untilize_config::InitUninitMode::InitAndUninit,
                reblock_untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(
                in0_num_subblocks,
                in1_num_subblocks,
                out_subblock_num_tiles,
                out_subblock_h,
                mm_partials_buf,
                mm_out_buf);
        }

        if constexpr (enable_global_cb) {
            // Release in1
            cb_reserve_back(sync_cb, 1);
            cb_push_back(sync_cb, 1);
            UNPACK(
                (update_local_cb_rd_ptr(in1_cb_id, in1_rd_ptr_start_addr)));  // reset rd_ptr back to the initial addr
            UNPACK((update_rd_ptr_to_ring_index(
                in1_cb_id, in1_block_size_bytes, ring_size, in1_tensor_split)));  // update to next tensor addr
        }
    }
}
