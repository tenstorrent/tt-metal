// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "internal/mod_div_lib.h"

#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers_advanced.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reblock_untilize_helpers.hpp"

#ifdef SFPU_ACTIVATION
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"
#endif

enum class CORE_TYPE : uint8_t { IDLE_CORE = 0, WORKER_CORE = 1, HOP_CORE = 2 };

// ── Local CB rd_ptr utilities (used by ENABLE_GLOBAL_CB ring management) ─────
// reload_from_cb_to_dst is gone — the matmul_block helper does its own reload
// (copy_tile_to_dst_init_short_with_dt + copy_block_matmul_partials +
// mm_block_init_short_with_dt) inline as part of its K-loop.

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

// Named CB arg lookup tables for batch-indexed output and partials CBs.
// The factory emits "cb_mm_out_0" .. "cb_mm_out_N" and "cb_mm_partials_0" .. "cb_mm_partials_N"
// as named compile-time args. These tables let fill_named_cb_array resolve them by index.
constexpr const char* mm_out_cb_names[] = {
    "cb_mm_out_0",
    "cb_mm_out_1",
    "cb_mm_out_2",
    "cb_mm_out_3",
    "cb_mm_out_4",
    "cb_mm_out_5",
    "cb_mm_out_6",
    "cb_mm_out_7",
    "cb_mm_out_8",
    "cb_mm_out_9",
    "cb_mm_out_10",
    "cb_mm_out_11",
    "cb_mm_out_12",
    "cb_mm_out_13",
    "cb_mm_out_14",
    "cb_mm_out_15",
};
constexpr const char* mm_partials_cb_names[] = {
    "cb_mm_partials_0",
    "cb_mm_partials_1",
    "cb_mm_partials_2",
    "cb_mm_partials_3",
    "cb_mm_partials_4",
    "cb_mm_partials_5",
    "cb_mm_partials_6",
    "cb_mm_partials_7",
    "cb_mm_partials_8",
    "cb_mm_partials_9",
    "cb_mm_partials_10",
    "cb_mm_partials_11",
    "cb_mm_partials_12",
    "cb_mm_partials_13",
    "cb_mm_partials_14",
    "cb_mm_partials_15",
};

template <uint32_t N>
constexpr std::array<uint32_t, N> fill_named_cb_array(const char* const* names) {
    std::array<uint32_t, N> arr{};
    for (uint32_t i = 0; i < N; ++i) {
        arr[i] = get_named_compile_time_arg_val(names[i]);
    }
    return arr;
}

// ── Functors threaded into compute_kernel_lib::matmul_block ─────────────────
//
// SFPU activation is fused via the matmul_block helper's Activation template
// parameter, built as ActivationOp<activation_type, activation_param0,
// activation_param1, activation_param2> from named CT args below. It runs on
// the packer thread (TRISC2) at
// the per-subblock pack stage of the last K-block — overlaps with the next
// math iteration and frees math-thread DST register pressure. No separate
// PostComputeFn struct is needed.

// Per-K-block PreKBlockFn. Three responsibilities, all gated by compile-time switches:
//   block-0 self-prime → reserve_back + push_back on the local in0 CB on block 0.
//   ENABLE_GLOBAL_CB   → compute the next in1 ring rd_ptr (UNPACK-only side effect).
//   PACK_RELU + untilize_out → enable ZERO_RELU on the last K-block (the untilize
//      path via the Interm target doesn't auto-enable relu the way OutWithRelu does).
template <
    bool EnableGlobalCb,
    bool EnableReluOnLast,
    uint32_t In0CbId,
    uint32_t In0BlockNumTiles,
    uint32_t In1CbId,
    uint32_t NumBlocks,
    uint32_t In1BlockSizeBytes>
struct GatheredPreKBlock {
    uint32_t* curr_block_index_ptr;
    uint32_t cb_start_addr;
    uint32_t rd_ptr_start_addr;
    bool tensor_split;
    uint32_t* next_block_index_ptr;
    uint32_t* next_rd_ptr_addr_ptr;

    ALWI void operator()(uint32_t block, uint32_t /*num_k_blocks*/, bool last_out) const {
        // Block-0 self-prime: the production gathered factory expects compute to
        // self-prime its local in0 CB for the first ring step, so on block 0 this
        // PreKBlockFn reserves+pushes one in0 block before the matmul consumes it.
        if (block == 0) {
            CircularBuffer in0_cb(In0CbId);
            in0_cb.reserve_back(In0BlockNumTiles);
            in0_cb.push_back(In0BlockNumTiles);
        }
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
// computed by GatheredPreKBlock. Fires after the helper's in0/in1 pop_front and
// after the L1_ACC drain, so the rd_ptr advance observes a fully-consumed K-block.
template <bool EnableGlobalCb, uint32_t In1CbId>
struct GatheredPostKBlock {
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

// Per-K-block KBlockInnerDimFn. Returns the unpadded inner-dim step count for
// this K-block (read from the runtime-arg table indexed by the active ring
// position). The LLK call's kt_dim arg stays In0BlockW — only the loop bound
// shrinks; in1 is still strided as if the K-tile span were full-width.
template <uint32_t RingSize>
struct GatheredInnerDimFn {
    const uint32_t* unpadded_widths;
    uint32_t ring_idx;

    ALWI uint32_t operator()(uint32_t block, uint32_t /*block_k*/) const {
        const uint32_t curr_ring_idx = (ring_idx + block) % RingSize;
        return unpadded_widths[curr_ring_idx];
    }
};

// In0SourceFn hook — the helper calls this once per K-block (before the matmul) to
// pick which in0 CB to unpack. Block 0 reads the self-primed local in0 CB; blocks
// 1..N read the remote/mcast in2 CB, as the gather streams the remaining K-slabs in
// over the ring. The two CBs MUST share a dataformat — the unpacker is configured
// once for in0_cb_id (see the helper's In0SourceFn contract).
template <uint32_t In2CbId>
struct GatheredIn0Source {
    ALWI uint32_t operator()(uint32_t block, uint32_t in0_cb_id) const { return block == 0 ? in0_cb_id : In2CbId; }
};

// In1BaseOffsetFn hook — the helper calls this once per K-block to shift the in1
// read base within the fronted region. Zero when in1 is a single fronted block
// (ENABLE_GLOBAL_CB receiver or DRAM in1); otherwise steps one in1 block per ring
// position — In1BlockNumTiles * curr_ring_idx, with curr_ring_idx = (ring_idx +
// block) % RingSize — to index the K-slab for the active ring position.
template <bool NeedsOffset, uint32_t In1BlockNumTiles, uint32_t RingSize>
struct GatheredIn1BaseOffsetFn {
    uint32_t ring_idx;

    ALWI uint32_t operator()(uint32_t block) const {
        if constexpr (!NeedsOffset) {
            return 0;
        } else {
            const uint32_t curr_ring_idx = (ring_idx + block) % RingSize;
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
    [[maybe_unused]] constexpr uint32_t in0_subblock_num_tiles =
        get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
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
    [[maybe_unused]] constexpr uint32_t out_block_num_tiles =
        get_compile_time_arg_val(14);                                          // number of tiles in out_block
    constexpr bool untilize_out = get_compile_time_arg_val(15);                // untilize output
    constexpr bool in1_is_dram_interleaved = get_compile_time_arg_val(16);     // in1 is in dram
    constexpr bool in1_is_dram_sharded = get_compile_time_arg_val(17);
    constexpr uint32_t in0_cb_id = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t in1_cb_id = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t in2_cb_id = get_named_compile_time_arg_val("cb_in2");
    constexpr uint32_t sync_cb = get_named_compile_time_arg_val("cb_sync");
    constexpr uint32_t sync_cb2 = get_named_compile_time_arg_val("cb_sync2");

    constexpr std::array<uint32_t, batch> mm_out_cb_ids = fill_named_cb_array<batch>(mm_out_cb_names);
    constexpr std::array<uint32_t, batch> mm_partials_cb_ids = fill_named_cb_array<batch>(mm_partials_cb_names);

    // ── SFPU activation params (compile-time, named CT args) ──
    // Always declared; default to NONE/0 so the helper template arguments resolve
    // either way — when SFPU_ACTIVATION is undefined the helpers statically discard
    // their packer-side activation paths.
#ifdef SFPU_ACTIVATION
    constexpr KernelActivation activation_type =
        static_cast<KernelActivation>(get_named_compile_time_arg_val("activation_type"));
    constexpr uint32_t activation_param0 = get_named_compile_time_arg_val("activation_param0");
    constexpr uint32_t activation_param1 = get_named_compile_time_arg_val("activation_param1");
    constexpr uint32_t activation_param2 = get_named_compile_time_arg_val("activation_param2");
#else
    constexpr KernelActivation activation_type = KernelActivation::NONE;
    constexpr uint32_t activation_param0 = 0;
    constexpr uint32_t activation_param1 = 0;
    constexpr uint32_t activation_param2 = 0;
#endif

    constexpr uint32_t ring_size = num_blocks;
    constexpr bool in1_is_dram = in1_is_dram_interleaved || in1_is_dram_sharded;

    // Compile-time helper switches.
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

    // LastBlockTarget selection (FUSE_BIAS is host-side dead code in this
    // kernel — same as the llama variant — so the FUSE_BIAS arm is moot).
    // untilize_out uses Interm (not the old fused OutWithUntilize): the matmul accumulates
    // the fully-tiled block into interm, then a downstream reblock_and_untilize untilizes the
    // FULLY-accumulated block into out. The fused pack_untilize untilized a partial before the
    // block finished accumulating, corrupting under packer_l1_acc.
    constexpr LastBlockTarget last_block_target =
        untilize_out ? LastBlockTarget::Interm
                     : (pack_relu_defined ? LastBlockTarget::OutWithRelu : LastBlockTarget::Out);

    // On the Interm-target untilize path, LastBlockTarget::Interm does NOT auto-enable relu and the
    // downstream reblock does not apply activation, so the PreKBlockFn enables relu-on-last for the
    // untilize path — relu still runs (at the matmul's pack stage) before untilize.
    constexpr bool enable_relu_on_last_via_pre_k = pack_relu_defined && untilize_out;

    // in1_policy: DRAM in1 is produced per-K-block (wait/pop each block); a fronted / global-CB in1
    // is caller-managed, so the helper neither waits nor pops it.
    constexpr InputPolicy in1_policy_const = in1_is_dram ? InputPolicy::WaitAndPopPerKBlock : InputPolicy::NoWaitNoPop;

    // The per-K-block in1 base offset is non-zero only on the
    // (`!ENABLE_GLOBAL_CB && !in1_is_dram`) path.
    constexpr bool needs_in1_base_offset = !enable_global_cb && !in1_is_dram;

    // Buf wrappers used by the kernel's outer fabric-sync primitives.
    CircularBuffer in1_cb(in1_cb_id);
    CircularBuffer sync_buf(sync_cb);
    CircularBuffer sync2_buf(sync_cb2);

    // Runtime args
    uint32_t rt_args_idx = 0;
    uint32_t core_type = get_arg_val<uint32_t>(rt_args_idx++);
    if (core_type == (uint32_t)CORE_TYPE::IDLE_CORE || core_type == (uint32_t)CORE_TYPE::HOP_CORE) {
        return;
    }
    uint32_t ring_idx = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t* unpadded_in0_shard_widths_in_tiles = (uint32_t*)get_arg_addr(rt_args_idx);
    rt_args_idx += ring_size;

    constexpr uint32_t out_block_w = out_subblock_w * in1_num_subblocks;

    // Buf wrappers for the helper's wait_front / pop_front / LLK call hygiene.
    // The In0SourceFn alternates between in0_cb_id (block 0) and in2_cb_id
    // (blocks 1..N); the helper uses these via active_in0_buf inside the K-loop.
    CircularBuffer in0_buf(in0_cb_id);
    CircularBuffer in1_buf(in1_cb_id);

    // Boot-time matmul + activation init. The helper invocation below uses
    // InitMode::None so it doesn't re-init each call; the per-batch
    // pack_reconfig_data_format re-binds the packer to the current batch's
    // partials CB without disturbing the unpacker matmul state. mm_block_init is
    // deprecated: boot with compute_kernel_hw_startup (hw_configure) then
    // matmul_block_init (unpack/math init). ActivationInitHelper::init() is a
    // compile-time no-op when activation_type == KernelActivation::NONE.
    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb_id, in1_cb_id, mm_partials_cb_ids[0]);
    matmul_block_init(in0_cb_id, in1_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
    if constexpr (activation_type != KernelActivation::NONE) {
        ActivationInitHelper<activation_type, activation_param0, activation_param1>::init();
    }

    // Per-batch ring CB state (used by ENABLE_GLOBAL_CB; declared at outer scope
    // so the functors can hold pointers across helper invocations).
    [[maybe_unused]] uint32_t in1_cb_start_addr = 0;
    [[maybe_unused]] uint32_t in1_rd_ptr_start_addr = 0;
    [[maybe_unused]] uint32_t curr_in1_block_index = 0;
    [[maybe_unused]] bool in1_tensor_split = false;
    [[maybe_unused]] uint32_t next_in1_block_index = 0;
    [[maybe_unused]] uint32_t next_in1_rd_ptr_addr = 0;

    using PreFn = GatheredPreKBlock<
        enable_global_cb,
        enable_relu_on_last_via_pre_k,
        in0_cb_id,
        in0_block_num_tiles,
        in1_cb_id,
        num_blocks,
        in1_block_size_bytes>;
    using PostFnRing = GatheredPostKBlock<enable_global_cb, in1_cb_id>;
    using InnerDimFn = GatheredInnerDimFn<ring_size>;
    using In0SrcFn = GatheredIn0Source<in2_cb_id>;
    using In1OffsetFn = GatheredIn1BaseOffsetFn<needs_in1_base_offset, in1_block_num_tiles, ring_size>;

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
        sync2_buf.wait_front(1);
        sync2_buf.pop_front(1);

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
        InnerDimFn inner_dim_fn{/*unpadded_widths=*/unpadded_in0_shard_widths_in_tiles, /*ring_idx=*/ring_idx};
        In0SrcFn in0_source_fn{};
        In1OffsetFn in1_offset_fn{/*ring_idx=*/ring_idx};

        matmul_block_gathered<
            in1_transpose_tile,                                 // transpose
            l1_acc,                                             // packer_l1_acc
            last_block_target,                                  // last_block_target
            OutputCBLayout::SubblockMajor,                      // layout
            matmul_config::InitMode::None,                      // init_mode
            InputPolicy::WaitAndPopPerKBlock,                   // in0_policy
            in1_policy_const,                                   // in1_policy
            matmul_config::DataFormatReconfig::InputAndOutput,  // reconfig (was defaulted)
            ActivationOp<activation_type, activation_param0, activation_param1, activation_param2>,  // Activation
            NoPostCompute,  // PostComputeFn (math-thread; unused)
            PreFn,          // PreKBlockFn
            PostFnRing,     // PostKBlockFn
            InnerDimFn,     // KBlockInnerDimFn
            In0SrcFn,       // In0SourceFn
            In1OffsetFn>(   // In1BaseOffsetFn
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
            NoPostCompute{},
            pre_fn,
            post_fn,
            inner_dim_fn,
            in0_source_fn,
            in1_offset_fn);

        // ── Downstream untilize (Interm target) ─────────────────────────────
        // matmul_block accumulated the fully-tiled block into mm_partials_buf (SubblockMajor).
        // reblock_and_untilize gathers it into row-major and untilizes into mm_out_buf. This
        // untilizes the FULLY-accumulated block (fixes the packer_l1_acc corruption the old fused
        // OutWithUntilize had, where a partial got untilized before accumulation finished).
        if constexpr (untilize_out) {
#ifdef PACK_RELU
            // The PreKBlockFn (enable_relu_on_last_via_pre_k) enabled relu for the last K-block's
            // interm pack; restore a clean packer state before the untilize reblock.
            PACK((llk_pack_relu_config(ReluConfig::none())));
#endif
            // Reconfigure srcA / pack DF / l1_acc for the reblock read, then invoke with
            // NoReconfigure so the helper adds no reconfig of its own — mirrors the production
            // (non-gather) kernel's !FUSE_BIAS untilize phase.
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
            sync_buf.reserve_back(1);
            sync_buf.push_back(1);
            UNPACK(
                (update_local_cb_rd_ptr(in1_cb_id, in1_rd_ptr_start_addr)));  // reset rd_ptr back to the initial addr
            UNPACK((update_rd_ptr_to_ring_index(
                in1_cb_id, in1_block_size_bytes, ring_size, in1_tensor_split)));  // update to next tensor addr
        }
    }
}
