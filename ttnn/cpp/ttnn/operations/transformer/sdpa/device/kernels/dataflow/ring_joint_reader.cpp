// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#if defined(SOL_PACK_DEBUG) || defined(SOL_DEBUG)
#include "api/debug/dprint.h"
#endif
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "dataflow_common.hpp"
#include "chunked_prefill_utils.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_mla_packing_plan.hpp"
#include "ring_joint_kv_pad_derivation.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_derived_slots.hpp"
#include "metadata_scalar_read.hpp"
#include "chain_link.hpp"
#include "sol_bias_gen.hpp"
#include "fused_op_receiver.hpp"
#include "ring_utils.hpp"

// Device-profiler zones around the reader's waits (TT_SOL_STEP_PROFILE=1 under TT_METAL_DEVICE_PROFILER=1), so a
// stall shows up as a long RD_* zone on the core that waits instead of as a gap in the compute zones.
#if defined(PROFILE_KERNEL) && defined(SOL_STEP_PROFILE)
#define SOL_ZONE(name) DeviceZoneScopedN(name)
#else
#define SOL_ZONE(name)
#endif
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_chain_layout.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/sliding_window_work_plan.hpp"
#include "../ring_joint_ksplit.hpp"  // relative, so its chunked_prefill_utils.hpp is the same file this kernel includes

namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;

template <bool has_joint_inputs, bool has_gathered_joint_k, uint32_t joint_tensor_args_offset>
constexpr uint32_t get_post_tensor_args_offset() {
    if constexpr (has_joint_inputs) {
        constexpr auto joint_q_args = TensorAccessorArgs<joint_tensor_args_offset>();
        constexpr auto joint_k_args = TensorAccessorArgs<joint_q_args.next_compile_time_args_offset()>();
        constexpr auto joint_v_args = TensorAccessorArgs<joint_k_args.next_compile_time_args_offset()>();
        constexpr uint32_t after_joint_v = joint_v_args.next_compile_time_args_offset();
        if constexpr (has_gathered_joint_k) {
            // Two additional slots for gathered joint K and gathered joint V accessor args.
            // IMPORTANT: these shift post_tensor_args_offset; all downstream index constants must account for this.
            constexpr auto gathered_joint_k_args = TensorAccessorArgs<after_joint_v>();
            constexpr auto gathered_joint_v_args =
                TensorAccessorArgs<gathered_joint_k_args.next_compile_time_args_offset()>();
            return gathered_joint_v_args.next_compile_time_args_offset();
        } else {
            return after_joint_v;
        }
    } else {
        return joint_tensor_args_offset;
    }
}

template <
    bool has_joint_k,
    bool has_gathered_joint_k,
    uint32_t joint_tensor_args_offset,
    typename LocalKGenerator,
    typename GatheredKGenerator,
    typename LocalJointTileLogical,
    typename JointInputTileLogical,
    typename FetchK>
inline void fetch_k_from_source(
    bool kv_chunk_is_joint,
    bool joint_chunk_is_local,
    uint32_t ring_iter,
    uint32_t joint_k_addr,
    uint32_t gathered_joint_k_addr,
    const LocalKGenerator& local_k_generator,
    const GatheredKGenerator& gathered_k_generator,
    const LocalJointTileLogical& local_joint_tile_logical,
    const JointInputTileLogical& joint_input_tile_logical,
    const FetchK& fetch_k) {
    if constexpr (has_joint_k) {
        if (kv_chunk_is_joint) {
            constexpr auto joint_q_args = TensorAccessorArgs<joint_tensor_args_offset>();
            constexpr auto joint_k_args = TensorAccessorArgs<joint_q_args.next_compile_time_args_offset()>();
            if constexpr (has_gathered_joint_k) {
                // Sharded-joint: the fused AG does NOT write the local device's own slice into
                // the gathered buffer (each device already holds its slice locally). When this
                // chunk belongs to the local device (ring_id == ring_index), read from the local
                // joint K tensor; otherwise read from the gathered buffer.
                if (joint_chunk_is_local) {
                    const auto joint_k_reader = TensorAccessor(joint_k_args, joint_k_addr);
                    const auto joint_k_generator = PaddedAddrGenerator(joint_k_reader, local_joint_tile_logical);
                    fetch_k(joint_k_generator);
                } else {
                    constexpr auto joint_v_args = TensorAccessorArgs<joint_k_args.next_compile_time_args_offset()>();
                    constexpr auto gathered_joint_k_args =
                        TensorAccessorArgs<joint_v_args.next_compile_time_args_offset()>();
                    const auto gathered_joint_k_reader = TensorAccessor(gathered_joint_k_args, gathered_joint_k_addr);
                    const auto gathered_joint_k_generator =
                        PaddedAddrGenerator(gathered_joint_k_reader, joint_input_tile_logical);
                    fetch_k(gathered_joint_k_generator);
                }
            } else {
                // Replicated-joint path: read joint K from the local (full L) joint tensor.
                const auto joint_k_reader = TensorAccessor(joint_k_args, joint_k_addr);
                const auto joint_k_generator = PaddedAddrGenerator(joint_k_reader, joint_input_tile_logical);
                fetch_k(joint_k_generator);
            }
        } else if (ring_iter == 0) {
            fetch_k(local_k_generator);
        } else {
            fetch_k(gathered_k_generator);
        }
    } else {
        if (ring_iter == 0) {
            fetch_k(local_k_generator);
        } else {
            fetch_k(gathered_k_generator);
        }
    }
}

template <
    bool has_joint_q,
    uint32_t joint_tensor_args_offset,
    typename QGenerator,
    typename JointInputTileLogical,
    typename ReadQ>
inline void read_q_from_source(
    bool is_joint_q,
    uint32_t joint_q_addr,
    const QGenerator& q_generator,
    const JointInputTileLogical& joint_input_tile_logical,
    const ReadQ& read_q) {
    if constexpr (has_joint_q) {
        if (is_joint_q) {
            constexpr auto joint_q_args = TensorAccessorArgs<joint_tensor_args_offset>();
            const auto joint_q_reader = TensorAccessor(joint_q_args, joint_q_addr);
            const auto joint_q_generator = PaddedAddrGenerator(joint_q_reader, joint_input_tile_logical);
            read_q(joint_q_generator);
        } else {
            read_q(q_generator);
        }
    } else {
        read_q(q_generator);
    }
}

template <
    bool has_joint_k,
    bool has_gathered_joint_k,
    uint32_t joint_tensor_args_offset,
    typename LocalVGenerator,
    typename GatheredVGenerator,
    typename LocalJointTileLogical,
    typename JointInputTileLogical,
    typename FetchV>
inline void fetch_v_from_source(
    bool kv_chunk_is_joint,
    bool joint_chunk_is_local,
    uint32_t ring_iter,
    uint32_t joint_v_addr,
    uint32_t gathered_joint_v_addr,
    const LocalVGenerator& local_v_generator,
    const GatheredVGenerator& gathered_v_generator,
    const LocalJointTileLogical& local_joint_tile_logical,
    const JointInputTileLogical& joint_input_tile_logical,
    const FetchV& fetch_v) {
    if constexpr (has_joint_k) {
        if (kv_chunk_is_joint) {
            constexpr auto joint_q_args = TensorAccessorArgs<joint_tensor_args_offset>();
            constexpr auto joint_k_args = TensorAccessorArgs<joint_q_args.next_compile_time_args_offset()>();
            constexpr auto joint_v_args = TensorAccessorArgs<joint_k_args.next_compile_time_args_offset()>();
            if constexpr (has_gathered_joint_k) {
                // Sharded-joint: read local device's slice from the local joint V tensor; all other
                // slices come from the gathered buffer (which doesn't hold the local device's data).
                if (joint_chunk_is_local) {
                    const auto joint_v_reader = TensorAccessor(joint_v_args, joint_v_addr);
                    const auto joint_v_generator = PaddedAddrGenerator(joint_v_reader, local_joint_tile_logical);
                    fetch_v(joint_v_generator);
                } else {
                    constexpr auto gathered_joint_k_args =
                        TensorAccessorArgs<joint_v_args.next_compile_time_args_offset()>();
                    constexpr auto gathered_joint_v_args =
                        TensorAccessorArgs<gathered_joint_k_args.next_compile_time_args_offset()>();
                    const auto gathered_joint_v_reader = TensorAccessor(gathered_joint_v_args, gathered_joint_v_addr);
                    const auto gathered_joint_v_generator =
                        PaddedAddrGenerator(gathered_joint_v_reader, joint_input_tile_logical);
                    fetch_v(gathered_joint_v_generator);
                }
            } else {
                // Replicated-joint path: read joint V from the local (full L) joint tensor.
                const auto joint_v_reader = TensorAccessor(joint_v_args, joint_v_addr);
                const auto joint_v_generator = PaddedAddrGenerator(joint_v_reader, joint_input_tile_logical);
                fetch_v(joint_v_generator);
            }
        } else if (ring_iter == 0) {
            fetch_v(local_v_generator);
        } else {
            fetch_v(gathered_v_generator);
        }
    } else {
        if (ring_iter == 0) {
            fetch_v(local_v_generator);
        } else {
            fetch_v(gathered_v_generator);
        }
    }
}

template <typename LocalVGenerator, typename GatheredVGenerator>
struct VSourceGenerators {
    LocalVGenerator local;
    GatheredVGenerator gathered;
};

template <uint32_t cb_v_in, uint32_t v_cb_entry_tiles, uint32_t Sk_chunk_t, uint32_t vDHt, uint32_t k_tile_bytes>
inline void materialize_v_prefix_from_k(Noc noc, uint32_t kt_base_addr, uint32_t rows_to_materialize) {
    CircularBuffer cb_v(cb_v_in);
    cb_v.reserve_back(v_cb_entry_tiles);
    uint32_t v_write_ptr = cb_v.get_write_ptr();
    const uint8_t noc_id = noc.get_noc_id();
    const uint32_t my_noc_x = my_x[noc_id];
    const uint32_t my_noc_y = my_y[noc_id];
    UnicastEndpoint kt_src;
    noc.set_async_read_state<NocOptions::DEFAULT, k_tile_bytes>(
        kt_src, k_tile_bytes, {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = kt_base_addr});
    for (uint32_t sk = 0; sk < rows_to_materialize; ++sk) {
        uint32_t kt_read_ptr = kt_base_addr + sk * k_tile_bytes;
        for (uint32_t vd = 0; vd < vDHt; ++vd) {
            noc.async_read_with_state<NocOptions::DEFAULT, k_tile_bytes>(
                kt_src,
                CoreLocalMem<uint32_t>(v_write_ptr),
                k_tile_bytes,
                {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = kt_read_ptr},
                {});
            v_write_ptr += k_tile_bytes;
            kt_read_ptr += Sk_chunk_t * k_tile_bytes;
        }
    }
    noc.async_read_barrier();
    cb_v.push_back(v_cb_entry_tiles);
}

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t NHK = get_compile_time_arg_val(2);
    constexpr uint32_t DHt = get_compile_time_arg_val(3);
    constexpr uint32_t vDHt = get_compile_time_arg_val(4);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(6);
    constexpr uint32_t q_local_padded_Nt = get_compile_time_arg_val(7);
    constexpr uint32_t kv_local_padded_Nt = get_compile_time_arg_val(8);
    constexpr uint32_t Lt = get_compile_time_arg_val(9);
    constexpr uint32_t L = get_compile_time_arg_val(10);
    constexpr uint32_t num_local_q_chunks = get_compile_time_arg_val(11);
    constexpr uint32_t num_joint_q_chunks = get_compile_time_arg_val(12);
    constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(13);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(14);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(15);
    constexpr uint32_t ring_size = get_compile_time_arg_val(16);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(17);
    constexpr uint32_t is_causal = get_compile_time_arg_val(18);
    constexpr uint32_t is_balanced = get_compile_time_arg_val(19);
    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(20) == 1;
    // Reader's slot-21 carries chunked_enabled.
    constexpr bool chunked_enabled = get_compile_time_arg_val(21) == 1;
    constexpr uint32_t num_q_readers = get_compile_time_arg_val(22);
    constexpr uint32_t chunk_size_t = get_compile_time_arg_val(23);
    constexpr uint32_t kv_stripe_split = get_named_compile_time_arg_val("kv_stripe_split");
    static_assert(
        kv_stripe_split > 0 && q_local_padded_Nt % kv_stripe_split == 0, "Q slab must split into whole KV regions");
    constexpr uint32_t kv_region_Nt = q_local_padded_Nt / kv_stripe_split;
    static_assert(kv_region_Nt * ring_size == chunk_size_t, "chunk_size_t must span one KV region per ring rank");
    constexpr uint32_t q_ring_size = ring_size / kv_stripe_split;
    constexpr bool indexed_kv_cache = get_compile_time_arg_val(24) == 1;
    constexpr bool kv_pad_rotation_enabled = get_compile_time_arg_val(25) == 1;
    constexpr uint32_t NHV = get_compile_time_arg_val(26);
    // Latent-V mode: absent V is materialized from the prefix of K tiles already in L1.
    constexpr bool v_shares_k_buffer = get_compile_time_arg_val(27) == 1;
    constexpr bool use_attention_sink = get_compile_time_arg_val(28) == 1;
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(29);
    constexpr bool has_sliding_window = sliding_window_size > 0;
    // Sol-Attn block gathering: per-Q-chunk K/V content, so no K/V chains.
    constexpr bool sol_enabled = get_named_compile_time_arg_val("sol_enabled") == 1;
    // Chunk mode streams every chunk through the chains; only compute skips.
    constexpr bool sol_chunk_mode = get_named_compile_time_arg_val("sol_chunk_mode") == 1;
    constexpr uint32_t sol_nkc = get_named_compile_time_arg_val("sol_nkc");
    // No sol_bias tensor: the pooled-chunk column-bias tiles are generated here (sol_bias_gen.hpp).
    constexpr bool sol_gen_bias = get_named_compile_time_arg_val("sol_gen_bias") == 1;
    constexpr uint32_t sol_logical_n_ct = get_named_compile_time_arg_val("sol_logical_n");
    // Compact mode: block mask, streamed like chunk mode, own kept blocks left-justified in L1.
    constexpr bool sol_compact = get_named_compile_time_arg_val("sol_compact") == 1;
    constexpr bool sol_even_up = get_named_compile_time_arg_val("sol_even_up") == 1;  // compact even-up rule
    constexpr bool sol_even_up_all = get_named_compile_time_arg_val("sol_even_up_all") == 1;  // debug knob
    constexpr bool enable_kv_chains = !has_sliding_window && !(sol_enabled && !sol_chunk_mode);
    constexpr uint32_t gathered_padded_Nt = get_compile_time_arg_val(30);
    // Slots 31/32 are trace-safe metadata controls. Attention-sink and sliding-window
    // fields occupy 28..30 in this branch.
    // Slot 31: trace-safe slot select. When set, kv_cache_batch_idx is read from the slot_id tensor[0]
    // on-device (common runtime arg 0 = slot_id DRAM addr) instead of the per-core runtime arg, so a
    // captured trace replays across cache slots.
    constexpr bool slot_from_metadata = get_compile_time_arg_val(31) == 1;
    // Slot 32: trace-safe KV-pad derivation. When set, the reader reads kv_actual_isl from the
    // kv_actual_isl tensor[0] (common runtime arg 4 = its DRAM addr), derives logical_nt / q-mapping /
    // ring masks on-device, and hands the compute-needed values to the compute kernel via
    // cb_kv_pad_derived (compute cannot NoC-read the DRAM tensor).
    constexpr bool kv_pad_from_metadata = get_compile_time_arg_val(32) == 1;
    constexpr bool gqa_grouped_kv = ring_joint::is_gqa_grouped_kv_head_mode(v_shares_k_buffer, NH, NHK, NHV);
    constexpr bool k_uses_batch_chain = ring_joint::uses_shared_k_batch_chain(gqa_grouped_kv, NHK);
    constexpr bool use_head_chain = ring_joint::uses_v_head_chain(enable_kv_chains, gqa_grouped_kv, v_shares_k_buffer);
    // In-place latent-V (single-tile Q): the compute kernel reads V straight from K^T, so the
    // reader never materializes V. Shared with the program factory and compute kernel.
    constexpr bool kt_inplace_v = kt_inplace_v_enabled(v_shares_k_buffer, Sq_chunk_t);
    constexpr uint32_t q_heads_per_v = NH / NHV;
    // Slots 33-35: sharded-joint scalars appended by the factory after upstream's attention-sink /
    // sliding-window / metadata fields (slots 28-32 above).
    // Lt_local: per-device joint-Q tile count (Lt/ring_size on sharded path, Lt on replicated).
    constexpr uint32_t Lt_local = get_compile_time_arg_val(33);
    constexpr bool joint_is_sharded = get_compile_time_arg_val(34) == 1;
    // Slot 35: true (unpadded) joint length in tiles (twins spatial logical_nt). Joint K chunks whose
    // global start tile is at/after logical_lt are pure padding and are skipped.
    constexpr uint32_t logical_lt_compile = get_compile_time_arg_val(35);
    // Slots 36-39: transport-to-tensor rank mapping.
    constexpr bool full_mesh_rank_mapping = get_compile_time_arg_val(36) == 1;
    constexpr auto snake_orientation = static_cast<ttnn::ccl::snake_ring::Orientation>(get_compile_time_arg_val(37));
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(38);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(39);
    // Slot 40: circular sliding KV slab count (0 = unbounded). Wraps the sliding work plan's
    // local slab addressing (sliding_window_work_plan.hpp) — must match host halo layout and compute.
    constexpr uint32_t circular_kv_slab_count = get_compile_time_arg_val(40);
    // Slots 41-42: read live logical_n / logical_l from device tensors (common runtime args below), then
    // re-derive logical_nt / logical_lt / the ring masks from them.
    constexpr bool has_logical_n_tensor = get_compile_time_arg_val(41) == 1;
    constexpr bool has_logical_l_tensor = get_compile_time_arg_val(42) == 1;
    constexpr bool has_logical_length_tensor = has_logical_n_tensor || has_logical_l_tensor;
    // Live joint tail in tiles; equals the compile-time value unless logical_l arrives as a tensor.
    uint32_t logical_lt = logical_lt_compile;

    // Joint-path compile-time gating. When zero, joint Q/K branches are statically dead
    // and dropped by the compiler, eliminating runtime ternaries and joint generator uses.
    constexpr bool has_joint_q = num_joint_q_chunks > 0;
    constexpr bool has_joint_k = num_joint_k_chunks > 0;
    constexpr bool has_joint_inputs = has_joint_q || has_joint_k;
    // Sharded joint requires the gathered joint K/V buffers (only meaningful when joint K is present).
    constexpr bool has_gathered_joint_k = joint_is_sharded && has_joint_k;

    // Slots 33-35 are the sharded-joint scalars (Lt_local, joint_is_sharded, logical_lt), 36-39 the rank
    // mapping, slot 40 the bounded sliding KV slab count, and 41-42 the logical-length transport flags, so
    // the tensor accessors start at slot 43.
    constexpr uint32_t kFirstAccessorArgOffset = 43;
    constexpr auto q_args = TensorAccessorArgs<kFirstAccessorArgOffset>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto gathered_k_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto gathered_v_args = TensorAccessorArgs<gathered_k_args.next_compile_time_args_offset()>();
    constexpr uint32_t joint_tensor_args_offset = gathered_v_args.next_compile_time_args_offset();
    constexpr uint32_t post_joint_tensor_args_offset =
        get_post_tensor_args_offset<has_joint_inputs, has_gathered_joint_k, joint_tensor_args_offset>();
    constexpr auto attention_sink_args = TensorAccessorArgs<post_joint_tensor_args_offset>();
    constexpr uint32_t post_tensor_args_offset = attention_sink_args.next_compile_time_args_offset();
    // The metadata accessor (metadata path only) follows the tensor accessors and precedes the chain
    // semaphore compile args. Gate its offset on slot_from_metadata: when absent, fall back to a VALID
    // (unused) accessor offset (q_args' slot) so TensorAccessorArgs<> -- instantiated unconditionally
    // here -- never names a non-accessor compile arg (which would fail its internal static_assert).
    // The chain/CB compile args then start after the metadata accessor when present.
    constexpr uint32_t meta_args_offset = slot_from_metadata ? post_tensor_args_offset : kFirstAccessorArgOffset;
    constexpr auto meta_args = TensorAccessorArgs<meta_args_offset>();  // slot_id accessor
    // kv_actual_isl gets its OWN accessor (a separately-allocated single-page DRAM tensor can land in a
    // different DRAM bank than slot_id, so the slot accessor's dspec reads the wrong bank for it -- the kv
    // read silently returned 0). Appended right after slot's when kv_pad_from_metadata; otherwise fall back
    // to a VALID accessor offset so the unconditional TensorAccessorArgs<> never names a non-accessor arg.
    constexpr uint32_t kv_meta_args_offset =
        kv_pad_from_metadata ? meta_args.next_compile_time_args_offset() : meta_args_offset;
    constexpr auto kv_meta_args = TensorAccessorArgs<kv_meta_args_offset>();
#ifdef KV_VALID_END
    constexpr bool has_kv_valid_end = kv_pad_from_metadata;
#else
    constexpr bool has_kv_valid_end = false;
#endif
    constexpr uint32_t kv_valid_end_args_offset =
        has_kv_valid_end ? kv_meta_args.next_compile_time_args_offset() : kv_meta_args_offset;
    constexpr auto kv_valid_end_args = TensorAccessorArgs<kv_valid_end_args_offset>();
    constexpr uint32_t chains_base_no_kv_pad =
        slot_from_metadata ? meta_args.next_compile_time_args_offset() : post_tensor_args_offset;
    constexpr uint32_t post_meta_args_offset =
        kv_pad_from_metadata ? kv_valid_end_args.next_compile_time_args_offset() : chains_base_no_kv_pad;
    // logical_n / logical_l accessors follow the metadata accessors, appended as a pair so these offsets do
    // not depend on which one was supplied. Same VALID-fallback trick as meta_args above.
    constexpr uint32_t logical_n_args_offset =
        has_logical_length_tensor ? post_meta_args_offset : kFirstAccessorArgOffset;
    constexpr auto logical_n_args = TensorAccessorArgs<logical_n_args_offset>();
    constexpr uint32_t logical_l_args_offset =
        has_logical_length_tensor ? logical_n_args.next_compile_time_args_offset() : logical_n_args_offset;
    constexpr auto logical_l_args = TensorAccessorArgs<logical_l_args_offset>();
    constexpr uint32_t pre_sol_args_offset =
        has_logical_length_tensor ? logical_l_args.next_compile_time_args_offset() : post_meta_args_offset;
    // Sol-Attn accessors (mask, pooled K, pooled V, bias) follow the logical-length pair. Same VALID-fallback
    // trick as meta_args when disabled.
    constexpr uint32_t sol_mask_args_offset = sol_enabled ? pre_sol_args_offset : kFirstAccessorArgOffset;
    constexpr auto sol_mask_args = TensorAccessorArgs<sol_mask_args_offset>();
    constexpr uint32_t sol_pk_args_offset =
        sol_enabled ? sol_mask_args.next_compile_time_args_offset() : kFirstAccessorArgOffset;
    constexpr auto sol_pk_args = TensorAccessorArgs<sol_pk_args_offset>();
    constexpr uint32_t sol_pv_args_offset =
        sol_enabled ? sol_pk_args.next_compile_time_args_offset() : kFirstAccessorArgOffset;
    constexpr auto sol_pv_args = TensorAccessorArgs<sol_pv_args_offset>();
    // Resident static max: the per-row max bound tensor's accessor follows pooled V's iff sol_static_max; the bias
    // accessor and the chain args continue after it (see the factory's Sol-Attn accessor layout).
    constexpr bool sol_static_max = get_named_compile_time_arg_val("sol_static_max") == 1;
    constexpr uint32_t sol_rm_args_offset =
        (sol_enabled && sol_static_max) ? sol_pv_args.next_compile_time_args_offset() : kFirstAccessorArgOffset;
    constexpr auto sol_rm_args = TensorAccessorArgs<sol_rm_args_offset>();
    constexpr uint32_t sol_after_pv_offset = (sol_enabled && sol_static_max)
                                                 ? sol_rm_args.next_compile_time_args_offset()
                                                 : sol_pv_args.next_compile_time_args_offset();
    constexpr uint32_t sol_bias_args_offset =
        (sol_enabled && !sol_gen_bias) ? sol_after_pv_offset : kFirstAccessorArgOffset;
    constexpr auto sol_bias_args = TensorAccessorArgs<sol_bias_args_offset>();
    constexpr uint32_t chains_base_offset =
        sol_enabled ? (sol_gen_bias ? sol_after_pv_offset : sol_bias_args.next_compile_time_args_offset())
                    : pre_sol_args_offset;

    uint32_t argidx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t gathered_k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t gathered_v_addr = get_arg_val<uint32_t>(argidx++);
    uint32_t joint_q_addr = 0;
    uint32_t joint_k_addr = 0;
    uint32_t joint_v_addr = 0;
    if constexpr (has_joint_inputs) {
        joint_q_addr = get_arg_val<uint32_t>(argidx++);
        joint_k_addr = get_arg_val<uint32_t>(argidx++);
        joint_v_addr = get_arg_val<uint32_t>(argidx++);
    }
    const uint32_t attention_sink_addr = get_arg_val<uint32_t>(argidx++);
    // Gathered joint K/V buffer addresses (sharded path only) — pushed by the factory right after the
    // attention-sink address, so they must be read here before global_q_start to keep argidx aligned.
    uint32_t gathered_joint_k_addr = 0;
    uint32_t gathered_joint_v_addr = 0;
    if constexpr (has_gathered_joint_k) {
        gathered_joint_k_addr = get_arg_val<uint32_t>(argidx++);
        gathered_joint_v_addr = get_arg_val<uint32_t>(argidx++);
    }
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);
    uint32_t kv_cache_batch_idx = get_arg_val<uint32_t>(argidx++);
    const uint32_t q_per_core = global_q_end - global_q_start;

    ChainConfig head_cfg;
    if constexpr (use_head_chain) {
        head_cfg = ChainConfig::read_from_args(argidx);
    }

    // Batch chain runtime args (only present for non-GQA shared-K modes).
    ChainConfig batch_cfg;  // default zero-initialized
    uint32_t max_q_per_core = 0;
    if constexpr (enable_kv_chains && k_uses_batch_chain) {
        batch_cfg = ChainConfig::read_from_args(argidx);
        max_q_per_core = get_arg_val<uint32_t>(argidx++);
    }
    ChainConfig gqa_cfg;  // default zero-initialized
    uint32_t gqa_max_q_per_core = 0;
    if constexpr (enable_kv_chains && gqa_grouped_kv) {
        gqa_cfg = ChainConfig::read_from_args(argidx);
        gqa_max_q_per_core = get_arg_val<uint32_t>(argidx++);
    }

    uint32_t logical_nt = get_arg_val<uint32_t>(argidx++);
    // Token-exact logical length for the Sol bias generator (block lengths at the ragged tail); the live
    // logical_n tensor overrides the compile-time placeholder below, like logical_nt.
    [[maybe_unused]] uint32_t sol_logical_n_tokens = sol_logical_n_ct;
    uint32_t active_ring_iter_mask = get_arg_val<uint32_t>(argidx++);
    ring_joint::ChunkedQMapping qmap{
        get_arg_val<uint32_t>(argidx++),
        get_arg_val<uint32_t>(argidx++),
        get_arg_val<uint32_t>(argidx++),
        get_arg_val<uint32_t>(argidx++)};
    RingSDPAOpReceiver fused_op_receiver = RingSDPAOpReceiver(
        true, /* wait_for_op_signal */
        argidx);

    // Rotated Q split: first runtime arg past the fused-op signaler block. Captured here because
    // argidx is final at this point; the compile-time sizes are derived below with the other CT args.
    const uint32_t rotated_args_base = argidx;

    // Compile-time semaphore ids and chain flags are appended after all TensorAccessorArgs().
    // ChainLink takes semaphore IDs directly (the new Semaphore<> wrapper resolves them to L1 addrs).
    constexpr uint32_t chain_sender_semaphore_arg_offset = ring_joint::kChainSenderSemaphoreCompileArgOffset;
    constexpr uint32_t chain_receiver_semaphore_arg_offset = ring_joint::kChainReceiverSemaphoreCompileArgOffset;
    constexpr uint32_t chain_valid_semaphore_arg_offset = ring_joint::kChainValidSemaphoreCompileArgOffset;
    constexpr uint32_t chain_mcast_enabled_arg_offset = ring_joint::kChainMcastEnabledCompileArgOffset;
    constexpr uint32_t chain_compile_arg_count = ring_joint::kChainCompileArgCount;

    uint32_t head_sender_semaphore_id = 0;
    uint32_t head_receiver_semaphore_id = 0;
    uint32_t head_valid_semaphore_id = 0;
    constexpr bool head_mcast_enabled = []() {
        if constexpr (use_head_chain) {
            return get_compile_time_arg_val(chains_base_offset + chain_mcast_enabled_arg_offset) == 1;
        }
        return false;
    }();
    if constexpr (use_head_chain) {
        head_sender_semaphore_id = get_compile_time_arg_val(chains_base_offset + chain_sender_semaphore_arg_offset);
        head_receiver_semaphore_id = get_compile_time_arg_val(chains_base_offset + chain_receiver_semaphore_arg_offset);
        head_valid_semaphore_id = get_compile_time_arg_val(chains_base_offset + chain_valid_semaphore_arg_offset);
    }
    constexpr uint32_t head_chain_arg_count = use_head_chain ? chain_compile_arg_count : 0;
    constexpr uint32_t batch_chain_arg_count = enable_kv_chains && k_uses_batch_chain ? chain_compile_arg_count : 0;
    constexpr uint32_t gqa_chain_arg_count = enable_kv_chains && gqa_grouped_kv ? chain_compile_arg_count : 0;
    constexpr uint32_t batch_chain_ct_offset = chains_base_offset + head_chain_arg_count;
    constexpr uint32_t gqa_chain_ct_offset = batch_chain_ct_offset + batch_chain_arg_count;

    // Batch chain semaphores (only present for non-GQA shared-K modes).
    // Initialize to 0; will be overwritten if k_uses_batch_chain. Non-participants never use them.
    uint32_t batch_sender_semaphore_id = 0;
    uint32_t batch_receiver_semaphore_id = 0;
    uint32_t batch_valid_semaphore_id = 0;

    // batch_mcast_enabled: read from compile-time args if present, else false (for template instantiation)
    constexpr bool batch_mcast_enabled = []() {
        if constexpr (enable_kv_chains && k_uses_batch_chain) {
            return get_compile_time_arg_val(batch_chain_ct_offset + chain_mcast_enabled_arg_offset) == 1;
        }
        return false;
    }();

    if constexpr (enable_kv_chains && k_uses_batch_chain) {
        batch_sender_semaphore_id = get_compile_time_arg_val(batch_chain_ct_offset + chain_sender_semaphore_arg_offset);
        batch_receiver_semaphore_id =
            get_compile_time_arg_val(batch_chain_ct_offset + chain_receiver_semaphore_arg_offset);
        batch_valid_semaphore_id = get_compile_time_arg_val(batch_chain_ct_offset + chain_valid_semaphore_arg_offset);
    }

    uint32_t gqa_sender_semaphore_id = 0;
    uint32_t gqa_receiver_semaphore_id = 0;
    uint32_t gqa_valid_semaphore_id = 0;
    constexpr bool gqa_mcast_enabled = []() {
        if constexpr (enable_kv_chains && gqa_grouped_kv) {
            return get_compile_time_arg_val(gqa_chain_ct_offset + chain_mcast_enabled_arg_offset) == 1;
        }
        return false;
    }();
    if constexpr (enable_kv_chains && gqa_grouped_kv) {
        gqa_sender_semaphore_id = get_compile_time_arg_val(gqa_chain_ct_offset + chain_sender_semaphore_arg_offset);
        gqa_receiver_semaphore_id = get_compile_time_arg_val(gqa_chain_ct_offset + chain_receiver_semaphore_arg_offset);
        gqa_valid_semaphore_id = get_compile_time_arg_val(gqa_chain_ct_offset + chain_valid_semaphore_arg_offset);
    }

    constexpr uint32_t cb_arg_offset =
        chains_base_offset + head_chain_arg_count + batch_chain_arg_count + gqa_chain_arg_count;
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(cb_arg_offset + 0);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(cb_arg_offset + 1);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(cb_arg_offset + 2);
    constexpr uint32_t cb_attention_sink = get_compile_time_arg_val(cb_arg_offset + 3);
    constexpr uint32_t cb_kv_pad_derived = get_compile_time_arg_val(cb_arg_offset + 4);

    // Maximum owned chunk count (base plus one remainder unit), or zero for static scheduling.
    // Per active ordinal the factory sends [remainder_start, group_has_remainder].
    // global_q_start/end describe the fixed base range when rotation is enabled.
    // The factory pushes rotated_max_slots as the final compile-time arg of every kernel, so
    // read it from there rather than tracking a per-kernel index into the block above.
    constexpr uint32_t rotated_max_slots = get_ct_arg<kernel_compile_time_args.size() - 1>();
    constexpr bool rotated_q_split_enabled = rotated_max_slots > 0;
    constexpr uint32_t ksplit_count = get_named_compile_time_arg_val("ksplit_count");
    constexpr bool dense_causal_skip = get_named_compile_time_arg_val("dense_causal_skip") == 1;
    constexpr bool ksplit_enabled = ksplit_count > 1;
    static_assert(!ksplit_enabled || !rotated_q_split_enabled);
    [[maybe_unused]] const uint32_t ksplit_idx = ksplit_enabled ? get_arg_val<uint32_t>(rotated_args_base) : 0;
    constexpr uint32_t rotated_iter_stride = kRotatedReaderIterWords;
    // Sol-Attn buffer addresses: the factory appends them last; sol excludes the rotated and K splits.
    static_assert(!sol_enabled || (!rotated_q_split_enabled && !ksplit_enabled));
    [[maybe_unused]] const uint32_t sol_mask_addr = sol_enabled ? get_arg_val<uint32_t>(rotated_args_base + 0) : 0;
    [[maybe_unused]] const uint32_t sol_pk_addr = sol_enabled ? get_arg_val<uint32_t>(rotated_args_base + 1) : 0;
    [[maybe_unused]] const uint32_t sol_pv_addr = sol_enabled ? get_arg_val<uint32_t>(rotated_args_base + 2) : 0;
    // The row-max bound's address sits at +3 iff sol_static_max; everything after it shifts by one word.
    constexpr uint32_t kSolRM = sol_static_max ? 1u : 0u;
    [[maybe_unused]] const uint32_t sol_rm_addr =
        (sol_enabled && sol_static_max) ? get_arg_val<uint32_t>(rotated_args_base + 3) : 0;
    [[maybe_unused]] const uint32_t sol_bias_addr =
        sol_enabled ? get_arg_val<uint32_t>(rotated_args_base + 3 + kSolRM) : 0;
    // Chunk mode: head-chain members (q_chunk start within head, count) for the chain-wide kept union.
    [[maybe_unused]] const uint32_t sol_n_members =
        sol_enabled ? get_arg_val<uint32_t>(rotated_args_base + 4 + kSolRM) : 0;
    [[maybe_unused]] const uint32_t sol_my_start =
        sol_enabled ? get_arg_val<uint32_t>(rotated_args_base + 5 + kSolRM) : 0;
    // Generated bias only: packed bf16 bias-per-token-count table (sol_bias_gen::kTableWords words) precedes the
    // chain members; with a sol_bias tensor the members follow my_start directly.
    [[maybe_unused]] const uint32_t sol_len_bias_table_base = rotated_args_base + 6 + kSolRM;
    [[maybe_unused]] const uint32_t sol_members_base =
        rotated_args_base + 6 + kSolRM + (sol_gen_bias ? sol_bias_gen::kTableWords : 0u);
    constexpr uint32_t sol_nb = get_named_compile_time_arg_val("sol_nb");
    constexpr uint32_t sol_nb_pad = get_named_compile_time_arg_val("sol_nb_pad");
    constexpr uint32_t sol_n_blocks_pad = get_named_compile_time_arg_val("sol_n_blocks_pad");
    constexpr uint32_t sol_nqc_pad = get_named_compile_time_arg_val("sol_nqc_pad");
    constexpr uint32_t sol_pooled_chunks = get_named_compile_time_arg_val("sol_pooled_chunks");
    constexpr uint32_t sol_pooled_last_tiles = get_named_compile_time_arg_val("sol_pooled_last_tiles");
    constexpr uint32_t sol_pooled_off = get_named_compile_time_arg_val("sol_pooled_off");
    constexpr uint32_t sol_mask_row_bytes = get_named_compile_time_arg_val("sol_mask_row_bytes");
    constexpr uint32_t cb_sol_plan = get_named_compile_time_arg_val("cb_sol_plan");
    constexpr uint32_t cb_sol_bias = get_named_compile_time_arg_val("cb_sol_bias");
    // Cross-chunk packing: kept blocks are appended into full pages instead of compacted inside their own
    // chunk, so compute pays one visit per PAGE rather than one per kept chunk.
    //
    // Roles invert under sol_pack. Raw chunks are downloaded into cb_k_stage / cb_v_stage and forwarded to the
    // chain from there; those CBs are never pushed, so compute never sees a raw chunk and never drains one. The
    // packed pages go out on cb_k_in / cb_v_in in place of chunks, so compute's inner-step call site is the same
    // as in the dense path.
    constexpr bool sol_pack = get_named_compile_time_arg_val("sol_pack") == 1;
    // Resident mode: chunk-mode semantics, one pass of each K/V piece per ring iteration shared by every Q chunk
    // the core owns. See the branch in the ring loop below.
    constexpr bool sol_resident = get_named_compile_time_arg_val("sol_resident") == 1;
    // Words of per-shard kept-block bits in every resident plan page (block-form mask, as in compact mode).
    constexpr uint32_t sol_res_block_words = get_named_compile_time_arg_val("sol_res_block_words");
    static_assert(!sol_resident || (sol_chunk_mode && !sol_compact && !sol_pack), "resident mode is a chunk mode");
    static_assert(!sol_resident || sol_res_block_words * 32 >= sol_nb, "resident: block words cover the shard");
    // D1 (sparse_attn_design_2026-10-23.md 4.2): the resident piece stream replaces the chain with one producer per
    // head group that reads each piece from DRAM and multicasts it to every other member (no forwarding). The head
    // chain's membership and semaphores are reused: sender = the producer's credit count (+1 per consumer per
    // piece), receiver = each consumer's delivered count (set to the running piece count by multicast), valid = the
    // producer's local source word for that set. The group arguments follow the Sol chain members.
    constexpr bool sol_res_mcast = get_named_compile_time_arg_val("sol_res_mcast") == 1;
    static_assert(!sol_res_mcast || sol_resident, "group delivery is the resident stream");
    // Resident iteration boundary (U3). rec_base != 0: one plan page per ring iteration holding every chunk's record
    // from that word on (0 = one page per chunk). no_dummies: no phase-alignment pushes at the end of an iteration;
    // every member of a chain or group pushes the same sequence, so their CB pointers stay in step without them, and
    // the dummies' reserves held the reader back until compute had nearly drained the iteration.
    constexpr uint32_t sol_res_plan_rec_base = get_named_compile_time_arg_val("sol_res_plan_rec_base");
    constexpr bool sol_res_no_dummies = get_named_compile_time_arg_val("sol_res_no_dummies") == 1;
    static_assert(sol_res_plan_rec_base == 0 || sol_resident, "the one-page plan is the resident plan");
    static_assert(!sol_res_no_dummies || sol_resident, "dummies are dropped in resident mode only");
    constexpr uint32_t cb_k_stage = get_named_compile_time_arg_val("cb_k_stage");
    constexpr uint32_t cb_v_stage = get_named_compile_time_arg_val("cb_v_stage");
    constexpr uint32_t sol_blocks_per_page = get_named_compile_time_arg_val("sol_blocks_per_page");
    // K/V CB depth in chunks (factory kv_cb_pages, default 2); the end-of-iteration phase alignment and the group
    // delivery's credit lookahead use it.
    constexpr uint32_t kv_cb_pages = get_named_compile_time_arg_val("kv_cb_pages");
    constexpr uint32_t kSolBlockTiles = 2;  // 64-token blocks
    constexpr uint32_t sol_bias_tile_bytes = get_tile_size(cb_sol_bias);

    // Common runtime args: metadata block first when present, then the logical-length pair.
    constexpr uint32_t logical_length_common_arg_base =
        slot_from_metadata ? ring_joint::kReaderMetadataCommonArgCount : 0;
    // kv_valid_end is validated incompatible with the logical_n / logical_l tensors, so its arg directly
    // follows the metadata block.
    static_assert(!has_kv_valid_end || !has_logical_length_tensor);
    [[maybe_unused]] constexpr uint32_t kv_valid_end_common_arg = logical_length_common_arg_base;

    [[maybe_unused]] uint32_t global_n_partial_col_live = 0;
    [[maybe_unused]] uint32_t joint_l_partial_col_live = 0;

    if constexpr (slot_from_metadata || kv_pad_from_metadata || has_logical_length_tensor) {
        Noc meta_noc;
        CircularBuffer cb_q_scratch(cb_q_in);
        const uint32_t meta_l1 = cb_q_scratch.get_write_ptr();
        if constexpr (slot_from_metadata) {
            const uint32_t slot_id =
                trace_metadata::read_metadata_scalar_u32(meta_noc, meta_args, get_common_arg_val<uint32_t>(0), meta_l1);
            kv_cache_batch_idx = trace_metadata::bounded_cache_batch_idx(
                slot_id,
                get_common_arg_val<uint32_t>(1),
                get_common_arg_val<uint32_t>(2),
                get_common_arg_val<uint32_t>(3));
        }
        if constexpr (has_logical_n_tensor) {
            const uint32_t logical_n_live = trace_metadata::read_metadata_scalar_u32(
                meta_noc, logical_n_args, get_common_arg_val<uint32_t>(logical_length_common_arg_base), meta_l1);
            ASSERT(logical_n_live >= 1);
            logical_nt = ring_joint::tiles_for(logical_n_live);
            global_n_partial_col_live = ring_joint::tile_partial_col(logical_n_live);
            sol_logical_n_tokens = logical_n_live;
        }
        if constexpr (has_logical_l_tensor) {
            const uint32_t logical_l_live = trace_metadata::read_metadata_scalar_u32(
                meta_noc, logical_l_args, get_common_arg_val<uint32_t>(logical_length_common_arg_base + 1), meta_l1);
            logical_lt = ring_joint::tiles_for(logical_l_live);
            joint_l_partial_col_live = ring_joint::tile_partial_col(logical_l_live);
        }
        if constexpr (kv_pad_from_metadata) {
            uint32_t kv_actual_isl = trace_metadata::read_metadata_scalar_u32(
                meta_noc, kv_meta_args, get_common_arg_val<uint32_t>(4), meta_l1);
            if constexpr (has_sliding_window) {
                kv_actual_isl = trace_metadata::bounded_sliding_kv_actual_isl(
                    kv_actual_isl, q_local_padded_Nt, ring_size, kv_local_padded_Nt, SLIDING_HALO_SLOT_COUNT);
            } else {
                kv_actual_isl =
                    trace_metadata::bounded_kv_actual_isl(kv_actual_isl, chunk_size_t, kv_local_padded_Nt * ring_size);
            }
            const uint32_t kv_actual_tile_count = kv_actual_isl / 32;
            logical_nt = trace_metadata::logical_tile_rows_clamped_to_cache(
                kv_actual_isl, chunk_size_t, kv_local_padded_Nt * ring_size);
            if constexpr (has_kv_valid_end) {
                const uint32_t kv_valid_end = trace_metadata::read_metadata_scalar_u32(
                    meta_noc, kv_valid_end_args, get_common_arg_val<uint32_t>(kv_valid_end_common_arg), meta_l1);
                logical_nt =
                    trace_metadata::logical_tile_rows_clamped_to_valid_end(logical_nt, kv_actual_isl, kv_valid_end);
            }
            const uint32_t tensor_rank =
                ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
                    fused_op_receiver.seq.ring_index, mesh_rows, mesh_cols, snake_orientation);
            qmap = ring_joint::build_chunked_q_mapping(
                kv_actual_tile_count, logical_nt, q_local_padded_Nt, q_ring_size, tensor_rank / kv_stripe_split);
            // Joint trio stays defaulted: KV-pad rotation is validated incompatible with a sharded joint.
            const auto masks = ring_joint::build_ring_work_masks_device<full_mesh_rank_mapping>({
                .transport_rank = fused_op_receiver.seq.ring_index,
                .ring_size = ring_size,
                .mesh_rows = mesh_rows,
                .mesh_cols = mesh_cols,
                .snake_orientation = snake_orientation,
                .backward_writes_expected = fused_op_receiver.seq.expected[0],
                .forward_writes_expected = fused_op_receiver.seq.expected[1],
                .num_local_k_chunks = num_local_k_chunks,
                .k_chunk_tile_count = Sk_chunk_t,
                .kv_local_padded_Nt = kv_local_padded_Nt,
                .kernel_chunked = chunked_enabled,
                .q_chunk_group_tile_count = chunk_size_t,
                .kv_region_Nt = kv_region_Nt,
                .logical_nt = logical_nt,
                .num_joint_k_chunks = num_joint_k_chunks,
                .joint_seq_len = L,
                .kv_pad_rotation_enabled = kv_pad_rotation_enabled,
                .kernel_is_causal = is_causal != 0,
                .is_balanced = is_balanced != 0,
            });
            active_ring_iter_mask = masks.active_ring_iter_mask;
        } else if constexpr (has_logical_length_tensor) {
            // Host masks assumed the placeholder (every iteration active). A smaller live length can empty
            // an iteration, and an active bit there leaves compute waiting on work that never arrives.
            const auto masks = ring_joint::build_ring_work_masks_device<full_mesh_rank_mapping>({
                .transport_rank = fused_op_receiver.seq.ring_index,
                .ring_size = ring_size,
                .mesh_rows = mesh_rows,
                .mesh_cols = mesh_cols,
                .snake_orientation = snake_orientation,
                .backward_writes_expected = fused_op_receiver.seq.expected[0],
                .forward_writes_expected = fused_op_receiver.seq.expected[1],
                .num_local_k_chunks = num_local_k_chunks,
                .k_chunk_tile_count = Sk_chunk_t,
                .kv_local_padded_Nt = kv_local_padded_Nt,
                .kernel_chunked = chunked_enabled,
                .q_chunk_group_tile_count = chunk_size_t,
                .kv_region_Nt = kv_region_Nt,
                .logical_nt = logical_nt,
                .num_joint_k_chunks = num_joint_k_chunks,
                .joint_seq_len = L,
                .kv_pad_rotation_enabled = kv_pad_rotation_enabled,
                .kernel_is_causal = is_causal != 0,
                .is_balanced = is_balanced != 0,
                .joint_is_sharded = joint_is_sharded,
                .joint_local_padded_Nt = Lt_local,
                .logical_lt = logical_lt,
            });
            active_ring_iter_mask = masks.active_ring_iter_mask;
        }

        if constexpr (kv_pad_from_metadata || has_logical_length_tensor) {
            // Hand compute the values it cannot read itself (compute RISCs cannot NoC-read DRAM).
            CircularBuffer cb_derived(cb_kv_pad_derived);
            cb_derived.reserve_back(1);
            CoreLocalMem<volatile uint32_t> d(cb_derived.get_write_ptr());
            d[ring_joint::kDerivedLogicalNt] = logical_nt;
            if constexpr (kv_pad_from_metadata) {
                d[ring_joint::kDerivedQPreWrapStartTile] = qmap.q_pre_wrap_start_tile;
                d[ring_joint::kDerivedQPreWrapTileCount] = qmap.q_pre_wrap_tile_count;
                d[ring_joint::kDerivedQPostWrapStartTile] = qmap.q_post_wrap_start_tile;
                d[ring_joint::kDerivedQValidTileCount] = qmap.q_valid_tile_count;
            }
            d[ring_joint::kDerivedActiveRingIterMask] = active_ring_iter_mask;
            if constexpr (has_logical_n_tensor) {
                d[ring_joint::kDerivedGlobalNPartialCol] = global_n_partial_col_live;
            }
            if constexpr (has_logical_l_tensor) {
                d[ring_joint::kDerivedLogicalLt] = logical_lt;
                d[ring_joint::kDerivedJointLPartialCol] = joint_l_partial_col_live;
            }
            cb_derived.push_back(1);
        }
    }

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);
    constexpr uint32_t attention_sink_tile_bytes = get_tile_size(cb_attention_sink);

    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;
    constexpr uint32_t v_cb_entry_tiles = v_shares_k_buffer ? k_chunk_tiles : v_chunk_tiles;

    Noc noc;

    // Head chain (query-head level): MHA uses it for K/V; separate-V shared-K uses it for V only.
    ChainLink<head_mcast_enabled, true> head_chain(
        head_cfg.participates,
        head_cfg.is_injector,
        head_cfg.is_sink,
        head_sender_semaphore_id,
        head_receiver_semaphore_id,
        head_valid_semaphore_id,
        head_cfg.signal_target_x<head_mcast_enabled>(),
        head_cfg.signal_target_y<head_mcast_enabled>(),
        head_cfg.next_physical_x,
        head_cfg.next_physical_y,
        head_cfg.mcast_start_x,
        head_cfg.mcast_start_y,
        head_cfg.mcast_end_x,
        head_cfg.mcast_end_y,
        head_cfg.mcast_num_dests,
        v_chunk_tiles,
        v_tile_bytes,
        head_cfg.head,
        head_cfg.next_core_q_chunks);

    // Shared-K chain used by K in non-GQA shared-K modes.
    ChainLink<batch_mcast_enabled, false> batch_chain(
        batch_cfg.participates,
        batch_cfg.is_injector,
        batch_cfg.is_sink,
        batch_sender_semaphore_id,
        batch_receiver_semaphore_id,
        batch_valid_semaphore_id,
        batch_cfg.signal_target_x<batch_mcast_enabled>(),
        batch_cfg.signal_target_y<batch_mcast_enabled>(),
        batch_cfg.next_physical_x,
        batch_cfg.next_physical_y,
        batch_cfg.mcast_start_x,
        batch_cfg.mcast_start_y,
        batch_cfg.mcast_end_x,
        batch_cfg.mcast_end_y,
        batch_cfg.mcast_num_dests,
        k_chunk_tiles,
        k_tile_bytes,
        0,  // chain_head unused for the shared-K chain
        batch_cfg.next_core_q_chunks);

    // GQA grouped chain (head-level where head means KV head): used by both K and V in GQA_GROUPED_KV.
    ChainLink<gqa_mcast_enabled, true> gqa_chain(
        gqa_cfg.participates,
        gqa_cfg.is_injector,
        gqa_cfg.is_sink,
        gqa_sender_semaphore_id,
        gqa_receiver_semaphore_id,
        gqa_valid_semaphore_id,
        gqa_cfg.signal_target_x<gqa_mcast_enabled>(),
        gqa_cfg.signal_target_y<gqa_mcast_enabled>(),
        gqa_cfg.next_physical_x,
        gqa_cfg.next_physical_y,
        gqa_cfg.mcast_start_x,
        gqa_cfg.mcast_start_y,
        gqa_cfg.mcast_end_x,
        gqa_cfg.mcast_end_y,
        gqa_cfg.mcast_num_dests,
        v_chunk_tiles,
        v_tile_bytes,
        gqa_cfg.head,
        gqa_cfg.next_core_q_chunks);

    // Non-shared V uses the head chain, except GQA where K and V share the grouped KV-head chain.
    auto& v_chain = [&]() -> auto& {
        if constexpr (gqa_grouped_kv) {
            return gqa_chain;
        } else {
            return head_chain;
        }
    }();

    // K uses the grouped GQA chain, the shared-K chain, or the query-head chain.
    auto& k_chain = [&]() -> auto& {
        if constexpr (gqa_grouped_kv) {
            return gqa_chain;
        } else if constexpr (k_uses_batch_chain) {
            return batch_chain;
        } else {
            return head_chain;
        }
    }();

    constexpr uint32_t q_num_subblocks = Sq_chunk_t / qk_subblock_h;
    constexpr bool use_q_subblock_push = (q_num_subblocks > 1);
    constexpr uint32_t q_heads_per_k = NH / NHK;
    static_assert(!has_sliding_window || !gqa_mcast_enabled, "Sliding windows use direct per-core K/V reads");
    static_assert(!has_sliding_window || chunked_enabled, "Sliding windows require chunked prefill");

    // Throttle Q DRAM reads so many readers don't saturate the NoC outstanding-read budget.
    constexpr uint32_t q_barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_q_readers>();

    const auto q_reader = TensorAccessor(q_args, q_addr);
    const auto local_k_reader = TensorAccessor(k_args, k_addr);
    const auto gathered_k_reader = TensorAccessor(gathered_k_args, gathered_k_addr);

    const uint32_t kv_batch_dim = indexed_kv_cache ? kv_cache_batch_idx + 1 : B;
    // The fused all-gather wrote the active slot to gathered slot 0, so address it as batch-1.
    const uint32_t gathered_kv_batch_dim = indexed_kv_cache ? 1 : B;
    const auto input_q_tile_logical = TensorTileShape(B, NH, q_local_padded_Nt, DHt);
    // Packed KV: K/V rows can be wider than the head dims; V is the last vDHt columns of its row.
    constexpr uint32_t k_row_Wt = get_named_compile_time_arg_val("k_row_Wt");
    constexpr uint32_t v_row_Wt = get_named_compile_time_arg_val("v_row_Wt");
    constexpr uint32_t v_col_offset_t = v_row_Wt - vDHt;
    const auto input_k_tile_logical = TensorTileShape(kv_batch_dim, NHK, kv_local_padded_Nt, k_row_Wt);
    const auto gathered_k_input_tile_logical =
        TensorTileShape(gathered_kv_batch_dim, NHK, gathered_padded_Nt, k_row_Wt);
    // Joint K/V addressing: full gathered length (Lt rows). Used for the gathered joint buffer on the
    // sharded path and for the local full-L joint tensor on the replicated path.
    const auto joint_input_tile_logical = TensorTileShape(B, NH, Lt, DHt);
    // Joint Q addressing: per-device shard (Lt_local rows). Equals Lt on the replicated path
    // (Lt_local == Lt when joint is not sharded), so this is bit-identical there.
    const auto joint_q_input_tile_logical = TensorTileShape(B, NH, Lt_local, DHt);

    const auto q_generator = PaddedAddrGenerator(q_reader, input_q_tile_logical);
    const auto local_k_generator = PaddedAddrGenerator(local_k_reader, input_k_tile_logical);
    const auto gathered_k_generator = PaddedAddrGenerator(gathered_k_reader, gathered_k_input_tile_logical);
    const auto local_v_reader = TensorAccessor(v_args, v_addr);
    const auto input_v_tile_logical = TensorTileShape(kv_batch_dim, NHV, kv_local_padded_Nt, v_row_Wt);
    const auto gathered_v_reader = TensorAccessor(gathered_v_args, gathered_v_addr);
    const auto gathered_v_input_tile_logical =
        TensorTileShape(gathered_kv_batch_dim, NHV, gathered_padded_Nt, v_row_Wt);
    const auto local_v_generator = PaddedAddrGenerator(local_v_reader, input_v_tile_logical);
    const auto gathered_v_generator = PaddedAddrGenerator(gathered_v_reader, gathered_v_input_tile_logical);
    [[maybe_unused]] const auto v_generators =
        VSourceGenerators<decltype(local_v_generator), decltype(gathered_v_generator)>{
            local_v_generator, gathered_v_generator};
    const auto attention_sink_reader = TensorAccessor(attention_sink_args, attention_sink_addr);
    const auto attention_sink_tile_shape = TensorTileShape(1, NH, 1, 1);
    CircularBuffer cb_attn_sink(cb_attention_sink);
    // Sol-Attn sources: pooled K/V [1, NH, n_blocks_pad, DH] tiles, bias [1, NH, nqc_pad*32, n_blocks_pad] tiles,
    // mask [1, NH, nqc_pad, n_blocks_pad] uint32 row-major (one page per row).
    [[maybe_unused]] const auto sol_mask_reader = TensorAccessor(sol_mask_args, sol_mask_addr);
    [[maybe_unused]] const auto sol_pk_generator = PaddedAddrGenerator(
        TensorAccessor(sol_pk_args, sol_pk_addr), TensorTileShape(1, NH, sol_n_blocks_pad / 32, DHt));
    [[maybe_unused]] const auto sol_pv_generator = PaddedAddrGenerator(
        TensorAccessor(sol_pv_args, sol_pv_addr), TensorTileShape(1, NH, sol_n_blocks_pad / 32, vDHt));
    // The per-row max bound, bf16 tile [1, NH, Q padded rows, 32] (one tile column, value in column 0).
    [[maybe_unused]] const auto sol_rm_generator =
        PaddedAddrGenerator(TensorAccessor(sol_rm_args, sol_rm_addr), TensorTileShape(1, NH, q_local_padded_Nt, 1));
    [[maybe_unused]] const auto sol_bias_generator = PaddedAddrGenerator(
        TensorAccessor(sol_bias_args, sol_bias_addr), TensorTileShape(1, NH, sol_nqc_pad, sol_n_blocks_pad / 32));
    // Pooled chunks are consumed on the first active ring iteration of every Q chunk.
    [[maybe_unused]] bool sol_pooled_pending = true;
    // Compact mode helpers. Block b of shard s is exact only if it is a FULL 64-token block (partial shard / logical_n
    // tails are always pooled, so a set mask bit on them is ignored here and in the bias generator alike).
    [[maybe_unused]] const auto sol_block_is_full = [&](uint32_t s, uint32_t b) -> bool {
        return sol_bias_gen::block_len(
                   sol_bias_gen::Geometry{
                       kv_local_padded_Nt * 32, sol_logical_n_tokens, sol_nb, sol_nb_pad, sol_nkc, Sk_chunk_t},
                   s,
                   b) == sol_bias_gen::kBlockTiles * 32;
    };
    // Compact-mode exactness of block b of shard s, read from a shard slice at `row` (slice index = b): own bit on a
    // full block, or the even-up rule (sol_bias_gen::block_exact; the bias generator applies the same predicate).
    [[maybe_unused]] const auto sol_block_exact =
        [&](CoreLocalMem<volatile uint32_t> row, uint32_t s, uint32_t b) -> bool {
        sol_bias_gen::Geometry g{
            kv_local_padded_Nt * 32, sol_logical_n_tokens, sol_nb, sol_nb_pad, sol_nkc, Sk_chunk_t};
        g.even_up = sol_even_up;
        g.even_up_all = sol_even_up_all;
        return sol_bias_gen::block_exact(g, row, s * sol_nb_pad, s, b);
    };
    // This (ring iter, Q chunk)'s own block-row slice (nb_pad words of shard ring_id) in the plan page scratch.
    [[maybe_unused]] uint32_t sol_row_addr = 0;
    // Left-justify the kept blocks of local chunk k_chunk inside a chunk page in place. The page is `rows` rows of
    // `row_stride` bytes (K^T: DHt rows of Sk_chunk_t tiles; V: one row), block j at row offset j * block_bytes.
    // Move-order rule: moves are local NoC reads and NoC ordering across different (src, dst) pairs is not
    // guaranteed, so a move whose destination slot is <= an earlier move's source block waits for the earlier reads
    // first. Called after the chain forward (its writes are flushed), before the push.
    [[maybe_unused]] const auto sol_compact_blocks =
        [&](uint32_t s, uint32_t k_chunk, uint32_t base, uint32_t rows, uint32_t row_stride, uint32_t block_bytes) {
            if constexpr (sol_compact) {
                constexpr uint32_t bpc = Sk_chunk_t / kSolBlockTiles;
                CoreLocalMem<volatile uint32_t> row(sol_row_addr);
                const uint8_t noc_id = noc.get_noc_id();
                UnicastEndpoint src;
                uint32_t slot = 0;
                uint32_t last_src = 0;
                bool moved = false;
                for (uint32_t j = 0; j < bpc; ++j) {
                    const uint32_t b = k_chunk * bpc + j;
                    if (!sol_block_exact(row, s, b)) {
                        continue;
                    }
                    if (slot != j) {
                        if (!moved) {
                            // The chain forward above waited for its writes to be flushed (source page fully read
                            // by the NoC); a full write barrier before the first in-place move removes any
                            // dependence on that semantic.
                            noc.async_write_barrier();
                        }
                        if (moved && slot <= last_src) {
                            noc.async_read_barrier();
                        }
                        for (uint32_t rr = 0; rr < rows; ++rr) {
                            noc.async_read(
                                src,
                                CoreLocalMem<uint32_t>(base + rr * row_stride + slot * block_bytes),
                                block_bytes,
                                {.noc_x = my_x[noc_id],
                                 .noc_y = my_y[noc_id],
                                 .addr = base + rr * row_stride + j * block_bytes},
                                {});
                        }
                        moved = true;
                        last_src = j;
                    }
                    ++slot;
                }
                if (moved) {
                    noc.async_read_barrier();
                }
            }
        };
    // Pack one chunk's kept blocks into the packed pages, flushing whenever a page fills.
    //
    // K and V are packed together in one pass and the full-page test runs after every block, because a page
    // boundary can fall INSIDE a chunk: testing only after a whole chunk would let the fill overrun the page and
    // write past its end into neighbouring L1.
    //
    // Returns the new fill. Source and destination are disjoint (staging -> packed), so the in-place overlap
    // ordering rule that sol_compact_blocks needs does not apply here.
    [[maybe_unused]] const auto sol_pack_chunk = [&](uint32_t s,
                                                     uint32_t k_chunk,
                                                     uint32_t k_src,
                                                     uint32_t v_src,
                                                     uint32_t& k_dst,
                                                     uint32_t& v_dst,
                                                     uint32_t& pages_pushed,
                                                     uint32_t fill) -> uint32_t {
        if constexpr (sol_pack) {
            CoreLocalMem<volatile uint32_t> row(sol_row_addr);
            const uint8_t noc_id = noc.get_noc_id();
            UnicastEndpoint src;
            const uint32_t k_block_bytes = kSolBlockTiles * k_tile_bytes;
            const uint32_t v_block_bytes = kSolBlockTiles * vDHt * v_tile_bytes;
            const uint32_t k_row_stride = Sk_chunk_t * k_tile_bytes;
            bool moved = false;
            // Only LOCAL chunks are packed. plan word 5 sums row_chunk_kept over [0, num_local_k_chunks), so a
            // joint chunk packed here would create pages the plan never counted -- and its block index
            // k_chunk * blocks_per_page + j runs past the row's valid range, reading whatever follows.
            if (k_chunk >= num_local_k_chunks) {
                return fill;
            }
            for (uint32_t j = 0; j < sol_blocks_per_page; ++j) {
                const uint32_t b = k_chunk * sol_blocks_per_page + j;
                if (!sol_block_exact(row, s, b)) {
                    continue;
                }
                if (!moved) {
                    noc.async_write_barrier();  // the chain forward read these pages; flush before re-reading
                    moved = true;
                }
                // K^T page is [DHt][Sk_chunk_t] tiles: block j is tile columns [2j, 2j+2) of every row.
                for (uint32_t rr = 0; rr < DHt; ++rr) {
                    noc.async_read(
                        src,
                        CoreLocalMem<uint32_t>(k_dst + rr * k_row_stride + fill * k_block_bytes),
                        k_block_bytes,
                        {.noc_x = my_x[noc_id], .noc_y = my_y[noc_id], .addr = k_src + rr * k_row_stride + j * k_block_bytes},
                        {});
                }
                // V page is [Sk_chunk_t][vDHt] tiles: block j is tile rows [2j, 2j+2), one contiguous run.
                noc.async_read(
                    src,
                    CoreLocalMem<uint32_t>(v_dst + fill * v_block_bytes),
                    v_block_bytes,
                    {.noc_x = my_x[noc_id], .noc_y = my_y[noc_id], .addr = v_src + j * v_block_bytes},
                    {});
                ++fill;
                if (fill == sol_blocks_per_page) {
                    // Page full MID-CHUNK: hand it over and open the next one before packing another block.
                    noc.async_read_barrier();  // every block of this page must have landed before the push
                    CircularBuffer cb_kp(cb_k_in), cb_vp(cb_v_in);
                    cb_kp.push_back(k_chunk_tiles);
                    cb_vp.push_back(v_cb_entry_tiles);
#ifdef SOL_PACK_DEBUG
                    // Printed before the reserve: reserve_back is where the reader blocks, so the last PUSH
                    // line names the page it could not get space for.
                    DPRINT("SP {} {}\n", k_chunk, pages_pushed + 1);
#endif
                    cb_kp.reserve_back(k_chunk_tiles);
                    cb_vp.reserve_back(v_cb_entry_tiles);
                    k_dst = cb_kp.get_write_ptr();
                    v_dst = cb_vp.get_write_ptr();
                    ++pages_pushed;
                    fill = 0;
                    moved = false;  // the new page needs its own barrier before the next move
                }
            }
            if (moved) {
                noc.async_read_barrier();
            }
        }
        return fill;
    };
    // Pooled K/V chunks + this Q chunk's bias tile rows (both Sol modes; every core reads its own copy).
    [[maybe_unused]] const auto sol_push_pooled = [&](uint32_t nq, uint32_t q_chunk, uint32_t& kv_pushed) {
        if constexpr (sol_enabled) {
            constexpr uint32_t pooled_rows_t = sol_n_blocks_pad / 32;
            // Generated bias: the Q chunk's own mask row goes back into the plan page scratch (compute only
            // reads the plan words at offset 0; the scratch is the reader's) and the block geometry is fixed.
            [[maybe_unused]] uint32_t sol_mask_scratch = 0;
            [[maybe_unused]] sol_bias_gen::Geometry sol_bias_geo{};
            if constexpr (sol_gen_bias && !sol_compact) {
                CircularBuffer cb_plan_scratch(cb_sol_plan);
                sol_mask_scratch = cb_plan_scratch.get_write_ptr() + 64;
                noc.async_read(
                    sol_mask_reader,
                    CoreLocalMem<uint32_t>(sol_mask_scratch),
                    sol_mask_row_bytes,
                    {.page_id = nq * sol_nqc_pad + q_chunk},
                    {});
                noc.async_read_barrier();
                sol_bias_geo = sol_bias_gen::Geometry{
                    kv_local_padded_Nt * 32, sol_logical_n_tokens, sol_nb, sol_nb_pad, sol_nkc, Sk_chunk_t};
            } else if constexpr (sol_gen_bias) {
                // Compact: the block row is too long for the plan scratch; each pooled chunk reads its own slice.
                sol_bias_geo = sol_bias_gen::Geometry{
                    kv_local_padded_Nt * 32, sol_logical_n_tokens, sol_nb, sol_nb_pad, sol_nkc, Sk_chunk_t};
                sol_bias_geo.even_up = sol_even_up;
                sol_bias_geo.even_up_all = sol_even_up_all;
            }
            for (uint32_t pc = 0; pc < sol_pooled_chunks; ++pc) {
                const uint32_t tiles = pc + 1 == sol_pooled_chunks ? sol_pooled_last_tiles : Sk_chunk_t;
                const uint32_t row0 = pc * Sk_chunk_t;
                CircularBuffer cb_k(cb_k_in);
                cb_k.reserve_back(k_chunk_tiles);
                sol_pk_generator.issue_reads(
                    Slice(0, nq, row0, row0 + tiles, 0, DHt),
                    pooled_rows_t,
                    cb_k_in,
                    cb_k.get_write_ptr(),
                    k_tile_bytes,
                    k_tile_bytes * Sk_chunk_t,
                    0);
                noc.async_read_barrier();
                noc.write_zeros_l1_barrier();
                cb_k.push_back(k_chunk_tiles);
                kv_pushed++;
                CircularBuffer cb_bias(cb_sol_bias);
                cb_bias.reserve_back(Sk_chunk_t);
                if constexpr (sol_gen_bias && sol_compact) {
                    // Slice [row0*32, row0*32 + tiles*32) of the block row (<= 2 KB) into the last tile slot of the
                    // reserved bias region; generate_tiles computes that slot's values before overwriting it.
                    const uint32_t slice_addr = cb_bias.get_write_ptr() + (Sk_chunk_t - 1) * sol_bias_tile_bytes;
                    noc.async_read(
                        sol_mask_reader,
                        CoreLocalMem<uint32_t>(slice_addr),
                        tiles * 32 * sizeof(uint32_t),
                        {.page_id = nq * sol_nqc_pad + q_chunk, .offset_bytes = row0 * 32 * sizeof(uint32_t)},
                        {});
                    noc.async_read_barrier();
                    sol_bias_geo.mask_base_block = row0 * 32;
                    sol_bias_gen::even_up_row(
                        sol_bias_geo,
                        CoreLocalMem<volatile uint32_t>(slice_addr),
                        sol_bias_geo.mask_base_block,
                        sol_bias_geo.mask_base_block,
                        tiles * 32);
                    sol_bias_gen::generate_tiles<false, sol_bias_tile_bytes>(
                        noc,
                        cb_bias.get_write_ptr(),
                        row0,
                        tiles,
                        sol_bias_geo,
                        CoreLocalMem<volatile uint32_t>(slice_addr),
                        sol_len_bias_table_base);
                } else if constexpr (sol_gen_bias) {
                    sol_bias_gen::generate_tiles<sol_chunk_mode, sol_bias_tile_bytes>(
                        noc,
                        cb_bias.get_write_ptr(),
                        row0,
                        tiles,
                        sol_bias_geo,
                        CoreLocalMem<volatile uint32_t>(sol_mask_scratch),
                        sol_len_bias_table_base);
                } else {
                    sol_bias_generator.issue_reads(
                        Slice(0, nq, q_chunk, q_chunk + 1, row0, row0 + tiles),
                        sol_nqc_pad,
                        cb_sol_bias,
                        cb_bias.get_write_ptr(),
                        tiles * sol_bias_tile_bytes,
                        sol_bias_tile_bytes,
                        0);
                    noc.async_read_barrier();
                    noc.write_zeros_l1_barrier();
                }
                cb_bias.push_back(Sk_chunk_t);
                CircularBuffer cb_v(cb_v_in);
                cb_v.reserve_back(v_cb_entry_tiles);
                sol_pv_generator.issue_reads(
                    Slice(0, nq, row0, row0 + tiles, 0, vDHt),
                    pooled_rows_t,
                    cb_v_in,
                    cb_v.get_write_ptr(),
                    vDHt * v_tile_bytes,
                    v_tile_bytes,
                    0);
                noc.async_read_barrier();
                noc.write_zeros_l1_barrier();
                cb_v.push_back(v_cb_entry_tiles);
            }
        }
    };

    // Tracks whether Q has been pushed for q_per_core == 1 optimization.
    // When q_per_core == 1, Q is identical across ring iterations so we only push it once.
    bool q_pushed = false;
    // D1 (resident group delivery): pieces this core has pushed through the group stream since launch. The
    // producer's delivered count and the consumers' credits are running totals of it, never reset in a launch.
    [[maybe_unused]] uint32_t res_group_pieces = 0;

    /**
     * Iterate over ring indices.
     * On the first iteration, read from local K, V.
     * On subsequent iterations, read from gathered K, V. Sync with AllGather fused signaler.
     */
    const uint32_t ring_index =
        ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
            fused_op_receiver.seq.ring_index, mesh_rows, mesh_cols, snake_orientation);
    uint32_t half_sequence = num_q_chunks / 2;
    // Sliding consumes local and halo ranges in one logical Q pass. Without a K split the plan puts the local K chunks
    // first, so the halo is waited for only before the first halo chunk; a core that reads none waits at the end, so a
    // cached program cannot observe the previous invocation's token.
    bool halo_synchronized = !has_sliding_window;
    auto synchronize_halo = [&]() {
        if (halo_synchronized) {
            return;
        }
        halo_synchronized = true;
        // Sliding has a compact write plan (local slab + cyclic predecessor tails).
        const uint32_t synchronization_iters =
            1 + fused_op_receiver.seq.expected[0] + fused_op_receiver.seq.expected[1];
        // One signal per halo regardless of hop count: the collecting exchange waits for every hop's
        // arrival before signalling (ring_attention_neighbor_halo_reader.cpp).
        for (uint32_t ring_iter = 0; ring_iter < synchronization_iters; ++ring_iter) {
            fused_op_receiver.get_next_ring_id_and_consume_one_signal();
        }
    };
    // K chunks past this device's last Q row are fully masked; compute skips the same ones.
    const uint32_t causal_end_nt = dense_causal_skip
                                       ? chunked_q_global_end_tile<kv_pad_rotation_enabled, q_local_padded_Nt>(
                                             logical_nt,
                                             ring_index,
                                             ring_size,
                                             qmap.q_pre_wrap_start_tile,
                                             qmap.q_pre_wrap_tile_count,
                                             qmap.q_post_wrap_start_tile,
                                             qmap.q_valid_tile_count)
                                       : logical_nt;
    auto arrival_preview = fused_op_receiver.seq;
    // Chosen at compile time, and the guard below folds packed_sources to false in the classic ring's
    // binary so it carries none of the packed-source path.
    constexpr bool packed_kv_enabled = GROUPED_KV_SOURCE_COUNT > 1;
    const ring_joint::PackedKVSchedule packed_schedule = ring_joint::make_packed_kv_schedule<GROUPED_KV_SOURCE_COUNT>(
        ring_size,
        kv_local_padded_Nt,
        logical_nt,
        active_ring_iter_mask,
        kv_region_Nt,
        Sk_chunk_t,
        has_sliding_window,
        [&] {
            return ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
                arrival_preview.get_next_ring_id([](uint32_t, uint32_t) {}), mesh_rows, mesh_cols, snake_orientation);
        });
    const uint32_t source_group_size = packed_schedule.source_group_size;
    const bool packed_sources = packed_kv_enabled && packed_schedule.packed();
    const uint32_t sdpa_ring_iterations = packed_schedule.ring_iterations;
#ifdef SOL_DEBUG
    // TT_SOL_DEBUG=1: the arrival-signal semaphores as this core first sees them (program-local, initial value 0,
    // incremented by the all-gather workers once per slab), and the wait thresholds the sequencer will use.
    DPRINT(
        "SYNC start sem0={} sem1={} exp0={} exp1={} split={} split_shard={} split_wait={} ring_index={}\n",
        Semaphore<>(fused_op_receiver.signal_op_semaphore_ids[0]).value(),
        Semaphore<>(fused_op_receiver.signal_op_semaphore_ids[1]).value(),
        fused_op_receiver.seq.expected[0],
        fused_op_receiver.seq.expected[1],
        fused_op_receiver.split_forwarding_enabled ? 1u : 0u,
        fused_op_receiver.split_shard_id,
        fused_op_receiver.split_second_half_wait,
        ring_index);
#endif
    for (uint32_t ring_iter = 0; ring_iter < sdpa_ring_iterations; ++ring_iter) {
        const ring_joint::PackedKVGroupPlan packed_kv = packed_schedule.pass_plan(ring_iter);
        uint32_t packed_source_ids[GROUPED_KV_SOURCE_COUNT];
        ring_joint::PackedKVSourceReadiness packed_readiness;
        if (packed_sources) {
            // Planning identities must not wait for every member. Compute and
            // writer also derive this order without gather waits; this reader
            // alone gates access to the corresponding gathered source data.
            auto preview = fused_op_receiver.seq;
            for (uint32_t source = 0; source < source_group_size; ++source) {
                packed_source_ids[source] =
                    ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
                        preview.get_next_ring_id([](uint32_t, uint32_t) {}), mesh_rows, mesh_cols, snake_orientation);
            }
        }
        const auto wait_packed_source = [&](uint32_t source) {
            const uint32_t ready_id =
                ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
                    fused_op_receiver.get_next_ring_id_and_sync(), mesh_rows, mesh_cols, snake_orientation);
            ASSERT(ready_id == packed_source_ids[source]);
        };
        const bool ring_iter_is_active =
            packed_sources || has_sliding_window || ((active_ring_iter_mask >> ring_iter) & 1u) != 0;
        // Sliding already advanced/synchronized the sequencer above and uses a synthetic local
        // iteration whose K loop decodes the real source ring ID for each chunk.
        const uint32_t ring_id = [&]() -> uint32_t {
            SOL_ZONE("RD_SYNC");  // the wait for this iteration's all-gather arrival signal
            return packed_sources ? packed_source_ids[0]
                   : has_sliding_window
                       ? ring_index
                       : ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
                             fused_op_receiver.get_next_ring_id_and_sync(), mesh_rows, mesh_cols, snake_orientation);
        }();
#ifdef SOL_DEBUG
        // After the wait: the semaphore values the sync passed on. A value far above the iteration count means
        // the wait was already satisfied before this op's slabs arrived.
        DPRINT(
            "SYNC it={} rid={} sem0={} sem1={}\n",
            ring_iter,
            ring_id,
            Semaphore<>(fused_op_receiver.signal_op_semaphore_ids[0]).value(),
            Semaphore<>(fused_op_receiver.signal_op_semaphore_ids[1]).value());
#endif
        // Host precomputes which ring iterations have useful SDPA work; sync/ring-id sequencing
        // still advances above so reader stays aligned with compute, writer, and all-gather.
        if (!ring_iter_is_active) {
            continue;
        }
        // Compute consumes one sink tile for each real Q only when it normalizes on the
        // final active ring iteration. Keep the producer cadence identical, including
        // balanced Q chunks whose final iteration takes the normalize-only path.
        const bool is_last_ring_iter =
            has_sliding_window || (packed_sources ? ring_iter + 1 == sdpa_ring_iterations
                                                  : is_last_active_ring_iter(active_ring_iter_mask, ring_iter));
        // Iterate over KV blocks gathered on ring.
        // Sharded joint: the fused all-gather delivers one remote L/P shard per ring iteration (local
        // slice is read from the local joint tensor, same as spatial). Each shard is available right
        // after its ring_id sync, so we process joint K/V on EVERY ring iteration — no need to batch
        // at the end.
        // Replicated joint: process joint when ring_id == ring_size-1
        const bool do_joint_kv = has_gathered_joint_k ? true : (ring_id == ring_size - 1);
        uint32_t num_kv_chunks = num_local_k_chunks;
        if constexpr (has_joint_k) {
            if (do_joint_kv) {
                num_kv_chunks += num_joint_k_chunks;
            }
        }

        uint32_t KV_chunks_processed_in_iter = 0;
        uint32_t iter_num_kv_chunks = packed_sources ? packed_kv.chunk_count() : num_kv_chunks;

        // K split: this core's slice of the iteration's local K chunks, as compute derives it. Idle cores in a split
        // row use the row's index too, since they must skip exactly the chunks the row's multicast skips.
        ring_joint::KSplitRange ksplit_k_range = ring_joint::kKSplitAll;
        if constexpr (ksplit_enabled) {
            const uint32_t num_valid = ring_joint::ksplit_valid_local_k_chunks<
                kv_pad_rotation_enabled,
                chunked_enabled,
                kv_local_padded_Nt,
                chunk_size_t,
                q_local_padded_Nt,
                Sk_chunk_t>(num_local_k_chunks, ring_id, logical_nt, causal_end_nt);
            ksplit_k_range = ring_joint::ksplit_range(num_valid, ksplit_idx, ksplit_count);
        }

        // In causal balanced case processing KV received from other devices:
        //
        // We will have two logical chunks of the input sequence, logical indexes are:
        // ring_index and (seq_len / 2 * num_devices) - ring_index
        //
        // With this in mind we have two distinct cases when receiving from other device:
        // - 1st part of the sequence precedes both chunks on the sender device, 2nd part attends to both
        // - both chunks preced 2nd part of the sequence in received KV
        // Indexes are updated accordingly; compute is skipped
        if constexpr (!has_sliding_window) {
            if (is_causal && is_balanced && ring_index > ring_id) {
                iter_num_kv_chunks /= 2;
                // Mirror compute's K-loop extension: include the straddle chunk so K/V tiles
                // for it get loaded. Compute -inf-masks its late-half columns via lw_mask.
                using Straddle = KCausalStraddleInfo<kv_local_padded_Nt, Sk_chunk_t>;
                if constexpr (Straddle::has_straddle) {
                    iter_num_kv_chunks = Straddle::straddle_chunk_id + 1;
                }
            }
        }

        // Rotated: the factory precomputes this lockstep GROUP's mcast slot count per ring iteration
        // (the group max, so members with less real work still run the padded handshakes),
        // superseding the static per-chain maxima -- both max_q_per_core (shared-K) and
        // gqa_max_q_per_core (GQA), which is why this branch needs no per-family case.
        // Indexed by ACTIVE ordinal -- see rotated_active_ordinal.
        // Static: when K/V mcast is enabled, loop the per-chain max so receivers with less real Q
        // work still participate in padded multicast handshakes. Padded slots push their K/V entries,
        // which compute drains without computing, and never push Q.
        uint32_t loop_q_count;
        uint32_t rotated_my_count = 0;
        RotatedQSlots rotated_slots;
        constexpr uint32_t rotation_unit_chunks = use_zigzag_balancing ? 2 : 1;
        if constexpr (rotated_q_split_enabled) {
            // Grouped split-KV runs only with a full active mask, where ordinal == ring_iter.
            ASSERT(!packed_sources || active_ring_iter_mask == (ring_joint::all_sources_mask(ring_size)));
            const uint32_t rotated_ordinal = rotated_active_ordinal(active_ring_iter_mask, ring_iter);
            const uint32_t rotated_iter_base =
                ::rotated_iter_base(rotated_args_base, rotated_iter_stride, rotated_ordinal);
            rotated_slots = {global_q_start, q_per_core, get_arg_val<uint32_t>(rotated_iter_base)};
            loop_q_count = q_per_core + get_arg_val<uint32_t>(rotated_iter_base + 1) * rotation_unit_chunks;
            rotated_my_count = rotated_slots.count(rotation_unit_chunks);
        } else {
            loop_q_count = q_per_core;
            if constexpr (k_uses_batch_chain && batch_mcast_enabled) {
                loop_q_count = max_q_per_core;
            }
            if constexpr (gqa_grouped_kv && gqa_mcast_enabled) {
                loop_q_count = gqa_max_q_per_core;
            }
        }
        uint32_t gqa_group_q_iter = 0;

        [[maybe_unused]] const uint32_t do_pooled = (sol_pooled_pending && !sol_pooled_off) ? 1u : 0u;
        if constexpr (sol_resident) {
            // ---- Sol-Attn resident mode. One pass of each K/V piece per ring iteration, applied by compute to
            // every resident Q chunk (sdpa_ring_resident). This reader:
            //   1. on the first active iteration reads every resident Q chunk into cb_q_in, once for the whole op;
            //   2. per head the core's chunks fall in: the chain-wide kept union (per piece) over every chunk of
            //      every chain member, then one plan page per resident chunk [this shard's kept-block words,
            //      union piece bits lo, hi, do_pooled]; the mask is the block-form one, a piece is kept by a
            //      chunk when the chunk keeps any 64-token block of it, and compute steps only on those blocks;
            //   3. streams the shard's pieces once each, heads interleaved per k in head order, skipping a piece
            //      no member of the head's chain needs (compute mirrors both rules);
            //   4. on the first active iteration the pooled K/V chunks once per head, then one bias row per
            //      resident chunk of that head -- K and V before the biases, so compute's first step on the entry
            //      never waits for V behind a bias row the reader could not push yet.
            constexpr uint32_t kResMax = 32;
            const uint32_t n_res = q_per_core;
            ASSERT(n_res >= 1 && n_res <= kResMax);
            uint32_t res_nb[kResMax];
            uint32_t res_nq[kResMax];
            uint32_t res_qc[kResMax];
            for (uint32_t i = 0; i < n_res; ++i) {
                const auto d = decompose_global_q_index(global_q_start + i, num_q_chunks, NH, use_zigzag_balancing);
                res_nb[i] = d.nb;
                res_nq[i] = d.nq;
                res_qc[i] = d.q_chunk;
            }
            // 1. Q, once. Before any chain forward of this op, so no NoC write is in flight across the read barrier.
            if (!q_pushed) {
                SOL_ZONE("RD_RES_Q");
                for (uint32_t i = 0; i < n_res; ++i) {
                    const Slice q_slice(
                        res_nb[i], res_nq[i], res_qc[i] * Sq_chunk_t, res_qc[i] * Sq_chunk_t + Sq_chunk_t, 0, DHt);
                    read_block(
                        q_generator, q_slice, q_local_padded_Nt, cb_q_in, q_tile_bytes, false, q_barrier_threshold);
                }
                // The frozen per-slot max is set by compute from each slot's first step, so the sol_row_max tensor,
                // when given, is accepted but unused.
                q_pushed = true;
            }
            // 2. Kept bits and plans. The plan page's scratch (past the 64-byte header) holds one chunk row at a time:
            //    the whole block-form row (n_blocks_pad words, every shard), which the bias generator needs too.
            CircularBuffer cb_plan(cb_sol_plan);
            cb_plan.reserve_back(1);
            uint32_t scratch = cb_plan.get_write_ptr() + 64;
            const auto read_chunk_row = [&](uint32_t nq, uint32_t qc) {
                noc.async_read(
                    sol_mask_reader,
                    CoreLocalMem<uint32_t>(scratch),
                    sol_mask_row_bytes,
                    {.page_id = nq * sol_nqc_pad + qc},
                    {});
                noc.async_read_barrier();
            };
            // This shard's slice of a row (nb_pad words from block ring_id * nb_pad) into the scratch: all the bit
            // extraction needs, and a fraction of the whole row for every member row on the chain's critical path.
            const auto read_shard_slice = [&](uint32_t nq, uint32_t qc) {
                noc.async_read(
                    sol_mask_reader,
                    CoreLocalMem<uint32_t>(scratch),
                    sol_nb_pad * sizeof(uint32_t),
                    {.page_id = nq * sol_nqc_pad + qc, .offset_bytes = ring_id * sol_nb_pad * sizeof(uint32_t)},
                    {});
                noc.async_read_barrier();
            };
            // This shard's kept FULL blocks of the slice in the scratch, as bits (block b: word b/32, bit b%32).
            // Block 0 forced: every (Q chunk, shard) computes at least one block, compute creates each chunk's state
            // on piece 0, and the pre-pass forces the same block (the bias row is patched to match below).
            // (>= 1 so the arrays below are well-formed in the non-resident builds, where this branch is discarded.)
            constexpr uint32_t kBlockWords = sol_res_block_words > 0 ? sol_res_block_words : 1;
            constexpr uint32_t kBlocksPerPiece = Sk_chunk_t / sol_bias_gen::kBlockTiles;
            uint32_t full_bits[kBlockWords];  // the shard's full blocks, once per ring iteration
            for (uint32_t w = 0; w < kBlockWords; ++w) {
                full_bits[w] = 0;
            }
            for (uint32_t b = 0; b < sol_nb; ++b) {
                if (sol_block_is_full(ring_id, b)) {
                    full_bits[b >> 5] |= 1u << (b & 31);
                }
            }
            const auto row_block_bits = [&](uint32_t* words) {
                CoreLocalMem<volatile uint32_t> row(scratch);
                for (uint32_t w = 0; w < kBlockWords; ++w) {
                    words[w] = 0;
                }
                for (uint32_t b = 0; b < sol_nb; ++b) {
                    if (row[b] != 0) {
                        words[b >> 5] |= 1u << (b & 31);
                    }
                }
                for (uint32_t w = 0; w < kBlockWords; ++w) {
                    words[w] &= full_bits[w];
                }
                words[0] |= 1u;
            };
            // Piece k is kept when any of its blocks is: the streaming / chain-union granularity.
            const auto piece_bits = [&](const uint32_t* words, uint32_t& lo, uint32_t& hi) {
                lo = 0;
                hi = 0;
                for (uint32_t k = 0; k < num_local_k_chunks; ++k) {
                    bool kept = false;
                    for (uint32_t j = 0; j < kBlocksPerPiece && !kept; ++j) {
                        const uint32_t b = k * kBlocksPerPiece + j;
                        kept = b < sol_nb && ((words[b >> 5] >> (b & 31)) & 1u) != 0;
                    }
                    if (kept) {
                        if (k < 32) {
                            lo |= 1u << k;
                        } else {
                            hi |= 1u << (k - 32);
                        }
                    }
                }
            };
            const auto row_kept_bits = [&](uint32_t& lo, uint32_t& hi) {
                uint32_t words[kBlockWords];
                row_block_bits(words);
                piece_bits(words, lo, hi);
            };
            uint32_t res_lo[kResMax];
            uint32_t res_hi[kResMax];
            uint32_t seg_ulo[kResMax];
            uint32_t seg_uhi[kResMax];
            // One-page plan (sol_res_plan_rec_base != 0): every chunk's record [kept-block words][union lo, hi]
            // [do_pooled] in the one plan page, from word rec_base (past the shard slice the scratch reads), pushed
            // once per iteration. The per-chunk pages made the reader wait for compute to pop each one, so the next
            // iteration's plan could not go out before compute had finished this iteration's pieces.
            constexpr uint32_t kPlanRec = kBlockWords + 3;
            [[maybe_unused]] const auto plan_rec = [&](uint32_t i) {
                return CoreLocalMem<volatile uint32_t>(
                    cb_plan.get_write_ptr() + (sol_res_plan_rec_base + i * kPlanRec) * sizeof(uint32_t));
            };
            {
                SOL_ZONE("RD_RES_BITS");  // mask rows of this core's chunks and of every chain member's
                for (uint32_t i = 0; i < n_res; ++i) {
                    read_shard_slice(res_nq[i], res_qc[i]);
                    if constexpr (sol_res_plan_rec_base != 0) {
                        // One read per chunk: the block words go straight into the chunk's record.
                        uint32_t words[kBlockWords];
                        row_block_bits(words);
                        auto rec = plan_rec(i);
                        for (uint32_t w = 0; w < kBlockWords; ++w) {
                            rec[w] = words[w];
                        }
                        piece_bits(words, res_lo[i], res_hi[i]);
                    } else {
                        row_kept_bits(res_lo[i], res_hi[i]);
                    }
                }
            }
            // Head segments and their chain-wide unions. The chain members cover the head this core participates
            // in; a segment of another head is a non-participant there (it reads its own K/V), so union = own.
            for (uint32_t i0 = 0; i0 < n_res;) {
                uint32_t i1 = i0;
                uint32_t ulo = 0;
                uint32_t uhi = 0;
                while (i1 < n_res && res_nq[i1] == res_nq[i0]) {
                    ulo |= res_lo[i1];
                    uhi |= res_hi[i1];
                    ++i1;
                }
                if (sol_n_members > 0 && head_cfg.participates && res_nq[i0] == head_cfg.head) {
                    // Chain / group members stream every piece that holds a full block of this shard (block 0
                    // forced, as in row_block_bits): a superset of the members' union that needs no mask reads (the
                    // union read every member's row per ring iteration, ~200 us while compute idled, and at these
                    // densities it is nearly every piece anyway). The same on every member (it depends on the shard
                    // only) and inside every live piece, so compute agrees; it skips a piece it keeps no block of.
                    uint32_t words[kBlockWords];
                    for (uint32_t w = 0; w < kBlockWords; ++w) {
                        words[w] = full_bits[w];
                    }
                    words[0] |= 1u;
                    piece_bits(words, ulo, uhi);
                }
                for (uint32_t i = i0; i < i1; ++i) {
                    seg_ulo[i] = ulo;
                    seg_uhi[i] = uhi;
                }
                i0 = i1;
            }
            if constexpr (sol_res_plan_rec_base != 0) {
                SOL_ZONE("RD_RES_PLAN");
                for (uint32_t i = 0; i < n_res; ++i) {
                    auto rec = plan_rec(i);
                    rec[kBlockWords] = seg_ulo[i];
                    rec[kBlockWords + 1] = seg_uhi[i];
                    rec[kBlockWords + 2] = do_pooled;
                }
                cb_plan.push_back(1);
            } else {
                SOL_ZONE("RD_RES_PLAN");  // long when compute is behind: the reserve waits for the previous plan pop
                for (uint32_t i = 0; i < n_res; ++i) {
                    if (i > 0) {
                        cb_plan.reserve_back(1);  // one page: waits for compute to pop the previous plan
                        scratch = cb_plan.get_write_ptr() + 64;
                    }
                    // The chunk's own block bits, from its slice again (the scratch held the members' since).
                    read_shard_slice(res_nq[i], res_qc[i]);
                    uint32_t words[kBlockWords];
                    row_block_bits(words);
                    CoreLocalMem<volatile uint32_t> plan_words(cb_plan.get_write_ptr());
                    for (uint32_t w = 0; w < kBlockWords; ++w) {
                        plan_words[w] = words[w];
                    }
                    plan_words[kBlockWords] = seg_ulo[i];
                    plan_words[kBlockWords + 1] = seg_uhi[i];
                    plan_words[kBlockWords + 2] = do_pooled;
                    cb_plan.push_back(1);
                }
            }
            // 3. The shard's pieces, once each; heads interleaved per k in head order.
            const bool source_is_local = ring_id == ring_index;
            const uint32_t source_start_tile = ring_id * kv_local_padded_Nt;
            const uint32_t source_valid_kv_tiles =
                logical_nt > source_start_tile ? std::min(logical_nt - source_start_tile, kv_local_padded_Nt) : 0;
            const uint32_t end_seq_tile =
                source_is_local ? source_valid_kv_tiles : source_start_tile + source_valid_kv_tiles;
            // Chain overlap. The chain protocol allows one outstanding transfer per LINK: a member signals
            // "ready", the upstream writes and flags valid. In MHA the K and V transfers of a piece go over the
            // same link (k_chain and v_chain are the same head_chain, one semaphore set), so the link's handshakes
            // form one sequence: K(p), V(p), K(p+1), ... Signalling each only after the previous forward completes
            // would make the chain a bucket brigade, every hop idle while its neighbours move. Instead a member
            // signals the link's NEXT handshake before forwarding the current chunk, whenever that chunk's page is
            // free: V(p)'s ready before forwarding K(p), K(p+1)'s ready before forwarding V(p). The upstream's
            // write then overlaps this core's forward, and the link never has more than one transfer outstanding
            // (the next signal always follows the wait for the previous data). An armed K page carries to the
            // next stream_piece call; nothing is armed after the iteration's last piece (the pooled fetches and
            // the dummy pushes reserve pages themselves). Fetchers (injectors, standalone two-head cores) signal
            // nothing and only gain the early reserve.
            bool k_armed = false;
            uint32_t k_armed_addr = 0;
            const auto stream_piece = [&](uint32_t nb, uint32_t nq, uint32_t k, bool has_next) {
                const uint32_t nk = nq / q_heads_per_k;
                const uint32_t nv = nq / q_heads_per_v;
                const uint32_t kv_batch = indexed_kv_cache ? kv_cache_batch_idx : nb;
                const uint32_t gathered_kv_batch = indexed_kv_cache ? 0 : nb;
                const uint32_t src_batch = source_is_local ? kv_batch : gathered_kv_batch;
                const uint32_t row0 = (source_is_local ? 0 : source_start_tile) + k * Sk_chunk_t;
                const Slice k_slice(src_batch, nk, row0, row0 + Sk_chunk_t, 0, DHt);
                // Chain scope keys: the query head for MHA chains, the KV head for grouped ones (as the dense loop).
                const uint32_t k_head = gqa_grouped_kv ? nk : nq;
                const bool k_rx = k_chain.should_receive(k_head);
                const bool v_rx = v_chain.should_receive(nv);
                CircularBuffer cb_k(cb_k_in);
                uint32_t k_addr;
                if (k_armed) {
                    // Armed pages are only created for chain-received pieces, and a chain member receives every
                    // piece of its stream (cores whose chunks span two heads are kept off the chains), so an armed
                    // page always meets a chain piece: its ready signal has already been sent.
                    ASSERT(k_rx);
                    k_addr = k_armed_addr;
                    k_armed = false;
                } else {
                    cb_k.reserve_back(k_chunk_tiles);
                    k_addr = cb_k.get_write_ptr();
                    if (k_rx) {
                        k_chain.signal_ready(noc);
                    }
                }
                if (k_rx) {
                    SOL_ZONE("RD_CHAIN_RX");  // member: the wait for the upstream write (the ready signal went early)
                    k_chain.wait_data();
                } else if (source_is_local) {
                    SOL_ZONE("RD_FETCH_K");
                    fetch_block(local_k_generator, k_slice, end_seq_tile, cb_k_in, k_addr, k_tile_bytes, true);
                } else {
                    SOL_ZONE("RD_FETCH_K");
                    fetch_block(gathered_k_generator, k_slice, end_seq_tile, cb_k_in, k_addr, k_tile_bytes, true);
                }
                // Push before the forward: the forward only reads the page, its NoC write is flushed inside
                // forward(), and the page is not reused before piece p+2's reserve.
                cb_k.push_back(k_chunk_tiles);
                // The link's next handshake is V(p)'s: arm it (reserve V's page, signal ready) before forwarding
                // K(p) when the page is free now; otherwise reserve it after the forward.
                const Slice v_slice(src_batch, nv, row0, row0 + Sk_chunk_t, v_col_offset_t, v_col_offset_t + vDHt);
                CircularBuffer cb_v(cb_v_in);
                uint32_t v_addr = 0;
                bool v_reserved = false;
                if (v_rx && cb_v.pages_reservable_at_back(v_cb_entry_tiles)) {
                    cb_v.reserve_back(v_cb_entry_tiles);
                    v_addr = cb_v.get_write_ptr();
                    v_reserved = true;
                    v_chain.signal_ready(noc);
                }
                if (k_chain.should_forward(k_head, 0)) {
                    SOL_ZONE("RD_CHAIN_TX");  // the wait for the downstream ready signal, then the write
                    k_chain.forward(noc, k_addr, k_chunk_tiles, k_tile_bytes);
                }
                if (!v_reserved) {
                    cb_v.reserve_back(v_cb_entry_tiles);
                    v_addr = cb_v.get_write_ptr();
                    if (v_rx) {
                        v_chain.signal_ready(noc);
                    }
                }
                if (v_rx) {
                    v_chain.wait_data();
                } else if (source_is_local) {
                    fetch_block(local_v_generator, v_slice, end_seq_tile, cb_v_in, v_addr, v_tile_bytes, false);
                } else {
                    fetch_block(gathered_v_generator, v_slice, end_seq_tile, cb_v_in, v_addr, v_tile_bytes, false);
                }
                cb_v.push_back(v_cb_entry_tiles);
                // The link's next handshake is K(p+1)'s: arm it before forwarding V(p) when there is a next piece
                // and K's page is free now. The armed page is consumed by the next stream_piece call.
                if (k_rx && has_next && cb_k.pages_reservable_at_back(k_chunk_tiles)) {
                    cb_k.reserve_back(k_chunk_tiles);
                    k_armed_addr = cb_k.get_write_ptr();
                    k_armed = true;
                    k_chain.signal_ready(noc);
                }
                if (v_chain.should_forward(nv, 0)) {
                    v_chain.forward(noc, v_addr);
                }
                KV_chunks_processed_in_iter++;
            };
            uint32_t n_pieces = 0;  // pieces streamed this iteration (the first pass counts them)
            // D1: group delivery. Arguments (after the Sol chain members): role (0 = not in a group: reads its own
            // K/V; 1 = producer; 2 = consumer), producer NoC x, y, receiver count, rectangle count, then per
            // rectangle (x_start, y_start, x_end, y_end, cores), producer only. Every member pushes the same page
            // sequence into cb_k_in / cb_v_in (same pieces, same pooled pages), so
            // the producer's write pointer for a piece is every consumer's page for it.
            //   consumer: reserve the piece's K and V pages, credit the producer (+1), wait until the delivered
            //     count reaches the piece, push. At most one credit is outstanding (the next piece's), and never
            //     past the iteration's last piece (the pooled and dummy pages after it are this core's own).
            //   producer: reserve its own pages, read K and V from DRAM, push them for its compute, wait for
            //     receivers x (piece + 1) credits, multicast K and V to every rectangle, then the new count.
            // The credits share one counter, so the sum says every consumer has freed its page only because no
            // consumer can credit two pieces ahead: when the producer waits for piece p, every consumer has
            // received at most p - 1 and so credited at most p, and the sum reaches receivers x p only if each one
            // has. With a deeper lookahead, a fast consumer's extra credits would stand in for a slow one's and the
            // multicast would overwrite a page the slow core's compute has not popped. The CB still holds
            // kv_cb_pages pieces: the lookahead bounds only the producer's lead over the slowest receipt.
            [[maybe_unused]] const uint32_t g_base = sol_members_base + 2 * sol_n_members;
            [[maybe_unused]] const uint32_t g_role = sol_res_mcast ? get_arg_val<uint32_t>(g_base) : 0;
            [[maybe_unused]] uint32_t g_granted = 0;  // pieces credited this iteration
            [[maybe_unused]] uint32_t g_ahead = 0;    // pieces credited, not yet pushed
            const auto g_try_grant = [&](bool blocking) -> bool {
                if (g_granted >= n_pieces) {
                    return false;
                }
                CircularBuffer cb_k(cb_k_in);
                CircularBuffer cb_v(cb_v_in);
                const uint32_t k_need = k_chunk_tiles * (g_ahead + 1);
                const uint32_t v_need = v_cb_entry_tiles * (g_ahead + 1);
                if (!blocking &&
                    (g_ahead > 0 || !cb_k.pages_reservable_at_back(k_need) || !cb_v.pages_reservable_at_back(v_need))) {
                    return false;
                }
                cb_k.reserve_back(k_need);
                cb_v.reserve_back(v_need);
                ++g_ahead;
                ++g_granted;
                Semaphore<>(head_sender_semaphore_id)
                    .up(noc, get_arg_val<uint32_t>(g_base + 1), get_arg_val<uint32_t>(g_base + 2), 1);
                return true;
            };
            const auto g_grant_ahead = [&]() {
                while (g_try_grant(false)) {
                }
            };
            const auto stream_piece_group = [&](uint32_t nb, uint32_t nq, uint32_t k) {
                CircularBuffer cb_k(cb_k_in);
                CircularBuffer cb_v(cb_v_in);
                if (g_role == 2) {
                    if (g_ahead == 0) {
                        g_try_grant(true);
                    }
                    {
                        SOL_ZONE("RD_GROUP_RX");  // the wait for the producer's multicast (the credit went out earlier)
                        Semaphore<>(head_receiver_semaphore_id).wait_min(res_group_pieces + 1);
                    }
                    cb_k.push_back(k_chunk_tiles);
                    cb_v.push_back(v_cb_entry_tiles);
                    --g_ahead;
                    ++res_group_pieces;
                    g_grant_ahead();
                    KV_chunks_processed_in_iter++;
                    return;
                }
                // Producer.
                const uint32_t nk = nq / q_heads_per_k;
                const uint32_t nv = nq / q_heads_per_v;
                const uint32_t kv_batch = indexed_kv_cache ? kv_cache_batch_idx : nb;
                const uint32_t gathered_kv_batch = indexed_kv_cache ? 0 : nb;
                const uint32_t src_batch = source_is_local ? kv_batch : gathered_kv_batch;
                const uint32_t row0 = (source_is_local ? 0 : source_start_tile) + k * Sk_chunk_t;
                const Slice k_slice(src_batch, nk, row0, row0 + Sk_chunk_t, 0, DHt);
                const Slice v_slice(src_batch, nv, row0, row0 + Sk_chunk_t, v_col_offset_t, v_col_offset_t + vDHt);
                cb_k.reserve_back(k_chunk_tiles);
                cb_v.reserve_back(v_cb_entry_tiles);
                const uint32_t k_addr = cb_k.get_write_ptr();
                const uint32_t v_addr = cb_v.get_write_ptr();
                {
                    SOL_ZONE("RD_FETCH_K");
                    if (source_is_local) {
                        fetch_block(local_k_generator, k_slice, end_seq_tile, cb_k_in, k_addr, k_tile_bytes, true);
                        fetch_block(local_v_generator, v_slice, end_seq_tile, cb_v_in, v_addr, v_tile_bytes, false);
                    } else {
                        fetch_block(gathered_k_generator, k_slice, end_seq_tile, cb_k_in, k_addr, k_tile_bytes, true);
                        fetch_block(gathered_v_generator, v_slice, end_seq_tile, cb_v_in, v_addr, v_tile_bytes, false);
                    }
                }
                // Push first: the multicast only reads the pages, and they are not reserved again before this
                // core's compute has popped them and the writes below are flushed.
                cb_k.push_back(k_chunk_tiles);
                cb_v.push_back(v_cb_entry_tiles);
                ++res_group_pieces;
                const uint32_t n_recv = get_arg_val<uint32_t>(g_base + 3);
                const uint32_t n_rects = get_arg_val<uint32_t>(g_base + 4);
                {
                    SOL_ZONE("RD_GROUP_TX");  // the wait for the slowest consumer's credit, then the multicasts
                    Semaphore<>(head_sender_semaphore_id).wait_min(n_recv * res_group_pieces);
                    Semaphore<> count_src(head_valid_semaphore_id);
                    count_src.set(res_group_pieces);
                    for (uint32_t r = 0; r < n_rects; ++r) {
                        const uint32_t a = g_base + 5 + 5 * r;
                        // NoC 1 runs the other way: its rectangle starts at the high corner.
                        const bool rev = noc.get_noc_id() == 1;
                        const uint32_t x0 = get_arg_val<uint32_t>(a + (rev ? 2 : 0));
                        const uint32_t y0 = get_arg_val<uint32_t>(a + (rev ? 3 : 1));
                        const uint32_t x1 = get_arg_val<uint32_t>(a + (rev ? 0 : 2));
                        const uint32_t y1 = get_arg_val<uint32_t>(a + (rev ? 1 : 3));
                        const uint32_t n_dests = get_arg_val<uint32_t>(a + 4);
                        // K and V linked, then the count: back to back, no flush in between (ChainLink::forward).
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(k_addr),
                            MulticastEndpoint{},
                            k_chunk_tiles * k_tile_bytes,
                            n_dests,
                            {},
                            {.noc_x_start = x0, .noc_y_start = y0, .noc_x_end = x1, .noc_y_end = y1, .addr = k_addr},
                            /*linked=*/true);
                        noc.async_write_multicast(
                            CoreLocalMem<uint32_t>(v_addr),
                            MulticastEndpoint{},
                            v_cb_entry_tiles * v_tile_bytes,
                            n_dests,
                            {},
                            {.noc_x_start = x0, .noc_y_start = y0, .noc_x_end = x1, .noc_y_end = y1, .addr = v_addr},
                            /*linked=*/true);
                        count_src.relay_multicast(
                            noc, Semaphore<>(head_receiver_semaphore_id), x0, y0, x1, y1, n_dests, /*linked=*/false);
                    }
                    noc.async_writes_flushed();
                }
                KV_chunks_processed_in_iter++;
            };
            {
                SOL_ZONE("RD_RES_PIECES");  // the whole piece stream of this iteration (fetch / chain / CB waits)
                // Two passes over the same (piece, head group) order: a count first, so each streamed piece knows
                // whether another follows in this iteration (the lookahead arms the next page), then the stream.
                uint32_t j = 0;
                for (uint32_t pass = 0; pass < 2; ++pass) {
                    if constexpr (sol_res_mcast) {
                        if (pass == 1 && g_role == 2) {
                            g_grant_ahead();  // the first pieces' credits, before the first wait
                        }
                    }
                    for (uint32_t k = 0; k < num_local_k_chunks; ++k) {
                        if (!chunked_kv_chunk_is_live<
                                kv_pad_rotation_enabled,
                                chunked_enabled,
                                kv_local_padded_Nt,
                                chunk_size_t,
                                kv_region_Nt>(ring_id, k, Sk_chunk_t, logical_nt, causal_end_nt)) {
                            continue;
                        }
                        for (uint32_t i0 = 0; i0 < n_res;) {
                            uint32_t i1 = i0;
                            while (i1 < n_res && res_nq[i1] == res_nq[i0]) {
                                ++i1;
                            }
                            const bool union_kept =
                                k < 32 ? ((seg_ulo[i0] >> k) & 1u) != 0 : ((seg_uhi[i0] >> (k - 32)) & 1u) != 0;
                            if (union_kept) {
                                if (pass == 0) {
                                    ++n_pieces;
                                } else {
                                    ++j;
                                    if constexpr (sol_res_mcast) {
                                        // A group serves one head; a member's chunks never span two (factory).
                                        if (g_role != 0) {
                                            stream_piece_group(res_nb[i0], res_nq[i0], k);
                                        } else {
                                            stream_piece(res_nb[i0], res_nq[i0], k, j < n_pieces);
                                        }
                                    } else {
                                        stream_piece(res_nb[i0], res_nq[i0], k, j < n_pieces);
                                    }
                                }
                            }
                            i0 = i1;
                        }
                    }
                }
            }
            // 4. Pooled chunks, once per head, then one bias row per resident chunk of the head.
            if constexpr (sol_gen_bias) {
                if (do_pooled) {
                    SOL_ZONE("RD_RES_POOLED");
                    if constexpr (sol_res_plan_rec_base != 0) {
                        // The row reads below fill the whole scratch, records included: take the page back first,
                        // which waits until compute has popped (read) this iteration's plan. Short iterations
                        // can push every piece before compute starts. The next iteration's reserve returns it.
                        cb_plan.reserve_back(1);
                        scratch = cb_plan.get_write_ptr() + 64;
                    }
                    constexpr uint32_t pooled_rows_t = sol_n_blocks_pad / 32;
                    const sol_bias_gen::Geometry geo{
                        kv_local_padded_Nt * 32, sol_logical_n_tokens, sol_nb, sol_nb_pad, sol_nkc, Sk_chunk_t};
                    for (uint32_t i0 = 0; i0 < n_res;) {
                        uint32_t i1 = i0;
                        while (i1 < n_res && res_nq[i1] == res_nq[i0]) {
                            ++i1;
                        }
                        const uint32_t nq = res_nq[i0];
                        for (uint32_t pc = 0; pc < sol_pooled_chunks; ++pc) {
                            const uint32_t tiles = pc + 1 == sol_pooled_chunks ? sol_pooled_last_tiles : Sk_chunk_t;
                            const uint32_t row0 = pc * Sk_chunk_t;
                            CircularBuffer cb_k(cb_k_in);
                            cb_k.reserve_back(k_chunk_tiles);
                            sol_pk_generator.issue_reads(
                                Slice(0, nq, row0, row0 + tiles, 0, DHt),
                                pooled_rows_t,
                                cb_k_in,
                                cb_k.get_write_ptr(),
                                k_tile_bytes,
                                k_tile_bytes * Sk_chunk_t,
                                0);
                            noc.async_read_barrier();
                            noc.write_zeros_l1_barrier();
                            cb_k.push_back(k_chunk_tiles);
                            CircularBuffer cb_v(cb_v_in);
                            cb_v.reserve_back(v_cb_entry_tiles);
                            sol_pv_generator.issue_reads(
                                Slice(0, nq, row0, row0 + tiles, 0, vDHt),
                                pooled_rows_t,
                                cb_v_in,
                                cb_v.get_write_ptr(),
                                vDHt * v_tile_bytes,
                                v_tile_bytes,
                                0);
                            noc.async_read_barrier();
                            noc.write_zeros_l1_barrier();
                            cb_v.push_back(v_cb_entry_tiles);
                            KV_chunks_processed_in_iter++;
                            for (uint32_t i = i0; i < i1; ++i) {
                                // The chunk's own row back into the plan scratch: compute reads only the header
                                // words of a plan page, and every page has been popped by now.
                                read_chunk_row(nq, res_qc[i]);
                                // Block 0 of every live shard is forced exact by every ring iteration's bits (above),
                                // so mark it exact in the row the bias is generated from (no double counting).
                                {
                                    CoreLocalMem<volatile uint32_t> row(scratch);
                                    for (uint32_t s = 0; s < ring_size; ++s) {
                                        if (sol_block_is_full(s, 0)) {
                                            row[s * sol_nb_pad] = 1u;
                                        }
                                    }
                                }
                                CircularBuffer cb_bias(cb_sol_bias);
                                cb_bias.reserve_back(Sk_chunk_t);
                                // Block-form row: exact iff the block's own bit is set on a full block.
                                sol_bias_gen::generate_tiles<false, sol_bias_tile_bytes>(
                                    noc,
                                    cb_bias.get_write_ptr(),
                                    row0,
                                    tiles,
                                    geo,
                                    CoreLocalMem<volatile uint32_t>(scratch),
                                    sol_len_bias_table_base);
                                cb_bias.push_back(Sk_chunk_t);
                            }
                        }
                        i0 = i1;
                    }
                }
            }
#ifdef SOL_DEBUG
            // TT_SOL_DEBUG=1 (factory define): one line per (core, ring iteration) with what was decided and
            // pushed. Compute pops exactly KV_chunks_processed_in_iter entries (plus the same dummy padding) if the
            // two sides agree; a count that differs from a chain neighbour's is a phase fault.
            DPRINT(
                "RES it={} rid={} n={} heads={}/{} pieces0={} union0={} pushed={} pooled={} qread={}\n",
                ring_iter,
                ring_id,
                n_res,
                n_res ? res_nq[0] : 0u,
                n_res ? res_nq[n_res - 1] : 0u,
                n_res ? res_lo[0] : 0u,
                n_res ? seg_ulo[0] : 0u,
                KV_chunks_processed_in_iter,
                do_pooled,
                q_pushed ? 1u : 0u);
#endif
        } else if constexpr (sol_enabled && !sol_chunk_mode) {
            // ---- Sol-Attn block gathering: per Q chunk, gather this shard's kept 64-token blocks into dense
            // Sk-sized K/V chunks (8 blocks each); on the first active ring iteration also stream the pooled
            // K/V chunks with their per-Q-chunk column bias. Compute reads the plan words (n_gather,
            // last_active_tiles, do_pooled) from cb_sol_plan and runs its dense chunk step.
            const bool source_is_local = ring_id == ring_index;
            const uint32_t source_start_tile = ring_id * kv_local_padded_Nt;
            const uint32_t source_valid_kv_tiles =
                logical_nt > source_start_tile ? std::min(logical_nt - source_start_tile, kv_local_padded_Nt) : 0;
            const uint32_t source_end_seq_tile =
                source_is_local ? source_valid_kv_tiles : source_start_tile + source_valid_kv_tiles;
            const uint32_t source_row_base = source_is_local ? 0 : source_start_tile;
            for (uint32_t q_iter = 0; q_iter < q_per_core; ++q_iter) {
                const auto decoded_q =
                    decompose_global_q_index(global_q_start + q_iter, num_q_chunks, NH, use_zigzag_balancing);
                const uint32_t nb = decoded_q.nb;
                const uint32_t nq = decoded_q.nq;
                const uint32_t q_chunk = decoded_q.q_chunk;
                const uint32_t nk = nq / q_heads_per_k;
                const uint32_t nv = nq / q_heads_per_v;
                const uint32_t kv_batch = indexed_kv_cache ? kv_cache_batch_idx : nb;
                const uint32_t gathered_kv_batch = indexed_kv_cache ? 0 : nb;
                const uint32_t src_batch = source_is_local ? kv_batch : gathered_kv_batch;

                // 1. Mask row for (head, Q chunk) into the plan page scratch; scan this shard's blocks.
                CircularBuffer cb_plan(cb_sol_plan);
                cb_plan.reserve_back(1);
                const uint32_t plan_addr = cb_plan.get_write_ptr();
                CoreLocalMem<volatile uint32_t> plan_words(plan_addr);
                const uint32_t mask_row_addr = plan_addr + 64;
                noc.async_read(
                    sol_mask_reader,
                    CoreLocalMem<uint32_t>(mask_row_addr),
                    sol_mask_row_bytes,
                    {.page_id = nq * sol_nqc_pad + q_chunk},
                    {});
                noc.async_read_barrier();
                CoreLocalMem<volatile uint32_t> mask_row(mask_row_addr);
                const uint32_t shard_mask_base = ring_id * sol_nb_pad;
                // Only FULL blocks are gathered, matching compact mode and the bias generator (block_bit_full): a set
                // bit on a partial tail block would otherwise be computed here AND pooled by the bias -- counted
                // twice. The producer zeroes such bits (full_row), so this is defence in depth.
                uint32_t n_kept = 0;
                for (uint32_t b = 0; b < sol_nb; ++b) {
                    n_kept += (mask_row[shard_mask_base + b] != 0 && sol_block_is_full(ring_id, b)) ? 1u : 0u;
                }
                constexpr uint32_t blocks_per_chunk = Sk_chunk_t / kSolBlockTiles;
                const bool force_block0 = n_kept == 0;  // never leave an iteration without a chunk
                if (force_block0) {
                    n_kept = 1;
                }
                const uint32_t n_gather = (n_kept + blocks_per_chunk - 1) / blocks_per_chunk;
                const uint32_t last_active_tiles = (n_kept - (n_gather - 1) * blocks_per_chunk) * kSolBlockTiles;
                plan_words[0] = n_gather;
                plan_words[1] = last_active_tiles;
                plan_words[2] = do_pooled;
                plan_words[3] = 0;
                // Publish the plan before gathering: compute needs it before its first chunk, and the K/V CBs
                // hold only two chunks. The CB is one page: compute's pop does not touch the memory, and this
                // reader only reserves (and overwrites) the page again at its next Q chunk, after the scan below.
                cb_plan.push_back(1);

                // 2. Gathered K/V chunks.
                const Slice q_slice(nb, nq, q_chunk * Sq_chunk_t, q_chunk * Sq_chunk_t + Sq_chunk_t, 0, DHt);
                const bool need_q_read = (q_per_core > 1) || !q_pushed;
                bool first_chunk_for_q = true;
                uint32_t scan_b = 0;
                for (uint32_t g = 0; g < n_gather; ++g) {
                    const uint32_t blocks_here =
                        g + 1 == n_gather ? last_active_tiles / kSolBlockTiles : blocks_per_chunk;
                    // Collect this chunk's block ids (<= 8) so K and V issue the same gather.
                    uint32_t block_ids[Sk_chunk_t / kSolBlockTiles];
                    for (uint32_t slot = 0; slot < blocks_here; ++slot) {
                        if (force_block0) {
                            block_ids[slot] = 0;
                            continue;
                        }
                        while (mask_row[shard_mask_base + scan_b] == 0 || !sol_block_is_full(ring_id, scan_b)) {
                            ++scan_b;
                        }
                        block_ids[slot] = scan_b++;
                    }
                    // K^T chunk: block at slot s occupies K^T tile columns [2s, 2s+2) of the [DHt][Sk_chunk_t] layout.
                    CircularBuffer cb_k(cb_k_in);
                    cb_k.reserve_back(k_chunk_tiles);
                    const uint32_t k_dst = cb_k.get_write_ptr();
                    for (uint32_t slot = 0; slot < blocks_here; ++slot) {
                        const uint32_t row0 = source_row_base + block_ids[slot] * kSolBlockTiles;
                        const Slice k_slice(src_batch, nk, row0, row0 + kSolBlockTiles, 0, DHt);
                        if (source_is_local) {
                            local_k_generator.issue_reads(
                                k_slice,
                                source_end_seq_tile,
                                cb_k_in,
                                k_dst + slot * kSolBlockTiles * k_tile_bytes,
                                k_tile_bytes,
                                k_tile_bytes * Sk_chunk_t,
                                0);
                        } else {
                            gathered_k_generator.issue_reads(
                                k_slice,
                                source_end_seq_tile,
                                cb_k_in,
                                k_dst + slot * kSolBlockTiles * k_tile_bytes,
                                k_tile_bytes,
                                k_tile_bytes * Sk_chunk_t,
                                0);
                        }
                    }
                    noc.async_read_barrier();
                    noc.write_zeros_l1_barrier();
                    cb_k.push_back(k_chunk_tiles);
                    KV_chunks_processed_in_iter++;
                    if (first_chunk_for_q && need_q_read) {
                        read_block(
                            q_generator, q_slice, q_local_padded_Nt, cb_q_in, q_tile_bytes, false, q_barrier_threshold);
                        q_pushed = true;
                    }
                    first_chunk_for_q = false;
                    // V chunk: block at slot s occupies V tile rows [2s, 2s+2) of the [Sk_chunk_t][vDHt] layout.
                    CircularBuffer cb_v(cb_v_in);
                    cb_v.reserve_back(v_cb_entry_tiles);
                    const uint32_t v_dst = cb_v.get_write_ptr();
                    for (uint32_t slot = 0; slot < blocks_here; ++slot) {
                        const uint32_t row0 = source_row_base + block_ids[slot] * kSolBlockTiles;
                        const Slice v_slice(
                            src_batch, nv, row0, row0 + kSolBlockTiles, v_col_offset_t, v_col_offset_t + vDHt);
                        if (source_is_local) {
                            local_v_generator.issue_reads(
                                v_slice,
                                source_end_seq_tile,
                                cb_v_in,
                                v_dst + slot * kSolBlockTiles * vDHt * v_tile_bytes,
                                vDHt * v_tile_bytes,
                                v_tile_bytes,
                                0);
                        } else {
                            gathered_v_generator.issue_reads(
                                v_slice,
                                source_end_seq_tile,
                                cb_v_in,
                                v_dst + slot * kSolBlockTiles * vDHt * v_tile_bytes,
                                vDHt * v_tile_bytes,
                                v_tile_bytes,
                                0);
                        }
                    }
                    noc.async_read_barrier();
                    noc.write_zeros_l1_barrier();
                    cb_v.push_back(v_cb_entry_tiles);
                }

                // 3. Pooled K/V chunks + bias tile rows (first active ring iteration only).
                if (do_pooled) {
                    sol_push_pooled(nq, q_chunk, KV_chunks_processed_in_iter);
                }
            }
        } else {
            for (uint32_t q_iter = 0; q_iter < loop_q_count; ++q_iter) {
                // Check if this is a real iteration or only padded chain/mcast synchronization.
                bool is_padded_iter;
                uint32_t flat_q_index;
                if constexpr (rotated_q_split_enabled) {
                    is_padded_iter = (q_iter >= rotated_my_count);
                    // Padded iterations only handshake; decode a member of a valid owned unit so
                    // downstream indices and balanced causal skips match the multicast slot.
                    // Every core owns at least one complete base unit.
                    ASSERT(rotated_my_count >= rotation_unit_chunks && rotated_my_count <= rotated_max_slots);
                    ASSERT(rotated_my_count % rotation_unit_chunks == 0);
                    // Pad with the corresponding member of the last owned unit. A balanced low
                    // slot must skip on every multicast peer, including cores without a remainder.
                    flat_q_index = is_padded_iter ? padded_slot_flat_q<true>(
                                                        rotated_slots, 0, rotated_my_count, q_iter, rotation_unit_chunks)
                                                  : rotated_slots.at(q_iter);
                } else {
                    is_padded_iter = (q_iter >= q_per_core);
                    flat_q_index = padded_slot_flat_q<false>({}, global_q_start, q_per_core, q_iter, 1);
                }

                // Same-core GQA uses group-major (batch, Q chunk, head) scheduling so consecutive
                // iterations reuse one K/V window.  Every other specialization retains the ordinary
                // head-major flat order and optional causal zigzag remap.
                const auto decoded_q = decompose_global_q_index(flat_q_index, num_q_chunks, NH, use_zigzag_balancing);
                const uint32_t nb = decoded_q.nb;
                const uint32_t nq = decoded_q.nq;
                const uint32_t q_chunk = decoded_q.q_chunk;
                const uint32_t nk = nq / q_heads_per_k;
                const auto q_row_start_tile = q_chunk * Sq_chunk_t;
                const bool is_joint_q = has_joint_q ? (q_chunk >= num_local_q_chunks) : false;
                const uint32_t q_iter_local = [&]() {
                    if constexpr (enable_kv_chains && gqa_grouped_kv) {
                        return gqa_group_q_iter;
                    } else {
                        return q_iter;
                    }
                }();
                if constexpr (enable_kv_chains && gqa_grouped_kv) {
                    if (nk == gqa_cfg.head) {
                        gqa_group_q_iter++;
                    }
                }

                const bool balanced_skip_q =
                    !has_sliding_window && q_chunk < half_sequence && is_balanced && ring_index < ring_id;

                // A sink is one virtual key in the global softmax. It is consumed exactly once,
                // during final-iteration normalization. Decode its head from this iteration's
                // scheduled chunk, including migrated chunks; no sink state crosses ring iterations.
                // Padded chain/mcast iterations have no consumer and must not produce a sink tile.
                if constexpr (use_attention_sink) {
                    if (is_last_ring_iter && !is_padded_iter) {
                        constexpr uint32_t sink_tiles = 1;
                        cb_attn_sink.reserve_back(sink_tiles);
                        const uint32_t sink_write_ptr = cb_attn_sink.get_write_ptr();
                        const uint32_t sink_tile_id = attention_sink_tile_shape.id_of(0, nq, 0, 0);
                        noc.async_read(
                            attention_sink_reader,
                            CoreLocalMem<uint32_t>(sink_write_ptr),
                            attention_sink_tile_bytes,
                            {.page_id = sink_tile_id},
                            {});
                        noc.async_read_barrier();
                        cb_attn_sink.push_back(sink_tiles);
                    }
                }

                // Balanced causal skip: this Q chunk is handled by the paired device. Reader sends
                // nothing (no Q, no K/V) — compute's normalize-only path on the last ring iter does
                // not read Q (normalize uses only restored sum/out).
                // Skip logic applies to all iterations (including padded) so injector and receivers
                // make the same skip decisions, keeping chain/mcast sync aligned.
                if (balanced_skip_q) {
                    continue;
                }

                // Chunk mode: chain-wide kept union for this stream position (== own mask off the chain).
                [[maybe_unused]] uint32_t sol_union_lo = 0xffffffffu, sol_union_hi = 0xffffffffu;
                if constexpr (sol_chunk_mode) {
                    // ---- Sol-Attn chunk mode: publish this (ring iter, Q chunk)'s kept-chunk bitmask, the pooled
                    // flag and the chain-union bitmask. The K/V stream below is the dense one (chains intact) minus
                    // the chunks the whole chain skips; compute pops the streamed chunks whose own bit is clear
                    // without computing them, then runs the pooled chunks pushed after the loop.
                    if (!is_padded_iter) {
                        CircularBuffer cb_plan(cb_sol_plan);
                        cb_plan.reserve_back(1);
                        const uint32_t plan_addr = cb_plan.get_write_ptr();
                        CoreLocalMem<volatile uint32_t> plan_words(plan_addr);
                        // Header: chunk mode 16 words [kept lo, hi, pooled, union lo, hi]; compact mode 32 words,
                        // adding the per-chunk kept-block counts, one byte each, at words 8..23. The row scratch
                        // follows the header.
                        constexpr uint32_t sol_plan_header_bytes = sol_compact ? 128 : 64;
                        const uint32_t mask_row_addr = plan_addr + sol_plan_header_bytes;
                        sol_row_addr = mask_row_addr;
                        CoreLocalMem<volatile uint32_t> mask_row(mask_row_addr);
                        // Chunk mode reads a whole chunk row (nkc_pad words, global chunk index); compact mode reads
                        // only this shard's slice of the block row (nb_pad words at word offset ring_id * nb_pad).
                        const auto read_row = [&](uint32_t page) {
                            if constexpr (sol_compact) {
                                noc.async_read(
                                    sol_mask_reader,
                                    CoreLocalMem<uint32_t>(mask_row_addr),
                                    sol_mask_row_bytes,
                                    {.page_id = page, .offset_bytes = ring_id * sol_nb_pad * 4},
                                    {});
                            } else {
                                noc.async_read(
                                    sol_mask_reader,
                                    CoreLocalMem<uint32_t>(mask_row_addr),
                                    sol_mask_row_bytes,
                                    {.page_id = page},
                                    {});
                            }
                            noc.async_read_barrier();
                            if constexpr (sol_compact) {
                                // Fold the even-up rule into the slice once (index = block; whole shard row).
                                sol_bias_gen::Geometry g{
                                    kv_local_padded_Nt * 32,
                                    sol_logical_n_tokens,
                                    sol_nb,
                                    sol_nb_pad,
                                    sol_nkc,
                                    Sk_chunk_t};
                                g.even_up = sol_even_up;
                                g.even_up_all = sol_even_up_all;
                                sol_bias_gen::even_up_row(
                                    g, mask_row, ring_id * sol_nb_pad, ring_id * sol_nb_pad, sol_nb_pad);
                            }
                        };
                        const uint32_t shard_base = sol_compact ? 0u : ring_id * sol_nkc;
                        // Kept count of local chunk k in the row now in the scratch: compact mode counts FULL blocks
                        // with a set bit (partial tail blocks are never exact); chunk mode reads the chunk word (0 or
                        // 1).
                        const auto row_chunk_kept = [&](uint32_t k) -> uint32_t {
                            if constexpr (sol_compact) {
                                constexpr uint32_t bpc = Sk_chunk_t / kSolBlockTiles;
                                uint32_t cnt = 0;
                                for (uint32_t b = k * bpc; b < (k + 1) * bpc; ++b) {
                                    cnt += sol_block_exact(mask_row, ring_id, b) ? 1u : 0u;
                                }
                                return cnt;
                            } else {
                                return mask_row[shard_base + k] != 0 ? 1u : 0u;
                            }
                        };
                        // Every (Q chunk, shard) must keep at least one block (an iteration with zero computed chunks
                        // is not a supported state of the accumulator staging). Producers force block 0 of every shard
                        // (SOL_ATTN_INTERFACE.md); as a fallback the reader forces it here on any all-zero row slice.
                        // The fallback is applied identically to member rows, so the chain union stays consistent. (If
                        // a producer violates the rule, block 0 is computed exactly and also pooled: a small deviation,
                        // not a hang.)
                        const auto force_block0_if_empty = [&]() {
                            if constexpr (sol_compact) {
                                for (uint32_t b = 0; b < sol_nb; ++b) {
                                    if (mask_row[b] != 0 && sol_block_is_full(ring_id, b)) {
                                        return;
                                    }
                                }
                                // Only a FULL block can be exact. A shard whose live tail is 1..63 tokens has none,
                                // so this iteration streams nothing and compute runs no step for it -- and with
                                // q_per_core > 1 the writer then waits forever for the per-Q-chunk signal. The host
                                // rejects that geometry for a static logical_n (ring_joint_sdpa_device_operation.cpp);
                                // a live logical_n tensor cannot be checked there, so trap it under the watcher.
                                ASSERT(sol_block_is_full(ring_id, 0));
                                mask_row[0] = 1u;
                            }
                        };
                        const uint32_t own_page = nq * sol_nqc_pad + q_chunk;
                        read_row(own_page);
                        force_block0_if_empty();
                        if constexpr (sol_compact) {
                            for (uint32_t w = 8; w < 24; ++w) {
                                plan_words[w] = 0;
                            }
                        }
                        uint32_t lo = 0, hi = 0;
                        [[maybe_unused]] uint32_t sol_total_kept = 0;
                        for (uint32_t k = 0; k < num_local_k_chunks; ++k) {
                            const uint32_t cnt = row_chunk_kept(k);
                            if (cnt != 0) {
                                if (k < 32) {
                                    lo |= 1u << k;
                                } else {
                                    hi |= 1u << (k - 32);
                                }
                                sol_total_kept += cnt;
                                if constexpr (sol_compact) {
                                    const uint32_t w = 8 + k / 4;
                                    plan_words[w] = plan_words[w] | (cnt << (8 * (k % 4)));
                                }
                            }
                        }
                        if constexpr (sol_pack) {
                            // Packing: publish the kept-block total so compute does not re-derive it. Compute
                            // accumulates after try_skip_oob_kv, the causal skip and `k < num_kv_chunks`, while
                            // the packing walk below applies none of those and runs to q_k_loop_count, so two
                            // independent counts can drift (joint chunks, out-of-range chunks, padded
                            // iterations), the reader out-produces compute and parks in cb_reserve_back.
                            //
                            // This loop is the single producer of the count: the same `cnt` values that drive
                            // the per-chunk counts compute already trusts. Publish the BLOCK total, not the page
                            // count: compute needs both the number of pages and the fill of the trailing partial
                            // page, and deriving both from one published number keeps them consistent.
                            plan_words[5] = sol_total_kept;
#ifdef SOL_PACK_DEBUG
                            DPRINT(
                                "SX {} {} {} {}\n",
                                ring_id,
                                q_chunk,
                                sol_total_kept,
                                (sol_total_kept + sol_blocks_per_page - 1) / sol_blocks_per_page);
#endif
                        }
                        uint32_t ulo = lo, uhi = hi;
                        if constexpr (sol_pack) {
                            // Packing streams every in-range chunk through the chain and drops the chain-wide
                            // union. Compute never sees a raw chunk under packing (pages only), so nothing on its
                            // side depends on which chunks were streamed, and the reader packs from its own row
                            // regardless. The union would only skip chunks no member needs, which is rare at the
                            // block densities this path targets, while computing it costs one DRAM read and a
                            // bitmask scan per member per (ring iter, Q chunk).
                            ulo = 0xffffffffu;
                            uhi = 0xffffffffu;
                        } else if (sol_n_members > 0 && head_chain.is_participant && nq == head_chain.chain_head) {
                            const uint32_t j = q_chunk - sol_my_start;  // stream position along the chain
                            for (uint32_t m = 0; m < sol_n_members; ++m) {
                                const uint32_t m_start = get_arg_val<uint32_t>(sol_members_base + 2 * m);
                                const uint32_t m_count = get_arg_val<uint32_t>(sol_members_base + 2 * m + 1);
                                if (j >= m_count || m_start + j == q_chunk) {
                                    continue;  // member idle at this position, or this core itself
                                }
                                read_row(nq * sol_nqc_pad + m_start + j);
                                force_block0_if_empty();
                                for (uint32_t k = 0; k < num_local_k_chunks; ++k) {
                                    if (row_chunk_kept(k) != 0) {
                                        if (k < 32) {
                                            ulo |= 1u << k;
                                        } else {
                                            uhi |= 1u << (k - 32);
                                        }
                                    }
                                }
                            }
                            if constexpr (sol_compact) {
                                read_row(own_page);  // the K loop compacts from this core's own slice
                                force_block0_if_empty();
                            }
                        }
                        sol_union_lo = ulo;
                        sol_union_hi = uhi;
                        plan_words[0] = lo;
                        plan_words[1] = hi;
                        plan_words[2] = do_pooled;
                        plan_words[3] = ulo;
                        plan_words[4] = uhi;
                        cb_plan.push_back(1);
                    }
                }

                // Default to local Q tensor; override below for joint Q when applicable.
                Slice q_slice(nb, nq, q_row_start_tile, q_row_start_tile + Sq_chunk_t, 0, DHt);
                uint32_t q_end_seq_tile = q_local_padded_Nt;
                if constexpr (has_joint_q) {
                    if (is_joint_q) {
                        const uint32_t joint_q_row_start_tile = (q_chunk - num_local_q_chunks) * Sq_chunk_t;
                        q_slice = Slice(nb, nq, joint_q_row_start_tile, joint_q_row_start_tile + Sq_chunk_t, 0, DHt);
                        // Lt_local: per-device joint-Q shard tile count (== Lt on the replicated path).
                        // joint_q is a local tensor starting at row 0; the mesh tensor tracks which global
                        // prompt slice this device owns, so no ring_id offset is added here.
                        q_end_seq_tile = Lt_local;
                    }
                }

                // When q_per_core == 1, Q is identical across ring iterations: compute keeps it
                // fronted in the CB, so we only need to read it once on the first active ring iteration.
                // Rotated: this core's owned count varies per ring iteration, so the "single Q chunk
                // stays fronted in L1" optimization never applies -- compute derives its q_per_core from
                // the fixed slot count and would wait on a Q chunk the reader never re-pushed.
                const bool need_q_read = rotated_q_split_enabled || (q_per_core > 1) || !q_pushed;
                // Q download, one subblock at a time when the chunk has several so compute can start QK^T early.
                // Called from the first K chunk actually processed (after K is forwarded, so no outstanding NoC
                // writes remain -- the read barrier inside would deadlock with in-flight writes), and from the
                // Sol fallback after the loop when nothing was streamed.
                const auto read_q = [&](const auto& q_gen) {
                    if constexpr (use_q_subblock_push) {
                        for (uint32_t q_sub = 0; q_sub < q_num_subblocks; ++q_sub) {
                            const uint32_t sb_row_start = q_slice.d2_start + q_sub * qk_subblock_h;
                            const uint32_t sb_row_end = sb_row_start + qk_subblock_h;
                            Slice q_sub_slice(q_slice.d0, q_slice.d1, sb_row_start, sb_row_end, 0, DHt);
                            read_block(
                                q_gen,
                                q_sub_slice,
                                q_end_seq_tile,
                                cb_q_in,
                                q_tile_bytes,
                                false /*transpose*/,
                                q_barrier_threshold);
                        }
                    } else {
                        read_block(
                            q_gen,
                            q_slice,
                            q_end_seq_tile,
                            cb_q_in,
                            q_tile_bytes,
                            false /*transpose*/,
                            q_barrier_threshold);
                    }
                };

                ring_joint::SlidingQWorkPlan<SLIDING_MAX_SOURCE_RANGES> sliding_q_plan;
                if constexpr (has_sliding_window) {
                    sliding_q_plan = ring_joint::build_sliding_q_work_plan<SLIDING_MAX_SOURCE_RANGES>(
                        q_chunk * Sq_chunk_t,
                        Sq_chunk_t,
                        ring_index,
                        q_local_padded_Nt,
                        ring_size,
                        sliding_window_size,
                        tt::constants::TILE_HEIGHT,
                        kv_local_padded_Nt,
                        Sk_chunk_t,
                        logical_nt,
                        circular_kv_slab_count,
                        kv_pad_rotation_enabled ? &qmap : nullptr,
                        /*local_first=*/SLIDING_LOCAL_FIRST && ksplit_count == 1);
                    ASSERT(sliding_q_plan.is_valid);
                    ASSERT(sliding_q_plan.total_k_chunk_count > 0);
                }
                // Sliding K split: this core's band of the work plan (ring_joint_ksplit.hpp).
                const auto sliding_band =
                    ring_joint::sliding_ksplit_range(sliding_q_plan.total_k_chunk_count, ksplit_idx, ksplit_count);
                const uint32_t q_k_loop_count = has_sliding_window ? sliding_band.end : iter_num_kv_chunks;
                // Q must be pushed on the first K chunk actually PROCESSED for this Q chunk — not a
                // hardcoded k_chunk == 0. When spatial k_chunk 0 is beyond logical_n it is skipped, so
                // anchoring Q to k_chunk == 0 would never push Q while compute still waits on it
                // (q_per_core > 1) -> deadlock. Reads Q exactly once per q_iter, so no extra work.
                bool first_k_for_q = true;
                // Packed-page state (sol_pack). A page is filled across K chunks and flushed when full; a partial
                // page is flushed after the loop, because shard `ring_id`'s K/V is only resident for this iteration
                // and a page may not span iterations. K and V share one fill so their slots stay aligned: the K
                // site records the staging address, the V site packs both and commits the fill.
                [[maybe_unused]] uint32_t sol_pack_fill = 0;
                [[maybe_unused]] uint32_t sol_pack_k_src = 0;  // K staging address, carried to the V site
                [[maybe_unused]] uint32_t sol_pages_pushed = 0;  // SOL_PACK_DEBUG: pages handed to compute
                [[maybe_unused]] uint32_t sol_packed_k_addr = 0;
                [[maybe_unused]] uint32_t sol_packed_v_addr = 0;
                if constexpr (sol_pack) {
                    // Packed pages go out on the ordinary K/V CBs, so compute's inner-step call is the dense one.
                    //
                    // Guarded on !is_padded_iter to match the plan block and the packing sites: a padded
                    // iteration writes no plan and packs nothing, so compute processes nothing for it. An
                    // unguarded reserve_back here would block waiting for space it never uses -- and after the
                    // last real iteration compute has stopped draining, so it would block forever.
                    if (!is_padded_iter) {
                        CircularBuffer cb_kp(cb_k_in), cb_vp(cb_v_in);
                        cb_kp.reserve_back(k_chunk_tiles);
                        cb_vp.reserve_back(v_cb_entry_tiles);
                        sol_packed_k_addr = cb_kp.get_write_ptr();
                        sol_packed_v_addr = cb_vp.get_write_ptr();
                    }
                }
                for (uint32_t k_chunk = has_sliding_window ? sliding_band.begin : 0; k_chunk < q_k_loop_count;
                     ++k_chunk) {
                    const auto sliding_k_chunk = sliding_q_plan.k_chunk_at(k_chunk);
                    const uint32_t source_ring_id = has_sliding_window ? sliding_k_chunk.source_ring_id : ring_id;
                    const uint32_t source_k_chunk = has_sliding_window ? sliding_k_chunk.source_k_chunk : k_chunk;
                    /**
                     * Iterate over all KV chunks for this Q chunk.
                     * If this is the last ring ID, we will also read from joint KV.
                     * If this k chunk is in the spatial input and beyond the logical N, we will skip it.
                     */
                    const bool kv_chunk_is_joint = !has_sliding_window && has_joint_k && k_chunk >= num_local_k_chunks;
                    const bool kv_chunk_is_beyond_logical_n = [&]() {
                        if (has_sliding_window || packed_sources || kv_chunk_is_joint) {
                            return false;
                        }
                        if (!ksplit_k_range.contains(source_k_chunk)) {
                            return true;
                        }
                        // causal_end_nt is logical_nt when the causal skip is off.
                        return !chunked_kv_chunk_is_live<
                            kv_pad_rotation_enabled,
                            chunked_enabled,
                            kv_local_padded_Nt,
                            chunk_size_t,
                            kv_region_Nt>(source_ring_id, source_k_chunk, Sk_chunk_t, logical_nt, causal_end_nt);
                    }();

                    // Sharded joint: this ring iteration serves shard `ring_id`, whose global joint tile
                    // range starts at ring_id * Lt_local. A joint K chunk whose global start tile is
                    // at/after logical_lt lies entirely in the padded joint tail — skip it, exactly like
                    // the spatial per-chunk logical_n skip above. Compute mirrors this skip, so the
                    // K/V CB producer/consumer counts stay in lockstep (no deadlock).
                    bool kv_chunk_is_beyond_logical_l = false;
                    if constexpr (has_gathered_joint_k) {
                        if (kv_chunk_is_joint) {
                            const uint32_t joint_global_start_tile =
                                ring_id * Lt_local + (k_chunk - num_local_k_chunks) * Sk_chunk_t;
                            kv_chunk_is_beyond_logical_l = (joint_global_start_tile >= logical_lt);
                        }
                    }

                    if (kv_chunk_is_beyond_logical_n || kv_chunk_is_beyond_logical_l) {
                        // KV chunk on spatial input beyond logical N, or joint chunk beyond logical L. Skip it.
                        // A sliding work plan is already clipped to logical_n, so this would otherwise
                        // desynchronize reader and compute K-loop counts.
                        ASSERT(!has_sliding_window);
                        continue;
                    }
                    if constexpr (sol_chunk_mode) {
                        // Chain-wide skip: no member needs this chunk at this stream position, so nobody reads,
                        // forwards or pushes it (every member computes the same union -> CB pointers stay aligned).
                        const bool union_kept = k_chunk < 32 ? ((sol_union_lo >> k_chunk) & 1u) != 0
                                                             : ((sol_union_hi >> (k_chunk - 32)) & 1u) != 0;
                        if (!is_padded_iter && !union_kept) {
                            continue;
                        }
                    }

                    // Default to local/gathered KV; override below for joint KV when applicable.
                    Slice k_slice;
                    uint32_t end_seq_tile;
                    uint32_t source_valid_kv_tiles = kv_local_padded_Nt;
                    if constexpr (!chunked_enabled) {
                        const uint32_t source_start_tile = source_ring_id * kv_local_padded_Nt;
                        source_valid_kv_tiles = logical_nt > source_start_tile
                                                    ? std::min(logical_nt - source_start_tile, kv_local_padded_Nt)
                                                    : 0;
                    }
                    // Local KV reads the indexed cache slot; gathered KV is at slot 0 of the scratch buffer.
                    const uint32_t kv_batch = indexed_kv_cache ? kv_cache_batch_idx : nb;
                    const uint32_t gathered_kv_batch = indexed_kv_cache ? 0 : nb;
                    const bool source_is_local = source_ring_id == ring_index;
                    if (source_is_local) {
                        const uint32_t local_k_start_tile = source_k_chunk * Sk_chunk_t;
                        k_slice = Slice(kv_batch, nk, local_k_start_tile, local_k_start_tile + Sk_chunk_t, 0, DHt);
                        end_seq_tile = source_valid_kv_tiles;
                    } else {
                        uint32_t gathered_start_tile =
                            source_ring_id * kv_local_padded_Nt + source_k_chunk * Sk_chunk_t;
                        if constexpr (has_sliding_window) {
                            synchronize_halo();
                            gathered_start_tile = sliding_k_chunk.compact_k_chunk * Sk_chunk_t;
                        }
                        k_slice =
                            Slice(gathered_kv_batch, nk, gathered_start_tile, gathered_start_tile + Sk_chunk_t, 0, DHt);
                        end_seq_tile = has_sliding_window ? gathered_padded_Nt
                                                          : source_ring_id * kv_local_padded_Nt + source_valid_kv_tiles;
                    }
                    if constexpr (has_joint_k) {
                        if (kv_chunk_is_joint) {
                            const uint32_t joint_chunk_offset = (k_chunk - num_local_k_chunks) * Sk_chunk_t;
                            if constexpr (has_gathered_joint_k) {
                                // Sharded: each ring iteration processes the shard belonging to ring_id.
                                // The fused AG does NOT write the local device's slice into the gathered
                                // buffer — read from local tensor (local frame) when ring_id == ring_index,
                                // from gathered buffer (global frame) for all other devices' slices.
                                if (ring_id == ring_index) {
                                    // Local slice: k_slice in local tensor frame (rows 0..Lt_local).
                                    k_slice =
                                        Slice(nb, nk, joint_chunk_offset, joint_chunk_offset + Sk_chunk_t, 0, DHt);
                                    end_seq_tile = Lt_local;
                                } else {
                                    // Remote slice: k_slice in gathered buffer frame.
                                    const uint32_t joint_k_row_start_tile = ring_id * Lt_local + joint_chunk_offset;
                                    k_slice = Slice(
                                        nb, nk, joint_k_row_start_tile, joint_k_row_start_tile + Sk_chunk_t, 0, DHt);
                                    end_seq_tile = ring_id * Lt_local + Lt_local;
                                }
                            } else {
                                // Replicated: read starting at tile 0 of the local full-L joint tensor.
                                k_slice = Slice(nb, nk, joint_chunk_offset, joint_chunk_offset + Sk_chunk_t, 0, DHt);
                                end_seq_tile = Lt;
                            }
                        }
                    }
                    const bool joint_chunk_is_local =
                        has_gathered_joint_k ? (kv_chunk_is_joint && ring_id == ring_index) : false;

                    if (packed_sources) {
                        // A chunk can span several sources. Wait through its last
                        // member before either a local read or chain receive; later
                        // Q chunks reuse the readiness already established here.
                        packed_readiness.wait_for_chunk(packed_kv, k_chunk, wait_packed_source);
                    }

                    // K: either read locally (injector or not participant) or receive from chain
                    const uint32_t k_chain_head = [&]() {
                        if constexpr (gqa_grouped_kv) {
                            return nk;
                        } else {
                            return nq;
                        }
                    }();
                    CircularBuffer cb_k(cb_k_in);
                    uint32_t cb_k_start_address;
                    if constexpr (sol_pack) {
                        // Raw chunks land in staging; cb_k_in carries the packed page, whose reserve/push is
                        // owned by the page lifecycle around this loop.
                        //
                        // Do NOT reserve cb_k_in here. A per-chunk reserve contends with that outstanding
                        // packed-page reserve, and the mcast branch below asks for 2 * k_chunk_tiles from a
                        // 2-page CB -- unsatisfiable the moment one page is in flight, so the reader would stall
                        // in cb_reserve_back for ever while compute never waits for data.
                        CircularBuffer cb_ks(cb_k_stage);
                        cb_ks.reserve_back(k_chunk_tiles);
                        cb_k_start_address = cb_ks.get_write_ptr();
                    } else {
                        // Padded slots push like real ones, so a multicast always targets this core's write pointer.
                        cb_k.reserve_back(k_chunk_tiles);
                        cb_k_start_address = cb_k.get_write_ptr();
                    }
                    bool received_k_from_chain = false;
                    if constexpr (!has_sliding_window) {
                        if (k_chain.should_receive(k_chain_head)) {
                            k_chain.receive(noc);
                            received_k_from_chain = true;
                        }
                    }
                    if (packed_sources && !received_k_from_chain) {
                        // Keep the physical K^T stride and forwarding size unchanged.
                        // A full packed chunk writes every tile at that stride, so
                        // prior CB contents cannot be observed after the read barrier.
                        // Partial chunks retain conservative whole-buffer zeroing.
                        if (packed_kv.valid_tiles(k_chunk) < Sk_chunk_t) {
                            for (uint32_t tile = 0; tile < k_chunk_tiles; ++tile) {
                                fill_zeros_async(noc, cb_k_in, k_tile_bytes, tile * k_tile_bytes);
                            }
                            noc.write_zeros_l1_barrier();
                        }
                        for (uint32_t dst_row = 0; dst_row < packed_kv.valid_tiles(k_chunk);) {
                            const uint32_t stream_row = k_chunk * Sk_chunk_t + dst_row;
                            const auto location = packed_kv.locate(stream_row, packed_source_ids);
                            const uint32_t source = location.rank;
                            const uint32_t local_row = location.local;
                            const uint32_t rows = packed_kv.segment_tiles(k_chunk, dst_row);
                            const bool local = source == ring_index;
                            const uint32_t source_row = local ? local_row : source * kv_local_padded_Nt + local_row;
                            const uint32_t source_end = local ? kv_local_padded_Nt : (source + 1) * kv_local_padded_Nt;
                            const Slice segment(
                                local ? kv_batch : gathered_kv_batch, nk, source_row, source_row + rows, 0, DHt);
                            const auto issue_segment = [&](const auto& generator) {
                                generator.issue_reads(
                                    segment,
                                    source_end,
                                    cb_k_in,
                                    cb_k_start_address + dst_row * k_tile_bytes,
                                    k_tile_bytes,
                                    Sk_chunk_t * k_tile_bytes,
                                    0);
                            };
                            if (local) {
                                issue_segment(local_k_generator);
                            } else {
                                issue_segment(gathered_k_generator);
                            }
                            dst_row += rows;
                        }
                        noc.async_read_barrier();
                    } else if (!received_k_from_chain) {
                        // Injector or non-participant: read K from DRAM. Dispatch directly so
                        // local and gathered tensors may use different accessor types.
                        const auto fetch_k = [&](const auto& k_gen) {
                            fetch_block(
                                k_gen,
                                k_slice,
                                end_seq_tile,
                                cb_k_in,
                                cb_k_start_address,
                                k_tile_bytes,
                                true /*transpose*/);
                        };
                        fetch_k_from_source<has_joint_k, has_gathered_joint_k, joint_tensor_args_offset>(
                            kv_chunk_is_joint,
                            joint_chunk_is_local,
                            // Local vs gathered spatial source is keyed off the source ring id, not the
                            // loop index, so the sliding-window plan's out-of-order K chunks resolve too.
                            source_is_local ? 0 : 1,
                            joint_k_addr,
                            gathered_joint_k_addr,
                            local_k_generator,
                            gathered_k_generator,
                            joint_q_input_tile_logical,
                            joint_input_tile_logical,
                            fetch_k);
                    }

                    // Forward K chunk via chain (uses K's data size explicitly)
                    if constexpr (!has_sliding_window) {
                        if (k_chain.should_forward(k_chain_head, q_iter_local)) {
                            k_chain.forward(noc, cb_k_start_address, k_chunk_tiles, k_tile_bytes);
                        }
                    }

                    // Padded mcast iterations skip Q and compute work. Only multicast rows pad, and sliding mode never
                    // builds chains. (main #60337: padded slots push their K/V entries like real ones and compute
                    // drains them; the former staging-only receive, incl. the sol_pack cb_v_stage variant, is gone.)
                    if (is_padded_iter) {
                        ASSERT(!has_sliding_window);
                        // Push every received entry so this core's CB advances with the injector's; compute drains
                        // padded slots without computing. Push a V entry whenever a real chunk would, received
                        // (GQA multicast) or empty, so both CBs keep the same phase as the real path.
                        cb_k.push_back(k_chunk_tiles);
                        KV_chunks_processed_in_iter++;
                        if constexpr (gqa_grouped_kv && gqa_mcast_enabled) {
                            const uint32_t nv = nq / q_heads_per_v;
                            CircularBuffer cb_v(cb_v_in);
                            cb_v.reserve_back(v_cb_entry_tiles);
                            if (v_chain.should_receive(nv)) {
                                v_chain.receive(noc);
                            }
                            cb_v.push_back(v_cb_entry_tiles);
                        } else if constexpr (!kt_inplace_v) {
                            CircularBuffer cb_v(cb_v_in);
                            cb_v.reserve_back(v_cb_entry_tiles);
                            cb_v.push_back(v_cb_entry_tiles);
                        }
                        continue;
                    }

                    if constexpr (sol_pack) {
                        if (!is_padded_iter) {
                            // Packing: nothing to do here. K stays in staging until the V site below, which packs
                            // both together in one bounded pass -- a page boundary can fall inside a chunk, so
                            // the two cannot be packed independently without the fill overrunning the page.
                            sol_pack_k_src = cb_k_start_address;
                        }
                    } else if constexpr (sol_compact) {
                        if (!is_padded_iter) {
                            // K^T page is [DHt][Sk_chunk_t] tiles: block j = tile columns [2j, 2j+2) of every row.
                            sol_compact_blocks(
                                ring_id,
                                k_chunk,
                                cb_k_start_address,
                                DHt,
                                Sk_chunk_t * k_tile_bytes,
                                kSolBlockTiles * k_tile_bytes);
                        }
                    }
                    // Make K available to compute. Under sol_pack the unit of delivery is the packed page, pushed
                    // above when it filled and after the loop when partial -- a raw chunk is never handed over.
                    //
                    // The per-iteration K/V push count feeds dummy_kv_chunks_for_phase_alignment below, and compute
                    // runs the same function on ITS count. Compute counts pages under sol_pack (raw chunks never
                    // reach it), so the reader must count pages too -- see the trailing flush -- or the two sides
                    // push and pop different numbers of dummy pages and the 2-page CBs desync on the next ring
                    // iteration.
                    if constexpr (!sol_pack) {
                        cb_k.push_back(k_chunk_tiles);
                        KV_chunks_processed_in_iter++;
                    }

                    // Download Q on the first K iteration — after K is downloaded and forwarded.
                    // Push Q one subblock at a time so compute can start QK matmul incrementally.
                    // Placed after K forward so no outstanding NOC writes remain
                    // (noc_async_read_barrier inside subblock read would deadlock with in-flight writes).
                    if (first_k_for_q && need_q_read) {
                        read_q_from_source<has_joint_q, joint_tensor_args_offset>(
                            is_joint_q, joint_q_addr, q_generator, joint_q_input_tile_logical, read_q);
                        q_pushed = true;
                    }
                    first_k_for_q = false;

                    // In-place latent-V (kt_inplace_v) materializes nothing: compute reads V straight
                    // from the K^T already pushed above, so neither branch below runs for that case.
                    if constexpr (v_shares_k_buffer && !kt_inplace_v) {
                        bool skip_v_materialization = false;
                        uint32_t v_rows_to_materialize = Sk_chunk_t;
                        if constexpr (is_causal && !chunked_enabled && !has_sliding_window) {
                            if (ring_iter == 0) {
                                // Local causal chunks beyond this limit are fully masked. Compute still
                                // advances the K/V FIFO phase, but does not consume V values for them.
                                const uint32_t causal_k_limit =
                                    (q_row_start_tile + Sq_chunk_t + Sk_chunk_t - 1) / Sk_chunk_t;
                                skip_v_materialization = k_chunk >= causal_k_limit;
                                if (!skip_v_materialization && k_chunk == causal_k_limit - 1) {
                                    const uint32_t active_rows = q_row_start_tile + Sq_chunk_t - k_chunk * Sk_chunk_t;
                                    if (active_rows < Sk_chunk_t) {
                                        // Compute narrows active_Sk to this same row count, so the unfilled tail
                                        // of the fixed-size V entry is kept for FIFO phase only and is never read.
                                        v_rows_to_materialize = active_rows;
                                    }
                                }
                            }
                        }

                        // Same physical CB as K. K^T is already pushed; reserve a second fixed-size
                        // FIFO entry whose prefix is compact V[Sk, vDHt] via local L1-to-L1 NoC reads.
                        if (skip_v_materialization) {
                            // Preserve the logical V FIFO entry for phase alignment. No fill is needed
                            // because compute skips the fully masked K chunk.
                            CircularBuffer cb_v_skip(cb_v_in);
                            cb_v_skip.reserve_back(v_cb_entry_tiles);
                            cb_v_skip.push_back(v_cb_entry_tiles);
                        } else {
                            materialize_v_prefix_from_k<cb_v_in, v_cb_entry_tiles, Sk_chunk_t, vDHt, k_tile_bytes>(
                                noc, cb_k_start_address, v_rows_to_materialize);
                        }
                    } else if constexpr (!v_shares_k_buffer) {
                        // V: either read locally (injector or not participant) or receive from chain.
                        const uint32_t nv = nq / q_heads_per_v;
                        const Slice v_slice(
                            k_slice.d0, nv, k_slice.d2_start, k_slice.d2_end, v_col_offset_t, v_col_offset_t + vDHt);
                        CircularBuffer cb_v(cb_v_in);
                        uint32_t cb_v_start_address;
                        if constexpr (sol_pack) {
                            CircularBuffer cb_vs(cb_v_stage);  // see the K note: cb_v_in holds the packed page
                            cb_vs.reserve_back(v_cb_entry_tiles);
                            cb_v_start_address = cb_vs.get_write_ptr();
                        } else {
                            cb_v.reserve_back(v_cb_entry_tiles);
                            cb_v_start_address = cb_v.get_write_ptr();
                        }
                        bool received_v_from_chain = false;
                        if constexpr (!has_sliding_window) {
                            if (v_chain.should_receive(nv)) {
                                v_chain.receive(noc);
                                received_v_from_chain = true;
                            }
                        }
                        if (!received_v_from_chain) {
                            const auto fetch_v = [&](const auto& v_gen) {
                                fetch_block(
                                    v_gen,
                                    v_slice,
                                    end_seq_tile,
                                    cb_v_in,
                                    cb_v_start_address,
                                    v_tile_bytes,
                                    false /*transpose*/);
                            };
                            fetch_v_from_source<has_joint_k, has_gathered_joint_k, joint_tensor_args_offset>(
                                kv_chunk_is_joint,
                                joint_chunk_is_local,
                                source_is_local ? 0 : 1,
                                joint_v_addr,
                                gathered_joint_v_addr,
                                v_generators.local,
                                v_generators.gathered,
                                joint_q_input_tile_logical,
                                joint_input_tile_logical,
                                fetch_v);
                        }

                        // Forward V to next core(s) before push_back — prevents compute from
                        // popping the buffer while the mcast is still reading from it.
                        if constexpr (!has_sliding_window) {
                            if (v_chain.should_forward(nv, q_iter_local)) {
                                v_chain.forward(noc, cb_v_start_address);
                            }
                        }

                        if constexpr (sol_pack) {
                            if (!is_padded_iter) {
                                // Pack K (from staging) and V together, flushing whenever a page fills mid-chunk;
                                // this call commits the fill. V page is [Sk_chunk_t][vDHt] tiles: block j = tile
                                // rows [2j, 2j+2), one run. The routine owns the push/reserve so the fill can never
                                // exceed the page.
                                sol_pack_fill = sol_pack_chunk(
                                    ring_id,
                                    k_chunk,
                                    sol_pack_k_src,
                                    cb_v_start_address,
                                    sol_packed_k_addr,
                                    sol_packed_v_addr,
                                    sol_pages_pushed,
                                    sol_pack_fill);
                            }
                        } else if constexpr (sol_compact) {
                            if (!is_padded_iter) {
                                // V page is [Sk_chunk_t][vDHt] tiles: block j = tile rows [2j, 2j+2), one contiguous
                                // run.
                                sol_compact_blocks(
                                    ring_id, k_chunk, cb_v_start_address, 1, 0, kSolBlockTiles * vDHt * v_tile_bytes);
                            }
                        }
                        // Make V available to compute. See the K push above: under sol_pack the packed page is
                        // the unit of delivery, so raw chunks are not pushed.
                        if constexpr (!sol_pack) {
                            cb_v.push_back(v_cb_entry_tiles);
                        }
                    }
                }
                // Flush the partial page. A page cannot span ring iterations (shard `ring_id` is resident only
                // now), so whatever is in it goes out here. The slack slots are NOT zero-filled and must never be
                // computed: compute derives the same fill from the kept bitmask and sets its matmul extent to
                // exactly `fill` blocks. Zero-filling would be worse than useless -- a zero K tile gives QK = 0,
                // hence exp(0) = 1, a full-weight softmax term against a zero V, inflating the denominator.
                if constexpr (sol_pack) {
                    CircularBuffer cb_kp(cb_k_in), cb_vp(cb_v_in);
                    if (!is_padded_iter && sol_pack_fill > 0) {
                        cb_kp.push_back(k_chunk_tiles);
                        cb_vp.push_back(v_cb_entry_tiles);
                        ++sol_pages_pushed;
                        sol_pack_fill = 0;
                    }
                    // Pages are the unit compute counts (one per inner step), so they are what the phase
                    // alignment at the end of the ring iteration must see. See the note at the K push site.
                    KV_chunks_processed_in_iter += sol_pages_pushed;

                    // The trailing reserve_back (from the loop, or the pre-loop one when nothing was kept) is
                    // left un-pushed: the write pointer does not advance, so the next iteration's reserve returns
                    // the same address. Same staging-area idiom the padded-mcast path uses above.
                }
                if constexpr (sol_chunk_mode) {
                    // Nothing was streamed for this (ring iter, Q chunk), so the in-loop Q push never ran. Push Q
                    // here: compute still waits for it (the pooled step on the first active iteration, and the
                    // per-Q-chunk Q pop when q_per_core > 1). Reachable only when the shard holds no full kept
                    // block, which force_block0_if_empty cannot repair (see its ASSERT); the host rejects that
                    // geometry when logical_n is static, and this keeps the dynamic-length path from wedging on Q.
                    if (!is_padded_iter && first_k_for_q && need_q_read) {
                        read_q_from_source<has_joint_q, joint_tensor_args_offset>(
                            is_joint_q, joint_q_addr, q_generator, joint_q_input_tile_logical, read_q);
                        q_pushed = true;
                        first_k_for_q = false;
                    }
                }
                if constexpr (sol_chunk_mode) {
                    if (!is_padded_iter && do_pooled) {
                        sol_push_pooled(nq, q_chunk, KV_chunks_processed_in_iter);
                    }
                }
            }
        }  // sol_enabled / dense q loop
        if constexpr (sol_enabled) {
            sol_pooled_pending = false;
        }
        if (packed_sources) {
            // Cores without Q work still consume the group in sequencer order,
            // so the next group and subsequent dispatches start at the right source.
            packed_readiness.drain(source_group_size, wait_packed_source);
        }
        if constexpr (!has_sliding_window && !sol_res_no_dummies) {
            SOL_ZONE("RD_DUMMY");  // phase-alignment pushes: the reserves wait for compute to drain the K/V CBs
            for (uint32_t dummy_chunk = 0;
                 dummy_chunk < dummy_kv_chunks_for_phase_alignment<v_shares_k_buffer, kt_inplace_v, kv_cb_pages>(
                                   KV_chunks_processed_in_iter);
                 ++dummy_chunk) {
                CircularBuffer cb_k_dummy(cb_k_in);
                cb_k_dummy.reserve_back(k_chunk_tiles);
                cb_k_dummy.push_back(k_chunk_tiles);
                if constexpr (!kt_inplace_v) {
                    CircularBuffer cb_v_dummy(cb_v_in);
                    cb_v_dummy.reserve_back(v_cb_entry_tiles);
                    cb_v_dummy.push_back(v_cb_entry_tiles);
                }
            }
        }
    }
    synchronize_halo();
}
