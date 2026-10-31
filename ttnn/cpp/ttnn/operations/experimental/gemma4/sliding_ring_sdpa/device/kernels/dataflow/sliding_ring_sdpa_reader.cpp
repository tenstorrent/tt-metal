// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/chunked_prefill_utils.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_mla_packing_plan.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_kv_pad_derivation.hpp"
#include "ttnn/operations/experimental/gemma4/sliding_ring_sdpa/device/kernels/sliding_ring_sdpa_derived_slots.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/metadata_scalar_read.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/fused_op_receiver.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_utils.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/sliding_window_work_plan.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_ksplit.hpp"

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
    if (ring_iter == 0) {
        fetch_k(local_k_generator);
    } else {
        fetch_k(gathered_k_generator);
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
    read_q(q_generator);
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
    if (ring_iter == 0) {
        fetch_v(local_v_generator);
    } else {
        fetch_v(gathered_v_generator);
    }
}

template <typename LocalVGenerator, typename GatheredVGenerator>
struct VSourceGenerators {
    LocalVGenerator local;
    GatheredVGenerator gathered;
};

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
    // The host builds this op only for sliding attention, which reads K/V independently on every core: no
    // store-and-forward chains, K/V multicast, rotated Q split, or dense ring all-gather.
    static_assert(has_sliding_window, "Gemma4 sliding ring SDPA requires a sliding window");
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
    // The metadata accessor (metadata path only) follows the tensor accessors and precedes the CB compile
    // args. Gate its offset on slot_from_metadata: when absent, fall back to a VALID (unused) accessor
    // offset (q_args' slot) so TensorAccessorArgs<> -- instantiated unconditionally here -- never names a
    // non-accessor compile arg (which would fail its internal static_assert). The CB compile args then
    // start after the metadata accessor when present.
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
    constexpr uint32_t post_slot_args_offset =
        slot_from_metadata ? meta_args.next_compile_time_args_offset() : post_tensor_args_offset;
    constexpr uint32_t post_meta_args_offset =
        kv_pad_from_metadata ? kv_valid_end_args.next_compile_time_args_offset() : post_slot_args_offset;
    // logical_n / logical_l accessors follow the metadata accessors, appended as a pair so these offsets do
    // not depend on which one was supplied. Same VALID-fallback trick as meta_args above.
    constexpr uint32_t logical_n_args_offset =
        has_logical_length_tensor ? post_meta_args_offset : kFirstAccessorArgOffset;
    constexpr auto logical_n_args = TensorAccessorArgs<logical_n_args_offset>();
    constexpr uint32_t logical_l_args_offset =
        has_logical_length_tensor ? logical_n_args.next_compile_time_args_offset() : logical_n_args_offset;
    constexpr auto logical_l_args = TensorAccessorArgs<logical_l_args_offset>();
    // Sliding builds no KV chains, so no chain semaphore compile args precede the CB block.
    constexpr uint32_t cb_arg_offset =
        has_logical_length_tensor ? logical_l_args.next_compile_time_args_offset() : post_meta_args_offset;

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

    uint32_t logical_nt = get_arg_val<uint32_t>(argidx++);
    uint32_t active_ring_iter_mask = get_arg_val<uint32_t>(argidx++);
    ring_joint::ChunkedQMapping qmap{
        get_arg_val<uint32_t>(argidx++),
        get_arg_val<uint32_t>(argidx++),
        get_arg_val<uint32_t>(argidx++),
        get_arg_val<uint32_t>(argidx++)};
    RingSDPAOpReceiver fused_op_receiver = RingSDPAOpReceiver(
        true, /* wait_for_op_signal */
        argidx);

    // The K split's band index follows the fused-op signaler block.
    const uint32_t ksplit_arg_index = argidx;

    constexpr uint32_t cb_q_in = get_compile_time_arg_val(cb_arg_offset + 0);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(cb_arg_offset + 1);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(cb_arg_offset + 2);
    constexpr uint32_t cb_attention_sink = get_compile_time_arg_val(cb_arg_offset + 3);
    constexpr uint32_t cb_kv_pad_derived = get_compile_time_arg_val(cb_arg_offset + 4);

    // The factory keeps the shared op's rotated Q split slot as the final compile-time arg of every kernel.
    // Sliding never rotates Q, so it is always 0.
    constexpr uint32_t rotated_max_slots = get_ct_arg<kernel_compile_time_args.size() - 1>();
    static_assert(rotated_max_slots == 0, "Sliding attention has no rotated Q split");
    constexpr uint32_t ksplit_count = get_named_compile_time_arg_val("ksplit_count");
    static_assert(
        get_named_compile_time_arg_val("dense_causal_skip") == 0, "Sliding attention has no dense causal K skip");
    constexpr bool ksplit_enabled = ksplit_count > 1;
    [[maybe_unused]] const uint32_t ksplit_idx = ksplit_enabled ? get_arg_val<uint32_t>(ksplit_arg_index) : 0;

    // Common runtime args: metadata block first when present, then the logical-length pair.
    constexpr uint32_t logical_length_common_arg_base =
        slot_from_metadata ? ring_joint::kReaderMetadataCommonArgCount : 0;
    // kv_valid_end is validated incompatible with the logical_n / logical_l tensors, so its arg directly
    // follows the metadata block.
    static_assert(!has_kv_valid_end || !has_logical_length_tensor);
    [[maybe_unused]] constexpr uint32_t kv_valid_end_common_arg = logical_length_common_arg_base;

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
        if constexpr (kv_pad_from_metadata) {
            uint32_t kv_actual_isl = trace_metadata::read_metadata_scalar_u32(
                meta_noc, kv_meta_args, get_common_arg_val<uint32_t>(4), meta_l1);
            kv_actual_isl = trace_metadata::bounded_sliding_kv_actual_isl(
                kv_actual_isl, q_local_padded_Nt, ring_size, kv_local_padded_Nt, SLIDING_HALO_SLOT_COUNT);
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
            cb_derived.push_back(1);
        }
    }

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);

    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;
    constexpr uint32_t v_cb_entry_tiles = v_shares_k_buffer ? k_chunk_tiles : v_chunk_tiles;

    Noc noc;

    constexpr uint32_t q_num_subblocks = Sq_chunk_t / qk_subblock_h;
    constexpr bool use_q_subblock_push = (q_num_subblocks > 1);
    constexpr uint32_t q_heads_per_k = NH / NHK;
    static_assert(chunked_enabled, "Sliding windows require chunked prefill");

    // Throttle Q DRAM reads so many readers don't saturate the NoC outstanding-read budget.
    constexpr uint32_t q_barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_q_readers>();

    const auto q_reader = TensorAccessor(q_args, q_addr);
    const auto local_k_reader = TensorAccessor(k_args, k_addr);
    const auto gathered_k_reader = TensorAccessor(gathered_k_args, gathered_k_addr);

    const uint32_t kv_batch_dim = indexed_kv_cache ? kv_cache_batch_idx + 1 : B;
    // The halo exchange wrote the active slot to gathered slot 0, so address it as batch-1.
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

    // Tracks whether Q has been pushed for q_per_core == 1 optimization.
    // When q_per_core == 1, Q is identical across ring iterations so we only push it once.
    bool q_pushed = false;

    /**
     * Sliding consumes the local slab and its predecessor halo in one synthetic ring iteration.
     * Local K/V chunks come from the local cache; halo chunks from the compact gathered buffer,
     * after the halo exchange signals its arrival.
     */
    const uint32_t ring_index =
        ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
            fused_op_receiver.seq.ring_index, mesh_rows, mesh_cols, snake_orientation);
    // Sliding consumes local and halo ranges in one logical Q pass. Without a K split the plan puts the local K chunks
    // first, so the halo is waited for only before the first halo chunk; a core that reads none waits at the end, so a
    // cached program cannot observe the previous invocation's token.
    bool halo_synchronized = false;
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
    auto arrival_preview = fused_op_receiver.seq;
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
    const uint32_t sdpa_ring_iterations = packed_schedule.ring_iterations;
    for (uint32_t ring_iter = 0; ring_iter < sdpa_ring_iterations; ++ring_iter) {
        for (uint32_t q_iter = 0; q_iter < q_per_core; ++q_iter) {
            const uint32_t flat_q_index = global_q_start + q_iter;
            const auto decoded_q = decompose_global_q_index(flat_q_index, num_q_chunks, NH, use_zigzag_balancing);
            const uint32_t nb = decoded_q.nb;
            const uint32_t nq = decoded_q.nq;
            const uint32_t q_chunk = decoded_q.q_chunk;
            const uint32_t nk = nq / q_heads_per_k;
            const auto q_row_start_tile = q_chunk * Sq_chunk_t;
            const bool is_joint_q = has_joint_q ? (q_chunk >= num_local_q_chunks) : false;

            Slice q_slice(nb, nq, q_row_start_tile, q_row_start_tile + Sq_chunk_t, 0, DHt);
            uint32_t q_end_seq_tile = q_local_padded_Nt;

            // When q_per_core == 1, Q is identical across ring iterations: compute keeps it
            // fronted in the CB, so we only need to read it once on the first ring iteration.
            const bool need_q_read = (q_per_core > 1) || !q_pushed;

            const ring_joint::SlidingQWorkPlan<SLIDING_MAX_SOURCE_RANGES> sliding_q_plan =
                ring_joint::build_sliding_q_work_plan<SLIDING_MAX_SOURCE_RANGES>(
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
            // Sliding K split: this core's band of the work plan (ring_joint_ksplit.hpp).
            const auto sliding_band =
                ring_joint::sliding_ksplit_range(sliding_q_plan.total_k_chunk_count, ksplit_idx, ksplit_count);
            // Q must be pushed on the first K chunk actually PROCESSED for this Q chunk, which is the
            // band's first chunk under a K split. Reads Q exactly once per q_iter, so no extra work.
            bool first_k_for_q = true;
            for (uint32_t k_chunk = sliding_band.begin; k_chunk < sliding_band.end; ++k_chunk) {
                const auto sliding_k_chunk = sliding_q_plan.k_chunk_at(k_chunk);
                const uint32_t source_ring_id = sliding_k_chunk.source_ring_id;
                const uint32_t source_k_chunk = sliding_k_chunk.source_k_chunk;
                /**
                 * Iterate over this Q chunk's K chunks from the sliding work plan, already clipped to
                 * logical_n. Joint K/V is never present with a sliding window.
                 */
                const bool kv_chunk_is_joint = false;

                // Default to local/gathered KV; override below for joint KV when applicable.
                Slice k_slice;
                uint32_t end_seq_tile;
                // Local KV reads the indexed cache slot; gathered KV is at slot 0 of the scratch buffer.
                const uint32_t kv_batch = indexed_kv_cache ? kv_cache_batch_idx : nb;
                const uint32_t gathered_kv_batch = indexed_kv_cache ? 0 : nb;
                const bool source_is_local = source_ring_id == ring_index;
                if (source_is_local) {
                    const uint32_t local_k_start_tile = source_k_chunk * Sk_chunk_t;
                    k_slice = Slice(kv_batch, nk, local_k_start_tile, local_k_start_tile + Sk_chunk_t, 0, DHt);
                    end_seq_tile = kv_local_padded_Nt;
                } else {
                    synchronize_halo();
                    const uint32_t gathered_start_tile = sliding_k_chunk.compact_k_chunk * Sk_chunk_t;
                    k_slice =
                        Slice(gathered_kv_batch, nk, gathered_start_tile, gathered_start_tile + Sk_chunk_t, 0, DHt);
                    end_seq_tile = gathered_padded_Nt;
                }
                const bool joint_chunk_is_local = false;

                CircularBuffer cb_k(cb_k_in);
                cb_k.reserve_back(k_chunk_tiles);
                uint32_t cb_k_start_address = cb_k.get_write_ptr();
                // Read K from DRAM. Dispatch directly so local and gathered tensors may use
                // different accessor types.
                const auto fetch_k = [&](const auto& k_gen) {
                    fetch_block(
                        k_gen, k_slice, end_seq_tile, cb_k_in, cb_k_start_address, k_tile_bytes, true /*transpose*/);
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

                // Make K available to compute.
                cb_k.push_back(k_chunk_tiles);

                // Download Q on the first K iteration, after K is downloaded.
                // Push Q one subblock at a time so compute can start QK matmul incrementally.
                if (first_k_for_q && need_q_read) {
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
                    read_q_from_source<has_joint_q, joint_tensor_args_offset>(
                        is_joint_q, joint_q_addr, q_generator, joint_q_input_tile_logical, read_q);
                    q_pushed = true;
                }
                first_k_for_q = false;

                if constexpr (!v_shares_k_buffer) {
                    // V: read from the same local or gathered source as K.
                    const uint32_t nv = nq / q_heads_per_v;
                    const Slice v_slice(
                        k_slice.d0, nv, k_slice.d2_start, k_slice.d2_end, v_col_offset_t, v_col_offset_t + vDHt);
                    CircularBuffer cb_v(cb_v_in);
                    cb_v.reserve_back(v_cb_entry_tiles);
                    uint32_t cb_v_start_address = cb_v.get_write_ptr();
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

                    // Make V available to compute.
                    cb_v.push_back(v_cb_entry_tiles);
                }
            }
        }
    }
    synchronize_halo();
}
