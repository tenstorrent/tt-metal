// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "dataflow_common.hpp"
#include "chunked_prefill_utils.hpp"
#include "chain_link.hpp"
#include "fused_op_receiver.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_chain_layout.hpp"

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
    constexpr uint32_t padded_Nt = get_compile_time_arg_val(9);
    // Slot 10: reader-unused (writer/compute consume it for constexpr mask-CB sizing).
    constexpr uint32_t logical_n [[maybe_unused]] = get_compile_time_arg_val(10);
    // Slot 11 is retained for compile-time arg index stability; live logical_nt is a runtime arg below.
    constexpr uint32_t logical_nt_compile [[maybe_unused]] = get_compile_time_arg_val(11);
    constexpr uint32_t Lt = get_compile_time_arg_val(12);
    constexpr uint32_t L = get_compile_time_arg_val(13);
    constexpr uint32_t num_local_q_chunks = get_compile_time_arg_val(14);
    constexpr uint32_t num_joint_q_chunks = get_compile_time_arg_val(15);
    constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(16);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(17);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(18);
    constexpr uint32_t ring_size = get_compile_time_arg_val(19);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(20);
    constexpr uint32_t is_causal = get_compile_time_arg_val(21);
    constexpr uint32_t is_balanced = get_compile_time_arg_val(22);
    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(23) == 1;
    // Reader's slot-24 carries chunked_enabled.
    constexpr bool chunked_enabled = get_compile_time_arg_val(24) == 1;
    constexpr uint32_t num_q_readers = get_compile_time_arg_val(25);
    constexpr uint32_t chunk_size_t = get_compile_time_arg_val(26);
    constexpr bool indexed_kv_cache = get_compile_time_arg_val(27) == 1;
    constexpr bool kv_pad_rotation_enabled = get_compile_time_arg_val(28) == 1;
    // Slot 29 is retained for compile-time arg index stability; live active-ring mask is a runtime arg below.
    constexpr uint32_t active_ring_iter_mask_compile [[maybe_unused]] = get_compile_time_arg_val(29);
    constexpr uint32_t NHV = get_compile_time_arg_val(30);
    // Latent-V mode: absent V is materialized from the prefix of K tiles already in L1.
    constexpr bool v_shares_k_buffer = get_compile_time_arg_val(31) == 1;
    constexpr bool gqa_grouped_kv = ring_joint::is_gqa_grouped_kv_head_mode(v_shares_k_buffer, NH, NHK, NHV);
    constexpr bool k_uses_batch_chain = ring_joint::uses_shared_k_batch_chain(gqa_grouped_kv, NHK);
    // In-place latent-V (single-tile Q): the compute kernel reads V straight from K^T, so the
    // reader never materializes V. Shared with the program factory and compute kernel.
    constexpr bool kt_inplace_v = kt_inplace_v_enabled(v_shares_k_buffer, Sq_chunk_t);
    constexpr uint32_t q_heads_per_v = NH / NHV;
    // Slots 32-33: sharded-joint scalars appended by the factory after the existing scalar block.
    // Lt_local: per-device joint-Q tile count (Lt/ring_size on sharded path, Lt on replicated).
    constexpr uint32_t Lt_local = get_compile_time_arg_val(32);
    constexpr bool joint_is_sharded = get_compile_time_arg_val(33) == 1;

    // Joint-path compile-time gating. When zero, joint Q/K branches are statically dead
    // and dropped by the compiler, eliminating runtime ternaries and joint generator uses.
    constexpr bool has_joint_q = num_joint_q_chunks > 0;
    constexpr bool has_joint_k = num_joint_k_chunks > 0;
    constexpr bool has_joint_inputs = has_joint_q || has_joint_k;
    // Sharded joint requires the gathered joint K/V buffers (only meaningful when joint K is present).
    constexpr bool has_gathered_joint_k = joint_is_sharded && has_joint_k;

    constexpr auto q_args = TensorAccessorArgs<34>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto gathered_k_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto gathered_v_args = TensorAccessorArgs<gathered_k_args.next_compile_time_args_offset()>();
    constexpr uint32_t joint_tensor_args_offset = gathered_v_args.next_compile_time_args_offset();
    constexpr uint32_t post_tensor_args_offset =
        get_post_tensor_args_offset<has_joint_inputs, has_gathered_joint_k, joint_tensor_args_offset>();

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
    // Gathered joint K/V buffer addresses (sharded path only) — pushed by the factory right after the
    // local joint addresses, so they must be read here before global_q_start to keep argidx aligned.
    uint32_t gathered_joint_k_addr = 0;
    uint32_t gathered_joint_v_addr = 0;
    if constexpr (has_gathered_joint_k) {
        gathered_joint_k_addr = get_arg_val<uint32_t>(argidx++);
        gathered_joint_v_addr = get_arg_val<uint32_t>(argidx++);
    }
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t kv_cache_batch_idx = get_arg_val<uint32_t>(argidx++);
    const uint32_t q_per_core = global_q_end - global_q_start;

    // Head chain runtime args (always present)
    const ChainConfig head_cfg = ChainConfig::read_from_args(argidx);

    // Batch chain runtime args (only present for non-GQA shared-K modes).
    ChainConfig batch_cfg;  // default zero-initialized
    uint32_t max_q_per_core = 0;
    if constexpr (k_uses_batch_chain) {
        batch_cfg = ChainConfig::read_from_args(argidx);
        max_q_per_core = get_arg_val<uint32_t>(argidx++);
    }
    ChainConfig gqa_cfg;  // default zero-initialized
    uint32_t gqa_max_q_per_core = 0;
    if constexpr (gqa_grouped_kv) {
        gqa_cfg = ChainConfig::read_from_args(argidx);
        gqa_max_q_per_core = get_arg_val<uint32_t>(argidx++);
    }

    const uint32_t logical_nt = get_arg_val<uint32_t>(argidx++);
    const uint32_t active_ring_iter_mask = get_arg_val<uint32_t>(argidx++);
    RingSDPAOpReceiver fused_op_receiver = RingSDPAOpReceiver(
        true, /* wait_for_op_signal */
        argidx);

    // Compile-time semaphore ids and chain flags are appended after all TensorAccessorArgs().
    // ChainLink takes semaphore IDs directly (the new Semaphore<> wrapper resolves them to L1 addrs).
    constexpr uint32_t chain_sender_semaphore_arg_offset = ring_joint::kChainSenderSemaphoreCompileArgOffset;
    constexpr uint32_t chain_receiver_semaphore_arg_offset = ring_joint::kChainReceiverSemaphoreCompileArgOffset;
    constexpr uint32_t chain_valid_semaphore_arg_offset = ring_joint::kChainValidSemaphoreCompileArgOffset;
    constexpr uint32_t chain_mcast_enabled_arg_offset = ring_joint::kChainMcastEnabledCompileArgOffset;
    constexpr uint32_t chain_compile_arg_count = ring_joint::kChainCompileArgCount;

    uint32_t head_sender_semaphore_id =
        get_compile_time_arg_val(post_tensor_args_offset + chain_sender_semaphore_arg_offset);
    uint32_t head_receiver_semaphore_id =
        get_compile_time_arg_val(post_tensor_args_offset + chain_receiver_semaphore_arg_offset);
    uint32_t head_valid_semaphore_id =
        get_compile_time_arg_val(post_tensor_args_offset + chain_valid_semaphore_arg_offset);
    constexpr bool head_mcast_enabled =
        get_compile_time_arg_val(post_tensor_args_offset + chain_mcast_enabled_arg_offset) == 1;
    constexpr uint32_t head_chain_arg_count = chain_compile_arg_count;
    constexpr uint32_t batch_chain_arg_count = k_uses_batch_chain ? chain_compile_arg_count : 0;
    constexpr uint32_t gqa_chain_arg_count = gqa_grouped_kv ? chain_compile_arg_count : 0;
    constexpr uint32_t batch_chain_ct_offset = post_tensor_args_offset + head_chain_arg_count;
    constexpr uint32_t gqa_chain_ct_offset = batch_chain_ct_offset + batch_chain_arg_count;

    // Batch chain semaphores (only present for non-GQA shared-K modes).
    // Initialize to 0; will be overwritten if k_uses_batch_chain. Non-participants never use them.
    uint32_t batch_sender_semaphore_id = 0;
    uint32_t batch_receiver_semaphore_id = 0;
    uint32_t batch_valid_semaphore_id = 0;

    // batch_mcast_enabled: read from compile-time args if present, else false (for template instantiation)
    constexpr bool batch_mcast_enabled = []() {
        if constexpr (k_uses_batch_chain) {
            return get_compile_time_arg_val(batch_chain_ct_offset + chain_mcast_enabled_arg_offset) == 1;
        }
        return false;
    }();

    if constexpr (k_uses_batch_chain) {
        batch_sender_semaphore_id = get_compile_time_arg_val(batch_chain_ct_offset + chain_sender_semaphore_arg_offset);
        batch_receiver_semaphore_id =
            get_compile_time_arg_val(batch_chain_ct_offset + chain_receiver_semaphore_arg_offset);
        batch_valid_semaphore_id = get_compile_time_arg_val(batch_chain_ct_offset + chain_valid_semaphore_arg_offset);
    }

    uint32_t gqa_sender_semaphore_id = 0;
    uint32_t gqa_receiver_semaphore_id = 0;
    uint32_t gqa_valid_semaphore_id = 0;
    constexpr bool gqa_mcast_enabled = []() {
        if constexpr (gqa_grouped_kv) {
            return get_compile_time_arg_val(gqa_chain_ct_offset + chain_mcast_enabled_arg_offset) == 1;
        }
        return false;
    }();
    if constexpr (gqa_grouped_kv) {
        gqa_sender_semaphore_id = get_compile_time_arg_val(gqa_chain_ct_offset + chain_sender_semaphore_arg_offset);
        gqa_receiver_semaphore_id = get_compile_time_arg_val(gqa_chain_ct_offset + chain_receiver_semaphore_arg_offset);
        gqa_valid_semaphore_id = get_compile_time_arg_val(gqa_chain_ct_offset + chain_valid_semaphore_arg_offset);
    }

    constexpr uint32_t cb_arg_offset =
        post_tensor_args_offset + head_chain_arg_count + batch_chain_arg_count + gqa_chain_arg_count;
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(cb_arg_offset + 0);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(cb_arg_offset + 1);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(cb_arg_offset + 2);

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);

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
        head_cfg.mcast_sender_wait,
        v_chunk_tiles,
        v_tile_bytes,
        head_cfg.batch,
        head_cfg.head,
        head_cfg.next_core_q_chunks);

    // Batch chain (batch-level): matches batch only, used by K in non-GQA shared-K modes.
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
        batch_cfg.mcast_sender_wait,
        k_chunk_tiles,
        k_tile_bytes,
        batch_cfg.batch,
        0,  // chain_head unused for batch-level chain
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
        gqa_cfg.mcast_sender_wait,
        v_chunk_tiles,
        v_tile_bytes,
        gqa_cfg.batch,
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

    // K uses the grouped GQA chain, the batch chain for shared-K modes, otherwise the query-head chain.
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

    // Throttle Q DRAM reads so many readers don't saturate the NoC outstanding-read budget.
    constexpr uint32_t q_barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_q_readers>();

    const auto q_reader = TensorAccessor(q_args, q_addr);
    const auto local_k_reader = TensorAccessor(k_args, k_addr);
    const auto gathered_k_reader = TensorAccessor(gathered_k_args, gathered_k_addr);

    const uint32_t kv_batch_dim = indexed_kv_cache ? kv_cache_batch_idx + 1 : B;
    // The fused all-gather wrote the active slot to gathered slot 0, so address it as batch-1.
    const uint32_t gathered_kv_batch_dim = indexed_kv_cache ? 1 : B;
    const auto input_q_tile_logical = TensorTileShape(B, NH, q_local_padded_Nt, DHt);
    const auto input_k_tile_logical = TensorTileShape(kv_batch_dim, NHK, kv_local_padded_Nt, DHt);
    const auto gathered_k_input_tile_logical = TensorTileShape(gathered_kv_batch_dim, NHK, padded_Nt, DHt);
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
    const auto input_v_tile_logical = TensorTileShape(kv_batch_dim, NHV, kv_local_padded_Nt, vDHt);
    const auto gathered_v_reader = TensorAccessor(gathered_v_args, gathered_v_addr);
    const auto gathered_v_input_tile_logical = TensorTileShape(gathered_kv_batch_dim, NHV, padded_Nt, vDHt);
    const auto local_v_generator = PaddedAddrGenerator(local_v_reader, input_v_tile_logical);
    const auto gathered_v_generator = PaddedAddrGenerator(gathered_v_reader, gathered_v_input_tile_logical);
    [[maybe_unused]] const auto v_generators =
        VSourceGenerators<decltype(local_v_generator), decltype(gathered_v_generator)>{
            local_v_generator, gathered_v_generator};

    // Tracks whether Q has been pushed for q_per_core == 1 optimization.
    // When q_per_core == 1, Q is identical across ring iterations so we only push it once.
    bool q_pushed = false;

    /**
     * Iterate over ring indices.
     * On the first iteration, read from local K, V.
     * On subsequent iterations, read from gathered K, V. Sync with AllGather fused signaler.
     */
    uint32_t ring_index = fused_op_receiver.seq.ring_index;
    uint32_t half_sequence = num_q_chunks / 2;
    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        // find out which is the latest ring_id that synchronized
        uint32_t ring_id = fused_op_receiver.get_next_ring_id_and_sync();
        // Host precomputes which ring iterations have useful SDPA work; sync/ring-id sequencing
        // still advances above so reader stays aligned with compute, writer, and all-gather.
        if (((active_ring_iter_mask >> ring_iter) & 1u) == 0) {
            continue;
        }
        // Sharded joint: the fused all-gather delivers one remote L/P shard per ring iteration (local
        // slice is read from the local joint tensor, same as spatial). Each shard is available right
        // after its ring_id sync, so we process joint K/V on EVERY ring iteration — no need to batch
        // at the end.
        // Replicated joint: gathered K/V is only complete at the last active iteration (processing it
        // earlier would read a partially-filled buffer and deadlock). See original ring_iter vs ring_id note.
        const bool do_joint_kv =
            has_gathered_joint_k ? true : is_last_active_ring_iter(active_ring_iter_mask, ring_iter);
        uint32_t num_kv_chunks = num_local_k_chunks;
        if constexpr (has_joint_k) {
            if (do_joint_kv) {
                num_kv_chunks += num_joint_k_chunks;
            }
        }

        uint32_t ring_iter_valid_kv_tiles = kv_local_padded_Nt;
        if constexpr (!chunked_enabled) {
            const uint32_t ring_iter_kv_start_tile = ring_id * kv_local_padded_Nt;
            ring_iter_valid_kv_tiles = 0;
            if (ring_iter_kv_start_tile < logical_nt) {
                const uint32_t remaining_kv_tiles = logical_nt - ring_iter_kv_start_tile;
                ring_iter_valid_kv_tiles =
                    remaining_kv_tiles < kv_local_padded_Nt ? remaining_kv_tiles : kv_local_padded_Nt;
            }
        }

        uint32_t KV_chunks_processed_in_iter = 0;
        uint32_t iter_num_kv_chunks = num_kv_chunks;

        // In causal balanced case processing KV received from other devices:
        //
        // We will have two logical chunks of the input sequence, logical indexes are:
        // ring_index and (seq_len / 2 * num_devices) - ring_index
        //
        // With this in mind we have two distinct cases when receiving from other device:
        // - 1st part of the sequence precedes both chunks on the sender device, 2nd part attends to both
        // - both chunks preced 2nd part of the sequence in received KV
        // Indexes are updated accordingly; compute is skipped
        if (is_causal && is_balanced && ring_index > ring_id) {
            iter_num_kv_chunks /= 2;
            // Mirror compute's K-loop extension: include the straddle chunk so K/V tiles
            // for it get loaded. Compute -inf-masks its late-half columns via lw_mask.
            using Straddle = KCausalStraddleInfo<kv_local_padded_Nt, Sk_chunk_t>;
            if constexpr (Straddle::has_straddle) {
                iter_num_kv_chunks = Straddle::straddle_chunk_id + 1;
            }
        }

        // When K/V mcast is enabled, loop the per-chain max so receivers with less real Q work
        // still participate in padded multicast handshakes without pushing compute-visible data.
        uint32_t loop_q_count = q_per_core;
        if constexpr (k_uses_batch_chain && batch_mcast_enabled) {
            loop_q_count = max_q_per_core;
        }
        if constexpr (gqa_grouped_kv && gqa_mcast_enabled) {
            loop_q_count = gqa_max_q_per_core;
        }
        uint32_t gqa_group_q_iter = 0;

        for (uint32_t q_iter = 0; q_iter < loop_q_count; ++q_iter) {
            // Check if this is a real iteration or only padded chain/mcast synchronization.
            const bool is_padded_iter = (q_iter >= q_per_core);

            // Calculate global_q_chunk for all iterations (including padded).
            // For padded iterations, global index may be out of bounds, but q_chunk = global_q_chunk % num_q_chunks
            // gives a valid position that correctly determines whether to skip this iteration.
            uint32_t global_q_chunk = remap_q_index(global_q_start + q_iter, num_q_chunks, use_zigzag_balancing);

            // global_q_chunk is index into `B * NH * num_q_chunks`. Need to get nb, nq, q_chunk from this.
            const uint32_t nb = global_q_chunk / (NH * num_q_chunks);
            const uint32_t nq = (global_q_chunk % (NH * num_q_chunks)) / num_q_chunks;
            const uint32_t q_chunk = global_q_chunk % num_q_chunks;
            const uint32_t nk = nq / q_heads_per_k;
            const auto q_row_start_tile = q_chunk * Sq_chunk_t;
            const bool is_joint_q = has_joint_q ? (q_chunk >= num_local_q_chunks) : false;
            const uint32_t q_iter_local = [&]() {
                if constexpr (gqa_grouped_kv) {
                    return gqa_group_q_iter;
                } else {
                    return q_iter;
                }
            }();
            if constexpr (gqa_grouped_kv) {
                if (nb == gqa_cfg.batch && nk == gqa_cfg.head) {
                    gqa_group_q_iter++;
                }
            }

            const bool balanced_skip_q = q_chunk < half_sequence && is_balanced && ring_index < ring_id;

            // Balanced causal skip: this Q chunk is handled by the paired device. Reader sends
            // nothing (no Q, no K/V) — compute's normalize-only path on the last ring iter does
            // not read Q (normalize uses only restored sum/out).
            // Skip logic applies to all iterations (including padded) so injector and receivers
            // make the same skip decisions, keeping chain/mcast sync aligned.
            if (balanced_skip_q) {
                continue;
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
            const bool need_q_read = (q_per_core > 1) || !q_pushed;

            // Q must be pushed on the first K chunk actually PROCESSED for this Q chunk — not a
            // hardcoded k_chunk == 0. When spatial k_chunk 0 is beyond logical_n it is skipped, so
            // anchoring Q to k_chunk == 0 would never push Q while compute still waits on it
            // (q_per_core > 1) -> deadlock. Reads Q exactly once per q_iter, so no extra work.
            bool q_read_done_this_qiter = false;

            for (uint32_t k_chunk = 0; k_chunk < iter_num_kv_chunks; ++k_chunk) {
                /**
                 * Iterate over all KV chunks for this Q chunk.
                 * If this is the last ring ID, we will also read from joint KV.
                 * If this k chunk is in the spatial input and beyond the logical N, we will skip it.
                 */
                const bool kv_chunk_is_joint = has_joint_k ? (k_chunk >= num_local_k_chunks) : false;
                const bool kv_chunk_is_beyond_logical_n =
                    !kv_chunk_is_joint && !kv_chunk_starts_before_logical_end<
                                              kv_pad_rotation_enabled,
                                              chunked_enabled,
                                              kv_local_padded_Nt,
                                              chunk_size_t,
                                              q_local_padded_Nt>(ring_id, k_chunk * Sk_chunk_t, logical_nt);

                if (kv_chunk_is_beyond_logical_n) {
                    // This is a KV chunk on spatial input beyond the logical N, and not joint KV. Skip it.
                    continue;
                }

                // Default to local/gathered KV; override below for joint KV when applicable.
                Slice k_slice;
                uint32_t end_seq_tile;
                // Local KV reads the indexed cache slot; gathered KV is at slot 0 of the scratch buffer.
                const uint32_t kv_batch = indexed_kv_cache ? kv_cache_batch_idx : nb;
                const uint32_t gathered_kv_batch = indexed_kv_cache ? 0 : nb;
                if (ring_iter == 0) {
                    const uint32_t local_k_start_tile = k_chunk * Sk_chunk_t;
                    k_slice = Slice(kv_batch, nk, local_k_start_tile, local_k_start_tile + Sk_chunk_t, 0, DHt);
                    end_seq_tile = ring_iter_valid_kv_tiles;
                } else {
                    const uint32_t gathered_start_tile = ring_id * kv_local_padded_Nt + k_chunk * Sk_chunk_t;
                    k_slice =
                        Slice(gathered_kv_batch, nk, gathered_start_tile, gathered_start_tile + Sk_chunk_t, 0, DHt);
                    end_seq_tile = ring_id * kv_local_padded_Nt + ring_iter_valid_kv_tiles;
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
                                k_slice = Slice(nb, nk, joint_chunk_offset, joint_chunk_offset + Sk_chunk_t, 0, DHt);
                                end_seq_tile = Lt_local;
                            } else {
                                // Remote slice: k_slice in gathered buffer frame.
                                const uint32_t joint_k_row_start_tile = ring_id * Lt_local + joint_chunk_offset;
                                k_slice =
                                    Slice(nb, nk, joint_k_row_start_tile, joint_k_row_start_tile + Sk_chunk_t, 0, DHt);
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

                // K: either read locally (injector or not participant) or receive from chain
                const uint32_t k_chain_head = [&]() {
                    if constexpr (gqa_grouped_kv) {
                        return nk;
                    } else {
                        return nq;
                    }
                }();
                CircularBuffer cb_k(cb_k_in);
                if constexpr ((k_uses_batch_chain && batch_mcast_enabled) || (gqa_grouped_kv && gqa_mcast_enabled)) {
                    // Ensures that compute has completed with the previous K chunk before we overwrite the buffer with
                    // the next K chunk for mcast.
                    const uint32_t reserve_tiles = is_padded_iter ? 2 * k_chunk_tiles : k_chunk_tiles;
                    cb_k.reserve_back(reserve_tiles);
                } else {
                    cb_k.reserve_back(k_chunk_tiles);
                }
                uint32_t cb_k_start_address = cb_k.get_write_ptr();
                if (k_chain.should_receive(nb, k_chain_head)) {
                    k_chain.receive(noc);
                } else {
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
                        ring_iter,
                        joint_k_addr,
                        gathered_joint_k_addr,
                        local_k_generator,
                        gathered_k_generator,
                        joint_q_input_tile_logical,
                        joint_input_tile_logical,
                        fetch_k);
                }

                // Forward K chunk via chain (uses K's data size explicitly)
                if (k_chain.should_forward(nb, k_chain_head, q_iter_local)) {
                    k_chain.forward(noc, cb_k_start_address, k_chunk_tiles, k_tile_bytes);
                }

                // Skip Q and compute-visible pushes for padded mcast iterations.
                // Note: push_back is intentionally skipped — without it, the write pointer
                // doesn't advance, so reserve_back returns the same address each iteration.
                // This lets the buffer act as a reusable staging area for the mcast.
                if (is_padded_iter) {
                    // Padded GQA receivers must also receive V, because the row injector multicasts
                    // both K and V and waits for every receiver's ready signal. The data remains
                    // staging-only because we intentionally do not push it to compute.
                    if constexpr (gqa_grouped_kv && gqa_mcast_enabled) {
                        const uint32_t nv = nq / q_heads_per_v;
                        CircularBuffer cb_v(cb_v_in);
                        cb_v.reserve_back(2 * v_cb_entry_tiles);
                        if (v_chain.should_receive(nb, nv)) {
                            v_chain.receive(noc);
                        }
                    }
                    continue;
                }

                // Make K available to compute.
                cb_k.push_back(k_chunk_tiles);
                KV_chunks_processed_in_iter++;

                // Download Q on the first K iteration — after K is downloaded and forwarded.
                // Push Q one subblock at a time so compute can start QK matmul incrementally.
                // Placed after K forward so no outstanding NOC writes remain
                // (noc_async_read_barrier inside subblock read would deadlock with in-flight writes).
                if (need_q_read && !q_read_done_this_qiter) {
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
                    q_read_done_this_qiter = true;
                }

                // In-place latent-V (kt_inplace_v) materializes nothing: compute reads V straight
                // from the K^T already pushed above, so neither branch below runs for that case.
                if constexpr (v_shares_k_buffer && !kt_inplace_v) {
                    bool skip_v_materialization = false;
                    uint32_t v_rows_to_materialize = Sk_chunk_t;
                    if constexpr (is_causal && !chunked_enabled) {
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
                    const Slice v_slice(k_slice.d0, nv, k_slice.d2_start, k_slice.d2_end, 0, vDHt);
                    CircularBuffer cb_v(cb_v_in);
                    cb_v.reserve_back(v_cb_entry_tiles);
                    uint32_t cb_v_start_address = cb_v.get_write_ptr();
                    if (v_chain.should_receive(nb, nv)) {
                        v_chain.receive(noc);
                    } else {
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
                            ring_iter,
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
                    if (v_chain.should_forward(nb, nv, q_iter_local)) {
                        v_chain.forward(noc, cb_v_start_address);
                    }

                    // Make V available to compute.
                    cb_v.push_back(v_cb_entry_tiles);
                }
            }
        }
        for (uint32_t dummy_chunk = 0;
             dummy_chunk <
             dummy_kv_chunks_for_phase_alignment<v_shares_k_buffer, kt_inplace_v>(KV_chunks_processed_in_iter);
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
