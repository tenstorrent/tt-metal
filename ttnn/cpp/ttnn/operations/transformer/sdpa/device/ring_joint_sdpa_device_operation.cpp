// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/ring_attention_all_gather_async_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/ring_joint_chain_layout.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation_types.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_perf_model.hpp"
#include "ttnn/tensor/types.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

using namespace experimental::ccl;

namespace {

namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;

// Chunked causal prefill does not have the same valid-pair geometry as a full causal
// Sq x Sk attention window. A new Q chunk attends to all prior K/V tokens as a full
// rectangle, then attends to its own chunk causally as a triangle:
//
//   valid_pairs_global = q_chunk * prefix_k + q_chunk * (q_chunk + 1) / 2
//
// Ring-joint shards Q rows across the ring, while every shard still sees the same
// global K prefix, so each device models 1 / ring_size of the global valid pairs.
// The generic SDPA causal model uses Sq * Sk / 2, which undercounts late chunked
// prefill chunks where most work is the prefix rectangle.
int compute_chunked_causal_sdpa_ideal_cycles(
    uint32_t batch_size,
    uint32_t num_heads_q,
    uint32_t q_global,
    uint32_t prefix_k_global,
    uint32_t ring_size,
    uint32_t DH,
    uint32_t DV,
    tt::tt_metal::MathFidelity math_fidelity,
    int num_cores) {
    if (ring_size == 0 || num_cores <= 0 || q_global == 0) {
        return 0;
    }

    const double q = static_cast<double>(q_global);
    const double prefix_k = static_cast<double>(prefix_k_global);
    const double valid_pairs_per_device = (q * prefix_k + (q * (q + 1.0) / 2.0)) / static_cast<double>(ring_size);
    return operations::transformer::sdpa::compute_sdpa_ideal_cycles_for_valid_pairs(
        batch_size, num_heads_q, valid_pairs_per_device, DH, DV, math_fidelity, num_cores);
}

void validate_ring_joint_all_gather_on_program_cache_miss(
    const ttnn::experimental::prim::RingAttentionAllGatherAsyncParams& operation_attributes,
    const ttnn::experimental::prim::RingAttentionAllGatherAsyncInputs& tensor_args,
    // Single-slot gather writes one cache slot to gathered slot 0, so allow a batch-1 output.
    bool allow_single_slot_output) {
    const auto& input_tensors = tensor_args.input_tensor;
    TT_FATAL(
        !input_tensors.empty(), "Error, Input tensor size should be greater than 0 but has {}", input_tensors.size());
    TT_FATAL(input_tensors[0].buffer() != nullptr, "Input tensor 0 must be allocated in buffers on device");

    const auto dtype = input_tensors[0].dtype();
    const auto page_size = input_tensors[0].buffer()->page_size();
    for (size_t i = 0; i < input_tensors.size(); ++i) {
        const auto& input_tensor = input_tensors[i];

        TT_FATAL(input_tensor.layout() == Layout::TILE, "Input tensor {} must be tiled", i);
        TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor {} must be on device", i);
        TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor {} must be allocated in buffers on device", i);
        TT_FATAL(
            input_tensor.dtype() == dtype,
            "All input tensors must have the same dtype. Input tensor {} has dtype {} but expected {}",
            i,
            input_tensor.dtype(),
            dtype);
        TT_FATAL(
            input_tensor.buffer()->page_size() == page_size,
            "All input tensors must have the same page size. Input tensor {} has page size {} but expected {}",
            i,
            input_tensor.buffer()->page_size(),
            page_size);
    }

    TT_FATAL(
        operation_attributes.num_links > 0,
        "Error, num_links should be more than 0 but has {}",
        operation_attributes.num_links);

    const auto& output_tensors = tensor_args.persistent_output_buffer;
    if (!output_tensors.empty()) {
        TT_FATAL(
            output_tensors.size() == input_tensors.size(),
            "Number of output tensors ({}) must match number of input tensors ({})",
            output_tensors.size(),
            input_tensors.size());

        for (size_t i = 0; i < output_tensors.size(); ++i) {
            TT_FATAL(output_tensors[i].has_value(), "RingJointSDPA requires persistent all-gather output tensor {}", i);
            const auto& output_tensor = output_tensors[i].value();

            TT_FATAL(output_tensor.layout() == Layout::TILE, "Output tensor {} must be tiled", i);
            TT_FATAL(output_tensor.storage_type() == StorageType::DEVICE, "Output tensor {} must be on device", i);
            TT_FATAL(output_tensor.buffer() != nullptr, "Output tensor {} must be allocated in buffers on device", i);
            TT_FATAL(
                output_tensor.dtype() == dtype,
                "Output tensor {} dtype should match input tensors but has {}",
                i,
                output_tensor.dtype());
            TT_FATAL(
                output_tensor.buffer()->page_size() == page_size,
                "Output tensor {} page size should match input tensors but has {}",
                i,
                output_tensor.buffer()->page_size());
            TT_FATAL(
                output_tensor.memory_config() == operation_attributes.output_mem_config,
                "Output tensor {} memory config should match output_mem_config",
                i);

            auto output_shape = output_tensor.logical_shape();
            auto expected_output_shape = input_tensors[i].logical_shape();
            expected_output_shape[operation_attributes.dim] *= operation_attributes.ring_size;
            for (int d = 0; d < static_cast<int>(output_shape.rank()); ++d) {
                if (d == operation_attributes.dim) {
                    TT_FATAL(
                        output_shape[d] >= expected_output_shape[d],
                        "Output tensor {} gather dim {} too small: got {}, expected >= {} "
                        "(= input_dim * ring_size {})",
                        i,
                        d,
                        output_shape[d],
                        expected_output_shape[d],
                        operation_attributes.ring_size);
                } else if (allow_single_slot_output && d == 0) {
                    // Single-slot gather targets gathered slot 0: batch-1 expected, full-batch also ok.
                    TT_FATAL(
                        output_shape[d] == 1 || output_shape[d] == expected_output_shape[d],
                        "Output tensor {} batch dim must be 1 (single-slot gather) or {}: got {}",
                        i,
                        expected_output_shape[d],
                        output_shape[d]);
                } else {
                    TT_FATAL(
                        output_shape[d] == expected_output_shape[d],
                        "Output tensor {} non-gather dim {} mismatch: got {}, expected {}",
                        i,
                        d,
                        output_shape[d],
                        expected_output_shape[d]);
                }
            }
        }
    }
}

// Re-validate the scalar args that are runtime-patched on a program-cache hit and therefore NOT part of
// compute_program_hash: kv_cache_batch_idx (indexed KV cache) and logical_n / kv_actual_isl (KV-pad
// rotation). Everything else is keyed by the hash, so a cache hit guarantees it already passed at miss
// time. Shared by validate_on_program_cache_miss and validate_on_program_cache_hit to avoid divergence.
void validate_runtime_patched_scalars(const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args) {
    if (args.has_indexed_kv_cache()) {
        const auto K_cache_batch = tensor_args.input_k.logical_shape()[0];
        const auto V_cache_batch =
            tensor_args.input_v.has_value() ? tensor_args.input_v->logical_shape()[0] : K_cache_batch;
        TT_FATAL(
            args.kv_cache_batch_idx.value() < K_cache_batch,
            "kv_cache_batch_idx={} is outside K cache batch={}",
            args.kv_cache_batch_idx.value(),
            K_cache_batch);
        TT_FATAL(
            args.kv_cache_batch_idx.value() < V_cache_batch,
            "kv_cache_batch_idx={} is outside V cache batch={}",
            args.kv_cache_batch_idx.value(),
            V_cache_batch);
    }

    if (args.has_kv_pad_rotation()) {
        const auto N_local_q = tensor_args.input_q.logical_shape()[2];
        const auto N_local_kv = tensor_args.local_kv_seq_len();
        const auto kv_actual_isl = args.kv_actual_isl.value();
        TT_FATAL(
            args.logical_n >= kv_actual_isl,
            "logical_n must be >= kv_actual_isl. Got logical_n={}, kv_actual_isl={}",
            args.logical_n,
            kv_actual_isl);
        const auto new_actual_isl = args.logical_n - kv_actual_isl;
        const auto chunk_capacity = N_local_q * args.ring_size;
        const auto cache_capacity = N_local_kv * args.ring_size;
        TT_FATAL(
            kv_actual_isl % tt::constants::TILE_HEIGHT == 0 && new_actual_isl % tt::constants::TILE_HEIGHT == 0,
            "KV-pad-aware rotation currently requires tile-aligned lengths. Got kv_actual_isl={}, "
            "new_actual_isl={} (logical_n - kv_actual_isl), TILE_HEIGHT={}",
            kv_actual_isl,
            new_actual_isl,
            tt::constants::TILE_HEIGHT);
        TT_FATAL(
            new_actual_isl > 0,
            "KV-pad-aware rotation requires at least one valid token in the current chunk. Got kv_actual_isl={}, "
            "logical_n={}",
            kv_actual_isl,
            args.logical_n);
        TT_FATAL(
            args.logical_n <= cache_capacity,
            "KV-pad-aware rotation logical_n exceeds physical K/V cache capacity. Got logical_n={}, "
            "cache capacity={}",
            args.logical_n,
            cache_capacity);
        TT_FATAL(
            new_actual_isl <= chunk_capacity,
            "KV-pad-aware rotation expects current valid Q to fit in one fixed chunk. Got new_actual_isl={}, "
            "chunk capacity={}",
            new_actual_isl,
            chunk_capacity);
    }
}

}  // namespace

void RingJointSDPADeviceOperation::validate_on_program_cache_miss(
    const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args) {
    const auto& input_tensor_q = tensor_args.input_q;
    const auto& gathered_input_tensor_k = tensor_args.gathered_k;

    const bool has_input_v = tensor_args.input_v.has_value();
    const bool has_gathered_v = tensor_args.gathered_v.has_value();
    const bool has_latent_v = tensor_args.has_latent_v();
    const bool has_joint_tensors =
        tensor_args.joint_q.has_value() || tensor_args.joint_k.has_value() || tensor_args.joint_v.has_value();
    TT_FATAL(
        tensor_args.joint_q.has_value() == has_joint_tensors && tensor_args.joint_k.has_value() == has_joint_tensors &&
            tensor_args.joint_v.has_value() == has_joint_tensors,
        "Joint tensors must be provided all together or omitted altogether");

    std::vector<Tensor> sdpa_input_tensors = {input_tensor_q, gathered_input_tensor_k};
    if (has_gathered_v) {
        sdpa_input_tensors.push_back(tensor_args.gathered_v.value());
    }
    if (has_joint_tensors) {
        sdpa_input_tensors.push_back(tensor_args.joint_q.value());
        sdpa_input_tensors.push_back(tensor_args.joint_k.value());
        sdpa_input_tensors.push_back(tensor_args.joint_v.value());
    }

    validate_ring_joint_all_gather_on_program_cache_miss(
        args.all_gather_operation_attributes, args.all_gather_tensor_args, args.has_indexed_kv_cache());

    // Check that SDPA coregrid does not overlap with AllGather coregrid
    TT_FATAL(args.program_config.has_value(), "Program config must be provided");
    const auto strategy = args.all_gather_operation_attributes.core_allocation_strategy;
    if (strategy == ttnn::ccl::CoreAllocationStrategy::COL_MAJOR) {
        TT_FATAL(
            args.ccl_core_grid_offset.x >= args.program_config.value().compute_with_storage_grid_size.x,
            "SDPA coregrid overlaps with AllGather coregrid (column-major)");
    } else {
        TT_FATAL(
            args.ccl_core_grid_offset.y >= args.program_config.value().compute_with_storage_grid_size.y,
            "SDPA coregrid overlaps with AllGather coregrid (row-major)");
    }

    // Validate joint strategy is 'rear'
    TT_FATAL(args.joint_strategy == "rear", "Joint strategy must be 'rear'. Got: {}", args.joint_strategy);

    // Get shapes
    const auto& q_shape = input_tensor_q.logical_shape();
    const auto& k_shape = gathered_input_tensor_k.logical_shape();
    const auto v_shape = has_gathered_v ? tensor_args.gathered_v->logical_shape() : k_shape;
    const auto joint_q_shape = has_joint_tensors ? tensor_args.joint_q.value().logical_shape() : q_shape;
    const auto joint_k_shape = has_joint_tensors ? tensor_args.joint_k.value().logical_shape() : q_shape;
    const auto joint_v_shape = has_joint_tensors ? tensor_args.joint_v.value().logical_shape() : q_shape;
    const bool has_indexed_kv_cache = args.has_indexed_kv_cache();
    const uint32_t NVH = tensor_args.v_num_heads();
    const uint32_t VDH = tensor_args.v_head_dim(args.latent_v_head_dim);

    // Chunked-prefill (`tensor_args.is_chunked()`): Q is shorter than the per-device K shard
    // (latest slab against a growing K cache). Chunk 0 has equal shapes and uses the regular
    // is_causal=True path.
    const bool is_chunked = tensor_args.is_chunked();

    const auto dtype = input_tensor_q.dtype();
    if ((!args.is_causal && !is_chunked) || args.is_cross) {
        for (const auto& tensor : sdpa_input_tensors) {
            TT_FATAL(
                tensor.dtype() == dtype,
                "All tensors must have the same dtype. Expected {}, got {}",
                dtype,
                tensor.dtype());
        }
    }

    // Validate storage types and buffers
    for (const auto& tensor : sdpa_input_tensors) {
        TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "Operands to Joint SDPA need to be on device");
        TT_FATAL(tensor.buffer() != nullptr, "Operands to Joint SDPA need to be allocated in buffers on device");
        TT_FATAL(tensor.layout() == Layout::TILE, "Inputs to Joint SDPA must be tilized");
        TT_FATAL(
            tensor.dtype() == DataType::BFLOAT16 || tensor.dtype() == DataType::BFLOAT8_B ||
                tensor.dtype() == DataType::BFLOAT4_B,
            "Inputs to Joint SDPA must be BF16 or BF8 or BF4");
        TT_FATAL(
            tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Operands to Joint SDPA need to be in DRAM");
    }

    // Validate input shapes match
    const auto B = q_shape[0];
    const auto NQH = q_shape[1];
    const auto NKH = k_shape[1];
    const auto N_local_q = q_shape[2];
    const auto N_local_kv = tensor_args.local_kv_seq_len();
    const auto N_global = k_shape[2];
    const auto L = has_joint_tensors ? joint_q_shape[2] : 0;
    const auto DH = q_shape[3];
    const uint32_t v_local_seq =
        has_input_v ? static_cast<uint32_t>(tensor_args.input_v->logical_shape()[2]) : N_local_kv;

    auto q_chunk_size = args.get_q_chunk_size();
    auto k_chunk_size = args.get_k_chunk_size();
    const bool has_kv_pad_rotation = args.has_kv_pad_rotation();

    TT_FATAL(!(L != 0 && args.is_causal), "Causality is enabled only for ring attention");

    TT_FATAL(
        args.logical_n > 0,
        "Logical sequence length must be > 0; kernels derive last-valid-tile = logical_nt - 1 and would underflow.");

    TT_FATAL(
        N_local_q <= N_local_kv,
        "Per-device Q seq length must be <= per-device K/V seq length. Equal: full-prefill path. Less: "
        "chunked-prefill path. Greater is undefined. Got N_local_q={}, N_local_kv={}",
        N_local_q,
        N_local_kv);

    TT_FATAL(
        !is_chunked || args.is_causal || args.is_cross,
        "Chunked-shaped prefill (N_local_q < N_local_kv) must be causal (incremental prefill) or cross "
        "(is_cross=True, non-causal short-Q/long-K). Got N_local_q={}, N_local_kv={}, is_causal={}, is_cross={}",
        N_local_q,
        N_local_kv,
        args.is_causal,
        args.is_cross);

    // Cross is the non-causal short-Q/long-K/V path: requires is_chunked, excludes is_causal and
    // balanced (causal-only) zigzag.
    if (args.is_cross) {
        TT_FATAL(
            is_chunked,
            "is_cross requires per-device Q seq length < K/V seq length; use the full-prefill non-causal "
            "path for equal lengths. Got N_local_q={}, N_local_kv={}",
            N_local_q,
            N_local_kv);
        TT_FATAL(
            !args.is_causal, "is_cross and is_causal are mutually exclusive (cross attention applies no triangle)");
        TT_FATAL(!args.is_balanced, "is_cross is non-causal; balanced zigzag load-balancing is causal-only");
    }

    // Value checks for the runtime-patched scalars (kv_cache_batch_idx, logical_n, kv_actual_isl).
    // Also invoked on every program-cache hit, where these values vary but the rest is hash-pinned.
    validate_runtime_patched_scalars(args, tensor_args);

    if (has_kv_pad_rotation) {
        // Shape/flag preconditions are pinned by the program hash. The logical_n / kv_actual_isl value
        // checks live in validate_runtime_patched_scalars (called below; also runs on cache hits).
        TT_FATAL(
            is_chunked,
            "kv_actual_isl enables KV-pad-aware rotation and requires chunked-prefill input (Q.seq < K.seq). "
            "Got N_local_q={}, N_local_kv={}",
            N_local_q,
            N_local_kv);
        TT_FATAL(
            args.is_causal,
            "kv_actual_isl enables KV-pad-aware rotation, which is causal-only. Got is_causal={}",
            args.is_causal);
        TT_FATAL(
            !args.is_balanced, "kv_actual_isl does not support balanced zigzag distribution. Pass is_balanced=false.");
        TT_FATAL(
            L == 0,
            "KV-pad-aware rotation currently supports ring attention without joint tokens. Got joint length L={}",
            L);
        TT_FATAL(
            N_local_kv % N_local_q == 0,
            "KV-pad-aware rotation expects K/V local sequence length to be an integer number of Q-sized slabs. "
            "Got N_local_kv={}, N_local_q={}",
            N_local_kv,
            N_local_q);
    }

    TT_FATAL(
        !(args.is_balanced && (N_local_q / 2) % q_chunk_size != 0),
        "q_chunk_size must divide half of local q seq_len in balanced case");

    TT_FATAL(
        has_input_v == has_gathered_v,
        "input_tensor_v and persistent_output_buffer_v must both be provided for tensor-V mode, or both omitted for "
        "latent-V mode");
    TT_FATAL(VDH > 0, "V head dimension must be provided and non-zero");
    if (has_latent_v) {
        TT_FATAL(NKH == 1, "Latent-V mode currently supports one shared KV head. Got K/V heads: {}", NKH);
        TT_FATAL(
            NVH == NKH,
            "Latent-V mode reads V from K's prefix, so V head count must match K head count. Got V: {}, K: {}",
            NVH,
            NKH);
        TT_FATAL(
            VDH < DH,
            "Latent-V mode reads V from K's strict prefix, so V head dim must be < K head dim. Got V: {}, K: {}",
            VDH,
            DH);
        TT_FATAL(
            VDH % tt::constants::TILE_WIDTH == 0,
            "Latent-V head dim must be tile aligned. Got V: {}, tile width: {}",
            VDH,
            tt::constants::TILE_WIDTH);
    }

    if (has_indexed_kv_cache) {
        const auto K_cache_batch = tensor_args.input_k.logical_shape()[0];
        const auto V_cache_batch = has_input_v ? tensor_args.input_v->logical_shape()[0] : K_cache_batch;
        TT_FATAL(
            B == 1,
            "kv_cache_batch_idx currently selects one shared K/V cache slot for the whole query batch; indexed K/V "
            "cache "
            "mode requires Q batch size 1. Got Q batch size {}",
            B);
        // kv_cache_batch_idx bounds are value checks → validate_runtime_patched_scalars (runs on hits too).
        // Single-slot gather writes cache slot kv_cache_batch_idx to gathered slot 0: a batch-1 buffer
        // is the efficient shape; a full-batch buffer is also accepted (only slot 0 is used).
        TT_FATAL(
            k_shape[0] == 1 || k_shape[0] == K_cache_batch,
            "Gathered K batch must be 1 (single-slot gather) or match input K cache batch {}. Got gathered K: {}",
            K_cache_batch,
            k_shape[0]);
        TT_FATAL(
            v_shape[0] == 1 || v_shape[0] == V_cache_batch,
            "Gathered V batch must be 1 (single-slot gather) or match input V cache batch {}. Got gathered V: {}",
            V_cache_batch,
            v_shape[0]);
    } else {
        TT_FATAL(k_shape[0] == B, "K batch size must match Q. Got Q: {}, K: {}", B, k_shape[0]);
        TT_FATAL(v_shape[0] == B, "V batch size must match Q. Got Q: {}, V: {}", B, v_shape[0]);
    }
    if (has_joint_tensors) {
        TT_FATAL(
            joint_q_shape[0] == B && joint_k_shape[0] == B && joint_v_shape[0] == B,
            "Batch sizes must match. Got Q: {}, joint_Q: {}, joint_K: {}, joint_V: {}",
            B,
            joint_q_shape[0],
            joint_k_shape[0],
            joint_v_shape[0]);
    }

    // Chunked-prefill targets MLA (K head dim == Q != V) — use is_causal's relaxed K-only check.
    // Cross is full attention (V head dim must equal Q), so it takes the strict check.
    if ((!args.is_causal && !is_chunked) || args.is_cross) {
        TT_FATAL(
            k_shape[3] == DH && VDH == DH, "Head dimensions must match. Got Q: {}, K: {}, V: {}", DH, k_shape[3], VDH);
        if (has_joint_tensors) {
            TT_FATAL(
                joint_q_shape[3] == DH && joint_k_shape[3] == DH && joint_v_shape[3] == DH,
                "Joint head dimensions must match Q. Got Q: {}, joint_Q: {}, joint_K: {}, joint_V: {}",
                DH,
                joint_q_shape[3],
                joint_k_shape[3],
                joint_v_shape[3]);
        }
    } else {
        TT_FATAL(k_shape[3] == DH, "Q/K head dimensions must match. Got Q: {}, K: {}", DH, k_shape[3]);
        if (has_joint_tensors) {
            TT_FATAL(
                joint_k_shape[3] == DH,
                "Q/joint_K head dimensions must match. Got Q: {}, joint_K: {}",
                DH,
                joint_k_shape[3]);
        }
    }

    TT_FATAL(
        args.logical_n <= N_global,
        "Logical sequence length must be less than or equal to global sequence length. Got logical sequence length: "
        "{}, global sequence length: {}",
        args.logical_n,
        N_global);

    if (has_joint_tensors) {
        TT_FATAL(
            joint_k_shape[2] == L && joint_v_shape[2] == L,
            "Joint sequence length must match. Got joint_K: {}, joint_V: {}",
            joint_k_shape[2],
            joint_v_shape[2]);
    }

    // Sharded-joint path validation
    if (tensor_args.joint_is_sharded()) {
        // Defensive guard: must be true by construction in the prim, but assert before any deref.
        TT_FATAL(
            tensor_args.gathered_joint_k.has_value() && tensor_args.gathered_joint_v.has_value(),
            "sharded joint path requires resolved gathered joint K/V buffers");

        TT_FATAL(args.logical_l > 0, "logical_l must be provided and > 0 for the sharded-joint path");
        TT_FATAL(
            args.logical_l % args.ring_size == 0,
            "logical_l ({}) must be divisible by ring_size ({})",
            args.logical_l,
            args.ring_size);
        TT_FATAL(
            L == args.logical_l / args.ring_size,
            "joint per-device seq ({}) must equal logical_l / ring_size ({} / {} = {})",
            L,
            args.logical_l,
            args.ring_size,
            args.logical_l / args.ring_size);
        TT_FATAL(
            L % tt::constants::TILE_HEIGHT == 0,
            "joint shard seq ({}) must be tile-aligned (TILE_HEIGHT={})",
            L,
            tt::constants::TILE_HEIGHT);
        TT_FATAL(
            tensor_args.gathered_joint_k->logical_shape()[2] == args.logical_l,
            "gathered joint K seq ({}) must equal logical_l ({})",
            tensor_args.gathered_joint_k->logical_shape()[2],
            args.logical_l);
        TT_FATAL(
            tensor_args.gathered_joint_v->logical_shape()[2] == args.logical_l,
            "gathered joint V seq ({}) must equal logical_l ({})",
            tensor_args.gathered_joint_v->logical_shape()[2],
            args.logical_l);

        // Mode incompatibilities: none of these combinations are reasoned about for sharded-joint.
        TT_FATAL(!args.is_causal, "sharded joint is incompatible with is_causal");
        TT_FATAL(!args.is_balanced, "sharded joint is incompatible with is_balanced (zigzag)");
        TT_FATAL(!args.is_cross, "sharded joint is incompatible with is_cross");
        TT_FATAL(!args.kv_cache_batch_idx.has_value(), "sharded joint is incompatible with indexed KV cache");
        TT_FATAL(!args.kv_actual_isl.has_value(), "sharded joint is incompatible with KV-pad rotation");

        // Page-size parity: joint K/V are appended to the same fused all-gather list as spatial K/V.
        // The AG validator enforces uniform page size; assert explicitly for a clear error on divergence.
        TT_FATAL(
            tensor_args.joint_k->buffer()->page_size() == tensor_args.input_k.buffer()->page_size(),
            "joint K page size ({}) must match spatial K page size ({}) for fused all-gather",
            tensor_args.joint_k->buffer()->page_size(),
            tensor_args.input_k.buffer()->page_size());
    } else if (args.logical_l > 0 && has_joint_tensors && L != args.logical_l) {
        TT_FATAL(
            false,
            "joint per-device seq ({}) must equal logical_l (replicated) or logical_l/ring_size (sharded). "
            "logical_l={}, ring_size={}",
            L,
            args.logical_l,
            args.ring_size);
    }

    TT_FATAL(
        N_global >= N_local_kv * args.ring_size,
        "Gathered K seq length must be >= per-device K shard times ring size. Got N_global: {}, N_local_kv: {}, "
        "ring_size: {}",
        N_global,
        N_local_kv,
        args.ring_size);
    TT_FATAL(
        k_shape[2] == v_shape[2],
        "K sequence length must be equal to V sequence length. Got K: {}, V: {}",
        k_shape[2],
        v_shape[2]);

    if (!has_latent_v) {
        const bool tensor_mha = (NQH == NKH) && (NKH == NVH);
        const bool tensor_separate_v_shared_k = (NKH == 1) && (NVH == NQH);
        const bool tensor_gqa_grouped_kv =
            ring_joint::is_gqa_grouped_kv_head_mode(/*v_shares_k_buffer=*/false, NQH, NKH, NVH);

        TT_FATAL(
            tensor_mha || tensor_separate_v_shared_k || tensor_gqa_grouped_kv,
            "Unsupported tensor-V head relationship. Expected MHA (NQH == NKH == NVH), separate-V shared-K "
            "(NKH == 1 && NVH == NQH), or GQA (NKH == NVH < NQH && NQH % NKH == 0). Got NQH: {}, "
            "NKH: {}, NVH: {}",
            NQH,
            NKH,
            NVH);
        TT_FATAL(
            !has_joint_tensors || !tensor_gqa_grouped_kv,
            "Ring joint SDPA GQA with joint tensors is unsupported. Got NQH: {}, NKH: {}, NVH: {}",
            NQH,
            NKH,
            NVH);
    } else {
        TT_FATAL(
            NKH == NVH || NKH == 1,
            "K num_heads must be equal to V num_heads or 1 in latent-V mode. Got K: {}, V: {}",
            NKH,
            NVH);
    }

    // Validate chunk sizes if program config is provided

    TT_FATAL(
        q_chunk_size % tt::constants::TILE_WIDTH == 0,
        "q_chunk_size must be divisible by TILE_SIZE. Got q_chunk_size: {}, TILE_SIZE: {}",
        q_chunk_size,
        tt::constants::TILE_WIDTH);
    TT_FATAL(
        k_chunk_size % tt::constants::TILE_WIDTH == 0,
        "k_chunk_size must be divisible by TILE_SIZE. Got k_chunk_size: {}, TILE_SIZE: {}",
        k_chunk_size,
        tt::constants::TILE_WIDTH);

    TT_FATAL(
        N_local_q % tt::constants::TILE_HEIGHT == 0,
        "Per-device Q seq length must be divisible by TILE_HEIGHT. Got N_local_q: {}, TILE_HEIGHT: {}",
        N_local_q,
        tt::constants::TILE_HEIGHT);
    TT_FATAL(
        N_local_kv % tt::constants::TILE_HEIGHT == 0,
        "Per-device K/V seq length must be divisible by TILE_HEIGHT. Got N_local_kv: {}, TILE_HEIGHT: {}",
        N_local_kv,
        tt::constants::TILE_HEIGHT);
    TT_FATAL(
        v_local_seq == N_local_kv,
        "V local seq length must match K local seq length. Got V: {}, K: {}",
        v_local_seq,
        N_local_kv);

    // Validate padding: Only the sequence dimension may be padded
    auto validate_padding = [](const Tensor& tensor) {
        const auto& logical_shape = tensor.logical_shape();
        const auto& padded_shape = tensor.padded_shape();
        TT_FATAL(logical_shape[0] == padded_shape[0], "Padding is not supported on the batch dimension");
        TT_FATAL(logical_shape[1] == padded_shape[1], "Padding is not supported on the num_heads dimension");
        TT_FATAL(logical_shape[3] == padded_shape[3], "Padding is not supported on the head_dim dimension");
    };

    for (const auto& tensor : sdpa_input_tensors) {
        validate_padding(tensor);
    }
}

void RingJointSDPADeviceOperation::validate_on_program_cache_hit(
    const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args) {
    // On a cache hit everything except the runtime-patched scalars is guaranteed by the program hash to
    // match a prior miss that already passed full validation. Re-check only the values that the hash no
    // longer keys on (kv_cache_batch_idx, logical_n, kv_actual_isl) — they are re-patched per dispatch.
    validate_runtime_patched_scalars(args, tensor_args);
}

RingJointSDPAResultSpec RingJointSDPADeviceOperation::compute_output_specs(
    const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args) {
    const auto& input = tensor_args.input_q;
    const bool has_joint_tensors = tensor_args.joint_q.has_value();
    const auto v_head_dim = tensor_args.v_head_dim(args.latent_v_head_dim);
    auto joint_output_shape = input.logical_shape();
    joint_output_shape[2] = 0;
    joint_output_shape[3] = v_head_dim;
    uint32_t joint_padded_seq = 0;
    if (has_joint_tensors) {
        joint_output_shape = tensor_args.joint_q.value().logical_shape();
        joint_output_shape[3] = v_head_dim;
        joint_padded_seq = tensor_args.joint_q.value().padded_shape()[2];
    }

    auto stats_shape = input.logical_shape();
    stats_shape[3] = 1;
    // 2× the sequence length: first half stores running max, second half stores running sum.
    // Used as DRAM scratch for multi-Q-chunk deferred norm round-trips between ring iterations.
    stats_shape[2] = (input.padded_shape()[2] + joint_padded_seq) * 2;

    auto out_shape = input.logical_shape();
    // head dim as v head dim
    out_shape[3] = v_head_dim;

    return {
        TensorSpec(out_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), args.output_memory_config)),
        TensorSpec(
            joint_output_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), args.output_memory_config)),
        TensorSpec(stats_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), args.output_memory_config))};
}

RingJointSDPAResult RingJointSDPADeviceOperation::create_output_tensors(
    const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args) {
    auto output_specs = compute_output_specs(args, tensor_args);
    return {
        create_device_tensor(output_specs[RING_JOINT_SDPA_OUTPUT_IDX], tensor_args.input_q.device()),
        create_device_tensor(output_specs[RING_JOINT_SDPA_JOINT_OUTPUT_IDX], tensor_args.input_q.device()),
        create_device_tensor(output_specs[RING_JOINT_SDPA_STATS_OUTPUT_IDX], tensor_args.input_q.device()),
    };
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensors> RingJointSDPADeviceOperation::create_op_performance_model(
    const RingJointSDPAParams& args, const RingJointSDPAInputs& tensor_args, RingJointSDPAResult& output_tensors) {
    Tensors input_tensors = {tensor_args.input_q, tensor_args.input_k};
    if (tensor_args.input_v.has_value()) {
        input_tensors.emplace_back(tensor_args.input_v.value());
    }
    if (tensor_args.joint_q.has_value()) {
        input_tensors.emplace_back(tensor_args.joint_q.value());
        input_tensors.emplace_back(tensor_args.joint_k.value());
    }
    if (tensor_args.joint_v.has_value()) {
        input_tensors.emplace_back(tensor_args.joint_v.value());
    }
    input_tensors.emplace_back(tensor_args.gathered_k);
    if (tensor_args.gathered_v.has_value()) {
        input_tensors.emplace_back(tensor_args.gathered_v.value());
    }

    auto& output_tensor = output_tensors[RING_JOINT_SDPA_OUTPUT_IDX];
    auto arch = output_tensor.storage_type() == StorageType::DEVICE ? output_tensor.device()->arch()
                                                                    : ttnn::GetDefaultDevice()->arch();

    if (arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
        log_warning(tt::LogOp, "RingJointSDPA perf model does not support arch '{}'", enchantum::to_string(arch));
        return operation::OpPerformanceModelGeneral<Tensors>(input_tensors, output_tensors, 0);
    }

    const auto& q_shape = tensor_args.input_q.logical_shape();
    const auto& gathered_k_shape = tensor_args.gathered_k.logical_shape();

    CoreCoord grid = args.program_config.has_value() ? args.program_config->compute_with_storage_grid_size
                                                     : output_tensor.device()->compute_with_storage_grid_size();
    tt::tt_metal::MathFidelity fidelity = ttnn::get_math_fidelity(args.compute_kernel_config);

    const uint32_t B = q_shape[0];
    const uint32_t NQH = q_shape[1];
    const uint32_t N_local = q_shape[2];
    const uint32_t N_global = gathered_k_shape[2];
    const bool has_joint_tensors =
        tensor_args.joint_q.has_value() || tensor_args.joint_k.has_value() || tensor_args.joint_v.has_value();
    const uint32_t L = tensor_args.joint_q.has_value() ? tensor_args.joint_q.value().logical_shape()[2] : 0;
    const uint32_t DH = q_shape[3];
    const uint32_t DV = tensor_args.v_head_dim(args.latent_v_head_dim);

    if (args.is_causal && args.has_kv_pad_rotation() && !has_joint_tensors) {
        const uint32_t logical_n = static_cast<uint32_t>(args.logical_n);
        const uint32_t prefix_k = args.kv_actual_isl.value();
        const uint32_t new_q_global = logical_n - prefix_k;
        const uint32_t ring_size = static_cast<uint32_t>(args.ring_size);
        int ideal_cycles = compute_chunked_causal_sdpa_ideal_cycles(
            B, NQH, new_q_global, prefix_k, ring_size, DH, DV, fidelity, grid.x * grid.y);
        return operation::OpPerformanceModelGeneral<Tensors>(input_tensors, output_tensors, ideal_cycles);
    }

    // RingJointSDPA: local Q and joint Q attend to (gathered K + joint K)
    // Total Q dimension: N_local + L, Total K dimension: N_global + L
    const uint32_t cat_Sq = N_local + L;
    const uint32_t cat_Sk = N_global + L;

    // Single attention pass over concatenated dimensions
    int ideal_cycles = operations::transformer::sdpa::compute_sdpa_ideal_cycles(
        B, NQH, cat_Sq, cat_Sk, DH, DV, args.is_causal, fidelity, grid.x * grid.y);

    return operation::OpPerformanceModelGeneral<Tensors>(input_tensors, output_tensors, ideal_cycles);
}

RingJointSDPAResult ring_joint_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const std::optional<ttnn::Tensor>& input_tensor_v,
    const std::optional<ttnn::Tensor>& joint_tensor_q,
    const std::optional<ttnn::Tensor>& joint_tensor_k,
    const std::optional<ttnn::Tensor>& joint_tensor_v,
    ttnn::Tensor& persistent_output_buffer_k,
    const std::optional<ttnn::Tensor>& persistent_output_buffer_v,
    const std::string& joint_strategy,
    const std::size_t logical_n,
    const std::size_t logical_l,
    ttnn::operations::transformer::SDPAProgramConfig program_config,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const uint32_t cluster_axis,
    const ttnn::MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    const CoreCoord ccl_core_grid_offset,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const bool is_causal,
    const bool is_balanced,
    const bool is_cross,
    const std::optional<float> scale,
    const std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const ttnn::ccl::CoreAllocationStrategy core_allocation_strategy,
    const std::optional<uint32_t> kv_cache_batch_idx,
    const std::optional<uint32_t> kv_actual_isl,
    const std::optional<uint32_t> latent_v_head_dim,
    const std::optional<ttnn::Tensor>& persistent_output_buffer_joint_k,
    const std::optional<ttnn::Tensor>& persistent_output_buffer_joint_v) {
    using OperationType = ttnn::prim::RingJointSDPADeviceOperation;

    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi2, true, false, false);

    log_debug(
        tt::LogOp,
        "Launching RingJointSDPA with core_allocation_strategy {}",
        enchantum::to_string(core_allocation_strategy));

    /**
     * Create RingAttentionAllGatherAsync struct.
     * It will be a member of the RingJointScaledDotProductAttention struct.
     */
    const auto& mesh_view = mesh_device.get_view();
    TT_FATAL(
        mesh_view.is_mesh_2d(),
        "all-gather invoked with cluster_axis API without 2D mesh, which is currently unsupported");
    std::size_t num_devices = (cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    int32_t rank = input_tensor_k.logical_shape().rank();
    int32_t gather_dim = (dim < 0) ? rank + dim : dim;

    TT_FATAL(
        gather_dim >= -rank && gather_dim <= rank - 1,
        "Dimension input should be in between -{} and {}, but has {}",
        rank,
        rank - 1,
        dim);

    auto all_gather_operation_attributes = ttnn::experimental::prim::RingAttentionAllGatherAsyncParams{
        {},
        gather_dim,
        num_links,
        num_devices,
        persistent_output_buffer_k.memory_config(),
        topology,
        multi_device_global_semaphore,
        subdevice_id,
        cluster_axis,
        core_allocation_strategy};
    std::vector<Tensor> all_gather_input_tensors = {input_tensor_k};
    std::vector<std::optional<Tensor>> all_gather_output_tensors = {persistent_output_buffer_k};
    if (input_tensor_v.has_value()) {
        TT_FATAL(
            persistent_output_buffer_v.has_value(),
            "persistent_output_buffer_v must be provided when input_tensor_v is provided");
        TT_FATAL(
            !latent_v_head_dim.has_value(),
            "latent_v_head_dim is only valid when input_tensor_v is omitted for latent-V mode");
        all_gather_input_tensors.push_back(input_tensor_v.value());
        all_gather_output_tensors.push_back(persistent_output_buffer_v);
    } else {
        TT_FATAL(
            !persistent_output_buffer_v.has_value(),
            "persistent_output_buffer_v must be omitted when input_tensor_v is omitted for latent-V mode");
        TT_FATAL(latent_v_head_dim.has_value(), "latent_v_head_dim must be provided when input_tensor_v is omitted");
        TT_FATAL(
            !joint_tensor_q.has_value() && !joint_tensor_k.has_value() && !joint_tensor_v.has_value(),
            "joint tensors must be omitted when input_tensor_v is omitted for latent-V mode");
    }

    // Detect sharded-joint path: joint per-device seq == logical_l / ring_size
    const std::size_t joint_seq =
        joint_tensor_k.has_value() ? static_cast<std::size_t>(joint_tensor_k->logical_shape()[2]) : 0;
    const bool joint_is_sharded = (logical_l > 0) && (joint_seq > 0) && (joint_seq == logical_l / num_devices);

    // For the sharded-joint path: allocate gather scratch buffers before building the AG list,
    // because all_gather_output_tensors must contain real tensors (nullopt is a fatal error).
    std::optional<Tensor> resolved_gathered_joint_k;
    std::optional<Tensor> resolved_gathered_joint_v;
    if (joint_is_sharded) {
        TT_FATAL(
            joint_tensor_k.has_value() && joint_tensor_v.has_value(),
            "Joint K and V must be provided for the sharded-joint path");
        const auto& jk = joint_tensor_k.value();
        auto jk_shape = jk.logical_shape();
        jk_shape[2] = static_cast<uint32_t>(logical_l);
        resolved_gathered_joint_k =
            persistent_output_buffer_joint_k.has_value()
                ? persistent_output_buffer_joint_k.value()
                : create_device_tensor(
                      TensorSpec(jk_shape, TensorLayout(jk.dtype(), PageConfig(Layout::TILE), jk.memory_config())),
                      jk.device());

        const auto& jv = joint_tensor_v.value();
        auto jv_shape = jv.logical_shape();
        jv_shape[2] = static_cast<uint32_t>(logical_l);
        resolved_gathered_joint_v =
            persistent_output_buffer_joint_v.has_value()
                ? persistent_output_buffer_joint_v.value()
                : create_device_tensor(
                      TensorSpec(jv_shape, TensorLayout(jv.dtype(), PageConfig(Layout::TILE), jv.memory_config())),
                      jv.device());

        all_gather_input_tensors.push_back(joint_tensor_k.value());
        all_gather_output_tensors.push_back(resolved_gathered_joint_k);
        all_gather_input_tensors.push_back(joint_tensor_v.value());
        all_gather_output_tensors.push_back(resolved_gathered_joint_v);
    }

    auto all_gather_tensor_args = ttnn::experimental::prim::RingAttentionAllGatherAsyncInputs{
        std::move(all_gather_input_tensors), std::move(all_gather_output_tensors)};

    auto operation_attributes = OperationType::operation_attributes_t(
        joint_strategy,
        scale,
        is_causal,
        is_balanced,
        is_cross,
        logical_n,
        logical_l,
        num_devices,
        tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        std::move(program_config),
        kernel_config_val,
        std::move(all_gather_operation_attributes),
        std::move(all_gather_tensor_args),
        ccl_core_grid_offset,
        kv_cache_batch_idx,
        kv_actual_isl,
        latent_v_head_dim.value_or(0));

    auto tensor_args = OperationType::tensor_args_t{
        .input_q = input_tensor_q,
        .input_k = input_tensor_k,
        .input_v = input_tensor_v,
        .joint_q = joint_tensor_q,
        .joint_k = joint_tensor_k,
        .joint_v = joint_tensor_v,
        .gathered_k = persistent_output_buffer_k,
        .gathered_v = persistent_output_buffer_v,
        .gathered_joint_k = resolved_gathered_joint_k,
        .gathered_joint_v = resolved_gathered_joint_v};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
