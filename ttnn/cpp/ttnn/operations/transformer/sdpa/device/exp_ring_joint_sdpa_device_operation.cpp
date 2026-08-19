// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_device_operation_types.hpp"
#include "ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_program_factory.hpp"
#include "ttnn/operations/transformer/sdpa/device/sdpa_perf_model.hpp"
#include "ttnn/tensor/types.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

void ExpRingJointSDPADeviceOperation::validate_on_program_cache_miss(
    const ExpRingJointSDPAParams& args, const ExpRingJointSDPAInputs& tensor_args) {
    const auto& input_tensor_q = tensor_args.input_q;

    const bool has_joint = tensor_args.joint_q.has_value();
    TT_FATAL(
        has_joint == tensor_args.joint_k.has_value() && has_joint == tensor_args.joint_v.has_value(),
        "Joint q/k/v must all be present or all absent");

    const auto& gathered_input_tensor_k = tensor_args.gathered_k;
    const auto& gathered_input_tensor_v = tensor_args.gathered_v;

    std::vector<Tensor> sdpa_input_tensors = {input_tensor_q, gathered_input_tensor_k, gathered_input_tensor_v};
    if (has_joint) {
        sdpa_input_tensors.push_back(tensor_args.joint_q.value());
        sdpa_input_tensors.push_back(tensor_args.joint_k.value());
        sdpa_input_tensors.push_back(tensor_args.joint_v.value());
    }

    TT_FATAL(args.program_config.has_value(), "Program config must be provided");

    // Validate joint strategy is 'rear'
    TT_FATAL(args.joint_strategy == "rear", "Joint strategy must be 'rear'. Got: {}", args.joint_strategy);

    // Validate all tensors have the same dtype
    const auto dtype = input_tensor_q.dtype();
    for (const auto& tensor : sdpa_input_tensors) {
        TT_FATAL(
            tensor.dtype() == dtype,
            "All tensors must have the same dtype. Expected {}, got {}",
            dtype,
            tensor.dtype());
    }

    // Get shapes
    const auto& q_shape = input_tensor_q.logical_shape();
    const auto& k_shape = gathered_input_tensor_k.logical_shape();
    const auto& v_shape = gathered_input_tensor_v.logical_shape();
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
    const auto N_local = q_shape[2];
    const auto N_global = k_shape[2];
    // Joint sequence length: 0 when there are no joint inputs (self-attention).
    const auto L = has_joint ? tensor_args.joint_q.value().logical_shape()[2] : 0;
    const auto DH = q_shape[3];

    TT_FATAL(
        k_shape[0] == B && v_shape[0] == B,
        "Batch sizes must match. Got Q: {}, K: {}, V: {}",
        B,
        k_shape[0],
        v_shape[0]);

    // Validate head dimensions match
    TT_FATAL(
        k_shape[3] == DH && v_shape[3] == DH,
        "Head dimensions must match. Got Q: {}, K: {}, V: {}",
        DH,
        k_shape[3],
        v_shape[3]);

    TT_FATAL(v_shape[1] == NKH, "Num heads must match. Got K: {}, V: {}", NKH, v_shape[1]);

    // Joint-input shape checks only apply when joint inputs are present.
    if (has_joint) {
        const auto& joint_q_shape = tensor_args.joint_q.value().logical_shape();
        const auto& joint_k_shape = tensor_args.joint_k.value().logical_shape();
        const auto& joint_v_shape = tensor_args.joint_v.value().logical_shape();
        TT_FATAL(
            joint_q_shape[0] == B && joint_k_shape[0] == B && joint_v_shape[0] == B,
            "Joint batch sizes must match. Got B: {}, joint_Q: {}, joint_K: {}, joint_V: {}",
            B,
            joint_q_shape[0],
            joint_k_shape[0],
            joint_v_shape[0]);
        TT_FATAL(
            joint_q_shape[3] == DH && joint_k_shape[3] == DH && joint_v_shape[3] == DH,
            "Joint head dimensions must match. Got DH: {}, joint_Q: {}, joint_K: {}, joint_V: {}",
            DH,
            joint_q_shape[3],
            joint_k_shape[3],
            joint_v_shape[3]);
        TT_FATAL(
            joint_q_shape[1] == NQH && joint_k_shape[1] == NKH && joint_v_shape[1] == NKH,
            "Joint num heads must match. Got NQH: {}, NKH: {}, joint_Q: {}, joint_K: {}, joint_V: {}",
            NQH,
            NKH,
            joint_q_shape[1],
            joint_k_shape[1],
            joint_v_shape[1]);
        TT_FATAL(
            joint_k_shape[2] == L && joint_v_shape[2] == L,
            "Joint sequence length must match. Got joint_K: {}, joint_V: {}",
            joint_k_shape[2],
            joint_v_shape[2]);
    }

    TT_FATAL(
        v_shape[2] == N_global,
        "V sequence length must be equal to global sequence length. Got V: {}, global sequence length: {}",
        v_shape[2],
        N_global);

    TT_FATAL(
        N_global == N_local * args.ring_size,
        "Global sequence length must be equal to local sequence length times ring size. Got global sequence length: "
        "{}, local sequence length: {}, ring size: {}",
        N_global,
        N_local,
        args.ring_size);

    TT_FATAL(
        args.logical_n <= N_global,
        "Logical sequence length must be less than or equal to global sequence length. Got logical sequence length: "
        "{}, global sequence length: {}",
        args.logical_n,
        N_global);

    TT_FATAL(
        (N_global - args.logical_n) < N_local,
        "Delta between global (padded) and logical (unpadded) sequence length must be less than local (per device) "
        "sequence length. Got delta: {}, local sequence length: {} "
        "This implies at least one device will have only padded tokens and no real tokens to process. Either "
        "reduce the ring size or reduce padding by reducing the chunk size.",
        N_global - args.logical_n,
        N_local);

    if (tensor_args.has_logical_n_tensor()) {
        const auto& t = tensor_args.logical_n_tensor.value();
        TT_FATAL(
            t.dtype() == DataType::UINT32 || t.dtype() == DataType::INT32,
            "logical_n tensor must be UINT32 or INT32 (the kernels read element 0 as a raw 32-bit word). Got {}",
            t.dtype());
        TT_FATAL(t.storage_type() == StorageType::DEVICE, "logical_n tensor must be on device");
        TT_FATAL(t.buffer() != nullptr, "logical_n tensor must be allocated on device");
        TT_FATAL(
            t.logical_volume() == 1,
            "logical_n tensor must hold exactly one value (the kernels read page 0, element 0). Got volume {}",
            t.logical_volume());
        // Live-value range is a caller contract (unverifiable on host, as with kv_actual_isl): a live
        // value violating (N_global - live_n) < N_local would empty a ring iteration, which the reader's
        // credit/gate counts assume never happens.
    }

    // Check shapes based on ring
    TT_FATAL(
        q_shape[2] * args.ring_size == k_shape[2],
        "Q sequence length times ring size must be equal to K sequence length. Got Q: {}, K: {}, ring_size: {}",
        q_shape[2],
        k_shape[2],
        args.ring_size);
    TT_FATAL(
        k_shape[2] == v_shape[2],
        "K sequence length must be equal to V sequence length. Got K: {}, V: {}",
        k_shape[2],
        v_shape[2]);

    TT_FATAL(NQH == NKH, "Q num_heads must be equal to K num_heads. Got Q: {}, K: {}", NQH, NKH);

    // Validate chunk sizes if program config is provided
    auto q_chunk_size = args.get_q_chunk_size();
    auto k_chunk_size = args.get_k_chunk_size();

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
        N_local % tt::constants::TILE_HEIGHT == 0,
        "Local sequence length must be divisible by TILE_HEIGHT. Got N_local: {}, TILE_HEIGHT: {}",
        N_local,
        tt::constants::TILE_HEIGHT);

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

    // --- Grid and chunk compatibility ---
    // The factory computes sdpa_grid = {user_grid.x - 1, user_grid.y} (last column = fabric MUX),
    // where user_grid is the program config's grid. Work is assigned row-aligned: each core row
    // hosts ceil(B*NQH / rows) heads and walks them as serial passes, one Q chunk per pass, with a
    // head's Q chunks filling its row. Mirror the factory's grid derivation exactly so validation
    // and the factory never disagree.

    TT_FATAL(
        DH % tt::constants::TILE_WIDTH == 0,
        "Head dimension ({}) must be divisible by TILE_WIDTH ({})",
        DH,
        tt::constants::TILE_WIDTH);

    TT_FATAL(args.num_links == 2, "Exp ring joint SDPA requires exactly 2 links. Got {}.", args.num_links);
    TT_FATAL(args.topology == ttnn::ccl::Topology::Ring, "Exp ring joint SDPA requires Ring topology.");

    const auto device_grid = input_tensor_q.device()->compute_with_storage_grid_size();
    TT_FATAL(
        device_grid.x >= 3,
        "Device grid must have at least 3 columns (2 reserved for fabric MUX + at least 1 for SDPA workers). "
        "Got {} columns.",
        device_grid.x);

    const CoreCoord user_grid =
        args.program_config.has_value() ? args.program_config->compute_with_storage_grid_size : device_grid;
    TT_FATAL(
        user_grid.x <= device_grid.x && user_grid.y <= device_grid.y,
        "Program config grid ({}x{}) exceeds device grid ({}x{}).",
        user_grid.x,
        user_grid.y,
        device_grid.x,
        device_grid.y);
    // Mirrors the factory's grid derivation, including the bottom-row MUX experiment.
    // Lower-bound the grid BEFORE the derivation: last-column mode subtracts the reserved MUX
    // column from x and bottom-row mode subtracts two rows from y, so an undersized
    // program-config grid would otherwise underflow unsigned here (and the num_q_chunks modulo
    // below would divide by zero) instead of failing with a clear error.
    const bool mux_on_bottom_row = exp_sdpa_mux_on_bottom_row();
    if (mux_on_bottom_row) {
        TT_FATAL(
            user_grid.y >= 3,
            "Program config grid ({}x{}) too short for bottom-row MUX placement: needs at least 3 "
            "rows (2 reserved for the MUX row and its spacer).",
            user_grid.x,
            user_grid.y);
    } else {
        TT_FATAL(
            user_grid.x >= 2,
            "Program config grid ({}x{}) too narrow: needs at least 2 columns (the last column is "
            "reserved for the fabric MUX kernels).",
            user_grid.x,
            user_grid.y);
    }
    const uint32_t sdpa_grid_x = mux_on_bottom_row ? user_grid.x : user_grid.x - 1;
    const uint32_t sdpa_grid_y = mux_on_bottom_row ? user_grid.y - 2 : user_grid.y;
    const uint32_t num_sdpa_cores = sdpa_grid_x * sdpa_grid_y;

    // Joint sequence must divide evenly (or be zero); last local Q chunk may be padded.
    TT_FATAL(
        L == 0 || L % q_chunk_size == 0,
        "Joint sequence length ({}) must be 0 or divisible by q_chunk_size ({}).",
        L,
        q_chunk_size);

    const uint32_t num_local_q_chunks = (N_local + q_chunk_size - 1) / q_chunk_size;
    const uint32_t num_joint_q_chunks = (L == 0) ? 0 : (L / q_chunk_size);
    const uint32_t num_q_chunks = num_local_q_chunks + num_joint_q_chunks;
    const uint32_t total_q_chunks = B * NQH * num_q_chunks;

    // Every head-segment must fill its row exactly: fewer chunks than columns would idle the
    // trailing columns, and the last two SDPA columns are the fabric MUX clients that drive the
    // K/V all-gather — an idle MUX column means that link never forwards its shard.
    TT_FATAL(
        num_q_chunks % sdpa_grid_x == 0,
        "Q chunks per head (num_local={} + num_joint={} = {}) must be a multiple of the SDPA grid "
        "columns ({}) on device grid {}×{}. Adjust q_chunk_size so ceil(N_local / q_chunk_size) is "
        "a multiple of {}.",
        num_local_q_chunks,
        num_joint_q_chunks,
        num_q_chunks,
        sdpa_grid_x,
        device_grid.x,
        device_grid.y,
        sdpa_grid_x);
    const uint32_t segs_per_head = num_q_chunks / sdpa_grid_x;
    const uint32_t total_segments = B * NQH * segs_per_head;

    // Every SDPA row must own at least one head-segment. An empty row builds no K/V chain and no
    // injector, and the MUX-writer columns of that row would then hit the row-has-injector
    // TT_FATAL during program construction — reject the shape here with an actionable message
    // instead.
    TT_FATAL(
        total_segments >= sdpa_grid_y,
        "Head-segments (B={} x NQH={} x segs_per_head={} = {}) must cover all {} SDPA grid rows; "
        "rows without a segment are not supported. Use a program_config grid with at most {} rows, "
        "or a smaller q_chunk_size to raise segs_per_head.",
        B,
        NQH,
        segs_per_head,
        total_segments,
        sdpa_grid_y,
        total_segments);

    // Segments per row: each core row hosts up to kMaxPasses head-segments, walked as serial
    // passes. Keep in lockstep with kMaxPasses in exp_ring_joint_sdpa_program_factory.cpp
    // (L1-bound).
    constexpr uint32_t kMaxPasses = 3;
    const uint32_t num_passes = (total_segments + sdpa_grid_y - 1) / sdpa_grid_y;
    TT_FATAL(
        num_passes <= kMaxPasses,
        "Number of head-segments (B={} × NQH={} × segs_per_head={} = {}) needs {} serial passes on "
        "{} SDPA grid rows (device grid {}×{}), but at most {} are supported. Reduce batch size or "
        "head count (e.g. via tensor parallelism), or use a larger q_chunk_size.",
        B,
        NQH,
        segs_per_head,
        total_segments,
        num_passes,
        sdpa_grid_y,
        device_grid.x,
        device_grid.y,
        kMaxPasses);

    // Final sanity: total Q chunks must fit the cores across all passes.
    TT_FATAL(
        total_q_chunks <= num_passes * num_sdpa_cores,
        "Total Q chunks (B={} × NQH={} × num_q_chunks={} = {}) exceeds SDPA cores ({}) across {} "
        "passes. The two constraints above should have caught this.",
        B,
        NQH,
        num_q_chunks,
        total_q_chunks,
        num_sdpa_cores,
        num_passes);
}

ExpRingJointSDPAResultSpec ExpRingJointSDPADeviceOperation::compute_output_specs(
    const ExpRingJointSDPAParams& args, const ExpRingJointSDPAInputs& tensor_args) {
    const auto& input = tensor_args.input_q;
    auto stats_shape = input.logical_shape();
    stats_shape[3] = 1;
    // Joint output is empty (zero joint sequence length) when there are no joint inputs.
    auto joint_output_shape = input.logical_shape();
    joint_output_shape[2] = 0;
    uint32_t joint_padded_seq = 0;
    if (tensor_args.joint_q.has_value()) {
        joint_output_shape = tensor_args.joint_q.value().logical_shape();
        joint_padded_seq = tensor_args.joint_q.value().padded_shape()[2];
    }
    // 2× the sequence length: first half stores running max, second half stores running sum.
    // Used as DRAM scratch for multi-Q-chunk deferred norm round-trips between ring iterations.
    stats_shape[2] = (input.padded_shape()[2] + joint_padded_seq) * 2;

    return {
        tt::tt_metal::TensorSpec(
            input.logical_shape(),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), args.output_memory_config)),
        tt::tt_metal::TensorSpec(
            joint_output_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), args.output_memory_config)),
        tt::tt_metal::TensorSpec(
            stats_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), args.output_memory_config))};
}

ExpRingJointSDPAResult ExpRingJointSDPADeviceOperation::create_output_tensors(
    const ExpRingJointSDPAParams& args, const ExpRingJointSDPAInputs& tensor_args) {
    auto output_specs = compute_output_specs(args, tensor_args);
    return {
        create_device_tensor(output_specs[EXP_RING_JOINT_SDPA_OUTPUT_IDX], tensor_args.input_q.device()),
        create_device_tensor(output_specs[EXP_RING_JOINT_SDPA_JOINT_OUTPUT_IDX], tensor_args.input_q.device()),
        create_device_tensor(output_specs[EXP_RING_JOINT_SDPA_STATS_OUTPUT_IDX], tensor_args.input_q.device()),
    };
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensors> ExpRingJointSDPADeviceOperation::create_op_performance_model(
    const ExpRingJointSDPAParams& args, const ExpRingJointSDPAInputs& tensor_args, ExpRingJointSDPAResult& output_tensors) {
    // Order mirrors compute_program_hash: q/k/v, then joints (if present), then gathered k/v.
    Tensors input_tensors = {tensor_args.input_q, tensor_args.input_k, tensor_args.input_v};
    if (tensor_args.joint_q.has_value()) {
        input_tensors.push_back(tensor_args.joint_q.value());
        input_tensors.push_back(tensor_args.joint_k.value());
        input_tensors.push_back(tensor_args.joint_v.value());
    }
    input_tensors.push_back(tensor_args.gathered_k);
    input_tensors.push_back(tensor_args.gathered_v);

    auto& output_tensor = output_tensors[EXP_RING_JOINT_SDPA_OUTPUT_IDX];
    auto arch = output_tensor.storage_type() == StorageType::DEVICE ? output_tensor.device()->arch()
                                                                    : ttnn::GetDefaultDevice()->arch();

    if (arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
        log_warning(tt::LogOp, "ExpRingJointSDPA perf model does not support arch '{}'", enchantum::to_string(arch));
        return operation::OpPerformanceModelGeneral<Tensors>(input_tensors, output_tensors, 0);
    }

    const auto& q_shape = tensor_args.input_q.logical_shape();
    const auto& gathered_k_shape = tensor_args.gathered_k.logical_shape();
    const auto& v_shape = tensor_args.gathered_v.logical_shape();

    CoreCoord grid = args.program_config.has_value() ? args.program_config->compute_with_storage_grid_size
                                                     : output_tensor.device()->compute_with_storage_grid_size();
    MathFidelity fidelity = ttnn::get_math_fidelity(args.compute_kernel_config);

    const uint32_t B = q_shape[0];
    const uint32_t NQH = q_shape[1];
    const uint32_t N_local = q_shape[2];
    const uint32_t N_global = gathered_k_shape[2];
    const uint32_t L = tensor_args.joint_q.has_value() ? tensor_args.joint_q.value().logical_shape()[2] : 0;
    const uint32_t DH = q_shape[3];
    const uint32_t DV = v_shape[3];

    // ExpRingJointSDPA: local Q and joint Q attend to (gathered K + joint K)
    // Total Q dimension: N_local + L, Total K dimension: N_global + L
    const uint32_t cat_Sq = N_local + L;
    const uint32_t cat_Sk = N_global + L;

    // Single attention pass over concatenated dimensions, non-causal
    int ideal_cycles = operations::transformer::sdpa::compute_sdpa_ideal_cycles(
        B, NQH, cat_Sq, cat_Sk, DH, DV, false, fidelity, grid.x * grid.y);

    return operation::OpPerformanceModelGeneral<Tensors>(input_tensors, output_tensors, ideal_cycles);
}

}  // namespace ttnn::prim

namespace ttnn::prim {

ExpRingJointSDPAResult exp_ring_joint_scaled_dot_product_attention(
    const ttnn::Tensor& input_tensor_q,
    const ttnn::Tensor& input_tensor_k,
    const ttnn::Tensor& input_tensor_v,
    const std::optional<ttnn::Tensor>& joint_tensor_q,
    const std::optional<ttnn::Tensor>& joint_tensor_k,
    const std::optional<ttnn::Tensor>& joint_tensor_v,
    ttnn::Tensor& persistent_output_buffer_k,
    ttnn::Tensor& persistent_output_buffer_v,
    const std::string& joint_strategy,
    const std::size_t logical_n,
    ttnn::operations::transformer::SDPAProgramConfig program_config,
    const int32_t dim,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
    const std::optional<float> scale,
    const std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const uint32_t num_workers_per_link,
    const uint32_t num_buffers_per_channel,
    const std::optional<ttnn::Tensor>& logical_n_tensor) {
    using OperationType = ttnn::prim::ExpRingJointSDPADeviceOperation;

    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi2, true, false, false);

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

    auto operation_attributes = OperationType::operation_attributes_t(
        joint_strategy,
        scale,
        logical_n,
        num_devices,
        tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        std::move(program_config),
        kernel_config_val,
        gather_dim,
        num_links,
        topology,
        multi_device_global_semaphore,
        subdevice_id,
        cluster_axis,
        num_workers_per_link,
        num_buffers_per_channel);

    auto tensor_args = OperationType::tensor_args_t{
        .input_q = input_tensor_q,
        .input_k = input_tensor_k,
        .input_v = input_tensor_v,
        .joint_q = joint_tensor_q,
        .joint_k = joint_tensor_k,
        .joint_v = joint_tensor_v,
        .gathered_k = persistent_output_buffer_k,
        .gathered_v = persistent_output_buffer_v,
        .logical_n_tensor = logical_n_tensor};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
