// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_minimal_matmul_async_device_operation.hpp"
#include <array>
#include <cstdint>
#include <optional>
#include <vector>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include "all_gather_minimal_matmul_async_program_factory.hpp"

#include <tt-metalium/hal.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

AllGatherMinimalMatmulAsyncOp::program_factory_t AllGatherMinimalMatmulAsyncOp::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return AllGatherMinimalMatmulAsyncProgramFactory{};
}

void AllGatherMinimalMatmulAsyncOp::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& act_tensor = tensor_args.input_tensor;
    const auto& weight_tensor = tensor_args.weight_tensor;
    const bool has_bias = tensor_args.bias_tensor.has_value();
    const Tensor* bias_ptr = has_bias ? &tensor_args.bias_tensor.value() : nullptr;

    // Topology check: only Ring topology is supported
    TT_FATAL(
        attributes.topology == tt::tt_fabric::Topology::Ring,
        "all_gather_minimal_matmul_async only supports Ring topology");

    // Basic device/storage checks
    TT_FATAL(
        act_tensor.storage_type() == StorageType::DEVICE && weight_tensor.storage_type() == StorageType::DEVICE,
        "all_gather_minimal_matmul_async operands must be on device");
    TT_FATAL(
        act_tensor.device() == weight_tensor.device(),
        "all_gather_minimal_matmul_async inputs must reside on the same device");
    TT_FATAL(
        act_tensor.buffer() != nullptr && weight_tensor.buffer() != nullptr,
        "all_gather_minimal_matmul_async inputs must be allocated in device buffers");
    if (has_bias) {
        const auto& bias_tensor = *bias_ptr;
        TT_FATAL(
            bias_tensor.storage_type() == StorageType::DEVICE,
            "all_gather_minimal_matmul_async bias must be on device");
        TT_FATAL(
            bias_tensor.device() == act_tensor.device(),
            "all_gather_minimal_matmul_async bias must be on the same device");
        TT_FATAL(
            bias_tensor.buffer() != nullptr,
            "all_gather_minimal_matmul_async bias must be allocated in a device buffer");
    }

    // Layout requirements: all inputs must be TILE layout
    TT_FATAL(
        act_tensor.layout() == Layout::TILE && weight_tensor.layout() == Layout::TILE,
        "all_gather_minimal_matmul_async requires TILE layout for activation and weight");
    if (has_bias) {
        TT_FATAL(bias_ptr->layout() == Layout::TILE, "all_gather_minimal_matmul_async requires TILE layout for bias");
    }

    // DType constraints: support BFLOAT16, BFLOAT8_B, BFLOAT4_B and FLOAT32
    auto dtype_supported = [](tt::tt_metal::DataType dt) {
        return dt == DataType::BFLOAT16 || dt == DataType::BFLOAT8_B || dt == DataType::BFLOAT4_B ||
               dt == DataType::FLOAT32;
    };
    TT_FATAL(
        dtype_supported(act_tensor.dtype()) && dtype_supported(weight_tensor.dtype()),
        "all_gather_minimal_matmul_async supports only BFLOAT16, BFLOAT8_B, BFLOAT4_B, and FLOAT32 for inputs");

    // Bias dtype constraint, if present
    if (has_bias) {
        TT_FATAL(
            dtype_supported(bias_ptr->dtype()),
            "all_gather_minimal_matmul_async supports only BFLOAT16, BFLOAT8_B, and BFLOAT4_B for bias");
    }

    // Shape constraints
    const auto& a_logical = act_tensor.logical_shape();
    const auto& w_logical = weight_tensor.logical_shape();
    TT_FATAL(
        a_logical.rank() >= 2 && w_logical.rank() >= 2, "all_gather_minimal_matmul_async expects rank >= 2 tensors");

    // Allow upper-dim broadcasting on activation (LHS): activation may have arbitrary upper dims
    for (int i = 0; i < static_cast<int>(w_logical.rank()) - 2; ++i) {
        TT_FATAL(w_logical[i] == 1, "all_gather_minimal_matmul_async weight must have 1 in all dims < -2");
    }

    const uint32_t M = a_logical[-2];
    const uint32_t K = a_logical[-1] * attributes.ring_size;
    // When FSDP fusion is active, the weight is sharded along K across `fsdp_ring_size`
    // devices, so the per-device weight only holds K/fsdp_ring_size of the K dim.
    const uint32_t K_w = w_logical[-2] * attributes.fsdp_ring_size;
    const uint32_t N = w_logical[-1];

    TT_FATAL(K == K_w, "all_gather_minimal_matmul_async inner dimensions must match, got K={} and K_w={}", K, K_w);
    TT_FATAL(M > 0 && K > 0 && N > 0, "all_gather_minimal_matmul_async dimensions must be positive");

    // FSDP fusion validation
    if (attributes.fsdp_cluster_axis.has_value()) {
        TT_FATAL(
            attributes.fsdp_ring_size > 1,
            "fsdp_cluster_axis is set but fsdp_ring_size is {} (expected > 1)",
            attributes.fsdp_ring_size);
        TT_FATAL(
            attributes.topology == ttnn::ccl::Topology::Linear,
            "FSDP-fused all_gather_minimal_matmul_async requires TP topology Linear (got {})",
            static_cast<uint32_t>(attributes.topology));
        TT_FATAL(
            attributes.fsdp_topology == ttnn::ccl::Topology::Linear,
            "FSDP-fused all_gather_minimal_matmul_async requires FSDP topology Linear (got {})",
            static_cast<uint32_t>(attributes.fsdp_topology));
        TT_FATAL(
            attributes.ring_size == attributes.fsdp_ring_size,
            "FSDP-fused all_gather_minimal_matmul_async requires ring_size == fsdp_ring_size (got {} vs {})",
            attributes.ring_size,
            attributes.fsdp_ring_size);
        TT_FATAL(
            !attributes.cluster_axis.has_value() ||
                attributes.cluster_axis.value() != attributes.fsdp_cluster_axis.value(),
            "fsdp_cluster_axis ({}) must not equal cluster_axis ({})",
            attributes.fsdp_cluster_axis.value(),
            attributes.cluster_axis.value_or(0));
        TT_FATAL(
            attributes.fsdp_semaphore.size() >= 2,
            "fsdp_semaphore must have at least 2 entries (ping-pong) when fsdp_cluster_axis is set, got {}",
            attributes.fsdp_semaphore.size());
        // Weight local K must be tile-aligned after FSDP sharding (use logical shape since
        // padded_shape isn't computed until later in this function).
        TT_FATAL(
            w_logical[-2] % TILE_HEIGHT == 0,
            "all_gather_minimal_matmul_async FSDP weight local K must be tile-aligned, got {}",
            w_logical[-2]);
        // persistent_weight_buffer must be provided
        TT_FATAL(
            tensor_args.persistent_weight_buffer.has_value(),
            "persistent_weight_buffer must be provided when fsdp_cluster_axis is set");
        const auto& pwb = tensor_args.persistent_weight_buffer.value();
        TT_FATAL(
            pwb.storage_type() == StorageType::DEVICE && pwb.buffer() != nullptr,
            "persistent_weight_buffer must be on device and allocated");
        TT_FATAL(pwb.layout() == Layout::TILE, "persistent_weight_buffer must be TILE layout");
        const auto& pwb_logical = pwb.logical_shape();
        TT_FATAL(pwb.dtype() == weight_tensor.dtype(), "persistent_weight_buffer dtype must match weight_tensor dtype");
        TT_FATAL(
            pwb_logical[-2] == K && pwb_logical[-1] == N,
            "persistent_weight_buffer shape must be [..., K={}, N={}], got [..., {}, {}]",
            K,
            N,
            pwb_logical[-2],
            pwb_logical[-1]);
    } else {
        TT_FATAL(
            attributes.fsdp_ring_size == 1,
            "fsdp_ring_size must be 1 when fsdp_cluster_axis is not set, got {}",
            attributes.fsdp_ring_size);
    }

    // Validate chunks and dim parameters
    const int32_t chunks = attributes.chunks;
    const int32_t dim = attributes.dim;
    TT_FATAL(chunks >= 1, "minimal_matmul requires chunks >= 1, got chunks={}", chunks);
    TT_FATAL(dim == -1, "minimal_matmul currently only supports dim=-1, got dim={}", dim);

    if (chunks > 1) {
        // Validate N is divisible by chunks
        TT_FATAL(N % chunks == 0, "Output width N={} must be divisible by chunks={}", N, chunks);

        // Validate each chunk is tile-aligned
        const uint32_t N_per_chunk = N / chunks;
        TT_FATAL(
            N_per_chunk % tt::constants::TILE_WIDTH == 0,
            "Each chunk size N/chunks={} must be a multiple of TILE_WIDTH={}",
            N_per_chunk,
            tt::constants::TILE_WIDTH);
    }

    if (has_bias) {
        const auto& b_logical = bias_ptr->logical_shape();
        TT_FATAL(b_logical.rank() >= 1, "all_gather_minimal_matmul_async bias must have rank >= 1");
        // All dims except the last must be 1 (i.e., shape is [..., 1, N])
        for (int i = 0; i < static_cast<int>(b_logical.rank()) - 1; ++i) {
            TT_FATAL(b_logical[i] == 1, "all_gather_minimal_matmul_async bias must be 1 in all dims except the last");
        }
        TT_FATAL(
            b_logical[-1] == N,
            "all_gather_minimal_matmul_async bias last dimension must equal N ({}), got {}",
            N,
            b_logical[-1]);
    }

    // Tile alignment checks (implicitly guaranteed by TILE layout, but assert inner two dims are tile-aligned)
    const auto& a_padded = act_tensor.padded_shape();
    const auto& w_padded = weight_tensor.padded_shape();
    TT_FATAL(
        a_padded[-2] % TILE_HEIGHT == 0 && a_padded[-1] % TILE_WIDTH == 0,
        "all_gather_minimal_matmul_async activation must be tile-aligned");
    TT_FATAL(
        w_padded[-2] % TILE_HEIGHT == 0 && w_padded[-1] % TILE_WIDTH == 0,
        "all_gather_minimal_matmul_async weight must be tile-aligned");
    if (has_bias) {
        const auto& b_padded = bias_ptr->padded_shape();
        TT_FATAL(
            b_padded[-1] % TILE_WIDTH == 0, "all_gather_minimal_matmul_async bias last dimension must be tile-aligned");
    }

    // Validate fused ternary tensors if present
    bool has_ternary_tensors =
        tensor_args.fused_ternary_input_a.has_value() && tensor_args.fused_ternary_input_b.has_value();
    bool has_fused_ternary = attributes.fused_ternary_scalar.has_value();
    TT_FATAL(
        !(has_ternary_tensors && !has_fused_ternary),
        "fused_ternary_scalar must be provided when addcmul input tensors are provided");
    if (has_fused_ternary) {
        TT_FATAL(
            has_ternary_tensors,
            "If fused_ternary_scalar is provided, both fused_ternary_input_a and fused_ternary_input_b must be "
            "provided");

        TT_FATAL(
            !attributes.fused_activation.has_value(),
            "minimal_matmul does not support using fused_activation together with ternary inputs "
            "(dit_minimal_matmul_addcmul_fused). "
            "Please use either fused_activation or ternary inputs, not both.");

        const auto& ternary_a = tensor_args.fused_ternary_input_a.value();
        const auto& ternary_b = tensor_args.fused_ternary_input_b.value();

        TT_FATAL(ternary_a.storage_type() == StorageType::DEVICE, "fused_ternary_input_a must be on device");
        TT_FATAL(ternary_b.storage_type() == StorageType::DEVICE, "fused_ternary_input_b must be on device");
        TT_FATAL(ternary_a.device() == act_tensor.device(), "fused_ternary_input_a must be on same device");
        TT_FATAL(ternary_b.device() == act_tensor.device(), "fused_ternary_input_b must be on same device");
        TT_FATAL(ternary_a.buffer() != nullptr, "fused_ternary_input_a must be allocated");
        TT_FATAL(ternary_b.buffer() != nullptr, "fused_ternary_input_b must be allocated");

        TT_FATAL(ternary_a.layout() == Layout::TILE, "fused_ternary_input_a must be TILE layout");
        TT_FATAL(ternary_b.layout() == Layout::TILE, "fused_ternary_input_b must be TILE layout");

        TT_FATAL(
            dtype_supported(ternary_a.dtype()) && dtype_supported(ternary_b.dtype()),
            "fused_ternary tensors must have supported dtypes");

        const auto& ternary_a_logical = ternary_a.logical_shape();
        const auto& ternary_b_logical = ternary_b.logical_shape();

        // ternary_a matches output [M, N], ternary_b is broadcast [1, N]
        TT_FATAL(
            ternary_a_logical[-2] == M && ternary_a_logical[-1] == N,
            "fused_ternary_input_a shape must match output [M={}, N={}], got [{}, {}]",
            M,
            N,
            ternary_a_logical[-2],
            ternary_a_logical[-1]);
        TT_FATAL(
            (ternary_b_logical[-2] == 1 || ternary_b_logical[-2] == M) && ternary_b_logical[-1] == N,
            "fused_ternary_input_b shape must be [1, N={}] (broadcast) or [M={}, N={}] (full), got [{}, {}]",
            N,
            M,
            N,
            ternary_b_logical[-2],
            ternary_b_logical[-1]);
    }

    // Config constraints
    if (attributes.config.has_value()) {
        const auto& cfg = attributes.config.value();
        TT_FATAL(cfg.M_block_size > 0 && cfg.K_block_size > 0 && cfg.N_block_size > 0, "Block sizes must be > 0");

        const uint32_t K_tiles_per_device = a_padded[-1] / TILE_WIDTH;
        // Ring topology uses a bidirectional half-block scheme that requires K_block_size to
        // evenly divide K_tiles_per_device (no tail-block support). Linear topology uses a
        // unidirectional full-block scheme that supports a tail block of K_tiles_per_device %
        // K_block_size tiles (zero-padded in L1 to keep the K_block_size row stride).
        if (attributes.topology != ttnn::ccl::Topology::Linear) {
            TT_FATAL(
                K_tiles_per_device % cfg.K_block_size == 0,
                "K_block_size ({}) must evenly divide the number of K tiles per device ({}) for Ring topology",
                cfg.K_block_size,
                K_tiles_per_device);
        }
        TT_FATAL(
            cfg.K_block_size <= K_tiles_per_device,
            "K_block_size ({}) must be <= K tiles per device ({})",
            cfg.K_block_size,
            K_tiles_per_device);
        TT_FATAL(cfg.subblock_h > 0 && cfg.subblock_w > 0, "Subblock sizes must be > 0");
        TT_FATAL(
            (cfg.M_block_size % cfg.subblock_h) == 0,
            "M_block_size ({}) must be divisible by subblock_h ({})",
            cfg.M_block_size,
            cfg.subblock_h);
        TT_FATAL(
            (cfg.N_block_size % cfg.subblock_w) == 0,
            "N_block_size ({}) must be divisible by subblock_w ({})",
            cfg.N_block_size,
            cfg.subblock_w);

        // Grid must be at least 2x2
        TT_FATAL(
            cfg.compute_with_storage_grid_size.x >= 2 && cfg.compute_with_storage_grid_size.y >= 2,
            "compute_with_storage_grid_size must be >= 2x2");

        // Additional grid checks are performed when creating the program
        auto device_grid = act_tensor.device()->compute_with_storage_grid_size();
        TT_FATAL(
            cfg.compute_with_storage_grid_size.x <= device_grid.x &&
                cfg.compute_with_storage_grid_size.y <= device_grid.y,
            "compute_with_storage_grid_size must be <= device grid size");

        const uint32_t max_dest_volume = get_dest_reg_count(attributes.compute_kernel_config);
        TT_FATAL(
            cfg.subblock_h * cfg.subblock_w <= max_dest_volume, "subblock_h * subblock_w must be <= max_dest_volume");
    }
}

AllGatherMinimalMatmulAsyncOp::spec_return_value_t AllGatherMinimalMatmulAsyncOp::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& in0_input_tensor = tensor_args.input_tensor;
    const auto& in1_input_tensor = tensor_args.weight_tensor;
    const auto& in0_input_tensor_shape = in0_input_tensor.logical_shape();
    const auto& in1_input_tensor_shape = in1_input_tensor.logical_shape();
    const uint32_t N = in1_input_tensor_shape[-1];
    const int32_t chunks = attributes.chunks;
    const bool fsdp_fused = attributes.fsdp_cluster_axis.has_value();

    ttnn::Shape intermediate_shape(in0_input_tensor_shape);
    intermediate_shape[-1] = intermediate_shape[-1] * attributes.ring_size;

    const auto& memory_config = attributes.output_mem_config.value_or(in0_input_tensor.memory_config());
    auto dtype = attributes.output_dtype.value_or(in0_input_tensor.dtype());

    // Create specs for output tensors
    // Layout: [activation_gather_intermediate, (optional: weight_gather_intermediate), chunks...]
    std::vector<TensorSpec> output_specs;
    output_specs.reserve(chunks + 1 + (fsdp_fused ? 1 : 0));

    output_specs.push_back(
        TensorSpec(intermediate_shape, TensorLayout(dtype, PageConfig(Layout::TILE), memory_config)));

    if (fsdp_fused) {
        // Gathered weight intermediate: [K_full, N_local] = [K_local * fsdp_ring_size, N_local].
        // Derive from in1_input_tensor_shape so we don't depend on persistent_weight_buffer being provided.
        ttnn::Shape weight_intermediate_shape(in1_input_tensor_shape);
        weight_intermediate_shape[-2] = weight_intermediate_shape[-2] * attributes.fsdp_ring_size;
        output_specs.push_back(TensorSpec(
            weight_intermediate_shape,
            TensorLayout(in1_input_tensor.dtype(), PageConfig(Layout::TILE), in1_input_tensor.memory_config())));
    }

    const uint32_t N_per_chunk = N / chunks;
    for (int32_t i = 0; i < chunks; ++i) {
        ttnn::Shape output_shape(in0_input_tensor_shape);
        output_shape[-1] = N_per_chunk;
        output_specs.push_back(TensorSpec(output_shape, TensorLayout(dtype, PageConfig(Layout::TILE), memory_config)));
    }

    return output_specs;
}

AllGatherMinimalMatmulAsyncOp::tensor_return_value_t AllGatherMinimalMatmulAsyncOp::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    std::vector<Tensor> output_tensors;
    auto* device = tensor_args.input_tensor.device();
    const auto output_specs = compute_output_specs(attributes, tensor_args);
    output_tensors.reserve(output_specs.size());
    const bool fsdp_fused = attributes.fsdp_cluster_axis.has_value();

    // Slot 0: activation gather buffer (persistent_output_buffer or new alloc)
    if (tensor_args.persistent_output_buffer.has_value()) {
        output_tensors.emplace_back(tensor_args.persistent_output_buffer.value());
    } else {
        output_tensors.emplace_back(create_device_tensor(output_specs[0], device));
    }

    // Slot 1 (if FSDP fused): gathered weight buffer (persistent_weight_buffer or new alloc)
    size_t next_idx = 1;
    if (fsdp_fused) {
        if (tensor_args.persistent_weight_buffer.has_value()) {
            output_tensors.emplace_back(tensor_args.persistent_weight_buffer.value());
        } else {
            output_tensors.emplace_back(create_device_tensor(output_specs[1], device));
        }
        next_idx = 2;
    }

    // Remaining slots: chunk outputs
    for (size_t i = next_idx; i < output_specs.size(); ++i) {
        output_tensors.emplace_back(create_device_tensor(output_specs[i], device));
    }

    return output_tensors;
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<ttnn::Tensor> all_gather_minimal_matmul_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<ttnn::Tensor>& bias_tensor,
    const std::optional<float> scalar,
    const std::optional<ttnn::Tensor>& addcmul_input_tensor1,
    const std::optional<ttnn::Tensor>& addcmul_input_tensor2,
    std::optional<ttnn::operations::unary::UnaryWithParam> fused_activation,
    const std::optional<const experimental::prim::MinimalMatmulConfig>& config,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const ttnn::ccl::Topology topology,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<ttnn::Tensor>& persistent_output_buffer,
    uint32_t num_links,
    std::optional<uint32_t> cluster_axis,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    const bool force_transpose,
    uint32_t num_workers_per_link,
    uint32_t num_buffers_per_channel,
    int32_t chunks,
    int32_t dim,
    std::optional<uint32_t> fsdp_cluster_axis,
    const std::vector<GlobalSemaphore>& fsdp_multi_device_global_semaphore,
    const std::optional<ttnn::Tensor>& persistent_weight_buffer,
    std::optional<ttnn::ccl::Topology> fsdp_topology) {
    using OperationType = ttnn::experimental::prim::AllGatherMinimalMatmulAsyncOp;

    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        compute_kernel_config,
        tt::tt_metal::MathFidelity::HiFi2,
        false /*approx_mode*/,
        true /*fp32_acc*/,
        true /*packer_acc*/);

    uint32_t num_devices = ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);
    uint32_t fsdp_num_devices =
        fsdp_cluster_axis.has_value() ? ttnn::ccl::get_topological_dimension(input_tensor, fsdp_cluster_axis) : 1;

    bool using_persistent_buffers = persistent_output_buffer.has_value();
    bool using_persistent_weight_buffer = persistent_weight_buffer.has_value();

    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);
    tt::tt_fabric::Topology fsdp_topology_ =
        fsdp_cluster_axis.has_value() ? ::ttnn::ccl::get_usable_topology(input_tensor, fsdp_topology, fsdp_cluster_axis)
                                      : fsdp_topology.value_or(ttnn::ccl::Topology::Ring);

    auto operation_attributes = OperationType::operation_attributes_t{
        config,
        std::move(fused_activation),
        memory_config,
        dtype,
        kernel_config_val,
        num_links,
        num_devices,
        topology_,
        multi_device_global_semaphore,
        cluster_axis,
        barrier_semaphore,
        using_persistent_buffers,
        force_transpose,
        num_workers_per_link,
        num_buffers_per_channel,
        scalar,
        chunks,
        dim,
        fsdp_cluster_axis,
        fsdp_num_devices,
        fsdp_multi_device_global_semaphore,
        using_persistent_weight_buffer,
        fsdp_topology_};
    auto tensor_args = OperationType::tensor_args_t{
        input_tensor,
        weight_tensor,
        bias_tensor,
        persistent_output_buffer,
        addcmul_input_tensor1,
        addcmul_input_tensor2,
        persistent_weight_buffer};

    std::vector<Tensor> returned_tensors =
        ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
    // Strip the activation-gather intermediate (slot 0) and, if present, the weight-gather
    // intermediate (slot 1). What's returned are just the chunked matmul outputs.
    size_t strip_count = 1 + (fsdp_cluster_axis.has_value() ? 1 : 0);
    return std::vector<Tensor>(returned_tensors.begin() + strip_count, returned_tensors.end());
}

}  // namespace ttnn::prim
