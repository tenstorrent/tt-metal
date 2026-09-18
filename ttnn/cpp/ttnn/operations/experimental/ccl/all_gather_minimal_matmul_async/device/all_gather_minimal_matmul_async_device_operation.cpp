// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_minimal_matmul_async_device_operation.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <optional>
#include <vector>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include "all_gather_minimal_matmul_async_program_factory.hpp"
#include "../registry/agmm_config_registry.hpp"
#include "ttnn/operations/matmul/device/config/matmul_config_registry.hpp"

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

    if (attributes.fuse_swiglu) {
        TT_FATAL(
            !attributes.fused_activation.has_value() && !attributes.fused_ternary_scalar.has_value(),
            "all_gather_minimal_matmul_async fuse_swiglu is mutually exclusive with fused_activation / ternary");
        TT_FATAL(attributes.chunks == 1, "all_gather_minimal_matmul_async fuse_swiglu does not yet support chunks > 1");
        TT_FATAL(
            N % (2 * tt::constants::TILE_WIDTH) == 0,
            "all_gather_minimal_matmul_async fuse_swiglu requires weight width N={} to be a multiple of "
            "2*TILE_WIDTH={}",
            N,
            2 * tt::constants::TILE_WIDTH);
    }

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

    const auto& explicit_chunk_sizes = attributes.chunk_sizes;
    if (!explicit_chunk_sizes.empty()) {
        // Variable-width chunks: `chunks` widths (elements) that may differ, summing to N.
        TT_FATAL(
            static_cast<int32_t>(explicit_chunk_sizes.size()) == chunks,
            "chunk_sizes must have exactly chunks={} entries, got {}",
            chunks,
            explicit_chunk_sizes.size());
        uint32_t chunk_sizes_sum = 0;
        for (size_t i = 0; i < explicit_chunk_sizes.size(); ++i) {
            const uint32_t width = explicit_chunk_sizes[i];
            TT_FATAL(width > 0, "chunk_sizes[{}] must be > 0", i);
            TT_FATAL(
                width % tt::constants::TILE_WIDTH == 0,
                "chunk_sizes[{}]={} must be a multiple of TILE_WIDTH={}",
                i,
                width,
                tt::constants::TILE_WIDTH);
            chunk_sizes_sum += width;
        }
        TT_FATAL(chunk_sizes_sum == N, "chunk_sizes must sum to the output width N={}, got {}", N, chunk_sizes_sum);
    } else if (chunks > 1) {
        // Uniform split (the default): N must divide evenly and each chunk must be tile-aligned.
        TT_FATAL(N % chunks == 0, "Output width N={} must be divisible by chunks={}", N, chunks);

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
    // SwiGLU halves the output along N (weight is the packed [gate|up] of width 2N).
    const uint32_t N = attributes.fuse_swiglu ? (in1_input_tensor_shape[-1] / 2) : in1_input_tensor_shape[-1];
    const int32_t chunks = attributes.chunks;
    const bool fsdp_fused = attributes.fsdp_cluster_axis.has_value();

    ttnn::Shape intermediate_shape(in0_input_tensor_shape);
    intermediate_shape[-1] = intermediate_shape[-1] * attributes.ring_size;

    const auto& memory_config = attributes.output_mem_config.value_or(in0_input_tensor.memory_config());
    auto dtype = attributes.output_dtype.value_or(in0_input_tensor.dtype());

    // Create specs for output tensors
    // Layout: [activation_gather_intermediate, (optional: weight_gather_intermediate), chunks...]
    std::vector<tt::tt_metal::TensorSpec> output_specs;
    output_specs.reserve(chunks + 1 + (fsdp_fused ? 1 : 0));

    output_specs.push_back(
        tt::tt_metal::TensorSpec(intermediate_shape, TensorLayout(dtype, PageConfig(Layout::TILE), memory_config)));

    if (fsdp_fused) {
        // Gathered weight intermediate: [K_full, N_local] = [K_local * fsdp_ring_size, N_local].
        // Derive from in1_input_tensor_shape so we don't depend on persistent_weight_buffer being provided.
        ttnn::Shape weight_intermediate_shape(in1_input_tensor_shape);
        weight_intermediate_shape[-2] = weight_intermediate_shape[-2] * attributes.fsdp_ring_size;
        output_specs.push_back(tt::tt_metal::TensorSpec(
            weight_intermediate_shape,
            TensorLayout(in1_input_tensor.dtype(), PageConfig(Layout::TILE), in1_input_tensor.memory_config())));
    }

    // Per-chunk widths: explicit when given, else the uniform N/chunks split.
    const auto chunk_sizes = resolve_chunk_sizes(attributes, N);
    for (const uint32_t chunk_width : chunk_sizes) {
        ttnn::Shape output_shape(in0_input_tensor_shape);
        output_shape[-1] = chunk_width;
        output_specs.push_back(
            tt::tt_metal::TensorSpec(output_shape, TensorLayout(dtype, PageConfig(Layout::TILE), memory_config)));
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
namespace {

namespace agmm_registry = ttnn::experimental::all_gather_minimal_matmul_registry;

agmm_registry::compact::TensorDescriptor describe_tensor(const ttnn::Tensor& tensor) {
    agmm_registry::compact::TensorDescriptor result;
    const auto& logical_shape = tensor.logical_shape();
    const auto& padded_shape = tensor.padded_shape();
    if (logical_shape.rank() > agmm_registry::compact::kMaxTensorRank || padded_shape.rank() != logical_shape.rank()) {
        return result;
    }
    result.rank = static_cast<std::uint8_t>(logical_shape.rank());
    for (std::size_t axis = 0; axis < logical_shape.rank(); ++axis) {
        result.logical_shape[axis] = logical_shape[axis];
        result.padded_shape[axis] = padded_shape[axis];
    }
    const auto& tile = tensor.tensor_spec().tile();
    const auto& memory_config = tensor.memory_config();
    result.dtype = static_cast<std::uint32_t>(tensor.dtype());
    result.layout = static_cast<std::uint32_t>(tensor.layout());
    result.memory_layout = static_cast<std::uint32_t>(memory_config.memory_layout());
    result.buffer_type = static_cast<std::uint32_t>(memory_config.buffer_type());
    result.tile_height = tile.get_height();
    result.tile_width = tile.get_width();
    result.tile_transpose_of_faces = tile.get_transpose_of_faces();
    result.tile_transpose_within_face = tile.get_transpose_within_face();
    return result;
}

agmm_registry::compact::OptionalTensorDescriptor describe_optional_tensor(const std::optional<ttnn::Tensor>& tensor) {
    return tensor
               ? agmm_registry::compact::OptionalTensorDescriptor{.present = true, .tensor = describe_tensor(*tensor)}
               : agmm_registry::compact::OptionalTensorDescriptor{};
}

agmm_registry::RegistryRequestFacts make_registry_facts(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<ttnn::Tensor>& bias_tensor,
    const std::optional<float> scalar,
    const std::optional<ttnn::Tensor>& ternary_input_a,
    const std::optional<ttnn::Tensor>& ternary_input_b,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& fused_activation,
    const std::vector<GlobalSemaphore>& semaphores,
    tt::tt_fabric::Topology topology,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<const DataType>& output_dtype,
    const std::optional<ttnn::Tensor>& persistent_output,
    std::uint32_t num_links,
    std::uint32_t ring_size,
    const std::optional<std::uint32_t>& cluster_axis,
    const std::optional<GlobalSemaphore>& barrier_semaphore,
    bool force_transpose,
    std::uint32_t num_workers_per_link,
    std::uint32_t num_buffers_per_channel,
    std::int32_t chunks,
    std::int32_t dim,
    const std::vector<std::uint32_t>& chunk_sizes,
    const std::optional<std::uint32_t>& fsdp_cluster_axis,
    std::uint32_t fsdp_ring_size,
    const std::vector<GlobalSemaphore>& fsdp_semaphores,
    const std::optional<ttnn::Tensor>& persistent_weight,
    tt::tt_fabric::Topology fsdp_topology,
    bool fuse_swiglu) {
    auto* mesh = input_tensor.device();
    const auto grid = mesh->compute_with_storage_grid_size();
    const auto& input_logical = input_tensor.logical_shape();
    const auto& input_padded = input_tensor.padded_shape();
    const auto& weight_logical = weight_tensor.logical_shape();
    const auto& weight_padded = weight_tensor.padded_shape();
    std::uint64_t batch = 1;
    for (std::size_t axis = 0; axis + 2 < input_logical.rank(); ++axis) {
        batch *= input_logical[axis];
    }

    agmm_registry::compact::OperationDescriptor operation{
        .topology = static_cast<std::uint32_t>(topology),
        .fsdp_topology = static_cast<std::uint32_t>(fsdp_topology),
        .num_links = num_links,
        .ring_size = ring_size,
        .cluster_axis_present = cluster_axis.has_value(),
        .cluster_axis = cluster_axis.value_or(0),
        .fsdp_cluster_axis_present = fsdp_cluster_axis.has_value(),
        .fsdp_cluster_axis = fsdp_cluster_axis.value_or(0),
        .fsdp_ring_size = fsdp_ring_size,
        .semaphore_count = static_cast<std::uint32_t>(semaphores.size()),
        .fsdp_semaphore_count = static_cast<std::uint32_t>(fsdp_semaphores.size()),
        .barrier_semaphore_present = barrier_semaphore.has_value(),
        .persistent_output_present = persistent_output.has_value(),
        .persistent_weight_present = persistent_weight.has_value(),
        .force_transpose = force_transpose,
        .num_workers_per_link = num_workers_per_link,
        .num_buffers_per_channel = num_buffers_per_channel,
        .scalar_present = scalar.has_value(),
        .scalar_f32_bits = scalar ? std::bit_cast<std::uint32_t>(*scalar) : 0,
        .chunks = chunks,
        .dim = dim,
        .fuse_swiglu = fuse_swiglu,
        .activation_present = fused_activation.has_value(),
        .activation_op = fused_activation ? static_cast<std::uint32_t>(fused_activation->op_type) : 0,
        .output_dtype_present = output_dtype.has_value(),
        .output_dtype = output_dtype ? static_cast<std::uint32_t>(*output_dtype) : 0,
        .output_memory_config_present = output_memory_config.has_value(),
        .output_memory_layout =
            output_memory_config ? static_cast<std::uint32_t>(output_memory_config->memory_layout()) : 0,
        .output_buffer_type =
            output_memory_config ? static_cast<std::uint32_t>(output_memory_config->buffer_type()) : 0,
        .output_layout = static_cast<std::uint32_t>(Layout::TILE),
        .output_tile_height = 32,
        .output_tile_width = 32};
    if (chunk_sizes.size() <= operation.chunk_sizes.size()) {
        operation.chunk_size_count = static_cast<std::uint8_t>(chunk_sizes.size());
        std::copy(chunk_sizes.begin(), chunk_sizes.end(), operation.chunk_sizes.begin());
    } else {
        operation.chunk_size_count = static_cast<std::uint8_t>(operation.chunk_sizes.size() + 1);
    }
    if (fused_activation && fused_activation->params.size() <= operation.activation_parameter_f32_bits.size()) {
        operation.activation_parameter_count = static_cast<std::uint8_t>(fused_activation->params.size());
        for (std::size_t index = 0; index < fused_activation->params.size(); ++index) {
            operation.activation_parameter_f32_bits[index] =
                std::bit_cast<std::uint32_t>(fused_activation->params[index]);
        }
    } else if (fused_activation) {
        operation.activation_parameter_count =
            static_cast<std::uint8_t>(operation.activation_parameter_f32_bits.size() + 1);
    }

    return agmm_registry::RegistryRequestFacts{
        .device =
            agmm_registry::compact::DeviceDescriptor{
                .architecture = static_cast<std::uint32_t>(mesh->arch()),
                .device_count = static_cast<std::uint16_t>(mesh->num_devices()),
                .mesh_rows = static_cast<std::uint16_t>(mesh->num_rows()),
                .mesh_cols = static_cast<std::uint16_t>(mesh->num_cols()),
                .compute_grid_x = static_cast<std::uint16_t>(grid.x),
                .compute_grid_y = static_cast<std::uint16_t>(grid.y)},
        .workload =
            agmm_registry::compact::WorkloadDescriptor{
                .logical_m = input_logical[-2],
                .logical_k = static_cast<std::uint64_t>(input_logical[-1]) * ring_size,
                .logical_n = weight_logical[-1],
                .padded_m = input_padded[-2],
                .padded_k = static_cast<std::uint64_t>(input_padded[-1]) * ring_size,
                .padded_n = weight_padded[-1],
                .batch = batch},
        .operation = operation,
        .input = describe_tensor(input_tensor),
        .weight = describe_tensor(weight_tensor),
        .bias = describe_optional_tensor(bias_tensor),
        .ternary_input_a = describe_optional_tensor(ternary_input_a),
        .ternary_input_b = describe_optional_tensor(ternary_input_b),
        .persistent_output = describe_optional_tensor(persistent_output),
        .persistent_weight = describe_optional_tensor(persistent_weight)};
}

}  // namespace

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
    std::optional<ttnn::ccl::Topology> fsdp_topology,
    bool fuse_swiglu,
    const std::vector<uint32_t>& chunk_sizes) {
    using OperationType = ttnn::experimental::prim::AllGatherMinimalMatmulAsyncOp;

    uint32_t num_devices = ttnn::ccl::get_topological_dimension(input_tensor, cluster_axis);
    uint32_t fsdp_num_devices =
        fsdp_cluster_axis.has_value() ? ttnn::ccl::get_topological_dimension(input_tensor, fsdp_cluster_axis) : 1;

    bool using_persistent_buffers = persistent_output_buffer.has_value();
    bool using_persistent_weight_buffer = persistent_weight_buffer.has_value();

    tt::tt_fabric::Topology topology_ = ::ttnn::ccl::get_usable_topology(input_tensor, topology, cluster_axis);
    tt::tt_fabric::Topology fsdp_topology_ =
        fsdp_cluster_axis.has_value() ? ::ttnn::ccl::get_usable_topology(input_tensor, fsdp_topology, fsdp_cluster_axis)
                                      : fsdp_topology.value_or(ttnn::ccl::Topology::Ring);

    const auto registry_mode = ttnn::operations::matmul::registry::current_mode();
    const bool registry_fallback_is_error = ttnn::operations::matmul::registry::fallback_is_error(registry_mode);
    std::optional<agmm_registry::Recipe> registry_recipe;
    if (input_tensor.logical_shape().rank() >= 2 && weight_tensor.logical_shape().rank() >= 2) {
        registry_recipe = agmm_registry::select_recipe(
            registry_mode,
            make_registry_facts(
                input_tensor,
                weight_tensor,
                bias_tensor,
                scalar,
                addcmul_input_tensor1,
                addcmul_input_tensor2,
                fused_activation,
                multi_device_global_semaphore,
                topology_,
                memory_config,
                dtype,
                persistent_output_buffer,
                num_links,
                num_devices,
                cluster_axis,
                barrier_semaphore,
                force_transpose,
                num_workers_per_link,
                num_buffers_per_channel,
                chunks,
                dim,
                chunk_sizes,
                fsdp_cluster_axis,
                fsdp_num_devices,
                fsdp_multi_device_global_semaphore,
                persistent_weight_buffer,
                fsdp_topology_,
                fuse_swiglu));
    }
    if (registry_fallback_is_error && !registry_recipe) {
        TT_THROW("AGMM registry required an exact recipe, but dispatch fell back: ineligible request");
    }
    auto selected_config = config;
    auto selected_kernel_config = compute_kernel_config;
    if (registry_recipe) {
        selected_config.emplace(registry_recipe->config);
        selected_kernel_config = registry_recipe->compute_kernel_config;
    }
    auto kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        selected_kernel_config,
        tt::tt_metal::MathFidelity::HiFi2,
        false /*approx_mode*/,
        true /*fp32_acc*/,
        true /*packer_acc*/);

    auto operation_attributes = OperationType::operation_attributes_t{
        selected_config,
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
        fsdp_topology_,
        fuse_swiglu,
        chunk_sizes};
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
