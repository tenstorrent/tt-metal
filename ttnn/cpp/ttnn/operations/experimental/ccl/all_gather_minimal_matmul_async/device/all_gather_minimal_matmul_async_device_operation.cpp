// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

namespace ttnn::operations::experimental::all_gather_minimal_matmul_async {

void AllGatherMinimalMatmulAsyncOp::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(
        input_tensors.size() == 2, "all_gather_minimal_matmul_async expects exactly 2 inputs: activation and weight");

    const auto& act_tensor = input_tensors.at(0);
    const auto& weight_tensor = input_tensors.at(1);
    const bool has_bias = (optional_input_tensors.size() == 1) && optional_input_tensors.at(0).has_value();
    const Tensor* bias_ptr = has_bias ? &optional_input_tensors.at(0).value() : nullptr;

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
    const uint32_t K = a_logical[-1] * this->ring_size;
    const uint32_t K_w = w_logical[-2];
    const uint32_t N = w_logical[-1];

    TT_FATAL(K == K_w, "all_gather_minimal_matmul_async inner dimensions must match, got K={} and K_w={}", K, K_w);
    TT_FATAL(M > 0 && K > 0 && N > 0, "all_gather_minimal_matmul_async dimensions must be positive");

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

    // Config constraints
    if (config.has_value()) {
        const auto& cfg = config.value();
        TT_FATAL(cfg.M_block_size > 0 && cfg.K_block_size > 0 && cfg.N_block_size > 0, "Block sizes must be > 0");
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

        // Grid must be at least 1x1
        TT_FATAL(
            cfg.compute_with_storage_grid_size.x >= 2 && cfg.compute_with_storage_grid_size.y >= 2,
            "compute_with_storage_grid_size must be >= 2x2");

        // Additional grid checks are performed when creating the program
        auto device_grid = act_tensor.device()->compute_with_storage_grid_size();
        TT_FATAL(
            cfg.compute_with_storage_grid_size.x <= device_grid.x &&
                cfg.compute_with_storage_grid_size.y <= device_grid.y,
            "compute_with_storage_grid_size must be <= device grid size");

        const uint32_t max_dest_volume = get_dest_reg_count(this->compute_kernel_config);
        TT_FATAL(
            cfg.subblock_h * cfg.subblock_w <= max_dest_volume, "subblock_h * subblock_w must be <= max_dest_volume");
    }
}

std::vector<TensorSpec> AllGatherMinimalMatmulAsyncOp::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& in0_input_tensor = input_tensors.at(0);
    const auto& in1_input_tensor = input_tensors.at(1);
    const auto& in0_input_tensor_shape = in0_input_tensor.logical_shape();
    const auto& in1_input_tensor_shape = in1_input_tensor.logical_shape();
    uint32_t N = in1_input_tensor_shape[-1];

    ttnn::Shape intermediate_shape(in0_input_tensor_shape);
    intermediate_shape[-1] = intermediate_shape[-1] * this->ring_size;

    ttnn::Shape output_shape(in0_input_tensor_shape);
    output_shape[-1] = N;

    const auto& memory_config = this->output_mem_config.value_or(in0_input_tensor.memory_config());
    auto dtype = this->output_dtype.value_or(in0_input_tensor.dtype());

    return {
        TensorSpec(intermediate_shape, TensorLayout(dtype, PageConfig(Layout::TILE), memory_config)),
        TensorSpec(output_shape, TensorLayout(dtype, PageConfig(Layout::TILE), memory_config))};
}

std::vector<Tensor> AllGatherMinimalMatmulAsyncOp::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    std::vector<Tensor> output_tensors;
    const auto& device = input_tensors.at(0).device();
    const auto& output_specs = this->compute_output_specs(input_tensors);
    output_tensors.reserve(output_specs.size());

    if (!optional_output_tensors.empty() and optional_output_tensors[0].has_value()) {
        output_tensors.emplace_back(optional_output_tensors[0].value());
        output_tensors.emplace_back(create_device_tensor(output_specs[1], device));
    } else {
        for (const auto& output_spec : output_specs) {
            output_tensors.emplace_back(create_device_tensor(output_spec, device));
        }
    }

    return output_tensors;
}

tt::tt_metal::operation::MeshWorkloadWithCallbacks AllGatherMinimalMatmulAsyncOp::create_mesh_workload(
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    return ccl::create_mesh_workload_from_programs(
        tensor_coords, input_tensors, output_tensors, [&, this](const ttnn::MeshCoordinate& coord) {
            return create_program_at(coord, input_tensors, optional_input_tensors, output_tensors);
        });
}

tt::tt_metal::operation::ProgramWithCallbacks AllGatherMinimalMatmulAsyncOp::create_program_at(
    const MeshCoordinate& coord,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    uint32_t device_index = ccl::get_linearized_index_from_physical_coord(input_tensors[0], coord, this->cluster_axis);

    std::optional<MeshCoordinate> forward_coord =
        ccl::get_physical_neighbor_from_physical_coord(input_tensors[0], coord, 1, this->topology, this->cluster_axis);

    std::optional<MeshCoordinate> backward_coord =
        ccl::get_physical_neighbor_from_physical_coord(input_tensors[0], coord, -1, this->topology, this->cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "DEBUG: forward_coord or backward_coord is null");

    const auto& act_tensor = input_tensors.at(0);
    const auto& weight_tensor = input_tensors.at(1);
    const auto& bias_tensor = optional_input_tensors.at(0);
    const auto& ag_output_tensor = output_tensors.at(0);
    const auto& mm_output_tensor = output_tensors.at(1);
    return detail::all_gather_minimal_matmul_async_factory(
        act_tensor,
        weight_tensor,
        bias_tensor,
        this->fused_activation,
        this->config,
        mm_output_tensor,
        ag_output_tensor,
        compute_kernel_config,
        coord,
        forward_coord,
        backward_coord,
        this->num_links,
        this->ring_size,
        device_index,
        this->topology,
        this->semaphore,
        this->barrier_semaphore,
        this->using_persistent_buffers,
        this->chunks_per_sync,
        this->num_workers_per_link,
        this->num_buffers_per_channel);
}

}  // namespace ttnn::operations::experimental::all_gather_minimal_matmul_async
