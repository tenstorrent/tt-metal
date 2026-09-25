// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "agmm_config_registry.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt_stl/assert.hpp>

#include "agmm_registry_data.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/matmul/device/config/matmul_config_registry.hpp"

namespace ttnn::experimental::all_gather_minimal_matmul_registry {
namespace {

bool is_default_tile(const compact::TensorDescriptor& tensor) noexcept {
    return tensor.layout == static_cast<std::uint32_t>(tt::tt_metal::Layout::TILE) &&
           tensor.tile_height == tt::constants::TILE_HEIGHT && tensor.tile_width == tt::constants::TILE_WIDTH &&
           !tensor.tile_transpose_of_faces && !tensor.tile_transpose_within_face;
}

bool tensor_is_complete(const compact::TensorDescriptor& tensor) noexcept {
    if (tensor.rank < 2 || tensor.rank > compact::kMaxTensorRank) {
        return false;
    }
    for (std::size_t axis = 0; axis < compact::kMaxTensorRank; ++axis) {
        if (axis < tensor.rank) {
            if (tensor.logical_shape[axis] == 0 || tensor.padded_shape[axis] < tensor.logical_shape[axis]) {
                return false;
            }
        } else if (tensor.logical_shape[axis] != 0 || tensor.padded_shape[axis] != 0) {
            return false;
        }
    }
    return true;
}

bool optional_tensor_is_complete(const compact::OptionalTensorDescriptor& tensor) noexcept {
    return tensor.present ? tensor_is_complete(tensor.tensor) && is_default_tile(tensor.tensor)
                          : tensor.tensor == compact::TensorDescriptor{};
}

std::optional<std::uint64_t> checked_product(std::uint64_t lhs, std::uint64_t rhs) noexcept {
    std::uint64_t product = 0;
    return tt::checked_mul(&product, lhs, rhs) ? std::nullopt : std::make_optional(product);
}

bool shape_matches_workload(const compact::KeyDescriptor& key) noexcept {
    const auto& input = key.input;
    const auto& weight = key.weight;
    if (key.operation.ring_size == 0 || key.operation.fsdp_ring_size == 0) {
        return false;
    }
    const auto input_m = static_cast<std::size_t>(input.rank - 2);
    const auto input_k = static_cast<std::size_t>(input.rank - 1);
    const auto weight_k = static_cast<std::size_t>(weight.rank - 2);
    const auto weight_n = static_cast<std::size_t>(weight.rank - 1);
    const auto logical_k = checked_product(input.logical_shape[input_k], key.operation.ring_size);
    const auto logical_weight_k = checked_product(weight.logical_shape[weight_k], key.operation.fsdp_ring_size);
    const auto padded_k = checked_product(input.padded_shape[input_k], key.operation.ring_size);
    const auto padded_weight_k = checked_product(weight.padded_shape[weight_k], key.operation.fsdp_ring_size);
    if (!logical_k || !logical_weight_k || !padded_k || !padded_weight_k || logical_k != logical_weight_k ||
        padded_k != padded_weight_k) {
        return false;
    }
    std::uint64_t batch = 1;
    for (std::size_t axis = 0; axis < input_m; ++axis) {
        const auto next = checked_product(batch, input.logical_shape[axis]);
        if (!next) {
            return false;
        }
        batch = *next;
    }
    return key.workload.logical_m == input.logical_shape[input_m] && key.workload.logical_k == *logical_k &&
           key.workload.logical_n == weight.logical_shape[weight_n] &&
           key.workload.padded_m == input.padded_shape[input_m] && key.workload.padded_k == *padded_k &&
           key.workload.padded_n == weight.padded_shape[weight_n] && key.workload.batch == batch;
}

bool operation_is_consistent(const compact::KeyDescriptor& key) noexcept {
    const auto& operation = key.operation;
    if (operation.chunks < 1 || operation.dim != -1 || operation.chunk_size_count > compact::kMaxChunkSizes ||
        operation.activation_parameter_count > compact::kMaxActivationParameters ||
        (!operation.scalar_present && operation.scalar_f32_bits != 0) ||
        operation.scalar_present != (key.ternary_input_a.present && key.ternary_input_b.present) ||
        key.ternary_input_a.present != key.ternary_input_b.present ||
        operation.persistent_output_present != key.persistent_output.present ||
        operation.persistent_weight_present != key.persistent_weight.present ||
        (operation.scalar_present && operation.activation_present) ||
        (operation.fuse_swiglu &&
         (operation.scalar_present || operation.activation_present || operation.chunks != 1)) ||
        (!operation.cluster_axis_present && operation.cluster_axis != 0) ||
        (!operation.fsdp_cluster_axis_present && operation.fsdp_cluster_axis != 0) ||
        (!operation.activation_present &&
         (operation.activation_op != 0 || operation.activation_parameter_count != 0))) {
        return false;
    }
    constexpr std::uint32_t kLinearTopology = 1;
    if (operation.fsdp_cluster_axis_present) {
        if (operation.fsdp_ring_size <= 1 || operation.ring_size != operation.fsdp_ring_size ||
            operation.topology != kLinearTopology || operation.fsdp_topology != kLinearTopology ||
            !operation.persistent_weight_present || operation.fsdp_semaphore_count < 2 ||
            (operation.cluster_axis_present && operation.cluster_axis == operation.fsdp_cluster_axis)) {
            return false;
        }
    } else if (operation.fsdp_ring_size != 1) {
        return false;
    }
    std::uint64_t chunk_sum = 0;
    for (std::size_t index = 0; index < compact::kMaxChunkSizes; ++index) {
        const auto width = operation.chunk_sizes[index];
        if (index < operation.chunk_size_count) {
            if (width == 0 || width % tt::constants::TILE_WIDTH != 0) {
                return false;
            }
            chunk_sum += width;
        } else if (width != 0) {
            return false;
        }
    }
    return operation.chunk_size_count == 0
               ? operation.chunks == 1 || (key.workload.logical_n % operation.chunks == 0 &&
                                           (key.workload.logical_n / operation.chunks) %
                                                   tt::constants::TILE_WIDTH ==
                                               0)
               : operation.chunk_size_count == static_cast<std::size_t>(operation.chunks) &&
                     chunk_sum == key.workload.logical_n;
}

bool key_is_consistent(const compact::KeyDescriptor& key) noexcept {
    return key.schema_version == compact::kKeySchemaVersion && key.codegen_recipe_abi == compact::kCodegenRecipeAbi &&
           compact::is_supported_device(key.device) && tensor_is_complete(key.input) &&
           tensor_is_complete(key.weight) && is_default_tile(key.input) && is_default_tile(key.weight) &&
           optional_tensor_is_complete(key.bias) && optional_tensor_is_complete(key.ternary_input_a) &&
           optional_tensor_is_complete(key.ternary_input_b) && optional_tensor_is_complete(key.persistent_output) &&
           optional_tensor_is_complete(key.persistent_weight) && key.workload.logical_m != 0 &&
           key.workload.logical_k != 0 && key.workload.logical_n != 0 && key.workload.padded_m != 0 &&
           key.workload.padded_k != 0 && key.workload.padded_n != 0 && key.workload.batch != 0 &&
           operation_is_consistent(key) &&
           key.operation.output_layout == static_cast<std::uint32_t>(tt::tt_metal::Layout::TILE) &&
           key.operation.output_tile_height == tt::constants::TILE_HEIGHT &&
           key.operation.output_tile_width == tt::constants::TILE_WIDTH &&
           !key.operation.output_tile_transpose_of_faces &&
           !key.operation.output_tile_transpose_within_face && shape_matches_workload(key);
}

}  // namespace

std::optional<compact::KeyDescriptor> build_registry_key(const RegistryRequestFacts& facts) noexcept {
    auto key = compact::KeyDescriptor{
        .device = facts.device,
        .workload = facts.workload,
        .operation = facts.operation,
        .input = facts.input,
        .weight = facts.weight,
        .bias = facts.bias,
        .ternary_input_a = facts.ternary_input_a,
        .ternary_input_b = facts.ternary_input_b,
        .persistent_output = facts.persistent_output,
        .persistent_weight = facts.persistent_weight};
    return key_is_consistent(key) ? std::make_optional(key) : std::nullopt;
}

std::span<const compact::CohortDescriptor> cohorts_for_device_count(const std::uint16_t device_count) noexcept {
    if (device_count == 8) {
        return generated::blackhole_8_device_cohorts();
    }
    if (device_count == 32) {
        return generated::blackhole_32_device_cohorts();
    }
    return {};
}

const compact::EntryDescriptor* lookup(const compact::KeyDescriptor& key) noexcept {
    if (!key_is_consistent(key)) {
        return nullptr;
    }
    for (const auto& cohort : cohorts_for_device_count(key.device.device_count)) {
        // The campaign grid is a minimum capability, not a physical-device
        // identity: a larger live grid is legal, while a harvested/smaller
        // grid is not.
        const auto& required = cohort.device;
        if (key.device.architecture != required.architecture || key.device.device_count != required.device_count ||
            key.device.mesh_rows != required.mesh_rows || key.device.mesh_cols != required.mesh_cols ||
            key.device.compute_grid_x < required.compute_grid_x ||
            key.device.compute_grid_y < required.compute_grid_y) {
            continue;
        }
        auto normalized = key;
        normalized.device.compute_grid_x = required.compute_grid_x;
        normalized.device.compute_grid_y = required.compute_grid_y;
        if (const auto* entry = compact::lookup_exact(normalized, cohort.entries)) {
            return entry;
        }
    }
    return nullptr;
}

std::optional<Recipe> materialize_recipe(const compact::EntryDescriptor& descriptor) noexcept {
    if (!key_is_consistent(descriptor.key) || descriptor.replay.schema_version != compact::kReplaySchemaVersion) {
        return std::nullopt;
    }
    const auto& config = descriptor.replay.config;
    if (config.m_block_size == 0 || config.k_block_size == 0 || config.n_block_size == 0 || config.subblock_h == 0 ||
        config.subblock_w == 0 || config.compute_grid_x < 2 || config.compute_grid_y < 2 ||
        config.compute_grid_x > descriptor.key.device.compute_grid_x ||
        config.compute_grid_y > descriptor.key.device.compute_grid_y || config.m_block_size % config.subblock_h != 0 ||
        config.n_block_size % config.subblock_w != 0 ||
        (descriptor.key.operation.fuse_swiglu && config.n_block_size % 2 != 0)) {
        return std::nullopt;
    }
    const auto local_k_tiles =
        descriptor.key.input.padded_shape[descriptor.key.input.rank - 1] / tt::constants::TILE_WIDTH;
    if (config.k_block_size > local_k_tiles ||
        (descriptor.key.operation.topology != 1 && local_k_tiles % config.k_block_size != 0)) {
        return std::nullopt;
    }

    const auto fidelity = math_fidelity_from_raw_value(descriptor.replay.compute_kernel_config.math_fidelity);
    if (!fidelity) {
        return std::nullopt;
    }
    using ThrottleLevel = ttnn::operations::compute_throttle_utils::ThrottleLevel;
    const auto raw_throttle = descriptor.replay.compute_kernel_config.throttle_level;
    if (raw_throttle > static_cast<std::uint32_t>(ThrottleLevel::LEVEL_5)) {
        return std::nullopt;
    }
    const auto& kernel = descriptor.replay.compute_kernel_config;
    auto compute_kernel_config = DeviceComputeKernelConfig{
        .math_fidelity = *fidelity,
        .math_approx_mode = kernel.math_approx_mode,
        .fp32_dest_acc_en = kernel.fp32_dest_acc_en,
        .packer_l1_acc = kernel.packer_l1_acc,
        .dst_full_sync_en = kernel.dst_full_sync_en,
        .throttle_level = static_cast<ThrottleLevel>(raw_throttle)};
    if (config.subblock_h > get_dest_reg_count(compute_kernel_config) / config.subblock_w) {
        return std::nullopt;
    }
    return Recipe{
        .config =
            ttnn::experimental::prim::MinimalMatmulConfig{
                .M_block_size = config.m_block_size,
                .K_block_size = config.k_block_size,
                .N_block_size = config.n_block_size,
                .subblock_h = config.subblock_h,
                .subblock_w = config.subblock_w,
                .compute_with_storage_grid_size = {config.compute_grid_x, config.compute_grid_y}},
        .compute_kernel_config = compute_kernel_config};
}

std::optional<Recipe> select_recipe(const Mode mode, const RegistryRequestFacts& facts) {
    if (mode == Mode::Off) {
        return std::nullopt;
    }
    const auto key = build_registry_key(facts);
    const auto* entry = key ? lookup(*key) : nullptr;
    std::optional<Recipe> recipe;
    if (entry != nullptr) {
        recipe = materialize_recipe(*entry);
    }
    // This is host-side dispatch validation, before device-operation launch.
    TT_FATAL(
        recipe || !ttnn::operations::matmul::registry::fallback_is_error(mode),
        "AGMM registry required an exact recipe, but no exact match was found");
    return recipe;
}

}  // namespace ttnn::experimental::all_gather_minimal_matmul_registry
