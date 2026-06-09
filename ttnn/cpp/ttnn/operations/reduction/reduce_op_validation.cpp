// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_op_validation.hpp"

#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::prim {

using namespace tt::tt_metal;

namespace {

void validate_sharded_tensor(
    std::string_view op_name,
    std::string_view tensor_label,
    std::string_view shard_category,
    uint32_t shard_height,
    uint32_t shard_width,
    uint32_t tile_height,
    uint32_t tile_width) {
    TT_FATAL(
        shard_height > 0 && shard_width > 0,
        "{} {} {} dimensions must be positive: height={}, width={}",
        op_name,
        tensor_label,
        shard_category,
        shard_height,
        shard_width);
    TT_FATAL(
        shard_height % tile_height == 0,
        "{} {} {} height={} must be tile-height-aligned ({})",
        op_name,
        tensor_label,
        shard_category,
        shard_height,
        tile_height);
    TT_FATAL(
        shard_width % tile_width == 0,
        "{} {} {} width={} must be tile-width-aligned ({})",
        op_name,
        tensor_label,
        shard_category,
        shard_width,
        tile_width);
}

void validate_core_grids_within_device_grid(
    const Tensor& tensor_ref, std::string_view op_name, const ReduceOpDeviceGridValidationOptions& opts) {
    const auto device_grid_size = tensor_ref.device()->compute_with_storage_grid_size();
    TT_FATAL(
        device_grid_size.x > 0 && device_grid_size.y > 0,
        "{} requires non-empty device compute grid, got ({}, {})",
        op_name,
        device_grid_size.x,
        device_grid_size.y);

    const bool memory_has_shard_grids = opts.shard_grid_contained_in_device_grid != nullptr &&
                                        (opts.shard_grid_contained_in_device_grid->shard_spec().has_value() ||
                                         opts.shard_grid_contained_in_device_grid->nd_shard_spec().has_value());

    const bool needs_full_device_grid = opts.sub_grid_contained_in_device_grid != nullptr || memory_has_shard_grids;

    if (!needs_full_device_grid) {
        return;
    }

    const CoreRangeSet device_grid = num_cores_to_corerangeset(
        device_grid_size.x * device_grid_size.y, device_grid_size, opts.num_cores_use_last_core_divider);

    if (opts.sub_grid_contained_in_device_grid != nullptr) {
        TT_FATAL(
            device_grid.contains(*opts.sub_grid_contained_in_device_grid),
            "{} {} {} must be contained in device compute grid {}",
            op_name,
            opts.sub_grid_label,
            *opts.sub_grid_contained_in_device_grid,
            device_grid);
    }

    if (memory_has_shard_grids) {
        const auto& memory_config = *opts.shard_grid_contained_in_device_grid;
        auto validate_shard_grid = [&](const CoreRangeSet& shard_grid, bool is_nd) {
            const char* nd_word = is_nd ? "ND " : "";
            TT_FATAL(
                device_grid.contains(shard_grid),
                "{} {} {}shard grid {} must be contained in device compute grid {}",
                op_name,
                opts.memory_config_label,
                nd_word,
                shard_grid,
                device_grid);
        };
        if (memory_config.shard_spec().has_value()) {
            validate_shard_grid(memory_config.shard_spec().value().grid, false);
        }
        if (memory_config.nd_shard_spec().has_value()) {
            validate_shard_grid(memory_config.nd_shard_spec().value().grid, true);
        }
    }
}

}  // namespace

void validate_reduce_op_program_grid(
    std::string_view op_name,
    const CoreRangeSet& all_cores,
    const CoreCoord& device_compute_grid_size,
    const CoreRangeSet* sub_core_grids,
    bool num_cores_use_last_core_divider,
    std::initializer_list<ReduceOpProgramGridShardedTensor> sharded_tensors) {
    TT_FATAL(
        device_compute_grid_size.x > 0 && device_compute_grid_size.y > 0,
        "{} requires non-empty device compute grid, got ({}, {})",
        op_name,
        device_compute_grid_size.x,
        device_compute_grid_size.y);

    const CoreRangeSet device_grid = num_cores_to_corerangeset(
        device_compute_grid_size.x * device_compute_grid_size.y,
        device_compute_grid_size,
        num_cores_use_last_core_divider);

    TT_FATAL(
        device_grid.contains(all_cores),
        "{} program core grid {} must be contained in device compute grid {}",
        op_name,
        all_cores,
        device_grid);

    if (sub_core_grids != nullptr) {
        TT_FATAL(
            sub_core_grids->contains(all_cores),
            "{} program core grid {} must be contained in sub_core_grids {}",
            op_name,
            all_cores,
            *sub_core_grids);
    }

    for (const auto& sharded_tensor : sharded_tensors) {
        if (sharded_tensor.tensor == nullptr) {
            continue;
        }
        const auto& memory_config = sharded_tensor.tensor->memory_config();
        if (memory_config.shard_spec().has_value()) {
            const auto& shard_grid = memory_config.shard_spec().value().grid;
            TT_FATAL(
                all_cores.contains(shard_grid),
                "{} {} shard grid {} must be contained in program core grid {}",
                op_name,
                sharded_tensor.label,
                shard_grid,
                all_cores);
        }
        if (memory_config.nd_shard_spec().has_value()) {
            const auto& nd_shard_grid = memory_config.nd_shard_spec().value().grid;
            TT_FATAL(
                all_cores.contains(nd_shard_grid),
                "{} {} ND shard grid {} must be contained in program core grid {}",
                op_name,
                sharded_tensor.label,
                nd_shard_grid,
                all_cores);
        }
    }
}

void validate_reduce_op_tensor(
    const Tensor& tensor_ref,
    std::string_view op_name,
    std::string_view tensor_label,
    const ReduceOpDeviceGridValidationOptions* core_grids_within_device_grid,
    std::optional<TensorSpec> tensor_spec_ref) {
    const bool sharded_tensor_validation = tensor_spec_ref.has_value() || tensor_ref.is_sharded();
    if (sharded_tensor_validation) {
        if (tensor_spec_ref.has_value()) {
            const TensorSpec& tensor_spec = tensor_spec_ref.value();
            const MemoryConfig& memory_config = tensor_spec.memory_config();
            if (memory_config.nd_shard_spec().has_value()) {
                const auto& nd_shard_shape = memory_config.nd_shard_spec().value().shard_shape;
                const uint32_t tile_height = tensor_spec.tile().get_height();
                const uint32_t tile_width = tensor_spec.tile().get_width();
                if (nd_shard_shape.rank() >= 2) {
                    validate_sharded_tensor(
                        op_name,
                        tensor_label,
                        "ND shard",
                        nd_shard_shape[-2],
                        nd_shard_shape[-1],
                        tile_height,
                        tile_width);
                }
            }
        } else {
            const MemoryConfig& memory_config = tensor_ref.memory_config();
            if (memory_config.shard_spec().has_value()) {
                const auto& shard_shape = memory_config.shard_spec().value().shape;
                const uint32_t tile_height = tensor_ref.tensor_spec().tile().get_height();
                const uint32_t tile_width = tensor_ref.tensor_spec().tile().get_width();
                validate_sharded_tensor(
                    op_name, tensor_label, "shard", shard_shape[0], shard_shape[1], tile_height, tile_width);
            }
        }
    }

    if (core_grids_within_device_grid != nullptr) {
        validate_core_grids_within_device_grid(tensor_ref, op_name, *core_grids_within_device_grid);
    }
}

}  // namespace ttnn::prim
