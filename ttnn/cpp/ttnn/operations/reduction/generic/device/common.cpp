// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common.hpp"

#include <numeric>
#include <tuple>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/work_split.hpp>
#include <ttnn/tensor/layout/tensor_layout.hpp>
#include <ttnn/tensor/layout/page_config.hpp>

namespace ttnn::prim {
tt::tt_metal::ReduceOpParallelizationStrategy get_parallelization_strategy(
    const tt::tt_metal::Tensor& input_tensor, tt::tt_metal::ReduceOpDim reduce_dim) {
    uint32_t num_tiles = input_tensor.physical_volume() / input_tensor.tensor_spec().tile().get_tile_hw();
    if (reduce_dim == tt::tt_metal::ReduceOpDim::H) {
        return tt::tt_metal::ReduceOpParallelizationStrategy::MULTI_CORE_H;
    }
    if (reduce_dim == tt::tt_metal::ReduceOpDim::W) {
        return tt::tt_metal::ReduceOpParallelizationStrategy::MULTI_CORE_W;
    }
    if (reduce_dim == tt::tt_metal::ReduceOpDim::HW) {
        if (num_tiles > 1) {
            return tt::tt_metal::ReduceOpParallelizationStrategy::MULTI_CORE_HW;
        }
        return tt::tt_metal::ReduceOpParallelizationStrategy::SINGLE_CORE_HW;
    }
    TT_THROW("Unsupported reduce dim");
}

tt::tt_metal::TensorSpec build_reduce_output_tensor_spec(
    const tt::tt_metal::Shape& output_shape,
    tt::tt_metal::DataType output_dtype,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::MemoryConfig& input_mem_config,
    tt::tt_metal::ReduceOpDim reduce_dim,
    tt::tt_metal::Layout output_layout) {
    using namespace tt::tt_metal;

    TensorSpec tensor_spec(
        output_shape,
        TensorLayout(output_dtype, PageConfig(output_layout), MemoryConfig(output_mem_config.buffer_type())));

    TensorMemoryLayout mem_layout = output_mem_config.memory_layout();

    if (mem_layout == TensorMemoryLayout::WIDTH_SHARDED || mem_layout == TensorMemoryLayout::HEIGHT_SHARDED ||
        mem_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        // Grid and orientation are identical in both spec formats (nd_shard_spec and shard_spec)
        // when both are populated. Pick whichever is available from the output config,
        // falling back to the input tensor's shard spec for backward compatibility.
        const auto& nd = output_mem_config.nd_shard_spec();
        const auto& legacy = output_mem_config.shard_spec();
        const auto& input_nd = input_mem_config.nd_shard_spec();
        const auto& input_legacy = input_mem_config.shard_spec();
        auto get_grid_and_orientation = [&]() -> std::pair<const CoreRangeSet&, ShardOrientation> {
            if (nd) {
                return {nd->grid, nd->orientation};
            }
            if (legacy) {
                return {legacy->grid, legacy->orientation};
            }
            if (input_nd) {
                return {input_nd->grid, input_nd->orientation};
            }
            if (input_legacy) {
                return {input_legacy->grid, input_legacy->orientation};
            }
            TT_THROW(
                "Sharded memory layout {} requires either nd_shard_spec or shard_spec to be set "
                "on the output memory config or the input tensor",
                mem_layout);
        };
        const auto& [grid, orientation] = get_grid_and_orientation();

        // For width/height/block sharding modes, the output shard shape is fully determined
        // by the output physical shape and the core grid. Just delegate to the
        // appropriate TensorSpec builder.
        if (mem_layout == TensorMemoryLayout::WIDTH_SHARDED) {
            return tensor_spec.width_sharded(grid, orientation);
        }
        if (mem_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            return tensor_spec.height_sharded(grid, orientation);
        }
        TT_FATAL(
            grid.ranges().size() == 1,
            "Block sharding requires a single CoreRange, got {} ranges",
            grid.ranges().size());
        return tensor_spec.block_sharded(grid.bounding_box(), orientation);
    }

    // ND sharding: adjust per-logical-dimension shard shape for reduced dims.
    // Fall back to the input tensor's nd_shard_spec when the output config omits it.
    if (mem_layout == TensorMemoryLayout::ND_SHARDED) {
        const auto& nd_shard_spec = output_mem_config.nd_shard_spec();
        const auto& input_nd_shard_spec = input_mem_config.nd_shard_spec();
        TT_FATAL(
            nd_shard_spec.has_value() || input_nd_shard_spec.has_value(),
            "ND_SHARDED memory layout requires nd_shard_spec to be set "
            "on the output memory config or the input tensor");
        auto nd_shard_spec_copy = nd_shard_spec.has_value() ? *nd_shard_spec : *input_nd_shard_spec;
        if (reduce_dim == ReduceOpDim::W || reduce_dim == ReduceOpDim::HW) {
            nd_shard_spec_copy.shard_shape[-1] = 1;
        }
        if ((reduce_dim == ReduceOpDim::H || reduce_dim == ReduceOpDim::HW) &&
            nd_shard_spec_copy.shard_shape.rank() > 1) {
            nd_shard_spec_copy.shard_shape[-2] = 1;
        }
        return tensor_spec.sharded(std::move(nd_shard_spec_copy), TensorSpec::ShardShapeAlignment::REQUIRED);
    }

    // Guard against unexpected new memory layouts.
    TT_FATAL(mem_layout == TensorMemoryLayout::INTERLEAVED, "Unexpected memory layout: {}", mem_layout);
    // Interleaved tensor: tensor_spec already has everything we need.
    return tensor_spec;
}

void validate_reduce_sharded_buffer_types(
    const tt::tt_metal::MemoryConfig& input_mem_config,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    std::string_view op_name) {
    TT_FATAL(
        !output_mem_config.is_sharded() || output_mem_config.is_l1(),
        "{}: sharded output memory layout {} is only supported with L1 buffers, got buffer type {}",
        op_name,
        output_mem_config.memory_layout(),
        output_mem_config.buffer_type());
    TT_FATAL(
        !input_mem_config.is_sharded() || input_mem_config.is_l1(),
        "{}: sharded input memory layout {} is only supported with L1 buffers, got buffer type {}",
        op_name,
        input_mem_config.memory_layout(),
        input_mem_config.buffer_type());
}

bool h_reduce_negate_fits_in_l1(
    const tt::tt_metal::Tensor& input_tensor, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    using namespace tt::tt_metal;

    const auto& shape = input_tensor.padded_shape();
    const uint32_t tile_height = input_tensor.tensor_spec().tile().get_height();
    const uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();
    if (tile_height == 0 || tile_width == 0) {
        return true;
    }

    const uint32_t W = shape[3];
    const uint32_t H = shape[2];
    const uint32_t NC = shape[1] * shape[0];
    const uint32_t Wt = W / tile_width;
    const uint32_t Ht = H / tile_height;

    auto* device = input_tensor.device();
    const bool use_width_sharding = input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;

    uint32_t num_cols_per_core_group_1 = 0;
    uint32_t num_cols_per_core_group_2 = 0;
    if (use_width_sharding) {
        if (NC == 0 || !input_tensor.shard_spec().has_value()) {
            return true;
        }
        num_cols_per_core_group_1 = NC * (input_tensor.shard_spec().value().shape[1] / tile_width);
    } else {
        const uint32_t num_cols = NC * Wt;
        if (num_cols == 0) {
            return true;
        }
        const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        if (sub_core_grids.has_value()) {
            std::tie(
                std::ignore,
                std::ignore,
                std::ignore,
                std::ignore,
                num_cols_per_core_group_1,
                num_cols_per_core_group_2) = tt::tt_metal::split_work_to_cores(*sub_core_grids, num_cols);
        } else {
            std::tie(
                std::ignore,
                std::ignore,
                std::ignore,
                std::ignore,
                num_cols_per_core_group_1,
                num_cols_per_core_group_2) =
                tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_cols);
        }
    }

    // Match the kernel's per-core compute_Wt: width-sharded cores see shard_Wt,
    // non-sharded cores see num_cols_per_core directly (NC=1 in the kernel).
    const uint32_t compute_Wt_g1 =
        use_width_sharding ? (NC == 0 ? 0 : num_cols_per_core_group_1 / NC) : num_cols_per_core_group_1;
    const uint32_t compute_Wt_g2 = use_width_sharding ? 0 : num_cols_per_core_group_2;

    uint32_t per_nc_advance;
    if (compute_Wt_g1 == 0 && compute_Wt_g2 == 0) {
        return true;
    }
    if (compute_Wt_g2 == 0) {
        per_nc_advance = compute_Wt_g1;
    } else if (compute_Wt_g1 == 0) {
        per_nc_advance = compute_Wt_g2;
    } else {
        per_nc_advance = std::lcm(compute_Wt_g1, compute_Wt_g2);
    }
    const uint64_t negate_cb_tiles = static_cast<uint64_t>(Ht) * per_nc_advance;
    if (negate_cb_tiles == 0) {
        return true;
    }

    const tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);
    // Two CBs (cb_acc=c_4 and cb_ineg=c_5) are each sized at negate_cb_tiles.
    const uint64_t negate_cb_bytes = 2ull * negate_cb_tiles * dst_single_tile_size;

    const auto lowest_address = device->lowest_occupied_compute_l1_address();
    uint64_t max_l1_space = lowest_address.has_value() ? lowest_address.value() : device->l1_size_per_core();
    const uint64_t base_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    if (max_l1_space <= base_addr) {
        return false;
    }
    max_l1_space -= base_addr;

    return negate_cb_bytes <= max_l1_space;
}

}  // namespace ttnn::prim
