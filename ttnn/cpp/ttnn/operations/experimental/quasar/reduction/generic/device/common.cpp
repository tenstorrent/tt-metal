// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common.hpp"

#include <algorithm>
#include <numeric>
#include <tuple>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <ttnn/tensor/layout/tensor_layout.hpp>
#include <ttnn/tensor/layout/page_config.hpp>

namespace ttnn::prim::qsr {
RmPlan make_rm_plan(
    const tt::tt_metal::Shape& padded_shape,
    const tt::tt_metal::Shape& logical_shape,
    uint32_t tile_height,
    uint32_t tile_width,
    tt::DataFormat src_cb_data_format,
    tt::DataFormat dst_cb_data_format,
    tt::tt_metal::ReduceOpMath math_op,
    tt::tt_metal::ReduceOpDim dim) {
    RmPlan plan{};
    plan.H_logical = logical_shape[2];
    plan.W_logical = logical_shape[3];
    plan.rm_rows_per_tile = tile_height;
    plan.Wt = tt::div_up(padded_shape[3], tile_width);
    plan.Ht_rm = tt::div_up(plan.H_logical, plan.rm_rows_per_tile);

    // Only supports ReduceOpDim::W or ReduceOpDim::H.
    //
    // k_rm_max_tiles_per_chunk caps the reduction-axis chunk size (wt_tiles_per_chunk for W
    // reduce, ht_tiles_per_chunk for H reduce). It's an L1 staging-buffer budget on the reduction
    // axis. 8 was picked experimentally — the staging CB page lands at ~32 KB for bf16 and
    // ~64 KB for fp32 at chunk=8, which fits L1 comfortably alongside the other CBs. Tune later
    // if a different perf / L1-utilization trade-off is needed.
    constexpr uint32_t k_rm_max_tiles_per_chunk = 8;
    if (dim == tt::tt_metal::ReduceOpDim::W) {
        plan.wt_tiles_per_chunk = std::clamp(plan.Wt, 1u, k_rm_max_tiles_per_chunk);
        plan.ht_tiles_per_chunk = 1;
    } else {
        plan.wt_tiles_per_chunk = 1;
        plan.ht_tiles_per_chunk = std::clamp(plan.Ht_rm, 1u, k_rm_max_tiles_per_chunk);
    }

    // The RM dense path is gated to BF16/FP32 at validate_rm_preconditions;
    // so the unpacked-format byte sizes are always well-defined here.
    plan.src_datum_size = tt::datum_size(src_cb_data_format);
    plan.dst_datum_size = tt::datum_size(dst_cb_data_format);
    plan.chunk_row_bytes = plan.wt_tiles_per_chunk * tile_width * plan.src_datum_size;
    // One CB page = one logical RM row (chunk-wide). The compute kernel uses
    // compute_kernel_lib::tilize, whose asymmetric mode requires one input page per row so each
    // tile-block consumes up to TILE_HEIGHT pages.
    plan.rm_staging_page_size = plan.chunk_row_bytes;
    plan.padding_identity_bits = dense_rm_padding_identity_bits(src_cb_data_format, math_op);

    return plan;
}

void validate_rm_preconditions(
    const tt::tt_metal::MeshTensor& input,
    const tt::tt_metal::MeshTensor& output,
    tt::tt_metal::ReduceOpMath math_op,
    bool negate,
    tt::tt_metal::ReduceOpDim dim,
    std::string_view dim_label) {
    TT_FATAL(
        dim == tt::tt_metal::ReduceOpDim::W || dim == tt::tt_metal::ReduceOpDim::H,
        "{} RM path only supports ReduceOpDim::W or ReduceOpDim::H, got {}",
        dim_label,
        static_cast<int>(dim));
    TT_FATAL(
        input.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
            output.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "{} RM path only supports interleaved tensors (input layout {}, output layout {})",
        dim_label,
        static_cast<int>(input.memory_config().memory_layout()),
        static_cast<int>(output.memory_config().memory_layout()));
    TT_FATAL(
        math_op == tt::tt_metal::ReduceOpMath::SUM,
        "{} RM path only supports SUM (mean lowered from AVG), got {}",
        dim_label,
        math_op);
    TT_FATAL(!negate, "{} RM path does not currently support 'negate'", dim_label);
}

std::vector<uint32_t> build_rm_reader_ct_args(
    const RmPlan& plan, uint32_t scaler_bits, const tt::tt_metal::MeshTensor& src, tt::tt_metal::ReduceOpDim dim) {
    // Slots 0-7 are shared by both paths. The reader's REDUCE_COL (H) branch additionally consumes
    // H_logical at slot 8; the W path omits it, so the source TensorAccessor args follow at slot 8 (W)
    // or slot 9 (H). The kernel is templated on REDUCE_DIM so the unused slot is genuinely dropped.
    // Only supports ReduceOpDim::W or ReduceOpDim::H
    std::vector<uint32_t> args = {
        scaler_bits,
        plan.W_logical,
        plan.src_datum_size,
        plan.padding_identity_bits,
        plan.Wt,
        plan.wt_tiles_per_chunk,
        plan.rm_rows_per_tile,
        plan.ht_tiles_per_chunk,
    };
    if (dim == tt::tt_metal::ReduceOpDim::H) {
        args.push_back(plan.H_logical);
    }
    tt::tt_metal::TensorAccessorArgs(src).append_to(args);
    return args;
}

std::vector<uint32_t> build_rm_writer_ct_args(
    const RmPlan& plan, const tt::tt_metal::MeshTensor& dst, tt::tt_metal::ReduceOpDim dim) {
    // Slot 0 (datum_bytes) is shared. The writer's REDUCE_COL (H) branch additionally consumes
    // Wt, W_logical, and wt_tiles_per_chunk at slots 1-3; the W path omits them, so the dst
    // TensorAccessor args follow at slot 1 (W) or slot 4 (H). The kernel is templated on REDUCE_DIM
    // so the unused slots are genuinely dropped.
    // Only supports ReduceOpDim::W or ReduceOpDim::H
    std::vector<uint32_t> args = {
        plan.dst_datum_size,
    };
    if (dim == tt::tt_metal::ReduceOpDim::H) {
        args.push_back(plan.Wt);
        args.push_back(plan.W_logical);
        args.push_back(plan.wt_tiles_per_chunk);
    }
    tt::tt_metal::TensorAccessorArgs(dst).append_to(args);
    return args;
}

std::vector<uint32_t> build_rm_compute_ct_args(const RmPlan& plan, uint32_t Ht_arg, uint32_t post_mul_scaler_bits) {
    return {
        Ht_arg,
        plan.Wt,
        1u,  // NC (kept literal-1 per the existing RM compute contract; not hoisted into the plan)
        post_mul_scaler_bits,
        plan.wt_tiles_per_chunk,
        plan.ht_tiles_per_chunk,
    };
}

tt::tt_metal::ReduceOpParallelizationStrategy get_parallelization_strategy(
    const ttnn::Tensor& input_tensor, tt::tt_metal::ReduceOpDim reduce_dim) {
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

    tt::tt_metal::TensorSpec tensor_spec(
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
        // appropriate tt::tt_metal::TensorSpec builder.
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
        return tensor_spec.sharded(
            std::move(nd_shard_spec_copy), tt::tt_metal::TensorSpec::ShardShapeAlignment::REQUIRED);
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
    const ttnn::Tensor& input_tensor, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    using namespace tt::tt_metal;

    // The quasar Metal 2.0 ReduceMultiCoreH factory does not implement a fused in-kernel negate path
    // (the negative_tile LLK it would need is unported on Quasar, and the negate compute kernels were
    // removed). Force the host to take the external-negate fallback instead — min(x) == -reduce(MAX, H,
    // -x) computed via the regular (negate=false) reduce kernel, which IS on Metal 2.0 — so a MIN
    // H-reduce never reaches the (rejected) fused-negate path. Keep this `false` while negative_tile
    // stays unported.
    constexpr bool kFusedHNegateSupportedOnMetal2 = false;
    if (!kFusedHNegateSupportedOnMetal2) {
        return false;
    }

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

}  // namespace ttnn::prim::qsr
