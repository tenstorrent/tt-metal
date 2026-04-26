// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/indexed_fill/indexed_fill.hpp"
#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_device_operation.hpp"
#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_utils.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/core/core.hpp"

using namespace tt::tt_metal;

namespace ttnn {

Tensor indexed_fill(
    const Tensor& batch_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config,
    int64_t dim) {
    const auto rank = static_cast<int64_t>(input_tensor_a.logical_shape().rank());

    // Normalise negative dim (matches PyTorch index_copy convention).
    if (dim < 0) {
        dim += rank;
    }
    TT_FATAL(dim >= 0 && dim < rank, "indexed_fill: dim {} is out of bounds for rank {}", dim, rank);

    // Validate ranks before any per-dimension indexing: indexing b_shape[d] below
    // must not run off the end if input_b has a different rank than input_a.
    const auto& a_shape = input_tensor_a.logical_shape();
    const auto& b_shape = input_tensor_b.logical_shape();
    const auto b_rank = static_cast<int64_t>(b_shape.rank());
    TT_FATAL(
        rank == b_rank,
        "indexed_fill: input_a and input_b must have the same rank; got input_a.rank() = {} and input_b.rank() = {}",
        rank,
        b_rank);

    // Validate: all dims except `dim` must match between input_a and input_b.
    for (int64_t d = 0; d < rank; ++d) {
        if (d == dim) {
            continue;
        }
        TT_FATAL(
            a_shape[d] == b_shape[d],
            "indexed_fill: input_a and input_b must have the same size along every "
            "dimension except dim {}; mismatch at dim {}: input_a.size({}) = {}, input_b.size({}) = {}",
            dim,
            d,
            d,
            a_shape[d],
            d,
            b_shape[d]);
    }
    // Validate: number of indices must equal input_b's size along dim.
    TT_FATAL(
        static_cast<uint32_t>(b_shape[dim]) == batch_id.padded_shape()[-1],
        "indexed_fill: input_b.size({}) = {} must equal the number of indices (batch_id count = {})",
        dim,
        b_shape[dim],
        batch_id.padded_shape()[-1]);

    auto output_memory_config = memory_config.value_or(input_tensor_a.memory_config());

    // Native kernel path: the device op handles the 2D stride loop directly, avoiding
    // any permute.  Supported when the target dimension is coarser than the page unit:
    //   ROW_MAJOR: page = last-dim row  → native for dim < rank-1
    //   TILE:      page = 32×32 tile    → native for dim < rank-2
    const Layout layout = input_tensor_a.layout();
    // ROW_MAJOR: dim=rank-1 is sub-page (element within a row); requires permute.
    // TILE: dim=rank-1 and dim=rank-2 are both sub-tile (column/row within the 32×32 tile); require permute.
    const bool native_kernel =
        (layout == Layout::ROW_MAJOR && dim < rank - 1) || (layout == Layout::TILE && dim < rank - 2);

    if (native_kernel) {
        return ttnn::prim::indexed_fill(batch_id, input_tensor_a, input_tensor_b, output_memory_config, dim);
    }

    // Permute fallback for sub-page / sub-tile cases:
    //   ROW_MAJOR dim = rank-1  (element within a row)
    //   TILE      dim >= rank-2 (row or column within a tile)
    // Permute so that the target dimension becomes dim=0, run the primitive, then
    // permute the result back.

    // Build perm = [dim, 0, 1, ..., dim-1, dim+1, ..., rank-1]
    SmallVector<int64_t> perm;
    perm.reserve(rank);
    perm.push_back(dim);
    for (int64_t i = 0; i < rank; ++i) {
        if (i != dim) {
            perm.push_back(i);
        }
    }

    // Inverse permutation: inv_perm[perm[i]] = i.
    SmallVector<int64_t> inv_perm(rank);
    for (int64_t i = 0; i < rank; ++i) {
        inv_perm[perm[i]] = i;
    }

    // Intermediates: keep INTERLEAVED (we don't want to reason about shard-spec
    // compatibility with the transposed shape) but inherit input_a's buffer_type
    // so DRAM-resident inputs stay in DRAM through the wrapper.  Forcing L1 here
    // would regress large/DRAM cases by silently spilling them into L1.  Tensor
    // *layout* (ROW_MAJOR/TILE) is preserved by ttnn::permute itself.
    const MemoryConfig intermediate_mem_config{
        TensorMemoryLayout::INTERLEAVED, input_tensor_a.memory_config().buffer_type()};

    const auto a_perm = ttnn::permute(input_tensor_a, perm, intermediate_mem_config);
    const auto b_perm = ttnn::permute(input_tensor_b, perm, intermediate_mem_config);

    // The device primitive always sees dim=0 after permute.
    const auto out_perm = ttnn::prim::indexed_fill(batch_id, a_perm, b_perm, intermediate_mem_config, 0);

    // If the caller asked for a sharded output but didn't supply an explicit
    // shard_spec, derive one before passing it to the final permute (mirrors the
    // device op's compute_output_specs behavior).  Without this, the final
    // permute can fail when constructing the output tensor for a sharded-but-
    // shard_spec-less MemoryConfig.
    auto resolved_output_mem_config = output_memory_config;
    if (resolved_output_mem_config.is_sharded() && !resolved_output_mem_config.shard_spec().has_value()) {
        using namespace ttnn::operations::data_movement::indexed_fill;
        const auto& padded_out_shape = input_tensor_a.padded_shape();

        // Two-path resolution: adapt the existing shard_spec to the output shape when
        // input_a already carries one; otherwise generate a fresh spec covering all cores.
        auto derive_shard_spec = [&]() -> tt::tt_metal::ShardSpec {
            if (input_tensor_a.is_sharded() && input_tensor_a.memory_config().shard_spec().has_value()) {
                return adjust_to_shape(
                    *input_tensor_a.memory_config().shard_spec(), input_tensor_a.padded_shape(), padded_out_shape);
            }
            return generate_shard_spec_all_cores(
                input_tensor_a, padded_out_shape, resolved_output_mem_config.memory_layout());
        };
        resolved_output_mem_config = MemoryConfig(
            resolved_output_mem_config.memory_layout(), resolved_output_mem_config.buffer_type(), derive_shard_spec());
    }

    // Permute the result back to the original dimension order, placing it in the
    // Output layout matches input_tensor_a.
    return ttnn::permute(out_perm, inv_perm, resolved_output_mem_config);
}

}  // namespace ttnn
