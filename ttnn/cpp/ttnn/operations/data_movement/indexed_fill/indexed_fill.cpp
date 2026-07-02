// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/indexed_fill/indexed_fill.hpp"
#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_device_operation.hpp"
#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_utils.hpp"
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

    if (dim < 0) {
        dim += rank;
    }
    TT_FATAL(dim >= 0 && dim < rank, "indexed_fill: dim {} is out of bounds for rank {}", dim, rank);

    // The checks below intentionally mirror parts of the device op's
    // validate_on_program_cache_miss().  They are kept here to:
    //   1. Report errors against the original (logical) shapes/dim, not the
    //      post-permute coordinate frame.
    //   2. Fail-fast before the expensive permute call.
    // Do NOT remove them; the device op's checks are not a substitute.
    //
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
    const bool native_kernel =
        (layout == Layout::ROW_MAJOR && dim < rank - 1) || (layout == Layout::TILE && dim < rank - 2);

    if (native_kernel) {
        return ttnn::prim::indexed_fill(batch_id, input_tensor_a, input_tensor_b, output_memory_config, dim);
    }

    // Permute fallback: bring target dim to front, run the primitive at dim=0, permute back.
    // Build perm = [dim, 0, 1, ..., dim-1, dim+1, ..., rank-1]
    ttsl::SmallVector<int64_t> perm;
    perm.reserve(rank);
    perm.push_back(dim);
    for (int64_t i = 0; i < rank; ++i) {
        if (i != dim) {
            perm.push_back(i);
        }
    }

    ttsl::SmallVector<int64_t> inv_perm(rank);
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

    // TODO(nuked-op permute): restore real call
    const auto a_perm = input_tensor_a;
    // TODO(nuked-op permute): restore real call
    const auto b_perm = input_tensor_b;

    // The device primitive always sees dim=0 after permute.
    const auto out_perm = ttnn::prim::indexed_fill(batch_id, a_perm, b_perm, intermediate_mem_config, 0);

    // If the caller asked for a sharded output but didn't supply an explicit
    // shard_spec, derive one before passing it to the final permute (mirrors the
    // device op's compute_output_specs behavior).  Without this, the final
    // permute can fail when constructing the output tensor for a sharded-but-
    // shard_spec-less MemoryConfig.
    const auto resolved_output_mem_config = ttnn::operations::data_movement::indexed_fill::resolve_output_memory_config(
        input_tensor_a, input_tensor_a.padded_shape(), output_memory_config);

    // TODO(nuked-op permute): restore real call
    return out_perm;
}

}  // namespace ttnn
