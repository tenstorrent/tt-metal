// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <numeric>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct ConcatParams {
    uint32_t dim;
    unsigned int groups;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<ttnn::CoreRangeSet> sub_core_grids;
};

struct ConcatInputs {
    std::vector<Tensor> input_tensors;
};

// ConcatParams::dim is already normalized, so "is this the height/width dim?" has to be asked
// rank-relative: at rank 3 the width dim is 2, at rank 5 it is 4. Asking with the absolute
// literals 2 and 3 is what #55032 was. Every sharded factory needs the same answer, so they all
// ask here rather than each rederiving it.
//
// Phrased as dim + N == rank so a rank-1 input cannot wrap the unsigned subtraction. Width is
// meaningful at rank 1 (dim 0 is the last dim); height is not, hence the guard.
constexpr bool is_width_concat(uint32_t rank, uint32_t dim) { return dim + 1 == rank; }
constexpr bool is_height_concat(uint32_t rank, uint32_t dim) { return rank >= 2 && dim + 2 == rank; }

// A sharded tensor of shape (D0, ..., Dn-1) is laid out as the 2D flattening
// (prod(D0..Dn-2), Dn-1), so its shard rows carry the leading dims folded in. A height concat on
// dim == rank-2 therefore has to interleave: the output rows for leading index b are input 0's
// rows at b, then input 1's rows at b, and so on. This returns the number of those leading
// indices -- the "blocks" the copy has to be split into.
//
// It is 1 exactly when every dim before rank-2 is 1, the usual rank-4 model case (1, 1, H, W).
// At one block, appending whole shards and interleaving coincide, so that case cannot distinguish
// them (#55342).
//
// All inputs agree on this value: they differ only in the concat dim, which is not a leading dim.
inline uint32_t num_leading_blocks(const Tensor& tensor) {
    const auto& padded_shape = tensor.padded_shape();
    // Guarded so cend() - 2 cannot walk past the front; rank 2 gives an empty range and 1.
    if (padded_shape.rank() < 2) {
        return 1;
    }
    return std::accumulate(padded_shape.cbegin(), padded_shape.cend() - 2, 1u, std::multiplies<uint32_t>{});
}

}  // namespace ttnn::prim
