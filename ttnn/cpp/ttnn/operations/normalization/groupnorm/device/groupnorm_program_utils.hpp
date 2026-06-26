// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstdint>
#include <optional>

#include <tt-metalium/core_coord.hpp>
#include "ttnn/tensor/tensor.hpp"  // ttnn::Tensor, ttnn::TensorSpec, tt::tt_metal::DataType

namespace tt::tt_metal {
class IDevice;
}

namespace ttnn::prim {

enum class GroupNormMode : uint32_t { LEGACY = 0, WELFORD_NATIVE = 1, WELFORD_RECIPROCALS = 2 };

// Per-core byte sizes of the statically-allocated circular buffers used by the sharded
// group-norm program factory. These are the CBs that grow upward from l1_base; the in0
// (c_0) and output (c_16) CBs are bound to existing tensor buffers and are therefore NOT
// part of this set. Computed once by compute_sharded_gn_static_cb_sizes() and consumed
// BOTH by the program factory (to create the CBs) AND by the op-level L1 sizing helpers,
// so the factory and any size computation can never disagree. To add or resize a static
// CB, change compute_sharded_gn_static_cb_sizes() — the single source of truth.
struct GroupNormShardedStaticCbSizes {
    uint32_t in_CB_size = 0;                // c_1  tilized input (and c_30 untilize-out copy)
    uint32_t in2_CB_size = 0;               // c_2  scaler (and c_4 scaler-c when !welford)
    uint32_t in3_CB_size = 0;               // c_3  eps
    uint32_t in5_CB_size = 0;               // c_5  gamma
    uint32_t in6_CB_size = 0;               // c_6  beta
    uint32_t in_mask_CB_size = 0;           // c_7  input mask
    uint32_t in_negative_mask_CB_size = 0;  // c_14 negative mask
    uint32_t repack_CB_size = 0;            // c_11/c_12 repack
    uint32_t x_CB_size = 0;                 // c_13 x
    uint32_t ex_partial_CB_size = 0;        // c_8  ex_partial
    uint32_t ex_global_CB_size = 0;         // c_9/c_15 ex_global
    uint32_t ex2pe_CB_size = 0;             // c_17 ex2pe
    uint32_t single_tile_size = 0;          // c_10 ex_external and c_26 ones

    // Total per-core L1 occupied by the static CB region, matching exactly what the factory
    // allocates for the given configuration. `with_negative_mask` selects the negative-mask
    // CB (c_14) in place of the untilize-out copy (c_30). The flags mirror the factory's own
    // conditionals so the same CBs are counted here and created there.
    uint32_t total(
        bool with_negative_mask,
        bool untilize_out,
        bool has_gamma,
        bool has_beta,
        bool has_input_mask,
        bool reader_repack_output,
        bool use_welford) const;
};

// Single source of truth for the static-CB sizes above. `input` must be the sharded input
// tensor; gamma/beta dtypes are optional (absent => not present). Assumes the op's fixed
// conventions: bf16 compute data format and bf16 input/negative masks.
GroupNormShardedStaticCbSizes compute_sharded_gn_static_cb_sizes(
    const ttnn::Tensor& input,
    std::optional<tt::tt_metal::DataType> gamma_dtype,
    std::optional<tt::tt_metal::DataType> beta_dtype,
    bool use_welford,
    uint32_t num_groups);

// Exact per-core L1 (bytes) that running sharded group_norm ADDS to the device:
//   (1) the statically-allocated CB region — which INCLUDES the gamma, beta, input-mask and
//       negative-mask CBs (those are scratch CBs the op allocates), and
//   (2) the output buffer, only when it is a fresh L1 allocation (L1 and not in place).
// NOT counted: buffers that already exist before the op runs — the input, the reciprocals LUT,
// and the gamma/beta/input-mask SOURCE tensors (the c_5/c_6/c_7 CBs are counted, but the
// tensors feeding them are not) — nor any other tensor resident elsewhere in L1. No estimates:
// the CB region reuses compute_sharded_gn_static_cb_sizes(), and the output size comes from the
// output spec's own per-bank computation. This does NOT answer "will it fit" — that also depends
// on every other live L1 tensor and on where the allocator places the output (see the dispatch path).
uint32_t compute_sharded_gn_l1_footprint(
    const ttnn::Tensor& input,
    const ttnn::TensorSpec& output_spec,
    const tt::tt_metal::IDevice& device,
    std::optional<tt::tt_metal::DataType> gamma_dtype,
    std::optional<tt::tt_metal::DataType> beta_dtype,
    bool has_input_mask,
    bool with_negative_mask,
    bool use_welford,
    bool inplace,
    uint32_t num_groups);

int get_max_subblock(uint32_t n, uint32_t max_subblock_w);

bool is_rectangle_grid(const std::vector<CoreCoord>& core_coords);

void split_and_form_rectangle_grids(
    std::vector<CoreCoord>& group,
    std::vector<CoreCoord>& mcast_group_first,
    std::vector<CoreCoord>& mcast_group_mid,
    std::vector<CoreCoord>& mcast_group_last);

std::pair<uint32_t, uint32_t> find_max_tile_span(uint32_t W, uint32_t group_size, uint32_t tile_width = 32);

}  // namespace ttnn::prim
