// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/sub_device_types.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::extract {

struct ExtractParams {
    // Index into global_expert_idx_table. The kernel looks up
    //   global_expert_id = global_expert_idx_table[local_expert_id]
    // at runtime and uses the result to index start / counts.
    uint32_t local_expert_id;
    uint32_t max_dispatched_tokens_per_expert;
    // Optional sub-device to confine the op to. When set, the program runs on that
    // sub-device's worker cores instead of the full compute grid (used to overlap the
    // routed expert with the combine on disjoint cores). std::nullopt => full grid.
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt;

    static constexpr auto attribute_names =
        std::forward_as_tuple("local_expert_id", "max_dispatched_tokens_per_expert", "subdevice_id");

    auto attribute_values() const {
        return std::forward_as_tuple(local_expert_id, max_dispatched_tokens_per_expert, subdevice_id);
    }
};

struct ExtractInputs {
    Tensor global_tensor;
    Tensor start;
    Tensor counts;
    // 1D (or 2D with first dim == 1) UINT32 DRAM-interleaved tensor mapping
    // local_expert_id -> global_expert_id.
    Tensor global_expert_idx_table;
    // Optional preallocated output buffer. When set, create_output_tensors returns it
    // instead of allocating a fresh tensor. Must match the spec from compute_output_specs
    // (shape [.., max_tokens, hidden], global_tensor dtype, TILE, DRAM interleaved).
    std::optional<Tensor> optional_output_tensor = std::nullopt;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::extract
