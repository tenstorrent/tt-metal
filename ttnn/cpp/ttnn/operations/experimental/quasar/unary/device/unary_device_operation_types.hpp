// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim::qsr {

// Quasar (Metal 2.0) copy of ttnn::operations::unary::UnaryDeviceOperation::operation_attributes_t, minus the
// knobs Quasar has no use for (bfp8_pack_precise: no BFP formats; sub_core_grids: not wired yet). Every field
// is part of the default program hash, which also covers the input/output TensorSpecs (shape included), so a
// cache hit always replays the same per-node work split.
struct UnaryParams {
    std::vector<ttnn::operations::unary::EltwiseUnaryWithParam> op_chain;
    tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::BFLOAT16;
    tt::tt_metal::MemoryConfig output_memory_config;
    bool fp32_dest_acc_en = false;
    bool preserve_fp32_precision = false;
    tt::tt_metal::CoreRangeSet worker_grid;
};

struct UnaryInputs {
    Tensor input;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::prim::qsr
