// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "tilize_codegen_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct TilizeCodegenProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const TilizeCodegenParams& operation_attributes,
        const TilizeCodegenInputs& tensor_args,
        Tensor& tensor_return_value);
};

// Which builder create_descriptor() would select for these attributes, exposed so the perf gate
// reasons about the actual program instead of re-deriving the dispatch conditions.
enum class TilizeCodegenPath : uint8_t { RowSingleCore, Row, Column, Block };

struct TilizeCodegenDispatch {
    TilizeCodegenPath path = TilizeCodegenPath::Row;
    // Column path only: the number of column blocks, and the widest of them in tiles (the widths
    // differ by one tile when ncol does not divide Wt).
    uint32_t ncol = 1;
    uint32_t max_tiles_per_column_block = 0;
};

TilizeCodegenDispatch tilize_codegen_dispatch(
    tt::tt_metal::IDevice* device, const TilizeCodegenParams& operation_attributes, const Tensor& input_tensor);

// Whether the selected builder's circular-buffer plan fits per-core L1. Every path's CB footprint
// scales with the per-core tile count, and the reference raises rather than shrinking below the
// compute/writer contract, so the correctness gate has to answer this before `auto` commits: the
// factory can only abort where the gate would have fallen back to native.
bool tilize_codegen_cb_plan_fits(
    tt::tt_metal::IDevice* device, const TilizeCodegenParams& operation_attributes, const Tensor& input_tensor);

}  // namespace ttnn::prim
