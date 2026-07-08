// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

// Operation attributes - non-tensor parameters. Matches manifests/move.yaml cache_key_fields
// exactly; this struct (plus the tensor specs in MoveCodegenTensorArgs) IS the program-cache key —
// there is no custom hash.
struct MoveCodegenOperationAttributes {
    uint32_t total_pages;    // tiles (TILE) or sticks (ROW_MAJOR) moved, from input volume
    uint32_t page_bytes;     // aligned DRAM/L1 page pitch (tile_size or stick_size, alignment-padded)
    uint32_t read_batch;     // L1-clamped reader batch depth
    uint32_t write_batch;    // L1-clamped writer batch depth
    uint32_t cb_depth;       // circular buffer depth derived from read_batch/write_batch
    tt::tt_metal::MemoryConfig output_mem_config;  // target buffer type for the freshly allocated output tensor
};

// Tensor arguments - tensor parameters
struct MoveCodegenTensorArgs {
    Tensor input_tensor;
    Tensor output_tensor;
};

// Program factory for the codegen-generated move kernels (see move_codegen_program_factory.cpp).
struct MoveCodegenProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const MoveCodegenOperationAttributes& operation_attributes,
        const MoveCodegenTensorArgs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
