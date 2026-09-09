// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

// Fixed NABATCH used by the RM stick reader's non-aligned scratch sizing
// (reader_stick_interleaved_unified.cpp), matching the generator's `_NABATCH`.
inline constexpr uint32_t kReshapeNabatch = 8;

// Max single-transaction stick-merge size (bytes) used by merge_num_sticks_to_read
// on the coalescing RM path, matching the generator's `_MAX_READ_SIZE`.
inline constexpr uint32_t kReshapeMaxReadSize = 2048;

// Pipelined-write batch used by the TILE writer (writer_interleaved.cpp),
// matching the generator's `write_batch` default.
inline constexpr uint32_t kReshapeTileWriteBatch = 8;

// Host-computed parameters that fully determine one reshape_codegen program.
// Populated by the routing layer from the input/output tensors and consulted
// by both the program factory and supported.cpp's L1 gates.
struct ReshapeCodegenParams {
    // The output shapes are attributes, not tensor args: the device-operation
    // inspector (tt::tt_metal::experimental::inspector) recursively visits
    // tensor_args to count Tensors, and its reflection cannot descend through
    // ttnn::Shape -- it reaches the shape's span<const uint32_t> view and
    // throws "Unsupported visit of object of type". Attributes are never
    // visited by that path. This mirrors the native reshape op's
    // ReshapeOnDeviceParams, which carries the same two shapes.
    ttnn::Shape output_logical_shape;
    ttnn::Shape output_padded_shape;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct ReshapeCodegenInputs {
    Tensor input;
    std::optional<Tensor> optional_output_tensor;
};

// ROW_MAJOR stick-transport strategy (main path only; the two degenerate
// gather/scatter fast paths are excluded from this port's scope, see
// reshape_codegen_supported.cpp).
struct ReshapeCodegenRmProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ReshapeCodegenParams& operation_attributes,
        const ReshapeCodegenInputs& tensor_args,
        Tensor& tensor_return_value);
};

// TILE untilize/tilize compute strategy.
struct ReshapeCodegenTileProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ReshapeCodegenParams& operation_attributes,
        const ReshapeCodegenInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
