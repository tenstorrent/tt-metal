// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

// Matches ops/pad/spec.py's READ_BATCH / WRITE_BATCH defaults. RM batches may
// be clamped smaller by pad_rm_batches_for_l1() below; TILE always uses these.
inline constexpr uint32_t kPadReadBatch = 8;
inline constexpr uint32_t kPadWriteBatch = 4;

// ops/pad/spec.py's _L1_SAFETY_MARGIN, subtracted from the usable L1 budget
// before the RM batch-clamp loop. Shared with pad_codegen_supported.cpp so
// the routing gate and the factory can never disagree on the clamped batch.
inline constexpr uint32_t kPadL1SafetyMargin = 64 * 1024;

// Reduce RM pad read/write batching until the projected per-core CB footprint
// fits the usable L1 budget. Mirrors ops/pad/spec.py::_rm_pad_batches_for_l1
// exactly (including which of read/write shrinks first). `input_page` is the
// aligned staging-CB page (stage_buf_size); `output_page` is the aligned
// output-CB page (stick_size_out_aligned).
std::pair<uint32_t, uint32_t> pad_rm_batches_for_l1(
    uint32_t input_page,
    uint32_t output_page,
    uint32_t budget,
    uint32_t read_batch = kPadReadBatch,
    uint32_t write_batch = kPadWriteBatch);

// Packs one pad word in the output tensor's physical scalar format, exactly
// transcribing ops/pad/builder.py::_pack_pad_value's per-dtype rules (BF16
// RNE-duplicated into both halves, float32 raw IEEE bits with signed-infinity
// saturation, uint16 saturating round, int32/uint32 raw reinterpret).
uint32_t pack_pad_value(tt::tt_metal::DataType dtype, float value);

struct PadCodegenParams {
    // Output dims: RM stores element counts; TILE stores tile-page counts
    // (Ht_out/Wt_out), matching the manifest's layout_split-driven cache-key
    // convention.
    uint32_t N_out{};
    uint32_t C_out{};
    uint32_t H_out{};  // RM: elements; TILE: Ht_out (tile pages)
    uint32_t W_out{};  // RM: elements; TILE: Wt_out (tile pages)
    // Front offsets, same unit convention as above (RM: elements; TILE: tile
    // pages -- always 0 for TILE since front-pad is scope-rejected there).
    uint32_t front_n{};
    uint32_t front_c{};
    uint32_t front_h{};
    uint32_t front_w{};
    uint32_t packed_pad_value{};
    uint32_t read_batch{kPadReadBatch};
    uint32_t write_batch{kPadWriteBatch};
    tt::tt_metal::MemoryConfig output_mem_config;
    // Needed to build the output TensorSpec (compute_output_specs) without
    // recomputing tile-rounding logic a second time.
    ttnn::Shape output_logical_shape;
    ttnn::Shape output_padded_shape;
};

struct PadCodegenInputs {
    Tensor input;
};

struct PadCodegenProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PadCodegenParams& operation_attributes,
        const PadCodegenInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
