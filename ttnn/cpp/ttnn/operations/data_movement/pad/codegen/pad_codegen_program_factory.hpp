// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>

#include "pad_codegen_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

// ops/pad/spec.py: READ_BATCH / WRITE_BATCH. TILE always uses these fixed defaults; RM starts
// here and may shrink via rm_pad_batches_for_l1() at the wide-stick L1 cliff.
inline constexpr uint32_t kPadCodegenReadBatchDefault = 8;
inline constexpr uint32_t kPadCodegenWriteBatchDefault = 4;

struct PadCodegenProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PadCodegenParams& operation_attributes, const PadCodegenInputs& tensor_args, Tensor& output_tensor);
};

// Packs one pad word in the output tensor's physical scalar format. Transcribed from
// ops/pad/builder.py's ``_pack_pad_value`` (float32 keeps the exact IEEE-754 bit pattern;
// bfloat16 round-to-nearest-even's it into both halves of the word; int32/uint32 truncate).
// Shared by the program factory (to build ArgsPad's packed_pad_val) and is_demoted() (to
// recover which raw ledger value a cache-hit's packed_pad_value corresponds to).
uint32_t pack_pad_value(tt::tt_metal::DataType dtype, float value);

// RM-only batch shrink for the wide-stick L1 cliff. Transcribed from
// ops/pad/spec.py's ``_rm_pad_batches_for_l1``: halves read_batch/write_batch (larger first)
// until the CB footprint (depth*output_page + output_page + input_page) fits ``budget`` minus
// the L1 safety margin.
std::pair<uint32_t, uint32_t> rm_pad_batches_for_l1(
    uint32_t input_page_bytes, uint32_t output_page_bytes, uint32_t budget, uint32_t read_batch, uint32_t write_batch);

// Builds the cache-key attrs from a 4D-folded pad request (front/back offsets in element
// units, on all four N/C/H/W dims of the already-4D-unsqueezed input). Computes
// packed_pad_value and, for RM, the L1-clamped read_batch/write_batch -- the single place that
// assembles PadCodegenParams, shared by the free function (ttnn::pad) so the codegen prim's
// validate_on_program_cache_miss and is_demoted() see the exact same fields the routing decision
// was made on.
PadCodegenParams build_pad_codegen_params(
    const Tensor& input_4d,
    uint32_t front_n,
    uint32_t front_c,
    uint32_t front_h,
    uint32_t front_w,
    uint32_t back_n,
    uint32_t back_c,
    uint32_t back_h,
    uint32_t back_w,
    float pad_value,
    const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn::prim
