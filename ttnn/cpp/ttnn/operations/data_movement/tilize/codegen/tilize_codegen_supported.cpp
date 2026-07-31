// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_codegen_supported.hpp"

#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt_stl/assert.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

bool is_supported_dtype(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16:
        case DataType::FLOAT32:
        case DataType::UINT32:
        case DataType::INT32:
        case DataType::UINT16: return true;
        default: return false;
    }
}

// One exact-match branch per ledger entry (ungeneralized — no mechanism was identified that
// generalizes these into a predicate over normalized attrs). Each entry's fourth field is the
// sweep's `output_memory_config` (the only memory_config the sweep's vector_map maps into the
// case signature); dtype is the same for input and output because this port never routes a
// dtype-cast call to codegen (see supported_by_codegen).
struct DemotedCase {
    std::vector<uint32_t> shape;
    BufferType buffer_type;
    DataType dtype;
};

const std::vector<DemotedCase>& demoted_cases() {
    static const std::vector<DemotedCase> cases = {
        {{1, 1, 64, 64}, BufferType::DRAM, DataType::INT32},
        {{1, 1, 64, 64}, BufferType::DRAM, DataType::UINT16},
        {{1, 1, 64, 64}, BufferType::DRAM, DataType::UINT32},
        {{1, 1, 64, 64}, BufferType::L1, DataType::INT32},
        {{1, 1, 64, 64}, BufferType::L1, DataType::UINT16},
        {{1, 1, 64, 64}, BufferType::L1, DataType::UINT32},
        {{1, 10, 64, 64}, BufferType::DRAM, DataType::BFLOAT16},
        {{1, 10, 64, 64}, BufferType::L1, DataType::BFLOAT16},
        {{1, 3, 64, 64}, BufferType::DRAM, DataType::FLOAT32},
        {{1, 3, 64, 64}, BufferType::L1, DataType::FLOAT32},
        {{1, 32, 64}, BufferType::DRAM, DataType::INT32},
        {{1, 32, 64}, BufferType::DRAM, DataType::UINT16},
        {{1, 32, 64}, BufferType::DRAM, DataType::UINT32},
        {{1, 32, 64}, BufferType::L1, DataType::INT32},
        {{1, 32, 64}, BufferType::L1, DataType::UINT16},
        {{1, 32, 64}, BufferType::L1, DataType::UINT32},
        {{1, 4, 96, 32}, BufferType::DRAM, DataType::BFLOAT16},
        {{1, 4, 96, 32}, BufferType::L1, DataType::BFLOAT16},
        {{1, 96, 64}, BufferType::DRAM, DataType::FLOAT32},
        {{1, 96, 64}, BufferType::L1, DataType::FLOAT32},
        {{12, 32, 160}, BufferType::L1, DataType::BFLOAT16},
        {{2, 1, 96, 32}, BufferType::DRAM, DataType::INT32},
        {{2, 1, 96, 32}, BufferType::DRAM, DataType::UINT16},
        {{2, 1, 96, 32}, BufferType::DRAM, DataType::UINT32},
        {{2, 1, 96, 32}, BufferType::L1, DataType::INT32},
        {{2, 1, 96, 32}, BufferType::L1, DataType::UINT16},
        {{2, 1, 96, 32}, BufferType::L1, DataType::UINT32},
        {{2, 12, 64, 96}, BufferType::DRAM, DataType::BFLOAT16},
        {{2, 12, 64, 96}, BufferType::L1, DataType::BFLOAT16},
        {{2, 32, 32}, BufferType::DRAM, DataType::INT32},
        {{2, 32, 32}, BufferType::DRAM, DataType::UINT16},
        {{2, 32, 32}, BufferType::DRAM, DataType::UINT32},
        {{2, 32, 32}, BufferType::L1, DataType::INT32},
        {{2, 32, 32}, BufferType::L1, DataType::UINT16},
        {{2, 32, 32}, BufferType::L1, DataType::UINT32},
        {{2, 96, 32}, BufferType::DRAM, DataType::FLOAT32},
        {{2, 96, 32}, BufferType::L1, DataType::FLOAT32},
        {{224, 32}, BufferType::DRAM, DataType::BFLOAT16},
        {{224, 32}, BufferType::L1, DataType::BFLOAT16},
        {{224, 64}, BufferType::DRAM, DataType::BFLOAT16},
        {{224, 64}, BufferType::L1, DataType::BFLOAT16},
        {{3, 2, 128, 32}, BufferType::DRAM, DataType::FLOAT32},
        {{3, 2, 128, 32}, BufferType::L1, DataType::FLOAT32},
        {{3, 2, 64, 32}, BufferType::DRAM, DataType::FLOAT32},
        {{3, 2, 64, 32}, BufferType::L1, DataType::FLOAT32},
        {{3, 2, 96, 32}, BufferType::DRAM, DataType::FLOAT32},
        {{3, 2, 96, 32}, BufferType::L1, DataType::FLOAT32},
        {{3, 7, 64, 96}, BufferType::DRAM, DataType::BFLOAT16},
        {{3, 7, 64, 96}, BufferType::L1, DataType::BFLOAT16},
        {{3, 8, 96, 32}, BufferType::DRAM, DataType::BFLOAT16},
        {{3, 8, 96, 32}, BufferType::L1, DataType::BFLOAT16},
        {{3, 96, 32}, BufferType::DRAM, DataType::FLOAT32},
        {{3, 96, 32}, BufferType::L1, DataType::FLOAT32},
        {{32, 32}, BufferType::DRAM, DataType::INT32},
        {{32, 32}, BufferType::DRAM, DataType::UINT16},
        {{32, 32}, BufferType::DRAM, DataType::UINT32},
        {{32, 32}, BufferType::L1, DataType::INT32},
        {{32, 32}, BufferType::L1, DataType::UINT16},
        {{32, 32}, BufferType::L1, DataType::UINT32},
        {{32, 64}, BufferType::DRAM, DataType::INT32},
        {{32, 64}, BufferType::DRAM, DataType::UINT16},
        {{32, 64}, BufferType::DRAM, DataType::UINT32},
        {{32, 64}, BufferType::L1, DataType::INT32},
        {{32, 64}, BufferType::L1, DataType::UINT16},
        {{32, 64}, BufferType::L1, DataType::UINT32},
        {{4, 12, 96, 96}, BufferType::DRAM, DataType::BFLOAT16},
        {{4, 12, 96, 96}, BufferType::L1, DataType::BFLOAT16},
        {{4, 224, 160}, BufferType::DRAM, DataType::BFLOAT16},
        {{4, 224, 160}, BufferType::L1, DataType::BFLOAT16},
        {{4, 224, 64}, BufferType::DRAM, DataType::BFLOAT16},
        {{4, 224, 64}, BufferType::L1, DataType::BFLOAT16},
        {{4, 4, 32, 64}, BufferType::DRAM, DataType::FLOAT32},
        {{4, 4, 32, 64}, BufferType::L1, DataType::FLOAT32},
        {{4, 4, 64, 64}, BufferType::L1, DataType::FLOAT32},
        {{4, 6, 96, 32}, BufferType::DRAM, DataType::BFLOAT16},
        {{4, 6, 96, 32}, BufferType::L1, DataType::BFLOAT16},
        {{4, 7, 32, 64}, BufferType::DRAM, DataType::BFLOAT16},
        {{4, 7, 32, 64}, BufferType::L1, DataType::BFLOAT16},
        {{4, 9, 64, 32}, BufferType::DRAM, DataType::BFLOAT16},
        {{4, 9, 64, 32}, BufferType::L1, DataType::BFLOAT16},
        {{4, 96, 64}, BufferType::DRAM, DataType::BFLOAT16},
        {{4, 96, 64}, BufferType::L1, DataType::BFLOAT16},
        {{5, 128, 32}, BufferType::DRAM, DataType::BFLOAT16},
        {{5, 128, 32}, BufferType::L1, DataType::BFLOAT16},
        {{5, 160, 96}, BufferType::DRAM, DataType::BFLOAT16},
        {{5, 160, 96}, BufferType::L1, DataType::BFLOAT16},
        {{5, 3, 64, 32}, BufferType::DRAM, DataType::BFLOAT16},
        {{5, 3, 64, 32}, BufferType::L1, DataType::BFLOAT16},
        {{5, 8, 64, 64}, BufferType::L1, DataType::BFLOAT16},
        {{6, 10, 32, 64}, BufferType::DRAM, DataType::BFLOAT16},
        {{6, 10, 32, 64}, BufferType::L1, DataType::BFLOAT16},
        {{6, 4, 96, 64}, BufferType::L1, DataType::BFLOAT16},
        {{64, 32}, BufferType::DRAM, DataType::BFLOAT16},
        {{64, 32}, BufferType::DRAM, DataType::INT32},
        {{64, 32}, BufferType::DRAM, DataType::UINT16},
        {{64, 32}, BufferType::DRAM, DataType::UINT32},
        {{64, 32}, BufferType::L1, DataType::BFLOAT16},
        {{64, 32}, BufferType::L1, DataType::INT32},
        {{64, 32}, BufferType::L1, DataType::UINT16},
        {{64, 32}, BufferType::L1, DataType::UINT32},
        {{64, 64}, BufferType::DRAM, DataType::INT32},
        {{64, 64}, BufferType::DRAM, DataType::UINT16},
        {{64, 64}, BufferType::DRAM, DataType::UINT32},
        {{64, 64}, BufferType::L1, DataType::INT32},
        {{64, 64}, BufferType::L1, DataType::UINT16},
        {{64, 64}, BufferType::L1, DataType::UINT32},
        {{7, 96, 160}, BufferType::DRAM, DataType::BFLOAT16},
        {{7, 96, 160}, BufferType::L1, DataType::BFLOAT16},
        {{9, 128, 32}, BufferType::DRAM, DataType::BFLOAT16},
        {{9, 128, 32}, BufferType::L1, DataType::BFLOAT16},
        {{9, 160, 96}, BufferType::DRAM, DataType::BFLOAT16},
        {{9, 160, 96}, BufferType::L1, DataType::BFLOAT16},
    };
    return cases;
}

}  // namespace

bool supported_by_codegen(const TilizeCodegenParams& operation_attributes, const TilizeCodegenInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    if (input_tensor.layout() != Layout::ROW_MAJOR) {
        return false;
    }
    // codegen_sharded / codegen_rank are excluded from this manifest's sweep_suite: only the RM
    // interleaved, same-dtype nightly/codegen_dtype/broaden_suite surface is transcribed here.
    if (operation_attributes.input_mem_config.is_sharded() || operation_attributes.output_mem_config.is_sharded()) {
        return false;
    }
    // No sweep vector in codegen_tilize.py's run()/_run_broaden() ever passes a `dtype=` kwarg
    // to TilizeCodegen.tilize(), so a dtype-cast tilize call was never exercised for this port.
    if (operation_attributes.input_dtype != operation_attributes.output_dtype) {
        return false;
    }
    if (!is_supported_dtype(operation_attributes.input_dtype)) {
        return false;
    }

    const auto& shape = input_tensor.logical_shape();
    const uint32_t rank = shape.rank();
    // ttnn::tilize() squeezes rank>4 to 4D before this predicate ever runs (build_ndiml_tilize in
    // tilize.cpp); a rank<2 tensor has no H/W plane to tilize.
    if (rank < 2 || rank > 4) {
        return false;
    }
    for (uint32_t i = 0; i < rank; ++i) {
        if (shape[i] == 0) {
            return false;
        }
    }
    const uint32_t h = shape[rank - 2];
    const uint32_t w = shape[rank - 1];
    // Sub-tile inputs need PadCodegen's pad-then-tilize detour (ops/tilize/tilize.py's
    // `TilizeCodegen.tilize`); TilizeCodegenParams carries no logical/padded shape, so that path
    // is not transcribed here — sub-tile falls back to native.
    if (h % tt::constants::TILE_HEIGHT != 0 || w % tt::constants::TILE_WIDTH != 0) {
        return false;
    }
    return true;
}

bool is_demoted(const TilizeCodegenParams& operation_attributes, const TilizeCodegenInputs& tensor_args) {
    const auto& shape = tensor_args.input_tensor.logical_shape();
    std::vector<uint32_t> shape_vec;
    shape_vec.reserve(shape.rank());
    for (uint32_t i = 0; i < shape.rank(); ++i) {
        shape_vec.push_back(shape[i]);
    }
    const auto buffer_type = operation_attributes.output_mem_config.buffer_type();
    for (const auto& demoted : demoted_cases()) {
        if (demoted.buffer_type == buffer_type && demoted.dtype == operation_attributes.input_dtype &&
            demoted.shape == shape_vec) {
            return true;
        }
    }
    return false;
}

ImplementationSelector parse_implementation(std::string_view implementation) {
    if (implementation == "auto") {
        return ImplementationSelector::Auto;
    }
    if (implementation == "native") {
        return ImplementationSelector::Native;
    }
    if (implementation == "codegen") {
        return ImplementationSelector::Codegen;
    }
    TT_THROW("tilize: unknown implementation selector '{}'", implementation);
}

}  // namespace ttnn::prim
