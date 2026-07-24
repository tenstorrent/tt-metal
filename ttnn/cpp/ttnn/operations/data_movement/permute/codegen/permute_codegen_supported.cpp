// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute_codegen_supported.hpp"

#include <vector>

#include <tt_stl/assert.hpp>

#include "permute_codegen_device_operation.hpp"

namespace ttnn::operations::data_movement::permute_codegen {

namespace {

constexpr uint32_t kTileH = 32;
constexpr uint32_t kTileW = 32;
constexpr uint32_t kFusedMinNc = 6;
constexpr uint64_t kFusedMaxL1Bytes = 1024 * 1024;

// Transcribed from an internal reference implementation's _fused_wh_ok. Only evaluated when
// dims[-1] == rank - 2; a match means the op orchestration delegates the WHOLE dispatch to
// TransposeCodegen's fused WH kernel (transpose port scope, not this port's), so
// supported_by_codegen() must reject it.
bool fused_wh_ok(const Tensor& input_tensor, ttsl::Span<const uint32_t> dims) {
    const auto& shape = input_tensor.logical_shape();
    const uint32_t rank = static_cast<uint32_t>(dims.size());
    const auto dtype = input_tensor.dtype();
    const uint32_t elem_size = input_tensor.element_size();
    // dtype is already narrowed to BFLOAT16/FLOAT32/INT32 by supported_by_codegen() before this
    // runs, so only elem_size == 2 (BFLOAT16) needs a dtype-independent check here.
    if (!(elem_size == 2 || dtype == DataType::INT32 || dtype == DataType::FLOAT32)) {
        return false;
    }
    const uint32_t h = shape[rank - 2];
    const uint32_t w = shape[rank - 1];
    if (h % kTileH != 0 || w % kTileW != 0) {
        return false;
    }
    uint32_t nc = 1;
    for (uint32_t i = 0; i + 2 < rank; ++i) {
        nc *= shape[i];
    }
    if (nc < kFusedMinNc) {
        return false;
    }
    const uint32_t ht = (h + kTileH - 1) / kTileH;
    const uint32_t wt = (w + kTileW - 1) / kTileW;
    return static_cast<uint64_t>(ht) * wt * kTileH * kTileW * elem_size <= kFusedMaxL1Bytes;
}

struct DemotedEntry {
    std::vector<uint32_t> shape;
    std::vector<uint32_t> dims;
    DataType dtype;
};

// Enumerated perf-demote ledger: in-scope (supported_by_codegen == true) cases that lose to
// native on device. Consulted only by the free function's `auto` branch.
const std::vector<DemotedEntry>& demoted_entries() {
    static const std::vector<DemotedEntry> entries = {
        {{1, 2, 3, 64, 96}, {1, 2, 0, 3, 4}, DataType::BFLOAT16},
        {{1, 2, 3, 64, 96}, {1, 2, 0, 3, 4}, DataType::FLOAT32},
        {{1, 2, 3, 64, 96}, {1, 2, 0, 3, 4}, DataType::INT32},
        {{1, 2, 3, 64, 96}, {2, 1, 4, 3, 0}, DataType::BFLOAT16},
        {{1, 2, 3, 64, 96}, {2, 1, 4, 3, 0}, DataType::FLOAT32},
        {{1, 2, 3, 64, 96}, {2, 1, 4, 3, 0}, DataType::INT32},
        {{1, 2, 3, 64, 96}, {2, 3, 1, 4, 0}, DataType::BFLOAT16},
        {{1, 2, 3, 64, 96}, {2, 3, 1, 4, 0}, DataType::FLOAT32},
        {{1, 2, 3, 64, 96}, {2, 3, 1, 4, 0}, DataType::INT32},
        {{1, 2, 3, 64, 96}, {2, 3, 4, 0, 1}, DataType::BFLOAT16},
        {{1, 2, 3, 64, 96}, {2, 3, 4, 0, 1}, DataType::FLOAT32},
        {{1, 2, 3, 64, 96}, {2, 3, 4, 0, 1}, DataType::INT32},
        {{1, 2, 3, 64, 96}, {4, 0, 2, 3, 1}, DataType::BFLOAT16},
        {{1, 2, 3, 64, 96}, {4, 0, 2, 3, 1}, DataType::FLOAT32},
        {{1, 2, 3, 64, 96}, {4, 0, 2, 3, 1}, DataType::INT32},
        {{1, 4, 96, 128}, {1, 3, 2, 0}, DataType::BFLOAT16},
        {{1, 4, 96, 128}, {1, 3, 2, 0}, DataType::FLOAT32},
        {{1, 4, 96, 128}, {1, 3, 2, 0}, DataType::INT32},
        {{1, 4, 96, 128}, {3, 2, 0, 1}, DataType::BFLOAT16},
        {{1, 4, 96, 128}, {3, 2, 0, 1}, DataType::FLOAT32},
        {{1, 4, 96, 128}, {3, 2, 0, 1}, DataType::INT32},
        {{1, 4, 96, 128}, {3, 2, 1, 0}, DataType::BFLOAT16},
        {{1, 4, 96, 128}, {3, 2, 1, 0}, DataType::FLOAT32},
        {{1, 4, 96, 128}, {3, 2, 1, 0}, DataType::INT32},
        {{2, 3, 4, 32, 64}, {2, 1, 4, 3, 0}, DataType::BFLOAT16},
        {{2, 3, 4, 32, 64}, {2, 1, 4, 3, 0}, DataType::FLOAT32},
        {{2, 3, 4, 32, 64}, {2, 1, 4, 3, 0}, DataType::INT32},
        {{2, 3, 4, 32, 64}, {2, 3, 1, 4, 0}, DataType::BFLOAT16},
        {{2, 3, 4, 32, 64}, {2, 3, 1, 4, 0}, DataType::FLOAT32},
        {{2, 3, 4, 32, 64}, {2, 3, 1, 4, 0}, DataType::INT32},
        {{2, 3, 4, 32, 64}, {2, 3, 4, 0, 1}, DataType::BFLOAT16},
        {{2, 3, 4, 32, 64}, {2, 3, 4, 0, 1}, DataType::FLOAT32},
        {{2, 3, 4, 32, 64}, {2, 3, 4, 0, 1}, DataType::INT32},
        {{2, 3, 4, 32, 64}, {4, 0, 2, 3, 1}, DataType::BFLOAT16},
        {{2, 3, 4, 32, 64}, {4, 0, 2, 3, 1}, DataType::FLOAT32},
        {{2, 3, 4, 32, 64}, {4, 0, 2, 3, 1}, DataType::INT32},
        {{2, 3, 64, 96}, {1, 3, 2, 0}, DataType::BFLOAT16},
        {{2, 3, 64, 96}, {1, 3, 2, 0}, DataType::FLOAT32},
        {{2, 3, 64, 96}, {1, 3, 2, 0}, DataType::INT32},
        {{2, 3, 64, 96}, {3, 2, 0, 1}, DataType::BFLOAT16},
        {{2, 3, 64, 96}, {3, 2, 0, 1}, DataType::FLOAT32},
        {{2, 3, 64, 96}, {3, 2, 0, 1}, DataType::INT32},
        {{2, 3, 64, 96}, {3, 2, 1, 0}, DataType::BFLOAT16},
        {{2, 3, 64, 96}, {3, 2, 1, 0}, DataType::FLOAT32},
        {{2, 3, 64, 96}, {3, 2, 1, 0}, DataType::INT32},
        {{2, 96, 128}, {0, 2, 1}, DataType::BFLOAT16},
        {{2, 96, 128}, {0, 2, 1}, DataType::FLOAT32},
        {{2, 96, 128}, {0, 2, 1}, DataType::INT32},
        {{2, 96, 128}, {1, 2, 0}, DataType::BFLOAT16},
        {{2, 96, 128}, {1, 2, 0}, DataType::FLOAT32},
        {{2, 96, 128}, {1, 2, 0}, DataType::INT32},
        {{2, 96, 128}, {2, 0, 1}, DataType::BFLOAT16},
        {{2, 96, 128}, {2, 0, 1}, DataType::FLOAT32},
        {{2, 96, 128}, {2, 0, 1}, DataType::INT32},
        {{3, 64, 96}, {0, 2, 1}, DataType::BFLOAT16},
        {{3, 64, 96}, {0, 2, 1}, DataType::FLOAT32},
        {{3, 64, 96}, {0, 2, 1}, DataType::INT32},
        {{3, 64, 96}, {1, 2, 0}, DataType::BFLOAT16},
        {{3, 64, 96}, {1, 2, 0}, DataType::FLOAT32},
        {{3, 64, 96}, {1, 2, 0}, DataType::INT32},
        {{3, 64, 96}, {2, 0, 1}, DataType::BFLOAT16},
        {{3, 64, 96}, {2, 0, 1}, DataType::FLOAT32},
        {{3, 64, 96}, {2, 0, 1}, DataType::INT32},
        {{64, 96}, {1, 0}, DataType::BFLOAT16},
        {{64, 96}, {1, 0}, DataType::FLOAT32},
        {{64, 96}, {1, 0}, DataType::INT32},
        {{96, 64}, {1, 0}, DataType::BFLOAT16},
        {{96, 64}, {1, 0}, DataType::FLOAT32},
        {{96, 64}, {1, 0}, DataType::INT32},
    };
    return entries;
}

}  // namespace

bool supported_by_codegen(
    const Tensor& input_tensor, ttsl::Span<const uint32_t> dims, const MemoryConfig& output_memory_config) {
    const uint32_t rank = static_cast<uint32_t>(dims.size());
    if (rank < 2 || rank > ttnn::operations::data_movement::PermuteCodegenDeviceOperation::MAX_DIMS) {
        return false;
    }
    // TILE is entirely out of scope for this port.
    if (input_tensor.layout() != Layout::ROW_MAJOR) {
        return false;
    }
    // Sharded and non-DRAM-interleaved input are out of scope: the two kernel sets this port
    // wires (build_permute_rm / build_permute_rm_blocked) both assume a plain DRAM-interleaved
    // TensorAccessor, and the reference implementation's invalidate_vector rejects L1-interleaved
    // input.
    if (input_tensor.memory_config().is_sharded() || input_tensor.memory_config().buffer_type() != BufferType::DRAM) {
        return false;
    }
    // bfloat8_b requires TILE (shared-exponent block-float has no row-major representation);
    // any other non-covered dtype is likewise out of scope.
    const auto dtype = input_tensor.dtype();
    if (dtype != DataType::BFLOAT16 && dtype != DataType::FLOAT32 && dtype != DataType::INT32) {
        return false;
    }

    if (dims[rank - 1] == rank - 1) {
        // Row-invariant: build_permute_rm. Its writer moves whole pages via
        // noc_async_write_page(), which resolves for both interleaved and sharded destinations.
        return true;
    }
    if (output_memory_config.is_sharded()) {
        // W-changing: build_permute_rm_blocked's writer scatters partial pages, addressing each
        // sub-page write via the free-function get_noc_addr(page_id, accessor) overload. That
        // overload only resolves for the interleaved TensorAccessor specialization; a sharded
        // destination has no matching overload and fails to compile the kernel.
        return false;
    }
    if (dims[rank - 1] == rank - 2 && fused_wh_ok(input_tensor, dims)) {
        // Left-out-for-now: delegates to TransposeCodegen's fused WH kernel, not a
        // permute-owned path (e.g. dims=[0,3,1,2] on [2,3,64,96]).
        return false;
    }
    // W-changing: build_permute_rm_blocked.
    return true;
}

bool is_demoted(const Tensor& input_tensor, ttsl::Span<const uint32_t> dims) {
    const auto& shape = input_tensor.logical_shape();
    const uint32_t rank = static_cast<uint32_t>(dims.size());
    std::vector<uint32_t> shape_vec(rank);
    for (uint32_t i = 0; i < rank; ++i) {
        shape_vec[i] = shape[i];
    }
    std::vector<uint32_t> dims_vec(dims.begin(), dims.end());
    const auto dtype = input_tensor.dtype();
    for (const auto& entry : demoted_entries()) {
        if (entry.dtype == dtype && entry.shape == shape_vec && entry.dims == dims_vec) {
            return true;
        }
    }
    return false;
}

ImplementationSelector parse_implementation(std::string_view implementation) {
    if (implementation == "native") {
        return ImplementationSelector::Native;
    }
    if (implementation == "codegen") {
        return ImplementationSelector::Codegen;
    }
    TT_FATAL(implementation == "auto", "Unknown permute implementation selector: {}", implementation);
    return ImplementationSelector::Auto;
}

}  // namespace ttnn::operations::data_movement::permute_codegen
