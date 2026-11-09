// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/gguf/gguf_matmul_device_operation_types.hpp"

namespace ttnn::prim {

struct GgufMatmulSharedVariables {
    std::array<tt::tt_metal::KernelHandle, 2> reader{}, writer{};  // by the core's column parity
    std::vector<CoreCoord> cores;
};

// One row block of the input. The weight's (N tile, K group) records are split evenly over the grid, so an N tile can
// be cut between up to eight cores: the core holding its first groups adds the others' fp32 partials. When the N
// tiles are a multiple of the DRAM banks (an N tile's records then sit in one bank), each core takes records of the
// banks nearest it. One collector core reads the input and multicasts it to the grid in chunks. When the input does not
// fit L1, bands of grid rows each take a share of K and receive only that share from their own collector.
struct GgufMatmulProgramFactory {
    using shared_variables_t = GgufMatmulSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const GgufMatmulParams&, const GgufMatmulInputs&, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program, const GgufMatmulParams&, const GgufMatmulInputs&, Tensor& output);
};

// More than one row block of the input. Bands of whole grid rows share M rows and split N over their cores; each core
// decodes its weight tiles once per M chunk into L1 and multiplies them against the chunk's rows, which a sender core
// of the band multicasts in blocks. With few rows, two bands split K and one adds the other's sums.
struct GgufMatmulPrefillProgramFactory {
    using shared_variables_t = GgufMatmulSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const GgufMatmulParams&, const GgufMatmulInputs&, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program, const GgufMatmulParams&, const GgufMatmulInputs&, Tensor& output);
};

// True when GgufMatmulPrefillProgramFactory takes this input (whole row blocks) and weight (its formats, a blocking
// that fits L1).
bool gguf_prefill_supported(const Tensor& input, const Tensor& weight);

}  // namespace ttnn::prim
