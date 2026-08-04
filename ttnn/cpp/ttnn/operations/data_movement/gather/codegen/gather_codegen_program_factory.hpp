// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct GatherCodegenParams;
struct GatherCodegenInputs;

// Host-computed tile-page geometry shared by all three factories AND by select_program_factory's
// L1-fit / core-count routing (ops/gather/gather.py Step 6, `_gather_impl`). Derived purely from
// tensor shapes (like native GatherDeviceOperation's own inline Ht/Wt_input/Wt_index computation),
// so none of it needs to live in GatherCodegenParams for cache-key purposes.
struct GatherGeometry {
    uint32_t Ht = 0;
    uint32_t Wt_input = 0;
    uint32_t Wt_index = 0;
    uint32_t index_valid_h_last = 0;
    uint32_t index_valid_w_last = 0;
    uint32_t index_ht_per_batch = 0;
};

GatherGeometry compute_gather_geometry(const Tensor& input_tensor, const Tensor& input_index_tensor);

// Mirrors ops/gather/gather.py::_interleaved_fits_l1: whether the row-buffered kernel's three CBs
// (Wt_input + 1 + max(4, Wt_index) tile pages, the SAME depths the Interleaved/Tiled factories
// allocate) fit the device's real per-core L1 budget.
bool gather_interleaved_fits_l1(
    const Tensor& input_tensor, const Tensor& input_index_tensor, uint32_t Wt_input, uint32_t Wt_index);

// Row-buffered: full Wt_input row resident in L1 (kernels/gather_reader.cpp, gather_writer.cpp).
struct GatherCodegenProgramFactoryInterleaved {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const GatherCodegenParams& attributes, const GatherCodegenInputs& tensor_args, Tensor& output_tensor);
};

// Per-output-tile split: high parallelism for small Ht (kernels/gather_reader_tiled.cpp,
// gather_writer_tiled.cpp).
struct GatherCodegenProgramFactoryTiled {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const GatherCodegenParams& attributes, const GatherCodegenInputs& tensor_args, Tensor& output_tensor);
};

// Double-buffered streaming: large Wt_input that doesn't fit the row-buffered L1 budget
// (kernels/gather_reader_streaming.cpp, gather_writer_streaming.cpp).
struct GatherCodegenProgramFactoryStreaming {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const GatherCodegenParams& attributes, const GatherCodegenInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim
