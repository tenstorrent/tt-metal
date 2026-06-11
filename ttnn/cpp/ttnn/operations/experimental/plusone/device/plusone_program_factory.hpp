// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "plusone_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"

namespace ttnn::experimental::prim {

// Metal 2.0 plusone factory — the degenerate ProgramSpecFactoryConcept.
//
// plusone has no per-call dynamic scalar (no seed-like value): every run-arg is a pure function of the
// input layout, the sub-core grid, and the skip_negative_entries flag, so there is nothing to vary
// between two dispatches that share a cache entry. So we use the simplest concept:
// create_program_artifacts returns the spec PLUS ALL run-args bundled together; on a cache hit the
// framework just refreshes the tensor bindings (UpdateTensorArgs). No extract_immutable_info, no
// create_per_enqueue_args, no custom hash — the framework hashes the generated ProgramSpec (and the
// attrs + tensor_args) for the cache key.
//
// In-place op: the output IS the input (tensor_return_value == input). A single reader kernel both
// NoC-reads the input into an L1 scratch DFB, increments, and NoC-writes back to the SAME buffer, so the
// kernel is bound as BOTH producer and consumer of the one DFB. When the input is sharded the DFB
// borrows the input's L1 storage directly (.borrowed_from = "input") and the increment happens in place
// with no NoC traffic; for the interleaved path the DFB is plain L1 scratch and the address comes from
// the TensorAccessor("input") binding.
struct PlusOneProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const PlusoneParams& operation_attributes, const Tensor& input, Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
