// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>

#include <tt-metalium/core_coord.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/metal2_artifacts.hpp"
#include "ttnn/operations/reduction/sampling/device/sampling_device_operation_types.hpp"

namespace ttnn::prim {

// Metal 2.0 sampling factory — AdvancedProgramSpecFactoryConcept with the per-enqueue split (Option 3++).
//
// The legacy sampling op baked the RNG seed as a COMPILE-TIME arg, so it lived in the program and every
// distinct seed forced a recompile (≈1 s/call on the realistic per-token decode path). This port moves
// the seed to a per-enqueue RUNTIME arg AND keys the cache on a structural ImmutableInfo that doesn't
// contain the seed at all — so two calls differing only in seed map to the same cache entry, and a
// mutable value cannot leak into the key or the spec because the builder never sees anything but the
// ImmutableInfo. (No custom hash function — the ImmutableInfo struct IS the key.)
//
//   - extract_immutable_info → the cache key AND the sole input to create_program_artifacts: the
//     structural projection of the request (the six tensor specs + the compute grid + sub-core grid).
//   - create_program_artifacts → the immutable ProgramSpec (18 DFBs, 3 kernels, named-arg schemas, the
//     6 tensor parameters) PLUS the ENQUEUE-INVARIANT run-args: the per-core core_id (a function of the
//     grid, fixed for a cache entry; declared enqueue-invariant in the spec).
//   - create_per_enqueue_args → the PER-ENQUEUE run-args: the per-core RNG seed + the 6 tensor addresses,
//     re-applied on every dispatch via UpdateProgramRunArgs.
struct SamplingProgramFactory {
    struct immutable_info_t {
        tt::tt_metal::TensorSpec input_values_spec;
        tt::tt_metal::TensorSpec input_indices_spec;
        tt::tt_metal::TensorSpec k_spec;
        tt::tt_metal::TensorSpec p_spec;
        tt::tt_metal::TensorSpec temp_spec;
        tt::tt_metal::TensorSpec output_spec;
        CoreCoord grid;
        std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids;
        // Architecture is structural: it selects the index intermediate width (Int32 on Quasar,
        // UInt16 on WH/BH) and the compute kernel's fp32_dest_acc_en, so it belongs in the key.
        tt::ARCH arch = tt::ARCH::Invalid;

        static constexpr auto attribute_names = std::forward_as_tuple(
            "input_values_spec",
            "input_indices_spec",
            "k_spec",
            "p_spec",
            "temp_spec",
            "output_spec",
            "grid",
            "sub_core_grids",
            "arch");
        auto attribute_values() const {
            return std::forward_as_tuple(
                input_values_spec,
                input_indices_spec,
                k_spec,
                p_spec,
                temp_spec,
                output_spec,
                grid,
                sub_core_grids,
                arch);
        }
    };

    static immutable_info_t extract_immutable_info(
        const SamplingParams& operation_attributes, const SamplingInputs& tensor_args);
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(const immutable_info_t& info);
    static tt::tt_metal::experimental::ProgramRunArgs create_per_enqueue_args(
        const SamplingParams& operation_attributes,
        const SamplingInputs& tensor_args,
        Tensor& output_tensor,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
};

}  // namespace ttnn::prim
