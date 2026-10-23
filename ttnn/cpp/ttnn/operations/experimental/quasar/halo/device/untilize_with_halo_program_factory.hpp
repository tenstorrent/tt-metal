// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/experimental/quasar/halo/device/halo_device_operation_types.hpp"

namespace ttnn::prim::qsr {

// Metal 2.0 host-API factory for untilize_with_halo (the "halo" op): returns
// ProgramArtifacts (ProgramSpec + ProgramRunArgs + op-owned tensors) rather than
// a WorkloadDescriptor.  Satisfies MetalV2FactoryConcept.
//
// create_program_artifacts() allocates the four sliding-window halo config
// tensors (pad_config0/1, gather_config0/1) on device and parks them on the
// returned ProgramArtifacts::op_owned_tensors so their backing buffers outlive
// the cached program.  They are declared as TensorParameters and read by the
// reader kernels as pure address sources.
struct UntilizeWithHaloProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const HaloParams& operation_attributes, const Tensor& input_tensor, Tensor& output_tensor);
};

}  // namespace ttnn::prim::qsr
