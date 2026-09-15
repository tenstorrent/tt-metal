// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::normalization::rmsnorm_distributed_bw {

// Fused dx, and optional on-chip dgamma reduction, as one generic_op.
// All tensors are fp32 TILE interleaved DRAM. When dgamma_out is nullopt, gamma is unused
// and may alias x.
tt::tt_metal::ProgramDescriptor create_rmsnorm_bw_apply_program_descriptor(
    const Tensor& x,
    const Tensor& dy,
    const Tensor& gamma,
    const Tensor& inv_rms,
    const Tensor& d,
    Tensor& dx_out,
    const std::optional<Tensor>& dgamma_out);

}  // namespace ttnn::operations::normalization::rmsnorm_distributed_bw
