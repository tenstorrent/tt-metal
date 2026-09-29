// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Re-exported from combine_fabric2d. What a ring-counter slot names is decided by CMBF2D_OVERLAPPED,
// which the including kernel defines before this header.

#pragma once

// This op's own re-export first: the shared arguments pull in combine_fabric2d's interface, which
// populates cmbf2d, not this namespace, and the kernel body names everything through this one.
#include "combine_fabric2d_kernel_interface.hpp"

#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/combine_fabric2d_sender_ct_args.hpp"

namespace hyb_cmbf2d {

using ::cmbf2d::SenderCtArgs;

}  // namespace hyb_cmbf2d
