// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Re-exported from combine_fabric2d: which tile-rows a group produces and in what order is the same
// question for both ops. The overlapped walk order and its ready gate are compiled in by
// CMBF2D_OVERLAPPED, which the including kernel defines before this header.

#pragma once

#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/combine_fabric2d_group_walk.hpp"

namespace hyb_cmbf2d {

using ::cmbf2d::ControlTables;
using ::cmbf2d::group_walk;
using ::cmbf2d::GroupWalk;

// Overlap-only: the routed expert's threshold-split walk order, and the gate on it having written.
using ::cmbf2d::expert_at_step;
using ::cmbf2d::in_fused_pass;
using ::cmbf2d::local_at_step;
using ::cmbf2d::ready_target;
using ::cmbf2d::wait_for_ready;

}  // namespace hyb_cmbf2d
