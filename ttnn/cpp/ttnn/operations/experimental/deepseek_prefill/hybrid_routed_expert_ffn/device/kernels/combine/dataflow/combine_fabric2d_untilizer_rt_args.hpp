// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The untilizer kernel's runtime arguments: combine_fabric2d's, instantiated in this op's namespace so the
// body binds to this op's DramBuffers. The overlapped build adds the id table's address.

#pragma once

#include "combine_fabric2d_kernel_interface.hpp"

#ifndef KERNEL_BUILD
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#endif

namespace hyb_cmbf2d {

#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/combine_fabric2d_untilizer_rt_args_body.hpp"

}  // namespace hyb_cmbf2d
