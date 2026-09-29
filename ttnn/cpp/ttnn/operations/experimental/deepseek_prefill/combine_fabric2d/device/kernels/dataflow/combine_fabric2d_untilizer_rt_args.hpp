// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The untilizer kernel's runtime arguments, the counterpart to combine_fabric2d_untilizer_ct_args.hpp.
//
// UntilizerRtArgManager owns the one thing the two sides have to agree on: the order. It emplaces the args on
// the host and reads them back on the kernel, and only it can build an UntilizerRtArgs.
//
// The args are the DRAM base addresses, which cannot be compile-time: a buffer's address is assigned by the
// allocator, so it describes an allocation and not a program, and a cached program is re-dispatched against
// whatever buffers the caller hands it. The manager keeps the buffers rather than their addresses so it
// emplaces them as buffers, which is what gets each position rebound on every dispatch.

#include "combine_fabric2d_kernel_interface.hpp"

#ifndef KERNEL_BUILD
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#endif

namespace cmbf2d {

#include "combine_fabric2d_untilizer_rt_args_body.hpp"

}  // namespace cmbf2d
