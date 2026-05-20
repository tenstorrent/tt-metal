// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host-side helpers for setting up compute_kernel_lib::reduce<> kernels.
//
// Each program factory that builds a kernel calling `compute_kernel_lib::reduce<...>` must pass
// the input data format as the 3rd template argument. The kernel side reads that argument from
// the `REDUCE_FORMAT` compile-time define, set by the host with `reduce_format_define(...)`
// below (mirroring how `REDUCE_OP` and `REDUCE_DIM` are wired up in reduce_op.cpp).

#include <string>

#include <tt_stl/assert.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace ttnn::kernel_lib {

// Returns the compute-kernel `DataFormat::...` identifier string matching `fmt`. Used as the
// value of the host-side `REDUCE_FORMAT` define so the kernel-side template parameter is the
// real input dtype (Int32/Float32 + MAX route to SFPU; everything else stays on FPU/GMPOOL).
inline std::string reduce_format_define(tt::DataFormat fmt) {
    switch (fmt) {
        case tt::DataFormat::Int32: return "DataFormat::Int32";
        case tt::DataFormat::UInt32: return "DataFormat::UInt32";
        case tt::DataFormat::Float32: return "DataFormat::Float32";
        case tt::DataFormat::Float16_b: return "DataFormat::Float16_b";
        case tt::DataFormat::Bfp8_b: return "DataFormat::Bfp8_b";
        case tt::DataFormat::Bfp4_b: return "DataFormat::Bfp4_b";
        case tt::DataFormat::UInt16: return "DataFormat::UInt16";
        default: TT_THROW("reduce: unsupported input DataFormat {}", static_cast<int>(fmt));
    }
}

}  // namespace ttnn::kernel_lib
