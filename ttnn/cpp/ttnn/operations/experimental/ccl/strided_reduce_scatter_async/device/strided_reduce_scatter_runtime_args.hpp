// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Common runtime-argument slots of the strided reduce-scatter kernels, shared by host and kernels.
// slice_Ht is the one per-call length word; the per-core split and page counts derive from it on device.
namespace ttnn::experimental::ccl::strided_reduce_scatter_common_args {

struct Reader {
    static constexpr uint32_t slice_Ht = 0;
    static constexpr uint32_t input = 1;
    static constexpr uint32_t intermediate = 2;
    static constexpr uint32_t semaphore_0 = 3;  // out-ready semaphore, backward direction
    static constexpr uint32_t semaphore_1 = 4;  // out-ready semaphore, forward direction
    static constexpr uint32_t addcmul_a = 5;
    static constexpr uint32_t addcmul_b = 6;
    static constexpr uint32_t count = 7;
};

struct Writer {
    static constexpr uint32_t slice_Ht = 0;
    static constexpr uint32_t intermediate = 1;
    static constexpr uint32_t output = 2;
    static constexpr uint32_t semaphore_0 = 3;
    static constexpr uint32_t semaphore_1 = 4;
    static constexpr uint32_t batch_ready = 5;
    static constexpr uint32_t barrier = 6;
    static constexpr uint32_t count = 7;
};

struct Reduce {
    static constexpr uint32_t slice_Ht = 0;
    static constexpr uint32_t addcmul_scalar = 1;  // float bits; 0 without the fused addcmul
    static constexpr uint32_t count = 2;
};

}  // namespace ttnn::experimental::ccl::strided_reduce_scatter_common_args
