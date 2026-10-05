// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::ccl {

// Common runtime-argument layouts shared by host factories and device kernels.
struct AllGatherCommonArgs {
    static constexpr uint32_t input = 0;
    static constexpr uint32_t output = 1;
    static constexpr uint32_t barrier = 2;
    static constexpr uint32_t semaphore_0 = 3;
    static constexpr uint32_t semaphore_1 = 4;
    static constexpr uint32_t count = 5;
};

struct LlamaGatherReaderCommonArgs {
    static constexpr uint32_t input = 0;
    static constexpr uint32_t count = 1;
};

struct LlamaGatherWriterCommonArgs {
    static constexpr uint32_t output = 0;
    static constexpr uint32_t semaphore = 1;
    static constexpr uint32_t barrier = 2;
    static constexpr uint32_t count = 3;
};

struct AllReduceReaderCommonArgs {
    static constexpr uint32_t input = 0;
    static constexpr uint32_t count = 1;
};

struct AllReduceSemaphoreCommonArgs {
    static constexpr uint32_t semaphore = 0;
    static constexpr uint32_t count = 1;
};

struct ReduceScatterCommonArgs {
    static constexpr uint32_t input = 0;
    static constexpr uint32_t intermediate = 1;
    static constexpr uint32_t output = 2;
    static constexpr uint32_t penult = 3;
    static constexpr uint32_t barrier = 4;
    static constexpr uint32_t semaphore_0 = 5;
    static constexpr uint32_t semaphore_1 = 6;
    static constexpr uint32_t ack = 7;
    static constexpr uint32_t count = 8;
};

}  // namespace ttnn::ccl
