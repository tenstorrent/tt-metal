// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace high_bw_all_gather {

enum class ReaderRuntimeArg : uint32_t {
    InputAddress,
    OutputAddress,
    InitialStripe,
    StripeStep,
    NumIters,
    TotalChunks,
    SliceStart,
    SliceCount,
    FinalStart,
    FinalCount,
    InputPageStart,
    InputPageEnd,
    DataValidSemaphore,
    Count,
};

enum class WriterRuntimeArg : uint32_t {
    OutputAddress,
    InitialStripe,
    StripeStep,
    NumIters,
    SliceStart,
    SliceCount,
    FinalStart,
    FinalCount,
    DoLocalWrite,
    DataValidSemaphore,
    DataValidNocX,
    DataValidNocY,
    NumGranularSends,
    NeighborDeviceId,
    NeighborMeshId,
    Count,
};

template <typename Enum>
constexpr uint32_t runtime_arg_index(Enum value) {
    return static_cast<uint32_t>(value);
}

}  // namespace high_bw_all_gather
