// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace csa_compressor::runtime_args {

enum class State : uint32_t {
    KvAddress,
    GateAddress,
    BiasAddress,
    BaseKvAddress,
    BaseScoreAddress,
    OutputKvAddress,
    OutputScoreAddress,
    LocalValid,
    AbsoluteStart,
    StateTiles,
    FirstStateTile,
    Count,
};

enum class Reader : uint32_t {
    KvAddress,
    GateAddress,
    BiasAddress,
    PredecessorKvAddress,
    PredecessorScoreAddress,
    OutputTiles,
    CompleteWindows,
    AbsoluteStart,
    FirstOutputTile,
    Count,
};

enum class Compute : uint32_t {
    OutputTiles,
    Count,
};

enum class Writer : uint32_t {
    OutputAddress,
    OutputTiles,
    FirstOutputTile,
    Count,
};

template <typename Arg>
constexpr uint32_t index(Arg arg) {
    return static_cast<uint32_t>(arg);
}

}  // namespace csa_compressor::runtime_args
