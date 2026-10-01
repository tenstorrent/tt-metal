// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::pcie_exchange {

enum class PcieExchangeMode : uint32_t {
    ALL_REDUCE = 0,  // out[core] = in[core] + peer_in[core]; output has the input's layout, on every device
    ALL_GATHER = 1,  // out = concat over devices along the last dim; output width-sharded over num_devices x N cores
};

struct PcieExchangeParams {
    PcieExchangeMode mode = PcieExchangeMode::ALL_REDUCE;
    MemoryConfig output_mem_config;
    uint32_t num_devices = 0;
    // ALL_REDUCE only: ship the shard as bfp8 (1088 B/tile instead of 2048) and sum the two bfp8 copies, so both
    // devices compute the identical bf16 result. Halves the link time; the input, output and landing areas keep
    // their bf16 sizes (the bfp8 landing fits inside the bf16-sized exchange shard).
    bool bfp8_payload = false;
};

struct PcieExchangeInputs {
    Tensor input;
    // Persistent per-core exchange buffer, L1 width-sharded on the same cores as `input`, bf16 TILE, shard at
    // least (2 * ntiles + 1) tiles wide: [landing 0][landing 1][control tile]. Same address on every device
    // (allocated once through the mesh) and alive for the whole run; the peer writes into it asynchronously.
    Tensor exchange;
    // Optional separate control tile (flag / sequence / staging), L1 width-sharded bf16 TILE on the same cores,
    // one tile per core. When given, `exchange` only needs the two landing buffers (2 * ntiles tiles) and may be
    // a fresh, uninitialised allocation per call, while the control tile stays persistent and zero-initialised
    // outside any trace: the peer may write its flag into the control tile before this device's op starts, so the
    // control words must never be re-zeroed per call.
    std::optional<Tensor> ctrl;
};

}  // namespace ttnn::operations::experimental::pcie_exchange
