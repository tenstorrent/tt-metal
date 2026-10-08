// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct ExchangeHistoriesParams {
    uint32_t sequence_parallel_axis;
    uint32_t local_rows;
    // The leading columns of the projection holding the convolution channels.
    uint32_t width;
    tt::tt_fabric::Topology topology;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct ExchangeHistoriesInputs {
    // Tiled [..., local_rows, columns] projection whose leading width columns are the channels.
    Tensor projected;
    Tensor actual_start;
    std::optional<Tensor> actual_end;
};

}  // namespace ttnn::experimental::prim
