// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/embedding/device/embedding_device_operation.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/work_split.hpp>

#include <tracy/Tracy.hpp>

#include "embedding_common.hpp"
#include "embeddings_fused_program_factory.hpp"
#include "embeddings_rm_program_factory.hpp"
#include "embeddings_tilized_indices_program_factory.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace ttnn::operations::embedding::detail {

tt::tt_metal::operation::ProgramWithCallbacks embeddings_(
    const Tensor& a,
    const Tensor& weights,
    Tensor& output,
    bool tilized,
    EmbeddingsType embeddings_type,
    std::optional<uint32_t> pad_token) {
    if (a.layout() == ttnn::TILE_LAYOUT) {
        return embeddings_tilized_indices(a, weights, output, embeddings_type, pad_token);
    } else if (tilized) {
        return embeddings_fused(a, weights, output, embeddings_type, pad_token);
    } else {
        return embeddings_rm(a, weights, output, embeddings_type, pad_token);
    }
}
}  // namespace ttnn::operations::embedding::detail
