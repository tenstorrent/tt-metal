// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "embedding_common.hpp"

namespace ttnn::operations::embedding::detail {

tt::tt_metal::operation::ProgramWithCallbacks embeddings_fused(
    const Tensor& a,
    const Tensor& weights,
    Tensor& output,
    EmbeddingsType embeddings_type,
    std::optional<uint32_t> pad_token);

}  // namespace ttnn::operations::embedding::detail
