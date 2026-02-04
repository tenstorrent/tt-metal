// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/host_api.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim::detail {

uint32_t get_packed_value(const tt::tt_metal::Tensor& tensor, const tt::tt_metal::PadValue& pad_value);

}  // namespace ttnn::prim::detail
