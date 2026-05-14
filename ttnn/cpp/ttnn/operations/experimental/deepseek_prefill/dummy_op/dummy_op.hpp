// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/sub_device_types.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dummy_op {

ttnn::Tensor dummy_op(
    const ttnn::Tensor& input_tensor,
    uint32_t num_iter,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::dummy_op

namespace ttnn {
using operations::experimental::deepseek_prefill::dummy_op::dummy_op;
}  // namespace ttnn
