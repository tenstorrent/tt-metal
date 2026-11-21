// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/types.hpp"

namespace ttnn::experimental::llama31_8b_spec {

ttnn::Tensor swiglu_decode(const ttnn::Tensor& gate, const ttnn::Tensor& up);

bool swiglu_decode_supported(const ttnn::Tensor& gate, const ttnn::Tensor& up);

}  // namespace ttnn::experimental::llama31_8b_spec
