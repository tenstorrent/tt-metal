// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>

#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::csa_compressor {

std::tuple<Tensor, Tensor, Tensor> csa_compressor(
    const Tensor& kv,
    const Tensor& gate,
    const Tensor& position_bias,
    const Tensor& initial_kv_state,
    const Tensor& initial_score_state,
    uint32_t seq_len_actual,
    uint32_t first_token_position = 0,
    uint32_t cluster_axis = 0,
    ::ttnn::ccl::Topology topology = ::ttnn::ccl::Topology::Linear);

}  // namespace ttnn::operations::experimental::deepseek_prefill::csa_compressor

namespace ttnn {
using operations::experimental::deepseek_prefill::csa_compressor::csa_compressor;
}  // namespace ttnn
