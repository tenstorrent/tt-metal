// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>

#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::compressor_state_exchange {

// Shift Blaze-compatible compressor states by one device along cluster_axis.
// Rank zero receives the injected temporal state; every other rank receives its
// predecessor's local state. The local inputs remain the outgoing states used
// by the next SP rank and, on the final active rank, by decode migration.
std::tuple<ttnn::Tensor, ttnn::Tensor> compressor_state_exchange(
    const ttnn::Tensor& local_kv_state,
    const ttnn::Tensor& local_score_state,
    const ttnn::Tensor& initial_kv_state,
    const ttnn::Tensor& initial_score_state,
    uint32_t cluster_axis = 0,
    ::ttnn::ccl::Topology topology = ::ttnn::ccl::Topology::Linear);

}  // namespace ttnn::operations::experimental::deepseek_prefill::compressor_state_exchange

namespace ttnn {
using operations::experimental::deepseek_prefill::compressor_state_exchange::compressor_state_exchange;
}  // namespace ttnn
