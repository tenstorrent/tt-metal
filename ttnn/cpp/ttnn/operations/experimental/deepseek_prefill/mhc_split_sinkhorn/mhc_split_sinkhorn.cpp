// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mhc_split_sinkhorn.hpp"
#include "device/mhc_split_sinkhorn_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::mhc_split_sinkhorn {

std::array<ttnn::Tensor, 3> mhc_split_sinkhorn(
    const ttnn::Tensor& mixes, const ttnn::Tensor& consts, uint32_t n, uint32_t sinkhorn_iters, float eps) {
    return ttnn::prim::mhc_split_sinkhorn(mixes, consts, n, sinkhorn_iters, eps);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::mhc_split_sinkhorn
