// SPDX-FileCopyrightText: (c) 2026
//
// SPDX-License-Identifier: Apache-2.0

#include "nemotron3_mamba2_decode_owned.hpp"

#include "device/nemotron3_mamba2_decode_owned_device_operation.hpp"

namespace ttnn::experimental {

std::tuple<Tensor, Tensor> nemotron3_mamba2_decode_owned(
    const Tensor& x,
    const Tensor& z,
    const Tensor& dt,
    const Tensor& dt_bias,
    const Tensor& A_log,
    const Tensor& D,
    const Tensor& B_in,
    const Tensor& C_in,
    const Tensor& ssm_state,
    bool debug_fill,
    uint32_t debug_mode,
    const std::optional<MemoryConfig>& output_memory_config,
    const std::optional<Tensor>& preallocated_y) {
    return ttnn::prim::nemotron3_mamba2_decode_owned(
        x,
        z,
        dt,
        dt_bias,
        A_log,
        D,
        B_in,
        C_in,
        ssm_state,
        debug_fill,
        debug_mode,
        output_memory_config,
        preallocated_y);
}

}  // namespace ttnn::experimental
