// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/buffer.hpp>
#include <hostdevcommon/tensor_accessor/arg_config.hpp>

#include <cstdint>
#include <vector>

namespace tt::tt_metal {

struct TensorAccessorArgs {
    explicit TensorAccessorArgs(
        const Buffer& buffer, tensor_accessor::ArgsConfig args_config = tensor_accessor::ArgConfig::None);

    std::vector<uint32_t> compile_time_args;
    std::vector<uint32_t> runtime_args;
};

}  // namespace tt::tt_metal
