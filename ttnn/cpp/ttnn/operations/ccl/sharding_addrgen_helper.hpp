// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/ccl/common/types/sharding_common.hpp"

namespace shard_builder {

using namespace tt::tt_metal;

void extend_sharding_compile_time_args(const tt::tt_metal::Tensor& t, std::vector<uint32_t>& args);
void extend_sharding_run_time_args(const tt::tt_metal::Tensor& t, std::vector<uint32_t>& args);
std::vector<uint32_t> generate_run_time_args(const tt::tt_metal::Tensor& t);
uint32_t get_sharding_core_count(const tt::tt_metal::Tensor& t);
std::vector<uint32_t> generate_compile_time_args(const tt::tt_metal::Tensor& t);
std::vector<CoreCoord> get_shard_cores(const tt::tt_metal::Tensor& t);
}  // namespace shard_builder
