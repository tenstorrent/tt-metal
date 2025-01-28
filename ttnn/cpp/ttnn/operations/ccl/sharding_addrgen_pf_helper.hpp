// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
namespace shard_pf_builder {
static void add_sharding_ct_to_existing_ct(
    const IDevice* d, const tt::tt_metal::Tensor& t, std::vector<uint32_t>& args);
static void add_sharding_rt_to_existing_rt(
    const IDevice* d, const tt::tt_metal::Tensor& t, std::vector<uint32_t>& args);
std::vector<uint32_t> get_linear_shard_list(const tt::tt_metal::IDevice* device, const tt::tt_metal::Tensor& t);
uint32_t get_sharding_core_count(const tt::tt_metal::Tensor& t);
std::vector<uint32_t> sharding_ct_table_builder(const tt::tt_metal::IDevice* device, const tt::tt_metal::Tensor& t);
std::vector<CoreCoord> get_vector_of_shard_cores(const tt::tt_metal::IDevice* device, const tt::tt_metal::Tensor& t);
}  // namespace shard_pf_builder
