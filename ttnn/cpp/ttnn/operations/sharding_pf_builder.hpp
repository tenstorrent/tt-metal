// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
namespace shard_pf_builder {
static void add_sharding_ct_to_existing_ct(const Device* d, const Tensor& t, std::vector<uint32_t>& args);
static void add_sharding_rt_to_existing_rt(const Device* d, const Tensor& t, std::vector<uint32_t>& args);
std::vector<uint32_t> get_linear_shard_list(const tt::tt_metal::Device* device, const Tensor& t);
uint32_t get_sharding_core_count(const Tensor t);
std::vector<uint32_t> sharding_ct_table_builder(const tt::tt_metal::Device* device, const Tensor& t);
}  // namespace shard_pf_builder
