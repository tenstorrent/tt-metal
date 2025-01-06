// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
// non-transposed - always row-major layout
// vec<logical row -> noc row>, vec<logicacal col -> noc col>
static std::pair<std::vector<uint32_t>, std::vector<uint32_t>> shard_noc_cores(
    const Device* d, const ShardSpec& shard_spec) {
    TT_ASSERT(d != nullptr);
    // Bounding box encompases all the shard cord range sets
    const auto& core_range = shard_spec.grid.bounding_box();
    std::vector<uint32_t> logical_to_noc_row_map;
    std::vector<uint32_t> logical_to_noc_col_map;
    for (uint32_t y = 0; y <= core_range.end_coord.y; y++) {
        CoreCoord noc_core = d->virtual_core_from_logical_core(CoreCoord(0, y), CoreType::WORKER);
        logical_to_noc_row_map.push_back(noc_core.y);
    }
    for (uint32_t x = 0; x <= core_range.end_coord.x; x++) {
        CoreCoord noc_core = d->virtual_core_from_logical_core(CoreCoord(x, 0), CoreType::WORKER);
        logical_to_noc_col_map.push_back(noc_core.x);
    }

    return {logical_to_noc_row_map, logical_to_noc_col_map};
}

std::vector<uint32_t> sharding_rt_table_builder(const Device* d, const Tensor& t) {
    std::vector<uint32_t> args;
    const auto& [row_map, col_map] = shard_noc_cores(d, t.shard_spec().value());
    args.push_back(row_map.size());
    for (uint32_t i = 0; i < row_map.size(); i++) {
        args.push_back(row_map.at(i));
    }
    args.push_back(col_map.size());
    for (uint32_t i = 0; i < col_map.size(); i++) {
        args.push_back(col_map.at(i));
    }

    return args;
}
