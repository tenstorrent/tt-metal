// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
// non-transposed - always row-major layout

namespace shard_pf_builder {

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

std::vector<uint32_t> sharding_ct_table_builder(const Tensor& t) {
    std::vector<uint32_t> args;
    TT_ASSERT(t.is_sharded());
    TT_FATAL(
        t.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED ||
            t.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED ||
            t.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED,
        "ShardedAddrGenArgBuilder::emit_ct_args was invoked with a tensor containing an unsupported (Sharded) Tensor "
        "Memory Layout: {}",
        t.memory_config().memory_layout);
    struct ShardSpec shard_spec = t.shard_spec().value();
    struct ShardSpecBuffer buf_shard_spec = t.buffer()->shard_spec();
    const auto& [pages_per_shard_y, pages_per_shard_x] = buf_shard_spec.shape_in_pages();
    // This takes a floor division of page_shape[0] and page_shape[1] from tensor_shard_spec.shape[0] and
    // tensor_shard_spec.shape[1] respectively Shouldn't it be a roof division?
    const auto core_ranges = buf_shard_spec.grid().ranges();
    const auto num_core_range = core_ranges.size();
    bool shard_grid_transposed =
        ((t.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED &&
          shard_spec.orientation == ShardOrientation::ROW_MAJOR) ||
         ((t.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED ||
           t.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED) &&
          shard_spec.orientation == ShardOrientation::COL_MAJOR));

    args.push_back(static_cast<uint32_t>(shard_grid_transposed));
    args.push_back(static_cast<uint32_t>(t.memory_config().memory_layout));
    args.push_back(static_cast<uint32_t>(num_core_range));
    args.push_back(static_cast<uint32_t>(t.buffer()->page_size()));
    args.push_back(pages_per_shard_x);
    args.push_back(pages_per_shard_y);
    // Todo figure out how many pages in total are in the last shard dimension not including padding pages
    // Push this back (pages_last_shard_dim)
    // CCL had the concept of a virtual NOC address ? Where they added VIRTUAL_TENSIX_START_X/Y to the start of the page
    // Push back a 1 if this is the case
    for (int i = 0; i < num_core_range; i++) {
        args.push_back(static_cast<uint32_t>(worker_core_from_logical_core(core_ranges.at(i).start_coord.x)));
        args.push_back(static_cast<uint32_t>(worker_core_from_logical_core(core_ranges.at(i).start_coord.y)));
        args.push_back(static_cast<uint32_t>(core_ranges.at(i).end_coord.x - core_ranges.at(i).start_coord.x));
        args.push_back(static_cast<uint32_t>(core_ranges.at(i).end_coord.y - core_ranges.at(i).start_coord.y));
    }
    /*
    // shard_grid_height (cores)
    args.push_back(shard_grid_end.y - shard_grid_start.y + 1);
    TT_FATAL(args.back() > 0, "Passed shard_grid height == 0 to sharded addrgen, which is invalid");
    // shard_grid_width (cores)
    args.push_back(shard_grid_end.x - shard_grid_start.x + 1);
    TT_FATAL(args.back() > 0, "Passed shard_grid width == 0 to sharded addrgen, which is invalid");
    // shard_grid_start_y
    args.push_back(shard_grid_start.y);
    // shard_grid_start_x
    args.push_back(shard_grid_start.x);
    // pages_per_shard_y
    args.push_back(pages_per_shard_y);
    TT_FATAL(args.back() > 0, "Passed pages per shard y == 0 to sharded addrgen, which is invalid");
    // pages_per_shard_x
    args.push_back(pages_per_shard_x);
    TT_FATAL(args.back() > 0, "Passed pages per shard x == 0 to sharded addrgen, which is invalid");
    // transposed grid
    args.push_back(static_cast<uint32_t>(shard_grid_transposed));
    */
    return args;
}

}  // namespace shard_pf_builder
