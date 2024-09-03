// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sliding_window.hpp"
#include <vector>

namespace ttnn::operations::sliding_window{
std::size_t SlidingWindowConfig::get_hash() const {
    return std::hash<std::string>{}(to_string());
}

/**
    * Return the input shape (excluding depth)
    */
Shape SlidingWindowConfig::get_input_shape() const {
    return Shape(std::vector<uint32_t>{batch_size, std::get<0>(input_hw), std::get<1>(input_hw)});
}

bool SlidingWindowConfig::has_parallel_config() const {
    return num_cores_nhw > 0 && !core_range_set.ranges().empty();
}
/**
    * Calculate the window op output shape, excludes the channel dimension since this config is independent of the depth.
    */
Shape SlidingWindowConfig::get_output_shape() const {
    uint32_t output_h = (input_hw.first + 2 * pad_hw.first - dilation_hw.first * window_hw.first) / stride_hw.first + 1;
    uint32_t output_w = (input_hw.second + 2 * pad_hw.second - dilation_hw.second * window_hw.second) / stride_hw.second + 1;
    // uint32_t output_h = (std::get<0>(input_hw) + 2 * std::get<0>(pad_hw) - std::get<0>(dilation_hw) * std::get<0>(window_hw)) / std::get<0>(stride_hw) + 1;
    // uint32_t output_w = (std::get<1>(input_hw) + 2 * std::get<1>(pad_hw) - std::get<1>(dilation_hw) * std::get<1>(window_hw)) / std::get<1>(stride_hw) + 1;
    log_debug(tt::LogOp, "output_size: {} {} {}", batch_size, output_h, output_w);
    return Shape( std::vector<uint32_t>{batch_size, output_h, output_w, 0});
}

/**
    * Calculate output tensor shard height
    */
uint32_t SlidingWindowConfig::get_output_shard_y(bool snap_to_tile) const {
    TT_ASSERT(has_parallel_config(), "Parallel config is not set in SlidingWindowConfig");
    Shape output_shape = get_output_shape();
    uint32_t output_nhw = output_shape[0] * output_shape[1] * output_shape[2];
    uint32_t output_nhw_padded = tt::round_up(output_nhw, num_cores_nhw * (snap_to_tile ? tt:: constants::TILE_HEIGHT : 1));
    log_debug(tt::LogOp, "output_nhw: {} output_nhw_padded: {} num_cores_nhw: {}", output_nhw, output_nhw_padded, num_cores_nhw);
    return (output_nhw_padded / num_cores_nhw);
}


std::vector<bool> generate_pad_metadata(const SlidingWindowConfig& config) {
    uint32_t padded_input_h = config.input_hw.first + 2 * config.pad_hw.first;
    uint32_t padded_input_w = config.input_hw.second + 2 * config.pad_hw.second;
    std::vector<bool> pad_metadata(config.batch_size * padded_input_h * padded_input_w, false);

    for (uint32_t b = 0; b < config.batch_size; ++b) {
        for (uint32_t h = 0; h < padded_input_h; ++h) {
            for (uint32_t w = 0; w < padded_input_w; ++w) {
                if (h < config.pad_hw.first || h >= config.pad_hw.first + config.input_hw.first ||
                    w < config.pad_hw.second || w >= config.pad_hw.second + config.input_hw.second) {
                    pad_metadata[b * padded_input_h * padded_input_w + h * padded_input_w + w] = true;
                }
            }
        }
    }
    return pad_metadata;
}

std::vector<uint32_t> generate_op_trace_metadata(const SlidingWindowConfig& config) {
    Shape output_shape = config.get_output_shape();
    uint32_t output_nhw = output_shape[0] * output_shape[1] * output_shape[2];
    uint32_t padded_input_h = config.input_hw.first + 2 * config.pad_hw.first;
    uint32_t padded_input_w = config.input_hw.second + 2 * config.pad_hw.second;
    uint32_t i = 0;
    std::vector<uint32_t> op_trace_metadata(output_nhw, 0);
    for (uint32_t b = 0; b < output_shape[0]; ++b) {
        for (uint32_t h = 0; h < output_shape[1]; ++h) {
            for (uint32_t w = 0; w < output_shape[2]; ++w) {
                uint32_t input_index = b * padded_input_h * padded_input_w + h * config.stride_hw.first * padded_input_w + w * config.stride_hw.second;
                op_trace_metadata[i++] = input_index;
            }
        }
    }
    return op_trace_metadata;
}

std::vector<std::pair<uint32_pair_t, uint32_pair_t>> generate_shard_boundaries(const SlidingWindowConfig& config, const std::vector<uint32_t>& op_trace_metadata) {
    std::vector<std::pair<uint32_pair_t, uint32_pair_t>> shard_boundaries;
    uint32_t num_cores = config.num_cores_nhw;
    uint32_t output_shard_h = config.get_output_shard_y(config.snap_to_tile);
    uint32_t padded_input_w = config.input_hw.second + 2 * config.pad_hw.second;
    uint32_t max_index = op_trace_metadata.size();
    uint32_t halo_with_pad_len = (config.window_hw.first - 1) * padded_input_w + config.window_hw.second - 1;
    uint32_t output_index_start = 0;
    for (uint32_t core = 0; core < num_cores; ++ core) {
        uint32_t output_index_end = std::min(output_index_start + output_shard_h, max_index) - 1;
        uint32_t input_index_start = op_trace_metadata[output_index_start];
        uint32_t input_index_end = op_trace_metadata[output_index_end] + halo_with_pad_len;
        if (input_index_start == 0 and output_index_start != 0) {
            input_index_start = op_trace_metadata[output_index_end] + 1;
            input_index_end = input_index_start - 1;
            log_debug(tt::LogOp, "core: {}, output_index_start: {}, output_index_end: {}, input_index_start: {}, input_index_end: {}", core, output_index_start, output_index_end, input_index_start, input_index_end);
        }
        shard_boundaries.push_back({{output_index_start, output_index_end}, {input_index_start, input_index_end}});
        output_index_start = output_index_end + 1;
    }
    #if 1
    for (auto [output_shard, input_shard] : shard_boundaries) {
        log_debug(tt::LogOp, "output_shard: ({}, {}), input_shard: ({}, {})", output_shard.first, output_shard.second, input_shard.first, input_shard.second);
    }
    #endif
    return shard_boundaries;
}

std::vector<std::pair<bool, uint32_pair_t>> generate_tensor_metadata(const std::vector<bool>& pad_metadata, const SlidingWindowConfig& config, uint32_t reshard_num_cores_nhw, bool is_in_tiled) {
    Shape input_shape = config.get_input_shape();
    uint32_t input_nhw = input_shape[0] * input_shape[1] * input_shape[2];
    uint32_t input_nhw_padded;
    if (is_in_tiled) {
        input_nhw_padded = tt::round_up(input_nhw, config.num_cores_nhw * tt:: constants::TILE_HEIGHT);
    } else {
        input_nhw_padded = tt::round_up(input_nhw, config.num_cores_nhw);
    }
    uint32_t input_shard_height = input_nhw_padded / config.num_cores_nhw;
    uint32_t input_reshard_height = reshard_num_cores_nhw == 0 ? input_shard_height : tt::round_up(input_nhw, reshard_num_cores_nhw * tt:: constants::TILE_HEIGHT) / reshard_num_cores_nhw;

    auto remap = [input_shard_height, input_reshard_height](uint32_t core_id, uint32_t local_idx) -> std::pair<uint32_t, uint32_t> {
        if (input_shard_height == input_reshard_height) {
            return std::make_pair(core_id, local_idx);
        } else {
            uint32_t global_idx = core_id * input_shard_height + local_idx;
            return std::make_pair(global_idx / input_reshard_height, global_idx % input_reshard_height);
        }
    };

    std::vector<std::pair<bool, uint32_pair_t>> tensor_metadata;
    uint32_t core_id = 0;
    uint32_t input_reshard_local_idx = 0;
    for (bool is_pad_stick : pad_metadata) {
        if (is_pad_stick) {
            tensor_metadata.push_back(std::make_pair(is_pad_stick, std::make_pair(0, 0)));
        } else {
            tensor_metadata.push_back(std::make_pair(is_pad_stick, remap(core_id, input_reshard_local_idx++)));
            if (input_reshard_local_idx == input_shard_height) {
                core_id++;
                input_reshard_local_idx = 0;
            }
        }
    }

    return tensor_metadata;
}

uint32_t generate_max_out_nsticks_per_core(const std::vector<std::pair<uint32_pair_t, uint32_pair_t>>& shard_boundaries) {
    // calculate max_out_nsticks_per_core
    uint32_t max_out_nsticks_per_core = 0;
    for (auto [_, in_shard] : shard_boundaries) {
        auto [in_start, in_end] = in_shard;
        max_out_nsticks_per_core = std::max(max_out_nsticks_per_core, in_end - in_start + 1);
    }
    return max_out_nsticks_per_core;
}

std::tuple<std::vector<std::vector<uint16_t>>, std::vector<std::vector<uint16_t>>, std::vector<std::vector<uint16_t>>> generate_halo_kernel_config_tensors(const std::vector<std::pair<bool, uint32_pair_t>>& tensor_metadata, const std::vector<std::pair<uint32_pair_t, uint32_pair_t>>& shard_boundaries, bool is_block_sharded, bool transpose_mcast, bool remote_read, Device* device) {
    auto core_id_to_noc_coords = [is_block_sharded, transpose_mcast, device](uint32_t core_id) -> CoreCoord {
        auto num_cores_x = device->compute_with_storage_grid_size().x;
        auto core_coord = is_block_sharded ? (transpose_mcast ? CoreCoord(core_id, 0) : CoreCoord(0, core_id)) : CoreCoord(core_id % num_cores_x, core_id / num_cores_x);
        return device->worker_core_from_logical_core(core_coord);
    };

    const uint16_t pad_local = 0xFFFF;
    std::map<uint32_pair_t, std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>> per_core_gather_data;

    uint32_t num_cores_nhw = shard_boundaries.size();

    uint32_t core_id = 0;
    for (auto [output_boundary, input_boundary] : shard_boundaries) {
        auto [input_start, input_end] = input_boundary;
        for (uint32_t global_idx = input_start; global_idx <= input_end; ++global_idx) {
            uint32_t dst_core_id = core_id;
            uint32_t local_idx = global_idx - input_start;
            auto [is_pad_stick, src_idx] = tensor_metadata[global_idx];
            auto [src_core_id, src_local_idx] = src_idx;
            TT_ASSERT(local_idx < pad_local && src_local_idx < pad_local, "Index overflow");
            if (is_pad_stick) {
                TT_ASSERT(src_local_idx == 0);
                src_core_id = pad_local;
            }
            if (per_core_gather_data.find({src_core_id, dst_core_id}) != per_core_gather_data.end()) {
                auto& [src_start, dst_start, length] = per_core_gather_data[{src_core_id, dst_core_id}].back();
                // src idx is 0 if it is a pad
                if ((src_local_idx == (src_start + length) || is_pad_stick) && local_idx == (dst_start + length)) {
                    ++ length;
                    continue;
                }
            }
            // insert new tuple
            per_core_gather_data[{src_core_id, dst_core_id}].push_back({src_local_idx, local_idx, 1});
        }
        ++ core_id;
    }

    // construct the config tensors
    /**
        * pad_config: length num_cores_nhw
        *     each element (for core i): [dst_start0, length0, dst_start1, length1, ...]
        * local_config: length num_cores_nhw
        *     each element (for core i): (nocx, nocy, len) -> [src_start0, dst_start0, length0, src_start1, dst_start1, length1, ...]
        * remote_config: length num_cores_nhw
        *     each element (for core i): { (nocx, nocy, len) -> [src_start0, dst_start0, length0, src_start1, dst_start1, length1, ...],
        *                                  (nocx, nocy, len) -> [src_start0, dst_start0, length0, src_start1, dst_start1, length1, ...], ...}
        */
    using uint32_triplet_t = std::tuple<uint32_t, uint32_t, uint32_t>;
    std::vector<std::vector<uint32_pair_t>> pad_config;
    std::vector<std::pair<uint32_triplet_t, std::vector<uint32_triplet_t>>> local_config;
    std::vector<std::vector<std::pair<uint32_triplet_t, std::vector<uint32_triplet_t>>>> remote_config;
    pad_config.resize(num_cores_nhw);
    local_config.resize(num_cores_nhw);
    remote_config.resize(num_cores_nhw);

    for (auto [src_dst, data] : per_core_gather_data) {
        auto [src_core_id, dst_core_id] = src_dst;
        bool is_pad = src_core_id == pad_local;
        bool is_local = src_core_id == dst_core_id;
        bool is_remote = !is_local && !is_pad;
        if (is_pad) {
            for (auto [src_start, dst_start, length] : data) {
                pad_config[dst_core_id].push_back({dst_start, length});
            }
        } else if (is_local) {
            CoreCoord noc_xy = core_id_to_noc_coords(dst_core_id);
            local_config[src_core_id].first = {noc_xy.x, noc_xy.y, 3 * data.size()};
            local_config[src_core_id].second = data;
        } else if (is_remote) {
            if (remote_read) {
                CoreCoord noc_xy = core_id_to_noc_coords(src_core_id);
                remote_config[dst_core_id].push_back({{noc_xy.x, noc_xy.y, 3 * data.size()}, data});
            } else {
                CoreCoord noc_xy = core_id_to_noc_coords(dst_core_id);
                remote_config[src_core_id].push_back({{noc_xy.x, noc_xy.y, 3 * data.size()}, data});
            }
        }
    }

    // flatten and uniformize the lengths of each config list
    auto flatten_pad_config = [](auto& config) -> std::vector<std::vector<uint16_t>> {
        // find max length
        size_t max_len = 0;
        for (auto& data : config) {
            max_len = std::max(max_len, 2 * data.size());   // each data is 2 * data.size()
        }
        std::vector<std::vector<uint16_t>> flattened_config;
        for (auto& data : config) {
            std::vector<uint16_t> flat_data(max_len, 0);
            uint32_t idx = 0;
            for (auto data_elem : data) {
                auto [dst_start, length] = data_elem;
                flat_data[idx++] = dst_start;
                flat_data[idx++] = length;
            }
            // null plug
            flat_data.emplace_back(0);
            flat_data.emplace_back(0);
            flattened_config.emplace_back(flat_data);
        }
        return flattened_config;
    };

    auto flatten_local_config = [](auto& config) -> std::vector<std::vector<uint16_t>> {
        // find max length
        size_t max_len = 0;
        for (auto& [_, data] : config) {
            max_len = std::max(max_len, 3 * data.size());   // each key is 3, data is 3 * data.size()
        }
        max_len += 3;   // key tuple
        std::vector<std::vector<uint16_t>> flattened_config;
        for (auto& [key, data]: config) {
            auto [nocx, nocy, len] = key;
            std::vector<uint16_t> flat_data(max_len, 0);
            flat_data[0] = nocx;
            flat_data[1] = nocy;
            flat_data[2] = len;
            uint32_t idx = 3;
            for (size_t i = 0; i < data.size(); ++i) {
                auto [src_start, dst_start, length] = data[i];
                flat_data[idx++] = src_start;
                flat_data[idx++] = dst_start;
                flat_data[idx++] = length;
            }
            // null plug
            flat_data.emplace_back(0);
            flat_data.emplace_back(0);
            flat_data.emplace_back(0);
            flattened_config.emplace_back(flat_data);
        }
        return flattened_config;
    };

    auto flatten_remote_config = [](auto& config) -> std::vector<std::vector<uint16_t>> {
        // find max length
        size_t max_len = 0;
        for (auto& core_config : config) {
            size_t curr_len = 0;
            for (auto& [key, subdata] : core_config) {
                curr_len += 3 + 3 * subdata.size();  // each key is len 3
            }
            max_len = std::max(max_len, curr_len);   // each key is 3, data is 3 * data.size()
        }
        std::vector<std::vector<uint16_t>> flattened_config;
        for (auto& core_config : config) {
            std::vector<uint16_t> flat_data(max_len, 0);
            uint32_t idx = 0;
            for (auto& key_data: core_config) {
                auto [nocx, nocy, len] = key_data.first;
                flat_data[idx++] = nocx;
                flat_data[idx++] = nocy;
                flat_data[idx++] = len;
                for (size_t i = 0; i < key_data.second.size(); ++i) {
                    auto [src_start, dst_start, length] = key_data.second[i];
                    flat_data[idx++] = src_start;
                    flat_data[idx++] = dst_start;
                    flat_data[idx++] = length;
                }
            }
            // null plug
            flat_data.emplace_back(0);
            flat_data.emplace_back(0);
            flat_data.emplace_back(0);
            flattened_config.emplace_back(flat_data);
        }
        return flattened_config;
    };

    auto flattened_pad_config = flatten_pad_config(pad_config);
    auto flattened_local_config = flatten_local_config(local_config);
    auto flattened_remote_config = flatten_remote_config(remote_config);

    auto align_config = [](auto& config, size_t align_granularity = 1, uint16_t align_value = 0) {
        size_t max_len = 0;
        for (auto& core_config : config) {
            max_len = std::max(max_len, core_config.size());
        }
        if (align_granularity > 1) {
            size_t align_amount = max_len % align_granularity;
            max_len = align_amount > 0 ? max_len + align_granularity - align_amount : max_len;
        }
        for (auto& core_config : config) {
            size_t curr_len = core_config.size();
            size_t extend_amount = max_len - core_config.size();
            if (extend_amount > 0) {
                std::vector<uint16_t> extend_v(extend_amount, align_value);
                core_config.insert(core_config.end(), extend_v.begin(), extend_v.end());
            }
        }
    };

    align_config(flattened_pad_config, 2);
    align_config(flattened_local_config, 2);
    align_config(flattened_remote_config, 2);

    return std::make_tuple(flattened_pad_config,
                            flattened_local_config,
                            flattened_remote_config);
}

std::vector<std::vector<uint16_t>> generate_sliding_window_op_config(const std::vector<uint32_t>& op_trace_metadata, const std::vector<std::pair<uint32_pair_t, uint32_pair_t>>& shard_boundaries, bool pad_tile, bool pad_last_core) {
    std::vector<std::vector<uint16_t>> sharded_input_top_left_indices;
    for(const auto& item : shard_boundaries) {
        const auto& [output_shard_start, output_shard_end] = item.first;
        const auto& [input_shard_start, input_shard_end] = item.second;
        // sanity check
        if (output_shard_start >= op_trace_metadata.size()) {
            // this core has no output
            continue;
        }
        // TT_ASSERT(output_shard_start < op_trace_metadata.size());
        TT_ASSERT(input_shard_start == op_trace_metadata[output_shard_start]);
        std::vector<uint16_t> local_top_left_indices;
        for(size_t i = output_shard_start; i < output_shard_end + 1; i++) {
            local_top_left_indices.push_back(op_trace_metadata[i] - op_trace_metadata[output_shard_start]);
        }
        sharded_input_top_left_indices.push_back(local_top_left_indices);
    }
    if (pad_tile) {
        // Pad indices to tile-multiple
        for(size_t i = 0; i < sharded_input_top_left_indices.size(); i++) {
            uint32_t extend_with_zeroes = sharded_input_top_left_indices[i].size() % 32;
            if (extend_with_zeroes > 0) {
                std::vector<uint16_t> extend_v(extend_with_zeroes, 0);
                sharded_input_top_left_indices[i].insert(sharded_input_top_left_indices[i].end(), extend_v.begin(), extend_v.end());
            }
        }
    }
    if (pad_last_core) {
        // Pad indices for last core if not equal to other cores
        uint32_t indices_length_per_core = sharded_input_top_left_indices[0].size();
        uint32_t indices_length_last_core = sharded_input_top_left_indices.back().size();
        TT_ASSERT(indices_length_last_core <= indices_length_per_core, "indices length for last core {} larger than indices length per core {}", indices_length_last_core, indices_length_per_core);
        if (indices_length_per_core - indices_length_last_core > 0) {
            std::vector<uint16_t> extend_v(indices_length_per_core - indices_length_last_core, 0);
            sharded_input_top_left_indices.back().insert(sharded_input_top_left_indices.back().end(), extend_v.begin(), extend_v.end());
        }

    }
    return sharded_input_top_left_indices;
}

std::vector<uint16_t> flatten(const std::vector<std::vector<uint16_t>>& input, uint32_t extend_with_zeroes) {
    std::vector<uint16_t> flattened_vector;
    for (auto sub_vec : input) {
        flattened_vector.insert(flattened_vector.end(), sub_vec.begin(), sub_vec.end());
        if (extend_with_zeroes > 0) {
            std::vector<uint16_t> extend_v(extend_with_zeroes, 0);
            flattened_vector.insert(flattened_vector.end(), extend_v.begin(), extend_v.end());
        }
    }
    log_debug(tt::LogOp, "flattened_vector size: {}", flattened_vector.size());
    return flattened_vector;
}

Tensor construct_on_host_config_tensor(const std::vector<std::vector<uint16_t>>& config, const SlidingWindowConfig& sw_config, const ParallelConfig& p_config) {
    // we need the last dim of tensors to be multiple of 2, pad if needed
    uint32_t extend_with_zeroes = config[0].size() % 2;
    extend_with_zeroes = extend_with_zeroes > 0 ? 2 - extend_with_zeroes : 0;
    Shape config_shape = Shape(std::vector<uint32_t>{(uint32_t) config.size(), (uint32_t) config[0].size() + extend_with_zeroes});
    std::vector<uint16_t> config_vector = flatten(config, extend_with_zeroes);
    if (p_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED) {
        auto config_buffer = owned_buffer::create<uint16_t>(std::move(config_vector));
        log_debug(tt::LogOp, "config_shape: ({}, {})", config_shape[0], config_shape[1]);
        return Tensor(OwnedStorage{config_buffer}, config_shape, DataType::UINT16, Layout::ROW_MAJOR);
    } else if(p_config.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED) {
            uint32_t repeat_factor = p_config.grid.num_cores();
            std::vector<uint16_t> repeat_config;
            for (uint32_t i = 0; i < repeat_factor; ++ i) {
                repeat_config.insert(repeat_config.end(), config_vector.begin(), config_vector.end());
            }
            auto config_buffer = owned_buffer::create<uint16_t>(std::move(repeat_config));
            config_shape = Shape(std::vector<uint32_t>{config_shape[0] * repeat_factor, config_shape[1]});
            return Tensor(OwnedStorage{config_buffer}, config_shape, DataType::UINT16, Layout::ROW_MAJOR);
    } else if (p_config.shard_scheme == TensorMemoryLayout::BLOCK_SHARDED) {
        TT_ASSERT(p_config.grid.ranges().size() == 1, "BLOCK_SHARDED should have just a single core range");
        // NOTE: it is assumed that the range start is always (0, 0)
        uint32_t ncores_y = p_config.grid.ranges().begin()->end_coord.y + 1;
        uint32_t ncores_x = p_config.grid.ranges().begin()->end_coord.x + 1;
        std::vector<uint16_t> repeat_config;
        uint32_t repeat_factor = 0;
        if (p_config.shard_orientation == ShardOrientation::ROW_MAJOR) {
            TT_ASSERT(config.size() == ncores_y, "Invalid config size {} (!= {}) for BLOCK_SHARDED ROW_MAJOR", config.size(), ncores_y);
            repeat_factor = ncores_x;
        } else if (p_config.shard_orientation == ShardOrientation::COL_MAJOR) {
            TT_ASSERT(config.size() == ncores_x, "Invalid config size {} (!= {}) for BLOCK_SHARDED COL_MAJOR", config.size(), ncores_x);
            repeat_factor = ncores_y;
        } else {
            TT_ASSERT(false, "Unsupported shard orientation");
        }
        for (uint32_t i = 0; i < repeat_factor; ++ i) {
            repeat_config.insert(repeat_config.end(), config_vector.begin(), config_vector.end());
        }
        auto config_buffer = owned_buffer::create<uint16_t>(std::move(repeat_config));
        config_shape = Shape(std::vector<uint32_t>{config_shape[0] * repeat_factor, config_shape[1]});
        return Tensor(OwnedStorage{config_buffer}, config_shape, DataType::UINT16, Layout::ROW_MAJOR);
    } else {
        TT_ASSERT(false, "Unsupported shard scheme");
        return Tensor();
    }
}

Tensor move_config_tensor_to_device(const Tensor& config_tensor, const ParallelConfig& p_config, bool is_block_sharded, Device* device) {
    auto shard_shape = std::array<uint32_t, 2>({1, (uint32_t) config_tensor.get_shape()[-1]});
    log_debug(tt::LogOp, "shard_shape: ({}, {})", shard_shape[0], shard_shape[1]);
    auto config_shard_orientation = is_block_sharded ? (p_config.shard_orientation == ShardOrientation::COL_MAJOR ? ShardOrientation::ROW_MAJOR : ShardOrientation::COL_MAJOR) : ShardOrientation::ROW_MAJOR;
    ShardSpec shard_spec(p_config.grid, shard_shape, config_shard_orientation, false);
    MemoryConfig memory_config{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1_SMALL, shard_spec};
    return config_tensor.to(device, memory_config);
}

std::string SlidingWindowConfig::to_string() const {
    return std::to_string(batch_size)
            + "_" + std::to_string(std::get<0>(input_hw)) + "_" + std::to_string(std::get<1>(input_hw))
            + "_" + std::to_string(std::get<0>(window_hw)) + "_" + std::to_string(std::get<1>(window_hw))
            + "_" + std::to_string(std::get<0>(stride_hw)) + "_" + std::to_string(std::get<1>(stride_hw))
            + "_" + std::to_string(std::get<0>(pad_hw)) + "_" + std::to_string(std::get<1>(pad_hw))
            + "_" + std::to_string(std::get<0>(dilation_hw)) + "_" + std::to_string(std::get<1>(dilation_hw))
            + "_" + std::to_string(num_cores_nhw) + "_" + core_range_set.str();
}

} // namespace ttnn::operations::sliding_window


auto fmt::formatter<ttnn::operations::sliding_window::ParallelConfig>::format(const ttnn::operations::sliding_window::ParallelConfig& t, format_context& ctx) const -> format_context::iterator {
        std::string shard_scheme_str = "";
        if(t.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED) {
            shard_scheme_str = "HEIGHT_SHARDED";
        } else if(t.shard_scheme == TensorMemoryLayout::BLOCK_SHARDED) {
            shard_scheme_str = "BLOCK_SHARDED";
        } else if(t.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED) {
            shard_scheme_str = "WIDTH_SHARDED";
        } else {
            shard_scheme_str = "NOT_SHARDED";
        }
        std::string shard_orientation_str = "";
        if(t.shard_orientation == ShardOrientation::COL_MAJOR){
            shard_orientation_str="COL_MAJOR";
        } else if(t.shard_orientation == ShardOrientation::ROW_MAJOR){
            shard_orientation_str="ROW_MAJOR";
        } else {
            shard_orientation_str="INVALID";
        }
        std::string str = fmt::format("ParallelConfig(grid={}, shard_scheme={}, shard_orientation={})", t.grid.str(), shard_scheme_str, shard_orientation_str);
        return fmt::format_to(ctx.out(), "{}", str);
}

auto fmt::formatter<ttnn::operations::sliding_window::SlidingWindowConfig>::format(const ttnn::operations::sliding_window::SlidingWindowConfig& t, format_context& ctx) const -> format_context::iterator {
        std::string str = fmt::format("SlidingWindowConfig(batch_size={}, input_hw=({},{}), window_hw=({},{}), stride_hw=({},{}), pad_hw=({},{}), dilation_hw=({},{}), num_cores_nhw={}, core_range_set_={})",
            t.batch_size,
            t.input_hw.first,
            t.input_hw.second,
            t.window_hw.first,
            t.window_hw.second,
            t.stride_hw.first,
            t.stride_hw.second,
            t.pad_hw.first,
            t.pad_hw.second,
            t.dilation_hw.first,
            t.dilation_hw.second,
            t.num_cores_nhw,
            t.core_range_set.str());
        return fmt::format_to(ctx.out(), "{}", str);
}
