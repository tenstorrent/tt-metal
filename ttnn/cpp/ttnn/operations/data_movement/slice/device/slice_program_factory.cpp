// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory.hpp"

#include <optional>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

// Helper functions used by multiple factories
namespace detail {

inline std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_slice_runtime_args_rm(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    uint32_t num_cores,
    const std::vector<CoreCoord>& all_cores_vec,
    const CoreRangeSet& core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_sticks_per_core_group_1,
    uint32_t num_sticks_per_core_group_2,
    uint32_t max_read_size) {
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.padded_shape();
    auto output_shape = output_tensor.padded_shape();

    uint32_t padded_row_size_bytes = input_shape[-1] * input_tensor.element_size();
    uint32_t unpadded_row_size_bytes = output_shape[-1] * input_tensor.element_size();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_unpadded_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_padded_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly
    // This currently just matches tile version where we iterate over the row as well
    num_unpadded_sticks_per_dim[0] = 1;
    num_padded_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;

    for (int32_t i = 1; i < num_dims; i++) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_sticks_per_dim[i] = num_unpadded_dim;
        num_padded_sticks_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    auto src_buffer_alignment = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto dst_buffer_alignment = output_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);
    uint32_t begins_bytes = output_tensor_start[-1] * input_tensor.element_size();
    uint32_t misalignment = begins_bytes % src_buffer_alignment;
    uint32_t unpadded_row_size_bytes_offset = tt::round_up(unpadded_row_size_bytes, alignment);
    uint32_t start_addr = input_tensor.buffer()->address();

    std::vector<uint32_t> common_reader_kernel_args = {
        start_addr + begins_bytes - misalignment,  // read from nearest aligned address,
        padded_row_size_bytes,
        unpadded_row_size_bytes,
        unpadded_row_size_bytes_offset,
        num_dims,
        misalignment,
        0,
        0,
        0,
        0};
    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_unpadded_sticks_per_dim.begin(), num_unpadded_sticks_per_dim.end());
    common_reader_kernel_args.insert(
        common_reader_kernel_args.end(), num_padded_sticks_per_dim.begin(), num_padded_sticks_per_dim.end());

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val;
    ret_val.reserve(num_cores);

    uint32_t start_offset = ttnn::operations::data_movement::get_rm_start_offset(input_tensor, output_tensor_start);
    uint32_t num_sticks_written = 0;
    for (const auto& core : all_cores_vec) {
        uint32_t num_sticks_per_core;
        if (core_group_1.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_2;
        } else {
            // no-op
            num_sticks_per_core = 0;
        }

        // issue more reads before calling barrier
        uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
        if (num_sticks_per_core != 0) {
            auto num_sticks_per_core_pad32 = num_sticks_per_core + ((32 - num_sticks_per_core % 32) % 32);
            num_sticks_per_core_read = tt::tt_metal::merge_num_sticks_to_read(
                num_sticks_per_core_pad32, unpadded_row_size_bytes_offset, max_read_size);
            num_read_per_barrier = num_sticks_per_core_pad32 / num_sticks_per_core_read;
        }

        id_per_dim[0] = num_sticks_written % num_unpadded_sticks_per_dim[0];
        uint32_t unpadded_written = num_sticks_written / num_unpadded_sticks_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;

        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = unpadded_written % num_unpadded_sticks_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }
        std::vector<uint32_t> reader_kernel_args = common_reader_kernel_args;
        //
        uint32_t addr_offset =
            6;  // input buffer addr, padded_row_size_bytes, unpadded_row_size_bytes, num_dims, misalignment
        reader_kernel_args[addr_offset++] = start_id;
        reader_kernel_args[addr_offset++] = num_sticks_per_core;
        reader_kernel_args[addr_offset++] = num_sticks_per_core_read;
        reader_kernel_args[addr_offset] = num_read_per_barrier;
        reader_kernel_args.insert(reader_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());

        std::vector<uint32_t> writer_kernel_args = {
            output_buffer->address(),
            unpadded_row_size_bytes,
            unpadded_row_size_bytes_offset,
            num_sticks_per_core,
            num_sticks_per_core_read,
            num_read_per_barrier,
            num_sticks_written,
            0};
        num_sticks_written += num_sticks_per_core;
        ret_val.emplace_back(reader_kernel_args, writer_kernel_args);
    }

    return ret_val;
}

constexpr uint32_t MAX_READ_SIZE = 4096;

std::tuple<uint32_t, uint32_t, uint32_t> compute_cb_size(
    const Tensor& input,
    const Tensor& output,
    const Shape& output_tensor_start,
    const uint32_t num_sticks_per_core_group_1,
    const uint32_t num_sticks_per_core_group_2) {
    auto src_buffer_alignment = input.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto dst_buffer_alignment = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);

    // if begins is not aligned then we need to pad the cb size, so that we can read from the nearest aligned address
    uint32_t begins_bytes = output_tensor_start[-1] * input.element_size();
    uint32_t misalignment = begins_bytes % src_buffer_alignment;

    if (misalignment != 0) {
        alignment *= 2;
    }
    const ttnn::Shape& output_shape = output.padded_shape();
    const uint32_t unpadded_row_size_bytes = output_shape[-1] * input.element_size();
    const uint32_t cb_page_size = tt::round_up(unpadded_row_size_bytes, alignment);
    const uint32_t num_input_pages = num_sticks_per_core_group_1 > num_sticks_per_core_group_2
                                         ? num_sticks_per_core_group_1
                                         : num_sticks_per_core_group_2;
    uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
    if (num_input_pages != 0) {
        auto num_sticks_per_core_pad32 = num_input_pages + ((32 - num_input_pages % 32) % 32);
        num_sticks_per_core_read =
            tt::tt_metal::merge_num_sticks_to_read(num_sticks_per_core_pad32, cb_page_size, MAX_READ_SIZE);
        num_read_per_barrier = num_sticks_per_core_pad32 / num_sticks_per_core_read;
    }

    return std::make_tuple(cb_page_size, num_read_per_barrier, misalignment);
}

inline std::vector<std::vector<uint32_t>> group_contiguous_values(std::vector<uint32_t>& values) {
    std::vector<std::vector<uint32_t>> chunks;
    if (values.empty()) {
        return chunks;
    }

    // Initialize the first chunk
    std::vector<uint32_t> current_chunk;
    current_chunk.push_back(values[0]);

    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] == values[i - 1] + 1) {
            current_chunk.push_back(values[i]);
        } else {
            chunks.push_back(current_chunk);
            current_chunk.clear();
            current_chunk.push_back(values[i]);
        }
    }
    // Add the last chunk
    chunks.push_back(current_chunk);
    return chunks;
}

inline std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_slice_runtime_args_rm_sharded(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    uint32_t num_cores_unpadded,
    bool row_major,
    uint32_t num_cores_x_unpadded,
    uint32_t num_cores_y_unpadded,
    uint32_t shard_height_unpadded,
    uint32_t shard_height_padded,
    uint32_t num_cores_x_padded,
    uint32_t num_cores_y_padded) {
    tt::tt_metal::IDevice* device = input_tensor.device();

    auto input_shape = input_tensor.padded_shape();
    auto output_shape = output_tensor.padded_shape();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_unpadded_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_padded_sticks_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly
    // This currently just matches tile version where we iterate over the row as well
    num_unpadded_sticks_per_dim[0] = 1;
    num_padded_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;

    for (int32_t i = 1; i < num_dims; i++) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_sticks_per_dim[i] = num_unpadded_dim;
        num_padded_sticks_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores_unpadded);

    uint32_t start_offset = ttnn::operations::data_movement::get_rm_start_offset(input_tensor, output_tensor_start);
    for (uint32_t i = 0, num_sticks_written = 0; i < num_cores_unpadded; i++) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x_unpadded, i / num_cores_x_unpadded};
        } else {
            core = {i / num_cores_y_unpadded, i % num_cores_y_unpadded};
        }
        uint32_t num_sticks_per_core_unpadded = shard_height_unpadded;
        uint32_t num_sticks_per_core_padded = shard_height_padded;

        // figure out the start read stick id for each core, and the start id for each dim
        id_per_dim[0] = num_sticks_written % num_unpadded_sticks_per_dim[0];
        uint32_t unpadded_written = num_sticks_written / num_unpadded_sticks_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;

        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = unpadded_written % num_unpadded_sticks_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }

        num_sticks_written += num_sticks_per_core_unpadded;

        // stores all sticks id for a core
        std::vector<uint32_t> stick_ids_per_core;
        uint32_t src_stick_id = start_id;
        for (uint32_t i = 0; i < num_sticks_per_core_unpadded; ++i) {
            stick_ids_per_core.push_back(src_stick_id);
            src_stick_id++;
            for (uint32_t j = 0; j < num_dims; j++) {
                id_per_dim[j]++;
                if (id_per_dim[j] == num_unpadded_sticks_per_dim[j]) {
                    id_per_dim[j] = 0;
                    src_stick_id += num_padded_sticks_per_dim[j];
                } else {
                    break;
                }
            }
        }

        // figure out the stick id in a shard, and the core id for the stick.
        std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> core_stick_map;
        for (uint32_t i = 0; i < num_sticks_per_core_unpadded; ++i) {
            uint32_t stick_id = stick_ids_per_core[i];
            uint32_t shard_id = stick_id / num_sticks_per_core_padded;
            uint32_t stick_id_in_shard = stick_id - (shard_id * num_sticks_per_core_padded);

            uint32_t shard_grid_inner_dim = row_major ? num_cores_x_padded : num_cores_y_padded;
            uint32_t shard_grid_outer_dim_id = shard_id / shard_grid_inner_dim;
            uint32_t shard_grid_inner_dim_id = shard_id - (shard_grid_outer_dim_id * shard_grid_inner_dim);

            uint32_t worker_y_logical = row_major ? shard_grid_outer_dim_id : shard_grid_inner_dim_id;
            uint32_t worker_x_logical = row_major ? shard_grid_inner_dim_id : shard_grid_outer_dim_id;

            if (worker_x_logical < num_cores_x_padded and worker_y_logical < num_cores_y_padded) {
                auto core_physical =
                    device->worker_core_from_logical_core(CoreCoord{worker_x_logical, worker_y_logical});
                // save stick id in a shard, and core coord into a map
                std::pair<uint32_t, uint32_t> xy_pair = row_major ? std::make_pair(core_physical.y, core_physical.x)
                                                                  : std::make_pair(core_physical.x, core_physical.y);
                core_stick_map[xy_pair].push_back(stick_id_in_shard);
            }
        }

        // reader rt args
        std::vector<uint32_t> reader_kernel_args;
        reader_kernel_args.push_back(core_stick_map.size());  // num_cores

        for (const auto& core_stick_pair : core_stick_map) {
            auto xy_pair = core_stick_pair.first;
            if (row_major) {
                reader_kernel_args.push_back(xy_pair.second);  // noc x
                reader_kernel_args.push_back(xy_pair.first);   // noc y
            } else {
                reader_kernel_args.push_back(xy_pair.first);   // noc x
                reader_kernel_args.push_back(xy_pair.second);  // noc y
            }
        }

        // coalesce the sticks into chunks
        std::vector<std::vector<std::vector<uint32_t>>> stick_chunks_per_core;
        for (auto core_stick_pair : core_stick_map) {
            auto stick_chunks = group_contiguous_values(core_stick_pair.second);
            stick_chunks_per_core.push_back(stick_chunks);

            reader_kernel_args.push_back(stick_chunks.size());  // num_chunks for current core
        }
        for (const auto& stick_chunks : stick_chunks_per_core) {
            for (auto chunk : stick_chunks) {
                reader_kernel_args.push_back(chunk[0]);      // start id of a chunk
                reader_kernel_args.push_back(chunk.size());  // length of a chunk
            }
        }

        std::vector<uint32_t> writer_kernel_args;
        ret_val[i] = {reader_kernel_args, writer_kernel_args};
    }

    return ret_val;
}

template <bool initialize_args>
inline __attribute__((always_inline)) void set_slice_runtime_args_tile(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    const uint32_t& num_cores,
    const CoreRangeSet& all_cores,
    const CoreRangeSet& core_group_1,
    const CoreRangeSet& core_group_2,
    const uint32_t& num_tiles_per_core_group_1,
    const uint32_t& num_tiles_per_core_group_2,
    const Program& program,
    const tt::tt_metal::KernelHandle& unary_reader_kernel_id,
    const tt::tt_metal::KernelHandle& unary_writer_kernel_id,
    std::vector<uint32_t>& accumulated_total_per_dim) {
    const auto input_buffer = input_tensor.buffer();
    const auto output_buffer = output_tensor.buffer();
    const auto& input_shape = input_tensor.padded_shape();
    const auto& output_shape = output_tensor.padded_shape();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());

    uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH;
    uint32_t num_total_Xt = input_shape[-1] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT;
    uint32_t num_total_Yt = input_shape[-2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;

    const auto set_common_reader_args = [&](
        uint32_t * reader_common_args, uint32_t * num_unpadded_tiles_per_dim, uint32_t * num_padded_tiles_per_dim)
        __attribute__((always_inline)) {
        reader_common_args[0] = input_buffer->address();
        num_unpadded_tiles_per_dim[0] = num_unpadded_Xt;
        num_unpadded_tiles_per_dim[1] = num_unpadded_Yt;
        num_padded_tiles_per_dim[0] = num_padded_Xt;
        num_padded_tiles_per_dim[1] = num_padded_Yt;
        accumulated_total_per_dim[0] = num_total_Xt;
        accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt;
        for (int32_t i = 2; i < num_dims; ++i) {
            uint32_t num_unpadded_dim = output_shape[-(i + 1)];
            uint32_t num_total_dim = input_shape[-(i + 1)];
            uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
            num_unpadded_tiles_per_dim[i] = num_unpadded_dim;
            num_padded_tiles_per_dim[i] = num_padded_dim;
            accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
        }
    };

    const auto set_reader_rt_args = [&](
        uint32_t * reader_rt_args,
        const uint32_t* num_unpadded_tiles_per_dim,
        const uint32_t* num_padded_tiles_per_dim,
        const uint32_t& num_tiles_per_core,
        const uint32_t& start_offset,
        const uint32_t& num_tiles_written) __attribute__((always_inline)) {
        reader_rt_args[2] = num_tiles_written % num_unpadded_tiles_per_dim[0];
        uint32_t unpadded_written = num_tiles_written / num_unpadded_tiles_per_dim[0];
        uint32_t start_id = reader_rt_args[2] + start_offset;
        for (uint32_t j = 1; j < num_dims; ++j) {
            reader_rt_args[2 + j] = unpadded_written % num_unpadded_tiles_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_tiles_per_dim[j];
            start_id += reader_rt_args[2 + j] * accumulated_total_per_dim[j - 1];
        }
        reader_rt_args[0] = start_id;
        reader_rt_args[1] = num_tiles_per_core;
    };

    if constexpr (initialize_args) {
        std::vector<uint32_t> reader_common_args(1 + (num_dims * 2));
        uint32_t* num_unpadded_tiles_per_dim = reader_common_args.data() + 1;
        uint32_t* num_padded_tiles_per_dim = num_unpadded_tiles_per_dim + num_dims;
        set_common_reader_args(reader_common_args.data(), num_unpadded_tiles_per_dim, num_padded_tiles_per_dim);
        SetCommonRuntimeArgs(program, unary_reader_kernel_id, reader_common_args);
    }
    auto& reader_common_args = GetCommonRuntimeArgs(program, unary_reader_kernel_id);
    uint32_t* num_unpadded_tiles_per_dim = reader_common_args.data() + 1;
    uint32_t* num_padded_tiles_per_dim = num_unpadded_tiles_per_dim + num_dims;
    if constexpr (!initialize_args) {
        set_common_reader_args(reader_common_args.data(), num_unpadded_tiles_per_dim, num_padded_tiles_per_dim);
    }

    uint32_t start_offset = ttnn::operations::data_movement::get_tiled_start_offset(input_tensor, output_tensor_start);

    auto& reader_kernel_args_by_core = GetRuntimeArgs(program, unary_reader_kernel_id);
    auto& writer_kernel_args_by_core = GetRuntimeArgs(program, unary_writer_kernel_id);
    uint32_t num_tiles_written = 0;
    for (const auto& core : corerange_to_cores(all_cores)) {
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op
            if constexpr (initialize_args) {
                std::vector<uint32_t> reader_kernel_args(2 + num_dims, 0);
                std::vector<uint32_t> writer_kernel_args(3, 0);
                tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_kernel_args);
                tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_kernel_args);
            } else {
                auto& reader_kernel_args = reader_kernel_args_by_core[core.x][core.y];
                reader_kernel_args[1] = 0;
                auto& writer_kernel_args = writer_kernel_args_by_core[core.x][core.y];
                writer_kernel_args[1] = 0;
            }
            continue;
        }

        if constexpr (initialize_args) {
            std::vector<uint32_t> reader_kernel_args(2 + num_dims);
            set_reader_rt_args(
                reader_kernel_args.data(),
                num_unpadded_tiles_per_dim,
                num_padded_tiles_per_dim,
                num_tiles_per_core,
                start_offset,
                num_tiles_written);
            SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_kernel_args);
        } else {
            auto& reader_kernel_args = reader_kernel_args_by_core[core.x][core.y];
            set_reader_rt_args(
                reader_kernel_args.data(),
                num_unpadded_tiles_per_dim,
                num_padded_tiles_per_dim,
                num_tiles_per_core,
                start_offset,
                num_tiles_written);
        }

        if constexpr (initialize_args) {
            const std::array writer_kernel_args = {output_buffer->address(), num_tiles_per_core, num_tiles_written};
            tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_kernel_args);
        } else {
            auto& writer_kernel_args = writer_kernel_args_by_core[core.x][core.y];
            writer_kernel_args[0] = output_buffer->address();
            writer_kernel_args[1] = num_tiles_per_core;
            writer_kernel_args[2] = num_tiles_written;
        }
        num_tiles_written += num_tiles_per_core;
    }
}

template <bool initialize_args>
inline __attribute__((always_inline)) void set_slice_runtime_args_tensor_args(
    const Tensor& input_tensor,
    const Tensor& start_tensor,
    const Tensor& end_tensor,
    const Tensor& output_tensor,
    const uint32_t& num_cores,
    const CoreRangeSet& all_cores,
    const CoreRangeSet& core_group_1,
    const CoreRangeSet& core_group_2,
    const uint32_t& num_tiles_per_core_group_1,
    const uint32_t& num_tiles_per_core_group_2,
    const Program& program,
    const tt::tt_metal::KernelHandle& unary_reader_kernel_id,
    const tt::tt_metal::KernelHandle& unary_writer_kernel_id,
    std::vector<uint32_t>& accumulated_total_per_dim);

template <bool initialize_args>
inline __attribute__((always_inline)) void set_slice_runtime_args_tensor_args(
    const Tensor& input_tensor,
    const Tensor& start_tensor,
    const Tensor& end_tensor,
    const Tensor& output_tensor,
    const uint32_t& num_cores,
    const CoreRangeSet& all_cores,
    const CoreRangeSet& core_group_1,
    const CoreRangeSet& core_group_2,
    const uint32_t& num_tiles_per_core_group_1,
    const uint32_t& num_tiles_per_core_group_2,
    const Program& program,
    const tt::tt_metal::KernelHandle& unary_reader_kernel_id,
    const tt::tt_metal::KernelHandle& unary_writer_kernel_id,
    std::vector<uint32_t>& accumulated_total_per_dim) {
    const auto input_buffer = input_tensor.buffer();
    const auto output_buffer = output_tensor.buffer();
    const auto& input_shape = input_tensor.padded_shape();
    const auto& output_shape = output_tensor.padded_shape();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());

    uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH;
    uint32_t num_total_Xt = input_shape[-1] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT;
    uint32_t num_total_Yt = input_shape[-2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;

    const auto set_common_reader_args = [&](uint32_t* reader_common_args,
                                            uint32_t* num_unpadded_tiles_per_dim,
                                            uint32_t* num_padded_tiles_per_dim,
                                            uint32_t* input_shape_args) __attribute__((always_inline)) {
        reader_common_args[0] = input_buffer->address();
        reader_common_args[1] = start_tensor.buffer()->address();
        reader_common_args[2] = end_tensor.buffer()->address();

        num_unpadded_tiles_per_dim[0] = num_unpadded_Xt;
        num_unpadded_tiles_per_dim[1] = num_unpadded_Yt;
        num_padded_tiles_per_dim[0] = num_padded_Xt;
        num_padded_tiles_per_dim[1] = num_padded_Yt;
        accumulated_total_per_dim[0] = num_total_Xt;
        accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt;
        for (int32_t i = 2; i < num_dims; ++i) {
            uint32_t num_unpadded_dim = output_shape[-(i + 1)];
            uint32_t num_total_dim = input_shape[-(i + 1)];
            uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
            num_unpadded_tiles_per_dim[i] = num_unpadded_dim;
            num_padded_tiles_per_dim[i] = num_padded_dim;
            accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
        }
        // Copy input shape to common args
        for (int32_t i = 0; i < num_dims; ++i) {
            input_shape_args[i] = input_shape[i];
        }
    };

    const auto set_reader_rt_args = [&](uint32_t* reader_rt_args,
                                        const uint32_t* num_unpadded_tiles_per_dim,
                                        const uint32_t* num_padded_tiles_per_dim,
                                        const uint32_t& num_tiles_per_core,
                                        const uint32_t& start_offset,
                                        const uint32_t& num_tiles_written) __attribute__((always_inline)) {
        reader_rt_args[2] = num_tiles_written % num_unpadded_tiles_per_dim[0];
        uint32_t unpadded_written = num_tiles_written / num_unpadded_tiles_per_dim[0];
        uint32_t start_id = reader_rt_args[2] + start_offset;
        for (uint32_t j = 1; j < num_dims; ++j) {
            reader_rt_args[2 + j] = unpadded_written % num_unpadded_tiles_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_tiles_per_dim[j];
            start_id += reader_rt_args[2 + j] * accumulated_total_per_dim[j - 1];
        }
        reader_rt_args[0] = start_id;
        reader_rt_args[1] = num_tiles_per_core;
    };

    if constexpr (initialize_args) {
        std::vector<uint32_t> reader_common_args(3 + (num_dims * 3));  // Added space for input_shape
        uint32_t* num_unpadded_tiles_per_dim = reader_common_args.data() + 3;
        uint32_t* num_padded_tiles_per_dim = num_unpadded_tiles_per_dim + num_dims;
        uint32_t* input_shape_args = num_padded_tiles_per_dim + num_dims;
        set_common_reader_args(
            reader_common_args.data(), num_unpadded_tiles_per_dim, num_padded_tiles_per_dim, input_shape_args);
        SetCommonRuntimeArgs(program, unary_reader_kernel_id, reader_common_args);
    }
    auto& reader_common_args = GetCommonRuntimeArgs(program, unary_reader_kernel_id);
    uint32_t* num_unpadded_tiles_per_dim = reader_common_args.data() + 3;
    uint32_t* num_padded_tiles_per_dim = num_unpadded_tiles_per_dim + num_dims;
    uint32_t* input_shape_args = num_padded_tiles_per_dim + num_dims;
    if constexpr (!initialize_args) {
        set_common_reader_args(
            reader_common_args.data(), num_unpadded_tiles_per_dim, num_padded_tiles_per_dim, input_shape_args);
    }

    uint32_t start_offset = 0;

    auto& reader_kernel_args_by_core = GetRuntimeArgs(program, unary_reader_kernel_id);
    auto& writer_kernel_args_by_core = GetRuntimeArgs(program, unary_writer_kernel_id);
    uint32_t num_tiles_written = 0;
    for (const auto& core : corerange_to_cores(all_cores)) {
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op
            if constexpr (initialize_args) {
                std::vector<uint32_t> reader_kernel_args(2 + num_dims, 0);
                std::vector<uint32_t> writer_kernel_args(3, 0);
                tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_kernel_args);
                tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_kernel_args);
            } else {
                auto& reader_kernel_args = reader_kernel_args_by_core[core.x][core.y];
                reader_kernel_args[1] = 0;
                auto& writer_kernel_args = writer_kernel_args_by_core[core.x][core.y];
                writer_kernel_args[1] = 0;
            }
            continue;
        }

        if constexpr (initialize_args) {
            std::vector<uint32_t> reader_kernel_args(2 + num_dims);
            set_reader_rt_args(
                reader_kernel_args.data(),
                num_unpadded_tiles_per_dim,
                num_padded_tiles_per_dim,
                num_tiles_per_core,
                start_offset,
                num_tiles_written);
            SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_kernel_args);
        } else {
            auto& reader_kernel_args = reader_kernel_args_by_core[core.x][core.y];
            set_reader_rt_args(
                reader_kernel_args.data(),
                num_unpadded_tiles_per_dim,
                num_padded_tiles_per_dim,
                num_tiles_per_core,
                start_offset,
                num_tiles_written);
        }

        if constexpr (initialize_args) {
            const std::array writer_kernel_args = {output_buffer->address(), num_tiles_per_core, num_tiles_written};
            tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_kernel_args);
        } else {
            auto& writer_kernel_args = writer_kernel_args_by_core[core.x][core.y];
            writer_kernel_args[0] = output_buffer->address();
            writer_kernel_args[1] = num_tiles_per_core;
            writer_kernel_args[2] = num_tiles_written;
        }
        num_tiles_written += num_tiles_per_core;
    }
}

}  // namespace detail

namespace slice::program {
SliceRmProgramFactory::cached_program_t SliceRmProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    tt::tt_metal::IDevice* device = input.device();

    uint32_t num_unpadded_sticks = output.physical_volume() / output.padded_shape()[-1];

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_sticks)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_sticks);

    tt::tt_metal::Buffer* src0_buffer = input.buffer();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    constexpr uint32_t src0_cb_index = 0;

    const auto [cb_page_size, num_read_per_barrier, misalignment] = detail::compute_cb_size(
        input, output, args.slice_start, num_sticks_per_core_group_1, num_sticks_per_core_group_2);

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_read_per_barrier * 2 * cb_page_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, cb_page_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    std::vector<uint32_t> writer_compile_time_args_vec = {(std::uint32_t)src0_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args_vec);

    std::vector<uint32_t> reader_compile_time_args_vec;
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args_vec);
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args_vec));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "slice_writer_unary_stick_layout_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args_vec));

    auto all_cores_vec = corerange_to_cores(all_cores);
    auto all_runtime_args = detail::get_slice_runtime_args_rm(
        input,
        output,
        args.slice_start,
        num_cores,
        all_cores_vec,
        core_group_1,
        core_group_2,
        num_sticks_per_core_group_1,
        num_sticks_per_core_group_2,
        detail::MAX_READ_SIZE);

    for (size_t i = 0; i < all_cores_vec.size(); ++i) {
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, all_cores_vec[i], all_runtime_args[i].first);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, all_cores_vec[i], all_runtime_args[i].second);
    }

    return {
        std::move(program),
        {unary_reader_kernel_id, unary_writer_kernel_id, compute_with_storage_grid_size, args.sub_core_grids, cb_src0}};
}

void SliceRmProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& src_tensor = tensor_args.input;
    const auto& slice_start = args.slice_start;
    uint32_t num_unpadded_sticks = output.physical_volume() / output.padded_shape()[-1];

    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        cached_program.shared_variables.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(
                  cached_program.shared_variables.sub_core_grids.value(), num_unpadded_sticks)
            : tt::tt_metal::split_work_to_cores(
                  cached_program.shared_variables.compute_with_storage_grid_size, num_unpadded_sticks);

    const auto [cb_page_size, num_read_per_barrier, misalignment] = detail::compute_cb_size(
        src_tensor, output, slice_start, num_sticks_per_core_group_1, num_sticks_per_core_group_2);

    const uint32_t cb_size_bytes = num_read_per_barrier * 2 * cb_page_size;
    UpdateCircularBufferTotalSize(cached_program.program, cached_program.shared_variables.cb_src0, cb_size_bytes);
    UpdateCircularBufferPageSize(cached_program.program, cached_program.shared_variables.cb_src0, 0, cb_page_size);

    auto all_cores_vec = corerange_to_cores(all_cores);
    auto all_runtime_args = detail::get_slice_runtime_args_rm(
        src_tensor,
        output,
        slice_start,
        num_cores,
        all_cores_vec,
        core_group_1,
        core_group_2,
        num_sticks_per_core_group_1,
        num_sticks_per_core_group_2,
        detail::MAX_READ_SIZE);

    for (size_t i = 0; i < all_cores_vec.size(); ++i) {
        auto& reader_runtime_args = GetRuntimeArgs(
            cached_program.program, cached_program.shared_variables.unary_reader_kernel_id, all_cores_vec[i]);
        std::copy(all_runtime_args[i].first.begin(), all_runtime_args[i].first.end(), reader_runtime_args.data());

        auto& writer_runtime_args = GetRuntimeArgs(
            cached_program.program, cached_program.shared_variables.unary_writer_kernel_id, all_cores_vec[i]);
        std::copy(all_runtime_args[i].second.begin(), all_runtime_args[i].second.end(), writer_runtime_args.data());
    }
}

// SliceRmShardedProgramFactory implementation
SliceRmShardedProgramFactory::cached_program_t SliceRmShardedProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    [[maybe_unused]] uint32_t num_padded_sticks = input.physical_volume() / input.padded_shape()[-1];
    [[maybe_unused]] uint32_t num_unpadded_sticks = output.physical_volume() / output.padded_shape()[-1];

    // stick sizes
    uint32_t W_padded = input.logical_shape()[-1];
    uint32_t W_unpadded = output.logical_shape()[-1];
    auto stick_size_padded = W_padded * input.element_size();
    auto stick_size_unpadded = W_unpadded * output.element_size();

    // input shard spec
    auto shard_spec_padded = input.shard_spec().value();
    uint32_t shard_height_padded = shard_spec_padded.shape[0];

    [[maybe_unused]] auto& all_cores_padded = shard_spec_padded.grid;
    [[maybe_unused]] uint32_t num_cores_padded = shard_spec_padded.num_cores();
    auto bbox_padded = shard_spec_padded.grid.bounding_box();
    CoreCoord grid_size_padded = {bbox_padded.end_coord.x + 1, bbox_padded.end_coord.y + 1};
    uint32_t num_cores_x_padded = grid_size_padded.x;
    uint32_t num_cores_y_padded = grid_size_padded.y;

    if (args.sub_core_grids.has_value()) {
        log_warning(tt::LogOp, "sub_core_grids is not used when input tensor is sharded");
    }

    log_debug(tt::LogOp, "num_padded_sticks: {}", num_padded_sticks);
    log_debug(tt::LogOp, "shard_height_padded: {}", shard_height_padded);
    log_debug(tt::LogOp, "all_cores_padded: {}", all_cores_padded);
    log_debug(tt::LogOp, "num_cores_padded: {}", num_cores_padded);

    // output shard spec
    auto shard_spec_unpadded = output.shard_spec().value();
    uint32_t shard_height_unpadded = shard_spec_unpadded.shape[0];
    bool row_major = shard_spec_unpadded.orientation == ShardOrientation::ROW_MAJOR;

    auto& all_cores_unpadded = shard_spec_unpadded.grid;
    uint32_t num_cores_unpadded = shard_spec_unpadded.num_cores();
    auto bbox_unpadded = all_cores_unpadded.bounding_box();
    CoreCoord grid_size_unpadded = {bbox_unpadded.end_coord.x + 1, bbox_unpadded.end_coord.y + 1};
    uint32_t num_cores_x_unpadded = grid_size_unpadded.x;
    uint32_t num_cores_y_unpadded = grid_size_unpadded.y;

    log_debug(tt::LogOp, "num_unpadded_sticks: {}", num_unpadded_sticks);
    log_debug(tt::LogOp, "shard_height_unpadded: {}", shard_height_unpadded);
    log_debug(tt::LogOp, "all_cores_unpadded: {}", all_cores_unpadded);
    log_debug(tt::LogOp, "num_cores_unpadded: {}", num_cores_unpadded);

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat dst_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(shard_height_padded * stick_size_padded, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, stick_size_padded)
            .set_globally_allocated_address(*input.buffer());
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores_unpadded, cb_src0_config);

    uint32_t output_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            shard_height_unpadded * stick_size_unpadded, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, stick_size_unpadded)
            .set_globally_allocated_address(*output.buffer());
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, all_cores_unpadded, cb_output_config);

    std::vector<uint32_t> reader_ct_args = {
        (std::uint32_t)stick_size_padded, (std::uint32_t)stick_size_unpadded, (std::uint32_t)shard_height_unpadded};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "slice_reader_unary_unpad_dims_rm_sharded.cpp",
        all_cores_unpadded,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    auto all_runtime_args = detail::get_slice_runtime_args_rm_sharded(
        input,
        output,
        args.slice_start,
        num_cores_unpadded,
        row_major,
        num_cores_x_unpadded,
        num_cores_y_unpadded,
        shard_height_unpadded,
        shard_height_padded,
        num_cores_x_padded,
        num_cores_y_padded);

    for (uint32_t i = 0; i < num_cores_unpadded; i++) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x_unpadded, i / num_cores_x_unpadded};
        } else {
            core = {i / num_cores_y_unpadded, i % num_cores_y_unpadded};
        }
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first);
    }

    return {std::move(program), {cb_src0, cb_output}};
}

void SliceRmShardedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto src_buffer_a = tensor_args.input.buffer();
    auto dst_buffer = output.buffer();

    UpdateDynamicCircularBufferAddress(cached_program.program, cached_program.shared_variables.cb_src0, *src_buffer_a);
    UpdateDynamicCircularBufferAddress(cached_program.program, cached_program.shared_variables.cb_output, *dst_buffer);
}

// SliceRmStrideProgramFactory implementation
SliceRmStrideProgramFactory::cached_program_t SliceRmStrideProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& input_tensor = tensor_args.input;
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    tt::tt_metal::IDevice* device = input_tensor.device();

    const auto& input_shape = input_tensor.padded_shape();
    const auto& output_shape = output.padded_shape();
    uint32_t element_size = input_tensor.element_size();

    // Calculate total output rows based on tensor rank
    uint32_t total_output_rows = output_shape.volume() / output_shape[-1];

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), total_output_rows)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_output_rows);

    // Select kernels based on tensor rank
    std::string reader_kernel_path, writer_kernel_path;
    if (input_shape.rank() <= 4) {
        reader_kernel_path =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_multicore_slice_4d.cpp";
        writer_kernel_path =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/writer_multicore_slice_4d.cpp";
    } else {
        reader_kernel_path =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_multicore_slice_nd.cpp";
        writer_kernel_path =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/writer_multicore_slice_nd.cpp";
    }

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t actual_output_w = output_shape[-1];
    uint32_t output_bytes_per_row = actual_output_w * element_size;
    uint32_t cb_page_size = output_bytes_per_row;

    auto src_buffer_alignment = input_tensor.buffer()->alignment();
    auto dst_buffer_alignment = output.buffer()->alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);

    uint32_t cb_page_size_aligned = tt::round_up(cb_page_size, alignment);
    uint32_t cb_total_size = 2 * cb_page_size_aligned;

    constexpr uint32_t in_cb = 0;

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_total_size, {{in_cb, cb_data_format}})
            .set_page_size(in_cb, cb_page_size_aligned);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    std::vector<uint32_t> reader_compile_time_args = {in_cb, element_size};
    TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {in_cb, element_size};
    TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program, reader_kernel_path, all_cores, tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program, writer_kernel_path, all_cores, tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Calculate runtime arguments helper function
    auto get_slice_runtime_args = [&](const Tensor& input_tensor,
                                      Tensor& output_tensor,
                                      const ttnn::Shape& slice_start,
                                      const ttnn::Shape& slice_end,
                                      const ttnn::Shape& slice_step,
                                      uint32_t num_cores,
                                      uint32_t total_output_rows,
                                      const std::string& reader_kernel_path) {
        const auto& input_shape = input_tensor.padded_shape();
        const auto& output_shape = output_tensor.padded_shape();
        uint32_t element_size = input_tensor.element_size();
        uint32_t tensor_rank = input_shape.rank();

        uint32_t base_rows_per_core = total_output_rows / num_cores;
        uint32_t extra_rows = total_output_rows % num_cores;

        std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores);

        uint32_t row_start_id = 0;
        uint32_t extra_rows_remaining = extra_rows;

        bool using_4d_kernels = (reader_kernel_path.find("4d") != std::string::npos);

        for (uint32_t core_idx = 0; core_idx < num_cores; ++core_idx) {
            uint32_t rows_for_this_core = base_rows_per_core;
            if (extra_rows_remaining > 0) {
                rows_for_this_core += 1;
                extra_rows_remaining -= 1;
            }

            std::vector<uint32_t> reader_args, writer_args;

            if (using_4d_kernels) {
                reader_args = {
                    input_tensor.buffer()->address(),
                    tensor_rank,
                    input_shape[-1],
                    input_shape[-2],
                    input_shape[-3],
                    input_shape[-4],
                    output_shape[-1],
                    output_shape[-2],
                    output_shape[-3],
                    output_shape[-4],
                    slice_start[-1],
                    slice_end[-1],
                    slice_step[-1],
                    slice_start[-2],
                    slice_end[-2],
                    slice_step[-2],
                    slice_start[-3],
                    slice_end[-3],
                    slice_step[-3],
                    slice_start[-4],
                    slice_end[-4],
                    slice_step[-4],
                    element_size,
                    rows_for_this_core,
                    row_start_id};

                writer_args = {
                    output_tensor.buffer()->address(),
                    tensor_rank,
                    output_shape[-1],
                    output_shape[-2],
                    output_shape[-3],
                    output_shape[-4],
                    element_size,
                    rows_for_this_core,
                    row_start_id};
            } else {
                reader_args = {
                    input_tensor.buffer()->address(), tensor_rank, element_size, rows_for_this_core, row_start_id};
                reader_args.insert(reader_args.end(), input_shape.cbegin(), input_shape.cend());
                reader_args.insert(reader_args.end(), output_shape.cbegin(), output_shape.cend());
                reader_args.insert(reader_args.end(), slice_start.cbegin(), slice_start.cend());
                reader_args.insert(reader_args.end(), slice_end.cbegin(), slice_end.cend());
                reader_args.insert(reader_args.end(), slice_step.cbegin(), slice_step.cend());

                writer_args = {
                    output_tensor.buffer()->address(), tensor_rank, element_size, rows_for_this_core, row_start_id};
                writer_args.insert(writer_args.end(), output_shape.cbegin(), output_shape.cend());
            }

            ret_val[core_idx] = {reader_args, writer_args};
            row_start_id += rows_for_this_core;
        }

        return ret_val;
    };

    auto all_runtime_args = get_slice_runtime_args(
        input_tensor,
        output,
        args.slice_start,
        args.slice_end,
        args.step,
        num_cores,
        total_output_rows,
        reader_kernel_path);

    auto all_cores_vec = corerange_to_cores(all_cores);
    for (size_t i = 0; i < all_cores_vec.size(); ++i) {
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, all_cores_vec[i], all_runtime_args[i].first);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, all_cores_vec[i], all_runtime_args[i].second);
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, all_cores_vec}};
}

void SliceRmStrideProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& src_tensor = tensor_args.input;
    const auto& dst_tensor = output;
    const auto& program = cached_program.program;
    const auto& all_cores_vec = cached_program.shared_variables.all_cores_vec;

    for (size_t i = 0; i < cached_program.shared_variables.all_cores_vec.size(); ++i) {
        auto& reader_runtime_args =
            tt::tt_metal::GetRuntimeArgs(program, cached_program.shared_variables.reader_kernel_id, all_cores_vec[i]);
        reader_runtime_args[0] = src_tensor.buffer()->address();

        auto& writer_runtime_args =
            tt::tt_metal::GetRuntimeArgs(program, cached_program.shared_variables.writer_kernel_id, all_cores_vec[i]);
        writer_runtime_args[0] = dst_tensor.buffer()->address();
    }
}

// Slice Tile Program Factory implementation
SliceTileProgramFactory::cached_program_t SliceTileProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    tt::tt_metal::IDevice* device = input.device();

    uint32_t num_unpadded_tiles = output.physical_volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_tiles)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    tt::tt_metal::Buffer* src0_buffer = input.buffer();

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    std::uint32_t num_dims = static_cast<std::uint32_t>(input.padded_shape().rank());

    // Reader compile-time args
    // Data is 32 byte aligned
    std::vector<uint32_t> reader_compile_time_args = {src0_cb_index, num_dims};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {src0_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "reader_unary_unpad_dims_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);
    detail::set_slice_runtime_args_tile<true>(
        input,
        output,
        args.slice_start,
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        num_tiles_per_core_group_1,
        num_tiles_per_core_group_2,
        program,
        unary_reader_kernel_id,
        unary_writer_kernel_id,
        accumulated_total_per_dim);

    return {
        std::move(program),
        {unary_reader_kernel_id,
         unary_writer_kernel_id,
         compute_with_storage_grid_size,
         args.sub_core_grids,
         accumulated_total_per_dim}};
}

void SliceTileProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const Tensor& src_tensor = tensor_args.input;
    const Tensor& dst_tensor = output;
    const auto& slice_start = args.slice_start;
    const auto& sub_core_grids = cached_program.shared_variables.sub_core_grids;
    const auto& compute_with_storage_grid_size = cached_program.shared_variables.compute_with_storage_grid_size;
    uint32_t num_unpadded_tiles = dst_tensor.physical_volume() / TILE_HW;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(sub_core_grids.value(), num_unpadded_tiles)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    detail::set_slice_runtime_args_tile<false>(
        src_tensor,
        dst_tensor,
        slice_start,
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        num_tiles_per_core_group_1,
        num_tiles_per_core_group_2,
        cached_program.program,
        cached_program.shared_variables.unary_reader_kernel_id,
        cached_program.shared_variables.unary_writer_kernel_id,
        cached_program.shared_variables.accumulated_total_per_dim);
}

// Slice Tile Tensor Args Program Factory implementation
SliceTileTensorArgsProgramFactory::cached_program_t SliceTileTensorArgsProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& input_tensor = tensor_args.input;
    const auto& start_tensor = tensor_args.start_tensor.value();
    const auto& end_tensor = tensor_args.end_tensor.value();

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    tt::tt_metal::IDevice* device = input_tensor.device();

    uint32_t num_unpadded_tiles = output.physical_volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_tiles)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    tt::tt_metal::Buffer* src_buffer = input_tensor.buffer();
    tt::tt_metal::Buffer* start_buffer = start_tensor.buffer();
    tt::tt_metal::Buffer* end_buffer = end_tensor.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();

    TT_ASSERT(src_buffer != nullptr, "Input buffer should be allocated on device!");
    TT_ASSERT(start_buffer != nullptr, "Start buffer should be allocated on device!");
    TT_ASSERT(end_buffer != nullptr, "End buffer should be allocated on device!");
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    uint32_t src0_cb_index = 0;
    uint32_t tensor_cb_index = 1;
    uint32_t num_input_tiles = 2;

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    tt::tt_metal::CircularBufferConfig cb_tensor_config =
        tt::tt_metal::CircularBufferConfig(single_tile_size, {{tensor_cb_index, cb_data_format}})
            .set_page_size(tensor_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_tensor_config);

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_tensor.padded_shape().rank());
    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    uint32_t tile_width = tile_shape[1];
    uint32_t tile_height = tile_shape[0];

    std::vector<uint32_t> reader_compile_time_args = {
        src0_cb_index, tensor_cb_index, num_dims, tile_width, tile_height};
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*start_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*end_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {src0_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);
    detail::set_slice_runtime_args_tensor_args<true>(
        input_tensor,
        start_tensor,
        end_tensor,
        output,
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        num_tiles_per_core_group_1,
        num_tiles_per_core_group_2,
        program,
        unary_reader_kernel_id,
        unary_writer_kernel_id,
        accumulated_total_per_dim);

    return {
        std::move(program),
        {unary_reader_kernel_id,
         unary_writer_kernel_id,
         compute_with_storage_grid_size,
         args.sub_core_grids,
         accumulated_total_per_dim}};
}

void SliceTileTensorArgsProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const Tensor& src_tensor = tensor_args.input;
    const Tensor& start_tensor = tensor_args.start_tensor.value();
    const Tensor& end_tensor = tensor_args.end_tensor.value();
    const Tensor& dst_tensor = output;
    const auto& sub_core_grids = cached_program.shared_variables.sub_core_grids;
    const auto& compute_with_storage_grid_size = cached_program.shared_variables.compute_with_storage_grid_size;
    uint32_t num_unpadded_tiles = dst_tensor.physical_volume() / TILE_HW;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(sub_core_grids.value(), num_unpadded_tiles)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    detail::set_slice_runtime_args_tensor_args<false>(
        src_tensor,
        start_tensor,
        end_tensor,
        dst_tensor,
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        num_tiles_per_core_group_1,
        num_tiles_per_core_group_2,
        cached_program.program,
        cached_program.shared_variables.unary_reader_kernel_id,
        cached_program.shared_variables.unary_writer_kernel_id,
        cached_program.shared_variables.accumulated_total_per_dim);
}

}  // namespace slice::program

}  // namespace ttnn::operations::data_movement
