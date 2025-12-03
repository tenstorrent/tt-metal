// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/host_buffer/functions.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/math.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "pad_program_factory.hpp"

static const uint32_t max_read_size = 2048;  // max read size in bytes for reader and writer kernels
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::pad::program {

inline void log_rt_args(const CoreCoord& core, std::vector<uint32_t>& args) {
    for ([[maybe_unused]] auto v : args) {
        log_debug(tt::LogOp, "{},{} :: {}", core.x, core.y, v);
    }
}

// This is currently mostly hardcoded for resnet shapes
inline std::tuple<uint32_t, uint32_t, uint32_t, CoreRangeSet, CoreRangeSet, uint32_t, uint32_t, uint32_t, uint32_t>
split_across_cores(CoreCoord grid_size, uint32_t nbatch, uint32_t nchannel, uint32_t ntiles_h, uint32_t ntiles_w) {
    uint32_t ncores, ncores_h, ncores_w, ntiles_per_core_h, ntiles_per_core_w, nbatch_per_core_h, ncores_per_batch_h;

    ncores_h = 1;

    // each batch needs to be padded independently
    switch (nbatch) {
        case 1:
            ncores_h = 1;
            nbatch_per_core_h = 1;
            ntiles_per_core_h = 1;
            switch (ntiles_h) {
                case 2:
                    ncores_h = 2;
                    ntiles_per_core_h = 1;
                    break;
                case 4:
                    ncores_h = 4;
                    ntiles_per_core_h = 1;
                    break;
                case 8:
                    ncores_h = 8;
                    ntiles_per_core_h = 1;
                    break;
                case 64:
                    ncores_h = 8;
                    ntiles_per_core_h = 8;
                    break;
                default: TT_THROW("Unsupported ntiles_h value {}", ntiles_h);
            }
            ncores_per_batch_h = ncores_h;
            break;

        case 2:
            ncores_h = 1;
            ncores_per_batch_h = 1;
            nbatch_per_core_h = 1;
            ntiles_per_core_h = 1;
            switch (ntiles_h) {
                case 2:
                    ncores_per_batch_h = 2;
                    ncores_h = ncores_per_batch_h * nbatch;
                    ntiles_per_core_h = 1;
                    break;
                case 4:
                    ncores_per_batch_h = 4;
                    ncores_h = ncores_per_batch_h * nbatch;
                    ntiles_per_core_h = 1;
                    break;
                case 8:
                    ncores_per_batch_h = 4;
                    ncores_h = ncores_per_batch_h * nbatch;
                    ntiles_per_core_h = 2;
                    break;
                case 64:
                    ncores_per_batch_h = 4;
                    ncores_h = ncores_per_batch_h * nbatch;
                    ntiles_per_core_h = 16;
                    break;
                default: TT_THROW("Unsupported ntiles_h value {}", ntiles_h);
            }
            break;

        case 8:
            ncores_h = 8;
            ncores_per_batch_h = 1;
            nbatch_per_core_h = 1;
            ntiles_per_core_h = ntiles_h;
            break;

        default:
            TT_ASSERT(false, "unhandled nbatch. TODO");

            // generic case -- TODO

            // one of the following will be 0 when grid_size.y != nbatch
            uint32_t nbatch_per_core_h = nbatch / grid_size.y;   // floor
            uint32_t ncores_per_batch_h = grid_size.y / nbatch;  // floor
            if (nbatch == grid_size.y) {
                nbatch_per_core_h = 1;
                ncores_per_batch_h = 1;
            }

            // currently uses hardcoded values for resnet50
            // TT_ASSERT(ntiles_h == 1 || ntiles_h == 2 || ntiles_h == 4 || ntiles_h == 16, "Only Resnet50 shapes are
            // supported in multicore version for now."); TT_ASSERT(ntiles_w == 64, "Only Resnet50 shapes are supported
            // in multicore version for now.");

            TT_ASSERT(nbatch <= grid_size.y, "Unsupported case with nbatch > grid_size.y!");

            if (nbatch_per_core_h == 0) {
                // there are multiple cores along h per batch
                nbatch_per_core_h = 1;
            } else if (ncores_per_batch_h == 0) {
                // unsupported case. TODO.
                TT_ASSERT(false);
                // there are multiple batch per core along h
                // ncores_per_batch_h = 1;
            } else {
                TT_THROW("Something went terribly wrong in splitting acrtoss cores");
            }
            break;
    }

    ncores_w = 1;
    switch (ntiles_w) {
        case 2: ncores_w = 2; break;
        case 4: ncores_w = 4; break;
        case 8:
        case 64: ncores_w = 8; break;
        default: TT_THROW("Unsupported ntiles_w value {}", ntiles_w);
    }
    ncores = ncores_h * ncores_w;
    ntiles_per_core_w = ntiles_w / ncores_w;
    std::set<CoreRange> all_cores;
    std::set<CoreRange> core_range;

    all_cores.insert(CoreRange(CoreCoord(0, 0), CoreCoord(ncores_w - 1, ncores_h - 1)));
    core_range.insert(CoreRange(CoreCoord(0, 0), CoreCoord(ncores_w - 1, ncores_h - 1)));

    return std::make_tuple(
        ncores,
        ncores_h,
        ncores_w,
        all_cores,
        core_range,
        ntiles_per_core_h,
        ntiles_per_core_w,
        nbatch_per_core_h,
        ncores_per_batch_h);
}

uint32_t get_num_stick_per_barrier(const Tensor& input_tensor) {
    uint32_t W = input_tensor.padded_shape()[3];
    uint32_t W_bytes = W * input_tensor.element_size();
    uint32_t num_stick_per_barrier = 0;
    for (uint32_t cur_bytes = 0; cur_bytes < max_read_size; cur_bytes += W_bytes) {
        num_stick_per_barrier++;
    }
    return num_stick_per_barrier;
}

std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_runtime_args_rm(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& input_tensor_start,
    uint32_t num_cores_total,
    uint32_t num_cores,
    uint32_t num_cores_y,
    const CoreRangeSet& core_group_1,
    uint32_t num_w_sticks_per_core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_w_sticks_per_core_group_2) {
    auto* input_buffer = input_tensor.buffer();
    auto* output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.padded_shape();
    auto output_shape = output_tensor.padded_shape();

    uint32_t H = input_shape[2], C = input_shape[1], N = input_shape[0];

    uint32_t H_padded = output_shape[2], C_padded = output_shape[1];

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> start_dim_offset(num_dims, 0);

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores_total);

    const auto& front_pad = input_tensor_start;
    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;
    for (uint32_t i = 0, curr_sticks_read = 0, curr_sticks_write = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_sticks_per_core;
        if (core_group_1.contains(core)) {
            num_sticks_per_core = num_w_sticks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_sticks_per_core = num_w_sticks_per_core_group_2;
        } else {
            // no-op
            num_sticks_per_core = 0;
        }

        uint32_t num_sticks_per_barrier = get_num_stick_per_barrier(input_tensor);
        // reader
        std::vector<uint32_t> reader_runtime_args = {
            input_buffer->address(),
            num_sticks_per_core,
            num_sticks_per_barrier,
            curr_sticks_read,
            front_pad[-4],
            front_pad[-3],
            front_pad[-2],
        };
        reader_runtime_args.insert(reader_runtime_args.end(), start_dim_offset.begin(), start_dim_offset.end());

        // writer
        std::vector<uint32_t> writer_runtime_args = {
            output_buffer->address(), num_sticks_per_core, num_sticks_per_barrier, curr_sticks_write};

        ret_val[i] = {reader_runtime_args, writer_runtime_args};

        curr_sticks_write += num_sticks_per_core;

        for (uint32_t i = 0; i < num_sticks_per_core; ++i) {
            if ((curr_h >= front_pad[-2] and curr_h < (H + front_pad[-2])) and
                (curr_c >= front_pad[-3] and curr_c < (C + front_pad[-3])) and
                (curr_n >= front_pad[-4] and curr_n < (N + front_pad[-4]))) {
                curr_sticks_read++;
            }

            curr_h++;
            if (curr_h == H_padded) {
                curr_c++;
                curr_h = 0;
                if (curr_c == C_padded) {
                    curr_n++;
                    curr_c = 0;
                }
            }
        }

        start_dim_offset = {0, curr_h, curr_c, curr_n};
    }

    return ret_val;
}

inline std::vector<std::vector<uint32_t>> group_contiguous_and_repeated_values(std::vector<uint32_t>& values) {
    std::vector<std::vector<uint32_t>> chunks;
    if (values.empty()) {
        return chunks;
    }

    // Initialize the first chunk
    std::vector<uint32_t> current_chunk;
    current_chunk.push_back(values[0]);

    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] == values[i - 1] + 1 or values[i] == values[i - 1]) {
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

inline std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_pad_runtime_args_rm_sharded(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& input_tensor_start,
    uint32_t num_cores_padded,
    bool row_major,
    uint32_t num_cores_x_padded,
    uint32_t num_cores_y_padded,
    uint32_t shard_height_padded,
    uint32_t shard_height_unpadded,
    uint32_t num_cores_x_unpadded,
    uint32_t num_cores_y_unpadded) {
    tt::tt_metal::IDevice* device = input_tensor.device();

    auto input_shape = input_tensor.padded_shape();
    auto output_shape = output_tensor.padded_shape();

    uint32_t H = input_shape[2], C = input_shape[1], N = input_shape[0];

    uint32_t H_padded = output_shape[2], C_padded = output_shape[1];

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> start_dim_offset(num_dims, 0);

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores_padded);

    const auto& front_pad = input_tensor_start;
    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;
    for (uint32_t i = 0, curr_sticks_read = 0; i < num_cores_padded; i++) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x_padded, i / num_cores_x_padded};
        } else {
            core = {i / num_cores_y_padded, i % num_cores_y_padded};
        }
        uint32_t num_sticks_per_core_unpadded = shard_height_unpadded;
        uint32_t num_sticks_per_core_padded = shard_height_padded;

        // writer rt args, set on top here as interleaved version.
        std::vector<uint32_t> writer_kernel_args = {
            num_sticks_per_core_padded,
            curr_sticks_read,
            front_pad[-4],
            front_pad[-3],
            front_pad[-2],
        };
        writer_kernel_args.insert(writer_kernel_args.end(), start_dim_offset.begin(), start_dim_offset.end());

        // figure out the start read stick id for each core, and the start id for each dim
        std::vector<int> stick_ids_per_core;
        int front_pad_stick_id = -2;
        int pad_stick_id = -1;
        for (uint32_t i = 0; i < num_sticks_per_core_padded; ++i) {
            if ((curr_h >= front_pad[-2] and curr_h < (H + front_pad[-2])) and
                (curr_c >= front_pad[-3] and curr_c < (C + front_pad[-3])) and
                (curr_n >= front_pad[-4] and curr_n < (N + front_pad[-4]))) {
                stick_ids_per_core.push_back(curr_sticks_read);
                curr_sticks_read++;
            } else {
                if (curr_h < front_pad[-2] or curr_c < front_pad[-3] or curr_n < front_pad[-4]) {
                    stick_ids_per_core.push_back(front_pad_stick_id);
                } else {
                    stick_ids_per_core.push_back(pad_stick_id);
                }
            }

            curr_h++;
            if (curr_h == H_padded) {
                curr_c++;
                curr_h = 0;
                if (curr_c == C_padded) {
                    curr_n++;
                    curr_c = 0;
                }
            }
        }

        start_dim_offset = {0, curr_h, curr_c, curr_n};

        // figure out the stick id in a shard, and the core id for the stick.
        std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> core_stick_map;
        auto first_core = device->worker_core_from_logical_core(CoreCoord{0, 0});
        std::pair<uint32_t, uint32_t> prev_xy_pair = std::make_pair(first_core.x, first_core.y);
        for (uint32_t j = 0; j < num_sticks_per_core_padded; ++j) {
            int stick_id = stick_ids_per_core[j];

            // if it is pad stick, we need to leave a gap between the previous non-pad stick and next non-pad stick.
            if (stick_id == -2) {  // front padding
                core_stick_map[prev_xy_pair].push_back(stick_id);
            } else if (stick_id == -1) {  // end padding
                core_stick_map[prev_xy_pair].push_back(stick_id);
            } else {
                uint32_t shard_id = stick_id / num_sticks_per_core_unpadded;
                uint32_t stick_id_in_shard = stick_id - (shard_id * num_sticks_per_core_unpadded);

                uint32_t shard_grid_inner_dim = row_major ? num_cores_x_unpadded : num_cores_y_unpadded;
                uint32_t shard_grid_outer_dim_id = shard_id / shard_grid_inner_dim;
                uint32_t shard_grid_inner_dim_id = shard_id - (shard_grid_outer_dim_id * shard_grid_inner_dim);

                uint32_t worker_y_logical = row_major ? shard_grid_outer_dim_id : shard_grid_inner_dim_id;
                uint32_t worker_x_logical = row_major ? shard_grid_inner_dim_id : shard_grid_outer_dim_id;

                if (worker_x_logical < num_cores_x_unpadded and worker_y_logical < num_cores_y_unpadded) {
                    auto core_physical =
                        device->worker_core_from_logical_core(CoreCoord{worker_x_logical, worker_y_logical});
                    // save stick id in a shard, and core coord into a map
                    std::pair<uint32_t, uint32_t> xy_pair = row_major
                                                                ? std::make_pair(core_physical.y, core_physical.x)
                                                                : std::make_pair(core_physical.x, core_physical.y);
                    core_stick_map[xy_pair].push_back(stick_id_in_shard);
                    prev_xy_pair = xy_pair;
                }
            }
        }

        // reader rt args
        std::vector<uint32_t> reader_kernel_args;
        reader_kernel_args.push_back(core_stick_map.size());  // num_cores

        for (const auto& core_stick_pair : core_stick_map) {
            auto xy_pair = core_stick_pair.first;
            if (row_major) {
                reader_kernel_args.push_back((std::uint32_t)xy_pair.second);  // noc x
                reader_kernel_args.push_back((std::uint32_t)xy_pair.first);   // noc y
            } else {
                reader_kernel_args.push_back((std::uint32_t)xy_pair.first);   // noc x
                reader_kernel_args.push_back((std::uint32_t)xy_pair.second);  // noc y
            }
        }

        // coalesce the sticks into chunks
        std::vector<std::vector<std::vector<uint32_t>>> stick_chunks_per_core;
        for (auto core_stick_pair : core_stick_map) {
            auto stick_chunks = group_contiguous_and_repeated_values(core_stick_pair.second);
            stick_chunks_per_core.push_back(stick_chunks);
            reader_kernel_args.push_back(stick_chunks.size());  // num_chunks for current core
        }
        for (const auto& stick_chunks : stick_chunks_per_core) {
            for (auto chunk : stick_chunks) {
                reader_kernel_args.push_back(chunk[0]);      // start id of a chunk
                reader_kernel_args.push_back(chunk.size());  // length of a chunk
            }
        }

        ret_val[i] = {reader_kernel_args, writer_kernel_args};
    }

    return ret_val;
}

}  // namespace ttnn::operations::data_movement::pad::program
