// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>
#include <cstdint>
#include <vector>

#include "upsample_op.hpp"
#include "ttnn/operations/cb_utils.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/math.hpp>

#include <tt_stl/reflection.hpp>
#include "ttnn/tensor/host_buffer/functions.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::upsample {
using namespace tt;

struct StickInterval {
    uint16_t core_x, core_y;
    uint16_t offset_start;
    uint16_t offset_end;

    StickInterval(uint16_t cx, uint16_t cy, uint16_t start) :
        core_x(cx), core_y(cy), offset_start(start), offset_end(start) {}
};

static Tensor create_config_tensor(
    IDevice* device,
    ShardSpec shard_spec,
    const uint32_t batch_size,
    const uint32_t in_h,
    const uint32_t in_w,
    const uint32_t scale_factor_h,
    const uint32_t scale_factor_w,
    const bool is_height_sharded) {
    uint16_t core_idx = 0;        // Tracks the current core being processed
    uint16_t core_idx_start = 0;  // Tracks the starting core index for each row of input, used when scale_factor_h > 1
    uint16_t stick_offset = 0;    // Tracks the current stick offset within the core
    uint16_t stick_offset_start =
        0;  // Tracks starting stick offset within the core before adding in_w sticks, used when scale_factor_h > 1
    uint32_t stick_cnt =
        0;  // Counts the number of sticks processed, used for splitting output sticks across NCRISC and BRISC
    uint16_t ch_start_core = 0;   // Starting core index where channels are distributed
    uint16_t ch_end_core = 0;     // Ending core index where channels are distributed
    uint16_t nhw_start_core = 0;  // Starting core index for NHW distribution
    const uint32_t input_nsticks_per_core = shard_spec.shape[0];
    const uint32_t output_nsticks_per_core = input_nsticks_per_core * scale_factor_h;
    uint32_t output_nsticks_per_core_reader =
        (output_nsticks_per_core + 1) / 2;  // Total number of sticks per core in the output

    auto logical_cores = corerange_to_cores(
        shard_spec.grid, shard_spec.num_cores(), shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    auto ranges = shard_spec.grid.ranges();

    if (!is_height_sharded) {
        auto all_cores = shard_spec.grid;
        if (shard_spec.orientation == ShardOrientation::ROW_MAJOR) {
            ch_start_core = all_cores.ranges().begin()->start_coord.x;
            ch_end_core = all_cores.ranges().begin()->end_coord.x;
            nhw_start_core = all_cores.ranges().begin()->start_coord.y;
        } else {
            ch_start_core = all_cores.ranges().begin()->start_coord.y;
            ch_end_core = all_cores.ranges().begin()->end_coord.y;
            nhw_start_core = all_cores.ranges().begin()->start_coord.x;
        }
    }

    std::vector<StickInterval> logical_core_to_stick_map;
    std::vector<uint16_t> dst_core_end_idx_map;  // used for padding the config vector per reader

    bool reader_sticks_reached = false;
    bool insert_new_interval = false;
    uint32_t last_ind = 0;
    uint32_t elems_per_core_reader = 0;  // represents number of intervals per reader (NCRISC/BRISC)
    uint32_t elem_num = 0;
    // Create map of core and respective offsets in input
    for (uint32_t b = 0; b < batch_size; ++b) {
        for (uint32_t h = 0; h < in_h; ++h) {
            for (uint32_t j = 0; j < scale_factor_h; ++j) {
                stick_offset_start = stick_offset;
                core_idx_start = core_idx;

                for (uint32_t w = 0; w < in_w; ++w, ++stick_offset, ++stick_cnt) {
                    if (stick_offset == input_nsticks_per_core) {
                        stick_offset = 0;
                        core_idx++;
                    }
                    reader_sticks_reached = stick_cnt == output_nsticks_per_core_reader;
                    insert_new_interval = insert_new_interval || stick_offset == 0 || reader_sticks_reached;

                    if (!insert_new_interval) {
                        logical_core_to_stick_map.back().offset_end++;
                        continue;
                    }
                    if (reader_sticks_reached) {
                        dst_core_end_idx_map.push_back(logical_core_to_stick_map.size());
                        if (output_nsticks_per_core > 1) {
                            output_nsticks_per_core_reader = output_nsticks_per_core - output_nsticks_per_core_reader;
                        }
                        stick_cnt = 0;
                        elem_num = logical_core_to_stick_map.size() - last_ind;
                        elems_per_core_reader = elem_num > elems_per_core_reader ? elem_num : elems_per_core_reader;
                        last_ind = logical_core_to_stick_map.size();
                    }
                    insert_new_interval = false;
                    if (is_height_sharded) {
                        logical_core_to_stick_map.emplace_back(
                            logical_cores[core_idx].x, logical_cores[core_idx].y, stick_offset);
                    } else {
                        logical_core_to_stick_map.emplace_back(0, nhw_start_core + core_idx, stick_offset);
                    }
                }
                if (j < scale_factor_h - 1) {
                    stick_offset = stick_offset_start;
                    core_idx = core_idx_start;
                    insert_new_interval = true;  // insert new entry in next loop
                }
            }
        }
    }
    dst_core_end_idx_map.push_back(logical_core_to_stick_map.size());

    elem_num = logical_core_to_stick_map.size() - last_ind;
    elems_per_core_reader = elem_num > elems_per_core_reader ? elem_num : elems_per_core_reader;

    /* Each entry in config_vector contains 4 elements:
     * {core_coords.x, core_coords.y, stick_offset_start, stick_offset_end(in input_cb)}
     * - core_coords.x: X coordinate of the core
     * - core_coords.y: Y coordinate of the core
     * - stick_offset_start: Offset start within the input circular buffer
     * - stick_offset_end: Offset end within the input circular buffer
     */
    std::vector<uint16_t> config_vector;

    const uint32_t config_buffer_entry_size = 4;
    elems_per_core_reader *= config_buffer_entry_size;
    const uint32_t elems_per_core =
        2 * elems_per_core_reader;  // because two readers per tensix core which get equal number of stick intervals

    // Based on core calculate physical location of cores
    CoreCoord core_coords;

    // In case last input shard is not full, fill the rest of the config vector with placeholder values
    const auto pad_uneven_shards = [config_buffer_entry_size](
                                       auto& config_vector, uint32_t elems_per_core_reader, size_t slice_begin = 0) {
        const uint32_t slice_length = config_vector.size() - slice_begin;
        const uint32_t remainder =
            (elems_per_core_reader - (slice_length % elems_per_core_reader)) % elems_per_core_reader;
        if (remainder != 0) {
            for (int i = 0; i < remainder / config_buffer_entry_size; i++) {
                config_vector.push_back(0);  // core x
                config_vector.push_back(0);  // core y
                config_vector.push_back(1);  // stick offset start
                config_vector.push_back(0);  // stick offset end
            }
        }
    };

    uint32_t per_core_start_idx = 0;
    for (size_t i = ch_start_core; i <= ch_end_core; i++) {
        for (size_t ind = 0, j = 0; ind < dst_core_end_idx_map.size(); ind++) {
            const size_t chan_slice_begin = config_vector.size();
            if (ind % 2 == 0) {
                per_core_start_idx = config_vector.size();
            }
            for (; j < dst_core_end_idx_map[ind]; ++j) {
                core_coords = device->worker_core_from_logical_core(
                    is_height_sharded
                        ? CoreCoord(logical_core_to_stick_map[j].core_x, logical_core_to_stick_map[j].core_y)
                        : CoreCoord(i, logical_core_to_stick_map[j].core_y));
                // Combine the x and y coordinates of the core into a single 16-bit value.
                config_vector.push_back(core_coords.x);
                config_vector.push_back(core_coords.y);
                config_vector.push_back(logical_core_to_stick_map[j].offset_start);
                config_vector.push_back(logical_core_to_stick_map[j].offset_end);
            }
            if (output_nsticks_per_core > 1) {
                pad_uneven_shards(config_vector, elems_per_core_reader, chan_slice_begin);
            } else {
                pad_uneven_shards(config_vector, elems_per_core, chan_slice_begin);
            }
        }
        pad_uneven_shards(config_vector, elems_per_core, per_core_start_idx);
    }

    TT_FATAL(
        config_vector.size() % elems_per_core == 0,
        "Config vector size {} should be multiple of {}",
        config_vector.size(),
        elems_per_core);

    ttnn::Shape config_shape({tt::div_up(config_vector.size(), elems_per_core), elems_per_core});
    auto config_buffer = HostBuffer(std::move(config_vector));
    return Tensor(std::move(config_buffer), config_shape, DataType::UINT16, Layout::ROW_MAJOR);
}

operation::ProgramWithCallbacks upsample_multi_core_sharded(
    const Tensor& input, Tensor& output, const uint32_t scale_factor_h, const uint32_t scale_factor_w) {
    Program program = CreateProgram();
    distributed::MeshDevice* device = input.device();

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    TT_FATAL(input.logical_shape()[-1] == output.logical_shape()[-1], "Expected input and output channels to match");
    TT_FATAL(
        input.layout() == tt_metal::Layout::ROW_MAJOR,
        "Only row-major layout is currently supported in nearest upsample");

    uint32_t input_stick_nbytes = input.padded_shape()[-1] * input.element_size();
    uint32_t output_stick_nbytes = output.padded_shape()[-1] * output.element_size();
    TT_FATAL(input_stick_nbytes == output_stick_nbytes, "Input and output sticks should have same size");

    uint32_t in_w = input.padded_shape()[2];

    auto shard_spec = input.shard_spec().value();
    auto all_cores = shard_spec.grid;
    uint32_t ncores = shard_spec.num_cores();
    uint32_t ncores_x = device->compute_with_storage_grid_size().x;

    auto out_shard_spec = output.shard_spec().value();
    TT_FATAL(
        out_shard_spec.num_cores() == ncores,
        "Output tensor should have same number of cores {} as input tensor {}",
        out_shard_spec.num_cores(),
        ncores);

    if (input.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        TT_THROW("Unsupported sharding layout");
    }

    // extra limitation to avoid post upsample step of resharding
    if (input.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        ncores_x = all_cores.ranges().begin()->end_coord.x - all_cores.ranges().begin()->start_coord.x + 1;
        input_stick_nbytes = input_stick_nbytes / ncores_x;
        output_stick_nbytes = output_stick_nbytes / ncores_x;
    }

    const uint32_t input_nsticks_per_core = shard_spec.shape[0];
    const uint32_t output_nsticks_per_core = input_nsticks_per_core * scale_factor_h * scale_factor_w;

    uint32_t next_cb_index = CBIndex::c_0;
    const uint32_t buffering_factor = 1;  // data is already fully buffered in the CBs since its sharded
    const uint32_t aligned_input_stick_nbytes = round_up_to_mul32(input_stick_nbytes);
    const uint32_t in_cb_pagesize = aligned_input_stick_nbytes;
    const uint32_t in_cb_npages = input_nsticks_per_core * buffering_factor;

    auto [in_cb_id, cb_src0] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, in_cb_pagesize, in_cb_npages, input_cb_data_format, input.buffer());

    // output sharded CB with upsampled data
    uint32_t out_cb_pagesize = round_up_to_mul32(output_stick_nbytes);  // aligned output stick n bytes
    uint32_t out_cb_npages = output_nsticks_per_core * buffering_factor;

    auto [out_cb_id, out_cb] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, out_cb_pagesize, out_cb_npages, output_cb_data_format, output.buffer());

    log_debug(LogOp, "input_cb: {}, npages: {}, pagesize: {}", in_cb_id, in_cb_npages, in_cb_pagesize);
    log_debug(LogOp, "output_cb: {}, npages: {}, pagesize: {}", out_cb_id, out_cb_npages, out_cb_pagesize);
    log_debug(LogOp, "input_stick_nbytes: {}, output_stick_nbytes: {}", input_stick_nbytes, output_stick_nbytes);
    log_debug(LogOp, "ncores: {}, ncores_x: {}", ncores, ncores_x);
    log_debug(
        LogOp,
        "input_nsticks_per_core: {}, output_nsticks_per_core: {}",
        input_nsticks_per_core,
        output_nsticks_per_core);

    // create config tensor
    Tensor config_tensor;
    if ((input.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) ||
        (input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED)) {
        config_tensor = create_config_tensor(
            device,
            shard_spec,
            input.padded_shape()[0],
            input.padded_shape()[1],
            in_w,
            scale_factor_h,
            scale_factor_w,
            input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED);
    } else {
        TT_THROW("Unsupported sharding layout");
    }
    auto shard_shape = std::array<uint32_t, 2>({1, (uint32_t)config_tensor.logical_shape()[-1]});
    auto config_tensor_shard_orientation = input.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED
                                               ? ShardOrientation::COL_MAJOR
                                               : shard_spec.orientation;
    ShardSpec config_shard_spec(input.shard_spec().value().grid, shard_shape, config_tensor_shard_orientation);
    MemoryConfig memory_config{TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1_SMALL, config_shard_spec};
    auto config_tensor_device = config_tensor.to_device(device, memory_config);

    tt::DataFormat config_df = tt::DataFormat::RawUInt16;
    const auto& config_storage = config_tensor_device.device_storage();
    auto config_buffer = config_storage.get_buffer();
    auto config_buffer_page_size = config_buffer->page_size();

    auto [config_cb_id, config_cb] = tt::tt_metal::create_cb(
        next_cb_index++, program, all_cores, config_buffer_page_size, 1, config_df, &*config_buffer);

    // Kernels

    std::vector<uint32_t> writer_compile_time_args = {
        in_cb_id,
        out_cb_id,
        false,
        config_cb_id,
        input_stick_nbytes,
        input_nsticks_per_core,
        scale_factor_h,
        scale_factor_w,
        // number of intervals in config tensor per core, 4 is number of bfloat16 elements per entry
        static_cast<uint32_t>(config_tensor.logical_shape()[-1] / 4),
    };
    std::string writer_kernel_fname =
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/writer_upsample_multi_core_sharded.cpp";
    auto writer_kernel =
        CreateKernel(program, writer_kernel_fname, all_cores, WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> reader_compile_time_args = writer_compile_time_args;
    reader_compile_time_args[2] = true;  // reader

    std::string reader_kernel_fname =
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/writer_upsample_multi_core_sharded.cpp";
    CreateKernel(program, reader_kernel_fname, all_cores, ReaderDataMovementConfig(reader_compile_time_args));

    // Capture config_buffer to cache this with the program
    auto override_runtime_args_callback = [writer_kernel, cb_src0, out_cb, config_cb, config_storage, config_buffer](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
        UpdateDynamicCircularBufferAddress(program, out_cb, *dst_buffer);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::upsample
