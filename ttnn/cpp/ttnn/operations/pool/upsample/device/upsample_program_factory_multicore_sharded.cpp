// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/pool/upsample/device/upsample_device_operation.hpp"

#include <sys/types.h>
#include <cstdint>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/pool/upsample/device/upsample_common.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {
using namespace tt;

namespace {

struct StickInterval {
    uint16_t core_x, core_y;
    uint16_t offset_start;
    uint16_t offset_end;

    StickInterval(uint16_t cx, uint16_t cy, uint16_t start) :
        core_x(cx), core_y(cy), offset_start(start), offset_end(start) {}
};

Tensor create_config_tensor(
    IDevice* device,
    ShardSpec shard_spec,
    const uint32_t batch_size,
    const uint32_t in_h,
    const uint32_t in_w,
    const uint32_t scale_factor_h,
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
            for (uint32_t i = 0; i < remainder / config_buffer_entry_size; i++) {
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

// Returns a reduced CoreRangeSet containing only cores that have actual work.
// For height sharding: returns first N cores from the grid.
// For block sharding: keeps all channel cores, reduces NHW dimension.
CoreRangeSet get_cores_with_work(
    const CoreRangeSet& all_cores,
    uint32_t total_nhw,
    uint32_t nsticks_per_core,
    bool is_height_sharded,
    ShardOrientation orientation) {
    const uint32_t num_cores = all_cores.num_cores();
    const uint32_t actual_nhw_cores = tt::div_up(total_nhw, nsticks_per_core);

    if (is_height_sharded) {
        if (actual_nhw_cores >= num_cores) {
            return all_cores;
        }
        return num_cores_to_corerangeset_in_subcoregrids(
            all_cores.ranges().begin()->start_coord,
            actual_nhw_cores,
            all_cores,
            orientation == ShardOrientation::ROW_MAJOR);
    }

    // Block sharding: keep all channel cores, reduce NHW dimension
    const auto& range = *all_cores.ranges().begin();
    const bool row_major = orientation == ShardOrientation::ROW_MAJOR;

    // NHW is on Y axis for ROW_MAJOR, X axis for COL_MAJOR
    const uint32_t nhw_start = row_major ? range.start_coord.y : range.start_coord.x;
    const uint32_t nhw_end = row_major ? range.end_coord.y : range.end_coord.x;
    const uint32_t nhw_cores_in_grid = nhw_end - nhw_start + 1;
    const uint32_t nhw_cores_needed = std::min(actual_nhw_cores, nhw_cores_in_grid);

    if (nhw_cores_needed >= nhw_cores_in_grid) {
        return all_cores;
    }

    const CoreCoord new_end = row_major ? CoreCoord(range.end_coord.x, nhw_start + nhw_cores_needed - 1)
                                        : CoreCoord(nhw_start + nhw_cores_needed - 1, range.end_coord.y);
    return CoreRangeSet(CoreRange(range.start_coord, new_end));
}

}  // namespace

UpsampleMultiCoreShardedProgramFactory::Resources UpsampleMultiCoreShardedProgramFactory::prepare_resources(
    const UpsampleParams& operation_attributes, const Tensor& input_tensor, Tensor& /*output_tensor*/) {
    const auto& input = input_tensor;
    TT_FATAL(
        operations::pool::upsample::is_integer_scale(operation_attributes.scale_factor_h) &&
            operations::pool::upsample::is_integer_scale(operation_attributes.scale_factor_w),
        "Sharded upsample factory requires integer scale factors, got scale_h={}, scale_w={}",
        operation_attributes.scale_factor_h,
        operation_attributes.scale_factor_w);
    const uint32_t scale_factor_h = static_cast<uint32_t>(operation_attributes.scale_factor_h);

    distributed::MeshDevice* device = input.device();
    const auto shard_spec = input.shard_spec().value();
    const uint32_t in_w = input.padded_shape()[2];
    const uint32_t input_nsticks_per_core = shard_spec.shape[0];
    const bool is_height_sharded = input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;
    const uint32_t total_nhw = input.padded_shape()[0] * input.padded_shape()[1] * in_w;

    const TensorMemoryLayout memory_layout = input.memory_config().memory_layout();
    TT_FATAL(
        memory_layout == TensorMemoryLayout::HEIGHT_SHARDED || memory_layout == TensorMemoryLayout::BLOCK_SHARDED,
        "Unsupported sharding layout");

    const CoreRangeSet cores_with_work = get_cores_with_work(
        shard_spec.grid, total_nhw, input_nsticks_per_core, is_height_sharded, shard_spec.orientation);

    Tensor config_tensor = create_config_tensor(
        device, shard_spec, input.padded_shape()[0], input.padded_shape()[1], in_w, scale_factor_h, is_height_sharded);

    const auto shard_shape = std::array<uint32_t, 2>({1, static_cast<uint32_t>(config_tensor.logical_shape()[-1])});
    const auto config_tensor_shard_orientation =
        input.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ? ShardOrientation::COL_MAJOR
                                                                                   : shard_spec.orientation;
    // Use cores_with_work for config tensor sharding - only cores that have actual work need config data
    const ShardSpec config_shard_spec(cores_with_work, shard_shape, config_tensor_shard_orientation);
    const MemoryConfig config_memory_config{
        TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1_SMALL, config_shard_spec};

    return Resources{.config_tensor_device = config_tensor.to_device(device, config_memory_config)};
}

namespace {
const Tensor& require_resources(const std::optional<Tensor>& cfg) {
    TT_FATAL(cfg.has_value(), "prepare_resources must run before create_descriptor for sharded upsample");
    return *cfg;
}
}  // namespace

ProgramDescriptor UpsampleMultiCoreShardedProgramFactory::create_descriptor(
    const UpsampleParams& operation_attributes,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    Resources& resources) {
    const auto& input = input_tensor;
    auto& output = output_tensor;

    const uint32_t scale_factor_h = static_cast<uint32_t>(operation_attributes.scale_factor_h);
    const uint32_t scale_factor_w = static_cast<uint32_t>(operation_attributes.scale_factor_w);

    distributed::MeshDevice* device = input.device();

    const tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());

    TT_FATAL(input.logical_shape()[-1] == output.logical_shape()[-1], "Expected input and output channels to match");
    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "Only row-major layout is currently supported in nearest upsample");

    uint32_t input_stick_nbytes = input.padded_shape()[-1] * input.element_size();
    uint32_t output_stick_nbytes = output.padded_shape()[-1] * output.element_size();
    TT_FATAL(input_stick_nbytes == output_stick_nbytes, "Input and output sticks should have same size");

    const uint32_t in_w = input.padded_shape()[2];

    const auto shard_spec = input.shard_spec().value();
    const auto all_cores = shard_spec.grid;
    const uint32_t ncores = shard_spec.num_cores();
    uint32_t ncores_x = device->compute_with_storage_grid_size().x;

    const auto out_shard_spec = output.shard_spec().value();
    TT_FATAL(
        out_shard_spec.num_cores() == ncores,
        "Output tensor should have same number of cores {} as input tensor {}",
        out_shard_spec.num_cores(),
        ncores);

    // extra limitation to avoid post upsample step of resharding
    if (input.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        ncores_x = all_cores.ranges().begin()->end_coord.x - all_cores.ranges().begin()->start_coord.x + 1;
        input_stick_nbytes = input_stick_nbytes / ncores_x;
        output_stick_nbytes = output_stick_nbytes / ncores_x;
    }

    const uint32_t input_nsticks_per_core = shard_spec.shape[0];
    const uint32_t output_nsticks_per_core = input_nsticks_per_core * scale_factor_h * scale_factor_w;
    const bool is_height_sharded = input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;
    const uint32_t total_nhw = input.padded_shape()[0] * input.padded_shape()[1] * in_w;

    // Reduced core set - only cores that actually have work
    const CoreRangeSet cores_with_work =
        get_cores_with_work(all_cores, total_nhw, input_nsticks_per_core, is_height_sharded, shard_spec.orientation);

    ProgramDescriptor desc;

    uint32_t next_cb_index = CBIndex::c_0;
    constexpr uint32_t buffering_factor = 1;  // data is already fully buffered in the CBs since its sharded
    const uint32_t aligned_input_stick_nbytes = tt::round_up(input_stick_nbytes, input.buffer()->alignment());
    const uint32_t in_cb_pagesize = aligned_input_stick_nbytes;
    const uint32_t in_cb_npages = input_nsticks_per_core * buffering_factor;

    const uint32_t in_cb_id = next_cb_index++;
    desc.cbs.push_back(CBDescriptor{
        .total_size = in_cb_pagesize * in_cb_npages,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(in_cb_id),
            .data_format = input_cb_data_format,
            .page_size = in_cb_pagesize,
        }}},
        .buffer = input.buffer(),
    });

    // output sharded CB with upsampled data
    const uint32_t out_cb_pagesize = tt::round_up(output_stick_nbytes, output.buffer()->alignment());
    const uint32_t out_cb_npages = output_nsticks_per_core * buffering_factor;

    const uint32_t out_cb_id = next_cb_index++;
    desc.cbs.push_back(CBDescriptor{
        .total_size = out_cb_pagesize * out_cb_npages,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(out_cb_id),
            .data_format = output_cb_data_format,
            .page_size = out_cb_pagesize,
        }}},
        .buffer = output.buffer(),
    });

    log_debug(LogOp, "input_cb: {}, npages: {}, pagesize: {}", in_cb_id, in_cb_npages, in_cb_pagesize);
    log_debug(LogOp, "output_cb: {}, npages: {}, pagesize: {}", out_cb_id, out_cb_npages, out_cb_pagesize);
    log_debug(LogOp, "input_stick_nbytes: {}, output_stick_nbytes: {}", input_stick_nbytes, output_stick_nbytes);
    log_debug(LogOp, "ncores: {}, ncores_x: {}", ncores, ncores_x);
    log_debug(
        LogOp,
        "input_nsticks_per_core: {}, output_nsticks_per_core: {}",
        input_nsticks_per_core,
        output_nsticks_per_core);

    // Config tensor lives on resources so its buffer outlives this descriptor
    // and the cached Program (the config CB references the buffer pointer
    // directly via UpdateDynamicCircularBufferAddress).
    Buffer* const config_buffer = require_resources(resources.config_tensor_device).buffer();
    constexpr tt::DataFormat config_df = tt::DataFormat::RawUInt16;
    const uint32_t config_buffer_page_size = config_buffer->page_size();

    // Create config CB only for cores that have work
    const uint32_t config_cb_id = next_cb_index++;
    desc.cbs.push_back(CBDescriptor{
        .total_size = config_buffer_page_size,
        .core_ranges = cores_with_work,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(config_cb_id),
            .data_format = config_df,
            .page_size = config_buffer_page_size,
        }}},
        .buffer = config_buffer,
    });

    KernelDescriptor::CompileTimeArgs writer_compile_time_args = {
        in_cb_id,
        out_cb_id,
        0,  // is_reader = false
        config_cb_id,
        input_stick_nbytes,
        input_nsticks_per_core,
        scale_factor_h,
        scale_factor_w,
        // number of intervals in config tensor per core, 4 is number of bfloat16 elements per entry
        static_cast<uint32_t>(require_resources(resources.config_tensor_device).logical_shape()[-1] / 4),
    };

    KernelDescriptor::CompileTimeArgs reader_compile_time_args = writer_compile_time_args;
    reader_compile_time_args[2] = 1;  // is_reader = true

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/writer_upsample_multi_core_sharded.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = cores_with_work;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor reader_desc;
    // Same kernel source as writer — branches on the is_reader CT arg.
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/writer_upsample_multi_core_sharded.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = cores_with_work;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(reader_desc));

    return desc;
}

}  // namespace ttnn::prim
