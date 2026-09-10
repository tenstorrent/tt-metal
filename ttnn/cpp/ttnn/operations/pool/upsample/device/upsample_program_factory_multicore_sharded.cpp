// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/pool/upsample/device/upsample_device_operation.hpp"

#include <sys/types.h>
#include <cstdint>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/pool/upsample/device/upsample_common.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

using namespace tt;
namespace metal2 = tt::tt_metal::experimental;

namespace {

struct StickInterval {
    std::uint16_t core_x, core_y;
    std::uint16_t offset_start;
    std::uint16_t offset_end;

    StickInterval(std::uint16_t cx, std::uint16_t cy, std::uint16_t start) :
        core_x(cx), core_y(cy), offset_start(start), offset_end(start) {}
};

Tensor create_config_tensor(
    IDevice* device,
    ShardSpec shard_spec,
    const std::uint32_t batch_size,
    const std::uint32_t in_h,
    const std::uint32_t in_w,
    const std::uint32_t scale_factor_h,
    const bool is_height_sharded) {
    std::uint16_t core_idx = 0;  // Tracks the current core being processed
    std::uint16_t core_idx_start =
        0;  // Tracks the starting core index for each row of input, used when scale_factor_h > 1
    std::uint16_t stick_offset = 0;  // Tracks the current stick offset within the core
    std::uint16_t stick_offset_start =
        0;  // Tracks starting stick offset within the core before adding in_w sticks, used when scale_factor_h > 1
    std::uint32_t stick_cnt =
        0;  // Counts the number of sticks processed, used for splitting output sticks across NCRISC and BRISC
    std::uint16_t ch_start_core = 0;   // Starting core index where channels are distributed
    std::uint16_t ch_end_core = 0;     // Ending core index where channels are distributed
    std::uint16_t nhw_start_core = 0;  // Starting core index for NHW distribution
    const std::uint32_t input_nsticks_per_core = shard_spec.shape[0];
    const std::uint32_t output_nsticks_per_core = input_nsticks_per_core * scale_factor_h;
    std::uint32_t output_nsticks_per_core_reader =
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
    std::vector<std::uint16_t> dst_core_end_idx_map;  // used for padding the config vector per reader

    bool reader_sticks_reached = false;
    bool insert_new_interval = false;
    std::uint32_t last_ind = 0;
    std::uint32_t elems_per_core_reader = 0;  // represents number of intervals per reader (NCRISC/BRISC)
    std::uint32_t elem_num = 0;
    // Create map of core and respective offsets in input
    for (std::uint32_t b = 0; b < batch_size; ++b) {
        for (std::uint32_t h = 0; h < in_h; ++h) {
            for (std::uint32_t j = 0; j < scale_factor_h; ++j) {
                stick_offset_start = stick_offset;
                core_idx_start = core_idx;

                for (std::uint32_t w = 0; w < in_w; ++w, ++stick_offset, ++stick_cnt) {
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
    std::vector<std::uint16_t> config_vector;

    const std::uint32_t config_buffer_entry_size = 4;
    elems_per_core_reader *= config_buffer_entry_size;
    const std::uint32_t elems_per_core =
        2 * elems_per_core_reader;  // because two readers per tensix core which get equal number of stick intervals

    // Based on core calculate physical location of cores
    CoreCoord core_coords;

    // In case last input shard is not full, fill the rest of the config vector with placeholder values
    const auto pad_uneven_shards =
        [config_buffer_entry_size](auto& config_vector, std::uint32_t elems_per_core_reader, size_t slice_begin = 0) {
            const std::uint32_t slice_length = config_vector.size() - slice_begin;
            const std::uint32_t remainder =
                (elems_per_core_reader - (slice_length % elems_per_core_reader)) % elems_per_core_reader;
            if (remainder != 0) {
                for (std::uint32_t i = 0; i < remainder / config_buffer_entry_size; i++) {
                    config_vector.push_back(0);  // core x
                    config_vector.push_back(0);  // core y
                    config_vector.push_back(1);  // stick offset start
                    config_vector.push_back(0);  // stick offset end
                }
            }
        };

    // Each entry of dst_core_end_idx_map is padded up to a whole reader block, or to a whole core block when there is a
    // single output stick per core; the extra entry accounts for the trailing per-core padding.
    const std::uint32_t elems_per_idx_map_entry = output_nsticks_per_core > 1 ? elems_per_core_reader : elems_per_core;
    config_vector.reserve(
        (ch_end_core - ch_start_core + 1) * (dst_core_end_idx_map.size() + 1) * elems_per_idx_map_entry);

    std::uint32_t per_core_start_idx = 0;
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
    std::uint32_t total_nhw,
    std::uint32_t nsticks_per_core,
    bool is_height_sharded,
    ShardOrientation orientation) {
    const std::uint32_t num_cores = all_cores.num_cores();
    const std::uint32_t actual_nhw_cores = tt::div_up(total_nhw, nsticks_per_core);

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
    const std::uint32_t nhw_start = row_major ? range.start_coord.y : range.start_coord.x;
    const std::uint32_t nhw_end = row_major ? range.end_coord.y : range.end_coord.x;
    const std::uint32_t nhw_cores_in_grid = nhw_end - nhw_start + 1;
    const std::uint32_t nhw_cores_needed = std::min(actual_nhw_cores, nhw_cores_in_grid);

    if (nhw_cores_needed >= nhw_cores_in_grid) {
        return all_cores;
    }

    const CoreCoord new_end = row_major ? CoreCoord(range.end_coord.x, nhw_start + nhw_cores_needed - 1)
                                        : CoreCoord(nhw_start + nhw_cores_needed - 1, range.end_coord.y);
    return CoreRangeSet(CoreRange(range.start_coord, new_end));
}

}  // namespace

ttnn::device_operation::ProgramArtifacts UpsampleMultiCoreShardedProgramFactory::create_program_artifacts(
    const UpsampleParams& operation_attributes, const Tensor& input_tensor, Tensor& output_tensor) {
    const metal2::KernelSpecName SHARD_WRITER{"upsample_shard_writer"};
    const metal2::KernelSpecName SHARD_READER{"upsample_shard_reader"};
    const metal2::DFBSpecName SHARD_IN{"upsample_shard_in"};
    const metal2::DFBSpecName SHARD_OUT{"upsample_shard_out"};
    const metal2::DFBSpecName SHARD_CONFIG{"upsample_shard_config"};
    const metal2::TensorParamName SHARD_INPUT{"upsample_shard_input"};
    const metal2::TensorParamName SHARD_OUTPUT{"upsample_shard_output"};
    const metal2::TensorParamName SHARD_CONFIG_TENSOR{"upsample_shard_config_tensor"};

    constexpr const char* SHARD_KERNEL =
        "ttnn/cpp/ttnn/operations/pool/upsample/device/kernels/dataflow/writer_upsample_multi_core_sharded.cpp";

    const auto& input = input_tensor;
    auto& output = output_tensor;
    const auto& input_mesh = input.mesh_tensor();
    const auto& output_mesh = output.mesh_tensor();
    TT_FATAL(
        operations::pool::upsample::is_integer_scale(operation_attributes.scale_factor_h) &&
            operations::pool::upsample::is_integer_scale(operation_attributes.scale_factor_w),
        "Sharded upsample factory requires integer scale factors, got scale_h={}, scale_w={}",
        operation_attributes.scale_factor_h,
        operation_attributes.scale_factor_w);
    const std::uint32_t scale_factor_h = static_cast<std::uint32_t>(operation_attributes.scale_factor_h);
    const std::uint32_t scale_factor_w = static_cast<std::uint32_t>(operation_attributes.scale_factor_w);

    distributed::MeshDevice* device = input.device();

    const tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());

    TT_FATAL(input.logical_shape()[-1] == output.logical_shape()[-1], "Expected input and output channels to match");
    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "Only row-major layout is currently supported in nearest upsample");

    std::uint32_t input_stick_nbytes = input.padded_shape()[-1] * input.element_size();
    std::uint32_t output_stick_nbytes = output.padded_shape()[-1] * output.element_size();
    TT_FATAL(input_stick_nbytes == output_stick_nbytes, "Input and output sticks should have same size");

    const std::uint32_t in_w = input.padded_shape()[2];

    const auto shard_spec = input.shard_spec().value();
    const auto all_cores = shard_spec.grid;
    const std::uint32_t ncores = shard_spec.num_cores();
    std::uint32_t ncores_x = device->compute_with_storage_grid_size().x;

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

    const std::uint32_t input_nsticks_per_core = shard_spec.shape[0];
    const std::uint32_t output_nsticks_per_core = input_nsticks_per_core * scale_factor_h * scale_factor_w;
    const bool is_height_sharded = input.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;
    const std::uint32_t total_nhw = input.padded_shape()[0] * input.padded_shape()[1] * in_w;

    // Reduced core set - only cores that actually have work
    const CoreRangeSet cores_with_work =
        get_cores_with_work(all_cores, total_nhw, input_nsticks_per_core, is_height_sharded, shard_spec.orientation);

    // --- Op-owned config tensor: construction UNCHANGED from legacy ---
    Tensor config_tensor = create_config_tensor(
        device, shard_spec, input.padded_shape()[0], input.padded_shape()[1], in_w, scale_factor_h, is_height_sharded);

    const auto shard_shape =
        std::array<std::uint32_t, 2>({1, static_cast<std::uint32_t>(config_tensor.logical_shape()[-1])});
    const auto config_tensor_shard_orientation =
        input.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ? ShardOrientation::COL_MAJOR
                                                                                   : shard_spec.orientation;
    // Use cores_with_work for config tensor sharding - only cores that have actual work need config data
    const ShardSpec config_shard_spec(cores_with_work, shard_shape, config_tensor_shard_orientation);
    const MemoryConfig config_memory_config{
        TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1_SMALL, config_shard_spec};

    Tensor config_tensor_dev = config_tensor.to_device(device, config_memory_config);
    const std::uint32_t config_tensor_width = static_cast<std::uint32_t>(config_tensor_dev.logical_shape()[-1]);
    const std::uint32_t config_buffer_page_size = config_tensor_dev.buffer()->page_size();

    // --- Release the MeshTensor into op_owned_tensors (the Metal 2.0 tail) ---
    std::vector<tt::tt_metal::MeshTensor> op_owned;
    op_owned.reserve(1);
    op_owned.push_back(config_tensor_dev.device_storage().release_mesh_tensor());
    const auto& config_mesh_tensor = op_owned.back();

    constexpr std::uint32_t buffering_factor = 1;  // data is already fully buffered in the CBs since its sharded
    const std::uint32_t aligned_input_stick_nbytes = tt::round_up(input_stick_nbytes, input.buffer()->alignment());
    const std::uint32_t in_cb_pagesize = aligned_input_stick_nbytes;
    const std::uint32_t in_cb_npages = input_nsticks_per_core * buffering_factor;

    metal2::DataflowBufferSpec in_dfb{
        .unique_id = SHARD_IN,
        .entry_size = in_cb_pagesize,
        .num_entries = in_cb_npages,
        .data_format_metadata = input_cb_data_format,
        .borrowed_from = SHARD_INPUT,
    };

    // output sharded CB with upsampled data
    const std::uint32_t out_cb_pagesize = tt::round_up(output_stick_nbytes, output.buffer()->alignment());
    const std::uint32_t out_cb_npages = output_nsticks_per_core * buffering_factor;

    metal2::DataflowBufferSpec out_dfb{
        .unique_id = SHARD_OUT,
        .entry_size = out_cb_pagesize,
        .num_entries = out_cb_npages,
        .data_format_metadata = output_cb_data_format,
        .borrowed_from = SHARD_OUTPUT,
    };

    // The config buffer lives on op_owned_tensors so it outlives this ProgramSpec and the cached
    // Program (the config DFB reresolves its backing address each dispatch via borrowed_from).
    constexpr tt::DataFormat config_df = tt::DataFormat::RawUInt16;

    metal2::DataflowBufferSpec config_dfb{
        .unique_id = SHARD_CONFIG,
        .entry_size = config_buffer_page_size,
        .num_entries = 1,
        .data_format_metadata = config_df,
        .borrowed_from = SHARD_CONFIG_TENSOR,
    };

    log_debug(LogOp, "in_dfb: {}, npages: {}, pagesize: {}", SHARD_IN, in_cb_npages, in_cb_pagesize);
    log_debug(LogOp, "out_dfb: {}, npages: {}, pagesize: {}", SHARD_OUT, out_cb_npages, out_cb_pagesize);
    log_debug(LogOp, "input_stick_nbytes: {}, output_stick_nbytes: {}", input_stick_nbytes, output_stick_nbytes);
    log_debug(LogOp, "ncores: {}, ncores_x: {}", ncores, ncores_x);
    log_debug(
        LogOp,
        "input_nsticks_per_core: {}, output_nsticks_per_core: {}",
        input_nsticks_per_core,
        output_nsticks_per_core);

    metal2::KernelSpec::CompileTimeArgs common_cta{
        {"stick_nbytes", input_stick_nbytes},
        {"in_nsticks_per_core", input_nsticks_per_core},
        {"scale_h", scale_factor_h},
        {"scale_w", scale_factor_w},
        // number of intervals in config tensor per core, 4 is number of bfloat16 elements per entry
        {"elem_per_core", config_tensor_width / 4},
    };

    auto make_cta = [&](std::uint32_t is_reader) {
        metal2::KernelSpec::CompileTimeArgs cta = common_cta;
        cta.insert({"is_reader", is_reader});
        return cta;
    };

    metal2::KernelSpec writer_spec{
        .unique_id = SHARD_WRITER,
        .source = std::filesystem::path{SHARD_KERNEL},
        .dfb_bindings =
            {metal2::DFBBinding{
                 .dfb_spec_name = SHARD_IN, .accessor_name = "in0", .endpoint_type = metal2::DFBEndpointType::PRODUCER},
             metal2::DFBBinding{
                 .dfb_spec_name = SHARD_OUT,
                 .accessor_name = "out0",
                 .endpoint_type = metal2::DFBEndpointType::PRODUCER},
             metal2::DFBBinding{
                 .dfb_spec_name = SHARD_CONFIG,
                 .accessor_name = "config",
                 .endpoint_type = metal2::DFBEndpointType::PRODUCER}},
        .compile_time_args = make_cta(/*is_reader=*/0),
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    metal2::KernelSpec reader_spec{
        .unique_id = SHARD_READER,
        .source = std::filesystem::path{SHARD_KERNEL},
        .dfb_bindings =
            {metal2::DFBBinding{
                 .dfb_spec_name = SHARD_IN, .accessor_name = "in0", .endpoint_type = metal2::DFBEndpointType::CONSUMER},
             metal2::DFBBinding{
                 .dfb_spec_name = SHARD_OUT,
                 .accessor_name = "out0",
                 .endpoint_type = metal2::DFBEndpointType::CONSUMER},
             metal2::DFBBinding{
                 .dfb_spec_name = SHARD_CONFIG,
                 .accessor_name = "config",
                 .endpoint_type = metal2::DFBEndpointType::CONSUMER}},
        .compile_time_args = make_cta(/*is_reader=*/1),
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    metal2::ProgramSpec spec{
        .name = "upsample_multicore_sharded",
        .kernels = {writer_spec, reader_spec},
        .dataflow_buffers = {in_dfb, out_dfb, config_dfb},
        .tensor_parameters =
            {
                {.unique_id = SHARD_INPUT, .spec = input_mesh.tensor_spec()},
                {.unique_id = SHARD_OUTPUT, .spec = output_mesh.tensor_spec()},
                {.unique_id = SHARD_CONFIG_TENSOR, .spec = config_mesh_tensor.tensor_spec()},
            },
        .work_units = {metal2::WorkUnitSpec{
            .name = "main",
            .kernels = {SHARD_WRITER, SHARD_READER},
            .target_nodes = cores_with_work,
        }},
    };

    // No runtime args on either kernel; provide empty entries so every kernel has a KernelRunArgs.
    metal2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        metal2::KernelRunArgs{.kernel = SHARD_WRITER}, metal2::KernelRunArgs{.kernel = SHARD_READER}};
    run_args.tensor_args = {
        {SHARD_INPUT, metal2::TensorArgument{input_mesh}},
        {SHARD_OUTPUT, metal2::TensorArgument{output_mesh}},
        {SHARD_CONFIG_TENSOR, metal2::TensorArgument{config_mesh_tensor}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec), .run_params = std::move(run_args), .op_owned_tensors = std::move(op_owned)};
}

}  // namespace ttnn::prim
