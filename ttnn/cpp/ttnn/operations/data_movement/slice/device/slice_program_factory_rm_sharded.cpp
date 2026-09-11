// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm_sharded.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::operations::data_movement {

namespace {

inline std::vector<std::vector<uint32_t>> group_contiguous_values(std::vector<uint32_t>& values) {
    std::vector<std::vector<uint32_t>> chunks;
    if (values.empty()) {
        return chunks;
    }

    // Contiguous values coalesce into far fewer chunks than there are values, so count the
    // runs up front instead of reserving the worst case
    size_t num_chunks = 1;
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] != values[i - 1] + 1) {
            ++num_chunks;
        }
    }
    chunks.reserve(num_chunks);

    // Initialize the first chunk
    std::vector<uint32_t> current_chunk;
    current_chunk.reserve(values.size());
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
    chunks.push_back(std::move(current_chunk));
    return chunks;
}

// One node's reader arguments.
//
// The gather plan is data-directed — num_cores_read source cores, then a NoC (x, y) pair per source
// core, then a chunk count per source core, then a (start_id, length) pair per chunk — and the
// counts come from the data itself, so nothing past num_cores_read has a per-element identity. That
// block travels as runtime varargs; num_cores_read is a named argument.
struct ShardedPerNodeArgs {
    uint32_t num_cores_read = 0;
    std::vector<uint32_t> reader_varargs;
};

inline std::vector<ShardedPerNodeArgs> get_slice_runtime_args_rm_sharded(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    uint32_t num_cores_unpadded,
    bool row_major,
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

    std::vector<ShardedPerNodeArgs> ret_val(num_cores_unpadded);

    uint32_t start_offset = ttnn::operations::data_movement::get_rm_start_offset(input_tensor, output_tensor_start);
    for (uint32_t i = 0, num_sticks_written = 0; i < num_cores_unpadded; i++) {
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
        stick_ids_per_core.reserve(num_sticks_per_core_unpadded);
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

        // reader args: num_cores_read is named, the gather plan below is the vararg block.
        ShardedPerNodeArgs node_args;
        node_args.num_cores_read = core_stick_map.size();
        std::vector<uint32_t>& reader_varargs = node_args.reader_varargs;
        reader_varargs.reserve(3 * core_stick_map.size() + 2 * num_sticks_per_core_unpadded);

        for (const auto& core_stick_pair : core_stick_map) {
            auto xy_pair = core_stick_pair.first;
            if (row_major) {
                reader_varargs.push_back(xy_pair.second);  // noc x
                reader_varargs.push_back(xy_pair.first);   // noc y
            } else {
                reader_varargs.push_back(xy_pair.first);   // noc x
                reader_varargs.push_back(xy_pair.second);  // noc y
            }
        }

        // coalesce the sticks into chunks
        std::vector<std::vector<std::vector<uint32_t>>> stick_chunks_per_core;
        stick_chunks_per_core.reserve(core_stick_map.size());
        for (auto core_stick_pair : core_stick_map) {
            auto stick_chunks = group_contiguous_values(core_stick_pair.second);
            reader_varargs.push_back(stick_chunks.size());  // num_chunks for current core

            stick_chunks_per_core.push_back(std::move(stick_chunks));
        }
        for (const auto& stick_chunks : stick_chunks_per_core) {
            for (auto chunk : stick_chunks) {
                reader_varargs.push_back(chunk[0]);      // start id of a chunk
                reader_varargs.push_back(chunk.size());  // length of a chunk
            }
        }

        ret_val[i] = std::move(node_args);
    }

    return ret_val;
}

}  // namespace

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {

namespace {

// Function-local for the same unity-build reason as the other slice factories.
struct ShardedSpecNames {
    KernelSpecName reader{"reader"};
    DFBSpecName in_shard{"in_shard"};
    DFBSpecName out_shard{"out_shard"};
    TensorParamName input{"input"};
    TensorParamName output{"output"};
};

}  // namespace

ttnn::device_operation::ProgramArtifacts SliceRmShardedProgramFactory::create_program_artifacts(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const ShardedSpecNames names;

    const auto& input = tensor_args.input;
    tt::tt_metal::IDevice* device = input.device();

    [[maybe_unused]] uint32_t num_padded_sticks = input.physical_volume() / input.padded_shape()[-1];
    [[maybe_unused]] uint32_t num_unpadded_sticks = output.physical_volume() / output.padded_shape()[-1];

    uint32_t W_unpadded = output.logical_shape()[-1];
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

    tt::DataFormat dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat dst_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    // Real per-row L1 stride is aligned_page_size(), not the compact payload (differs when W·E % 16 != 0).
    const uint32_t src_stride_bytes = input.buffer()->aligned_page_size();
    const uint32_t dst_stride_bytes = output.buffer()->aligned_page_size();
    const uint32_t begins_bytes = args.slice_start[-1] * input.element_size();
    TT_FATAL(
        begins_bytes % ::hal::get_l1_alignment() == 0,
        "SliceRmShardedProgramFactory: width-begin ({} bytes) must be L1-aligned.",
        begins_bytes);

    // Both DFBs borrow their backing memory from a tensor, so the framework re-points them from the
    // corresponding TensorArgument on every dispatch — no address ever travels as an argument.
    //
    // The reader is each one's only toucher, so it binds both endpoints of both (self-loop):
    //  - in_shard is sync-free, touched only by a raw get_write_ptr() peek and no FIFO ops. Keeping
    //    it borrowed from the input is load-bearing beyond addressing: the kernel uses that *local*
    //    pointer as the address of reads aimed at *other* cores, which is only correct because a
    //    sharded buffer lands at the same L1 offset on every core in the range.
    //  - out_shard is a locked producer (reserve_back / push_back) that nothing drains.
    const DataflowBufferSpec in_shard_dfb{
        .unique_id = names.in_shard,
        .entry_size = src_stride_bytes,
        .num_entries = shard_height_padded,
        .data_format_metadata = dfb_data_format,
        .borrowed_from = names.input,
    };
    const DataflowBufferSpec out_shard_dfb{
        .unique_id = names.out_shard,
        .entry_size = dst_stride_bytes,
        .num_entries = shard_height_unpadded,
        .data_format_metadata = dst_dfb_data_format,
        .borrowed_from = names.output,
    };

    const TensorParameter input_param{.unique_id = names.input, .spec = input.tensor_spec()};
    const TensorParameter output_param{.unique_id = names.output, .spec = output.tensor_spec()};

    auto all_runtime_args = ttnn::operations::data_movement::get_slice_runtime_args_rm_sharded(
        input,
        output,
        args.slice_start,
        num_cores_unpadded,
        row_major,
        shard_height_unpadded,
        shard_height_padded,
        num_cores_x_padded,
        num_cores_y_padded);

    // The gather plan's length is data-dependent (it grows with the number of source cores and
    // coalesced chunks a node reads from), but a KernelSpec declares one vararg count for every node
    // it runs on. Declare the longest block and zero-fill the shorter ones: the kernel walks the
    // block using the counts it reads out of it, so the tail is never read.
    uint32_t num_reader_varargs = 0;
    for (const auto& node_args : all_runtime_args) {
        num_reader_varargs = std::max<uint32_t>(num_reader_varargs, node_args.reader_varargs.size());
    }

    const KernelSpec reader{
        .unique_id = names.reader,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
            "slice_reader_unary_unpad_dims_rm_sharded.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = names.in_shard,
                    .accessor_name = "in_shard",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = names.in_shard,
                    .accessor_name = "in_shard",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = names.out_shard,
                    .accessor_name = "out_shard",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = names.out_shard,
                    .accessor_name = "out_shard",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .compile_time_args =
            {
                {"stick_size_unpadded", static_cast<uint32_t>(stick_size_unpadded)},
                {"num_sticks_unpadded", shard_height_unpadded},
                {"src_stride_bytes", src_stride_bytes},
                {"dst_stride_bytes", dst_stride_bytes},
                {"begins_bytes", begins_bytes},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_cores_read"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
        .advanced_options = {.num_runtime_varargs = num_reader_varargs},
    };

    KernelRunArgs reader_run_args{.kernel = names.reader};
    for (uint32_t i = 0; i < num_cores_unpadded; ++i) {
        CoreCoord node;
        if (row_major) {
            node = {i % num_cores_x_unpadded, i / num_cores_x_unpadded};
        } else {
            node = {i / num_cores_y_unpadded, i % num_cores_y_unpadded};
        }
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, node, {{"num_cores_read", all_runtime_args[i].num_cores_read}});
        AdvancedKernelRunArgs::Varargs reader_varargs = all_runtime_args[i].reader_varargs;
        reader_varargs.resize(num_reader_varargs, 0u);
        reader_run_args.advanced_options.runtime_varargs[node] = std::move(reader_varargs);
    }

    ProgramSpec spec{
        .name = "slice_rm_sharded",
        .kernels = {reader},
        .dataflow_buffers = {in_shard_dfb, out_shard_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "slice",
            .kernels = {names.reader},
            .target_nodes = all_cores_unpadded,
        }},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args)};
    run_args.tensor_args = {{names.input, input.mesh_tensor()}, {names.output, output.mesh_tensor()}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

tt::tt_metal::experimental::ProgramRunArgs SliceRmShardedProgramFactory::override_runtime_arguments(
    const SliceParams& /*args*/,
    const SliceInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const ShardedSpecNames names;

    // The legacy refresh for this factory re-pointed the two borrowed backing addresses and nothing
    // else, matching them positionally. Both now resolve by name from their backing tensor, so the
    // positional order that had to be kept in sync stops mattering.
    ProgramRunArgs run_args;
    run_args.tensor_args = {{names.input, tensor_args.input.mesh_tensor()}, {names.output, output.mesh_tensor()}};
    return run_args;
}

}  // namespace ttnn::prim
