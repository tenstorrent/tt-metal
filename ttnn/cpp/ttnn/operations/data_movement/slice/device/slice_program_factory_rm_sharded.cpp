// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_metal2_names.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm_sharded.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

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

        // reader rt args
        std::vector<uint32_t> reader_kernel_args;
        reader_kernel_args.reserve(1 + 3 * core_stick_map.size() + 2 * num_sticks_per_core_unpadded);
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
        stick_chunks_per_core.reserve(core_stick_map.size());
        for (auto core_stick_pair : core_stick_map) {
            auto stick_chunks = group_contiguous_values(core_stick_pair.second);
            reader_kernel_args.push_back(stick_chunks.size());  // num_chunks for current core

            stick_chunks_per_core.push_back(std::move(stick_chunks));
        }
        for (const auto& stick_chunks : stick_chunks_per_core) {
            for (auto chunk : stick_chunks) {
                reader_kernel_args.push_back(chunk[0]);      // start id of a chunk
                reader_kernel_args.push_back(chunk.size());  // length of a chunk
            }
        }

        std::vector<uint32_t> writer_kernel_args;
        ret_val[i] = {std::move(reader_kernel_args), std::move(writer_kernel_args)};
    }

    return ret_val;
}

}  // namespace

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {

namespace {

// This factory's two dataflow buffers, each borrowing its memory from one of the op's shards. The
// identifiers are factory-prefixed because sibling slice factories declare their own buffers and
// this target is a unity build.
const DFBSpecName SHARDED_IN{"in"};
const DFBSpecName SHARDED_OUT{"out"};

}  // namespace

ttnn::device_operation::ProgramArtifacts SliceRmShardedProgramFactory::create_program_artifacts(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    using slice_metal2::INPUT;
    using slice_metal2::OUTPUT;
    using slice_metal2::READER;

    const auto& input = tensor_args.input;
    const auto& input_mesh_tensor = input.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

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

    // Both buffers borrow their memory from a tensor, so their backing addresses resolve from the
    // tensor arguments on every dispatch. Their sizes vary with shard shape and element size, and are
    // fixed at spec construction; padded_shape is folded into compute_program_hash() to keep each
    // unique sizing in its own cache entry.
    DataflowBufferSpec in_dfb{
        .unique_id = SHARDED_IN,
        .entry_size = src_stride_bytes,
        .num_entries = shard_height_padded,
        .data_format_metadata = dfb_data_format,
        .borrowed_from = INPUT,
    };

    DataflowBufferSpec out_dfb{
        .unique_id = SHARDED_OUT,
        .entry_size = dst_stride_bytes,
        .num_entries = shard_height_unpadded,
        .data_format_metadata = dst_dfb_data_format,
        .borrowed_from = OUTPUT,
    };

    auto all_runtime_args = ttnn::operations::data_movement::get_slice_runtime_args_rm_sharded(
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

    // Argument 0 is the only nameable scalar. Everything after it is one variable-length stream whose
    // internal layout the kernel derives from that count, so it is a vararg block. Its length differs
    // per node, because both the number of source cores and the number of coalesced chunks are
    // data-directed. The schema carries a single count, so declare the longest block and leave the
    // shorter nodes zero-filled: the kernel walks the stream by the counts it reads out of it and
    // never touches the tail.
    uint32_t max_varargs = 0;
    for (uint32_t i = 0; i < num_cores_unpadded; ++i) {
        max_varargs = std::max<uint32_t>(max_varargs, all_runtime_args[i].first.size() - 1);
    }

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
            "slice_reader_unary_unpad_dims_rm_sharded.cpp",
        .dfb_bindings =
            {
                // Both buffers are touched only by this reader, so each is bound at both endpoints.
                // The input one is a pure address source: the kernel peeks its write pointer to learn
                // where the local input shard sits, then reads the same offset on peer cores.
                DFBBinding{
                    .dfb_spec_name = SHARDED_IN,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = SHARDED_IN,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                // The output one is filled through the FIFO and drained by nobody: the output shard
                // is resident, so the data is already where it belongs once written.
                DFBBinding{
                    .dfb_spec_name = SHARDED_OUT,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = SHARDED_OUT,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .compile_time_args =
            {
                {"stick_size_unpadded", static_cast<uint32_t>(stick_size_unpadded)},
                {"num_sticks_unpadded", static_cast<uint32_t>(shard_height_unpadded)},
                {"src_stride_bytes", src_stride_bytes},
                {"dst_stride_bytes", dst_stride_bytes},
                {"begins_bytes", begins_bytes},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_cores_read"}},
        .hw_config = ttnn::create_reader_datamovement_config(input.device()->arch()),
        .advanced_options = {.num_runtime_varargs = max_varargs},
    };

    KernelRunArgs reader_run_args{.kernel = READER};
    for (uint32_t i = 0; i < num_cores_unpadded; ++i) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x_unpadded, i / num_cores_x_unpadded};
        } else {
            core = {i / num_cores_y_unpadded, i % num_cores_y_unpadded};
        }
        const auto& core_args = all_runtime_args[i].first;
        AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"num_cores_read", core_args[0]}});

        AdvancedKernelRunArgs::Varargs varargs(max_varargs, 0);
        std::copy(core_args.begin() + 1, core_args.end(), varargs.begin());
        reader_run_args.advanced_options.runtime_varargs[core] = std::move(varargs);
    }

    ProgramSpec spec{
        .name = "slice_rm_sharded",
        .kernels = {std::move(reader)},
        .dataflow_buffers = {std::move(in_dfb), std::move(out_dfb)},
        // INPUT is declared but bound by no kernel. That is legal because a borrowed-memory buffer
        // names it, and the buffer's backing address resolves from its tensor argument.
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output_mesh_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {READER},
                    .target_nodes = all_cores_unpadded,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args)};
    run_args.tensor_args = {
        {INPUT, input_mesh_tensor},
        {OUTPUT, output_mesh_tensor},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

tt::tt_metal::experimental::ProgramRunArgs SliceRmShardedProgramFactory::override_runtime_arguments(
    const SliceParams& /*args*/,
    const SliceInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Every reader argument is shape-derived and cache-keyed, so the only per-dispatch state is where
    // the two shards live. Re-binding the tensors is what re-points both borrowed buffers, which is
    // exactly the pair of addresses the ported-from override patched.
    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {slice_metal2::INPUT, tensor_args.input.mesh_tensor()},
        {slice_metal2::OUTPUT, output.mesh_tensor()},
    };
    return run_args;
}

}  // namespace ttnn::prim
