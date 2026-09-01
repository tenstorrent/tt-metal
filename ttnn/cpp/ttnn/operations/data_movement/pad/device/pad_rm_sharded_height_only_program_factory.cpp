// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_rm_sharded_height_only_program_factory.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;
using namespace tt::constants;

namespace ttnn::prim {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

namespace {
// Names are prefixed per factory: all seven pad factories land in one unity-build
// translation unit, where every anonymous namespace is merged into a single scope.
const KernelSpecName SH_H_READER{"reader"};
const KernelSpecName SH_H_WRITER{"writer"};
const DFBSpecName SH_H_IN_SHARD{"in_shard"};
const DFBSpecName SH_H_OUT_SHARD{"out_shard"};
const DFBSpecName SH_H_PAD{"pad"};
const TensorParamName SH_H_INPUT{"input"};
const TensorParamName SH_H_OUTPUT{"output"};

// One core's worth of kernel arguments.
//
// The reader's gather plan is data-directed — num_cores source cores, then a NoC (x, y) pair per
// source core, then a chunk count per source core, then a (start_id, length) pair per chunk — and
// the counts come from the data itself, so nothing past num_cores has a per-element identity. That
// block travels as runtime varargs; everything the writer takes is a distinct named field.
struct ShardedHeightPerCoreArgs {
    uint32_t num_cores_read = 0;
    std::vector<uint32_t> reader_varargs;
    uint32_t num_sticks_per_core = 0;
    uint32_t start_id = 0;
    uint32_t start_dim_offset_h = 0;
    uint32_t start_dim_offset_c = 0;
    uint32_t start_dim_offset_n = 0;
};

inline std::vector<std::vector<uint32_t>> group_contiguous_and_repeated_values(std::vector<uint32_t>& values) {
    std::vector<std::vector<uint32_t>> chunks;
    if (values.empty()) {
        return chunks;
    }
    chunks.reserve(values.size());

    // Initialize the first chunk
    std::vector<uint32_t> current_chunk;
    current_chunk.reserve(values.size());
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
    chunks.push_back(std::move(current_chunk));
    return chunks;
}

inline std::vector<ShardedHeightPerCoreArgs> get_pad_runtime_args_rm_sharded(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& input_tensor_start,
    uint32_t num_cores_padded,
    bool row_major,
    uint32_t shard_height_padded,
    uint32_t shard_height_unpadded,
    const CoreCoord& unpadded_grid_start,
    uint32_t num_cores_x_unpadded,
    uint32_t num_cores_y_unpadded) {
    tt::tt_metal::IDevice* device = input_tensor.device();

    auto input_shape = input_tensor.padded_shape();
    auto output_shape = output_tensor.padded_shape();

    uint32_t H = input_shape[2], C = input_shape[1], N = input_shape[0];

    uint32_t H_padded = output_shape[2], C_padded = output_shape[1];

    std::vector<ShardedHeightPerCoreArgs> ret_val(num_cores_padded);

    const auto& front_pad = input_tensor_start;
    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;
    for (uint32_t i = 0, curr_sticks_read = 0; i < num_cores_padded; i++) {
        uint32_t num_sticks_per_core_unpadded = shard_height_unpadded;
        uint32_t num_sticks_per_core_padded = shard_height_padded;

        // Writer args, captured on top here as in the interleaved version. curr_h / curr_c /
        // curr_n hold this core's starting position in the padded output; the stick loop below
        // advances them for the next core.
        ShardedHeightPerCoreArgs core_args;
        core_args.num_sticks_per_core = num_sticks_per_core_padded;
        core_args.start_id = curr_sticks_read;
        core_args.start_dim_offset_h = curr_h;
        core_args.start_dim_offset_c = curr_c;
        core_args.start_dim_offset_n = curr_n;

        // figure out the start read stick id for each core, and the start id for each dim
        std::vector<int> stick_ids_per_core;
        stick_ids_per_core.reserve(num_sticks_per_core_padded);
        int front_pad_stick_id = -2;
        int pad_stick_id = -1;
        for (uint32_t j = 0; j < num_sticks_per_core_padded; ++j) {
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

        // figure out the stick id in a shard, and the core id for the stick.
        std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> core_stick_map;
        auto first_core = device->worker_core_from_logical_core(unpadded_grid_start);
        std::pair<uint32_t, uint32_t> prev_xy_pair = std::make_pair(first_core.x, first_core.y);
        for (uint32_t j = 0; j < num_sticks_per_core_padded; ++j) {
            int stick_id = stick_ids_per_core[j];

            // if it is pad stick, we need to leave a gap between the previous non-pad stick and next non-pad stick.
            if (stick_id == -2 || stick_id == -1) {  // front or end padding
                core_stick_map[prev_xy_pair].push_back(stick_id);
            } else {
                uint32_t shard_id = stick_id / num_sticks_per_core_unpadded;
                uint32_t stick_id_in_shard = stick_id - (shard_id * num_sticks_per_core_unpadded);

                uint32_t shard_grid_inner_dim = row_major ? num_cores_x_unpadded : num_cores_y_unpadded;
                uint32_t shard_grid_outer_dim_id = shard_id / shard_grid_inner_dim;
                uint32_t shard_grid_inner_dim_id = shard_id - (shard_grid_outer_dim_id * shard_grid_inner_dim);

                uint32_t worker_y_logical =
                    unpadded_grid_start.y + (row_major ? shard_grid_outer_dim_id : shard_grid_inner_dim_id);
                uint32_t worker_x_logical =
                    unpadded_grid_start.x + (row_major ? shard_grid_inner_dim_id : shard_grid_outer_dim_id);

                // worker_*_logical are absolute logical coordinates. Compare against absolute unpadded-grid bounds.
                uint32_t unpadded_grid_end_x = unpadded_grid_start.x + num_cores_x_unpadded;
                uint32_t unpadded_grid_end_y = unpadded_grid_start.y + num_cores_y_unpadded;
                if (worker_x_logical < unpadded_grid_end_x and worker_y_logical < unpadded_grid_end_y) {
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

        // reader varargs: the whole gather plan except num_cores, which is a named arg.
        core_args.num_cores_read = core_stick_map.size();
        std::vector<uint32_t>& reader_varargs = core_args.reader_varargs;
        reader_varargs.reserve(3 * core_stick_map.size() + 2 * num_sticks_per_core_padded);

        for (const auto& core_stick_pair : core_stick_map) {
            auto xy_pair = core_stick_pair.first;
            if (row_major) {
                reader_varargs.push_back((std::uint32_t)xy_pair.second);  // noc x
                reader_varargs.push_back((std::uint32_t)xy_pair.first);   // noc y
            } else {
                reader_varargs.push_back((std::uint32_t)xy_pair.first);   // noc x
                reader_varargs.push_back((std::uint32_t)xy_pair.second);  // noc y
            }
        }

        // coalesce the sticks into chunks
        std::vector<std::vector<std::vector<uint32_t>>> stick_chunks_per_core;
        stick_chunks_per_core.reserve(core_stick_map.size());
        for (auto core_stick_pair : core_stick_map) {
            auto stick_chunks = group_contiguous_and_repeated_values(core_stick_pair.second);
            reader_varargs.push_back(stick_chunks.size());  // num_chunks for current core
            stick_chunks_per_core.push_back(std::move(stick_chunks));
        }
        for (const auto& stick_chunks : stick_chunks_per_core) {
            for (auto chunk : stick_chunks) {
                reader_varargs.push_back(chunk[0]);      // start id of a chunk
                reader_varargs.push_back(chunk.size());  // length of a chunk
            }
        }

        ret_val[i] = std::move(core_args);
    }

    return ret_val;
}
}  // namespace

ttnn::device_operation::ProgramArtifacts PadRmShardedHeightOnlyProgramFactory::create_program_artifacts(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input;
    Tensor& output = tensor_return_value;
    const auto& input_mesh_tensor = a.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();
    const auto& output_padded_shape = operation_attributes.output_padded_shape;
    const auto& pad_value = operation_attributes.pad_value;
    const auto& input_tensor_start = operation_attributes.input_tensor_start;

    const auto& a_shape = a.logical_shape();
    uint32_t W = a_shape[3], H = a_shape[2], C = a_shape[1], N = a_shape[0];
    uint32_t num_unpadded_sticks = H * C * N;
    uint32_t W_padded = output_padded_shape[3], H_padded = output_padded_shape[2], C_padded = output_padded_shape[1],
             N_padded = output_padded_shape[0];

    const auto& front_pad = operation_attributes.input_tensor_start;

    log_debug(tt::LogOp, "H_padded: {}", H_padded);
    log_debug(tt::LogOp, "front_pad: {}", front_pad);

    // stick sizes
    auto stick_size_unpadded = W * a.element_size();
    auto stick_size_padded = W_padded * a.element_size();
    uint32_t row_major_min_bytes = 16;

    uint32_t zero_pad_stick_size = tt::tt_metal::find_max_divisor(stick_size_padded, 512);
    uint32_t num_zero_pad_sticks_read = stick_size_padded / zero_pad_stick_size;

    log_debug(tt::LogOp, "zero_pad_stick_size: {}", zero_pad_stick_size);
    log_debug(tt::LogOp, "num_zero_pad_sticks_read: {}", num_zero_pad_sticks_read);

    // TODO: add a general case, where we can pad on any dim.
    TT_FATAL(
        stick_size_unpadded == stick_size_padded,
        "sharded pad does not support pad on last dim currently as that will cause perf degradation");

    tt::DataFormat dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat dst_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    IDevice* device = a.device();

    // input shard spec
    auto shard_spec_unpadded = a.shard_spec().value();
    uint32_t shard_height_unpadded = shard_spec_unpadded.shape[0];
    bool row_major = shard_spec_unpadded.orientation == ShardOrientation::ROW_MAJOR;

    [[maybe_unused]] auto& all_cores_unpadded = shard_spec_unpadded.grid;
    [[maybe_unused]] uint32_t num_cores_unpadded = shard_spec_unpadded.num_cores();
    auto bbox_unpadded = shard_spec_unpadded.grid.bounding_box();
    CoreCoord grid_size_unpadded = {
        bbox_unpadded.end_coord.x - bbox_unpadded.start_coord.x + 1,
        bbox_unpadded.end_coord.y - bbox_unpadded.start_coord.y + 1};
    uint32_t num_cores_x_unpadded = grid_size_unpadded.x;
    uint32_t num_cores_y_unpadded = grid_size_unpadded.y;

    log_debug(tt::LogOp, "num_unpadded_sticks: {}", num_unpadded_sticks);
    log_debug(tt::LogOp, "shard_height_unpadded: {}", shard_height_unpadded);
    log_debug(tt::LogOp, "all_cores_unpadded: {}", all_cores_unpadded);
    log_debug(tt::LogOp, "num_cores_unpadded: {}", num_cores_unpadded);

    // output shard spec
    auto shard_spec_padded = output.shard_spec().value();
    uint32_t shard_height_padded = shard_spec_padded.shape[0];

    auto& all_cores_padded = shard_spec_padded.grid;
    uint32_t num_cores_padded = shard_spec_padded.num_cores();
    auto bbox_padded = shard_spec_padded.grid.bounding_box();
    CoreCoord grid_size_padded = {
        bbox_padded.end_coord.x - bbox_padded.start_coord.x + 1,
        bbox_padded.end_coord.y - bbox_padded.start_coord.y + 1};
    uint32_t num_cores_x_padded = grid_size_padded.x;
    uint32_t num_cores_y_padded = grid_size_padded.y;

    log_debug(tt::LogOp, "num_unpadded_sticks: {}", num_unpadded_sticks);
    log_debug(tt::LogOp, "shard_height_unpadded: {}", shard_height_unpadded);
    log_debug(tt::LogOp, "all_cores_unpadded: {}", all_cores_unpadded);
    log_debug(tt::LogOp, "num_cores_unpadded: {}", num_cores_unpadded);

    // Sharded input DFB — borrows the input buffer's L1 memory; the framework re-points it from
    // the input TensorArgument on every dispatch. The reader only takes its base pointer (a raw
    // peek, no FIFO ops), so the reader is its sole toucher and binds both endpoints (self-loop).
    // The entry count is clamped to the sticks the tensor actually holds: to_layout can hand pad
    // an input whose shard spec is taller than the whole tensor (from_torch builds the row-major
    // input with the requested tile-aligned output sharding, so e.g. 16 sticks arrive under a
    // 32-stick shard spec), and a full-shard DFB would then fail spec validation against the
    // borrowed tensor's packed size. The count is inert on device — the reader never runs FIFO
    // ops on this DFB — so the clamp only affects validation.
    DataflowBufferSpec in_shard_dfb{
        .unique_id = SH_H_IN_SHARD,
        .entry_size = stick_size_unpadded,
        .num_entries = std::min(shard_height_unpadded, num_unpadded_sticks),
        .data_format_metadata = dfb_data_format,
        .borrowed_from = SH_H_INPUT,
    };

    // Sharded output DFB — borrows the output buffer's L1 memory. Two touchers on every node: the
    // reader is a locked FIFO producer, and the writer raw-peeks the same buffer via get_write_ptr
    // with no FIFO ops of its own. A raw peek is role-free, so this is a plain 1P + 1C assignment
    // (reader producer, writer consumer) — not multi-binding. The writer must still be bound: in
    // Metal 2.0 a kernel may not touch a DFB it has not bound.
    DataflowBufferSpec out_shard_dfb{
        .unique_id = SH_H_OUT_SHARD,
        .entry_size = stick_size_padded,
        .num_entries = shard_height_padded,
        .data_format_metadata = dst_dfb_data_format,
        .borrowed_from = SH_H_OUTPUT,
    };

    // Const buffer holding one stick of the pad value. Writer-only, no FIFO ops — self-loop.
    DataflowBufferSpec pad_dfb{
        .unique_id = SH_H_PAD,
        .entry_size = stick_size_padded,
        .num_entries = 1,
        .data_format_metadata = dfb_data_format,
    };

    // construct const buffer with the pad_value
    bool not_pad_by_zero = pad_value != 0;

    uint32_t packed_pad_value;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32) {
        packed_pad_value = pad_value;
    } else if (a.dtype() == DataType::UINT16) {
        packed_pad_value = pack_two_uint16_into_uint32({float_to_uint16(pad_value), float_to_uint16(pad_value)});
    } else {
        packed_pad_value = pack_two_bfloat16_into_uint32({bfloat16(pad_value), bfloat16(pad_value)});
    }

    auto all_runtime_args = get_pad_runtime_args_rm_sharded(
        a,
        output,
        input_tensor_start,
        num_cores_padded,
        row_major,
        shard_height_padded,
        shard_height_unpadded,
        bbox_unpadded.start_coord,
        num_cores_x_unpadded,
        num_cores_y_unpadded);

    // The reader's vararg block length is data-dependent (it grows with the number of source
    // cores and coalesced chunks a core gathers from), but a KernelSpec declares one vararg count
    // for every node it runs on. Declare the longest block and zero-fill the shorter ones: the
    // kernel walks the block using the counts it reads out of it, so the tail is never read.
    uint32_t num_reader_varargs = 0;
    for (const auto& core_args : all_runtime_args) {
        num_reader_varargs = std::max<uint32_t>(num_reader_varargs, core_args.reader_varargs.size());
    }

    KernelSpec reader{
        .unique_id = SH_H_READER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/reader_pad_dims_rm_sharded.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = SH_H_IN_SHARD,
                    .accessor_name = "in_shard",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = SH_H_IN_SHARD,
                    .accessor_name = "in_shard",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = SH_H_OUT_SHARD,
                    .accessor_name = "out_shard",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .compile_time_args =
            {
                {"stick_size_bytes", static_cast<uint32_t>(stick_size_padded)},
                {"num_sticks_padded", shard_height_padded},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_cores_read"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
        .advanced_options = {.num_runtime_varargs = num_reader_varargs},
    };

    KernelSpec writer{
        .unique_id = SH_H_WRITER,
        .source = "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_pad_dims_rm_sharded.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = SH_H_OUT_SHARD,
                    .accessor_name = "out_shard",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = SH_H_PAD,
                    .accessor_name = "pad",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = SH_H_PAD,
                    .accessor_name = "pad",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .compile_time_args =
            {
                {"N", static_cast<uint32_t>(N + front_pad[-4])},
                {"H", static_cast<uint32_t>(H + front_pad[-2])},
                {"C", static_cast<uint32_t>(C + front_pad[-3])},
                {"stick_size_bytes", static_cast<uint32_t>(stick_size_padded)},
                {"N_padded", N_padded},
                {"H_padded", H_padded},
                {"C_padded", C_padded},
                {"num_zero_pad_sticks_read", num_zero_pad_sticks_read},
                {"zero_pad_stick_size", zero_pad_stick_size},
                {"not_pad_by_zero", static_cast<uint32_t>(not_pad_by_zero)},
                {"packed_pad_value", packed_pad_value},
                {"row_major_min_bytes", row_major_min_bytes},
                {"num_sticks_padded_read", static_cast<uint32_t>(stick_size_padded / row_major_min_bytes)},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"num_sticks_per_core",
                     "start_id",
                     "front_pad_n",
                     "front_pad_c",
                     "front_pad_h",
                     "start_dim_offset_h",
                     "start_dim_offset_c",
                     "start_dim_offset_n"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    KernelRunArgs reader_run_args{.kernel = SH_H_READER};
    KernelRunArgs writer_run_args{.kernel = SH_H_WRITER};

    for (uint32_t i = 0; i < num_cores_padded; i++) {
        CoreCoord core;
        if (row_major) {
            core = {
                bbox_padded.start_coord.x + i % num_cores_x_padded, bbox_padded.start_coord.y + i / num_cores_x_padded};
        } else {
            core = {
                bbox_padded.start_coord.x + i / num_cores_y_padded, bbox_padded.start_coord.y + i % num_cores_y_padded};
        }
        const auto& core_args = all_runtime_args[i];

        AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"num_cores_read", core_args.num_cores_read}});
        AdvancedKernelRunArgs::Varargs reader_varargs = core_args.reader_varargs;
        reader_varargs.resize(num_reader_varargs, 0u);
        reader_run_args.advanced_options.runtime_varargs[core] = std::move(reader_varargs);

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_sticks_per_core", core_args.num_sticks_per_core},
             {"start_id", core_args.start_id},
             {"front_pad_n", static_cast<uint32_t>(front_pad[-4])},
             {"front_pad_c", static_cast<uint32_t>(front_pad[-3])},
             {"front_pad_h", static_cast<uint32_t>(front_pad[-2])},
             {"start_dim_offset_h", core_args.start_dim_offset_h},
             {"start_dim_offset_c", core_args.start_dim_offset_c},
             {"start_dim_offset_n", core_args.start_dim_offset_n}});
    }

    ProgramSpec spec{
        .name = "pad_rm_sharded_height_only",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = {std::move(in_shard_dfb), std::move(out_shard_dfb), std::move(pad_dfb)},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = SH_H_INPUT, .spec = input_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = SH_H_OUTPUT, .spec = output_mesh_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {SH_H_READER, SH_H_WRITER},
                    .target_nodes = all_cores_padded,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {SH_H_INPUT, TensorArgument{input_mesh_tensor}},
        {SH_H_OUTPUT, TensorArgument{output_mesh_tensor}},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim
