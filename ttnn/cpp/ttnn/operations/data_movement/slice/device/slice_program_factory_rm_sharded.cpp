// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm_sharded.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile.hpp"

#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm_stride.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile_tensor_args.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_metal2_names.hpp"

#include <map>
#include <optional>
#include <vector>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

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

// input_cores lists the cores of the input tensor's shard grid in shard order, so input_cores[k] is the
// logical coordinate of the core holding input shard k. Every NOC coordinate written into an argument
// list is converted from an input_cores entry. num_padded_sticks is how many rows the input holds.
inline std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_slice_runtime_args_rm_sharded(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& output_tensor_start,
    const std::vector<CoreCoord>& input_cores,
    uint32_t num_cores_unpadded,
    uint32_t shard_height_unpadded,
    uint32_t shard_height_padded,
    uint32_t num_padded_sticks) {
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

    // The input's shards have to cover the input's rows, so every row id inside the input maps to a shard
    // index inside input_cores. Sharded TensorSpec construction rejects a shard height and grid that cannot
    // cover the tensor, so a failure here means the three values passed in do not describe one tensor.
    TT_FATAL(
        static_cast<uint64_t>(shard_height_padded) * input_cores.size() >= num_padded_sticks,
        "SliceRmShardedProgramFactory: the input's {} shards of {} rows cannot hold its {} rows.",
        input_cores.size(),
        shard_height_padded,
        num_padded_sticks);

    uint32_t start_offset = ttnn::operations::data_movement::get_rm_start_offset(input_tensor, output_tensor_start);
    for (uint32_t i = 0, num_sticks_written = 0; i < num_cores_unpadded; i++) {
        uint32_t num_sticks_per_core_unpadded = shard_height_unpadded;
        uint32_t num_sticks_per_core_padded = shard_height_padded;

        // Reducing num_sticks_written modulo the output dimensions gives this core's first source row,
        // and the start id for each dimension that the walk below advances from. A core whose slots all lie
        // past the final output row enters here with num_sticks_written already past the output row count,
        // so the reduction wraps and its first source row lands back inside the input. Such a core copies
        // rows into slots that lie outside the output and are never read back.
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

        // Group this core's source rows by the input shard holding them. Keying on the shard index puts the
        // groups in increasing shard index order, which is what the reader needs: it fills the output shard
        // front to back, one source core at a time in the order listed, and within a source core the rows
        // in the order listed. Row ids only grow as stick_ids_per_core is built above, so increasing shard
        // index order is also increasing output row order.
        std::map<uint32_t, std::vector<uint32_t>> shard_stick_map;
        for (uint32_t i = 0; i < num_sticks_per_core_unpadded; ++i) {
            uint32_t stick_id = stick_ids_per_core[i];
            uint32_t shard_id = stick_id / num_sticks_per_core_padded;
            uint32_t stick_id_in_shard = stick_id - (shard_id * num_sticks_per_core_padded);

            // When the core count does not divide the output row count, the output shards together have
            // room for more rows than the output holds, so a core that holds real rows can hold surplus
            // slots past the final output row as well. The walk above reaches those surplus slots by
            // running off the end of the input, returning ids at or past num_padded_sticks that name rows
            // the input does not have. Row ids only rise, so the ids at or past num_padded_sticks are the
            // tail of this core's list and stopping drops exactly them. The num_padded_sticks bound is
            // what keeps the reads of a core that holds both real rows and surplus slots inside the
            // input. A core whose slots all lie past the final output row is the separate case described
            // where start_id is derived above.
            if (stick_id >= num_padded_sticks) {
                break;
            }
            shard_stick_map[shard_id].push_back(stick_id_in_shard);
        }

        // reader rt args
        std::vector<uint32_t> reader_kernel_args;
        reader_kernel_args.reserve(1 + 3 * shard_stick_map.size() + 2 * num_sticks_per_core_unpadded);
        reader_kernel_args.push_back(shard_stick_map.size());  // num_cores

        for (const auto& shard_stick_pair : shard_stick_map) {
            auto core_physical = device->worker_core_from_logical_core(input_cores[shard_stick_pair.first]);
            reader_kernel_args.push_back(core_physical.x);  // noc x
            reader_kernel_args.push_back(core_physical.y);  // noc y
        }

        // coalesce the sticks into chunks
        std::vector<std::vector<std::vector<uint32_t>>> stick_chunks_per_core;
        stick_chunks_per_core.reserve(shard_stick_map.size());
        for (auto shard_stick_pair : shard_stick_map) {
            auto stick_chunks = group_contiguous_values(shard_stick_pair.second);
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

ttnn::device_operation::ProgramArtifacts SliceRmShardedProgramFactory::create_program_artifacts(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    using namespace ttnn::prim::slice_metal2;

    const auto& input = tensor_args.input;
    tt::tt_metal::IDevice* device = input.device();

    uint32_t num_padded_sticks = input.physical_volume() / input.padded_shape()[-1];
    [[maybe_unused]] uint32_t num_unpadded_sticks = output.physical_volume() / output.padded_shape()[-1];

    uint32_t W_unpadded = output.logical_shape()[-1];
    auto stick_size_unpadded = W_unpadded * output.element_size();

    // input shard spec
    auto shard_spec_padded = input.shard_spec().value();
    uint32_t shard_height_padded = shard_spec_padded.shape[0];

    auto& all_cores_padded = shard_spec_padded.grid;
    [[maybe_unused]] uint32_t num_cores_padded = shard_spec_padded.num_cores();
    // Which core holds which shard follows from a tensor's shard grid and its shard orientation, and the
    // input and the output each carry their own. The input's shard grid and orientation are read here; the
    // output's are read further down. corerange_to_cores walks the shard grid itself instead of the bounding
    // box around it, so a grid of several rectangles, or one placed away from core (0, 0), is listed
    // correctly. A sharded buffer builds its page mapping with the same call, so input_cores holds the input
    // buffer's own shard-to-core assignment.
    const bool row_major_padded = shard_spec_padded.orientation == ShardOrientation::ROW_MAJOR;
    const std::vector<CoreCoord> input_cores = corerange_to_cores(all_cores_padded, std::nullopt, row_major_padded);

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
    const bool row_major_unpadded = shard_spec_unpadded.orientation == ShardOrientation::ROW_MAJOR;

    auto& all_cores_unpadded = shard_spec_unpadded.grid;
    uint32_t num_cores_unpadded = shard_spec_unpadded.num_cores();
    const std::vector<CoreCoord> output_cores =
        corerange_to_cores(all_cores_unpadded, std::nullopt, row_major_unpadded);

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

    // Both DFBs are built on borrowed memory: their backing L1 address resolves from the input /
    // output TensorArgument each dispatch, so nothing here has to re-point them on a cache hit.
    // entry_size and num_entries vary with shard shape / element size, so padded_shape is folded
    // into compute_program_hash() to keep each unique sizing in its own cache entry; DFB sizing is
    // set once at spec construction and is not re-applied on a hit.
    DataflowBufferSpec dfb_in{
        .unique_id = SHARDED_IN,
        .entry_size = src_stride_bytes,
        .num_entries = shard_height_padded,
        .data_format_metadata = dfb_data_format,
        .borrowed_from = INPUT,
    };
    DataflowBufferSpec dfb_out{
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
        input_cores,
        num_cores_unpadded,
        shard_height_unpadded,
        shard_height_padded,
        num_padded_sticks);

    // The reader is the only kernel this factory builds, so it is the only toucher of either DFB
    // and binds both ends of each: a self-loop. Neither DFB carries data between kernels; each is a
    // window onto a resident shard that the reader addresses through its own cursor.
    KernelSpec reader{
        .unique_id = SHARDED_READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
            "slice_reader_unary_unpad_dims_rm_sharded.cpp",
        .dfb_bindings =
            {
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
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_cores_read"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // The reader runs on all_cores_unpadded, so every argument list must go to a core of that set.
    // get_slice_runtime_args_rm_sharded builds list i for output shard i, and output_cores[i] is the core
    // holding that shard.
    //
    // How many varargs a core takes depends on how many input shards its output shard draws from and
    // how those rows coalesce into chunks, so the count genuinely differs per core rather than being
    // one number for the kernel. num_runtime_varargs_per_node is the API's mechanism for that.
    KernelRunArgs reader_run_args{.kernel = SHARDED_READER};
    for (uint32_t i = 0; i < num_cores_unpadded; ++i) {
        const CoreCoord& core = output_cores[i];
        std::vector<uint32_t>& core_args = all_runtime_args[i].first;
        AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"num_cores_read", core_args[0]}});
        reader_run_args.advanced_options.runtime_varargs[core] =
            std::vector<uint32_t>(core_args.begin() + 1, core_args.end());
        reader.advanced_options.num_runtime_varargs_per_node[Nodes{core}] = static_cast<uint32_t>(core_args.size() - 1);
    }

    ProgramSpec spec{
        .name = "slice_rm_sharded",
        .kernels = {std::move(reader)},
        .dataflow_buffers = {std::move(dfb_in), std::move(dfb_out)},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {SHARDED_READER},
                    .target_nodes = all_cores_unpadded,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args)};
    run_args.tensor_args = {
        {INPUT, input.mesh_tensor()},
        {OUTPUT, output.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

// The per-dispatch run args of a cached slice program, for the factory that built it.
// Shared with MeshPartition, which drives these same factories directly, so the argument set has
// one home. Every shape-derived arg is keyed (both tensor specs, the slice params and
// factory.index() are folded into compute_program_hash), so the tensor bindings are all that move
// on a hit -- except on the two tile factories, whose per-core scalars are hash-excluded.
tt::tt_metal::experimental::ProgramRunArgs slice_program_run_args(
    const SliceDeviceOperation::program_factory_t& factory,
    const SliceParams& operation_attributes,
    const SliceInputs& tensor_args,
    Tensor& output) {
    using namespace ttnn::prim::slice_metal2;

    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {INPUT, tensor_args.input.mesh_tensor()},
        {OUTPUT, output.mesh_tensor()},
    };

    std::visit(
        [&](auto&& f) {
            using Factory = std::decay_t<decltype(f)>;
            if constexpr (std::is_same_v<Factory, SliceTileProgramFactory>) {
                // Divergent-partition hit leaves writer num_pages=0 -> all-zero output (#52651).
                const uint32_t start_offset = ttnn::operations::data_movement::get_tiled_start_offset(
                    tensor_args.input, operation_attributes.slice_start);
                run_args.kernel_run_args = slice_tile_run_args(
                    operation_attributes, tensor_args, output, start_offset, TILE_READER, TILE_WRITER);
            } else if constexpr (std::is_same_v<Factory, SliceTileTensorArgsProgramFactory>) {
                run_args.tensor_args.insert({START_TENSOR, tensor_args.start_tensor.value().mesh_tensor()});
                run_args.tensor_args.insert({END_TENSOR, tensor_args.end_tensor.value().mesh_tensor()});
                run_args.kernel_run_args = slice_tile_run_args(
                    operation_attributes, tensor_args, output, /*start_offset=*/0u, TA_READER, TA_WRITER);
            }
        },
        factory);

    return run_args;
}

tt::tt_metal::experimental::ProgramRunArgs SliceRmShardedProgramFactory::override_runtime_arguments(
    const SliceParams& args,
    const SliceInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    return slice_program_run_args(SliceRmShardedProgramFactory{}, args, tensor_args, output);
}

}  // namespace ttnn::prim
