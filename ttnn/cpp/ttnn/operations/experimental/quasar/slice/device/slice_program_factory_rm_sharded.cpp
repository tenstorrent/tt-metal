// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/experimental/quasar/slice/device/slice_program_factory_rm_sharded.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <utility>
#include <vector>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/spec_run_args.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::operations::experimental::quasar {

namespace {

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

    uint32_t start_offset =
        ttnn::operations::experimental::quasar::get_rm_start_offset(input_tensor, output_tensor_start);
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

}  // namespace

}  // namespace ttnn::operations::experimental::quasar

namespace ttnn::prim::qsr {

namespace {

// Spec resource names. Prefixed to stay distinct under unity builds.
const DFBSpecName SRC0{"slice_rmsh_c0"};
const DFBSpecName OUT{"slice_rmsh_c16"};
const TensorParamName INPUT{"slice_rmsh_input"};
const TensorParamName OUTPUT{"slice_rmsh_output"};
const KernelSpecName READER{"slice_rmsh_reader"};

}  // namespace

ttnn::device_operation::ProgramSpecArtifacts SliceRmShardedProgramFactory::create_program_spec(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;

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
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // ---- Build the ProgramSpec ----
    // Both dataflow buffers are borrowed: instead of allocating Program-lifetime L1, the DFB is
    // built on top of the input/output tensor's L1 buffer (DataflowBufferSpec::borrowed_from binds
    // to a TensorParameter; the runtime supplies the backing L1 address from the tensor argument).
    // This expresses the legacy dynamic-CB rebinding (CBDescriptor::buffer set to input/output
    // buffer). The spec is the program-cache key, so each unique sizing keys its own cache entry.
    ProgramSpec spec;
    spec.name = "slice_rm_sharded";

    // Tensor parameters (the borrowed-DFB backing memory is bound to these).
    spec.tensor_parameters = {
        TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
    };

    // src0 DFB (legacy CB index 0): borrowed onto the input buffer.
    // entry_size * num_entries == shard_height_padded * stick_size_padded (legacy total_size).
    DataflowBufferSpec src0_dfb{
        .unique_id = SRC0,
        .entry_size = static_cast<uint32_t>(stick_size_padded),
        .num_entries = shard_height_padded,
        .data_format_metadata = cb_data_format,
        .borrowed_from = INPUT,
    };

    // output DFB (legacy CB index c_16): borrowed onto the output buffer.
    // entry_size * num_entries == shard_height_unpadded * stick_size_unpadded (legacy total_size).
    DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = static_cast<uint32_t>(stick_size_unpadded),
        .num_entries = shard_height_unpadded,
        .data_format_metadata = dst_cb_data_format,
        .borrowed_from = OUTPUT,
    };

    spec.dataflow_buffers = {src0_dfb, out_dfb};

    // Reader kernel. It reads from the input shard (resident in L1) via raw noc x/y addresses and
    // writes the unpadded sticks into the borrowed output DFB; there is no writer kernel. The reader
    // binds NO TensorAccessor (it does not access the tensors via a TensorAccessor). It is a
    // self-supplied PRODUCER of both borrowed DFBs: it calls get_write_ptr() on cb_in (to find the
    // input shard's L1 base) and reserve_back/get_write_ptr/push_back on cb_out.
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/slice/device/kernels/dataflow/"
            "slice_reader_unary_unpad_dims_rm_sharded.cpp",
        .dfb_bindings =
            {ProducerOf(SRC0, "cb_in"), ProducerOf(OUT, "cb_out")},
        // CTAs (legacy compile_time_args {stick_size_padded, stick_size_unpadded, shard_height_unpadded}).
        .compile_time_args =
            {{"stick_size_padded", static_cast<uint32_t>(stick_size_padded)},
             {"stick_size_unpadded", static_cast<uint32_t>(stick_size_unpadded)},
             {"num_sticks_unpadded", shard_height_unpadded}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    // Build the variable-length per-core reader args (legacy reader_kernel_args).
    auto all_runtime_args = ttnn::operations::experimental::quasar::get_slice_runtime_args_rm_sharded(
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

    // The reader's per-core args are a single variable-length blob carried as positional runtime
    // varargs. Because each core's length differs (num_cores_read, chunk counts and chunk lists all
    // vary per core), we use the documented per-node-varying vararg-count mechanism
    // (KernelAdvancedOptions::num_runtime_varargs_per_node, a deprecated-but-purpose-built API for
    // exactly this case). num_runtime_varargs is the default for any node not in the table.
    AdvancedKernelRunArgs reader_run_advanced;
    uint32_t max_vararg_count = 0;
    for (uint32_t i = 0; i < num_cores_unpadded; ++i) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x_unpadded, i / num_cores_x_unpadded};
        } else {
            core = {i / num_cores_y_unpadded, i % num_cores_y_unpadded};
        }
        auto& core_args = all_runtime_args[i].first;
        max_vararg_count = std::max(max_vararg_count, static_cast<uint32_t>(core_args.size()));
        reader_run_advanced.runtime_varargs.emplace(NodeCoord{core}, std::move(core_args));
    }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    {
        // Per-node vararg COUNT: required because the reader's vararg length varies per core. This
        // is the documented mechanism for per-node-varying vararg counts.
        reader.advanced_options.num_runtime_varargs = max_vararg_count;
        for (uint32_t i = 0; i < num_cores_unpadded; ++i) {
            CoreCoord core;
            if (row_major) {
                core = {i % num_cores_x_unpadded, i / num_cores_x_unpadded};
            } else {
                core = {i / num_cores_y_unpadded, i % num_cores_y_unpadded};
            }
            reader.advanced_options.num_runtime_varargs_per_node.emplace(
                Nodes{NodeCoord{core}}, static_cast<uint32_t>(all_runtime_args[i].first.size()));
        }
    }
#pragma GCC diagnostic pop

    spec.kernels = {reader};

    // Single work unit: the reader runs on every unpadded core.
    spec.work_units = {WorkUnitSpec{
        .name = "slice_rm_sharded",
        .kernels = {READER},
        .target_nodes = all_cores_unpadded,
    }};

    // ---- Assemble ProgramRunArgs ----
    // No named runtime args and no common varargs: every per-core value lives in the variable blob.
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{
            .kernel = READER,
            .advanced_options = std::move(reader_run_advanced),
        },
    };
    run_args.tensor_args.emplace(INPUT, TensorArgument{input.mesh_tensor()});
    run_args.tensor_args.emplace(OUTPUT, TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramSpecArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::qsr
