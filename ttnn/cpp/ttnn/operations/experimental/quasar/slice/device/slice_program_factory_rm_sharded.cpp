// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/experimental/quasar/slice/device/slice_program_factory_rm_sharded.hpp"

#include <map>
#include <optional>
#include <vector>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::operations::experimental::quasar {

namespace {

inline std::vector<std::vector<uint32_t>> group_contiguous_values_sharded(std::vector<uint32_t>& values) {
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

// Builds the per-core packed work-description vararg vectors (one per unpadded core), matching the
// positional layout the reader kernel decodes:
//   [0] num_cores_read, then (noc_x,noc_y) pairs, then num_stick_chunks, then (start_id,len) chunk pairs.
// input_cores lists the cores of the input tensor's shard grid in shard order, so input_cores[k] is the
// logical coordinate of the core holding input shard k. Every NOC coordinate written into a vararg
// vector is converted from an input_cores entry. num_padded_sticks is how many rows the input holds.
inline std::vector<std::vector<uint32_t>> get_slice_runtime_varargs_rm_sharded(
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

    std::vector<std::vector<uint32_t>> ret_val(num_cores_unpadded);

    uint32_t start_offset =
        ttnn::operations::experimental::quasar::get_rm_start_offset(input_tensor, output_tensor_start);
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
        for (uint32_t s = 0; s < num_sticks_per_core_unpadded; ++s) {
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
        for (uint32_t s = 0; s < num_sticks_per_core_unpadded; ++s) {
            uint32_t stick_id = stick_ids_per_core[s];
            uint32_t shard_id = stick_id / num_sticks_per_core_padded;
            uint32_t stick_id_in_shard = stick_id - (shard_id * num_sticks_per_core_padded);

            // A core's output shard has room for shard_height_unpadded rows. When the core count does not
            // divide the output row count, the shards together have room for more rows than the output
            // holds, so the last slots lie past the final output row and have no source row. For those,
            // stick_ids_per_core can return an input row id at or past num_padded_sticks. Row ids only
            // rise, so the ids past the input's end are the tail of this core's list, and leaving them out
            // shifts none of the rows that do have a source.
            if (stick_id >= num_padded_sticks) {
                break;
            }
            // Every input row lies in one of the input's own shards, so the row-id bound above already
            // puts the shard index inside input_cores. A shard index outside input_cores would mean the
            // input's shard height, shard grid and row count do not describe the same tensor.
            TT_FATAL(
                shard_id < input_cores.size(),
                "qsr::SliceRmShardedProgramFactory: input row {} falls in shard {}, but the input shard grid "
                "holds only {} shards.",
                stick_id,
                shard_id,
                input_cores.size());
            shard_stick_map[shard_id].push_back(stick_id_in_shard);
        }

        // reader varargs
        std::vector<uint32_t> reader_varargs;
        reader_varargs.reserve(1 + (3 * shard_stick_map.size()));
        reader_varargs.push_back(shard_stick_map.size());  // num_cores

        for (const auto& shard_stick_pair : shard_stick_map) {
            auto core_physical = device->worker_core_from_logical_core(input_cores[shard_stick_pair.first]);
            reader_varargs.push_back(core_physical.x);  // noc x
            reader_varargs.push_back(core_physical.y);  // noc y
        }

        // coalesce the sticks into chunks
        std::vector<std::vector<std::vector<uint32_t>>> stick_chunks_per_core;
        stick_chunks_per_core.reserve(shard_stick_map.size());
        size_t num_chunks_total = 0;
        for (auto shard_stick_pair : shard_stick_map) {
            auto stick_chunks = group_contiguous_values_sharded(shard_stick_pair.second);
            num_chunks_total += stick_chunks.size();
            stick_chunks_per_core.push_back(stick_chunks);

            reader_varargs.push_back(stick_chunks.size());  // num_chunks for current core
        }
        reader_varargs.reserve(reader_varargs.size() + (2 * num_chunks_total));
        for (const auto& stick_chunks : stick_chunks_per_core) {
            for (auto chunk : stick_chunks) {
                reader_varargs.push_back(chunk[0]);      // start id of a chunk
                reader_varargs.push_back(chunk.size());  // length of a chunk
            }
        }

        ret_val[i] = std::move(reader_varargs);
    }

    return ret_val;
}

}  // namespace

}  // namespace ttnn::operations::experimental::quasar

namespace ttnn::prim::qsr {

ttnn::device_operation::ProgramArtifacts SliceRmShardedProgramFactory::create_program_artifacts(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;

    uint32_t W_unpadded = output.logical_shape()[-1];
    auto stick_size_unpadded = W_unpadded * output.element_size();

    // Real per-row L1 stride is aligned_page_size(), not the compact payload (differs when W·E % 16 != 0).
    const uint32_t src_stride_bytes = input.buffer()->aligned_page_size();
    const uint32_t dst_stride_bytes = output.buffer()->aligned_page_size();
    const uint32_t begins_bytes = args.slice_start[-1] * input.element_size();
    TT_FATAL(
        begins_bytes % ::hal::get_l1_alignment() == 0,
        "qsr::SliceRmShardedProgramFactory: width-begin ({} bytes) must be L1-aligned.",
        begins_bytes);

    // input shard spec
    auto shard_spec_padded = input.shard_spec().value();
    uint32_t shard_height_padded = shard_spec_padded.shape[0];

    // Which core holds which shard follows from a tensor's shard grid and its shard orientation, and the
    // input and the output each carry their own. The input's shard grid and orientation are read here; the
    // output's are read further down. corerange_to_cores walks the shard grid itself instead of the bounding
    // box around it, so a grid of several rectangles, or one placed away from core (0, 0), is listed
    // correctly. A sharded buffer builds its page mapping with the same call, so input_cores holds the input
    // buffer's own shard-to-core assignment.
    const bool row_major_padded = shard_spec_padded.orientation == ShardOrientation::ROW_MAJOR;
    const std::vector<CoreCoord> input_cores =
        corerange_to_cores(shard_spec_padded.grid, std::nullopt, row_major_padded);
    const uint32_t num_padded_sticks = input.physical_volume() / input.padded_shape()[-1];

    if (args.sub_core_grids.has_value()) {
        log_warning(tt::LogOp, "sub_core_grids is not used when input tensor is sharded");
    }

    // output shard spec
    auto shard_spec_unpadded = output.shard_spec().value();
    uint32_t shard_height_unpadded = shard_spec_unpadded.shape[0];
    const bool row_major_unpadded = shard_spec_unpadded.orientation == ShardOrientation::ROW_MAJOR;

    auto& all_cores_unpadded = shard_spec_unpadded.grid;
    uint32_t num_cores_unpadded = shard_spec_unpadded.num_cores();
    const std::vector<CoreCoord> output_cores =
        corerange_to_cores(all_cores_unpadded, std::nullopt, row_major_unpadded);

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat dst_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    // Resource names
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};

    // Both shards are address-only resident buffers: the reader reads the input shard's L1 base
    // (cross-core NoC source) and writes the sliced result into the output shard's L1 base. Both are
    // reached via local TensorAccessors (tensor::input / tensor::output) in the kernel, so there are no
    // borrowed self-loop DFBs (which DM kernels may no longer bind as PRODUCER+CONSUMER).
    (void)cb_data_format;
    (void)dst_cb_data_format;

    // --- Per-core runtime varargs ---
    auto all_runtime_varargs = ttnn::operations::experimental::quasar::get_slice_runtime_varargs_rm_sharded(
        input,
        output,
        args.slice_start,
        input_cores,
        num_cores_unpadded,
        shard_height_unpadded,
        shard_height_padded,
        num_padded_sticks);

    // Every node must supply exactly num_runtime_varargs words — pad each core's vector to the max.
    uint32_t max_varargs = 0;
    for (const auto& v : all_runtime_varargs) {
        max_varargs = std::max<uint32_t>(max_varargs, static_cast<uint32_t>(v.size()));
    }

    // --- Reader KernelSpec ---
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/slice/device/kernels/dataflow/"
            "slice_reader_unary_unpad_dims_rm_sharded.cpp",
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"},
             TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args =
            {{"stick_size_unpadded", static_cast<uint32_t>(stick_size_unpadded)},
             {"src_stride_bytes", src_stride_bytes},
             {"dst_stride_bytes", dst_stride_bytes},
             {"begins_bytes", begins_bytes}},
        .hw_config = ttnn::create_reader_datamovement_config(input.device()->arch()),
        .advanced_options = {.num_runtime_varargs = max_varargs},
    };

    // --- Per-core runtime args (varargs only) ---
    // The WorkUnit targets all_cores_unpadded, and a vararg vector may only be set for a node the kernel
    // runs on, so every destination here has to be a core of that set. get_slice_runtime_varargs_rm_sharded
    // builds vector i for output shard i, and output_cores[i] is the core holding that shard.
    AdvancedKernelRunArgs reader_run_advanced;
    for (uint32_t i = 0; i < num_cores_unpadded; ++i) {
        std::vector<uint32_t> padded = std::move(all_runtime_varargs[i]);
        padded.resize(max_varargs, 0);
        reader_run_advanced.runtime_varargs.emplace(output_cores[i], std::move(padded));
    }

    // --- TensorParameters ---
    TensorParameter input_param{.unique_id = INPUT, .spec = input.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // --- Assemble ProgramSpec ---
    ProgramSpec spec;
    spec.name = "slice_rm_sharded";
    spec.kernels = {reader};
    spec.tensor_parameters = {input_param, output_param};
    spec.work_units = {WorkUnitSpec{
        .name = "slice_rm_sharded_wu",
        .kernels = {READER},
        .target_nodes = all_cores_unpadded,
    }};

    // --- Assemble ProgramRunArgs ---
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = {},
            .common_runtime_arg_values = {},
            .advanced_options =
                AdvancedKernelRunArgs{.runtime_varargs = std::move(reader_run_advanced.runtime_varargs)},
        },
    };
    run_args.tensor_args.emplace(INPUT, input.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, output.mesh_tensor());

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::qsr
