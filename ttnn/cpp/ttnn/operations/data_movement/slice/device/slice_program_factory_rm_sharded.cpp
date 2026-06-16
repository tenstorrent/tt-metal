// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm_sharded.hpp"

#include <algorithm>
#include <map>
#include <optional>
#include <filesystem>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace m2 = tt::tt_metal::experimental;

namespace ttnn::operations::data_movement {

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

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor SliceRmShardedProgramFactory::create_descriptor(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    ProgramDescriptor desc;

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

    // Sharded CBs: total_size and page_size vary with shard shape / element size,
    // so padded_shape is folded into compute_program_hash() to keep each unique
    // sizing in its own cache entry.  On cache hit, the framework copies runtime
    // args and patches dynamic CB addresses (.buffer is set below); CB sizing
    // itself is not re-applied — it is carried by the cached descriptor.
    constexpr uint8_t src0_cb_index = 0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = shard_height_padded * stick_size_padded,
        .core_ranges = all_cores_unpadded,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = cb_data_format,
            .page_size = stick_size_padded,
        }}},
        .buffer = input.buffer(),
    });

    constexpr uint8_t output_cb_index = tt::CBIndex::c_16;
    desc.cbs.push_back(CBDescriptor{
        .total_size = shard_height_unpadded * stick_size_unpadded,
        .core_ranges = all_cores_unpadded,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = dst_cb_data_format,
            .page_size = stick_size_unpadded,
        }}},
        .buffer = output.buffer(),
    });

    std::vector<uint32_t> reader_ct_args = {
        static_cast<uint32_t>(stick_size_padded),
        static_cast<uint32_t>(stick_size_unpadded),
        static_cast<uint32_t>(shard_height_unpadded)};

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

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "slice_reader_unary_unpad_dims_rm_sharded.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores_unpadded;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};

    reader_desc.runtime_args.reserve(num_cores_unpadded);
    for (uint32_t i = 0; i < num_cores_unpadded; ++i) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x_unpadded, i / num_cores_x_unpadded};
        } else {
            core = {i / num_cores_y_unpadded, i % num_cores_y_unpadded};
        }
        reader_desc.runtime_args.emplace_back(core, std::move(all_runtime_args[i].first));
    }

    desc.kernels.push_back(std::move(reader_desc));

    return desc;
}

// Metal 2.0 (ProgramSpec) port of the ROW_MAJOR HEIGHT-sharded in/out factory. Mirrors
// create_descriptor's sizing and per-core runtime-arg computation exactly, but expresses it with
// the Metal 2.0 host API and points at the forked *_m2 reader kernel.
//
// Case-2 (bridge) port. The legacy factory had a SINGLE reader kernel that:
//   - borrows two CBs onto the input/output shard buffers (CBDescriptor::buffer = in/out buffer)
//     purely as L1 base-address sources (get_write_ptr()), and
//   - reads peer shards by a hand-rolled NoC walk over a host-computed physical core-coordinate
//     map plus per-shard stick-chunk descriptors threaded through runtime args.
// Arg mapping vs the legacy descriptor:
//   - The two borrowed CBs become two DataflowBufferSpecs with borrowed_from = src / dst. Each is
//     bound on the single reader as a self-loop (PRODUCER + CONSUMER) so the DFB endpoint invariant
//     is satisfied; the kernel reads their borrowed L1 base via get_write_ptr() exactly as before.
//     (cb_in is a pure address source; cb_out is produced into but has no consumer kernel in this
//     single-kernel factory — both are honored as self-loops.)
//   - The src/dst buffer addresses move from CBDescriptor::buffer into TensorParameter/TensorBinding
//     (the borrowed DFBs resolve their backing L1 address from the matching TensorArgument).
//   - stick_size_padded / stick_size_unpadded / num_sticks_unpadded: positional CTAs -> named CTAs.
//   - The per-core reader arg vector (num_cores_read, NoC x/y map, chunk descriptors) becomes
//     per-core runtime varargs (identical contents and layout).
ttnn::device_operation::ProgramArtifacts SliceRmShardedSpecProgramFactory::create_program_spec(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;

    // stick sizes
    uint32_t W_padded = input.logical_shape()[-1];
    uint32_t W_unpadded = output.logical_shape()[-1];
    auto stick_size_padded = W_padded * input.element_size();
    auto stick_size_unpadded = W_unpadded * output.element_size();

    // input shard spec
    auto shard_spec_padded = input.shard_spec().value();
    uint32_t shard_height_padded = shard_spec_padded.shape[0];

    auto bbox_padded = shard_spec_padded.grid.bounding_box();
    CoreCoord grid_size_padded = {bbox_padded.end_coord.x + 1, bbox_padded.end_coord.y + 1};
    uint32_t num_cores_x_padded = grid_size_padded.x;
    uint32_t num_cores_y_padded = grid_size_padded.y;

    if (args.sub_core_grids.has_value()) {
        log_warning(tt::LogOp, "sub_core_grids is not used when input tensor is sharded");
    }

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

    // Data formats are not needed in the spec path: the borrowed DFBs are DM-only (no compute
    // kernel binds them), so no data_format_metadata is required.
    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    m2::ProgramSpec spec;
    spec.name = "slice_rm_sharded";

    // --- DFBs (borrowed onto the src / dst shard buffers; sizing carried in the spec) ---
    // cb_in: borrowed onto src, used as an L1 base-address source for the peer-shard NoC reads.
    // cb_out: borrowed onto dst, produced into by the reader.
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"cb_in"},
            .entry_size = static_cast<uint32_t>(stick_size_padded),
            .num_entries = shard_height_padded,
            .borrowed_from = m2::TensorParamName{"src"},
        },
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"cb_out"},
            .entry_size = static_cast<uint32_t>(stick_size_unpadded),
            .num_entries = shard_height_unpadded,
            .borrowed_from = m2::TensorParamName{"dst"},
        },
    };

    // --- Reader kernel spec ---
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
                                        "slice_reader_unary_unpad_dims_rm_sharded_m2.cpp"},
        .compiler_options = {},
        .dfb_bindings =
            {
                // cb_in self-loop (pure address source onto src).
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"cb_in"},
                    .accessor_name = "cb_in",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER,
                },
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"cb_in"},
                    .accessor_name = "cb_in",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER,
                },
                // cb_out self-loop (produced into by the reader; no separate consumer kernel).
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"cb_out"},
                    .accessor_name = "cb_out",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER,
                },
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"cb_out"},
                    .accessor_name = "cb_out",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER,
                },
            },
        .compile_time_args =
            {
                {"stick_size_padded", static_cast<uint32_t>(stick_size_padded)},
                {"stick_size_unpadded", static_cast<uint32_t>(stick_size_unpadded)},
                {"num_sticks_unpadded", shard_height_unpadded},
            },
        .runtime_arg_schema = {},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
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

    // The per-core reader arg vector is variable-length (num_cores_read header + NoC x/y map +
    // chunk descriptors), so it is carried as per-core runtime varargs with identical contents and
    // layout. Per-node vararg counts differ, but the API's per-node count override is deprecated, so
    // a single uniform num_runtime_varargs (= the longest per-core vector) is declared and each
    // node's vector is zero-padded up to it. The kernel derives every index it reads from
    // get_vararg(0) (num_cores_read) and never touches the padding tail, so padding is inert.
    // --- ProgramRunArgs (computed first so the uniform vararg count is known for the KernelSpec) ---
    m2::ProgramRunArgs run;
    m2::KernelRunArgs reader_run{.kernel = m2::KernelSpecName{"reader"}};

    uint32_t max_varargs = 0;
    for (uint32_t i = 0; i < num_cores_unpadded; ++i) {
        max_varargs = std::max<uint32_t>(max_varargs, static_cast<uint32_t>(all_runtime_args[i].first.size()));
    }
    for (uint32_t i = 0; i < num_cores_unpadded; ++i) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x_unpadded, i / num_cores_x_unpadded};
        } else {
            core = {i / num_cores_y_unpadded, i % num_cores_y_unpadded};
        }
        const m2::NodeCoord node{core};
        m2::AdvancedKernelRunArgs::Varargs varargs(max_varargs, 0u);
        const auto& reader_args = all_runtime_args[i].first;
        std::copy(reader_args.begin(), reader_args.end(), varargs.begin());
        reader_run.advanced_options.runtime_varargs[node] = std::move(varargs);
    }
    reader.advanced_options.num_runtime_varargs = max_varargs;

    spec.kernels = {reader};
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"src"}, .spec = input.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"dst"}, .spec = output.tensor_spec()},
    };
    spec.work_units = std::vector<m2::WorkUnitSpec>{
        m2::WorkUnitSpec{
            .name = "slice_rm_sharded",
            .kernels = {m2::KernelSpecName{"reader"}},
            .target_nodes = all_cores_unpadded,
        },
    };

    run.kernel_run_args = {reader_run};
    run.tensor_args = {
        {m2::TensorParamName{"src"}, input.mesh_tensor()},
        {m2::TensorParamName{"dst"}, output.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::prim
