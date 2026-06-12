// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm_sharded.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"

#include <filesystem>
#include <map>
#include <optional>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

constexpr const char* RMSHD_READER_KERNEL =
    "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
    "slice_reader_unary_unpad_dims_rm_sharded_m2.cpp";

m2::NodeCoord rmshd_node_of(const CoreCoord& c) {
    return m2::NodeCoord{static_cast<std::size_t>(c.x), static_cast<std::size_t>(c.y)};
}

std::vector<std::vector<uint32_t>> group_contiguous_values(std::vector<uint32_t>& values) {
    std::vector<std::vector<uint32_t>> chunks;
    if (values.empty()) {
        return chunks;
    }

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
    chunks.push_back(current_chunk);
    return chunks;
}

// Enqueue-invariant work geometry: a pure function of shape + shard spec + slice params (all in the
// cache key). The reader runtime args are a single variable-length blob per core, decoded positionally
// on device — kept verbatim from the descriptor-era layout:
//   [num_cores_read, (noc_x, noc_y)*num_cores_read, num_chunks_per_core*num_cores_read,
//    (chunk_start, chunk_len)*total_chunks]
struct SliceRmShardedGeometry {
    CoreRangeSet all_cores_unpadded;
    std::vector<CoreCoord> cores;  // in the legacy row-major / col-major index order
    uint32_t stick_size_padded = 0;
    uint32_t stick_size_unpadded = 0;
    uint32_t shard_height_unpadded = 0;

    tt::DataFormat src_cb_data_format = tt::DataFormat::Invalid;
    tt::DataFormat dst_cb_data_format = tt::DataFormat::Invalid;
    uint32_t cb_in_total_size = 0;
    uint32_t cb_out_total_size = 0;

    // Per-core reader arg blob, parallel to `cores`.
    std::vector<std::vector<uint32_t>> reader_args;
};

SliceRmShardedGeometry rmshd_compute_geometry(const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    IDevice* device = input.device();

    SliceRmShardedGeometry g;

    auto input_shape = input.padded_shape();
    auto output_shape = output.padded_shape();

    // Stick sizes.
    uint32_t W_padded = input.logical_shape()[-1];
    uint32_t W_unpadded = output.logical_shape()[-1];
    g.stick_size_padded = W_padded * input.element_size();
    g.stick_size_unpadded = W_unpadded * output.element_size();

    g.src_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    g.dst_cb_data_format = datatype_to_dataformat_converter(output.dtype());

    // Input shard spec (padded).
    auto shard_spec_padded = input.shard_spec().value();
    uint32_t shard_height_padded = shard_spec_padded.shape[0];
    auto bbox_padded = shard_spec_padded.grid.bounding_box();
    CoreCoord grid_size_padded = {bbox_padded.end_coord.x + 1, bbox_padded.end_coord.y + 1};
    uint32_t num_cores_x_padded = grid_size_padded.x;
    uint32_t num_cores_y_padded = grid_size_padded.y;

    if (args.sub_core_grids.has_value()) {
        log_warning(tt::LogOp, "sub_core_grids is not used when input tensor is sharded");
    }

    // Output shard spec (unpadded).
    auto shard_spec_unpadded = output.shard_spec().value();
    g.shard_height_unpadded = shard_spec_unpadded.shape[0];
    bool row_major = shard_spec_unpadded.orientation == ShardOrientation::ROW_MAJOR;
    g.all_cores_unpadded = shard_spec_unpadded.grid;
    uint32_t num_cores_unpadded = shard_spec_unpadded.num_cores();
    auto bbox_unpadded = g.all_cores_unpadded.bounding_box();
    CoreCoord grid_size_unpadded = {bbox_unpadded.end_coord.x + 1, bbox_unpadded.end_coord.y + 1};
    uint32_t num_cores_x_unpadded = grid_size_unpadded.x;
    uint32_t num_cores_y_unpadded = grid_size_unpadded.y;

    // Sharded CB total sizes. padded_shape is in the cache key (default reflection hash over
    // op type + attrs + tensor spec), so each unique sizing gets its own cache entry.
    g.cb_in_total_size = shard_height_padded * g.stick_size_padded;
    g.cb_out_total_size = g.shard_height_unpadded * g.stick_size_unpadded;

    // Per-dim stick counts (same recurrence as the descriptor era).
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

    g.cores.reserve(num_cores_unpadded);
    g.reader_args.reserve(num_cores_unpadded);

    uint32_t start_offset = ttnn::operations::data_movement::get_rm_start_offset(input, args.slice_start);
    for (uint32_t i = 0, num_sticks_written = 0; i < num_cores_unpadded; i++) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x_unpadded, i / num_cores_x_unpadded};
        } else {
            core = {i / num_cores_y_unpadded, i % num_cores_y_unpadded};
        }
        g.cores.push_back(core);

        uint32_t num_sticks_per_core_unpadded = g.shard_height_unpadded;
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

        // figure out the stick id in a shard, and the core id for the stick.
        std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> core_stick_map;
        for (uint32_t s = 0; s < num_sticks_per_core_unpadded; ++s) {
            uint32_t stick_id = stick_ids_per_core[s];
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

        // Build the variable-length per-core reader arg blob (decoded positionally on device).
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

        g.reader_args.push_back(std::move(reader_kernel_args));
    }

    return g;
}

}  // namespace

m2::ProgramSpec SliceRmShardedProgramFactory::create_program_spec(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto g = rmshd_compute_geometry(args, tensor_args, output);

    // Both CBs are sharded -> borrowed-memory DFBs. The descriptor era pinned
    // CBDescriptor::buffer = input.buffer()/output.buffer(); here the backing L1 address is
    // supplied per dispatch from the tensor argument (borrowed_from = TensorParamName). entry_size
    // is the per-stick page size; num_entries the shard height (whole shard, single CB entry-set).
    m2::DataflowBufferSpec cb_in{
        .unique_id = m2::DFBSpecName{"cb_in"},
        .entry_size = g.stick_size_padded,
        .num_entries = g.cb_in_total_size / g.stick_size_padded,
        .data_format_metadata = g.src_cb_data_format,
        .borrowed_from = m2::TensorParamName{"input"},
    };
    m2::DataflowBufferSpec cb_out{
        .unique_id = m2::DFBSpecName{"cb_out"},
        .entry_size = g.stick_size_unpadded,
        .num_entries = g.cb_out_total_size / g.stick_size_unpadded,
        .data_format_metadata = g.dst_cb_data_format,
        .borrowed_from = m2::TensorParamName{"output"},
    };

    // The reader is a data-movement kernel, so binding the sharded tensors to it (for the borrowed
    // CBs) is permitted. The kernel uses neither TensorAccessor — the shard-to-shard NOC reads use
    // explicit (noc_x, noc_y, l1_addr) endpoints built from the borrowed CB write pointers and the
    // per-core noc-coord varargs — so no tensor_bindings accessor name is needed on the kernel. The
    // borrowed_from on each DFB is what wires the tensor L1 address into the CB.
    //
    // Self-loop bindings: with no separate writer kernel, the single reader plays BOTH the producer
    // and consumer endpoint of each borrowed CB (matching the plusone sharded pattern).
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{RMSHD_READER_KERNEL},
        .dfb_bindings =
            {m2::ProducerOf(m2::DFBSpecName{"cb_in"}, "cb_in"),
             m2::ConsumerOf(m2::DFBSpecName{"cb_in"}, "cb_in"),
             m2::ProducerOf(m2::DFBSpecName{"cb_out"}, "cb_out"),
             m2::ConsumerOf(m2::DFBSpecName{"cb_out"}, "cb_out")},
        .compile_time_args =
            {{"stick_size_padded", g.stick_size_padded},
             {"stick_size_unpadded", g.stick_size_unpadded},
             {"num_sticks_unpadded", g.shard_height_unpadded}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementHardwareConfig::RoleHint::READER},
    };

    // Per-core reader arg blob is a single variable-length runtime vararg vector. Its length differs
    // per core (the noc-coord list and chunk list depend on which remote shards a core reads from),
    // so the count is declared per node via num_runtime_varargs_per_node (the scalar
    // num_runtime_varargs only supports a uniform count).
    // NOTE: num_runtime_varargs_per_node is [[deprecated]] in the Metal 2.0 API, but it is the ONLY
    // mechanism for VARIABLE per-node vararg counts (the scalar num_runtime_varargs is uniform-only).
    // The rm-sharded reader blob length genuinely differs per core. Suppress the deprecation warning
    // locally; this should be revisited if/when typed std::array kernel args land (see advanced_options.hpp).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    for (size_t i = 0; i < g.cores.size(); ++i) {
        reader.advanced_options.num_runtime_varargs_per_node.emplace(
            m2::Nodes{rmshd_node_of(g.cores[i])}, static_cast<uint32_t>(g.reader_args[i].size()));
    }
#pragma GCC diagnostic pop

    m2::ProgramSpec spec;
    spec.name = "slice_rm_sharded";
    spec.kernels = {std::move(reader)};
    spec.dataflow_buffers = {std::move(cb_in), std::move(cb_out)};
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = tensor_args.input.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output.tensor_spec()}};
    spec.work_units = {m2::WorkUnitSpec{
        .name = "slice_rm_sharded",
        .kernels = {m2::KernelSpecName{"reader"}},
        .target_nodes = g.all_cores_unpadded}};
    return spec;
}

m2::ProgramRunArgs SliceRmShardedProgramFactory::create_invariant_run_args(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto g = rmshd_compute_geometry(args, tensor_args, output);

    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    for (size_t i = 0; i < g.cores.size(); ++i) {
        const auto node = rmshd_node_of(g.cores[i]);
        reader_args.advanced_options.runtime_varargs.emplace(
            node, m2::AdvancedKernelRunArgs::Varargs(g.reader_args[i].begin(), g.reader_args[i].end()));
    }

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_args));
    return run_args;
}

m2::ProgramRunArgs SliceRmShardedProgramFactory::create_per_enqueue_args(
    const SliceParams& /*args*/,
    const SliceInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    m2::ProgramRunArgs run_args;
    // Borrowed-memory DFBs draw their backing L1 address from these tensor arguments.
    run_args.tensor_args.emplace(
        m2::TensorParamName{"input"}, m2::ProgramRunArgs::TensorArgument{std::cref(tensor_args.input.mesh_tensor())});
    run_args.tensor_args.emplace(
        m2::TensorParamName{"output"}, m2::ProgramRunArgs::TensorArgument{std::cref(output.mesh_tensor())});
    return run_args;
}

}  // namespace ttnn::prim
