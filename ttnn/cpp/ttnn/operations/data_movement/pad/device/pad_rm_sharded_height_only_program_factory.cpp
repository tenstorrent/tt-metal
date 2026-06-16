// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_rm_sharded_height_only_program_factory.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {
using ttnn::operations::data_movement::float_to_uint16;
using ttnn::operations::data_movement::pack_two_uint16_into_uint32;

namespace {
inline std::vector<std::vector<uint32_t>> group_contiguous_and_repeated_values(std::vector<uint32_t>& values) {
    std::vector<std::vector<uint32_t>> chunks;
    if (values.empty()) {
        return chunks;
    }

    // Initialize the first chunk
    std::vector<uint32_t> current_chunk;
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
    chunks.push_back(current_chunk);
    return chunks;
}

inline std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_pad_runtime_args_rm_sharded(
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

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> start_dim_offset(num_dims, 0);

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores_padded);

    const auto& front_pad = input_tensor_start;
    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;
    for (uint32_t i = 0, curr_sticks_read = 0; i < num_cores_padded; i++) {
        uint32_t num_sticks_per_core_unpadded = shard_height_unpadded;
        uint32_t num_sticks_per_core_padded = shard_height_padded;

        // writer rt args, set on top here as interleaved version.
        std::vector<uint32_t> writer_kernel_args = {
            num_sticks_per_core_padded,
            curr_sticks_read,
            front_pad[-4],
            front_pad[-3],
            front_pad[-2],
        };
        writer_kernel_args.insert(writer_kernel_args.end(), start_dim_offset.begin(), start_dim_offset.end());

        // figure out the start read stick id for each core, and the start id for each dim
        std::vector<int> stick_ids_per_core;
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

        start_dim_offset = {0, curr_h, curr_c, curr_n};

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

        // reader rt args
        std::vector<uint32_t> reader_kernel_args;
        reader_kernel_args.push_back(core_stick_map.size());  // num_cores

        for (const auto& core_stick_pair : core_stick_map) {
            auto xy_pair = core_stick_pair.first;
            if (row_major) {
                reader_kernel_args.push_back((std::uint32_t)xy_pair.second);  // noc x
                reader_kernel_args.push_back((std::uint32_t)xy_pair.first);   // noc y
            } else {
                reader_kernel_args.push_back((std::uint32_t)xy_pair.first);   // noc x
                reader_kernel_args.push_back((std::uint32_t)xy_pair.second);  // noc y
            }
        }

        // coalesce the sticks into chunks
        std::vector<std::vector<std::vector<uint32_t>>> stick_chunks_per_core;
        for (auto core_stick_pair : core_stick_map) {
            auto stick_chunks = group_contiguous_and_repeated_values(core_stick_pair.second);
            stick_chunks_per_core.push_back(stick_chunks);
            reader_kernel_args.push_back(stick_chunks.size());  // num_chunks for current core
        }
        for (const auto& stick_chunks : stick_chunks_per_core) {
            for (auto chunk : stick_chunks) {
                reader_kernel_args.push_back(chunk[0]);      // start id of a chunk
                reader_kernel_args.push_back(chunk.size());  // length of a chunk
            }
        }

        ret_val[i] = {reader_kernel_args, writer_kernel_args};
    }

    return ret_val;
}
}  // namespace

ttnn::device_operation::ProgramArtifacts PadRmShardedHeightOnlyProgramFactory::create_program_spec(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input;
    Tensor& output = tensor_return_value;
    const auto& output_padded_shape = operation_attributes.output_padded_shape;
    const auto& pad_value = operation_attributes.pad_value;
    const auto& input_tensor_start = operation_attributes.input_tensor_start;

    const auto& a_shape = a.logical_shape();
    uint32_t W = a_shape[3], H = a_shape[2], C = a_shape[1], N = a_shape[0];
    [[maybe_unused]] uint32_t num_unpadded_sticks = H * C * N;
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

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat dst_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

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

    TT_ASSERT(a.buffer() != nullptr, "Input buffer should be allocated on device!");
    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

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

    // ---- ProgramSpec (immutable) ----
    m2::ProgramSpec spec;
    spec.name = "pad_rm_sharded_height_only";

    // Borrowed-memory DFBs: in0 views the input shard's L1, out0 views the output shard's L1
    // (both supplied at runtime via the src/dst tensor args).  pad is a fake CB the writer fills
    // with the pad value and reads by base pointer (self-loop).
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"in0"},
            .entry_size = stick_size_unpadded,
            .num_entries = shard_height_unpadded,
            .data_format_metadata = cb_data_format,
            .borrowed_from = m2::TensorParamName{"src"}},
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"out0"},
            .entry_size = stick_size_padded,
            .num_entries = shard_height_padded,
            .data_format_metadata = dst_cb_data_format,
            .borrowed_from = m2::TensorParamName{"dst"}},
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"pad"},
            .entry_size = stick_size_padded,
            .num_entries = 1,
            .data_format_metadata = cb_data_format},
    };

    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"src"}, .spec = a.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"dst"}, .spec = output.tensor_spec()},
    };

    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
                                        "reader_pad_dims_rm_sharded.cpp"},
        // in0 is read by base pointer only (no FIFO) -> self-loop; out0 is produced by the reader.
        .dfb_bindings =
            {m2::DFBBinding{
                 .dfb_spec_name = m2::DFBSpecName{"in0"},
                 .accessor_name = "in0",
                 .endpoint_type = m2::DFBEndpointType::PRODUCER},
             m2::DFBBinding{
                 .dfb_spec_name = m2::DFBSpecName{"in0"},
                 .accessor_name = "in0",
                 .endpoint_type = m2::DFBEndpointType::CONSUMER},
             m2::DFBBinding{
                 .dfb_spec_name = m2::DFBSpecName{"out0"},
                 .accessor_name = "out0",
                 .endpoint_type = m2::DFBEndpointType::PRODUCER}},
        .compile_time_args =
            {{"stick_size_bytes", (std::uint32_t)stick_size_padded},
             {"num_sticks_padded", (std::uint32_t)shard_height_padded}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_cores_read"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/"
                                        "writer_pad_dims_rm_sharded.cpp"},
        // out0 is consumed (written by base pointer) by the writer; pad is a writer self-loop.
        .dfb_bindings =
            {m2::DFBBinding{
                 .dfb_spec_name = m2::DFBSpecName{"out0"},
                 .accessor_name = "out0",
                 .endpoint_type = m2::DFBEndpointType::CONSUMER},
             m2::DFBBinding{
                 .dfb_spec_name = m2::DFBSpecName{"pad"},
                 .accessor_name = "pad",
                 .endpoint_type = m2::DFBEndpointType::PRODUCER},
             m2::DFBBinding{
                 .dfb_spec_name = m2::DFBSpecName{"pad"},
                 .accessor_name = "pad",
                 .endpoint_type = m2::DFBEndpointType::CONSUMER}},
        .compile_time_args =
            {{"N", (std::uint32_t)N + front_pad[-4]},
             {"H", (std::uint32_t)H + front_pad[-2]},
             {"C", (std::uint32_t)C + front_pad[-3]},
             {"stick_size_bytes", (std::uint32_t)stick_size_padded},
             {"N_padded", (std::uint32_t)N_padded},
             {"H_padded", (std::uint32_t)H_padded},
             {"C_padded", (std::uint32_t)C_padded},
             {"num_zero_pad_sticks_read", (std::uint32_t)num_zero_pad_sticks_read},
             {"zero_pad_stick_size", (std::uint32_t)zero_pad_stick_size},
             {"not_pad_by_zero", (std::uint32_t)not_pad_by_zero},
             {"packed_pad_value", (std::uint32_t)packed_pad_value},
             {"row_major_min_bytes", (std::uint32_t)row_major_min_bytes},
             {"num_sticks_padded_read", (std::uint32_t)(stick_size_padded / row_major_min_bytes)}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"num_sticks_per_core", "start_id", "front_pad_n", "front_pad_c", "front_pad_h"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };

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

    // The reader's packed per-core tail (noc x/y + stick chunks) is variable-length per core.
    // It rides as runtime varargs; num_runtime_varargs is fixed at the max tail length and
    // shorter cores are zero-padded (the kernel only indexes within its own num_cores_read range,
    // so the trailing pad is never read).  The writer's start_dim_offset (4) rides as varargs too.
    uint32_t max_reader_tail = 0;
    for (uint32_t i = 0; i < num_cores_padded; i++) {
        max_reader_tail = std::max<uint32_t>(max_reader_tail, all_runtime_args[i].first.size() - 1);
    }
    reader.advanced_options.num_runtime_varargs = max_reader_tail;
    constexpr uint32_t kNumDimOffsets = 4;
    writer.advanced_options.num_runtime_varargs = kNumDimOffsets;

    spec.kernels = {reader, writer};

    // out0 (borrowed) is produced by the reader and consumed by the writer; in0/pad self-loops.
    // All endpoints must share a WorkUnitSpec — both kernels run on all_cores_padded.
    spec.work_units = std::vector<m2::WorkUnitSpec>{
        m2::WorkUnitSpec{
            .name = "sharded",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}},
            .target_nodes = all_cores_padded},
    };

    // ---- ProgramRunArgs (mutable) ----
    m2::ProgramRunArgs run;
    m2::KernelRunArgs reader_run{.kernel = m2::KernelSpecName{"reader"}};
    m2::KernelRunArgs writer_run{.kernel = m2::KernelSpecName{"writer"}};

    for (uint32_t i = 0; i < num_cores_padded; i++) {
        CoreCoord core;
        if (row_major) {
            core = {
                bbox_padded.start_coord.x + i % num_cores_x_padded, bbox_padded.start_coord.y + i / num_cores_x_padded};
        } else {
            core = {
                bbox_padded.start_coord.x + i / num_cores_y_padded, bbox_padded.start_coord.y + i % num_cores_y_padded};
        }
        const m2::NodeCoord node{core};

        // Reader: num_cores_read (named) + packed tail (varargs, zero-padded to max).
        const auto& reader_args = all_runtime_args[i].first;
        reader_run.runtime_arg_values.push_back({node, {{"num_cores_read", reader_args[0]}}});
        m2::AdvancedKernelRunArgs::Varargs reader_tail;
        reader_tail.reserve(max_reader_tail);
        for (size_t k = 1; k < reader_args.size(); ++k) {
            reader_tail.push_back(reader_args[k]);
        }
        reader_tail.resize(max_reader_tail, 0u);
        reader_run.advanced_options.runtime_varargs[node] = std::move(reader_tail);

        // Writer: 5 named scalar args + start_dim_offset[4] varargs (legacy slots 5..8).
        const auto& writer_args = all_runtime_args[i].second;
        writer_run.runtime_arg_values.push_back(
            {node,
             {{"num_sticks_per_core", writer_args[0]},
              {"start_id", writer_args[1]},
              {"front_pad_n", writer_args[2]},
              {"front_pad_c", writer_args[3]},
              {"front_pad_h", writer_args[4]}}});
        m2::AdvancedKernelRunArgs::Varargs writer_dims;
        writer_dims.reserve(kNumDimOffsets);
        for (uint32_t d = 0; d < kNumDimOffsets; ++d) {
            writer_dims.push_back(d + 5 < writer_args.size() ? writer_args[d + 5] : 0u);
        }
        writer_run.advanced_options.runtime_varargs[node] = std::move(writer_dims);
    }

    run.kernel_run_args = {reader_run, writer_run};
    run.tensor_args = {
        {m2::TensorParamName{"src"}, a.mesh_tensor()},
        {m2::TensorParamName{"dst"}, output.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::prim
