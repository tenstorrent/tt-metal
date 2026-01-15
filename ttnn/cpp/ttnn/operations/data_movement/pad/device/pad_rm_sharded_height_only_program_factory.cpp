// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_rm_sharded_height_only_program_factory.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/work_split.hpp>

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::operations::data_movement::pad::program {

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
}  // namespace

namespace {
inline std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_pad_runtime_args_rm_sharded(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const ttnn::Shape& input_tensor_start,
    uint32_t num_cores_padded,
    bool row_major,
    uint32_t num_cores_x_padded,
    uint32_t num_cores_y_padded,
    uint32_t shard_height_padded,
    uint32_t shard_height_unpadded,
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
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x_padded, i / num_cores_x_padded};
        } else {
            core = {i / num_cores_y_padded, i % num_cores_y_padded};
        }
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
        for (uint32_t i = 0; i < num_sticks_per_core_padded; ++i) {
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
        auto first_core = device->worker_core_from_logical_core(CoreCoord{0, 0});
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

                uint32_t worker_y_logical = row_major ? shard_grid_outer_dim_id : shard_grid_inner_dim_id;
                uint32_t worker_x_logical = row_major ? shard_grid_inner_dim_id : shard_grid_outer_dim_id;

                if (worker_x_logical < num_cores_x_unpadded and worker_y_logical < num_cores_y_unpadded) {
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

PadRmShardedHeightOnlyProgramFactory::cached_program_t PadRmShardedHeightOnlyProgramFactory::create(
    const PadParams& operation_attributes, const PadInputs& tensor_args, Tensor& output) {
    const auto& a = tensor_args.input;
    const auto& output_padded_shape = operation_attributes.output_padded_shape;
    const auto& pad_value = operation_attributes.pad_value;
    const auto& input_tensor_start = operation_attributes.input_tensor_start;
    Program program{};

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

    IDevice* device = a.device();

    // input shard spec
    auto shard_spec_unpadded = a.shard_spec().value();
    uint32_t shard_height_unpadded = shard_spec_unpadded.shape[0];
    bool row_major = shard_spec_unpadded.orientation == ShardOrientation::ROW_MAJOR;

    [[maybe_unused]] auto& all_cores_unpadded = shard_spec_unpadded.grid;
    [[maybe_unused]] uint32_t num_cores_unpadded = shard_spec_unpadded.num_cores();
    auto bbox_unpadded = shard_spec_unpadded.grid.bounding_box();
    CoreCoord grid_size_unpadded = {bbox_unpadded.end_coord.x + 1, bbox_unpadded.end_coord.y + 1};
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
    CoreCoord grid_size_padded = {bbox_padded.end_coord.x + 1, bbox_padded.end_coord.y + 1};
    uint32_t num_cores_x_padded = grid_size_padded.x;
    uint32_t num_cores_y_padded = grid_size_padded.y;

    log_debug(tt::LogOp, "num_unpadded_sticks: {}", num_unpadded_sticks);
    log_debug(tt::LogOp, "shard_height_unpadded: {}", shard_height_unpadded);
    log_debug(tt::LogOp, "all_cores_unpadded: {}", all_cores_unpadded);
    log_debug(tt::LogOp, "num_cores_unpadded: {}", num_cores_unpadded);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    uint32_t src0_cb_index = 0;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(
            shard_height_unpadded * stick_size_unpadded, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, stick_size_unpadded)
            .set_globally_allocated_address(*a.buffer());
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);

    uint32_t output_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            shard_height_padded * stick_size_padded, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, stick_size_padded)
            .set_globally_allocated_address(*output.buffer());
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_output_config);

    // construct const buffer with the pad_value
    bool not_pad_by_zero = pad_value != 0;
    uint32_t src1_cb_index = 1;
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(stick_size_padded, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, stick_size_padded);
    tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src1_config);

    uint32_t packed_pad_value;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32) {
        packed_pad_value = pad_value;
    } else if (a.dtype() == DataType::UINT16) {
        packed_pad_value = pack_two_uint16_into_uint32({float_to_uint16(pad_value), float_to_uint16(pad_value)});
    } else {
        packed_pad_value = pack_two_bfloat16_into_uint32({bfloat16(pad_value), bfloat16(pad_value)});
    }

    std::vector<uint32_t> reader_ct_args = {(std::uint32_t)stick_size_padded, (std::uint32_t)shard_height_padded};

    std::vector<uint32_t> writer_ct_args = {
        (std::uint32_t)N + front_pad[-4],
        (std::uint32_t)H + front_pad[-2],
        (std::uint32_t)C + front_pad[-3],
        (std::uint32_t)stick_size_padded,
        (std::uint32_t)N_padded,
        (std::uint32_t)H_padded,
        (std::uint32_t)C_padded,
        (std::uint32_t)num_zero_pad_sticks_read,
        (std::uint32_t)zero_pad_stick_size,
        (std::uint32_t)not_pad_by_zero,
        (std::uint32_t)packed_pad_value,
        (std::uint32_t)row_major_min_bytes,
        (std::uint32_t)(stick_size_padded / row_major_min_bytes)};

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/reader_pad_dims_rm_sharded.cpp",
        all_cores_padded,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/pad/device/kernels/dataflow/writer_pad_dims_rm_sharded.cpp",
        all_cores_padded,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    auto all_runtime_args = get_pad_runtime_args_rm_sharded(
        a,
        output,
        input_tensor_start,
        num_cores_padded,
        row_major,
        num_cores_x_padded,
        num_cores_y_padded,
        shard_height_padded,
        shard_height_unpadded,
        num_cores_x_unpadded,
        num_cores_y_unpadded);

    for (uint32_t i = 0; i < num_cores_padded; i++) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x_padded, i / num_cores_x_padded};
        } else {
            core = {i / num_cores_y_padded, i % num_cores_y_padded};
        }
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, all_runtime_args[i].first);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, all_runtime_args[i].second);
    }

    return cached_program_t{std::move(program), {cb_src0, cb_output}};
}

void PadRmShardedHeightOnlyProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const PadParams& /*operation_attributes*/,
    const PadInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto* src_buffer_a = tensor_args.input.buffer();
    auto* dst_buffer = tensor_return_value.buffer();

    UpdateDynamicCircularBufferAddress(cached_program.program, cached_program.shared_variables.cb_src0, *src_buffer_a);
    UpdateDynamicCircularBufferAddress(cached_program.program, cached_program.shared_variables.cb_output, *dst_buffer);
}

}  // namespace ttnn::operations::data_movement::pad::program
