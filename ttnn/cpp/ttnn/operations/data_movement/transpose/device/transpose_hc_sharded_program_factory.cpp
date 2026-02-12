// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_hc_sharded_program_factory.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>

#include <map>
#include <set>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_runtime_args_hc_rm_sharded(
    const Tensor& input_tensor, uint32_t num_cores, uint32_t num_cores_x, uint32_t num_cores_y) {
    auto input_shape = input_tensor.padded_shape();

    uint32_t H = input_shape[2], C = input_shape[1];

    auto shard_spec = input_tensor.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    IDevice* device = input_tensor.device();

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores);

    std::vector<uint32_t> shard_grid_x_map;
    for (uint32_t i = 0; i < num_cores_x; ++i) {
        auto physical_core = device->worker_core_from_logical_core(CoreCoord(i, 0));
        shard_grid_x_map.push_back(physical_core.x);
    }
    std::vector<uint32_t> shard_grid_y_map;
    for (uint32_t i = 0; i < num_cores_y; ++i) {
        auto physical_core = device->worker_core_from_logical_core(CoreCoord(0, i));
        shard_grid_y_map.push_back(physical_core.y);
    }

    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;
    for (uint32_t i = 0, curr_sticks_read = 0; i < num_cores; i++) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x, i / num_cores_x};
        } else {
            core = {i / num_cores_y, i % num_cores_y};
        }
        uint32_t num_sticks_per_core = shard_height;

        std::vector<uint32_t> reader_runtime_args = {num_sticks_per_core, curr_sticks_read, curr_c, curr_h, curr_n};
        reader_runtime_args.insert(reader_runtime_args.end(), shard_grid_x_map.begin(), shard_grid_x_map.end());
        reader_runtime_args.insert(reader_runtime_args.end(), shard_grid_y_map.begin(), shard_grid_y_map.end());

        std::vector<uint32_t> writer_runtime_args;

        ret_val[i] = {reader_runtime_args, writer_runtime_args};

        for (uint32_t j = 0; j < num_sticks_per_core; ++j) {
            curr_c++;
            curr_sticks_read += H;
            if (curr_c == C) {
                curr_h++;
                curr_c = 0;
                if (curr_h == H) {
                    curr_n++;
                    curr_c = 0;
                    curr_h = 0;
                    curr_sticks_read = curr_sticks_read - H + 1;
                } else {
                    curr_sticks_read = curr_sticks_read - C * H + 1;
                }
            }
        }
    }

    return ret_val;
}

std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> get_runtime_args_hc_rm_sharded_special_case(
    const Tensor& input_tensor, uint32_t num_cores, uint32_t num_cores_x, uint32_t num_cores_y) {
    auto input_shape = input_tensor.padded_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];
    uint32_t total_height = N * C * H;
    uint32_t stick_size_bytes = W * input_tensor.element_size();

    auto shard_spec = input_tensor.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    IDevice* device = input_tensor.device();

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> ret_val(num_cores);

    uint32_t height = 0;
    std::vector<CoreCoord> cores;
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core;
        if (row_major) {
            core = {i % num_cores_x, i / num_cores_x};
        } else {
            core = {i / num_cores_y, i % num_cores_y};
        }
        log_debug(tt::LogOp, "core: {}", core);

        height += shard_height;

        if (height <= total_height) {
            cores.push_back(core);
        }
    }

    uint32_t num_H_per_core = shard_height / H > 0 ? shard_height / H : 1;  // the number of H blocks in a shard
    uint32_t num_C_blocks_per_core = shard_height > C ? shard_height / C : 1;

    uint32_t curr_c = 0, curr_h = 0;
    for (uint32_t i = 0, curr_sticks_read = 0; i < num_cores; i++) {
        std::vector<uint32_t> read_cores_indices;
        std::vector<uint32_t> read_cores_noc_x;
        std::vector<uint32_t> read_cores_noc_y;
        std::vector<uint32_t> read_stick_offset;

        uint32_t num_sticks_per_core = shard_height;

        std::vector<uint32_t> stick_ids_per_core;
        for (uint32_t j = 0; j < num_sticks_per_core; ++j) {
            stick_ids_per_core.push_back(curr_sticks_read);
            curr_c++;
            curr_sticks_read += H;
            if (curr_c == C) {
                curr_h++;
                curr_c = 0;
                if (curr_h == H) {
                    curr_c = 0;
                    curr_h = 0;
                    curr_sticks_read = curr_sticks_read - H + 1;
                } else {
                    curr_sticks_read = curr_sticks_read - C * H + 1;
                }
            }
        }

        // figure out the stick id in a shard, and the core id for the stick.
        std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> core_stick_map;
        for (uint32_t j = 0; j < num_sticks_per_core; ++j) {
            uint32_t stick_id = stick_ids_per_core[j];
            uint32_t shard_id = stick_id / num_sticks_per_core;
            uint32_t stick_id_in_shard = stick_id - (shard_id * num_sticks_per_core);

            uint32_t shard_grid_inner_dim = row_major ? num_cores_x : num_cores_y;
            uint32_t shard_grid_outer_dim_id = shard_id / shard_grid_inner_dim;
            uint32_t shard_grid_inner_dim_id = shard_id - (shard_grid_outer_dim_id * shard_grid_inner_dim);

            uint32_t worker_y_logical = row_major ? shard_grid_outer_dim_id : shard_grid_inner_dim_id;
            uint32_t worker_x_logical = row_major ? shard_grid_inner_dim_id : shard_grid_outer_dim_id;

            if (worker_x_logical < num_cores_x && worker_y_logical < num_cores_y) {
                auto core_physical =
                    device->worker_core_from_logical_core(CoreCoord{worker_x_logical, worker_y_logical});

                read_cores_indices.push_back(shard_id);
                read_stick_offset.push_back(stick_id_in_shard * stick_size_bytes);
                read_cores_noc_x.push_back(core_physical.x);
                read_cores_noc_y.push_back(core_physical.y);
            }
        }

        std::vector<uint32_t> non_repeat_stick_offset_values;
        std::vector<uint32_t> non_repeat_noc_x_values;
        std::vector<uint32_t> non_repeat_noc_y_values;

        uint32_t num_sticks_per_shard_core_reader = 0, num_sticks_per_shard_core_writer = 0;
        uint32_t writer_read_stick_offset = 0, writer_write_stick_offset = 0;
        uint32_t num_C_blocks_per_core_reader = num_C_blocks_per_core, num_C_blocks_per_core_writer = 0;

        uint32_t num_non_repeat_cores = read_cores_indices.size();
        uint32_t read_stick_stride = read_stick_offset.size() > 1 ? read_stick_offset[1] - read_stick_offset[0] : 0;

        if (num_H_per_core == 1) {  // each core only has one H block or part of H block
            for (uint32_t k = 1; k < read_cores_indices.size(); ++k) {
                if (read_cores_indices[k] == read_cores_indices[0]) {
                    num_non_repeat_cores = k;
                    read_stick_stride = read_stick_offset[k] - read_stick_offset[0];
                    break;
                }
            }

            uint32_t num_sticks_per_shard_core = shard_height / num_non_repeat_cores;
            num_sticks_per_shard_core_reader = num_sticks_per_shard_core;
            bool split_reader = num_sticks_per_shard_core > 2;
            if (split_reader) {
                num_sticks_per_shard_core_reader = num_sticks_per_shard_core / 2;
                num_sticks_per_shard_core_writer = num_sticks_per_shard_core - num_sticks_per_shard_core_reader;
                writer_read_stick_offset = num_sticks_per_shard_core_reader * read_stick_stride;
                writer_write_stick_offset = writer_read_stick_offset * num_non_repeat_cores;
            }

            for (uint32_t k = 0; k < num_non_repeat_cores; ++k) {
                non_repeat_stick_offset_values.push_back(read_stick_offset[k]);
                non_repeat_noc_x_values.push_back(read_cores_noc_x[k]);
                non_repeat_noc_y_values.push_back(read_cores_noc_y[k]);
            }
        } else {  // contains multiple H blocks
            std::set<uint32_t> unique_values(read_cores_indices.begin(), read_cores_indices.end());
            num_non_repeat_cores = unique_values.size();
            read_stick_stride = read_stick_offset[1] - read_stick_offset[0];

            // TODO: add the second batch args (num_non_repeat_cores, read_stick_offset, non_repeat_noc_x_values,
            // non_repeat_noc_y_values) to support multiple batch in a shard
            for (uint32_t k = 1; k < num_sticks_per_core; ++k) {
                if ((read_cores_indices[k - 1] == read_cores_indices[k]) &&
                    (read_stick_offset[k] == read_stick_offset[k - 1] + stick_size_bytes)) {
                    break;
                }
            }

            uint32_t num_sticks_per_shard_core = shard_height / num_non_repeat_cores / num_C_blocks_per_core;
            num_sticks_per_shard_core_reader = num_sticks_per_shard_core;
            num_sticks_per_shard_core_writer = num_sticks_per_shard_core;
            bool split_reader = num_C_blocks_per_core > 2;
            if (split_reader) {
                num_C_blocks_per_core_reader = num_C_blocks_per_core / 2;
                num_C_blocks_per_core_writer = num_C_blocks_per_core - num_C_blocks_per_core_reader;
                writer_read_stick_offset = num_C_blocks_per_core_reader * stick_size_bytes;
                writer_write_stick_offset =
                    num_C_blocks_per_core_reader * num_non_repeat_cores * num_sticks_per_shard_core * stick_size_bytes;
            }

            for (uint32_t k = 0; k < num_non_repeat_cores; ++k) {
                non_repeat_stick_offset_values.push_back(read_stick_offset[k * num_sticks_per_shard_core]);
                non_repeat_noc_x_values.push_back(read_cores_noc_x[k * num_sticks_per_shard_core]);
                non_repeat_noc_y_values.push_back(read_cores_noc_y[k * num_sticks_per_shard_core]);
            }
        }

        bool read_single_h_block_per_core = num_H_per_core == 1;

        std::vector<uint32_t> reader_runtime_args = {
            static_cast<uint32_t>(read_single_h_block_per_core),
            num_C_blocks_per_core_reader,
            num_sticks_per_shard_core_reader,
            num_non_repeat_cores,
            read_stick_stride,
        };

        reader_runtime_args.insert(
            reader_runtime_args.end(), non_repeat_stick_offset_values.begin(), non_repeat_stick_offset_values.end());
        reader_runtime_args.insert(
            reader_runtime_args.end(), non_repeat_noc_x_values.begin(), non_repeat_noc_x_values.end());
        reader_runtime_args.insert(
            reader_runtime_args.end(), non_repeat_noc_y_values.begin(), non_repeat_noc_y_values.end());

        std::vector<uint32_t> writer_runtime_args = {
            static_cast<uint32_t>(read_single_h_block_per_core),
            num_C_blocks_per_core_writer,
            num_sticks_per_shard_core_writer,
            num_non_repeat_cores,
            read_stick_stride,
            writer_read_stick_offset,
            writer_write_stick_offset,
        };

        writer_runtime_args.insert(
            writer_runtime_args.end(), non_repeat_stick_offset_values.begin(), non_repeat_stick_offset_values.end());
        writer_runtime_args.insert(
            writer_runtime_args.end(), non_repeat_noc_x_values.begin(), non_repeat_noc_x_values.end());
        writer_runtime_args.insert(
            writer_runtime_args.end(), non_repeat_noc_y_values.begin(), non_repeat_noc_y_values.end());

        ret_val[i] = {reader_runtime_args, writer_runtime_args};
    }

    return ret_val;
}

}  // namespace

TransposeHCShardedProgramFactory::cached_program_t TransposeHCShardedProgramFactory::create(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_hc needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_hc needs to be allocated in a buffer on device!");

    Program program = CreateProgram();

    tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());

    uint32_t W = input_tensor.logical_shape()[3], H = input_tensor.logical_shape()[2];
    uint32_t C = input_tensor.logical_shape()[1], N = input_tensor.logical_shape()[0];
    uint32_t stick_size_bytes = W * input_tensor.element_size();

    auto shard_spec = input_tensor.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    bool row_major_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    bool is_special_case = false;
    if ((shard_spec.shape[0] % H == 0 || H % shard_spec.shape[0] == 0) &&
        (shard_spec.shape[0] % C == 0 || C % shard_spec.shape[0] == 0) && (C % H == 0 || H % C == 0) &&
        (shard_height <= C * H)) {
        is_special_case = true;
    }

    auto& all_cores = shard_spec.grid;
    uint32_t num_cores = shard_spec.num_cores();

    log_debug(tt::LogOp, "all_cores: {}", all_cores);
    log_debug(tt::LogOp, "num_cores: {}", num_cores);

    auto bbox = shard_spec.grid.bounding_box();
    CoreCoord grid_size = {bbox.end_coord.x + 1, bbox.end_coord.y + 1};
    uint32_t num_cores_x = grid_size.x;
    uint32_t num_cores_y = grid_size.y;

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(shard_height * stick_size_bytes, {{src0_cb_index, src0_cb_data_format}})
            .set_page_size(src0_cb_index, stick_size_bytes)
            .set_globally_allocated_address(*input_tensor.buffer());
    auto cb_src0 = CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = tt::CBIndex::c_16;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(shard_height * stick_size_bytes, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, stick_size_bytes)
            .set_globally_allocated_address(*output_tensor.buffer());
    auto cb_output = CreateCircularBuffer(program, all_cores, cb_output_config);

    std::vector<uint32_t> reader_compile_time_args;
    if (is_special_case) {
        reader_compile_time_args = {src0_cb_index, output_cb_index, stick_size_bytes};
    } else {
        reader_compile_time_args = {
            src0_cb_index,
            output_cb_index,
            N,
            H,
            C,
            stick_size_bytes,
            static_cast<uint32_t>(row_major_orientation),
            num_cores_x,
            num_cores_y};
    }

    std::map<std::string, std::string> reader_defines;
    if (is_special_case) {
        reader_defines["USE_SPECIAL_CASE"] = "1";
    }

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_hc_sharded_rm.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    KernelHandle writer_kernel_id{};
    if (is_special_case) {
        std::vector<uint32_t> writer_compile_time_args = {src0_cb_index, output_cb_index, stick_size_bytes};

        writer_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
            "writer_unary_transpose_hc_sharded_rm.cpp",
            all_cores,
            WriterDataMovementConfig(writer_compile_time_args));
    }

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> all_runtime_args;
    if (is_special_case) {
        all_runtime_args =
            get_runtime_args_hc_rm_sharded_special_case(input_tensor, num_cores, num_cores_x, num_cores_y);
    } else {
        all_runtime_args = get_runtime_args_hc_rm_sharded(input_tensor, num_cores, num_cores_x, num_cores_y);
    }

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core;
        if (row_major_orientation) {
            core = {i % num_cores_x, i / num_cores_x};
        } else {
            core = {i / num_cores_y, i % num_cores_y};
        }

        SetRuntimeArgs(program, reader_kernel_id, core, all_runtime_args[i].first);
        SetRuntimeArgs(program, writer_kernel_id, core, all_runtime_args[i].second);
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .cb_src0 = cb_src0,
         .cb_output = cb_output,
         .num_cores_x = num_cores_x,
         .num_cores_y = num_cores_y}};
}

void TransposeHCShardedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TransposeParams& /*operation_attributes*/,
    const TransposeInputs& tensor_args,
    Tensor& output_tensor) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    auto* const src_buffer = tensor_args.input.buffer();
    auto* const dst_buffer = output_tensor.buffer();

    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_src0, *src_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_output, *dst_buffer);
}

}  // namespace ttnn::prim
