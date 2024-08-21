// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_decode_device_operation.hpp"
#include "ttnn/operations/core/work_split/work_split.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;
using namespace tt;

namespace ttnn::operations::experimental::transformer {

    operation::ProgramWithCallbacks multi_core_nlp_create_qkv_heads_decode(const Tensor &input_tensor, const uint32_t num_q_heads, const uint32_t num_kv_heads, const uint32_t head_dim, std::vector<Tensor>& output, CoreCoord compute_with_storage_grid_size) {

        tt_metal::Program program = tt_metal::CreateProgram();

        const auto& input_shape = input_tensor.get_legacy_shape();

        tt_metal::Device *device = input_tensor.device();

        tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

        uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

        uint32_t head_tiles = head_dim / TILE_WIDTH;
        uint32_t head_size = head_tiles * single_tile_size;

        uint32_t element_size = input_tensor.element_size();
        uint32_t sub_tile_line_bytes = 16 * element_size;
        auto q_shard_spec = output[0].shard_spec().value();
        auto q_cores = q_shard_spec.grid;
        auto q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / TILE_HW;
        auto in_shard_spec = input_tensor.shard_spec().value();
        auto in_cores = in_shard_spec.grid;
        auto in_num_tiles = in_shard_spec.shape[0] * in_shard_spec.shape[1] / TILE_HW;

        uint32_t q_output_cb_index = CB::c_out0;
        tt_metal::CircularBufferConfig cb_q_output_config =
            tt_metal::CircularBufferConfig(
                q_num_tiles * single_tile_size, {{q_output_cb_index, cb_data_format}})
                .set_page_size(q_output_cb_index, single_tile_size).set_globally_allocated_address(*output[0].buffer());
        auto cb_q_output = tt_metal::CreateCircularBuffer(program, q_cores, cb_q_output_config);

        auto k_shard_spec = output[1].shard_spec().value();
        auto k_cores = k_shard_spec.grid;
        auto k_num_tiles = k_shard_spec.shape[0] * k_shard_spec.shape[1] / TILE_HW;

        uint32_t k_output_cb_index = CB::c_out1;
        tt_metal::CircularBufferConfig cb_k_output_config =
            tt_metal::CircularBufferConfig(
                k_num_tiles * single_tile_size, {{k_output_cb_index, cb_data_format}})
                .set_page_size(k_output_cb_index, single_tile_size).set_globally_allocated_address(*output[1].buffer());
        auto cb_k_output = tt_metal::CreateCircularBuffer(program, k_cores, cb_k_output_config);

        auto v_shard_spec = output[0].shard_spec().value();
        auto v_cores = q_shard_spec.grid;
        auto v_num_tiles = v_shard_spec.shape[0] * v_shard_spec.shape[1] / TILE_HW;

        uint32_t v_output_cb_index = CB::c_out2;
        tt_metal::CircularBufferConfig cb_v_output_config =
            tt_metal::CircularBufferConfig(
                v_num_tiles * single_tile_size, {{v_output_cb_index, cb_data_format}})
                .set_page_size(v_output_cb_index, single_tile_size).set_globally_allocated_address(*output[2].buffer());
        auto cb_v_output = tt_metal::CreateCircularBuffer(program, v_cores, cb_v_output_config);

        uint32_t q_base_addr = input_tensor.buffer()->address();

        // cores to read and write to output
        uint32_t num_cores = q_cores.num_cores(); // number of cores of the output
        auto core_grid = q_cores.bounding_box();
        uint32_t num_cores_x = core_grid.end_coord.x + 1, num_cores_y = core_grid.end_coord.y + 1;
        const auto &cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, true);

        // cores for input
        uint32_t in_num_cores = in_cores.num_cores(); // number of cores of the input
        auto in_core_grid = in_cores.bounding_box();
        uint32_t in_num_cores_x = in_core_grid.end_coord.x + 1, in_num_cores_y = in_core_grid.end_coord.y + 1;

        std::vector<uint32_t> noc_x_coords;
        noc_x_coords.reserve(in_num_cores_x);
        for (uint32_t x = 0; x < in_num_cores_x; ++x) {
            noc_x_coords.push_back(device->worker_core_from_logical_core({x, 0}).x);
        }
        std::vector<uint32_t> noc_y_coords;
        noc_y_coords.reserve(in_num_cores_y);
        for (uint32_t y = 0; y < in_num_cores_y; ++y) {
            noc_y_coords.push_back(device->worker_core_from_logical_core({0, y}).y);
        }

        // We parallize the reader on risc0 and risc1, where each risc reads a sub-tile of the input (phase1 and phase2 of a tile respectively)
        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t) element_size,
            (std::uint32_t) sub_tile_line_bytes,
            q_output_cb_index,
            k_output_cb_index,
            v_output_cb_index,
            head_size,
            num_q_heads,
            num_kv_heads,
            head_tiles,
            1, // read the first phase
            in_num_cores_x,
            in_num_cores_y
        };
        auto reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp",
            q_cores,
            tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
        reader_compile_time_args[9] = 2;  // read the second phase
        auto writer_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp",
            q_cores,
            tt_metal::WriterDataMovementConfig(reader_compile_time_args));

        uint32_t q_start_addr = q_base_addr;

        for (uint32_t i = 0; i < num_cores; ++i) {
            uint32_t in_tile_offset_by_batch = i < 16 ? i * sub_tile_line_bytes : (i - 16) * sub_tile_line_bytes + 512*element_size;

            const auto& core = cores[i];
            std::vector<uint32_t> reader_runtime_args;
            reader_runtime_args.reserve(2 + in_num_cores_x + in_num_cores_y);
            reader_runtime_args = {
                in_tile_offset_by_batch,
                q_start_addr,
            };
            reader_runtime_args.insert(reader_runtime_args.end(), noc_x_coords.begin(), noc_x_coords.end());
            reader_runtime_args.insert(reader_runtime_args.end(), noc_y_coords.begin(), noc_y_coords.end());

            tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, reader_runtime_args);
        }

        auto override_runtime_arguments_callback = [
                reader_kernel_id,
                writer_kernel_id,
                num_cores,
                cb_q_output,
                cb_k_output,
                cb_v_output,
                cores,
                element_size,
                sub_tile_line_bytes
        ]
        (
            const void* operation,
            Program &program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors
        ) {

            auto src_buffer = input_tensors.at(0).buffer();

            uint32_t src_kv_buffer_addr = 0;

            auto dst_buffer_query = output_tensors.at(0).buffer();
            auto dst_buffer_key = output_tensors.at(1).buffer();
            auto dst_buffer_value = output_tensors.at(2).buffer();

            UpdateDynamicCircularBufferAddress(program, cb_q_output, *dst_buffer_query);
            UpdateDynamicCircularBufferAddress(program, cb_k_output, *dst_buffer_key);
            UpdateDynamicCircularBufferAddress(program, cb_v_output, *dst_buffer_value);

            uint32_t q_base_addr = input_tensors[0].buffer()->address();
            uint32_t q_start_addr = q_base_addr;

            for (uint32_t i = 0; i < num_cores; ++i) {
                uint32_t in_tile_offset_by_batch = i < 16 ? i * sub_tile_line_bytes : (i - 16) * sub_tile_line_bytes + 512*element_size;
                const auto& core = cores[i];
                auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = in_tile_offset_by_batch;
                runtime_args[1] = q_start_addr;

                auto &runtime_args_writer = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args_writer[0] = in_tile_offset_by_batch;
                runtime_args_writer[1] = q_start_addr;
            }
        };

        return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
    }
} // namespace ttnn::operations::experimental::transformer
