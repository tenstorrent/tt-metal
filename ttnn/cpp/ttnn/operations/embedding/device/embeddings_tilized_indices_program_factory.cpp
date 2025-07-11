// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "embeddings_tilized_indices_program_factory.hpp"
#include "embedding_common.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/embedding/device/embedding_device_operation.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/work_split.hpp>

#include <tracy/Tracy.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace ttnn::operations::embedding::detail {

tt::tt_metal::operation::ProgramWithCallbacks embeddings_tilized_indices(
    const Tensor& a,
    const Tensor& weights,
    Tensor& output,
    EmbeddingsType embeddings_type,
    std::optional<uint32_t> pad_token) {
    ////////////////////////////////////////////////////////////////////////////
    //                 Buffer Setup
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::Buffer* a_buffer = a.buffer();
    tt_metal::Buffer* weights_buffer = weights.buffer();
    tt_metal::Buffer* out_buffer = output.buffer();

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    auto device = a.device();
    auto dst_addr = output.buffer()->address();

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program{};

    bool in0_is_dram = a.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool weights_is_dram = weights.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool out_is_dram = output.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    uint32_t input_element_size_bytes = a.element_size();
    uint32_t weights_element_size_bytes = weights.element_size();
    uint32_t output_element_size_bytes = output.element_size();

    // row major, page size is last dim
    uint32_t input_page_size = a.logical_shape()[-1] * input_element_size_bytes;
    uint32_t weight_page_size = weights.padded_shape()[-1] * weights_element_size_bytes;
    uint32_t output_page_size = output.padded_shape()[-1] * output_element_size_bytes;

    // weights shape is [1, 1, num_embeddings, num_dim]
    uint32_t num_embeddings = weights.padded_shape()[-2];

    uint32_t batch_size = a.logical_shape()[0];  // num rows
    uint32_t num_cols = a.logical_shape()[-1];
    uint32_t volume = num_cols * batch_size;

    auto num_embedding_dims = weights.padded_shape()[-1];

    // setup problem and grid size
    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    uint32_t problem_size = volume;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    CoreSplitResult work = split_work_to_cores_aligned(compute_with_storage_grid_size, problem_size, FACE_HEIGHT);

    uint32_t num_cores = work.required_cores;
    CoreRangeSet all_cores = work.all_cores;
    CoreRangeSet core_group_1 = work.core_group_1;
    CoreRangeSet core_group_2 = work.core_group_2;
    uint32_t num_blocks_per_core_group_1 = work.units_per_core_group_1;
    uint32_t num_blocks_per_core_group_2 = work.units_per_core_group_2;

    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();

    // Create Buffers
    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());

    tt::DataFormat weights_cb_data_format = tt_metal::datatype_to_dataformat_converter(weights.dtype());
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());

    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    uint32_t rounded_weight_page_size = round_up_to_mul32(weight_page_size);
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(2 * rounded_weight_page_size, {{src0_cb_index, weights_cb_data_format}})
            .set_page_size(src0_cb_index, rounded_weight_page_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    constexpr uint32_t src1_cb_index = CBIndex::c_1;
    uint32_t index_page_size = round_up_to_mul32(input_element_size_bytes);
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(FACE_HEIGHT * index_page_size, {{src1_cb_index, input_cb_data_format}})
            .set_page_size(src1_cb_index, FACE_HEIGHT * index_page_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    constexpr uint32_t src2_cb_index = CBIndex::c_2;
    if (embeddings_type == EmbeddingsType::PADDED) {
        uint32_t cache_page_size = round_up_to_mul32(weight_page_size);
        tt_metal::CircularBufferConfig cb_src2_config =
            tt_metal::CircularBufferConfig(cache_page_size, {{src2_cb_index, weights_cb_data_format}})
                .set_page_size(src2_cb_index, cache_page_size);
        auto cb_src2 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src2_config);
    } else if (embeddings_type == EmbeddingsType::BINARY) {
        uint32_t cache_page_size = round_up_to_mul32(weight_page_size);
        tt_metal::CircularBufferConfig cb_src2_config =
            tt_metal::CircularBufferConfig(2 * cache_page_size, {{src2_cb_index, weights_cb_data_format}})
                .set_page_size(src2_cb_index, cache_page_size);
        auto cb_src2 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src2_config);
    }

    uint32_t output_cb_index = src0_cb_index;

    bool input_stick_size_is_power_of_two = is_power_of_two_at_least_32(input_page_size);
    uint32_t input_log2_stick_size = input_stick_size_is_power_of_two ? (std::uint32_t)std::log2(input_page_size) : 0;
    bool weight_stick_size_is_power_of_two = is_power_of_two_at_least_32(weight_page_size);
    uint32_t weight_log2_stick_size =
        weight_stick_size_is_power_of_two ? (std::uint32_t)std::log2(weight_page_size) : 0;

    // Create Kernels
    // reader
    std::vector<uint32_t> embedding_compile_time_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src1_cb_index,
        (std::uint32_t)src2_cb_index,
        (std::uint32_t)in0_is_dram,
        (std::uint32_t)input_page_size,
        (std::uint32_t)weights_is_dram,
        (std::uint32_t)weight_page_size,
        (std::uint32_t)a.logical_shape()[-1],  // width/length of a row
        (std::uint32_t)FACE_HEIGHT};

    EmbeddingsIndexType embeddings_index_type;
    if (a.dtype() == DataType::BFLOAT16) {
        embeddings_index_type = EmbeddingsIndexType::BFP16;
    } else {
        embeddings_index_type = EmbeddingsIndexType::UINT32;
    }

    std::map<std::string, std::string> embedding_defines = {
        {magic_enum::enum_name(embeddings_type).data(), "1"},
        {magic_enum::enum_name(embeddings_index_type).data(), "1"}};

    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embedding_ind_tilized.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(embedding_compile_time_args, embedding_defines));

    bool output_stick_size_is_power_of_two = is_power_of_two_at_least_32(output_page_size);
    uint32_t output_log2_stick_size =
        output_stick_size_is_power_of_two ? (std::uint32_t)std::log2(output_page_size) : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)output_cb_index,
        (std::uint32_t)out_is_dram,
        (std::uint32_t)output_stick_size_is_power_of_two,
        (std::uint32_t)output_log2_stick_size};

    // Tilized writer
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    uint32_t col_offset = 0;
    uint32_t weight_offset = 0;

    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);
    std::vector<uint32_t> reader_runtime_args = {
        (std::uint32_t)a.buffer()->address(),
        (std::uint32_t)weights.buffer()->address(),
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,
    };
    if (embeddings_type == EmbeddingsType::PADDED) {
        reader_runtime_args.push_back(pad_token.value());
    }
    std::vector<uint32_t> writer_runtime_args = {
        (std::uint32_t)output.buffer()->address(), (std::uint32_t)output_page_size, (std::uint32_t)0, (std::uint32_t)0};

    uint32_t row = 0;
    uint32_t tiles_per_tile_row = (num_cols + TILE_HEIGHT - 1) / TILE_HEIGHT;

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        col_offset = weight_offset % num_cols;
        row = weight_offset / num_cols;

        uint32_t local_num_blocks = i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;
        uint32_t r_f_offset = ((row % TILE_HEIGHT) / FACE_HEIGHT) * 2 * FACE_HW + (row % FACE_HEIGHT) * FACE_HEIGHT;
        // Offset by one face size if we are in the right half of the tile + where we are in the row
        uint32_t c_f_offset = ((col_offset % TILE_HEIGHT) / FACE_HEIGHT) * FACE_HW;
        uint32_t face_offset = r_f_offset + c_f_offset;
        uint32_t curr_tile = (row / TILE_HEIGHT) * tiles_per_tile_row + (col_offset / TILE_HEIGHT);

        // Reader
        {
            reader_runtime_args[2] = curr_tile;
            reader_runtime_args[3] = face_offset;
            reader_runtime_args[4] = local_num_blocks;
            reader_runtime_args[5] = col_offset;
            reader_runtime_args[6] = (col_offset % FACE_HEIGHT);  // starting col in the face row
            tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        }

        // Writer
        {
            writer_runtime_args[2] = local_num_blocks;
            writer_runtime_args[3] = weight_offset;
            tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        }

        weight_offset += local_num_blocks;
    }

    auto override_runtime_arguments_callback =
        [reader_kernel_id, writer_kernel_id, cores](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto output_buffer_address = output_tensors.at(0).buffer()->address();
            auto input_buffer_address = input_tensors.at(0).buffer()->address();
            auto weights_buffer_address = input_tensors.at(1).buffer()->address();

            auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id);
            auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id);

            for (const auto& core : cores) {
                {
                    auto& runtime_args = reader_runtime_args[core.x][core.y];
                    runtime_args[0] = input_buffer_address;
                    runtime_args[1] = weights_buffer_address;
                }

                {
                    auto& runtime_args = writer_runtime_args[core.x][core.y];
                    runtime_args[0] = output_buffer_address;
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::embedding::detail
