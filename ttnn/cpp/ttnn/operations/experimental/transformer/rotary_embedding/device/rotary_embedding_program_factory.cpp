// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/transformer/rotary_embedding/device/rotary_embedding_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::experimental::transformer::rotary_embedding::program {

using namespace tt::constants;

RotaryEmbeddingProgramFactory::cached_program_t RotaryEmbeddingProgramFactory::create(
    const RotaryEmbeddingParams& operation_attributes,
    const RotaryEmbeddingInputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& cos = tensor_args.cos;
    const auto& sin = tensor_args.sin;
    auto& output = tensor_return_value;
    const auto& token_idx = operation_attributes.token_idx;

    Program program{};

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    tt::DataFormat cos_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(cos.dtype());
    uint32_t cos_single_tile_size = tt::tile_size(cos_cb_data_format);

    tt::DataFormat sin_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(sin.dtype());
    uint32_t sin_single_tile_size = tt::tile_size(sin_cb_data_format);

    tt::DataFormat scalar_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t scalar_single_tile_size = tt::tile_size(scalar_cb_data_format);

    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    uint32_t num_rows = input.physical_volume() / input.padded_shape()[-1] / TILE_HEIGHT;
    uint32_t Ht = input.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t Wt = input.padded_shape()[-1] / TILE_WIDTH;
    uint32_t half_Wt = Wt / 2;
    uint32_t HtWt = Ht * Wt;
    uint32_t Wbytes = input.padded_shape()[-1] * sizeof(bfloat16);

    tt::tt_metal::IDevice* device = input.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    bool row_major;
    uint32_t num_cores, num_rows_per_core_group_1, num_rows_per_core_group_2;

    CoreRangeSet all_cores, core_group_1, core_group_2;

    bool in_sharded = input.shard_spec().has_value();
    bool out_sharded = output.shard_spec().has_value();
    std::optional<ShardSpec> shard_spec = in_sharded ? input.shard_spec() : output.shard_spec();

    uint32_t num_input_tiles, num_output_tiles;

    if (shard_spec.has_value()) {
        row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
        all_cores = shard_spec.value().grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet();
        num_rows_per_core_group_1 = shard_spec.value().shape[0] / TILE_HEIGHT;
        num_rows_per_core_group_2 = 0;
        num_input_tiles = in_sharded ? shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW : 2 * Wt;
        num_output_tiles = out_sharded ? shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW : 2 * Wt;
        auto bbox = all_cores.bounding_box();
        num_cores_x = bbox.end_coord.x + 1;
        num_cores_y = bbox.end_coord.y + 1;
    } else {
        row_major = true;
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows, row_major);
        num_input_tiles = 2 * Wt;
        num_output_tiles = num_input_tiles;
    }

    uint32_t input_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_input_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_tiles * input_single_tile_size, {{input_cb_index, input_cb_data_format}})
            .set_page_size(input_cb_index, input_single_tile_size);
    if (in_sharded) {
        cb_input_config = cb_input_config.set_globally_allocated_address(*input.buffer());
    }
    auto cb_input = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_input_config);

    uint32_t rotated_input_cb_index = tt::CBIndex::c_1;
    uint32_t num_rotated_input_tiles = 2 * Wt;
    tt::tt_metal::CircularBufferConfig cb_rotated_input_config =
        tt::tt_metal::CircularBufferConfig(
            num_rotated_input_tiles * input_single_tile_size, {{rotated_input_cb_index, input_cb_data_format}})
            .set_page_size(rotated_input_cb_index, input_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_rotated_input_config);

    uint32_t num_cos_sin_tiles = token_idx.has_value() ? Wt : 2 * Wt;
    uint32_t cos_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_cos_config =
        tt::tt_metal::CircularBufferConfig(
            num_cos_sin_tiles * cos_single_tile_size, {{cos_cb_index, cos_cb_data_format}})
            .set_page_size(cos_cb_index, cos_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_cos_config);

    uint32_t sin_cb_index = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig cb_sin_config =
        tt::tt_metal::CircularBufferConfig(
            num_cos_sin_tiles * sin_single_tile_size, {{sin_cb_index, sin_cb_data_format}})
            .set_page_size(sin_cb_index, sin_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_sin_config);

    // Used for bcast scalar
    uint32_t src_scalar_cb_index = tt::CBIndex::c_4;
    uint32_t num_scalar_tiles = 1;
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(
            num_scalar_tiles * scalar_single_tile_size, {{src_scalar_cb_index, scalar_cb_data_format}})
            .set_page_size(src_scalar_cb_index, scalar_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    uint32_t num_interm_tiles = 1;
    uint32_t rotated_input_interm_cb_index = tt::CBIndex::c_24;
    tt::tt_metal::CircularBufferConfig cb_rotated_input_interm_config =
        tt::tt_metal::CircularBufferConfig(
            num_interm_tiles * input_single_tile_size, {{rotated_input_interm_cb_index, input_cb_data_format}})
            .set_page_size(rotated_input_interm_cb_index, input_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_rotated_input_interm_config);

    uint32_t cos_interm_cb_index = tt::CBIndex::c_25;
    tt::tt_metal::CircularBufferConfig cb_cos_interm_config =
        tt::tt_metal::CircularBufferConfig(
            num_interm_tiles * cos_single_tile_size, {{cos_interm_cb_index, cos_cb_data_format}})
            .set_page_size(cos_interm_cb_index, cos_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_cos_interm_config);

    uint32_t sin_interm_cb_index = tt::CBIndex::c_26;
    tt::tt_metal::CircularBufferConfig cb_sin_interm_config =
        tt::tt_metal::CircularBufferConfig(
            num_interm_tiles * sin_single_tile_size, {{sin_interm_cb_index, sin_cb_data_format}})
            .set_page_size(sin_interm_cb_index, sin_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_sin_interm_config);

    uint32_t output_cb_index = tt::CBIndex::c_16;  // output operands start at index 16
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * output_single_tile_size, {{output_cb_index, output_cb_data_format}})
            .set_page_size(output_cb_index, output_single_tile_size);
    if (out_sharded) {
        cb_output_config = cb_output_config.set_globally_allocated_address(*output.buffer());
    }
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    uint32_t untilized_cos_interm_cb_index = tt::CBIndex::c_27;
    uint32_t untilized_cos_sync_cb_index = tt::CBIndex::c_5;
    uint32_t untilized_sin_interm_cb_index = tt::CBIndex::c_28;
    uint32_t untilized_sin_sync_cb_index = tt::CBIndex::c_6;
    uint32_t retilized_cos_cb_index = tt::CBIndex::c_29;
    uint32_t retilized_sin_cb_index = tt::CBIndex::c_30;
    std::map<std::string, std::string> reader_kernel_defines, writer_kernel_defines, compute_kernel_defines;
    if (token_idx.has_value()) {
        tt::tt_metal::CircularBufferConfig cb_cos2_config =
            tt::tt_metal::CircularBufferConfig(
                Wt * cos_single_tile_size, {{retilized_cos_cb_index, cos_cb_data_format}})
                .set_page_size(retilized_cos_cb_index, cos_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_cos2_config);

        tt::tt_metal::CircularBufferConfig cb_sin2_config =
            tt::tt_metal::CircularBufferConfig(
                Wt * sin_single_tile_size, {{retilized_sin_cb_index, sin_cb_data_format}})
                .set_page_size(retilized_sin_cb_index, sin_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_sin2_config);

        std::map<uint8_t, tt::DataFormat> cos_interim_data_format_spec = {
            {untilized_cos_interm_cb_index, scalar_cb_data_format},
            {untilized_cos_sync_cb_index, scalar_cb_data_format}};
        tt::tt_metal::CircularBufferConfig cb_untilized_cos_interm_config =
            tt::tt_metal::CircularBufferConfig(Wt * scalar_single_tile_size, cos_interim_data_format_spec)
                .set_page_size(untilized_cos_interm_cb_index, scalar_single_tile_size)
                .set_page_size(untilized_cos_sync_cb_index, scalar_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_untilized_cos_interm_config);

        std::map<uint8_t, tt::DataFormat> sin_interim_data_format_spec = {
            {untilized_sin_interm_cb_index, scalar_cb_data_format},
            {untilized_sin_sync_cb_index, scalar_cb_data_format}};
        tt::tt_metal::CircularBufferConfig cb_untilized_sin_interm_config =
            tt::tt_metal::CircularBufferConfig(Wt * scalar_single_tile_size, sin_interim_data_format_spec)
                .set_page_size(untilized_sin_interm_cb_index, scalar_single_tile_size)
                .set_page_size(untilized_sin_sync_cb_index, scalar_single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_untilized_sin_interm_config);
        reader_kernel_defines["DECODE_MODE"] = "1";
        writer_kernel_defines["DECODE_MODE"] = "1";
        compute_kernel_defines["DECODE_MODE"] = "1";
    }

    const uint16_t bfloat16_scalar = std::bit_cast<uint16_t>(bfloat16(-1.0f));

    auto* src_buffer = input.buffer();
    auto* cos_buffer = cos.buffer();
    auto* sin_buffer = sin.buffer();
    auto* dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args;
    if (in_sharded) {
        reader_compile_time_args = {
            (std::uint32_t)input_cb_index,
            (std::uint32_t)rotated_input_cb_index,
            (std::uint32_t)cos_cb_index,
            (std::uint32_t)sin_cb_index,
            (std::uint32_t)src_scalar_cb_index,
            (std::uint32_t)bfloat16_scalar,
            (std::uint32_t)Ht,
            (std::uint32_t)Wt,
            (std::uint32_t)HtWt,
            (std::uint32_t)half_Wt * input_single_tile_size,
        };
        tt::tt_metal::TensorAccessorArgs(cos_buffer).append_to(reader_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(sin_buffer).append_to(reader_compile_time_args);
    } else {
        reader_compile_time_args = {
            (std::uint32_t)input_cb_index,
            (std::uint32_t)rotated_input_cb_index,
            (std::uint32_t)cos_cb_index,
            (std::uint32_t)sin_cb_index,
            (std::uint32_t)src_scalar_cb_index,
            (std::uint32_t)bfloat16_scalar,
            (std::uint32_t)Ht,
            (std::uint32_t)Wt,
            (std::uint32_t)HtWt,
            (std::uint32_t)half_Wt,
        };
        tt::tt_metal::TensorAccessorArgs(src_buffer).append_to(reader_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(cos_buffer).append_to(reader_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(sin_buffer).append_to(reader_compile_time_args);
    }
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(writer_compile_time_args);

    if (token_idx.has_value()) {
        writer_compile_time_args.insert(
            writer_compile_time_args.end(),
            {untilized_cos_interm_cb_index,
             untilized_cos_sync_cb_index,
             untilized_sin_interm_cb_index,
             untilized_sin_sync_cb_index});
    }

    if (out_sharded) {
        writer_kernel_defines["OUT_SHARDED"] = "1";
    }

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        in_sharded ? "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/dataflow/"
                     "reader_rotary_embedding_interleaved_start_id_sharded.cpp"
                   : "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/dataflow/"
                     "reader_rotary_embedding_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_kernel_defines));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/dataflow/"
        "writer_rotary_embedding_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_kernel_defines));

    std::vector<uint32_t> compute_kernel_args = {
        (std::uint32_t)input_cb_index,
        (std::uint32_t)rotated_input_cb_index,
        (std::uint32_t)cos_cb_index,
        (std::uint32_t)sin_cb_index,
        (std::uint32_t)src_scalar_cb_index,
        (std::uint32_t)rotated_input_interm_cb_index,
        (std::uint32_t)cos_interm_cb_index,
        (std::uint32_t)sin_interm_cb_index,
        (std::uint32_t)output_cb_index,
        (std::uint32_t)num_rows_per_core_group_1,
        (std::uint32_t)Wt,
        (std::uint32_t)half_Wt};
    if (token_idx.has_value()) {
        compute_kernel_args.insert(
            compute_kernel_args.end(),
            {(std::uint32_t)untilized_cos_interm_cb_index,
             (std::uint32_t)untilized_cos_sync_cb_index,
             (std::uint32_t)untilized_sin_interm_cb_index,
             (std::uint32_t)untilized_sin_sync_cb_index,
             (std::uint32_t)retilized_cos_cb_index,
             (std::uint32_t)retilized_sin_cb_index});
    }

    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/compute/"
        "rotary_embedding.cpp",
        core_group_1,
        tt::tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = compute_kernel_defines});
    if (!core_group_2.ranges().empty()) {
        compute_kernel_args[9] = num_rows_per_core_group_2;

        tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/compute/"
            "rotary_embedding.cpp",
            core_group_2,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .compile_args = compute_kernel_args,
                .defines = compute_kernel_defines});
    }
    uint32_t cos_sin_offset = 0;
    uint32_t cos_sin_start_id = 0;
    if (token_idx.has_value()) {
        cos_sin_offset = token_idx.value() % TILE_HEIGHT * Wbytes;
        cos_sin_start_id = token_idx.value() / TILE_HEIGHT * Wt;
    }

    uint32_t g1_numcores = core_group_1.num_cores();

    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_rows_per_core = 0;
        if (i < g1_numcores) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else {
            num_rows_per_core = num_rows_per_core_group_2;
        }

        if (!token_idx.has_value()) {
            cos_sin_start_id = num_tiles_written % HtWt;
        }
        std::vector<uint32_t> reader_rt_args;
        if (in_sharded) {
            reader_rt_args = {
                cos_buffer->address(),
                sin_buffer->address(),
                num_rows_per_core,
                num_tiles_written / Wt % Ht,
                cos_sin_start_id};
        } else {
            reader_rt_args = {
                src_buffer->address(),
                cos_buffer->address(),
                sin_buffer->address(),
                num_rows_per_core,
                num_tiles_written,
                num_tiles_written / Wt % Ht,
                cos_sin_start_id};
        }
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);

        tt::tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {dst_buffer->address(), num_rows_per_core * Wt, num_tiles_written, cos_sin_offset, Wt, Wbytes});
        num_tiles_written += num_rows_per_core * Wt;
    }

    RotaryEmbeddingSharedVariables shared_variables{
        .unary_reader_kernel_id = unary_reader_kernel_id,
        .unary_writer_kernel_id = unary_writer_kernel_id,
        .cb_input = cb_input,
        .cb_output = cb_output,
        .cores = cores,
        .g1_numcores = g1_numcores,
        .num_rows_per_core_group_1 = num_rows_per_core_group_1,
        .num_rows_per_core_group_2 = num_rows_per_core_group_2,
        .Wbytes = Wbytes,
        .Wt = Wt,
        .HtWt = HtWt};

    return cached_program_t{std::move(program), std::move(shared_variables)};
}

void RotaryEmbeddingProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const RotaryEmbeddingParams& operation_attributes,
    const RotaryEmbeddingInputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt::constants;

    const auto& token_idx = operation_attributes.token_idx;

    auto* src_buffer = tensor_args.input.buffer();
    auto* cos_buffer = tensor_args.cos.buffer();
    auto* sin_buffer = tensor_args.sin.buffer();
    auto* dst_buffer = tensor_return_value.buffer();

    bool in_sharded = tensor_args.input.is_sharded();
    bool out_sharded = tensor_return_value.is_sharded();

    auto& program = cached_program.program;
    const auto& cores = cached_program.shared_variables.cores;
    const auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    const auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    const auto& cb_input = cached_program.shared_variables.cb_input;
    const auto& cb_output = cached_program.shared_variables.cb_output;
    const auto& g1_numcores = cached_program.shared_variables.g1_numcores;
    const auto& num_rows_per_core_group_1 = cached_program.shared_variables.num_rows_per_core_group_1;
    const auto& num_rows_per_core_group_2 = cached_program.shared_variables.num_rows_per_core_group_2;
    const auto& Wbytes = cached_program.shared_variables.Wbytes;
    const auto& Wt = cached_program.shared_variables.Wt;
    const auto& HtWt = cached_program.shared_variables.HtWt;

    if (in_sharded) {
        UpdateDynamicCircularBufferAddress(program, cb_input, *src_buffer);
    }

    if (out_sharded) {
        UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
    }

    uint32_t cos_sin_offset = 0;
    uint32_t cos_sin_start_id = 0;
    if (token_idx.has_value()) {
        cos_sin_offset = token_idx.value() % TILE_HEIGHT * Wbytes;
        cos_sin_start_id = token_idx.value() / TILE_HEIGHT * Wt;
    }

    for (uint32_t i = 0, num_tiles_written = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);
        uint32_t num_rows_per_core = 0;
        if (i < g1_numcores) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else {
            num_rows_per_core = num_rows_per_core_group_2;
        }
        if (!token_idx.has_value()) {
            cos_sin_start_id = num_tiles_written % HtWt;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            if (in_sharded) {
                runtime_args[0] = cos_buffer->address();
                runtime_args[1] = sin_buffer->address();
                runtime_args[4] = cos_sin_start_id;
            } else {
                runtime_args[0] = src_buffer->address();
                runtime_args[1] = cos_buffer->address();
                runtime_args[2] = sin_buffer->address();
                runtime_args[6] = cos_sin_start_id;
            }
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            runtime_args[3] = cos_sin_offset;
        }
        num_tiles_written += num_rows_per_core * Wt;
    }
}

}  // namespace ttnn::operations::experimental::transformer::rotary_embedding::program
