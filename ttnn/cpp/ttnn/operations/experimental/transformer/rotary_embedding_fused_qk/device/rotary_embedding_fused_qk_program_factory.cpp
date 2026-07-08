// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/transformer/rotary_embedding_fused_qk/device/rotary_embedding_fused_qk_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::experimental::prim {

using namespace tt::constants;

namespace {
// Reuse the proven, unchanged rotary_embedding multi-tile kernels (rotate_half / GPT-J
// convention) so numerics are byte-identical to two separate rotary_embedding calls.
constexpr const char* kReaderKernel =
    "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/dataflow/"
    "reader_rotary_embedding_interleaved_start_id.cpp";
constexpr const char* kWriterKernel =
    "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/dataflow/"
    "writer_rotary_embedding_interleaved_start_id.cpp";
constexpr const char* kComputeKernel =
    "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/compute/"
    "rotary_embedding.cpp";

struct RowAssign {
    tt::tt_metal::Buffer* src;
    tt::tt_metal::Buffer* dst;
    uint32_t within_row;  // row index within its own (src/dst) tensor
};
}  // namespace

RotaryEmbeddingFusedQKProgramFactory::cached_program_t RotaryEmbeddingFusedQKProgramFactory::create(
    const RotaryEmbeddingFusedQKParams& operation_attributes,
    const RotaryEmbeddingFusedQKInputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt::tt_metal;

    const auto& q = tensor_args.q;
    const auto& k = tensor_args.k;
    const auto& cos = tensor_args.cos;
    const auto& sin = tensor_args.sin;
    auto& q_out = std::get<0>(tensor_return_value);
    auto& k_out = std::get<1>(tensor_return_value);

    Program program{};

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(q.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat cos_cb_data_format = datatype_to_dataformat_converter(cos.dtype());
    uint32_t cos_single_tile_size = tt::tile_size(cos_cb_data_format);
    tt::DataFormat sin_cb_data_format = datatype_to_dataformat_converter(sin.dtype());
    uint32_t sin_single_tile_size = tt::tile_size(sin_cb_data_format);
    tt::DataFormat scalar_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t scalar_single_tile_size = tt::tile_size(scalar_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(q_out.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    uint32_t Ht = q.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t Wt = q.padded_shape()[-1] / TILE_WIDTH;
    uint32_t half_Wt = Wt / 2;
    uint32_t HtWt = Ht * Wt;

    uint32_t num_q_rows = q.physical_volume() / q.padded_shape()[-1] / TILE_HEIGHT;
    uint32_t num_k_rows = k.physical_volume() / k.padded_shape()[-1] / TILE_HEIGHT;
    uint32_t total_rows = num_q_rows + num_k_rows;

    IDevice* device = q.device();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    TT_FATAL(
        total_rows <= num_cores_x * num_cores_y,
        "fused-qk RoPE: total rows ({}) exceeds available cores ({}); shape not supported",
        total_rows,
        num_cores_x * num_cores_y);

    CoreRangeSet all_cores = num_cores_to_corerangeset(total_rows, compute_with_storage_grid_size, true);
    const auto& cores = grid_to_cores(total_rows, num_cores_x, num_cores_y, true);

    // Circular buffers (one row == Wt tiles in flight; mirror rotary_embedding multi-tile path).
    uint32_t input_cb_index = tt::CBIndex::c_0;
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(2 * Wt * input_single_tile_size, {{input_cb_index, input_cb_data_format}})
            .set_page_size(input_cb_index, input_single_tile_size));

    uint32_t rotated_input_cb_index = tt::CBIndex::c_1;
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(2 * Wt * input_single_tile_size, {{rotated_input_cb_index, input_cb_data_format}})
            .set_page_size(rotated_input_cb_index, input_single_tile_size));

    uint32_t cos_cb_index = tt::CBIndex::c_2;
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(2 * Wt * cos_single_tile_size, {{cos_cb_index, cos_cb_data_format}})
            .set_page_size(cos_cb_index, cos_single_tile_size));

    uint32_t sin_cb_index = tt::CBIndex::c_3;
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(2 * Wt * sin_single_tile_size, {{sin_cb_index, sin_cb_data_format}})
            .set_page_size(sin_cb_index, sin_single_tile_size));

    uint32_t src_scalar_cb_index = tt::CBIndex::c_4;
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(scalar_single_tile_size, {{src_scalar_cb_index, scalar_cb_data_format}})
            .set_page_size(src_scalar_cb_index, scalar_single_tile_size));

    uint32_t rotated_input_interm_cb_index = tt::CBIndex::c_24;
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(input_single_tile_size, {{rotated_input_interm_cb_index, input_cb_data_format}})
            .set_page_size(rotated_input_interm_cb_index, input_single_tile_size));

    uint32_t cos_interm_cb_index = tt::CBIndex::c_25;
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(cos_single_tile_size, {{cos_interm_cb_index, cos_cb_data_format}})
            .set_page_size(cos_interm_cb_index, cos_single_tile_size));

    uint32_t sin_interm_cb_index = tt::CBIndex::c_26;
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(sin_single_tile_size, {{sin_interm_cb_index, sin_cb_data_format}})
            .set_page_size(sin_interm_cb_index, sin_single_tile_size));

    uint32_t output_cb_index = tt::CBIndex::c_16;
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(2 * Wt * output_single_tile_size, {{output_cb_index, output_cb_data_format}})
            .set_page_size(output_cb_index, output_single_tile_size));

    const uint16_t bfloat16_scalar = std::bit_cast<uint16_t>(bfloat16(-1.0f));

    // q and k share dtype + buffer type (validated), so the same reader/writer compile-time
    // TensorAccessor args are valid for both; only the runtime buffer address differs per core.
    std::vector<uint32_t> reader_compile_time_args = {
        input_cb_index,
        rotated_input_cb_index,
        cos_cb_index,
        sin_cb_index,
        src_scalar_cb_index,
        (uint32_t)bfloat16_scalar,
        Ht,
        Wt,
        HtWt,
        half_Wt,
    };
    TensorAccessorArgs(q.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(cos.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(sin.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {output_cb_index};
    TensorAccessorArgs(q_out.buffer()).append_to(writer_compile_time_args);

    KernelHandle reader_kernel_id =
        CreateKernel(program, kReaderKernel, all_cores, ReaderDataMovementConfig(reader_compile_time_args));
    KernelHandle writer_kernel_id =
        CreateKernel(program, kWriterKernel, all_cores, WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_kernel_args = {
        input_cb_index,
        rotated_input_cb_index,
        cos_cb_index,
        sin_cb_index,
        src_scalar_cb_index,
        rotated_input_interm_cb_index,
        cos_interm_cb_index,
        sin_interm_cb_index,
        output_cb_index,
        1u,  // num_rows_per_core (1 row per core)
        Wt,
        half_Wt};
    CreateKernel(
        program,
        kComputeKernel,
        all_cores,
        ComputeConfig{
            .math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_kernel_args});

    // Build per-row assignments: q rows first (read/write q buffers), then k rows.
    std::vector<RowAssign> assigns;
    assigns.reserve(total_rows);
    for (uint32_t r = 0; r < num_q_rows; ++r) {
        assigns.push_back({q.buffer(), q_out.buffer(), r});
    }
    for (uint32_t r = 0; r < num_k_rows; ++r) {
        assigns.push_back({k.buffer(), k_out.buffer(), r});
    }

    auto* cos_buffer = cos.buffer();
    auto* sin_buffer = sin.buffer();
    for (uint32_t i = 0; i < total_rows; ++i) {
        const CoreCoord& core = cores.at(i);
        const RowAssign& a = assigns.at(i);
        uint32_t start_id = a.within_row * Wt;  // tile offset into this row's buffer
        uint32_t ht = a.within_row % Ht;        // seq-tile position
        uint32_t cos_sin_start_id = ht * Wt;

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {a.src->address(), cos_buffer->address(), sin_buffer->address(), 1u, start_id, ht, cos_sin_start_id});
        SetRuntimeArgs(program, writer_kernel_id, core, {a.dst->address(), Wt, start_id});
    }

    RotaryEmbeddingFusedQKSharedVariables shared_variables{
        .reader_kernel_id = reader_kernel_id,
        .writer_kernel_id = writer_kernel_id,
        .cores = cores,
        .Wt = Wt,
        .num_q_rows = num_q_rows,
        .num_k_rows = num_k_rows};

    return cached_program_t{std::move(program), std::move(shared_variables)};
}

void RotaryEmbeddingFusedQKProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const RotaryEmbeddingFusedQKParams& /*operation_attributes*/,
    const RotaryEmbeddingFusedQKInputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    const auto& sv = cached_program.shared_variables;
    const auto& cores = sv.cores;

    auto* q_in = tensor_args.q.buffer();
    auto* k_in = tensor_args.k.buffer();
    auto* cos_buffer = tensor_args.cos.buffer();
    auto* sin_buffer = tensor_args.sin.buffer();
    auto* q_out = std::get<0>(tensor_return_value).buffer();
    auto* k_out = std::get<1>(tensor_return_value).buffer();

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);
        bool is_q = i < sv.num_q_rows;
        auto* src = is_q ? q_in : k_in;
        auto* dst = is_q ? q_out : k_out;
        {
            auto& ra = GetRuntimeArgs(program, sv.reader_kernel_id, core);
            ra[0] = src->address();
            ra[1] = cos_buffer->address();
            ra[2] = sin_buffer->address();
        }
        {
            auto& wa = GetRuntimeArgs(program, sv.writer_kernel_id, core);
            wa[0] = dst->address();
        }
    }
}

}  // namespace ttnn::experimental::prim
