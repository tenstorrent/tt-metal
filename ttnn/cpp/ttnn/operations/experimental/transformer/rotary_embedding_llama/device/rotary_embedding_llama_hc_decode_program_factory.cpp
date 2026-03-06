// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_hc_decode_program_factory.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::experimental::prim {

RotaryEmbeddingLlamaHCDecode::cached_program_t RotaryEmbeddingLlamaHCDecode::create(
    const RotaryEmbeddingLlamaParams& operation_attributes,
    const RotaryEmbeddingLlamaInputs& tensor_args,
    tt::tt_metal::Tensor& output) {
    using namespace tt::constants;
    using namespace tt::tt_metal;
    using namespace tt;

    const auto& input = tensor_args.input_tensor;
    const auto& cos = tensor_args.cos_cache;
    const auto& sin = tensor_args.sin_cache;
    const auto& trans_mat = tensor_args.trans_mat;

    Program program{};

    const tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    const tt::DataFormat cos_cb_data_format = datatype_to_dataformat_converter(cos.dtype());
    const uint32_t cos_single_tile_size = tt::tile_size(cos_cb_data_format);
    const tt::DataFormat sin_cb_data_format = datatype_to_dataformat_converter(sin.dtype());
    const uint32_t sin_single_tile_size = tt::tile_size(sin_cb_data_format);
    const tt::DataFormat trans_mat_cb_data_format = datatype_to_dataformat_converter(trans_mat.dtype());
    const uint32_t trans_mat_single_tile_size = tt::tile_size(trans_mat_cb_data_format);
    const tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    // Input shape: [1, num_heads, batch_size, head_dim]  (padded to tile boundaries)
    const uint32_t num_heads = input.padded_shape()[1];
    const uint32_t batch_t = input.padded_shape()[2] / TILE_HEIGHT;    // ceil(batch/TILE_HEIGHT)
    const uint32_t head_dim_t = input.padded_shape()[3] / TILE_WIDTH;  // head_dim / TILE_WIDTH

    // Whether cos/sin have per-head frequencies (padded shape[1] == num_heads) or shared (shape[1]==1)
    const bool freq_per_head = cos.padded_shape()[1] == num_heads;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input.device()->arch(), operation_attributes.compute_kernel_config);

    // Parallelise over (head, batch_tile) pairs.
    // For simplicity, distribute whole heads across cores; each core handles
    // [head_start, head_end) × [0, batch_t).  If more cores are available than
    // heads, we further split the batch_t tiles within each head.
    //
    // total_work_units  = num_heads * batch_t
    // We use at most total_work_units cores.

    const uint32_t total_work_units = num_heads * batch_t;

    auto compute_with_storage_grid_size = input.device()->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const uint32_t max_cores = num_cores_x * num_cores_y;

    const uint32_t num_cores_used = std::min(total_work_units, max_cores);

    CoreRange all_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});
    const auto& cores = grid_to_cores(num_cores_used, num_cores_x, num_cores_y, /*row_major=*/true);

    auto* src_buffer = input.buffer();
    auto* cos_buffer = cos.buffer();
    auto* sin_buffer = sin.buffer();
    auto* trans_mat_buffer = trans_mat.buffer();
    auto* dst_buffer = output.buffer();

    uint32_t input_cb_index = CBIndex::c_0;
    uint32_t cos_cb_index = CBIndex::c_1;
    uint32_t sin_cb_index = CBIndex::c_2;
    uint32_t trans_mat_cb_index = CBIndex::c_3;
    uint32_t rotated_input_interm_cb_index = CBIndex::c_24;
    uint32_t cos_interm_cb_index = CBIndex::c_25;
    uint32_t sin_interm_cb_index = CBIndex::c_26;
    uint32_t output_cb_index = CBIndex::c_16;

    // CB sizes: I/O CBs are double-buffered to overlap NOC reads/writes with compute.
    // The reader can begin issuing NOC reads for iteration i+1 while compute
    // processes iteration i.  The output CB is doubled so the writer can drain
    // iteration i while compute fills iteration i+1.
    // Intermediate CBs are single-buffered: they are produced and consumed entirely
    // within one compute iteration and don't benefit from double-buffering.
    const uint32_t num_io_tiles = 2 * head_dim_t;
    const uint32_t num_interm_tiles = head_dim_t;

    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt_metal::CircularBufferConfig(num_io_tiles * input_single_tile_size, {{input_cb_index, input_cb_data_format}})
            .set_page_size(input_cb_index, input_single_tile_size));

    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt_metal::CircularBufferConfig(num_io_tiles * cos_single_tile_size, {{cos_cb_index, cos_cb_data_format}})
            .set_page_size(cos_cb_index, cos_single_tile_size));

    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt_metal::CircularBufferConfig(num_io_tiles * sin_single_tile_size, {{sin_cb_index, sin_cb_data_format}})
            .set_page_size(sin_cb_index, sin_single_tile_size));

    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt_metal::CircularBufferConfig(trans_mat_single_tile_size, {{trans_mat_cb_index, trans_mat_cb_data_format}})
            .set_page_size(trans_mat_cb_index, trans_mat_single_tile_size));

    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt_metal::CircularBufferConfig(
            num_interm_tiles * input_single_tile_size, {{rotated_input_interm_cb_index, input_cb_data_format}})
            .set_page_size(rotated_input_interm_cb_index, input_single_tile_size));

    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt_metal::CircularBufferConfig(
            num_interm_tiles * cos_single_tile_size, {{cos_interm_cb_index, cos_cb_data_format}})
            .set_page_size(cos_interm_cb_index, cos_single_tile_size));

    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt_metal::CircularBufferConfig(
            num_interm_tiles * sin_single_tile_size, {{sin_interm_cb_index, sin_cb_data_format}})
            .set_page_size(sin_interm_cb_index, sin_single_tile_size));

    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt_metal::CircularBufferConfig(
            num_io_tiles * output_single_tile_size, {{output_cb_index, output_cb_data_format}})
            .set_page_size(output_cb_index, output_single_tile_size));

    // ---- Reader kernel ----
    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)input_cb_index,
        (uint32_t)cos_cb_index,
        (uint32_t)sin_cb_index,
        (uint32_t)trans_mat_cb_index,
        (uint32_t)batch_t,
        (uint32_t)head_dim_t,
        (uint32_t)freq_per_head,
    };
    tt::tt_metal::TensorAccessorArgs(src_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(cos_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(sin_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(trans_mat_buffer).append_to(reader_compile_time_args);

    tt_metal::KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/dataflow/"
        "reader_rotary_embedding_llama_hc_decode.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // ---- Writer kernel ----
    std::vector<uint32_t> writer_compile_time_args = {
        (uint32_t)output_cb_index,
        (uint32_t)batch_t,
        (uint32_t)head_dim_t,
    };
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(writer_compile_time_args);

    tt_metal::KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/dataflow/"
        "writer_rotary_embedding_llama_hc_decode.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // ---- Compute kernel ----
    std::vector<uint32_t> compute_kernel_args = {
        (uint32_t)input_cb_index,
        (uint32_t)cos_cb_index,
        (uint32_t)sin_cb_index,
        (uint32_t)trans_mat_cb_index,
        (uint32_t)rotated_input_interm_cb_index,
        (uint32_t)cos_interm_cb_index,
        (uint32_t)sin_interm_cb_index,
        (uint32_t)output_cb_index,
        (uint32_t)head_dim_t,
    };

    tt_metal::KernelHandle compute_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/"
        "rotary_embedding_llama_hc_decode.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_kernel_args});

    // ---- Assign work ranges to cores ----
    // Work units are (head, batch_tile) pairs, indexed as flat = head * batch_t + bt.
    // We distribute contiguous flat ranges to cores so that each core's range is
    // head-row-aligned: it always starts at bt=0 of some head and ends at bt=batch_t-1
    // of some (possibly the same) head.  This is guaranteed when each core receives an
    // integer number of full head rows.
    //
    // When total_work_units % num_cores_used != 0, some cores receive one extra full
    // head row.  Because total_work_units = num_heads * batch_t and we never split a
    // head row across two cores, the runtime args express the range as
    //   [head_start, head_end) x [0, batch_t)
    // which is always correct.
    //
    // NOTE: The kernel already uses two nested loops (head, batch_tile), so passing
    // head_start/head_end with bt_start=0, bt_end=batch_t covers the full head(s)
    // assigned to this core.

    // To ensure head-aligned distribution we distribute at the *head* granularity
    // when batch_t == 1 (the common decode case), or at the work-unit granularity but
    // always rounded to full head rows.
    //
    // Algorithm: assign ceil(total_work_units / num_cores_used) work units per core,
    // but round up to the nearest multiple of batch_t so each core covers complete
    // head rows only.  When multiple full head rows don't divide evenly, use the
    // standard (base, base+1) split at the head granularity.

    const uint32_t heads_per_core_base = num_heads / num_cores_used;
    const uint32_t heads_remainder = num_heads % num_cores_used;
    // If num_cores_used > num_heads, some cores handle partial batch_t slices of a single head.
    const bool more_cores_than_heads = (num_cores_used > num_heads);

    std::vector<uint32_t> default_reader_args = {
        src_buffer->address(), cos_buffer->address(), sin_buffer->address(), trans_mat_buffer->address(), 0, 0, 0, 0};
    std::vector<uint32_t> default_writer_args = {dst_buffer->address(), 0, 0, 0, 0};
    std::vector<uint32_t> default_compute_args = {0, 0, 0, 0};

    std::vector<std::vector<uint32_t>> reader_args(cores.size(), default_reader_args);
    std::vector<std::vector<uint32_t>> writer_args(cores.size(), default_writer_args);
    std::vector<std::vector<uint32_t>> compute_args(cores.size(), default_compute_args);

    uint32_t num_active_cores = 0;

    if (!more_cores_than_heads) {
        // Distribute whole heads across cores; each core handles one or more complete heads.
        uint32_t head_idx = 0;
        for (uint32_t ci = 0; ci < num_cores_used; ++ci) {
            uint32_t n_heads_this_core = heads_per_core_base + (ci < heads_remainder ? 1 : 0);
            if (n_heads_this_core == 0) {
                break;
            }

            const uint32_t head_start = head_idx;
            const uint32_t head_end = head_idx + n_heads_this_core;

            auto& ra = reader_args[ci];
            ra[4] = head_start;
            ra[5] = head_end;
            ra[6] = 0;
            ra[7] = batch_t;

            auto& wa = writer_args[ci];
            wa[1] = head_start;
            wa[2] = head_end;
            wa[3] = 0;
            wa[4] = batch_t;

            auto& ca = compute_args[ci];
            ca[0] = head_start;
            ca[1] = head_end;
            ca[2] = 0;
            ca[3] = batch_t;

            head_idx += n_heads_this_core;
            num_active_cores = ci + 1;
        }
    } else {
        // More cores than heads: further subdivide batch_t within each head.
        // Each head gets cores_per_head = num_cores_used / num_heads cores.
        // Assign batch_t tiles to those cores.
        const uint32_t cores_per_head_base = num_cores_used / num_heads;
        const uint32_t cores_per_head_rem = num_cores_used % num_heads;

        uint32_t ci = 0;
        for (uint32_t h = 0; h < num_heads; ++h) {
            const uint32_t n_cores_this_head = cores_per_head_base + (h < cores_per_head_rem ? 1 : 0);
            const uint32_t bt_per_core_base = batch_t / n_cores_this_head;
            const uint32_t bt_per_core_rem = batch_t % n_cores_this_head;
            uint32_t bt_idx = 0;

            for (uint32_t sub = 0; sub < n_cores_this_head; ++sub, ++ci) {
                const uint32_t n_bt_this_core = bt_per_core_base + (sub < bt_per_core_rem ? 1 : 0);
                if (n_bt_this_core == 0) {
                    continue;
                }

                const uint32_t bt_start = bt_idx;
                const uint32_t bt_end = bt_idx + n_bt_this_core;

                auto& ra = reader_args[ci];
                ra[4] = h;
                ra[5] = h + 1;
                ra[6] = bt_start;
                ra[7] = bt_end;

                auto& wa = writer_args[ci];
                wa[1] = h;
                wa[2] = h + 1;
                wa[3] = bt_start;
                wa[4] = bt_end;

                auto& ca = compute_args[ci];
                ca[0] = h;
                ca[1] = h + 1;
                ca[2] = bt_start;
                ca[3] = bt_end;

                bt_idx += n_bt_this_core;
                num_active_cores = ci + 1;
            }
        }
    }

    tt_metal::SetRuntimeArgs(program, reader_kernel_id, cores, reader_args);
    tt_metal::SetRuntimeArgs(program, writer_kernel_id, cores, writer_args);
    tt_metal::SetRuntimeArgs(program, compute_kernel_id, cores, compute_args);

    RotaryEmbeddingLlamaHCDecode::shared_variables_t shared_variables;
    shared_variables.reader_kernel_id = reader_kernel_id;
    shared_variables.writer_kernel_id = writer_kernel_id;
    shared_variables.compute_kernel_id = compute_kernel_id;
    shared_variables.cores = cores;
    shared_variables.num_active_cores = num_active_cores;

    return {std::move(program), std::move(shared_variables)};
}

void RotaryEmbeddingLlamaHCDecode::override_runtime_arguments(
    cached_program_t& cached_program,
    const RotaryEmbeddingLlamaParams& /*operation_attributes*/,
    const RotaryEmbeddingLlamaInputs& tensor_args,
    tt::tt_metal::Tensor& output) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    auto* src_buffer = tensor_args.input_tensor.buffer();
    auto* cos_buffer = tensor_args.cos_cache.buffer();
    auto* sin_buffer = tensor_args.sin_cache.buffer();
    auto* trans_mat_buffer = tensor_args.trans_mat.buffer();
    auto* dst_buffer = output.buffer();

    auto& cached_reader_args = GetRuntimeArgs(program, shared_variables.reader_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, shared_variables.writer_kernel_id);

    const auto& cores = shared_variables.cores;
    const uint32_t num_active_cores = shared_variables.num_active_cores;

    for (uint32_t i = 0; i < num_active_cores; ++i) {
        const CoreCoord& core = cores.at(i);
        {
            auto& rt = cached_reader_args.at(core.x).at(core.y);
            rt[0] = src_buffer->address();
            rt[1] = cos_buffer->address();
            rt[2] = sin_buffer->address();
            rt[3] = trans_mat_buffer->address();
        }
        {
            auto& rt = cached_writer_args.at(core.x).at(core.y);
            rt[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::experimental::prim
