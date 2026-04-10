// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Program factory for TurboQuant SDPA decode.
// Creates CBs, kernels, and sets compile/runtime args for the fused
// BFP4-dequantize + SDPA decode kernel.

#include "sdpa_tq_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <cstring>

namespace ttnn::operations::experimental::turbo_quant {

namespace {
uint32_t sdpa_float_to_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return bits;
}
}  // namespace

SDPATQDeviceOperation::MultiCore::cached_program_t SDPATQDeviceOperation::MultiCore::create(
    const operation_attributes_t& attrs, const tensor_args_t& args, tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    Program program{};

    const auto& q = args.q;
    const auto& k_idx = args.k_indices;
    const auto& k_norms = args.k_norms;
    const auto& v_idx = args.v_indices;
    const auto& v_norms = args.v_norms;
    const auto& page_table = args.page_table;

    IDevice* device = q.device();
    auto grid = device->compute_with_storage_grid_size();

    // ── Dimensions ──
    const uint32_t B = q.padded_shape()[0];
    const uint32_t NQH = q.padded_shape()[1];
    const uint32_t NKH = k_idx.padded_shape()[1];
    const uint32_t DHt = q.padded_shape()[3] / tt::constants::TILE_WIDTH;
    const uint32_t vDHt = v_idx.padded_shape()[3] / tt::constants::TILE_WIDTH;
    const uint32_t Skt = k_idx.padded_shape()[2] / tt::constants::TILE_HEIGHT;

    // Chunk sizes (hardcoded for decode: Sq=1 tile, Sk=4 tiles = 128 positions)
    const uint32_t Sq_chunk_t = 1;
    const uint32_t Sk_chunk_t = std::min(Skt, (uint32_t)4);
    const uint32_t k_num_chunks = (Skt + Sk_chunk_t - 1) / Sk_chunk_t;

    const uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    const uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;
    const uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    const uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    const uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    // ── Tile sizes ──
    auto q_df = datatype_to_dataformat_converter(q.dtype());
    auto k_idx_df = datatype_to_dataformat_converter(k_idx.dtype());
    auto k_norms_df = datatype_to_dataformat_converter(k_norms.dtype());
    auto bf16_df = tt::DataFormat::Float16_b;
    auto im_df = tt::DataFormat::Float16_b;

    uint32_t q_tile_size = tile_size(q_df);
    uint32_t k_idx_tile_size = tile_size(k_idx_df);
    uint32_t k_norms_tile_size = tile_size(k_norms_df);
    uint32_t bf16_tile_size = tile_size(bf16_df);
    uint32_t im_tile_size = tile_size(im_df);

    // ── Work distribution: parallelize across (batch, Q heads) ──
    uint32_t total_work = B * NQH;
    uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_1, work_per_core_2] =
        split_work_to_cores(grid, total_work);

    // ── Circular Buffers ──
    // Standard SDPA CBs
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(q_chunk_tiles * q_tile_size, {{CBIndex::c_0, q_df}})
            .set_page_size(CBIndex::c_0, q_tile_size));

    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(k_chunk_tiles * bf16_tile_size, {{CBIndex::c_1, bf16_df}})
            .set_page_size(CBIndex::c_1, bf16_tile_size));

    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(v_chunk_tiles * bf16_tile_size, {{CBIndex::c_2, bf16_df}})
            .set_page_size(CBIndex::c_2, bf16_tile_size));

    // Scale + column identity
    uint32_t scalar_tile_size = tile_size(tt::DataFormat::Float16_b);
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(scalar_tile_size, {{CBIndex::c_5, tt::DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_5, scalar_tile_size));

    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(scalar_tile_size, {{CBIndex::c_7, tt::DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_7, scalar_tile_size));

    // Page table
    uint32_t page_table_stick_size = page_table.padded_shape()[-1] * sizeof(uint32_t);
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(page_table_stick_size, {{CBIndex::c_6, tt::DataFormat::Int32}})
            .set_page_size(CBIndex::c_6, page_table_stick_size));

    // TurboQuant raw index + norm CBs
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(k_chunk_tiles * k_idx_tile_size, {{CBIndex::c_10, k_idx_df}})
            .set_page_size(CBIndex::c_10, k_idx_tile_size));

    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(Sk_chunk_t * k_norms_tile_size, {{CBIndex::c_11, k_norms_df}})
            .set_page_size(CBIndex::c_11, k_norms_tile_size));

    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(v_chunk_tiles * k_idx_tile_size, {{CBIndex::c_12, k_idx_df}})
            .set_page_size(CBIndex::c_12, k_idx_tile_size));

    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(Sk_chunk_t * k_norms_tile_size, {{CBIndex::c_13, k_norms_df}})
            .set_page_size(CBIndex::c_13, k_norms_tile_size));

    // Dequantize temp (1 tile BF16)
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(bf16_tile_size, {{CBIndex::c_14, bf16_df}}).set_page_size(CBIndex::c_14, bf16_tile_size));

    // SDPA intermediates
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(qk_chunk_tiles * im_tile_size, {{CBIndex::c_24, im_df}})
            .set_page_size(CBIndex::c_24, im_tile_size));

    // Output ping-pong (c_25, c_26)
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(out_chunk_tiles * im_tile_size, {{CBIndex::c_25, im_df}})
            .set_page_size(CBIndex::c_25, im_tile_size));
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(out_chunk_tiles * im_tile_size, {{CBIndex::c_26, im_df}})
            .set_page_size(CBIndex::c_26, im_tile_size));

    // Max/sum ping-pong (c_27-c_30)
    for (auto cb : {CBIndex::c_27, CBIndex::c_28, CBIndex::c_29, CBIndex::c_30}) {
        CreateCircularBuffer(
            program,
            all_cores,
            CircularBufferConfig(Sq_chunk_t * im_tile_size, {{cb, im_df}}).set_page_size(cb, im_tile_size));
    }

    // exp_max_diff (c_31)
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(Sq_chunk_t * im_tile_size, {{CBIndex::c_31, im_df}})
            .set_page_size(CBIndex::c_31, im_tile_size));

    // Output (c_16)
    auto out_df = datatype_to_dataformat_converter(output.dtype());
    uint32_t out_tile_size = tile_size(out_df);
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(out_chunk_tiles * out_tile_size, {{CBIndex::c_16, out_df}})
            .set_page_size(CBIndex::c_16, out_tile_size));

    // ── Compute kernel ──
    // Matmul config for QK and out
    uint32_t qk_in0_block_w = DHt;
    uint32_t qk_subblock_w = std::min(Sk_chunk_t, (uint32_t)4);
    uint32_t qk_subblock_h = 1;
    uint32_t qk_in0_num_subblocks = 1;
    uint32_t qk_in1_num_subblocks = (Sk_chunk_t + qk_subblock_w - 1) / qk_subblock_w;
    uint32_t qk_num_blocks = 1;

    uint32_t out_in0_block_w = Sk_chunk_t;
    uint32_t out_subblock_w = std::min(vDHt, (uint32_t)4);
    uint32_t out_subblock_h = 1;
    uint32_t out_in0_num_subblocks = 1;
    uint32_t out_in1_num_subblocks = (vDHt + out_subblock_w - 1) / out_subblock_w;
    uint32_t out_num_blocks = 1;

    uint32_t num_levels = static_cast<uint32_t>(attrs.centroids.size());

    std::vector<uint32_t> compute_ct_args = {
        B,
        NQH,
        NKH,
        Skt,
        DHt,
        vDHt,
        Sq_chunk_t,
        1 /*q_num_chunks*/,
        Sk_chunk_t,
        k_num_chunks,
        qk_in0_block_w,
        qk_subblock_w,
        qk_subblock_h,
        qk_in0_num_subblocks,
        qk_in1_num_subblocks,
        qk_num_blocks,
        out_in0_block_w,
        out_subblock_w,
        out_subblock_h,
        out_in0_num_subblocks,
        out_in1_num_subblocks,
        out_num_blocks,
        num_cores,
        0,                                // is_causal = false
        0,                                // use_provided_mask = false
        0,                                // use_padded_mask = false
        1,                                // is_chunked = true (paged)
        sdpa_float_to_bits(attrs.scale),  // scale
        0,                                // sliding_window_size
        0,                                // use_attention_sink
        0,                                // use_streaming_compute
        Skt,                              // valid_Skt
        1,                                // uniform_dataformat
        // TQ args (index 33+)
        num_levels,
    };
    // Append centroid bit-patterns
    for (float c : attrs.centroids) {
        compute_ct_args.push_back(sdpa_float_to_bits(c));
    }

    KernelHandle compute_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/turbo_quant/sdpa/kernels/compute/sdpa_tq_decode.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .math_approx_mode = true,
            .compile_args = compute_ct_args,
            .defines = {{"SDPA_TQ_DECODE", "1"}}});

    // ── Reader kernel ──
    std::vector<uint32_t> reader_ct_args = {
        B,
        NQH,
        NKH,
        1 /*Sqt*/,
        Skt,
        DHt,
        vDHt,
        Sq_chunk_t,
        Sk_chunk_t,
        k_num_chunks,
        num_cores,
        Sk_chunk_t,  // block_size_t (simplified: block = chunk)
        page_table_stick_size,
    };
    TensorAccessorArgs(*q.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*k_idx.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*k_norms.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*v_idx.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*v_norms.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*page_table.buffer()).append_to(reader_ct_args);

    KernelHandle reader_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/turbo_quant/sdpa/kernels/dataflow/reader_tq_decode.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_ct_args));

    // ── Writer kernel (custom simple writer) ──
    std::vector<uint32_t> writer_ct_args = {
        B,
        NQH,
        Sq_chunk_t,
        vDHt,
        num_cores,
        sdpa_float_to_bits(attrs.scale),  // scale for identity tile
    };
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct_args);

    KernelHandle writer_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/turbo_quant/sdpa/kernels/dataflow/writer_tq_decode.cpp",
        all_cores,
        WriterDataMovementConfig(writer_ct_args));

    // ── Runtime args ──
    for (uint32_t i = 0, work_done = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t work_this_core = core_group_1.contains(core) ? work_per_core_1 : work_per_core_2;

        // Map work items to (batch, head) pairs
        uint32_t batch_start = work_done / NQH;
        uint32_t head_start = work_done % NQH;
        uint32_t batch_end = (work_done + work_this_core - 1) / NQH + 1;
        uint32_t head_end = NQH;  // simplified: process all heads per batch

        SetRuntimeArgs(
            program,
            reader_kernel,
            core,
            {
                q.buffer()->address(),
                k_idx.buffer()->address(),
                k_norms.buffer()->address(),
                v_idx.buffer()->address(),
                v_norms.buffer()->address(),
                page_table.buffer()->address(),
                i,  // core_id
                batch_start,
                std::min(batch_end, B),
                head_start,
                head_end,
            });

        SetRuntimeArgs(
            program,
            compute_kernel,
            core,
            {
                i,  // core_id
                batch_start,
                std::min(batch_end, B),
                head_start,
                head_end,
                (uint32_t)0,  // local_q_start
                (uint32_t)1,  // local_q_end
                (uint32_t)1,  // num_phases
                (uint32_t)0,  // use_chunk_start_idx_tensor
                (uint32_t)0,  // chunked_q_chunk_offset
            });

        SetRuntimeArgs(
            program,
            writer_kernel,
            core,
            {
                output.buffer()->address(),
                i,
                batch_start,
                std::min(batch_end, B),
                head_start,
                head_end,
                (uint32_t)0,
                (uint32_t)1,
            });

        work_done += work_this_core;
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel,
         .compute_kernel_id = compute_kernel,
         .writer_kernel_id = writer_kernel,
         .num_cores = num_cores,
         .num_cores_y = num_cores_y}};
}

void SDPATQDeviceOperation::MultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*attrs*/,
    const tensor_args_t& args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    auto num_cores = cached_program.shared_variables.num_cores;
    auto num_cores_y = cached_program.shared_variables.num_cores_y;

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        auto& reader_args = GetRuntimeArgs(program, cached_program.shared_variables.reader_kernel_id, core);
        reader_args[0] = args.q.buffer()->address();
        reader_args[1] = args.k_indices.buffer()->address();
        reader_args[2] = args.k_norms.buffer()->address();
        reader_args[3] = args.v_indices.buffer()->address();
        reader_args[4] = args.v_norms.buffer()->address();
        reader_args[5] = args.page_table.buffer()->address();

        auto& writer_args = GetRuntimeArgs(program, cached_program.shared_variables.writer_kernel_id, core);
        writer_args[0] = output.buffer()->address();
    }
}

}  // namespace ttnn::operations::experimental::turbo_quant
