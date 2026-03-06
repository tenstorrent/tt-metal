// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_hc_decode_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

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

    // Whether trans_mat / cos+sin are pre-loaded into per-core L1 shards.
    // When sharded, CBs are globally-allocated — no NOC reads needed in the reader.
    const bool trans_mat_sharded = trans_mat.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;
    const bool cos_sin_sharded = cos.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input.device()->arch(), operation_attributes.compute_kernel_config);

    // Parallelise over (head, batch_tile) pairs.
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

    // ---- Decide reload_cos_sin for the DRAM (non-sharded) cos/sin path ----
    // When cos/sin are sharded this decision is irrelevant — the sharded path
    // always signals one Wt-tile batch per (head, bt) iteration with zero NOC reads.
    const bool more_cores_than_heads = (num_cores_used > num_heads);
    const uint32_t max_batch_t_per_core =
        !more_cores_than_heads ? batch_t : (batch_t + (num_cores_used / num_heads) - 1) / (num_cores_used / num_heads);

    // Fixed CB budget for non-cos/sin CBs (bytes):
    //   input (2x) + output (2x) + trans_mat (1x) + 3 intermediates (1x each)
    const uint32_t fixed_cb_bytes = (7 * head_dim_t + 1) * input_single_tile_size;

    // Budget for cos/sin cache: conservative 1060 KB out of 1464 KB Wormhole L1.
    constexpr uint32_t l1_cb_budget_bytes = 1060 * 1024;
    const uint32_t cs_budget_bytes = (l1_cb_budget_bytes > fixed_cb_bytes) ? (l1_cb_budget_bytes - fixed_cb_bytes) : 0;
    const uint32_t cs_cache_bytes = max_batch_t_per_core * head_dim_t * (cos_single_tile_size + sin_single_tile_size);

    const bool more_heads_than_cores = !more_cores_than_heads && (num_heads > num_cores_used);
    const bool reload_cos_sin =
        cos_sin_sharded || freq_per_head || !more_heads_than_cores || (cs_cache_bytes > cs_budget_bytes);

    log_debug(
        tt::LogOp,
        "RotaryEmbeddingLlamaHCDecode: trans_mat_sharded={}, cos_sin_sharded={}, freq_per_head={}, "
        "more_heads_than_cores={}, max_batch_t_per_core={}, cs_cache_bytes={}, cs_budget_bytes={}, "
        "reload_cos_sin={}",
        trans_mat_sharded,
        cos_sin_sharded,
        freq_per_head,
        more_heads_than_cores,
        max_batch_t_per_core,
        cs_cache_bytes,
        cs_budget_bytes,
        reload_cos_sin);

    // ---- CB sizes ----
    // Input/output: double-buffered to overlap NOC and compute.
    // trans_mat: 1 tile; globally-allocated when sharded (already in L1), otherwise normal.
    // cos/sin: globally-allocated from shard when sharded; otherwise sized for caching or double-buffering.
    // Intermediates: single-buffered (produced and consumed within one compute iteration).
    const uint32_t num_io_tiles = 2 * head_dim_t;
    const uint32_t num_interm_tiles = head_dim_t;
    // cos/sin CB size in the DRAM path: cache-size when caching, double-buffer otherwise.
    const uint32_t num_cs_tiles_dram = reload_cos_sin ? num_io_tiles : (max_batch_t_per_core * head_dim_t);
    // cos/sin CB size in the sharded path: exactly one shard row (head_dim_t tiles).
    const uint32_t num_cs_tiles = cos_sin_sharded ? head_dim_t : num_cs_tiles_dram;

    tt_metal::CreateCircularBuffer(
        program,
        all_cores,
        tt_metal::CircularBufferConfig(num_io_tiles * input_single_tile_size, {{input_cb_index, input_cb_data_format}})
            .set_page_size(input_cb_index, input_single_tile_size));

    // cos/sin CBs: globally-allocated when sharded, otherwise plain.
    std::optional<CBHandle> cb_cos_handle;
    std::optional<CBHandle> cb_sin_handle;
    if (cos_sin_sharded) {
        auto cos_cb_cfg =
            tt_metal::CircularBufferConfig(num_cs_tiles * cos_single_tile_size, {{cos_cb_index, cos_cb_data_format}})
                .set_page_size(cos_cb_index, cos_single_tile_size)
                .set_globally_allocated_address(*cos_buffer);
        cb_cos_handle = tt_metal::CreateCircularBuffer(program, all_cores, cos_cb_cfg);

        auto sin_cb_cfg =
            tt_metal::CircularBufferConfig(num_cs_tiles * sin_single_tile_size, {{sin_cb_index, sin_cb_data_format}})
                .set_page_size(sin_cb_index, sin_single_tile_size)
                .set_globally_allocated_address(*sin_buffer);
        cb_sin_handle = tt_metal::CreateCircularBuffer(program, all_cores, sin_cb_cfg);
    } else {
        tt_metal::CreateCircularBuffer(
            program,
            all_cores,
            tt_metal::CircularBufferConfig(num_cs_tiles * cos_single_tile_size, {{cos_cb_index, cos_cb_data_format}})
                .set_page_size(cos_cb_index, cos_single_tile_size));
        tt_metal::CreateCircularBuffer(
            program,
            all_cores,
            tt_metal::CircularBufferConfig(num_cs_tiles * sin_single_tile_size, {{sin_cb_index, sin_cb_data_format}})
                .set_page_size(sin_cb_index, sin_single_tile_size));
    }

    // trans_mat CB: globally-allocated when sharded, otherwise plain.
    std::optional<CBHandle> cb_trans_mat_handle;
    if (trans_mat_sharded) {
        auto tm_cb_cfg =
            tt_metal::CircularBufferConfig(trans_mat_single_tile_size, {{trans_mat_cb_index, trans_mat_cb_data_format}})
                .set_page_size(trans_mat_cb_index, trans_mat_single_tile_size)
                .set_globally_allocated_address(*trans_mat_buffer);
        cb_trans_mat_handle = tt_metal::CreateCircularBuffer(program, all_cores, tm_cb_cfg);
    } else {
        tt_metal::CreateCircularBuffer(
            program,
            all_cores,
            tt_metal::CircularBufferConfig(trans_mat_single_tile_size, {{trans_mat_cb_index, trans_mat_cb_data_format}})
                .set_page_size(trans_mat_cb_index, trans_mat_single_tile_size));
    }

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
    // When a tensor is sharded (globally-allocated CB), no TensorAccessorArgs are
    // needed for it — the reader does not issue NOC reads for that tensor.
    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)input_cb_index,
        (uint32_t)cos_cb_index,
        (uint32_t)sin_cb_index,
        (uint32_t)trans_mat_cb_index,
        (uint32_t)batch_t,
        (uint32_t)head_dim_t,
        (uint32_t)freq_per_head,
        (uint32_t)reload_cos_sin,
        (uint32_t)trans_mat_sharded,
        (uint32_t)cos_sin_sharded,
    };
    tt::tt_metal::TensorAccessorArgs(src_buffer).append_to(reader_compile_time_args);
    if (!cos_sin_sharded) {
        tt::tt_metal::TensorAccessorArgs(cos_buffer).append_to(reader_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(sin_buffer).append_to(reader_compile_time_args);
    }
    if (!trans_mat_sharded) {
        tt::tt_metal::TensorAccessorArgs(trans_mat_buffer).append_to(reader_compile_time_args);
    }

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
        (uint32_t)reload_cos_sin,
    };

    tt_metal::KernelHandle compute_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/"
        "rotary_embedding_llama_hc_decode.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_kernel_args});

    // ---- Assign work ranges to cores ----
    const uint32_t heads_per_core_base = num_heads / num_cores_used;
    const uint32_t heads_remainder = num_heads % num_cores_used;

    // Runtime args layout:
    //   reader [0] = src_addr
    //   reader [1] = cos_addr   (only used when !cos_sin_sharded; slot always present)
    //   reader [2] = sin_addr   (only used when !cos_sin_sharded; slot always present)
    //   reader [3] = trans_mat_addr (only used when !trans_mat_sharded; slot always present)
    //   reader [4..7] = head_start, head_end, batch_t_start, batch_t_end
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
    shared_variables.cb_trans_mat = cb_trans_mat_handle;
    shared_variables.cb_cos = cb_cos_handle;
    shared_variables.cb_sin = cb_sin_handle;

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
            // Slots [1], [2], [3] for cos/sin/trans_mat addresses are always
            // present in the runtime-args vector so override_runtime_arguments
            // can update them unconditionally.  The reader kernel only reads
            // these when the corresponding tensor is not sharded.
            rt[1] = cos_buffer->address();
            rt[2] = sin_buffer->address();
            rt[3] = trans_mat_buffer->address();
        }
        {
            auto& rt = cached_writer_args.at(core.x).at(core.y);
            rt[0] = dst_buffer->address();
        }
    }

    // For globally-allocated (sharded) CBs, update the CB base address so it
    // tracks the new buffer allocation on successive invocations.
    if (shared_variables.cb_trans_mat.has_value()) {
        UpdateDynamicCircularBufferAddress(program, *shared_variables.cb_trans_mat, *trans_mat_buffer);
    }
    if (shared_variables.cb_cos.has_value()) {
        UpdateDynamicCircularBufferAddress(program, *shared_variables.cb_cos, *cos_buffer);
    }
    if (shared_variables.cb_sin.has_value()) {
        UpdateDynamicCircularBufferAddress(program, *shared_variables.cb_sin, *sin_buffer);
    }
}

}  // namespace ttnn::experimental::prim
