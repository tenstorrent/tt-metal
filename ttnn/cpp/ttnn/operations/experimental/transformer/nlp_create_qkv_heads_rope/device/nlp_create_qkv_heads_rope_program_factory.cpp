// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_rope/device/nlp_create_qkv_heads_rope_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::experimental::prim {

using namespace tt::constants;

namespace {
// Reuse the proven, unchanged rotary_embedding multi-tile compute/writer (rotate_half / GPT-J
// convention). The q/k reader is specialized for this op (Ht == 1, num_rows == 1): functionally
// identical to the generic interleaved reader, but batches all tile reads behind a single NoC barrier
// instead of one barrier per tile (the generic per-tile serialization is this op's dominant cost).
constexpr const char* kRopeReader =
    "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_rope/device/kernels/dataflow/"
    "reader_create_qkv_heads_rope.cpp";
constexpr const char* kRopeWriter =
    "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/dataflow/"
    "writer_rotary_embedding_interleaved_start_id.cpp";
constexpr const char* kRopeCompute =
    "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding/device/kernels/compute/"
    "rotary_embedding.cpp";
// Generic interleaved tile reader (reads into CB c_0) for the un-rotated v copy.
constexpr const char* kCopyReader =
    "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp";
// Datacopy compute (c_0 -> c_16) so the v core has the SAME reader->compute->writer structure as
// the q/k cores (a reader->writer pair with no compute is not trace-replay-safe).
constexpr const char* kCopyCompute =
    "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/compute/eltwise_copy.cpp";
}  // namespace

NlpCreateQkvHeadsRopeProgramFactory::cached_program_t NlpCreateQkvHeadsRopeProgramFactory::create(
    const NlpCreateQkvHeadsRopeParams& operation_attributes,
    const NlpCreateQkvHeadsRopeInputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt::tt_metal;

    const auto& qkv = tensor_args.qkv;
    const auto& cos = tensor_args.cos;
    const auto& sin = tensor_args.sin;
    auto& q_out = std::get<0>(tensor_return_value);
    auto& k_out = std::get<1>(tensor_return_value);
    auto& v_out = std::get<2>(tensor_return_value);

    Program program{};

    tt::DataFormat in_df = datatype_to_dataformat_converter(qkv.dtype());
    uint32_t in_tile = tt::tile_size(in_df);
    tt::DataFormat cos_df = datatype_to_dataformat_converter(cos.dtype());
    uint32_t cos_tile = tt::tile_size(cos_df);
    tt::DataFormat sin_df = datatype_to_dataformat_converter(sin.dtype());
    uint32_t sin_tile = tt::tile_size(sin_df);
    tt::DataFormat scalar_df = tt::DataFormat::Float16_b;
    uint32_t scalar_tile = tt::tile_size(scalar_df);
    tt::DataFormat out_df = datatype_to_dataformat_converter(q_out.dtype());
    uint32_t out_tile = tt::tile_size(out_df);

    uint32_t hd = operation_attributes.head_dim;
    uint32_t Wt = hd / TILE_WIDTH;
    uint32_t half_Wt = Wt / 2;
    uint32_t Ht = 1;     // enforced in validate
    uint32_t HtWt = Wt;  // Ht * Wt
    uint32_t nq = operation_attributes.num_q_heads;
    uint32_t nkv = operation_attributes.num_kv_heads;

    uint32_t num_q_rows = nq;  // Ht == 1
    uint32_t num_k_rows = nkv;
    uint32_t num_v_rows = nkv;
    uint32_t num_qk = num_q_rows + num_k_rows;
    uint32_t total = num_qk + num_v_rows;

    IDevice* device = qkv.device();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    auto grid = device->compute_with_storage_grid_size();
    uint32_t gx = grid.x, gy = grid.y;
    TT_FATAL(total <= gx * gy, "rows ({}) exceed cores ({})", total, gx * gy);

    CoreRangeSet all_cores = num_cores_to_corerangeset(total, grid, true);
    const auto& cores = grid_to_cores(total, gx, gy, true);
    std::vector<CoreRange> qk_ranges, v_ranges;
    for (uint32_t i = 0; i < num_qk; ++i) {
        qk_ranges.emplace_back(cores[i], cores[i]);
    }
    for (uint32_t i = num_qk; i < total; ++i) {
        v_ranges.emplace_back(cores[i], cores[i]);
    }
    CoreRangeSet qk_cores(qk_ranges);
    CoreRangeSet v_cores(v_ranges);

    // ---- Circular buffers (on all cores) ----
    auto cb = [&](uint32_t idx, uint32_t n, tt::DataFormat df, uint32_t ts) {
        CreateCircularBuffer(program, all_cores, CircularBufferConfig(n * ts, {{idx, df}}).set_page_size(idx, ts));
    };
    uint32_t c_in = tt::CBIndex::c_0;
    uint32_t c_rot = tt::CBIndex::c_1;
    uint32_t c_cos = tt::CBIndex::c_2;
    uint32_t c_sin = tt::CBIndex::c_3;
    uint32_t c_scalar = tt::CBIndex::c_4;
    uint32_t c_rot_interm = tt::CBIndex::c_24;
    uint32_t c_cos_interm = tt::CBIndex::c_25;
    uint32_t c_sin_interm = tt::CBIndex::c_26;
    uint32_t c_out = tt::CBIndex::c_16;
    cb(c_in, 2 * Wt, in_df, in_tile);
    cb(c_rot, 2 * Wt, in_df, in_tile);
    cb(c_cos, 2 * Wt, cos_df, cos_tile);
    cb(c_sin, 2 * Wt, sin_df, sin_tile);
    cb(c_scalar, 1, scalar_df, scalar_tile);
    cb(c_rot_interm, 1, in_df, in_tile);
    cb(c_cos_interm, 1, cos_df, cos_tile);
    cb(c_sin_interm, 1, sin_df, sin_tile);
    cb(c_out, 2 * Wt, out_df, out_tile);

    const uint16_t neg_one = std::bit_cast<uint16_t>(bfloat16(-1.0f));

    // ---- RoPE kernels on q/k cores (read slices of the fused qkv) ----
    std::vector<uint32_t> rope_reader_ct = {
        c_in, c_rot, c_cos, c_sin, c_scalar, (uint32_t)neg_one, Ht, Wt, HtWt, half_Wt};
    TensorAccessorArgs(qkv.buffer()).append_to(rope_reader_ct);
    TensorAccessorArgs(cos.buffer()).append_to(rope_reader_ct);
    TensorAccessorArgs(sin.buffer()).append_to(rope_reader_ct);
    KernelHandle qk_reader = CreateKernel(program, kRopeReader, qk_cores, ReaderDataMovementConfig(rope_reader_ct));

    std::vector<uint32_t> compute_ct = {
        c_in, c_rot, c_cos, c_sin, c_scalar, c_rot_interm, c_cos_interm, c_sin_interm, c_out, 1u, Wt, half_Wt};
    CreateKernel(
        program,
        kRopeCompute,
        qk_cores,
        ComputeConfig{
            .math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_ct});

    // ---- v copy kernels on v cores: generic reader -> c_0, datacopy compute c_0 -> c_16 ----
    // Uniform reader->compute->writer structure (matches q/k cores) -> trace-replay-safe.
    std::vector<uint32_t> v_reader_ct;
    TensorAccessorArgs(qkv.buffer()).append_to(v_reader_ct);  // generic reader -> hardcoded cb c_0
    KernelHandle v_reader = CreateKernel(program, kCopyReader, v_cores, ReaderDataMovementConfig(v_reader_ct));
    KernelHandle v_compute = CreateKernel(
        program,
        kCopyCompute,
        v_cores,
        ComputeConfig{.math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en});

    // ---- uniform RoPE writer on ALL cores (drains c_16 -> q_out/k_out/v_out) ----
    std::vector<uint32_t> writer_ct = {c_out};
    TensorAccessorArgs(q_out.buffer()).append_to(writer_ct);  // q/k/v share interleaved layout
    KernelHandle writer = CreateKernel(program, kRopeWriter, all_cores, WriterDataMovementConfig(writer_ct));

    // ---- per-core runtime args ----
    auto* qkv_buf = qkv.buffer();
    auto* cos_buf = cos.buffer();
    auto* sin_buf = sin.buffer();
    for (uint32_t i = 0; i < total; ++i) {
        const CoreCoord& core = cores[i];
        if (i < num_qk) {
            bool is_q = i < num_q_rows;
            uint32_t within = is_q ? i : (i - num_q_rows);
            uint32_t qkv_start = is_q ? (within * Wt) : ((nq + within) * Wt);
            SetRuntimeArgs(
                program,
                qk_reader,
                core,
                {qkv_buf->address(), cos_buf->address(), sin_buf->address(), 1u, qkv_start, 0u, 0u});
            auto* dst = is_q ? q_out.buffer() : k_out.buffer();
            SetRuntimeArgs(program, writer, core, {dst->address(), Wt, within * Wt});
        } else {
            uint32_t within = i - num_qk;
            uint32_t qkv_start = (nq + nkv + within) * Wt;
            SetRuntimeArgs(program, v_reader, core, {qkv_buf->address(), Wt, qkv_start});
            SetRuntimeArgs(program, v_compute, core, {Wt});  // per_core_tile_cnt
            SetRuntimeArgs(program, writer, core, {v_out.buffer()->address(), Wt, within * Wt});
        }
    }

    NlpCreateQkvHeadsRopeSharedVariables sv{
        .qk_reader_kernel_id = qk_reader,
        .writer_kernel_id = writer,
        .v_reader_kernel_id = v_reader,
        .cores = cores,
        .Wt = Wt,
        .num_q_rows = num_q_rows,
        .num_k_rows = num_k_rows,
        .num_v_rows = num_v_rows};
    return cached_program_t{std::move(program), std::move(sv)};
}

void NlpCreateQkvHeadsRopeProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const NlpCreateQkvHeadsRopeParams& /*operation_attributes*/,
    const NlpCreateQkvHeadsRopeInputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt::tt_metal;
    auto& program = cached_program.program;
    const auto& sv = cached_program.shared_variables;
    const auto& cores = sv.cores;
    uint32_t num_qk = sv.num_q_rows + sv.num_k_rows;

    auto* qkv = tensor_args.qkv.buffer();
    auto* cos = tensor_args.cos.buffer();
    auto* sin = tensor_args.sin.buffer();
    auto* q_out = std::get<0>(tensor_return_value).buffer();
    auto* k_out = std::get<1>(tensor_return_value).buffer();
    auto* v_out = std::get<2>(tensor_return_value).buffer();

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        if (i < num_qk) {
            bool is_q = i < sv.num_q_rows;
            {
                auto& ra = GetRuntimeArgs(program, sv.qk_reader_kernel_id, core);
                ra[0] = qkv->address();
                ra[1] = cos->address();
                ra[2] = sin->address();
            }
            {
                auto& wa = GetRuntimeArgs(program, sv.writer_kernel_id, core);
                wa[0] = (is_q ? q_out : k_out)->address();
            }
        } else {
            {
                auto& ra = GetRuntimeArgs(program, sv.v_reader_kernel_id, core);
                ra[0] = qkv->address();
            }
            {
                auto& wa = GetRuntimeArgs(program, sv.writer_kernel_id, core);
                wa[0] = v_out->address();
            }
        }
    }
}

}  // namespace ttnn::experimental::prim
