// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_fused_qkv_device_operation.hpp"

#include <bit>
#include <cstdint>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/device.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::transformer::deepseek_fused_qkv {

using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

constexpr auto kReaderKvKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/transformer/deepseek_fused_qkv/device/kernels/dataflow/"
    "reader_kv_fused.cpp";
constexpr auto kComputeKvKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/transformer/deepseek_fused_qkv/device/kernels/compute/"
    "compute_kv_fused.cpp";
constexpr auto kWriterKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/transformer/deepseek_fused_qkv/device/kernels/dataflow/"
    "writer_fused.cpp";
constexpr auto kReaderQKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/transformer/deepseek_fused_qkv/device/kernels/dataflow/"
    "reader_q_fused.cpp";
constexpr auto kComputeQKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/transformer/deepseek_fused_qkv/device/kernels/compute/"
    "compute_q_fused.cpp";

// Largest divisor of n that is <= cap (>= 1).
uint32_t largest_divisor_leq(uint32_t n, uint32_t cap) {
    for (uint32_t d = cap; d >= 1; --d) {
        if (n % d == 0) {
            return d;
        }
    }
    return 1;
}

}  // namespace

DeepseekFusedQkvDeviceOperation::program_factory_t DeepseekFusedQkvDeviceOperation::select_program_factory(
    const operation_attributes_t& /*args*/, const tensor_args_t& /*tensor_args*/) {
    return MultiCoreProgramFactory{};
}

void DeepseekFusedQkvDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& hidden = tensor_args.hidden;
    const auto& wqa = tensor_args.wqa;
    const auto& wqb = tensor_args.wqb;
    const auto& wkv = tensor_args.wkv;

    TT_FATAL(hidden.storage_type() == StorageType::DEVICE, "hidden must be on device");
    TT_FATAL(hidden.layout() == Layout::TILE, "hidden must be TILE layout");

    for (const auto* w : {&wqa, &wqb, &wkv}) {
        TT_FATAL(w->storage_type() == StorageType::DEVICE, "weights must be on device");
        TT_FATAL(w->layout() == Layout::TILE, "weights must be TILE layout");
        TT_FATAL(
            w->memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
                w->memory_config().buffer_type() == BufferType::DRAM,
            "weights must be DRAM WIDTH_SHARDED (got layout {} / buffer {})",
            w->memory_config().memory_layout(),
            w->memory_config().buffer_type());
    }

    const uint32_t D = hidden.padded_shape()[-1];
    const uint32_t Kqa = wqa.padded_shape()[-2];
    const uint32_t Kkv = wkv.padded_shape()[-2];
    TT_FATAL(Kqa == D, "wqa K ({}) must equal hidden D ({})", Kqa, D);
    TT_FATAL(Kkv == D, "wkv K ({}) must equal hidden D ({})", Kkv, D);

    const uint32_t q_lora = wqa.padded_shape()[-1];
    const uint32_t Kqb = wqb.padded_shape()[-2];
    TT_FATAL(Kqb == q_lora, "wqb K ({}) must equal q_lora ({})", Kqb, q_lora);

    const uint32_t N_qb = wqb.padded_shape()[-1];
    TT_FATAL(N_qb % args.num_heads == 0, "wqb N ({}) must be divisible by num_heads ({})", N_qb, args.num_heads);

    const uint32_t Rd = args.rope_dim;
    TT_FATAL(Rd % TILE_WIDTH == 0, "rope_dim ({}) must be tile-aligned", Rd);
}

DeepseekFusedQkvDeviceOperation::spec_return_value_t DeepseekFusedQkvDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& wqb = tensor_args.wqb;
    const auto& wkv = tensor_args.wkv;

    const uint32_t N_qb = wqb.padded_shape()[-1];
    const uint32_t kv_dim = wkv.padded_shape()[-1];

    // q is produced flat ([1, 1, 1, H*Dh]); the per-head RMSNorm + RoPE run on the
    // head blocks along the width. The Python wrapper reshapes to [1, 1, H, Dh].
    const ttnn::Shape q_shape({1, 1, 1, N_qb});
    const ttnn::Shape kv_shape({1, 1, 1, kv_dim});

    const auto q_spec = TensorSpec(
        q_shape,
        tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), args.q_mem_config));
    const auto kv_spec = TensorSpec(
        kv_shape,
        tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), args.kv_mem_config));

    return {q_spec, kv_spec};
}

DeepseekFusedQkvDeviceOperation::tensor_return_value_t DeepseekFusedQkvDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.hidden.device();
    std::vector<Tensor> outputs;
    outputs.reserve(specs.size());
    for (const auto& spec : specs) {
        outputs.push_back(create_device_tensor(spec, device));
    }
    return outputs;
}

tt::tt_metal::ProgramDescriptor DeepseekFusedQkvDeviceOperation::MultiCoreProgramFactory::create_descriptor(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    // ------------------------------------------------------------------------------------------
    // Correctness-first v1: KV compute partition only, on a single core.
    //   kv = rmsnorm_w(hidden @ Wkv)  ->  partial RoPE(Rd)
    // hidden (in0) is streamed once and kept resident; Wkv (in1) is streamed from DRAM in
    // K-blocks per N-subblock so it never has to fit L1 whole. The Q path + DRAM-BW-saturating
    // multi-bank parallelism land in the later plan stages (q-a-bridge / q-b-epilogue /
    // fuse-parallel / tune-integrate).
    // ------------------------------------------------------------------------------------------
    const auto& hidden = tensor_args.hidden;
    const auto& wkv = tensor_args.wkv;
    const auto& kv_norm_w = tensor_args.kv_norm_w;
    const auto& cos = tensor_args.cos;
    const auto& sin = tensor_args.sin;
    const auto& trans_mat = tensor_args.trans_mat;
    Tensor& kv_out = output[1];

    const tt::DataFormat df_in = datatype_to_dataformat_converter(hidden.dtype());
    const tt::DataFormat df_w = datatype_to_dataformat_converter(wkv.dtype());
    const tt::DataFormat df_gain = datatype_to_dataformat_converter(kv_norm_w.dtype());
    const tt::DataFormat df_cos = datatype_to_dataformat_converter(cos.dtype());
    const tt::DataFormat df_sin = datatype_to_dataformat_converter(sin.dtype());
    const tt::DataFormat df_tm = datatype_to_dataformat_converter(trans_mat.dtype());
    const tt::DataFormat df_out = datatype_to_dataformat_converter(kv_out.dtype());
    // All norm/rope intermediates are bf16 (matmul + reduce still accumulate in fp32 DST).
    const tt::DataFormat df_interm = tt::DataFormat::Float16_b;

    const uint32_t tile_in = tt::tile_size(df_in);
    const uint32_t tile_w = tt::tile_size(df_w);
    const uint32_t tile_gain = tt::tile_size(df_gain);
    const uint32_t tile_cos = tt::tile_size(df_cos);
    const uint32_t tile_sin = tt::tile_size(df_sin);
    const uint32_t tile_tm = tt::tile_size(df_tm);
    const uint32_t tile_out = tt::tile_size(df_out);
    const uint32_t tile_interm = tt::tile_size(df_interm);

    const uint32_t D = hidden.padded_shape()[-1];
    const uint32_t kv_dim = wkv.padded_shape()[-1];
    const uint32_t Kt = D / TILE_WIDTH;
    const uint32_t Nt = kv_dim / TILE_WIDTH;       // kv output tiles
    const uint32_t Nt_full = kv_dim / TILE_WIDTH;  // Wkv tile-grid width (page stride)
    const uint32_t Rd = args.rope_dim;
    const uint32_t rope_Wt = Rd / TILE_WIDTH;
    const uint32_t nope_Wt = Nt - rope_Wt;

    const uint32_t in0_block_w = largest_divisor_leq(Kt, 8);
    const uint32_t subblock_w = largest_divisor_leq(Nt, 4);
    const uint32_t num_kb = Kt / in0_block_w;
    const uint32_t num_nsub = Nt / subblock_w;

    const uint32_t eps_bits = std::bit_cast<uint32_t>(args.eps);
    const uint32_t scaler_bits = std::bit_cast<uint32_t>(1.0f / static_cast<float>(kv_dim));

    auto* device = hidden.device();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), args.compute_kernel_config);

    // Single KV core (v1).
    const CoreCoord kv_core{0, 0};
    const CoreRangeSet kv_cores{CoreRange{kv_core, kv_core}};

    tt::tt_metal::ProgramDescriptor desc;

    // CB indices.
    constexpr uint8_t in0_cb = tt::CBIndex::c_0;
    constexpr uint8_t in1_cb = tt::CBIndex::c_1;
    constexpr uint8_t scaler_cb = tt::CBIndex::c_2;
    constexpr uint8_t gain_cb = tt::CBIndex::c_3;
    constexpr uint8_t cos_cb = tt::CBIndex::c_4;
    constexpr uint8_t sin_cb = tt::CBIndex::c_5;
    constexpr uint8_t trans_mat_cb = tt::CBIndex::c_6;
    constexpr uint8_t mm_cb = tt::CBIndex::c_7;
    constexpr uint8_t x2_cb = tt::CBIndex::c_8;
    constexpr uint8_t recip_cb = tt::CBIndex::c_9;
    constexpr uint8_t normed_cb = tt::CBIndex::c_10;
    constexpr uint8_t normed_g_cb = tt::CBIndex::c_11;
    constexpr uint8_t rotated_cb = tt::CBIndex::c_12;
    constexpr uint8_t cos_interm_cb = tt::CBIndex::c_13;
    constexpr uint8_t sin_interm_cb = tt::CBIndex::c_14;
    constexpr uint8_t out_cb = tt::CBIndex::c_16;

    auto add_cb =
        [&](const CoreRangeSet& cores, uint8_t index, uint32_t n_tiles, tt::DataFormat df, uint32_t tile_bytes) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = n_tiles * tile_bytes,
                .core_ranges = cores,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = index, .data_format = df, .page_size = tile_bytes}}},
            });
        };

    add_cb(kv_cores, in0_cb, Kt, df_in, tile_in);
    add_cb(kv_cores, in1_cb, 2 * in0_block_w * subblock_w, df_w, tile_w);  // double-buffered weight blocks
    add_cb(kv_cores, scaler_cb, 1, df_interm, tile_interm);
    add_cb(kv_cores, gain_cb, Nt, df_gain, tile_gain);
    add_cb(kv_cores, cos_cb, rope_Wt, df_cos, tile_cos);
    add_cb(kv_cores, sin_cb, rope_Wt, df_sin, tile_sin);
    add_cb(kv_cores, trans_mat_cb, 1, df_tm, tile_tm);
    add_cb(kv_cores, mm_cb, Nt, df_interm, tile_interm);
    add_cb(kv_cores, x2_cb, Nt, df_interm, tile_interm);
    add_cb(kv_cores, recip_cb, 1, df_interm, tile_interm);
    add_cb(kv_cores, normed_cb, Nt, df_interm, tile_interm);
    add_cb(kv_cores, normed_g_cb, Nt, df_interm, tile_interm);
    add_cb(kv_cores, rotated_cb, rope_Wt, df_interm, tile_interm);
    add_cb(kv_cores, cos_interm_cb, rope_Wt, df_interm, tile_interm);
    add_cb(kv_cores, sin_interm_cb, rope_Wt, df_interm, tile_interm);
    add_cb(kv_cores, out_cb, Nt, df_out, tile_out);

    auto* hidden_buffer = hidden.buffer();
    auto* wkv_buffer = wkv.buffer();
    auto* gain_buffer = kv_norm_w.buffer();
    auto* cos_buffer = cos.buffer();
    auto* sin_buffer = sin.buffer();
    auto* trans_mat_buffer = trans_mat.buffer();
    auto* kv_out_buffer = kv_out.buffer();

    // ---- Reader ----
    KernelDescriptor::CompileTimeArgs reader_args = {
        (uint32_t)in0_cb,
        (uint32_t)in1_cb,
        (uint32_t)gain_cb,
        (uint32_t)cos_cb,
        (uint32_t)sin_cb,
        (uint32_t)trans_mat_cb,
        (uint32_t)scaler_cb,
        Kt,
        Nt,
        Nt_full,
        in0_block_w,
        subblock_w,
        num_kb,
        num_nsub,
        rope_Wt,
        scaler_bits,
    };
    TensorAccessorArgs(*hidden_buffer).append_to(reader_args);
    TensorAccessorArgs(*wkv_buffer).append_to(reader_args);
    TensorAccessorArgs(*gain_buffer).append_to(reader_args);
    TensorAccessorArgs(*cos_buffer).append_to(reader_args);
    TensorAccessorArgs(*sin_buffer).append_to(reader_args);
    TensorAccessorArgs(*trans_mat_buffer).append_to(reader_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kReaderKvKernelPath;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = kv_cores;
    reader_desc.compile_time_args = std::move(reader_args);
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.emplace_runtime_args(
        kv_core, {hidden_buffer, wkv_buffer, gain_buffer, cos_buffer, sin_buffer, trans_mat_buffer});

    // ---- Compute ----
    KernelDescriptor::CompileTimeArgs compute_args = {
        (uint32_t)in0_cb,
        (uint32_t)in1_cb,
        (uint32_t)scaler_cb,
        (uint32_t)gain_cb,
        (uint32_t)cos_cb,
        (uint32_t)sin_cb,
        (uint32_t)trans_mat_cb,
        (uint32_t)mm_cb,
        (uint32_t)x2_cb,
        (uint32_t)recip_cb,
        (uint32_t)normed_cb,
        (uint32_t)normed_g_cb,
        (uint32_t)rotated_cb,
        (uint32_t)cos_interm_cb,
        (uint32_t)sin_interm_cb,
        (uint32_t)out_cb,
        Kt,
        Nt,
        in0_block_w,
        subblock_w,
        num_kb,
        num_nsub,
        nope_Wt,
        rope_Wt,
        eps_bits,
    };
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = kComputeKvKernelPath;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = kv_cores;
    compute_desc.compile_time_args = std::move(compute_args);
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };
    compute_desc.runtime_args.emplace_back(kv_core, KernelDescriptor::CoreRuntimeArgs{});

    // ---- Writer ----
    KernelDescriptor::CompileTimeArgs writer_args = {
        (uint32_t)out_cb,
        Nt,
    };
    TensorAccessorArgs(*kv_out_buffer).append_to(writer_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kWriterKernelPath;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = kv_cores;
    writer_desc.compile_time_args = std::move(writer_args);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.emplace_runtime_args(kv_core, {kv_out_buffer, (uint32_t)0});

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(compute_desc));
    desc.kernels.push_back(std::move(writer_desc));

    // ==========================================================================================
    // Q compute partition (disjoint cores, row y=1). Each Q core recomputes q_a locally (no
    // cross-core bridge yet) and owns a contiguous head-slice of q_b.
    // ==========================================================================================
    const auto& wqa = tensor_args.wqa;
    const auto& wqb = tensor_args.wqb;
    const auto& qa_norm_w = tensor_args.qa_norm_w;
    Tensor& q_out = output[0];

    const uint32_t q_lora = wqa.padded_shape()[-1];
    const uint32_t N_qb = wqb.padded_shape()[-1];
    const uint32_t num_heads = args.num_heads;
    const uint32_t Dh = N_qb / num_heads;

    const uint32_t Kt_qa = D / TILE_WIDTH;
    const uint32_t Nqa = q_lora / TILE_WIDTH;
    const uint32_t Nqa_full = q_lora / TILE_WIDTH;
    const uint32_t Kt_qb = q_lora / TILE_WIDTH;
    const uint32_t Nqb_full = N_qb / TILE_WIDTH;
    const uint32_t Dht_head = Dh / TILE_WIDTH;
    const uint32_t nope_Wt_head = (Dh - Rd) / TILE_WIDTH;

    // num_qcores = largest divisor of num_heads that is <= 8; each core owns whole heads.
    const uint32_t num_qcores = largest_divisor_leq(num_heads, 8);
    const uint32_t heads_per_core = num_heads / num_qcores;
    const uint32_t Nqb_core = heads_per_core * Dht_head;

    const uint32_t in0_block_w_qa = largest_divisor_leq(Kt_qa, 8);
    const uint32_t subblock_w_qa = largest_divisor_leq(Nqa, 4);
    const uint32_t num_kb_qa = Kt_qa / in0_block_w_qa;
    const uint32_t num_nsub_qa = Nqa / subblock_w_qa;
    const uint32_t in0_block_w_qb = largest_divisor_leq(Kt_qb, 8);
    const uint32_t subblock_w_qb = largest_divisor_leq(Nqb_core, 4);
    const uint32_t num_kb_qb = Kt_qb / in0_block_w_qb;
    const uint32_t num_nsub_qb = Nqb_core / subblock_w_qb;

    const uint32_t scaler_qa_bits = std::bit_cast<uint32_t>(1.0f / static_cast<float>(q_lora));
    const uint32_t scaler_head_bits = std::bit_cast<uint32_t>(1.0f / static_cast<float>(Dh));

    const CoreRange q_core_range{CoreCoord{0, 1}, CoreCoord{num_qcores - 1, 1}};
    const CoreRangeSet q_cores{q_core_range};

    // Q CB indices (per-core, independent of the KV core's CBs).
    constexpr uint8_t q_in0_cb = tt::CBIndex::c_0;
    constexpr uint8_t q_in1_cb = tt::CBIndex::c_1;
    constexpr uint8_t q_scaler_qa_cb = tt::CBIndex::c_2;
    constexpr uint8_t q_scaler_head_cb = tt::CBIndex::c_3;
    constexpr uint8_t q_gain_cb = tt::CBIndex::c_4;
    constexpr uint8_t q_cos_cb = tt::CBIndex::c_5;
    constexpr uint8_t q_sin_cb = tt::CBIndex::c_6;
    constexpr uint8_t q_trans_mat_cb = tt::CBIndex::c_7;
    constexpr uint8_t q_mm_qa_cb = tt::CBIndex::c_8;
    constexpr uint8_t q_qa_cb = tt::CBIndex::c_9;
    constexpr uint8_t q_mm_qb_cb = tt::CBIndex::c_10;
    constexpr uint8_t q_x2_cb = tt::CBIndex::c_11;
    constexpr uint8_t q_recip_cb = tt::CBIndex::c_12;
    constexpr uint8_t q_normed_cb = tt::CBIndex::c_13;
    constexpr uint8_t q_rotated_cb = tt::CBIndex::c_14;
    constexpr uint8_t q_cos_interm_cb = tt::CBIndex::c_15;
    constexpr uint8_t q_out_cb = tt::CBIndex::c_16;
    constexpr uint8_t q_sin_interm_cb = tt::CBIndex::c_17;

    const uint32_t blk_qa = in0_block_w_qa * subblock_w_qa;
    const uint32_t blk_qb = in0_block_w_qb * subblock_w_qb;
    const uint32_t in1_q_tiles = 2 * (blk_qa > blk_qb ? blk_qa : blk_qb);

    add_cb(q_cores, q_in0_cb, Kt_qa, df_in, tile_in);
    add_cb(q_cores, q_in1_cb, in1_q_tiles, df_w, tile_w);
    add_cb(q_cores, q_scaler_qa_cb, 1, df_interm, tile_interm);
    add_cb(q_cores, q_scaler_head_cb, 1, df_interm, tile_interm);
    add_cb(q_cores, q_gain_cb, Nqa, df_gain, tile_gain);
    add_cb(q_cores, q_cos_cb, rope_Wt, df_cos, tile_cos);
    add_cb(q_cores, q_sin_cb, rope_Wt, df_sin, tile_sin);
    add_cb(q_cores, q_trans_mat_cb, 1, df_tm, tile_tm);
    add_cb(q_cores, q_mm_qa_cb, Nqa, df_interm, tile_interm);
    add_cb(q_cores, q_qa_cb, Nqa, df_interm, tile_interm);
    add_cb(q_cores, q_mm_qb_cb, Nqb_core, df_interm, tile_interm);
    add_cb(q_cores, q_x2_cb, Nqa, df_interm, tile_interm);
    add_cb(q_cores, q_recip_cb, 1, df_interm, tile_interm);
    add_cb(q_cores, q_normed_cb, Nqa, df_interm, tile_interm);
    add_cb(q_cores, q_rotated_cb, rope_Wt, df_interm, tile_interm);
    add_cb(q_cores, q_cos_interm_cb, rope_Wt, df_interm, tile_interm);
    add_cb(q_cores, q_out_cb, Nqb_core, df_out, tile_out);
    add_cb(q_cores, q_sin_interm_cb, rope_Wt, df_interm, tile_interm);

    auto* wqa_buffer = wqa.buffer();
    auto* wqb_buffer = wqb.buffer();
    auto* qa_gain_buffer = qa_norm_w.buffer();
    auto* q_out_buffer = q_out.buffer();

    // ---- Q reader ----
    KernelDescriptor::CompileTimeArgs q_reader_args = {
        (uint32_t)q_in0_cb,
        (uint32_t)q_in1_cb,
        (uint32_t)q_gain_cb,
        (uint32_t)q_cos_cb,
        (uint32_t)q_sin_cb,
        (uint32_t)q_trans_mat_cb,
        (uint32_t)q_scaler_qa_cb,
        (uint32_t)q_scaler_head_cb,
        Kt_qa,
        Nqa,
        Nqa_full,
        in0_block_w_qa,
        subblock_w_qa,
        num_kb_qa,
        num_nsub_qa,
        Kt_qb,
        Nqb_core,
        Nqb_full,
        in0_block_w_qb,
        subblock_w_qb,
        num_kb_qb,
        num_nsub_qb,
        rope_Wt,
        scaler_qa_bits,
        scaler_head_bits,
    };
    TensorAccessorArgs(*hidden_buffer).append_to(q_reader_args);
    TensorAccessorArgs(*wqa_buffer).append_to(q_reader_args);
    TensorAccessorArgs(*wqb_buffer).append_to(q_reader_args);
    TensorAccessorArgs(*qa_gain_buffer).append_to(q_reader_args);
    TensorAccessorArgs(*cos_buffer).append_to(q_reader_args);
    TensorAccessorArgs(*sin_buffer).append_to(q_reader_args);
    TensorAccessorArgs(*trans_mat_buffer).append_to(q_reader_args);

    KernelDescriptor q_reader_desc;
    q_reader_desc.kernel_source = kReaderQKernelPath;
    q_reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    q_reader_desc.core_ranges = q_cores;
    q_reader_desc.compile_time_args = std::move(q_reader_args);
    q_reader_desc.config = ReaderConfigDescriptor{};

    // ---- Q compute ----
    KernelDescriptor::CompileTimeArgs q_compute_args = {
        (uint32_t)q_in0_cb,
        (uint32_t)q_in1_cb,
        (uint32_t)q_scaler_qa_cb,
        (uint32_t)q_scaler_head_cb,
        (uint32_t)q_gain_cb,
        (uint32_t)q_cos_cb,
        (uint32_t)q_sin_cb,
        (uint32_t)q_trans_mat_cb,
        (uint32_t)q_mm_qa_cb,
        (uint32_t)q_qa_cb,
        (uint32_t)q_mm_qb_cb,
        (uint32_t)q_x2_cb,
        (uint32_t)q_recip_cb,
        (uint32_t)q_normed_cb,
        (uint32_t)q_rotated_cb,
        (uint32_t)q_cos_interm_cb,
        (uint32_t)q_out_cb,
        (uint32_t)q_sin_interm_cb,
        Kt_qa,
        Nqa,
        in0_block_w_qa,
        subblock_w_qa,
        num_kb_qa,
        num_nsub_qa,
        Kt_qb,
        Nqb_core,
        in0_block_w_qb,
        subblock_w_qb,
        num_kb_qb,
        num_nsub_qb,
        heads_per_core,
        Dht_head,
        nope_Wt_head,
        rope_Wt,
        eps_bits,
    };
    KernelDescriptor q_compute_desc;
    q_compute_desc.kernel_source = kComputeQKernelPath;
    q_compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    q_compute_desc.core_ranges = q_cores;
    q_compute_desc.compile_time_args = std::move(q_compute_args);
    q_compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };

    // ---- Q writer ----
    KernelDescriptor::CompileTimeArgs q_writer_args = {
        (uint32_t)q_out_cb,
        Nqb_core,
    };
    TensorAccessorArgs(*q_out_buffer).append_to(q_writer_args);

    KernelDescriptor q_writer_desc;
    q_writer_desc.kernel_source = kWriterKernelPath;
    q_writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    q_writer_desc.core_ranges = q_cores;
    q_writer_desc.compile_time_args = std::move(q_writer_args);
    q_writer_desc.config = WriterConfigDescriptor{};

    // Per-core Q runtime args: head-slice offset (in tiles) into Wqb and the output.
    const auto& q_core_list = corerange_to_cores(q_cores, std::nullopt, /*row_wise=*/true);
    for (uint32_t c = 0; c < q_core_list.size(); ++c) {
        const uint32_t n_start_tile = c * Nqb_core;
        q_reader_desc.emplace_runtime_args(
            q_core_list[c],
            {hidden_buffer,
             wqa_buffer,
             wqb_buffer,
             qa_gain_buffer,
             cos_buffer,
             sin_buffer,
             trans_mat_buffer,
             n_start_tile});
        q_compute_desc.runtime_args.emplace_back(q_core_list[c], KernelDescriptor::CoreRuntimeArgs{});
        q_writer_desc.emplace_runtime_args(q_core_list[c], {q_out_buffer, n_start_tile});
    }

    desc.kernels.push_back(std::move(q_reader_desc));
    desc.kernels.push_back(std::move(q_compute_desc));
    desc.kernels.push_back(std::move(q_writer_desc));
    return desc;
}

}  // namespace ttnn::operations::experimental::transformer::deepseek_fused_qkv

namespace ttnn::prim {

std::vector<ttnn::Tensor> deepseek_fused_qkv(
    const ttnn::Tensor& hidden,
    const ttnn::Tensor& wqa,
    const ttnn::Tensor& wqb,
    const ttnn::Tensor& wkv,
    const ttnn::Tensor& qa_norm_w,
    const ttnn::Tensor& kv_norm_w,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& trans_mat,
    float eps,
    uint32_t rope_dim,
    uint32_t num_heads,
    const std::optional<tt::tt_metal::MemoryConfig>& q_mem_config,
    const std::optional<tt::tt_metal::MemoryConfig>& kv_mem_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    using OperationType =
        ttnn::operations::experimental::transformer::deepseek_fused_qkv::DeepseekFusedQkvDeviceOperation;

    auto arch = hidden.storage_type() == tt::tt_metal::StorageType::DEVICE ? hidden.device()->arch()
                                                                           : ttnn::GetDefaultDevice()->arch();
    auto kernel_config_val = init_device_compute_kernel_config(
        arch, compute_kernel_config, tt::tt_metal::MathFidelity::HiFi4, true, true, false);

    auto attrs = OperationType::operation_attributes_t{
        .eps = eps,
        .rope_dim = rope_dim,
        .num_heads = num_heads,
        .q_mem_config = q_mem_config.value_or(tt::tt_metal::MemoryConfig{}),
        .kv_mem_config = kv_mem_config.value_or(tt::tt_metal::MemoryConfig{}),
        .compute_kernel_config = kernel_config_val,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .hidden = hidden,
        .wqa = wqa,
        .wqb = wqb,
        .wkv = wkv,
        .qa_norm_w = qa_norm_w,
        .kv_norm_w = kv_norm_w,
        .cos = cos,
        .sin = sin,
        .trans_mat = trans_mat};
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
