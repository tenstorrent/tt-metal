// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "kv_sdpa_device_operation.hpp"

#include "tt-metalium/constants.hpp"
#include "tt-metalium/work_split.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::kv_sdpa {

using namespace tt;
using namespace tt::tt_metal;

// FlashFused: one core per Q head; the compute kernel calls the production transformer-SDPA
// sdpa_standard() online-softmax routine (matches production speed). MQA: every core's reader reads
// the single KV head, chunk-by-chunk. Interleaved I/O for now; sharded folds come next.
ProgramDescriptor KvSdpaDeviceOperation::FlashFused::create_descriptor(
    const operation_attributes_t& attrs, const tensor_args_t& ta, tensor_return_value_t& out) {
    IDevice* device = ta.q.device();

    const auto qdf = datatype_to_dataformat_converter(ta.q.dtype());
    const auto kdf = datatype_to_dataformat_converter(ta.k.dtype());
    const auto vdf = datatype_to_dataformat_converter(ta.v.dtype());
    const auto odf = datatype_to_dataformat_converter(out.dtype());
    constexpr auto bf16 = tt::DataFormat::Float16_b;  // im/stats/scalar all bf16 (matches prod sdpa)

    const auto qtile = ta.q.tensor_spec().tile();
    const auto ktile = ta.k.tensor_spec().tile();
    const auto vtile = ta.v.tensor_spec().tile();
    const auto otile = out.tensor_spec().tile();
    const uint32_t q_ts = qtile.get_tile_size(qdf);
    const uint32_t k_ts = ktile.get_tile_size(kdf);
    const uint32_t v_ts = vtile.get_tile_size(vdf);
    const uint32_t o_ts = otile.get_tile_size(odf);
    const uint32_t bf16_ts = qtile.get_tile_size(bf16);

    const auto& qs = ta.q.padded_shape();
    const auto& ks = ta.k.padded_shape();
    const uint32_t NQH = qs[1];
    const uint32_t NKH = ks[1];
    const uint32_t DHt = qs[3] / tt::constants::TILE_WIDTH;
    const uint32_t vDHt = DHt;
    const uint32_t group = NQH / NKH;
    // Total KV = optional resident prefix (past_k/past_v) followed by the new/suffix K/V. The reader
    // reads both ranges, so the caller need not pre-concatenate.
    const bool has_past = ta.past_k.has_value();
    const uint32_t prefix_Kt = has_past ? (ta.past_k->padded_shape()[2] / tt::constants::TILE_HEIGHT) : 0;
    const uint32_t suffix_Kt = ks[2] / tt::constants::TILE_HEIGHT;
    const uint32_t Kt = prefix_Kt + suffix_Kt;

    // KV chunk size: largest small divisor of Kt (matches prod's k_chunk≈96 -> 3 tiles for Kt=33).
    uint32_t Sk_chunk_t = 1;
    for (uint32_t cand : {4u, 3u, 2u}) {
        if (Kt % cand == 0) {
            Sk_chunk_t = cand;
            break;
        }
    }
    const uint32_t k_num_chunks = Kt / Sk_chunk_t;
    const uint32_t qk_subblock_w = Sk_chunk_t;  // 1 x Sk_chunk_t qk subblock (Sq_chunk_t==1)
    const uint32_t out_subblock_w = vDHt;       // 1 x vDHt out subblock

    const uint32_t Sq_chunk_t = 1;
    const uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    const uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    const uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;
    const uint32_t out_tiles = Sq_chunk_t * vDHt;

    const CoreRangeSet cores =
        tt::tt_metal::num_cores_to_corerangeset(NQH, device->compute_with_storage_grid_size(), /*row_wise=*/true);
    const auto core_vec = corerange_to_cores(cores, std::nullopt, true);

    ProgramDescriptor desc;
    auto add_cb = [&](uint32_t idx, uint32_t ntiles, tt::DataFormat df, uint32_t ts, const auto& tile) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = ntiles * ts,
            .core_ranges = cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = idx, .data_format = df, .page_size = ts, .tile = TileDescriptor{tile}}}}});
    };
    using C = CBIndex;
    add_cb(C::c_0, q_chunk_tiles, qdf, q_ts, qtile);                 // cb_q_in
    add_cb(C::c_1, k_chunk_tiles * 2, kdf, k_ts, ktile);             // cb_k_in (double-buffered)
    add_cb(C::c_2, v_chunk_tiles * 2, vdf, v_ts, vtile);             // cb_v_in
    add_cb(C::c_5, 1, bf16, bf16_ts, qtile);                         // cb_identity_scale_in
    add_cb(C::c_7, 1, bf16, bf16_ts, qtile);                         // cb_col_identity
    add_cb(C::c_24, Sq_chunk_t * Sk_chunk_t, bf16, bf16_ts, qtile);  // cb_qk_im
    add_cb(C::c_25, out_tiles, bf16, bf16_ts, qtile);                // cb_out_im_A
    add_cb(C::c_26, out_tiles, bf16, bf16_ts, qtile);                // cb_out_im_B
    add_cb(C::c_27, Sq_chunk_t, bf16, bf16_ts, qtile);               // cb_max_A
    add_cb(C::c_28, Sq_chunk_t, bf16, bf16_ts, qtile);               // cb_max_B
    add_cb(C::c_29, Sq_chunk_t, bf16, bf16_ts, qtile);               // cb_sum_A
    add_cb(C::c_30, Sq_chunk_t, bf16, bf16_ts, qtile);               // cb_sum_B
    add_cb(C::c_31, Sq_chunk_t, bf16, bf16_ts, qtile);               // cb_exp_max_diff
    add_cb(C::c_16, out_tiles, odf, o_ts, otile);                    // cb_out (interleaved output)

    // 1.0 packed as bf16 for the reduce/bcast scalars (identity_scalar_packed in prod writer).
    constexpr uint32_t identity_scalar_packed = 0x3F803F80u;

    // ---- Reader ----
    // Suffix-relative geometry: prefix_Kt tiles come from past_k/past_v, the rest (suffix_Kt) from k/v.
    KernelDescriptor::CompileTimeArgs reader_cta = {
        NQH, DHt, Kt, Sk_chunk_t, k_num_chunks, prefix_Kt, (uint32_t)has_past};
    TensorAccessorArgs(*ta.q.buffer()).append_to(reader_cta);
    TensorAccessorArgs(*ta.k.buffer()).append_to(reader_cta);
    TensorAccessorArgs(*ta.v.buffer()).append_to(reader_cta);
    // Always append the prefix accessors so the reader's compile-time offsets are valid; when there is
    // no real past they alias k/v (placeholders) and the reader never reads them (has_past gates use).
    Buffer* pk_buf = has_past ? ta.past_k->buffer() : ta.k.buffer();
    Buffer* pv_buf = has_past ? ta.past_v->buffer() : ta.v.buffer();
    TensorAccessorArgs(*pk_buf).append_to(reader_cta);
    TensorAccessorArgs(*pv_buf).append_to(reader_cta);
    KernelDescriptor reader{};
    reader.kernel_source = "ttnn/cpp/ttnn/operations/kv_sdpa/device/kernels/dataflow/reader_fused.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.compile_time_args = reader_cta;
    reader.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_0};

    // ---- Writer (generates the sdpa scalars + drains cb_out to this core's Q head of the output) ----
    KernelDescriptor::CompileTimeArgs writer_cta = {DHt, identity_scalar_packed};
    TensorAccessorArgs(*out.buffer()).append_to(writer_cta);
    KernelDescriptor writer{};
    writer.kernel_source = "ttnn/cpp/ttnn/operations/kv_sdpa/device/kernels/dataflow/writer_fused.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = writer_cta;
    writer.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_1};

    // ---- Compute ----
    const auto ckc = ttnn::init_device_compute_kernel_config(
        device->arch(), attrs.compute_kernel_config, MathFidelity::HiFi2, false, false, false);
    KernelDescriptor compute{};
    compute.kernel_source = "ttnn/cpp/ttnn/operations/kv_sdpa/device/kernels/compute/flash_fused.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = {Sk_chunk_t, DHt, Kt, k_num_chunks, attrs.scale_bits, qk_subblock_w, out_subblock_w};
    // Granularity defines compute_common.hpp requires (loop-unroll factors; must divide their counts).
    // For Sq_chunk_t==1: stats/reduce over 1 -> 1; sub_exp/mul_bcast over Sk_chunk_t; dht over DHt.
    compute.defines = {
        {"STATS_GRANULARITY", "1"},
        {"SUB_EXP_GRANULARITY", std::to_string(Sk_chunk_t)},
        {"MUL_BCAST_GRANULARITY", std::to_string(Sk_chunk_t)},
        {"DHT_GRANULARITY", std::to_string(DHt)},
        {"REDUCE_GRANULARITY", "1"},
        {"EXP_APPROX_MODE", "0"}};
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = ckc.math_fidelity,
        .fp32_dest_acc_en = ckc.fp32_dest_acc_en,
        .dst_full_sync_en = ckc.dst_full_sync_en,
        .math_approx_mode = ckc.math_approx_mode};

    for (uint32_t h = 0; h < NQH; ++h) {
        const uint32_t kv_head = h / group;
        reader.emplace_runtime_args(
            core_vec[h], {ta.q.buffer(), ta.k.buffer(), ta.v.buffer(), h, kv_head, pk_buf, pv_buf});
        writer.emplace_runtime_args(core_vec[h], {out.buffer(), h});
    }
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::operations::kv_sdpa
