// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Program factory of the fused KDA decode block. Work items are (head, 32-user tile row), dealt
// round-robin over the compute cores; a core walks its items one after another and, per item, its
// users sequentially with the recurrent state streamed through L1 (read once, written once).

#include "kda_decode_program_factory.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::prim {

namespace {

uint32_t f32_bits(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

// CB index plan (kept in sync with the kernels)
namespace kcb {
constexpr uint32_t raw = 0, hist = 1, taps = 2, gf = 3, gog = 4, gb = 5, cc = 6, onw = 7, sel = 8, S = 9;
constexpr uint32_t out = 10, state_out = 11, rawrm = 12, qn = 13, kn = 14, vc = 15, dec = 16, bcol = 17;
constexpr uint32_t gate = 18, ht = 19, tmp = 20, inv = 21, rows = 22, cols = 23, vread = 24, u = 25;
constexpr uint32_t sd = 26, s2 = 27, orow = 28, bs = 29, selT = 30, oblk = 31;
}  // namespace kcb

}  // namespace

tt::tt_metal::ProgramDescriptor KdaDecodeProgramFactory::create_descriptor(
    const KdaDecodeParams& attrs, const KdaDecodeInputs& in, std::vector<Tensor>& outputs) {
    const uint32_t B = attrs.B;
    const uint32_t H = attrs.H;
    const uint32_t Kt = attrs.key_dim / TILE_WIDTH;
    const uint32_t Vt = attrs.val_dim / TILE_WIDTH;
    const uint32_t Bt = (B + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t HK_t = H * Kt;
    const uint32_t HV_t = H * Vt;
    const uint32_t Hb_t = (H + TILE_WIDTH - 1) / TILE_WIDTH;
    const uint32_t Wq_t = 3 * HK_t;
    const uint32_t Wg_t = HK_t + HV_t + Hb_t;
    const uint32_t kv = Kt * Vt;
    TT_FATAL(
        Kt == 4 && Vt == 4,
        "kda_decode: the kernel is written for K = V = 128 (got K {}, V {})",
        attrs.key_dim,
        attrs.val_dim);

    auto* device = in.qkv.device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t grid_y = grid.y;
    // Work items: (head, user tile row, part). The users of a tile row are split over `parts` cores when
    // heads x tile rows alone would leave cores idle (TP meshes hold few local heads), so the per-user
    // serial recurrence runs on as many cores as possible.
    const uint32_t n_cores = grid.x * grid_y;
    const uint32_t parts = std::max<uint32_t>(1u, std::min<uint32_t>(TILE_HEIGHT, n_cores / (H * Bt)));
    const uint32_t NI = H * Bt * parts;
    const uint32_t NC = std::min<uint32_t>(NI, n_cores);
    TT_FATAL(NC >= 1, "kda_decode: no compute cores");

    std::vector<CoreCoord> core_of(NC);
    std::set<CoreRange> core_set;
    for (uint32_t c = 0; c < NC; c++) {
        core_of[c] = CoreCoord{c / grid_y, c % grid_y};
        core_set.insert(CoreRange{core_of[c], core_of[c]});
    }
    CoreRangeSet cores{core_set};

    ProgramDescriptor desc;
    const tt::DataFormat f32 = tt::DataFormat::Float32;
    const tt::DataFormat f16 = tt::DataFormat::Float16_b;
    const uint32_t t32 = tt::tile_size(f32);
    const uint32_t t16 = tt::tile_size(f16);
    auto add_cb = [&](uint32_t idx, tt::DataFormat fmt, uint32_t page, uint32_t n_pages) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = n_pages * page,
            .core_ranges = cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(idx), .data_format = fmt, .page_size = page}}}});
    };
    add_cb(kcb::raw, f16, t16, 3 * Kt);
    add_cb(kcb::hist, f16, t16, 3 * 3 * Kt);  // 9 RM blocks of [32, K] bf16 (4 bf16-tile-sized pages each)
    add_cb(kcb::taps, f32, t32, 3 * Kt);
    add_cb(kcb::gf, f32, t32, Kt);
    add_cb(kcb::gog, f32, t32, Vt);
    add_cb(kcb::gb, f32, t32, 1);
    add_cb(kcb::cc, f32, t32, Kt);
    add_cb(kcb::onw, f32, t32, Vt);
    add_cb(kcb::sel, f16, t16, 33);
    add_cb(kcb::S, f32, t32, kv);
    add_cb(kcb::out, f16, t16, 2 * Vt);  // RM [32, V] bf16 block (one 256 B row per user)
    add_cb(kcb::state_out, f32, t32, 2 * kv);
    add_cb(kcb::rawrm, f16, t16, 3 * Kt);
    add_cb(kcb::qn, f32, t32, Kt);
    add_cb(kcb::kn, f32, t32, Kt);
    add_cb(kcb::vc, f32, t32, Vt);
    add_cb(kcb::dec, f32, t32, Kt);
    add_cb(kcb::bcol, f32, t32, 1);
    add_cb(kcb::gate, f32, t32, Vt);
    add_cb(kcb::ht, f32, t32, 3 * Kt);
    add_cb(kcb::tmp, f32, t32, 2 * Kt);
    add_cb(kcb::inv, f32, t32, 1);
    add_cb(kcb::rows, f32, t32, 4 * Kt);
    add_cb(kcb::cols, f32, t32, 2 * Kt);
    add_cb(kcb::vread, f32, t32, Vt);
    add_cb(kcb::u, f32, t32, Vt);
    add_cb(kcb::sd, f32, t32, kv);
    add_cb(kcb::s2, f32, t32, kv);
    add_cb(kcb::orow, f32, t32, Vt);
    add_cb(kcb::bs, f32, t32, 1);
    add_cb(kcb::selT, f32, t32, 1);
    add_cb(kcb::oblk, f32, t32, 2 * Vt);

    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/kda_decode/device/kernels/";
    const float qscale = 1.0f / std::sqrt(static_cast<float>(attrs.key_dim));
    const std::vector<uint32_t> compute_ct = {
        Kt,
        Vt,
        Bt,
        B,
        parts,
        f32_bits(attrs.gate_lower_bound),
        f32_bits(attrs.l2_eps),
        f32_bits(attrs.norm_eps),
        f32_bits(qscale),
        f32_bits(1.0f / static_cast<float>(attrs.val_dim))};

    std::vector<uint32_t> reader_ct = {Kt, Vt, Bt, B, H, Wq_t, Wg_t, HK_t, HV_t, parts};
    TensorAccessorArgs(*in.qkv.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.conv_state.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.taps.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.gates.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.chan_consts.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.o_norm_w.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.selectors.buffer()).append_to(reader_ct);
    TensorAccessorArgs(*in.recurrent.buffer()).append_to(reader_ct);

    std::vector<uint32_t> writer_ct = {Kt, Vt, Bt, B, H, HK_t, HV_t, parts};
    TensorAccessorArgs(*outputs[0].buffer()).append_to(writer_ct);
    TensorAccessorArgs(*outputs[1].buffer()).append_to(writer_ct);
    TensorAccessorArgs(*outputs[2].buffer()).append_to(writer_ct);

    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_kda_decode.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.compile_time_args = reader_ct;
    reader.config = ReaderConfigDescriptor{};
    reader.runtime_args.reserve(NC);

    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_kda_decode.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = writer_ct;
    writer.config = WriterConfigDescriptor{};
    writer.runtime_args.reserve(NC);

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/kda_decode.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = compute_ct;
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), attrs.compute_kernel_config);
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode};
    compute.runtime_args.reserve(NC);

    const uint32_t conv_page_bytes = in.conv_state.buffer()->aligned_page_size();
    const uint32_t o_page_bytes = outputs[0].buffer()->aligned_page_size();  // one RM output row [H*V] bf16
    auto* qkv_buf = in.qkv.buffer();
    auto* conv_buf = in.conv_state.buffer();
    auto* taps_buf = in.taps.buffer();
    auto* gates_buf = in.gates.buffer();
    auto* cc_buf = in.chan_consts.buffer();
    auto* onw_buf = in.o_norm_w.buffer();
    auto* sel_buf = in.selectors.buffer();
    auto* st_buf = in.recurrent.buffer();
    auto* o_buf = outputs[0].buffer();

    for (uint32_t c = 0; c < NC; c++) {
        const auto& core = core_of[c];
        const uint32_t n_items = (NI - c + NC - 1) / NC;
        reader.emplace_runtime_args(
            core,
            {c,
             n_items,
             NC,
             qkv_buf,
             conv_buf,
             taps_buf,
             gates_buf,
             cc_buf,
             onw_buf,
             sel_buf,
             st_buf,
             conv_page_bytes});
        writer.emplace_runtime_args(core, {c, n_items, NC, o_buf, st_buf, conv_buf, conv_page_bytes, o_page_bytes});
        compute.emplace_runtime_args(core, {c, n_items, NC});
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::prim
