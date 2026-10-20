// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Program factory of the fused DSA decode state maintenance: Bt "role A" cores (one per 32-user tile row:
// the q_a RMSNorm) and B "role B" cores (one per user: its latent / ring / pool updates and the pooling).

#include "dsa_decode_state_program_factory.hpp"

#include <algorithm>
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

namespace dcb {
constexpr uint32_t qa = 0, wq = 1, kva = 2, kblk = 3, gblk = 4, wkva = 5, wk = 6, bk = 7, ape = 8, sel = 9;
constexpr uint32_t lat_old = 10, ring_old = 11, pool_old = 12, selrow = 13, qout = 14, lat_new = 15, ring_new = 16;
constexpr uint32_t pool_new = 17, tmp = 18, inv = 19, normed = 20, knorm = 21, rows = 22, selT = 23, keep = 24;
constexpr uint32_t ringf = 25, pscr = 26, prow = 27, run = 28, meta_w = 29, idx = 30;
}  // namespace dcb

}  // namespace

tt::tt_metal::ProgramDescriptor DsaDecodeStateProgramFactory::create_descriptor(
    const DsaDecodeStateParams& a, const DsaDecodeStateInputs& in, std::vector<Tensor>& outputs) {
    const uint32_t B = a.B;
    const uint32_t Bt = (B + TILE_HEIGHT - 1) / TILE_HEIGHT;
    const uint32_t Lq_t = a.Lq / TILE_WIDTH;
    const uint32_t L_t = a.L / TILE_WIDTH;
    const uint32_t D_t = a.D / TILE_WIDTH;
    const uint32_t Ws_t = in.stem.logical_shape()[3] / TILE_WIDTH;
    TT_FATAL(Lq_t % 4 == 0 && L_t % 4 == 0 && D_t == 4, "dsa_decode_state: the kernel wants Lq, L multiples of 128 and D = 128 (got {}, {}, {})", a.Lq, a.L, a.D);

    auto* device = in.stem.device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t NC = Bt + B;
    TT_FATAL(NC <= grid.x * grid.y, "dsa_decode_state: needs {} cores (tile rows + users), the grid has {}", NC, grid.x * grid.y);
    std::vector<CoreCoord> core_of(NC);
    std::set<CoreRange> core_set;
    for (uint32_t c = 0; c < NC; c++) {
        core_of[c] = CoreCoord{c / grid.y, c % grid.y};
        core_set.insert(CoreRange{core_of[c], core_of[c]});
    }
    CoreRangeSet cores{core_set};

    ProgramDescriptor desc;
    const tt::DataFormat f32 = tt::DataFormat::Float32;
    const tt::DataFormat f16 = tt::DataFormat::Float16_b;
    const tt::DataFormat u32 = tt::DataFormat::UInt32;
    const uint32_t t32 = tt::tile_size(f32);
    const uint32_t t16 = tt::tile_size(f16);
    auto add_cb = [&](uint32_t idx, tt::DataFormat fmt, uint32_t page, uint32_t n_pages) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = n_pages * page,
            .core_ranges = cores,
            .format_descriptors = {
                {CBFormatDescriptor{.buffer_index = static_cast<uint8_t>(idx), .data_format = fmt, .page_size = page}}}});
    };
    add_cb(dcb::qa, f16, t16, Lq_t);
    add_cb(dcb::wq, f32, t32, 8);
    add_cb(dcb::kva, f16, t16, L_t);
    add_cb(dcb::kblk, f16, t16, D_t);
    add_cb(dcb::gblk, f16, t16, D_t);
    add_cb(dcb::wkva, f32, t32, L_t);
    add_cb(dcb::wk, f32, t32, D_t);
    add_cb(dcb::bk, f32, t32, D_t);
    add_cb(dcb::ape, f32, t32, D_t);
    add_cb(dcb::sel, f16, t16, 33);
    add_cb(dcb::lat_old, f16, t16, L_t);
    add_cb(dcb::ring_old, f16, t16, 2 * D_t);
    add_cb(dcb::pool_old, f16, t16, D_t);
    add_cb(dcb::selrow, f16, t16, 3);
    add_cb(dcb::qout, f16, t16, 8);
    add_cb(dcb::lat_new, f16, t16, L_t);
    add_cb(dcb::ring_new, f16, t16, 2 * D_t);
    add_cb(dcb::pool_new, f16, t16, D_t);
    add_cb(dcb::tmp, f32, t32, 8);
    add_cb(dcb::inv, f32, t32, 1);
    add_cb(dcb::normed, f32, t32, L_t);
    add_cb(dcb::knorm, f32, t32, D_t);
    add_cb(dcb::rows, f32, t32, L_t + 2 * D_t);
    add_cb(dcb::selT, f32, t32, 3);
    add_cb(dcb::keep, f32, t32, 3);
    add_cb(dcb::ringf, f32, t32, 2 * D_t);
    add_cb(dcb::pscr, f32, t32, 2 * D_t);
    add_cb(dcb::prow, f32, t32, D_t);
    add_cb(dcb::run, f32, t32, 2);
    add_cb(dcb::meta_w, u32, 32, 1);
    add_cb(dcb::idx, u32, 512, 1);

    const std::string kdir = "ttnn/cpp/ttnn/operations/transformer/dsa_decode_state/device/kernels/";
    const std::vector<uint32_t> compute_ct = {
        Lq_t,
        L_t,
        D_t,
        f32_bits(a.eps_rms),
        f32_bits(a.eps_ln),
        f32_bits(1.0f / static_cast<float>(a.Lq)),
        f32_bits(1.0f / static_cast<float>(a.L)),
        f32_bits(1.0f / static_cast<float>(a.D))};

    std::vector<uint32_t> reader_ct = {
        Lq_t, L_t, D_t, Ws_t, a.qa_off / TILE_WIDTH, a.kva_off / TILE_WIDTH, a.k_off / TILE_WIDTH, a.g_off / TILE_WIDTH, a.P, a.pool_blocks, Bt};
    for (const Tensor* t : {&in.stem, &in.latent_cache, &in.latent_page_table, &in.positions, &in.ring_k, &in.ring_g,
                            &in.pool_cache, &in.qa_norm_w, &in.kva_norm_w, &in.k_norm_w, &in.k_norm_b, &in.ape, &in.selectors}) {
        TensorAccessorArgs(*t->buffer()).append_to(reader_ct);
    }
    std::vector<uint32_t> writer_ct = {Lq_t, L_t, D_t};
    const Tensor& q_out = outputs[0];
    for (const Tensor* t : {&q_out, &in.latent_cache, &in.ring_k, &in.ring_g, &in.pool_cache}) {
        TensorAccessorArgs(*t->buffer()).append_to(writer_ct);
    }

    KernelDescriptor reader;
    reader.kernel_source = kdir + "dataflow/reader_dsa_decode_state.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = cores;
    reader.compile_time_args = reader_ct;
    reader.config = ReaderConfigDescriptor{};
    reader.runtime_args.reserve(NC);

    KernelDescriptor writer;
    writer.kernel_source = kdir + "dataflow/writer_dsa_decode_state.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = cores;
    writer.compile_time_args = writer_ct;
    writer.config = WriterConfigDescriptor{};
    writer.runtime_args.reserve(NC);

    KernelDescriptor compute;
    compute.kernel_source = kdir + "compute/dsa_decode_state.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = cores;
    compute.compile_time_args = compute_ct;
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), a.compute_kernel_config);
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode};
    compute.runtime_args.reserve(NC);

    const uint32_t lpt_page_bytes = in.latent_page_table.buffer()->aligned_page_size();
    const uint32_t pos_page_bytes = in.positions.buffer()->aligned_page_size();
    auto* stem_buf = in.stem.buffer();
    auto* lat_buf = in.latent_cache.buffer();
    auto* lpt_buf = in.latent_page_table.buffer();
    auto* pos_buf = in.positions.buffer();
    auto* rk_buf = in.ring_k.buffer();
    auto* rg_buf = in.ring_g.buffer();
    auto* pool_buf = in.pool_cache.buffer();
    auto* wq_buf = in.qa_norm_w.buffer();
    auto* wkva_buf = in.kva_norm_w.buffer();
    auto* wk_buf = in.k_norm_w.buffer();
    auto* bk_buf = in.k_norm_b.buffer();
    auto* ape_buf = in.ape.buffer();
    auto* sel_buf = in.selectors.buffer();
    auto* q_buf = outputs[0].buffer();

    for (uint32_t c = 0; c < NC; c++) {
        const auto& core = core_of[c];
        const uint32_t role = (c < Bt) ? 0u : 1u;
        const uint32_t idx = (c < Bt) ? c : (c - Bt);
        reader.emplace_runtime_args(
            core,
            {role, idx, stem_buf, lat_buf, lpt_buf, pos_buf, rk_buf, rg_buf, pool_buf, wq_buf, wkva_buf, wk_buf, bk_buf, ape_buf, sel_buf, lpt_page_bytes, pos_page_bytes});
        writer.emplace_runtime_args(core, {role, idx, q_buf, lat_buf, rk_buf, rg_buf, pool_buf});
        compute.emplace_runtime_args(core, {role, idx});
    }

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));
    return desc;
}

}  // namespace ttnn::prim
