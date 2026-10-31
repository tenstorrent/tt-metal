// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.mhc_post(..., stats_weight=W): kernels/mhc_post_stats_{reader,writer,compute}.cpp.
//
// Work: column blocks of B = cols_per_block tiles (the plan, mhc_post_stats_device_operation.cpp); a unit is one block
// b of one token tile-row r, units block-major (u = b * RT + r) split contiguously over the grid, so a core keeps the
// same W block (n * B tiles) in L1 across its units. Per unit the core mixes the B column tiles of all n output
// streams (fp32 SFPU, the mhc_post arithmetic), writes them, and forms the unit's partial row: X'_block @ W_block on
// the FPU (fp32 DEST) with the block's sum of squares in column n (n + 2); the partial goes to a DRAM scratch page
// (r * NB + b) and increments the semaphore of row r's ROOT core (the owner of unit ((r mod NB) RT + r)). A root,
// after its own units, waits for NB partials per rooted row, sums them in block order (fp32 SFPU) and writes the row.

#include "mhc_post_stats_device_operation.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::bringup::mhc_post_ttnn {

namespace {

using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::DataType;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ProgramDescriptor;
using tt::tt_metal::SemaphoreDescriptor;

constexpr const char* KERNEL_DIR = "ttnn/ttnn/bringup/mhc_post_ttnn/kernels/";
constexpr uint32_t TILE_HW = 32;
constexpr size_t NUM_CIRCULAR_BUFFERS = 64;

constexpr uint8_t CB_IN = 0;       // F + X_0..X_{n-1} tiles of a unit's B columns      reader -> compute
constexpr uint8_t CB_W = 1;        // the W block (n x B tiles)                       reader -> compute
constexpr uint8_t CB_CF = 2;       // coefficient tiles of a unit (top + bottom half)  writer -> compute
constexpr uint8_t CB_RAW = 3;      // raw post / comb row tiles, writer scratch
constexpr uint8_t CB_SOA = 4;      // the unit's sum-of-squares tile, lane = token row writer -> compute
constexpr uint8_t CB_SOA_OUT = 5;  // its row totals (slot 0)                          compute -> writer
constexpr uint8_t CB_PART = 6;     // a row's partials from the scratch (root)         reader -> compute
constexpr uint8_t CB_F = 7;        // bf16 F: the unit's B F tiles (MHC_POST_STATS_F16)  reader -> compute
constexpr uint8_t CB_OUT = 16;     // X' tiles of a unit (n B)                          compute -> writer
constexpr uint8_t CB_XS = 17;      // X' tiles of the unit (matmul in0)                compute -> compute
constexpr uint8_t CB_SSC = 18;     // per-column sum of squares tiles                 compute -> compute
constexpr uint8_t CB_MFIN = 19;    // the unit's partial X' @ W tile                  compute -> writer
constexpr uint8_t CB_SFIN = 20;    // the unit's lane-wise sum-of-squares tile         compute -> writer
constexpr uint8_t CB_FIN = 21;     // a root's summed row                              compute -> writer
constexpr uint32_t SEM_ROOT = 0;

constexpr uint32_t READER_KERNEL = 0, WRITER_KERNEL = 1;

CBDescriptor make_cb(
    uint8_t index,
    uint32_t page_bytes,
    uint32_t num_pages,
    const CoreRangeSet& cores,
    DataType dt = DataType::FLOAT32) {
    CBDescriptor cb;
    cb.total_size = num_pages * page_bytes;
    cb.core_ranges = cores;
    cb.format_descriptors.push_back(CBFormatDescriptor{
        .buffer_index = index,
        .data_format = tt::tt_metal::datatype_to_dataformat_converter(dt),
        .page_size = page_bytes});
    return cb;
}

KernelDescriptor make_kernel(
    const char* file,
    const CoreRangeSet& cores,
    std::vector<uint32_t> ct,
    KernelDescriptor::RuntimeArgs rt,
    KernelDescriptor::ConfigDescriptor config) {
    KernelDescriptor k;
    k.kernel_source = std::string(KERNEL_DIR) + file;
    k.source_type = KernelDescriptor::SourceType::FILE_PATH;
    k.core_ranges = cores;
    k.compile_time_args = std::move(ct);
    k.runtime_args = std::move(rt);
    k.config = std::move(config);
    return k;
}

uint32_t addr(const Tensor& t) { return t.buffer()->address(); }

void append_accessor(std::vector<uint32_t>& ct, const Tensor& t) {
    const auto a = tt::tt_metal::TensorAccessorArgs(*t.buffer()).get_compile_time_args();
    ct.insert(ct.end(), a.begin(), a.end());
}

}  // namespace

tt::tt_metal::ProgramDescriptor MhcPostStatsProgramFactory::create_descriptor(
    const MhcPostStatsParams& params, const MhcPostStatsInputs& in, std::vector<Tensor>& outputs) {
    const Tensor& f = in.input;
    const Tensor& x = in.residual;
    const Tensor& out = outputs.at(0);
    const Tensor& stats = outputs.at(1);
    const Tensor& scratch = outputs.at(2);
    auto* device = f.device();
    const MhcPostStatsPlan p = mhc_post_stats_plan(in, params.max_grid);
    const uint32_t n = p.n, B = p.cols_per_block, NB = p.blocks, RT = p.row_tiles;
    const uint32_t terms = n * (n + 1);
    const uint32_t coef_tiles = (terms * 4 + TILE_HW - 1) / TILE_HW;  // per half (4 row blocks a half)
    const uint32_t page = TILE_HW * TILE_HW * 4;
    // P.19: a bf16 F goes through its own CB (unpacked to the fp32 DEST exactly); fp32 F: the program it always was
    const bool f16 = f.dtype() == DataType::BFLOAT16;
    const uint32_t f_page = f.buffer()->page_size();
    TT_FATAL(f_page == (f16 ? page / 2 : page), "mhc_post stats: F must be fp32 or bf16 tiles");
    // P.20: bf16 X (with bf16 F): X / X' tiles in bf16 pages (CB_IN, CB_OUT, CB_XS; define MHC_POST_STATS_X16), the
    // matmul reads the bf16 X'; fp32 X: the program it always was
    const bool x16 = x.dtype() == DataType::BFLOAT16;
    TT_FATAL(!x16 || f16, "mhc_post stats: a bf16 residual needs a bf16 F");
    const uint32_t x_page = x16 ? page / 2 : page;
    const DataType x_dt = x16 ? DataType::BFLOAT16 : DataType::FLOAT32;
    for (const Tensor* t : {&x, &out}) {
        TT_FATAL(t->buffer()->page_size() == x_page, "mhc_post stats: X / X' must be fp32 or bf16 tiles");
    }
    for (const Tensor* t : {&in.post, &in.comb, &in.weight, &stats, &scratch}) {
        TT_FATAL(t->buffer()->page_size() == page, "mhc_post stats: post / comb / W must be fp32 tiles");
    }
    TT_FATAL(coef_tiles + n + 1 <= 8, "mhc_post stats: n={} does not fit the 8 fp32 DEST tiles", n);

    const CoreCoord grid = cut_grid(device->compute_with_storage_grid_size(), params.max_grid);
    auto [num_cores, all_cores, group_1, group_2, units_1, units_2] =
        tt::tt_metal::split_work_to_cores(grid, p.units, true);
    std::vector<CoreCoord> cores;
    std::vector<uint32_t> starts, counts;
    uint32_t start = 0;
    for (const auto& [group, per_core] : {std::pair{group_1, units_1}, std::pair{group_2, units_2}}) {
        for (const auto& core : tt::tt_metal::corerange_to_cores(group, std::nullopt, true)) {
            cores.push_back(core);
            starts.push_back(start);
            counts.push_back(per_core);
            start += per_core;
        }
    }
    TT_FATAL(start == p.units, "mhc_post stats: work split covers {} of {} units", start, p.units);
    auto owner = [&](uint32_t u) {
        const auto it = std::upper_bound(starts.begin(), starts.end(), u);
        return static_cast<size_t>(std::distance(starts.begin(), it) - 1);
    };
    auto root_unit = [&](uint32_t r) { return (r % NB) * RT + r; };
    std::vector<std::vector<uint32_t>> rooted(cores.size());
    for (uint32_t r = 0; r < RT; ++r) {
        rooted[owner(root_unit(r))].push_back(r);
    }
    std::vector<uint32_t> root_xy(RT);
    for (uint32_t r = 0; r < RT; ++r) {
        const CoreCoord v = device->worker_core_from_logical_core(cores[owner(root_unit(r))]);
        root_xy[r] = (static_cast<uint32_t>(v.x) << 16) | static_cast<uint32_t>(v.y);
    }

    ProgramDescriptor desc;
    desc.cbs.push_back(make_cb(CB_IN, x_page, 2 * (n + 1) * B, all_cores, x_dt));
    desc.cbs.push_back(make_cb(CB_W, page, 2 * n * B, all_cores));
    desc.cbs.push_back(make_cb(CB_CF, page, 2 * 2 * coef_tiles, all_cores));
    desc.cbs.push_back(make_cb(CB_RAW, page, 2, all_cores));
    desc.cbs.push_back(make_cb(CB_SOA, page, 1, all_cores));
    desc.cbs.push_back(make_cb(CB_SOA_OUT, page, 1, all_cores));
    TT_FATAL(NB <= 64, "mhc_post stats: {} partials per row", NB);
    desc.cbs.push_back(make_cb(CB_PART, page, NB, all_cores));  // one row's partials
    desc.cbs.push_back(make_cb(CB_OUT, x_page, 2 * n * B, all_cores, x_dt));
    desc.cbs.push_back(make_cb(CB_XS, x_page, n * B, all_cores, x_dt));
    desc.cbs.push_back(make_cb(CB_SSC, page, B, all_cores));
    desc.cbs.push_back(make_cb(CB_MFIN, page, 2, all_cores));
    desc.cbs.push_back(make_cb(CB_SFIN, page, 2, all_cores));
    desc.cbs.push_back(make_cb(CB_FIN, page, 2, all_cores));
    if (f16) {
        desc.cbs.push_back(make_cb(CB_F, f_page, 2 * B, all_cores, DataType::BFLOAT16));
    }

    std::vector<uint32_t> common_ct = {n, p.col_tiles, RT, B, NB, page, coef_tiles};
    std::vector<uint32_t> reader_ct = common_ct;
    reader_ct.insert(reader_ct.end(), {CB_IN, CB_W, CB_PART, SEM_ROOT});
    for (const Tensor* t : {&f, &x, &in.weight, &scratch}) {
        append_accessor(reader_ct, *t);
    }
    if (f16) {  // appended only when on: the fp32-F program keeps its CT args
        reader_ct.insert(reader_ct.end(), {CB_F, f_page});
    }
    std::vector<uint32_t> writer_ct = common_ct;
    writer_ct.insert(
        writer_ct.end(),
        {CB_OUT,
         CB_CF,
         CB_RAW,
         CB_MFIN,
         CB_SFIN,
         CB_SOA,
         CB_SOA_OUT,
         CB_FIN,
         SEM_ROOT,
         n * (n + 2),
         static_cast<uint32_t>(params.comb_transposed)});
    for (const Tensor* t : {&out, &in.post, &in.comb, &scratch, &stats}) {
        append_accessor(writer_ct, *t);
    }
    std::vector<uint32_t> compute_ct = common_ct;
    compute_ct.insert(
        compute_ct.end(),
        {CB_IN, CB_W, CB_CF, CB_OUT, CB_XS, CB_SSC, CB_MFIN, CB_SFIN, CB_SOA, CB_SOA_OUT, CB_PART, CB_FIN});
    if (f16) {
        compute_ct.push_back(CB_F);
    }

    KernelDescriptor::RuntimeArgs reader_rt, writer_rt, compute_rt;
    for (size_t i = 0; i < cores.size(); ++i) {
        const uint32_t s = starts[i], c = counts[i];
        std::vector<uint32_t> r = {addr(f), addr(x), addr(in.weight), addr(scratch), s, c};
        r.push_back(static_cast<uint32_t>(rooted[i].size()));
        r.insert(r.end(), rooted[i].begin(), rooted[i].end());
        reader_rt.emplace_back(cores[i], std::move(r));
        std::vector<uint32_t> w = {addr(out), addr(in.post), addr(in.comb), addr(scratch), addr(stats), s, c};
        w.push_back(static_cast<uint32_t>(rooted[i].size()));
        w.insert(w.end(), rooted[i].begin(), rooted[i].end());
        for (uint32_t u = s; u < s + c; ++u) {
            w.push_back(root_xy[u % RT]);  // unit u's row r = u mod RT
        }
        TT_FATAL(w.size() < 300, "mhc_post stats: {} runtime args on one core", w.size());
        writer_rt.emplace_back(cores[i], std::move(w));
        compute_rt.emplace_back(cores[i], std::vector<uint32_t>{s, c, static_cast<uint32_t>(rooted[i].size())});
    }

    ComputeConfigDescriptor cc;
    cc.math_fidelity = params.compute_config.math_fidelity;
    cc.fp32_dest_acc_en = true;
    cc.math_approx_mode = false;
    cc.dst_full_sync_en = true;  // 8 fp32 DEST tiles: the coefficient tiles + F + X_0..X_{n-1}
    cc.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    for (uint8_t cb : {CB_IN, CB_CF, CB_SSC, CB_SOA, CB_PART}) {
        cc.unpack_to_dest_mode[cb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;  // exact copies
    }
    if (f16) {
        cc.unpack_to_dest_mode[CB_F] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;  // bf16 -> fp32 DEST, exact
    }
    // CB_XS / CB_W stay Default: the matmul's operands go through srcB / srcA like ttnn.matmul's fp32 inputs.

    desc.kernels.push_back(make_kernel(
        "mhc_post_stats_reader.cpp", all_cores, reader_ct, reader_rt, tt::tt_metal::ReaderConfigDescriptor{}));
    desc.kernels.push_back(make_kernel(
        "mhc_post_stats_writer.cpp", all_cores, writer_ct, writer_rt, tt::tt_metal::WriterConfigDescriptor{}));
    desc.kernels.push_back(make_kernel("mhc_post_stats_compute.cpp", all_cores, compute_ct, compute_rt, cc));
    if (f16) {  // absent for fp32 F: the reader / compute unchanged
        for (size_t k : {size_t{0}, size_t{2}}) {
            desc.kernels[k].defines.emplace_back("MHC_POST_STATS_F16", "1");
        }
    }
    if (x16) {  // absent for fp32 X: every kernel unchanged (P.20)
        for (auto& k : desc.kernels) {
            k.defines.emplace_back("MHC_POST_STATS_X16", "1");
        }
    }
    if (params.post_col != 0 || params.comb_col != 0) {  // absent when 0 / 0: the writer unchanged
        TT_FATAL(
            params.post_col + n <= TILE_HW && params.comb_col + n * n <= TILE_HW,
            "mhc_post stats: post / comb beyond one tile row");
        desc.kernels[WRITER_KERNEL].defines.emplace_back("MHC_POST_POST_COL", std::to_string(params.post_col));
        desc.kernels[WRITER_KERNEL].defines.emplace_back("MHC_POST_COMB_COL", std::to_string(params.comb_col));
    }
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = SEM_ROOT, .core_type = tt::CoreType::WORKER, .core_ranges = all_cores, .initial_value = 0});
    return desc;
}

void MhcPostStatsProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const MhcPostStatsParams& /*operation_attributes*/,
    const MhcPostStatsInputs& in,
    std::vector<Tensor>& outputs,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const uint32_t scratch = addr(outputs.at(2));
    for (auto& col : tt::tt_metal::GetRuntimeArgs(program, READER_KERNEL)) {
        for (auto& args : col) {
            if (args.size() >= 7) {
                args[0] = addr(in.input);
                args[1] = addr(in.residual);
                args[2] = addr(in.weight);
                args[3] = scratch;
            }
        }
    }
    for (auto& col : tt::tt_metal::GetRuntimeArgs(program, WRITER_KERNEL)) {
        for (auto& args : col) {
            if (args.size() >= 8) {
                args[0] = addr(outputs.at(0));
                args[1] = addr(in.post);
                args[2] = addr(in.comb);
                args[3] = scratch;
                args[4] = addr(outputs.at(1));
            }
        }
    }
}

}  // namespace ttnn::operations::bringup::mhc_post_ttnn
