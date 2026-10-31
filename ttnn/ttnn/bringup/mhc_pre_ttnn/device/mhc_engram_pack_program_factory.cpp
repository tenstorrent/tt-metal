// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// mhc_pre_xing_pack engram mode: kernels/mhc_engram_pack_{reader,writer,compute}.cpp (plan:
// mhc_engram_pack_device_operation.cpp). Per unit (stream i, column block g, token tile-row r) one DEST window
// accumulates h w k, h^2, k^2 lane-wise over the block's column tiles (fp32 SFPU); the writer re-lays the three
// accumulators coefficient-major, the compute sums their 32 column slots, the writer puts the three totals in columns
// i, n + i, 2n + i of a zero partial row tile -> scratch page r (n G) + (i G + g), then one semaphore increment on row
// r's root. A root, after its own units, sums its rows' n G partials in unit order (fp32 SFPU) -> the output row.

#include "mhc_engram_pack_device_operation.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::bringup::mhc_pre_ttnn {

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

constexpr const char* KERNEL_DIR = "ttnn/ttnn/bringup/mhc_pre_ttnn/kernels/";
constexpr uint32_t TILE_HW = 32;
constexpr size_t NUM_CIRCULAR_BUFFERS = 64;

constexpr uint8_t CB_IN = 0;        // h / k tiles of a unit's columns (h at 2c, k at 2c + 1)  reader -> compute
constexpr uint8_t CB_W = 1;         // the weight block (Bc tiles)                           reader -> compute
constexpr uint8_t CB_SOA = 2;       // the 3 accumulators, coefficient-major                 writer -> compute
constexpr uint8_t CB_PART = 3;      // a row's partials from the scratch (root)              reader -> compute
constexpr uint8_t CB_STAGE = 4;     // the partial row tile, writer scratch
constexpr uint8_t CB_ACC = 16;      // the 3 lane-wise accumulators of a unit                compute -> writer
constexpr uint8_t CB_SOA_OUT = 17;  // their row totals (slot 0 of each)                     compute -> writer
constexpr uint8_t CB_FIN = 18;      // a root's summed row                                   compute -> writer
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

tt::tt_metal::ProgramDescriptor MhcEngramPackProgramFactory::create_descriptor(
    const MhcEngramPackParams& params, const MhcEngramPackInputs& in, std::vector<Tensor>& outputs) {
    const Tensor& row = outputs.at(0);
    const Tensor& scratch = outputs.at(1);
    auto* device = in.streams.device();
    const MhcEngramPackPlan p = mhc_engram_pack_plan(params, in);
    const uint32_t n = p.n, B = p.cols_per_block, NU = p.n * p.groups, RT = p.row_tiles;
    const uint32_t page = TILE_HW * TILE_HW * 4;
    // P.20: bf16 streams and keys (CB_IN in bf16 pages; define MHC_ENGRAM_X16); fp32: the program it always was
    const bool x16 = in.streams.dtype() == DataType::BFLOAT16;
    const uint32_t in_page = x16 ? page / 2 : page;
    for (const Tensor* t : {&in.streams, &in.keys}) {
        TT_FATAL(t->buffer()->page_size() == in_page, "mhc_pre_xing_pack engram: streams / keys must share a dtype");
    }
    for (const Tensor* t : {&in.weight, &row, &scratch}) {
        TT_FATAL(t->buffer()->page_size() == page, "mhc_pre_xing_pack engram: weight / row must be fp32 tiles");
    }

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
    TT_FATAL(start == p.units, "mhc_pre_xing_pack engram: work split covers {} of {} units", start, p.units);
    auto owner = [&](uint32_t u) {
        const auto it = std::upper_bound(starts.begin(), starts.end(), u);
        return static_cast<size_t>(std::distance(starts.begin(), it) - 1);
    };
    auto root_unit = [&](uint32_t r) { return (r % NU) * RT + r; };
    std::vector<std::vector<uint32_t>> rooted(cores.size());
    std::vector<uint32_t> root_xy(RT);
    for (uint32_t r = 0; r < RT; ++r) {
        const size_t o = owner(root_unit(r));
        rooted[o].push_back(r);
        const CoreCoord v = device->worker_core_from_logical_core(cores[o]);
        root_xy[r] = (static_cast<uint32_t>(v.x) << 16) | static_cast<uint32_t>(v.y);
    }

    ProgramDescriptor desc;
    desc.cbs.push_back(make_cb(CB_IN, in_page, 2 * 2 * B, all_cores, x16 ? DataType::BFLOAT16 : DataType::FLOAT32));
    desc.cbs.push_back(make_cb(CB_W, page, 2 * B, all_cores));
    desc.cbs.push_back(make_cb(CB_SOA, page, 3, all_cores));
    TT_FATAL(NU <= 64, "mhc_pre_xing_pack engram: {} partials per row", NU);
    desc.cbs.push_back(make_cb(CB_PART, page, NU, all_cores));  // one row's partials
    desc.cbs.push_back(make_cb(CB_STAGE, page, 1, all_cores));
    desc.cbs.push_back(make_cb(CB_ACC, page, 3, all_cores));
    desc.cbs.push_back(make_cb(CB_SOA_OUT, page, 3, all_cores));
    desc.cbs.push_back(make_cb(CB_FIN, page, 2, all_cores));

    const std::vector<uint32_t> common_ct = {n, p.col_tiles, p.key_tiles, RT, B, p.groups, page};
    std::vector<uint32_t> reader_ct = common_ct;
    reader_ct.insert(reader_ct.end(), {CB_IN, CB_W, CB_PART, SEM_ROOT});
    for (const Tensor* t : {&in.streams, &in.keys, &in.weight, &scratch}) {
        append_accessor(reader_ct, *t);
    }
    std::vector<uint32_t> writer_ct = common_ct;
    writer_ct.insert(writer_ct.end(), {CB_ACC, CB_SOA, CB_SOA_OUT, CB_STAGE, CB_FIN, SEM_ROOT});
    for (const Tensor* t : {&scratch, &row}) {
        append_accessor(writer_ct, *t);
    }
    std::vector<uint32_t> compute_ct = common_ct;
    compute_ct.insert(compute_ct.end(), {CB_IN, CB_W, CB_ACC, CB_SOA, CB_SOA_OUT, CB_PART, CB_FIN});
    const bool gate = params.gate_eps.has_value();
    // P.24 gate epilogue: fp32 constants of the op chain (1 / C, eps, C^-0.5, the 1e-6 floor), as fp32 bits
    if (gate) {
        TT_FATAL(4 * n <= TILE_HW, "mhc_pre_xing_pack engram gate: 4n = {} columns exceed a tile row", 4 * n);
        const double c = static_cast<double>(p.col_tiles) * TILE_HW;  // the whole hidden (the option's contract)
        compute_ct.insert(
            compute_ct.end(),
            {std::bit_cast<uint32_t>(static_cast<float>(1.0 / c)),
             std::bit_cast<uint32_t>(*params.gate_eps),
             std::bit_cast<uint32_t>(static_cast<float>(1.0 / std::sqrt(c))),
             std::bit_cast<uint32_t>(1e-6f)});
    }

    KernelDescriptor::RuntimeArgs reader_rt, writer_rt, compute_rt;
    for (size_t i = 0; i < cores.size(); ++i) {
        const uint32_t s = starts[i], c = counts[i];
        std::vector<uint32_t> r = {
            addr(in.streams),
            addr(in.keys),
            addr(in.weight),
            addr(scratch),
            s,
            c,
            static_cast<uint32_t>(rooted[i].size())};
        r.insert(r.end(), rooted[i].begin(), rooted[i].end());
        reader_rt.emplace_back(cores[i], std::move(r));
        std::vector<uint32_t> w = {addr(scratch), addr(row), s, c, static_cast<uint32_t>(rooted[i].size())};
        w.insert(w.end(), rooted[i].begin(), rooted[i].end());
        for (uint32_t u = s; u < s + c; ++u) {
            w.push_back(root_xy[u % RT]);
        }
        TT_FATAL(w.size() < 300, "mhc_pre_xing_pack engram: {} runtime args on one core", w.size());
        writer_rt.emplace_back(cores[i], std::move(w));
        compute_rt.emplace_back(cores[i], std::vector<uint32_t>{s, c, static_cast<uint32_t>(rooted[i].size())});
    }

    ComputeConfigDescriptor cc;
    cc.math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
    cc.fp32_dest_acc_en = true;
    cc.math_approx_mode = false;
    cc.dst_full_sync_en = true;  // 3 accumulators + h, k, w in the 8 fp32 DEST tiles
    cc.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    for (uint8_t cb : {CB_IN, CB_W, CB_SOA, CB_PART}) {
        cc.unpack_to_dest_mode[cb] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;  // exact copies
    }

    desc.kernels.push_back(make_kernel(
        "mhc_engram_pack_reader.cpp", all_cores, reader_ct, reader_rt, tt::tt_metal::ReaderConfigDescriptor{}));
    desc.kernels.push_back(make_kernel(
        "mhc_engram_pack_writer.cpp", all_cores, writer_ct, writer_rt, tt::tt_metal::WriterConfigDescriptor{}));
    desc.kernels.push_back(make_kernel("mhc_engram_pack_compute.cpp", all_cores, compute_ct, compute_rt, cc));
    // P.24: the writer lays partial rows coefficient-major, the compute appends the gates (absent when off)
    if (gate) {
        desc.kernels[WRITER_KERNEL].defines.emplace_back("MHC_ENGRAM_GATE", "1");
        desc.kernels.back().defines.emplace_back("MHC_ENGRAM_GATE", "1");
    }
    if (x16) {  // absent for fp32: the reader / compute unchanged (P.20)
        for (auto& k : desc.kernels) {
            if (&k != &desc.kernels[WRITER_KERNEL]) {
                k.defines.emplace_back("MHC_ENGRAM_X16", "1");
            }
        }
    }
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = SEM_ROOT, .core_type = tt::CoreType::WORKER, .core_ranges = all_cores, .initial_value = 0});
    return desc;
}

void MhcEngramPackProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const MhcEngramPackParams& /*operation_attributes*/,
    const MhcEngramPackInputs& in,
    std::vector<Tensor>& outputs,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    for (auto& col : tt::tt_metal::GetRuntimeArgs(program, READER_KERNEL)) {
        for (auto& args : col) {
            if (args.size() >= 7) {
                args[0] = addr(in.streams);
                args[1] = addr(in.keys);
                args[2] = addr(in.weight);
                args[3] = addr(outputs.at(1));
            }
        }
    }
    for (auto& col : tt::tt_metal::GetRuntimeArgs(program, WRITER_KERNEL)) {
        for (auto& args : col) {
            if (args.size() >= 5) {
                args[0] = addr(outputs.at(1));
                args[1] = addr(outputs.at(0));
            }
        }
    }
}

}  // namespace ttnn::operations::bringup::mhc_pre_ttnn
