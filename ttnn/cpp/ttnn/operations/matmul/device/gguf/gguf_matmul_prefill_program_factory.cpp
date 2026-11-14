// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/gguf/gguf_matmul_program_factory.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <string>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/matmul/device/gguf/gguf_matmul_program_common.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace gguf_program;

namespace {

constexpr uint32_t kPrefillInputSlots = 4;          // input blocks in flight per core (c_1)
constexpr uint32_t kPrefillSenders = 2;             // input senders per band, on different NoCs
constexpr uint32_t kPrefillDestTiles = 4;           // Float32 tiles in half of DEST
constexpr uint32_t kPrefillOutTiles = 24;           // c_16: a multiple of every subblock size, so no subblock wraps
constexpr uint32_t kPrefillL1Budget = 1250 * 1024;  // bytes of circular buffers per core, at most
constexpr double kPrefillDecodeCycles = 300.0;      // one weight tile: page copy, decode, pack
constexpr double kPrefillSumCycles = 150.0;         // one tile of a K split's sums: send, add

// The formats with a decoder of fixed per-tile cost and no codebook lookups (the expanded layouts are not among them).
bool prefill_format_supported(DataType dtype) {
    switch (dtype) {
        case DataType::GGUF_Q8_0:
        case DataType::GGUF_Q4_0:
        case DataType::GGUF_Q4_1:
        case DataType::GGUF_Q5_0:
        case DataType::GGUF_Q5_1:
        case DataType::GGUF_Q4_K:
        case DataType::GGUF_Q6_K: return true;
        default: return false;
    }
}

// Bands of whole grid rows. Without a K split (ks = 1): mg bands of equal rows, each with its share of the M rows and
// all of K. With a K split (ks = 2): two bands with all M rows, band 0 of rows0 grid rows taking K tiles [0, kt0) and
// band 1 the rest; band 1 sends its fp32 sums of each N tile, per unit of rows, to the core of band 0 that holds the
// tile. A band's cores split N, nc[band] tiles each: bands of unequal rows with unequal K shares even out the cores'
// work when the N tiles do not split evenly over one band. A band takes M in chunks of mc tiles and K in blocks of kb
// tiles (whole groups); the input arrives in units of ur rows x kb tiles. N passes (np > 0, no K split): a core takes
// its N tiles np at a time, and its band's input streams once per pass.
struct PrefillPlan {
    uint32_t mg = 0, ks = 1, rows0 = 0, kt0 = 0, mc = 0, kb = 0, ur = 1;
    std::array<uint32_t, 2> nc{};
    uint32_t np = 0;
    double cost = 1e30;
};

// K split: every core sends to or receives from at most two cores of the other band.
bool prefill_partners_fit(uint32_t n_tiles, uint32_t nc0, uint32_t nc1) {
    for (const auto& [mine, other] : {std::pair{nc0, nc1}, std::pair{nc1, nc0}}) {
        for (uint32_t first = 0; first < n_tiles; first += mine) {
            const uint32_t last = std::min(n_tiles, first + mine) - 1;
            if (last / other - first / other > 1) {
                return false;
            }
        }
    }
    return true;
}

uint32_t prefill_fixed_l1(const GgufTraits& f) { return weight_reader_l1_bytes(f) + kPrefillOutTiles * 2048; }

// The least modeled time within the L1 budget (kPrefillL1Budget or the L1 left to programs, the less): per core, the
// decodes (once per M chunk), the multiplies, a fixed cost per block and, on band 0 of a K split, the sums' exchange,
// against the band's input stream (measured multicast rates per band count). N passes only where no blocking takes a
// core's N tiles at once.
std::optional<PrefillPlan> prefill_plan(
    const GgufTraits& f, uint32_t m_tiles, uint32_t k_tiles, uint32_t n_tiles, CoreCoord grid, uint32_t l1_budget) {
    const uint32_t gt = f.group_tiles;
    const uint32_t base_l1 = prefill_fixed_l1(f);
    PrefillPlan best;
    auto core_cycles = [](uint32_t nc, uint32_t kt, uint32_t mb, uint32_t mc, uint32_t kb) {
        return double(nc) * kt * (mb / mc) * kPrefillDecodeCycles + double(nc) * mb * kt * 32.0 +
               double(mb / mc) * (kt / kb) * 2000.0;
    };
    auto consider = [&](PrefillPlan plan, double core) {
        // Input bytes per cycle per band stream: ~30 for one band, ~40 for two or five, ~70 for ten.
        const double rate = plan.mg == 1 ? 30.0 : plan.mg < 10 ? 40.0 : 70.0;
        const uint32_t passes = plan.np > 0 ? (plan.nc[0] + plan.np - 1) / plan.np : 1;
        const double input = double(m_tiles) * k_tiles * 2048 * passes / rate;
        plan.cost = std::max(core, input) + 0.2 * std::min(core, input);
        if (plan.cost < best.cost) {
            best = plan;
        }
    };
    auto fits = [&](uint32_t mc, uint32_t ur, uint32_t kb, uint32_t nc, bool ksplit) {
        return base_l1 + kPrefillInputSlots * ur * kb * 2048 + kb * nc * 4096 + mc * nc * 4096 +
                   (ksplit ? 4 * ur * nc * 4096 : 0) <=
               l1_budget;
    };
    for (uint32_t mg = 1; mg <= grid.y; mg++) {
        if (grid.y % mg || m_tiles % mg) {
            continue;
        }
        const uint32_t nc = (n_tiles + grid.x * (grid.y / mg) - 1) / (grid.x * (grid.y / mg)), mb = m_tiles / mg;
        for (uint32_t mc = 1; mc <= mb; mc++) {
            const uint32_t ur = mc % 4 == 0 ? 4 : mc % 2 == 0 ? 2 : 1;
            for (uint32_t kb = gt; mb % mc == 0 && kb <= k_tiles && kb <= 16; kb += gt) {
                if (k_tiles % kb == 0 && fits(mc, ur, kb, nc, false)) {
                    consider({mg, 1, 0, 0, mc, kb, ur, {nc, nc}}, core_cycles(nc, k_tiles, mb, mc, kb));
                }
            }
        }
    }
    for (uint32_t rows0 = 1; rows0 < grid.y; rows0++) {
        const uint32_t nc0 = (n_tiles + grid.x * rows0 - 1) / (grid.x * rows0);
        const uint32_t nc1 = (n_tiles + grid.x * (grid.y - rows0) - 1) / (grid.x * (grid.y - rows0));
        // Band 0, which adds the sums, takes the fewer N tiles per core (measured faster than the reverse).
        if (nc0 > nc1 || !prefill_partners_fit(n_tiles, nc0, nc1)) {
            continue;
        }
        const uint32_t nc = std::max(nc0, nc1), mb = m_tiles;
        for (uint32_t mc = 1; mc <= mb; mc++) {
            const uint32_t ur = mc % 4 == 0 ? 4 : mc % 2 == 0 ? 2 : 1;
            for (uint32_t kb = gt; mb % mc == 0 && 2 * kb <= k_tiles && kb <= 16; kb += gt) {
                if (k_tiles % kb || !fits(mc, ur, kb, nc, true)) {
                    continue;
                }
                // Band 0's K share: the multiples of kb next to the share that evens out the two bands' work.
                const uint32_t even = uint32_t(double(k_tiles) * nc1 / (nc0 + nc1) / kb) * kb;
                for (const uint32_t kt0 : {even, even + kb}) {
                    if (kt0 == 0 || kt0 >= k_tiles) {
                        continue;
                    }
                    const double core = std::max(
                        core_cycles(nc0, kt0, mb, mc, kb) + double(mb) * nc0 * kPrefillSumCycles,
                        core_cycles(nc1, k_tiles - kt0, mb, mc, kb));
                    consider({2, 2, rows0, kt0, mc, kb, ur, {nc0, nc1}}, core);
                }
            }
        }
    }
    const bool one_pass = best.mg != 0;
    for (uint32_t mg = 1; !one_pass && mg <= grid.y; mg++) {
        if (grid.y % mg || m_tiles % mg) {
            continue;
        }
        const uint32_t nc = (n_tiles + grid.x * (grid.y / mg) - 1) / (grid.x * (grid.y / mg)), mb = m_tiles / mg;
        for (uint32_t np = 1; np < nc; np++) {
            const uint32_t passes = (nc + np - 1) / np;
            for (uint32_t mc = 1; mc <= mb; mc++) {
                const uint32_t ur = mc % 4 == 0 ? 4 : mc % 2 == 0 ? 2 : 1;
                for (uint32_t kb = gt; mb % mc == 0 && kb <= k_tiles && kb <= 16; kb += gt) {
                    if (k_tiles % kb == 0 && fits(mc, ur, kb, np, false)) {
                        consider(
                            {mg, 1, 0, 0, mc, kb, ur, {nc, nc}, np},
                            core_cycles(nc, k_tiles, mb, mc, kb) + (passes - 1) * (mb / mc) * (k_tiles / kb) * 2000.0);
                    }
                }
            }
        }
    }
    if (best.mg == 0) {
        return std::nullopt;
    }
    return best;
}

uint32_t prefill_rows(const Tensor& input) { return input.padded_shape().volume() / input.padded_shape()[-1]; }

std::optional<PrefillPlan> prefill_plan(const Tensor& input, const Tensor& weight) {
    if (!prefill_format_supported(weight.dtype()) || prefill_rows(input) % tt::constants::TILE_HEIGHT != 0) {
        return std::nullopt;
    }
    return prefill_plan(
        gguf_traits(weight.dtype()),
        prefill_rows(input) / tt::constants::TILE_HEIGHT,
        weight.logical_shape()[0] / tt::constants::TILE_WIDTH,
        weight.logical_shape()[1] / tt::constants::TILE_WIDTH,
        input.device()->compute_with_storage_grid_size(),
        std::min(kPrefillL1Budget, program_l1_bytes(input.device())));
}

}  // namespace

bool gguf_prefill_supported(const Tensor& input, const Tensor& weight) {
    return prefill_plan(input, weight).has_value();
}

std::vector<uint32_t> gguf_prefill_plan_key(const Tensor& input, const Tensor& weight) {
    const std::optional<PrefillPlan> p = prefill_plan(input, weight);
    if (!p.has_value()) {
        return {};
    }
    return {p->mg, p->ks, p->rows0, p->kt0, p->mc, p->kb, p->ur, p->nc[0], p->nc[1], p->np};
}

GgufMatmulPrefillProgramFactory::cached_program_t GgufMatmulPrefillProgramFactory::create(
    const GgufMatmulParams&, const GgufMatmulInputs& inputs, Tensor& output) {
    const Tensor& input = inputs.input;
    const Tensor& weight = inputs.weight;
    auto* device = input.device();
    const GgufTraits& f = gguf_traits(weight.dtype());
    const uint32_t k_tiles = weight.logical_shape()[0] / 32, n_tiles = weight.logical_shape()[1] / 32;
    const uint32_t m_tiles = prefill_rows(input) / 32;
    const CoreCoord grid = device->compute_with_storage_grid_size();

    const std::optional<PrefillPlan> p = prefill_plan(input, weight);
    TT_FATAL(
        p.has_value(),
        "GGUF prefill matmul: no blocking fits the {} bytes of L1 left to programs ({} M tiles, {} K tiles)",
        program_l1_bytes(device),
        m_tiles,
        k_tiles);
    const uint32_t mg = p->mg, mc = p->mc, kb = p->kb, ur = p->ur, ks = p->ks;
    const uint32_t nc = p->np > 0 ? p->np : std::max(p->nc[0], p->nc[1]);  // N tiles the circular buffers are sized for
    const uint32_t passes = p->np > 0 ? (p->nc[0] + p->np - 1) / p->np : 1;
    const uint32_t mb = ks == 2 ? m_tiles : m_tiles / mg, mchunks = mb / mc;
    struct Band {
        uint32_t row0, rows, nc, k0, kt;
    };
    std::vector<Band> bands;
    if (ks == 2) {
        bands = {{0, p->rows0, p->nc[0], 0, p->kt0}, {p->rows0, grid.y - p->rows0, p->nc[1], p->kt0, k_tiles - p->kt0}};
    } else {
        for (uint32_t band = 0; band < mg; band++) {
            bands.push_back({band * (grid.y / mg), grid.y / mg, p->nc[0], 0, k_tiles});
        }
    }
    log_debug(
        tt::LogOp,
        "GGUF prefill: M {} K {} N {} tiles: bands {} K split {} (band 0: {} rows, {} K tiles) nc {} / {} passes {} mc "
        "{} kb {} ur {}",
        m_tiles,
        k_tiles,
        n_tiles,
        mg,
        ks,
        bands[0].rows,
        bands[0].kt,
        p->nc[0],
        p->nc[1],
        passes,
        mc,
        kb,
        ur);

    Program program = CreateProgram();
    const CoreRangeSet all(CoreRange(CoreCoord(0, 0), CoreCoord(grid.x - 1, grid.y - 1)));
    // Every core of the grid has the same circular buffers, so each is at the same address on all of them.
    add_cb(program, all, tt::CBIndex::c_1, tt::DataFormat::Float16_b, 2048, kPrefillInputSlots * ur * kb);
    add_cb(program, all, tt::CBIndex::c_4, tt::DataFormat::Float32, 4096, kb * nc);
    add_cb(program, all, tt::CBIndex::c_24, tt::DataFormat::Float32, 4096, mc * nc);
    if (ks == 2) {
        add_cb(program, all, tt::CBIndex::c_17, tt::DataFormat::Float32, 4096, 2 * ur * nc);
        add_cb(program, all, tt::CBIndex::c_18, tt::DataFormat::Float32, 4096, 2 * ur * nc);
    }
    add_cb(program, all, tt::CBIndex::c_16, tt::DataFormat::Float16_b, 2048, kPrefillOutTiles);
    add_weight_reader_cbs(program, all, f);
    const uint32_t request_sem = CreateSemaphore(program, all, 0);
    const uint32_t answer_sem = CreateSemaphore(program, all, 0);
    const uint32_t address_sem = CreateSemaphore(program, all, 0);
    std::array<uint32_t, kPrefillInputSlots> ready_sems{};
    for (auto& sem : ready_sems) {
        sem = CreateSemaphore(program, all, 0);
    }
    const uint32_t count_sem0 = CreateSemaphore(program, all, 0);
    const uint32_t count_sem1 = CreateSemaphore(program, all, 0);
    const uint32_t source_sem = CreateSemaphore(program, all, 0);
    const uint32_t partial_sem0 = CreateSemaphore(program, all, 0);
    const uint32_t partial_sem1 = CreateSemaphore(program, all, 0);
    const uint32_t senders = std::min<uint32_t>(kPrefillSenders, grid.x);

    const std::vector<uint32_t> reader_ct = weight_reader_args(weight, request_sem, answer_sem, address_sem);
    std::vector<uint32_t> writer_ct = {
        k_tiles,
        n_tiles,
        mc,
        kb,
        mchunks,
        kPrefillInputSlots,
        ready_sems[0],
        ready_sems[1],
        ready_sems[2],
        ready_sems[3],
        count_sem0,
        count_sem1,
        source_sem,
        senders,
        ur,
        partial_sem0,
        partial_sem1,
        nc,
        passes};
    TensorAccessorArgs(*input.buffer()).append_to(writer_ct);
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct);

    // The reader generates the core's records itself; the prefill formats have no codebook lookups.
    std::map<std::string, std::string> reader_defines = lookup_defines(weight);
    reader_defines["GGUF_PREFILL"] = "1";
    shared_variables_t shared;
    for (uint32_t parity = 0; parity < 2; parity++) {
        CoreRangeSet ranges;
        for (uint32_t x = parity; x < grid.x; x += 2) {
            ranges = ranges.merge(CoreRangeSet(CoreRange(CoreCoord(x, 0), CoreCoord(x, grid.y - 1))));
        }
        if (ranges.empty()) {
            continue;
        }
        add_movers(
            program,
            ranges,
            parity,
            "gguf_matmul_reader.cpp",
            "gguf_matmul_prefill_writer.cpp",
            reader_ct,
            writer_ct,
            reader_defines,
            {},
            shared);
    }
    const KernelHandle compute_kernel = CreateKernel(
        program,
        std::string(kKernels) + "gguf_matmul_prefill_compute.cpp",
        all,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = false,
            .compile_args = {f.format_id, mc, kb, mchunks, ur, passes, kPrefillOutTiles}});

    auto band_core = [&](uint32_t band, uint32_t j) { return CoreCoord(j % grid.x, bands[band].row0 + j / grid.x); };
    for (uint32_t band = 0; band < bands.size(); band++) {
        const Band& bd = bands[band];
        const uint32_t band_cores = grid.x * bd.rows, bnc = bd.nc;
        const CoreCoord v0 = device->worker_core_from_logical_core(CoreCoord(0, bd.row0));
        const CoreCoord v1 = device->worker_core_from_logical_core(CoreCoord(grid.x - 1, bd.row0 + bd.rows - 1));
        // Sender s: a core of the band in a column of parity 1 - s (its writer on NoC s), preferably without N tiles.
        std::array<uint32_t, 2> sender = {band_cores, band_cores};
        for (uint32_t s = 0; s < senders; s++) {
            for (uint32_t pass = 0; pass < 2 && sender[s] == band_cores; pass++) {
                for (uint32_t jj = band_cores; jj-- > 0;) {
                    const uint32_t j = pass ? (jj + band_cores / 2) % band_cores : jj;
                    const bool idle = j * bnc >= n_tiles;
                    if ((j % grid.x) % 2 == 1 - s && (pass == 1 || idle)) {
                        sender[s] = j;
                    }
                }
            }
        }
        std::array<CoreCoord, 2> sender_v;
        for (uint32_t s = 0; s < 2; s++) {
            sender_v[s] = device->worker_core_from_logical_core(band_core(band, sender[std::min(s, senders - 1)]));
        }
        for (uint32_t j = 0; j < band_cores; j++) {
            const CoreCoord core = band_core(band, j);
            const uint32_t parity = core.x % 2;
            const uint32_t n0 = std::min(n_tiles, j * bnc), my_nc = std::min(n_tiles, (j + 1) * bnc) - n0;
            // K split: band 1 (krole 1) sends the sums of its N tiles to the cores of band 0 (krole 2) holding them,
            // at most two either way. Per partner: NoC x, y, this core's index among the partner's partners, and for a
            // sender the tile offsets in its unit of sums and in the partner's, and the tiles.
            const uint32_t krole = ks == 2 && my_nc > 0 ? (band == 1 ? 1u : 2u) : 0u;
            std::vector<uint32_t> partners;
            if (krole != 0) {
                const uint32_t other = 1 - band, onc = bands[other].nc;
                for (uint32_t o = n0 / onc; o <= (n0 + my_nc - 1) / onc; o++) {
                    const uint32_t lo = std::max(n0, o * onc), hi = std::min(n0 + my_nc, (o + 1) * onc);
                    const CoreCoord ov = device->worker_core_from_logical_core(band_core(other, o));
                    partners.insert(
                        partners.end(),
                        {uint32_t(ov.x),
                         uint32_t(ov.y),
                         j - (o * onc) / bnc,
                         (lo - n0) * ur,
                         (lo - o * onc) * ur,
                         (hi - lo) * ur});
                }
            }
            // Subblock: sh x sw tiles within half of DEST, sw dividing the N tiles of each pass, sh dividing ur; the
            // largest. With a K split, ur x 1, so that a unit of sums holds each N tile's rows together.
            uint32_t sh = ks == 2 ? ur : 1, sw = 1;
            for (uint32_t w = 1; ks == 1 && w <= std::min(kPrefillDestTiles, std::max(my_nc, 1u)); w++) {
                for (uint32_t h = 1; h * w <= kPrefillDestTiles; h++) {
                    if (my_nc % w == 0 && (passes == 1 || nc % w == 0) && ur % h == 0 &&
                        kPrefillOutTiles % (h * w) == 0 && h * w > sh * sw) {
                        sh = h;
                        sw = w;
                    }
                }
            }
            SetRuntimeArgs(
                program,
                shared.reader[parity],
                core,
                {uint32_t(weight.buffer()->address()),
                 n_tiles,
                 n0,
                 my_nc,
                 bd.kt / f.group_tiles,
                 mchunks,
                 bd.k0 / f.group_tiles,
                 nc});
            std::vector<uint32_t> writer_args = {
                uint32_t(input.buffer()->address()),
                uint32_t(output.buffer()->address()),
                parity ? uint32_t(v0.x) : uint32_t(v1.x),
                parity ? uint32_t(v0.y) : uint32_t(v1.y),
                parity ? uint32_t(v1.x) : uint32_t(v0.x),
                parity ? uint32_t(v1.y) : uint32_t(v0.y),
                ks == 2 ? 0u : band * mb,
                n0,
                my_nc,
                sh,
                sw,
                j == sender[0]                    ? 1u
                : (senders > 1 && j == sender[1]) ? 2u
                                                  : 0u,
                uint32_t(sender_v[0].x),
                uint32_t(sender_v[0].y),
                uint32_t(sender_v[1].x),
                uint32_t(sender_v[1].y),
                bd.k0,
                krole,
                nc,
                bd.kt / kb,
                band_cores,
                uint32_t(partners.size() / 6)};
            writer_args.insert(writer_args.end(), partners.begin(), partners.end());
            SetRuntimeArgs(program, shared.writer[parity], core, writer_args);
            SetRuntimeArgs(program, compute_kernel, core, {my_nc, nc, sh, sw, krole, bd.kt / kb});
            shared.cores.push_back(core);
        }
    }
    return {std::move(program), std::move(shared)};
}

void GgufMatmulPrefillProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program, const GgufMatmulParams&, const GgufMatmulInputs& inputs, Tensor& output) {
    override_addresses(cached_program, inputs, output, 0);
}

}  // namespace ttnn::prim
