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

namespace ttnn::prim {

using namespace tt::tt_metal;

namespace {

constexpr auto kPrefillKernels = "ttnn/cpp/ttnn/operations/matmul/device/kernels/gguf/";
constexpr uint32_t kPrefillGroupsPerRead = 4;
constexpr uint32_t kPrefillPageUnits = 4;
constexpr uint32_t kPrefillRecordSlots = 3;
constexpr uint32_t kPrefillInputSlots = 4;          // input blocks in flight per core (c_1)
constexpr uint32_t kPrefillSenders = 2;             // input senders per band, on different NoCs
constexpr uint32_t kPrefillDestTiles = 4;           // Float32 tiles in half of DEST
constexpr uint32_t kPrefillOutTiles = 24;           // c_16: a multiple of every subblock size, so no subblock wraps
constexpr uint32_t kPrefillL1Budget = 1250 * 1024;  // bytes of circular buffers per core
constexpr double kPrefillDecodeCycles = 300.0;      // one weight tile: page copy, decode, pack

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

// Bands of mg whole grid rows share the band's M rows; the band's cores split N, nc tiles each. A band takes its M
// tiles in chunks of mc and K in blocks of kb tiles (whole groups); the input arrives in units of ur rows x kb tiles.
// ks = 2: K split, two bands with all M rows and half of K each, band 1 sending its fp32 sums per unit of rows to
// band 0 through a two-unit ring.
struct PrefillPlan {
    uint32_t mg = 0, nc = 0, mc = 0, kb = 0, ur = 1, ks = 1;
    double cost = 1e30;
};

uint32_t prefill_fixed_l1(const GgufTraits& f) {
    const uint32_t pages_per_group = f.group_tiles / f.page_tiles;
    const uint32_t unit = pages_per_group > 1 ? pages_per_group : kPrefillGroupsPerRead;
    return kPrefillPageUnits * unit * f.l1_page_bytes() +
           (pages_per_group > 1 ? kPrefillRecordSlots * f.page_bytes() : 0) + kPrefillOutTiles * 2048;
}

// The least modeled time within the L1 budget: per core, the decodes (once per M chunk), the multiplies and a fixed
// cost per block, against the band's input stream (measured multicast rates per band count).
std::optional<PrefillPlan> prefill_plan(
    const GgufTraits& f, uint32_t m_tiles, uint32_t k_tiles, uint32_t n_tiles, CoreCoord grid) {
    const uint32_t gt = f.group_tiles, groups = k_tiles / gt;
    const uint32_t base_l1 = prefill_fixed_l1(f);
    PrefillPlan best;
    for (uint32_t ks = 1; ks <= 2; ks++) {
        if (ks == 2 && (grid.y % 2 || groups % 2)) {
            continue;
        }
        for (uint32_t mg = ks; mg <= grid.y; mg++) {
            if (grid.y % mg || (ks == 1 && m_tiles % mg) || (ks == 2 && mg != 2)) {
                continue;
            }
            const uint32_t band_cores = grid.x * (grid.y / mg), mb = ks == 2 ? m_tiles : m_tiles / mg;
            const uint32_t kt = k_tiles / ks;
            const uint32_t nc = (n_tiles + band_cores - 1) / band_cores;
            for (uint32_t mc = 1; mc <= mb; mc++) {
                if (mb % mc) {
                    continue;
                }
                const uint32_t ur = mc % 4 == 0 ? 4 : mc % 2 == 0 ? 2 : 1;
                for (uint32_t kb = gt; kb <= kt && kb <= 16; kb += gt) {
                    if (kt % kb) {
                        continue;
                    }
                    const uint32_t l1 = base_l1 + kPrefillInputSlots * ur * kb * 2048 + kb * nc * 4096 +
                                        mc * nc * 4096 + (ks == 2 ? 4 * ur * nc * 4096 : 0);
                    if (l1 > kPrefillL1Budget) {
                        continue;
                    }
                    const double blocks = double(mb / mc) * (kt / kb);
                    const double core = double(nc) * kt * (mb / mc) * kPrefillDecodeCycles +
                                        double(nc) * mb * kt * 32.0 + blocks * 2000.0 +
                                        (ks == 2 ? double(mb) * nc * 600.0 : 0.0);
                    // Input bytes per cycle per band stream: ~30 for one band, ~40 for two or five, ~70 for ten.
                    const double rate = mg == 1 ? 30.0 : mg < 10 ? 40.0 : 70.0;
                    const double input = double(m_tiles) * k_tiles * 2048 / rate;
                    const double cost = std::max(core, input) + 0.2 * std::min(core, input);
                    if (cost < best.cost) {
                        best = {mg, nc, mc, kb, ur, ks, cost};
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

void add_prefill_cb(
    Program& program,
    const CoreRangeSet& cores,
    tt::CBIndex index,
    tt::DataFormat format,
    uint32_t page_bytes,
    uint32_t pages,
    std::optional<tt::tt_metal::Tile> tile = std::nullopt) {
    CircularBufferConfig config =
        CircularBufferConfig(page_bytes * pages, {{index, format}}).set_page_size(index, page_bytes);
    if (tile) {
        config.set_tile_dims(index, *tile);
    }
    CreateCircularBuffer(program, cores, config);
}

uint32_t prefill_rows(const Tensor& input) { return input.padded_shape().volume() / input.padded_shape()[-1]; }

}  // namespace

bool gguf_prefill_supported(const Tensor& input, const Tensor& weight) {
    if (!prefill_format_supported(weight.dtype()) || prefill_rows(input) % tt::constants::TILE_HEIGHT != 0) {
        return false;
    }
    return prefill_plan(
               gguf_traits(weight.dtype()),
               prefill_rows(input) / tt::constants::TILE_HEIGHT,
               weight.logical_shape()[0] / tt::constants::TILE_WIDTH,
               weight.logical_shape()[1] / tt::constants::TILE_WIDTH,
               input.device()->compute_with_storage_grid_size())
        .has_value();
}

GgufMatmulPrefillProgramFactory::cached_program_t GgufMatmulPrefillProgramFactory::create(
    const GgufMatmulParams&, const GgufMatmulInputs& inputs, Tensor& output) {
    const Tensor& input = inputs.input;
    const Tensor& weight = inputs.weight;
    auto* device = input.device();
    const GgufTraits& f = gguf_traits(weight.dtype());
    const uint32_t k_tiles = weight.logical_shape()[0] / 32, n_tiles = weight.logical_shape()[1] / 32;
    const uint32_t m_tiles = prefill_rows(input) / 32;
    const uint32_t groups = k_tiles / f.group_tiles, pages_per_group = f.group_tiles / f.page_tiles;
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const uint32_t l1_page = f.l1_page_bytes();
    const uint32_t unit = pages_per_group > 1 ? pages_per_group : kPrefillGroupsPerRead;

    const std::optional<PrefillPlan> p = prefill_plan(f, m_tiles, k_tiles, n_tiles, grid);
    TT_FATAL(p.has_value(), "GGUF prefill matmul: no blocking fits ({} M tiles, {} K tiles)", m_tiles, k_tiles);
    const uint32_t mg = p->mg, nc = p->nc, mc = p->mc, kb = p->kb, ur = p->ur, ks = p->ks;
    const uint32_t rows = grid.y / mg, band_cores = grid.x * rows, mb = ks == 2 ? m_tiles : m_tiles / mg;
    const uint32_t kt = k_tiles / ks, mchunks = mb / mc, kblocks = kt / kb;
    log_debug(
        tt::LogOp,
        "GGUF prefill: M {} K {} N {} tiles: bands {} K split {} nc {} mc {} kb {} ur {}",
        m_tiles,
        k_tiles,
        n_tiles,
        mg,
        ks,
        nc,
        mc,
        kb,
        ur);

    Program program = CreateProgram();
    const CoreRangeSet all(CoreRange(CoreCoord(0, 0), CoreCoord(grid.x - 1, grid.y - 1)));
    // Every core of the grid has the same circular buffers, so each is at the same address on all of them.
    add_prefill_cb(program, all, tt::CBIndex::c_1, tt::DataFormat::Float16_b, 2048, kPrefillInputSlots * ur * kb);
    add_prefill_cb(program, all, tt::CBIndex::c_4, tt::DataFormat::Float32, 4096, kb * nc);
    add_prefill_cb(program, all, tt::CBIndex::c_24, tt::DataFormat::Float32, 4096, mc * nc);
    if (ks == 2) {
        add_prefill_cb(program, all, tt::CBIndex::c_17, tt::DataFormat::Float32, 4096, 2 * ur * nc);
        add_prefill_cb(program, all, tt::CBIndex::c_18, tt::DataFormat::Float32, 4096, 2 * ur * nc);
    }
    add_prefill_cb(program, all, tt::CBIndex::c_16, tt::DataFormat::Float16_b, 2048, kPrefillOutTiles);
    const tt::tt_metal::Tile page_tile(
        l1_page == 1024   ? std::array<uint32_t, 2>{16, 16}
        : l1_page == 2048 ? std::array<uint32_t, 2>{16, 32}
                          : std::array<uint32_t, 2>{32, 32});
    add_prefill_cb(
        program, all, tt::CBIndex::c_0, tt::DataFormat::UInt32, l1_page, kPrefillPageUnits * unit, page_tile);
    if (pages_per_group > 1) {
        add_prefill_cb(program, all, tt::CBIndex::c_2, tt::DataFormat::UInt32, kPrefillRecordSlots * f.page_bytes(), 1);
    }
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
    const uint32_t partial_sem = CreateSemaphore(program, all, 0);
    const uint32_t senders = std::min<uint32_t>(kPrefillSenders, grid.x);

    const bool two_in_flight = weight.dtype() == DataType::GGUF_Q4_0 || weight.dtype() == DataType::GGUF_Q4_1;
    std::vector<uint32_t> reader_ct = {
        f.format_id,
        kPrefillPageUnits * unit,
        kPrefillGroupsPerRead,
        kPrefillRecordSlots,
        request_sem,
        answer_sem,
        address_sem,
        two_in_flight ? 2u : 1u};
    TensorAccessorArgs(*weight.buffer()).append_to(reader_ct);
    std::vector<uint32_t> writer_ct = {
        k_tiles,
        n_tiles,
        mc,
        kb,
        kblocks,
        mchunks,
        kPrefillInputSlots,
        band_cores,
        ready_sems[0],
        ready_sems[1],
        ready_sems[2],
        ready_sems[3],
        count_sem0,
        count_sem1,
        source_sem,
        senders,
        ur,
        partial_sem};
    TensorAccessorArgs(*input.buffer()).append_to(writer_ct);
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct);

    // The reader generates the core's records itself; the prefill formats have no codebook lookups.
    const std::map<std::string, std::string> reader_defines = {
        {"GGUF_PREFILL", "1"},
        {"GGUF_EXPANDED", "0"},
        {"GGUF_LOOKUP_PARTS", "2"},
        {"GGUF_READER_PARTS", "1"},
        {"GGUF_WRITER_PARTS", "1"},
        {"GGUF_MATH_PARTS", "0"},
        {"GGUF_UNPACK_PARTS", "0"}};
    shared_variables_t shared;
    for (uint32_t parity = 0; parity < 2; parity++) {
        CoreRangeSet ranges;
        for (uint32_t x = parity; x < grid.x; x += 2) {
            ranges = ranges.merge(CoreRangeSet(CoreRange(CoreCoord(x, 0), CoreCoord(x, grid.y - 1))));
        }
        if (ranges.empty()) {
            continue;
        }
        shared.reader[parity] = CreateKernel(
            program,
            std::string(kPrefillKernels) + "gguf_matmul_reader.cpp",
            ranges,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = parity ? NOC::NOC_1 : NOC::NOC_0,
                .compile_args = reader_ct,
                .defines = reader_defines});
        shared.writer[parity] = CreateKernel(
            program,
            std::string(kPrefillKernels) + "gguf_matmul_prefill_writer.cpp",
            ranges,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = parity ? NOC::NOC_0 : NOC::NOC_1,
                .compile_args = writer_ct});
    }
    const KernelHandle compute_kernel = CreateKernel(
        program,
        std::string(kPrefillKernels) + "gguf_matmul_prefill_compute.cpp",
        all,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = false,
            .compile_args = {f.format_id, mc, kb, kblocks, mchunks, ur}});

    for (uint32_t band = 0; band < mg; band++) {
        const CoreCoord v0 = device->worker_core_from_logical_core(CoreCoord(0, band * rows));
        const CoreCoord v1 = device->worker_core_from_logical_core(CoreCoord(grid.x - 1, (band + 1) * rows - 1));
        // Sender s: a core of the band in a column of parity 1 - s (its writer on NoC s), preferably without N tiles.
        std::array<uint32_t, 2> sender = {band_cores, band_cores};
        for (uint32_t s = 0; s < senders; s++) {
            for (uint32_t pass = 0; pass < 2 && sender[s] == band_cores; pass++) {
                for (uint32_t jj = band_cores; jj-- > 0;) {
                    const uint32_t j = pass ? (jj + band_cores / 2) % band_cores : jj;
                    const bool idle = j * nc >= n_tiles;
                    if ((j % grid.x) % 2 == 1 - s && (pass == 1 || idle)) {
                        sender[s] = j;
                    }
                }
            }
        }
        std::array<CoreCoord, 2> sender_v;
        for (uint32_t s = 0; s < 2; s++) {
            const uint32_t j = sender[std::min(s, senders - 1)];
            sender_v[s] = device->worker_core_from_logical_core(CoreCoord(j % grid.x, band * rows + j / grid.x));
        }
        for (uint32_t j = 0; j < band_cores; j++) {
            const CoreCoord core(j % grid.x, band * rows + j / grid.x);
            const uint32_t parity = core.x % 2;
            const uint32_t n0 = std::min(n_tiles, j * nc), my_nc = std::min(n_tiles, (j + 1) * nc) - n0;
            // K split: band 1 sends to the core of band 0 with the same N tiles.
            const uint32_t krole = ks == 2 ? (band == 1 ? 1u : 2u) : 0u;
            const CoreCoord partner_v = device->worker_core_from_logical_core(
                CoreCoord(core.x, (band ^ (ks == 2 ? 1 : 0)) * rows + j / grid.x));
            // Subblock: sh x sw tiles within half of DEST, sw dividing nc, sh dividing ur; the largest.
            uint32_t sh = 1, sw = 1;
            for (uint32_t w = 1; w <= std::min(kPrefillDestTiles, std::max(my_nc, 1u)); w++) {
                for (uint32_t h = 1; h * w <= kPrefillDestTiles; h++) {
                    if (my_nc % w == 0 && ur % h == 0 && kPrefillOutTiles % (h * w) == 0 && h * w > sh * sw) {
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
                 groups / ks,
                 mchunks,
                 band % ks * (groups / ks)});
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
                band % ks * kt,
                krole,
                uint32_t(partner_v.x),
                uint32_t(partner_v.y),
                nc};
            SetRuntimeArgs(program, shared.writer[parity], core, writer_args);
            SetRuntimeArgs(program, compute_kernel, core, {my_nc, nc, sh, sw, krole});
            shared.cores.push_back(core);
        }
    }
    return {std::move(program), std::move(shared)};
}

void GgufMatmulPrefillProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program, const GgufMatmulParams&, const GgufMatmulInputs& inputs, Tensor& output) {
    const shared_variables_t& shared = cached_program.shared_variables;
    for (const CoreCoord& core : shared.cores) {
        const uint32_t parity = core.x % 2;
        GetRuntimeArgs(cached_program.program, shared.reader[parity], core)[0] = inputs.weight.buffer()->address();
        auto& writer_args = GetRuntimeArgs(cached_program.program, shared.writer[parity], core);
        writer_args[0] = inputs.input.buffer()->address();
        writer_args[1] = output.buffer()->address();
    }
}

}  // namespace ttnn::prim
