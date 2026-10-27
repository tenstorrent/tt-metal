// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_plan.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <tuple>

#include <fmt/format.h>
#include <tt_stl/indestructible.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "ttnn/operations/ccl/all_gather/shared_with_host/all_gather_walk.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::ccl::all_gather_plan {

namespace {

using tt::tt_metal::NOC;

// ---- tuning ----
// Measured on BH 8x p150b, mesh (1, 8), 2 links, payload 15168: bf16 tiles gathered on the last dim at 64 KB - 42 MB,
// row-major pages of 256 B - 16 KB, plus other dtypes and layouts held out of the fit.
constexpr Tuning kBlackhole{
    .arch = tt::ARCH::BLACKHOLE,
    .payload = 15168,
    .links = 2,
    // Least (T - B / BW) over 64 KB - 4 MB, init handshake included. Lower pulls small sizes toward that algorithm.
    .fixed_us = {{{20.5, 15.0}, {16.0, 12.0}}},
    // Slope over 12 - 42 MB. Higher pulls large sizes toward that algorithm.
    .bw_gbps = {{{44.0, 38.0}, {37.0, 25.5}}},
    // One DRAM bank sustains about 64 GB/s (ND-sharded output with one bank per stripe); L1 is charged the same.
    .bank_gbps = 64.0,
    // Ring: 3 workers are 1.4-7% faster than 2 at 256 KB - 9 MB (2 workers cannot keep up with the link).
    .ring_n3 = true,
    // Line: 2 workers are 10% faster at 121 packets, 3 as fast or faster from 242.
    .n3_min_packets = 160,
    // Two full packets per read barrier: 24% less worker time per packet than one.
    .slot_read_bytes = 2 * 15168,
    // Each signal is a packet: fewer save link time, more start the downstream relay sooner. 1-packet cadence is up to
    // 1.5x slower at 8-26 MB; 2-3 is best at 2 MB; a 16-packet cap is 1.7% faster than 8 at 26 MB.
    .signals_per_iter = 4,
    .cadence_min_packets = 2,
    .cadence_max_packets = 16,
    // 1 worker with 2 buffers is 12-14% slower than with 4; from 2 workers 2 buffers are within 3% of 4.
    .mux_buffers = {4, 2, 2},
    // The forwarder writes every packet into an ethernet core, 1.8-2.3x slower on NOC1.
    .mux_forwarder_noc = NOC::NOC_0,
    // Writers on NOC0: multicast 1.42x (ring) / 1.13x (line) faster; unicast 5-10% faster on a ring, 3-8% slower on
    // a line.
    .writer_noc = {{{NOC::NOC_1, NOC::NOC_0}, {NOC::NOC_0, NOC::NOC_0}}},
};
constexpr std::array kTunings = {kBlackhole};

const char* algo_name(Algo algo) {
    switch (algo) {
        case Algo::Unicast: return "unicast";
        case Algo::Multicast: return "multicast";
        case Algo::Composite: return "composite";
    }
    TT_THROW("Unreachable");
}

// make_plan runs on every call, so a given shortfall is warned about once per process, not once per call.
void warn_once(const std::string& message) {
    struct Warned {
        std::mutex mutex;
        std::set<std::string> messages;
    };
    static ttsl::Indestructible<Warned> warned;
    const std::lock_guard lock(warned.get().mutex);
    if (warned.get().messages.insert(message).second) {
        log_warning(tt::LogOp, "{}", message);
    }
}

// ---- facts ----

// NOC_{L1,DRAM}_WRITE_ALIGNMENT_BYTES on Wormhole and Blackhole; no public host API exposes it.
constexpr uint32_t kNocWriteAlignment = 16;

// Fabric packet-header pool entries per data-movement RISC (NUM_PACKET_HEADERS / MaxDMProcessorsPerCoreType in the
// arch's dev_mem_map.h); no public host API exposes it. An unknown arch gets the smallest known pool, so the multicast
// header rule errs toward illegal.
uint32_t headers_per_risc(tt::ARCH arch) { return arch == tt::ARCH::BLACKHOLE ? 12 : 8; }

// Exclusive end of a Mux V2 L1 map with n channels. Memoized: the config walks the fabric context, and the plan runs
// on every call. No ceiling is passed: legality compares the end with l1_small_floor itself, so an n that does not fit
// is a verdict rather than the config's hard error.
uint32_t mux_map_end(
    const tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::ARCH arch,
    uint32_t n,
    uint32_t channel_bytes,
    uint32_t l1_base) {
    using Key = std::tuple<tt::ARCH, tt::tt_fabric::FabricConfig, uint32_t, uint32_t, uint32_t, uint32_t>;
    struct Memo {
        std::mutex mutex;
        std::map<Key, uint32_t> ends;
    };
    static ttsl::Indestructible<Memo> memo;
    const uint32_t buffers = mux_buffers(arch, n);
    const Key key{arch, tt::tt_fabric::GetFabricConfig(), n, buffers, channel_bytes, l1_base};
    const std::lock_guard lock(memo.get().mutex);
    auto& ends = memo.get().ends;
    if (const auto it = ends.find(key); it != ends.end()) {
        return it->second;
    }
    const tt::tt_fabric::FabricMuxV2Config config(
        mesh_device, static_cast<uint8_t>(n), static_cast<uint8_t>(buffers), channel_bytes, l1_base);
    return ends.emplace(key, static_cast<uint32_t>(config.get_memory_map_end_address())).first->second;
}

// ---- geometry helpers ----

// L1 a worker's slot CB may use: everything between the allocator base and the L1_SMALL floor (static limits only;
// live allocator state would break the program cache).
uint32_t l1_budget(const Facts& f) { return f.l1_small_floor > f.l1_base ? f.l1_small_floor - f.l1_base : 0; }

uint32_t slot_bytes(uint32_t k, const Geometry& geo, const Facts& f) {
    return ag::slot_offsets(k, geo.unit_bytes, f.header_bytes, geo.packet_cap).slot_bytes;
}

// Input pages along [gather dim .. last dim], i.e. what one device contributes to one output row. 0 when a row-major
// row is not a whole number of pages: the walk cannot place them.
uint32_t pages_per_stripe_row(const tt::tt_metal::TensorSpec& in, int32_t dim_from_end, uint32_t page_bytes) {
    const auto& shape = in.padded_shape();
    const bool tiled = in.layout() == tt::tt_metal::Layout::TILE;
    const auto tile = in.tile();
    uint64_t pages = 1;
    for (int32_t i = dim_from_end; i < 0; i++) {
        if (i == -1) {
            if (tiled) {
                pages *= shape[i] / tile.get_width();
                continue;
            }
            // A row-major row can span several pages (width sharding), so count pages, not elements.
            const uint64_t row_bytes =
                uint64_t{shape[i]} * tt::datum_size(tt::tt_metal::datatype_to_dataformat_converter(in.data_type()));
            if (row_bytes % page_bytes != 0) {
                return 0;
            }
            pages *= row_bytes / page_bytes;
        } else if (tiled && i == -2) {
            pages *= shape[i] / tile.get_height();
        } else {
            pages *= shape[i];
        }
    }
    return pages > std::numeric_limits<uint32_t>::max() ? 0 : static_cast<uint32_t>(pages);
}

// The shape rules no native algorithm handles, then the page-grid conditions the walk adds. Empty: a native algorithm
// may run.
std::string shape_reason(
    const AllGatherParams& args, const tt::tt_metal::TensorSpec& in, const tt::tt_metal::TensorSpec& out) {
    if (auto padded = gather_dim_padding_reason(in, args.dim_from_end)) {
        return *padded;
    }
    if (uint64_t{in.padded_shape()[args.dim_from_end]} * args.num_devices > std::numeric_limits<uint32_t>::max()) {
        return "the gathered dim does not fit in 32 bits";
    }
    // The kernel walks the output page grid as the input's with only the gather dim scaled, so any other difference
    // breaks it -- e.g. an output shard width that doesn't divide the row, which pads the last dim.
    auto gathered_padded_shape = in.padded_shape();
    gathered_padded_shape[args.dim_from_end] *= args.num_devices;
    if (out.padded_shape() != gathered_padded_shape) {
        return fmt::format(
            "output stored as {} instead of {}; its shard shape must divide the gathered shape evenly",
            out.padded_shape(),
            gathered_padded_shape);
    }

    // Page sizes are UNALIGNED (content) sizes, so a matched-but-padded gather reads as matched, not concat.
    const auto& f = args.facts;
    const auto input_page_size = static_cast<uint32_t>(in.compute_page_size_bytes());
    const auto output_page_size = static_cast<uint32_t>(out.compute_page_size_bytes());
    if (input_page_size == 0 || output_page_size == 0) {
        return "the tensor has no pages";
    }
    const uint32_t input_alignment = in.memory_config().is_dram() ? f.dram_align : f.l1_align;
    const uint32_t output_alignment = out.memory_config().is_dram() ? f.dram_align : f.l1_align;
    const uint32_t input_aligned_page_size = tt::align(input_page_size, input_alignment);
    if (std::max(input_page_size, output_page_size) % std::min(input_page_size, output_page_size) != 0) {
        return fmt::format(
            "input and output rows are {} B and {} B; one shard width must be a whole multiple of the other",
            input_page_size,
            output_page_size);
    }
    // concat and split move data at content granularity, so the input page must have no padding.
    if (input_page_size != output_page_size && input_page_size != input_aligned_page_size) {
        return fmt::format(
            "input rows ({} B) are padded to the {} B memory alignment; resharding needs unpadded rows",
            input_page_size,
            input_alignment);
    }
    if (input_page_size > output_page_size && output_page_size % f.noc_write_align != 0) {
        return fmt::format(
            "output is sharded finer than the input; its rows ({} B) must be a multiple of {} B",
            output_page_size,
            f.noc_write_align);
    }
    // matched and concat write a whole *aligned* input page into each output page slot, so the slot must be at least
    // that large. They differ only across memories with unequal alignments.
    const uint32_t output_aligned_page_size = tt::align(output_page_size, output_alignment);
    if (input_page_size <= output_page_size && output_aligned_page_size < input_aligned_page_size) {
        return fmt::format(
            "input rows pad to {} B but the output reserves {} B; use the same buffer type, or a row size "
            "that is a multiple of both DRAM and L1 alignments",
            input_aligned_page_size,
            output_aligned_page_size);
    }
    // Specs pad a sharded row to whole shards, so this guards against one that does not.
    if (pages_per_stripe_row(in, args.dim_from_end, input_page_size) == 0) {
        return "a row-major row is not a whole number of pages";
    }
    return {};
}

// Multicast connections of the busiest device: one per direction with a neighbor.
uint32_t mcast_connections(const PlanIn& in) {
    uint32_t conns = 0;
    for (uint32_t a = 0; a < 2; ++a) {
        if (in.axis_devices[a] > 1) {
            conns += (in.axis_devices[a] > 2 || in.axis_ring[a]) ? 2 : 1;
        }
    }
    return conns;
}

// Packets per slot (k): about slot_read_bytes of packets (at the packet cap) per read barrier, at least kCbSlots slots
// per worker iteration so both slots work, and at most what fits the L1 budget (at least 1).
uint32_t packets_per_slot(const Tuning& t, const Facts& f, const Geometry& geo, uint32_t workers) {
    const uint64_t iter_bytes = uint64_t{tt::div_up(geo.elements, workers)} * geo.element_units * geo.unit_bytes;
    const uint64_t iter_packets = tt::div_up(iter_bytes, uint64_t{geo.packet_cap});
    uint32_t k = static_cast<uint32_t>(std::clamp<uint64_t>(
        std::min<uint64_t>(tt::div_up(t.slot_read_bytes, geo.packet_cap), iter_packets / kCbSlots),
        1,
        kMaxPacketsPerSlot));
    while (k > 1 && kCbSlots * slot_bytes(k, geo, f) > l1_budget(f)) {
        --k;
    }
    return k;
}

// The largest (links, n) of one legal algorithm that fits: n shrinks first (3 -> 2 -> 1), then links.
std::optional<Choice> fit(const PlanIn& in, Algo algo, const Geometry& geo) {
    const bool unicast = algo == Algo::Unicast;
    const Facts& f = in.facts;
    const Tuning& t = tuning(f.arch);
    const uint32_t links_want = in.links;
    const uint32_t n_want =
        unicast ? detail::model_workers(t, in.ring, detail::busiest_link_packets(geo, in.N, in.ring, links_want)) : 1;
    for (uint32_t links = links_want; links >= 1; --links) {
        for (uint32_t n = n_want; n >= 1; --n) {
            if (unicast && f.mux_map_end[n - 1] > f.l1_small_floor) {
                continue;
            }
            const bool placed = unicast ? place_unicast(f.worker_cores, links, n).has_value()
                                        : place_multicast(f.worker_cores, links).has_value();
            if (!placed) {
                continue;
            }
            if (n < n_want || links < links_want) {
                warn_once(fmt::format(
                    "all_gather {} uses {} worker(s) x {} link(s), wanted {} x {}: the worker cores (or L1 below "
                    "L1_SMALL for the mux) do not fit more. This may lead to performance loss.",
                    algo_name(algo),
                    n,
                    links,
                    n_want,
                    links_want));
            }
            return Choice{
                .algo = algo, .workers = n, .links = links, .packets_per_slot = packets_per_slot(t, f, geo, links * n)};
        }
    }
    return std::nullopt;
}

// Composite gathers through all_broadcast, which multicasts every device's input to every other device: it needs a
// second device and, past 2 devices, routes beyond the neighbor.
Plan composite(const PlanIn& in, std::string reason) {
    TT_FATAL(in.N >= 2, "all_gather needs at least 2 devices on the gathered axis, got {}", in.N);
    TT_FATAL(
        !in.neighbor_exchange || in.N <= 2,
        "all_gather cannot run across {} devices on a neighbor-exchange fabric: {}; and the composite fallback needs "
        "multi-hop multicast, which that fabric does not provide",
        in.N,
        reason);
    Plan plan;
    plan.choice = Choice{.algo = Algo::Composite};
    plan.N = in.N;
    plan.ring = in.ring;
    plan.arch = in.facts.arch;
    plan.composite_reason = std::move(reason);
    return plan;
}

// Pages::stripe_banks of an interleaved output, where page p is in bank p % banks. Device d's stripe is its W bytes of
// each N * W byte row, i.e. pages [first, first + span) of each row of row_pages pages. Row r starts at bank
// (r * row_pages) % banks, a multiple of g = gcd(row_pages, banks), and successive rows take banks / g distinct
// starts. Spans no longer than g never overlap; with every start taken, each start's g banks hold min(span, g).
uint32_t interleaved_stripe_banks(uint64_t rows, uint64_t W, uint64_t page_bytes, uint32_t N, uint32_t banks) {
    const uint64_t row_pages = (W * N) / page_bytes;
    const uint64_t g = std::gcd(row_pages, uint64_t{banks});
    const uint64_t starts = std::min(rows, banks / g);
    uint64_t fewest = banks;
    for (uint64_t d = 0; d < N; ++d) {
        const uint64_t first = d * W / page_bytes;
        const uint64_t span = (((d + 1) * W) - 1) / page_bytes - first + 1;
        uint64_t held = 0;
        if (span <= g || starts == banks / g) {
            held = starts * std::min(span, g);
        } else {
            // Fewer rows than distinct starts (rows < banks), spans that may overlap: mark them.
            std::vector<bool> bank(banks, false);
            for (uint64_t r = 0; r < rows; ++r) {
                for (uint64_t j = 0; j < std::min(span, uint64_t{banks}); ++j) {
                    bank[((r * row_pages) + first + j) % banks] = true;
                }
            }
            held = static_cast<uint64_t>(std::count(bank.begin(), bank.end(), true));
        }
        fewest = std::min(fewest, held);
    }
    return static_cast<uint32_t>(fewest);
}

// Pages::stripe_banks of a sharded output, from the distribution spec's page map (which walks every page).
uint32_t count_sharded_stripe_banks(
    const tt::tt_metal::BufferDistributionSpec& dist, uint64_t out_pages, uint64_t W, uint64_t page_bytes, uint32_t N) {
    const auto mapping = dist.compute_page_mapping();
    const size_t cores = mapping.core_host_page_indices.size();
    // held[d][c]: core c holds a byte of device d's stripe. Page p holds bytes [p, p + 1) * page_bytes of the output,
    // and byte x belongs to device (x / W) % N.
    std::vector<std::vector<bool>> held(N, std::vector<bool>(cores, false));
    for (size_t c = 0; c < cores; ++c) {
        for (const uint32_t p : mapping.core_host_page_indices[c]) {
            if (p == tt::tt_metal::UncompressedBufferPageMapping::PADDING || p >= out_pages) {
                continue;
            }
            const uint64_t x = uint64_t{p} * page_bytes;
            for (uint64_t s = x / W; s <= (x + page_bytes - 1) / W; ++s) {
                held[s % N][c] = true;
            }
        }
    }
    uint32_t fewest = static_cast<uint32_t>(cores);
    for (const auto& h : held) {
        fewest = std::min(fewest, static_cast<uint32_t>(std::count(h.begin(), h.end(), true)));
    }
    return fewest;
}

// Memoized per layout: the page map costs about 15 us per MB of 2 KB pages, and the plan runs on every call. The lock
// guards only the lookup and the insert; the memo is cleared when it reaches kMemoEntries layouts.
uint32_t sharded_stripe_banks(
    const tt::tt_metal::BufferDistributionSpec& dist, uint64_t out_pages, uint64_t W, uint64_t page_bytes, uint32_t N) {
    constexpr size_t kMemoEntries = 256;
    using Key = std::
        tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<CoreCoord>, int, uint64_t, uint64_t, uint32_t>;
    struct Memo {
        std::mutex mutex;
        std::map<Key, uint32_t> banks;
    };
    static ttsl::Indestructible<Memo> memo;
    const auto& ts = dist.tensor_shape_in_pages();
    const auto& ss = dist.shard_shape_in_pages();
    Key key{
        std::vector<uint32_t>(ts.cbegin(), ts.cend()),
        std::vector<uint32_t>(ss.cbegin(), ss.cend()),
        dist.cores(),
        static_cast<int>(dist.shard_distribution_strategy()),
        W,
        page_bytes,
        N};
    {
        const std::lock_guard lock(memo.get().mutex);
        if (const auto it = memo.get().banks.find(key); it != memo.get().banks.end()) {
            return it->second;
        }
    }
    const uint32_t banks = count_sharded_stripe_banks(dist, out_pages, W, page_bytes, N);
    const std::lock_guard lock(memo.get().mutex);
    if (memo.get().banks.size() >= kMemoEntries) {
        memo.get().banks.clear();
    }
    memo.get().banks.emplace(std::move(key), banks);
    return banks;
}

// The fewest banks holding one device's stripe; 0 (unknown) when the output's rows are not whole pages.
uint32_t stripe_banks(
    const std::optional<tt::tt_metal::BufferDistributionSpec>& dist,
    uint32_t interleaved_banks,
    uint64_t out_pages,
    uint64_t W,
    uint64_t page_bytes,
    uint32_t N) {
    if (page_bytes == 0 || W == 0 || N == 0 || (W * N) % page_bytes != 0) {
        return 0;
    }
    if (dist.has_value()) {
        return sharded_stripe_banks(*dist, out_pages, W, page_bytes, N);
    }
    return interleaved_banks == 0
               ? 0
               : interleaved_stripe_banks(out_pages / ((W * N) / page_bytes), W, page_bytes, N, interleaved_banks);
}

// Every value of the Tuning entry was measured at its arch, payload and link count; anything else borrows them.
void warn_if_uncalibrated(const PlanIn& in) {
    const Tuning& t = tuning(in.facts.arch);
    if (t.arch != in.facts.arch || t.payload != in.facts.payload || t.links != in.links) {
        warn_once(fmt::format(
            "all_gather tuning uncalibrated for arch={}, payload={}, links={}: using the {} entry (payload {}, links "
            "{}). Performance may be off; correctness is unaffected.",
            in.facts.arch,
            in.facts.payload,
            in.links,
            t.arch,
            t.payload,
            t.links));
    }
}

}  // namespace

const Tuning& tuning(tt::ARCH arch) {
    for (const Tuning& t : kTunings) {
        if (t.arch == arch) {
            return t;
        }
    }
    return kTunings.front();
}

uint32_t mux_buffers(tt::ARCH arch, uint32_t n) {
    TT_FATAL(n >= 1 && n <= kMaxWorkers, "all_gather: no Mux V2 buffer count for {} channels", n);
    return tuning(arch).mux_buffers[n - 1];
}

// ---- facts ----

Facts make_facts(
    const tt::tt_metal::distributed::MeshDevice& mesh_device,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    const auto& allocator = mesh_device.allocator();
    Facts f;
    f.arch = tt::tt_metal::hal::get_arch();
    f.payload = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());
    f.header_bytes = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes());
    f.channel_bytes = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes());
    f.dram_align = tt::tt_metal::hal::get_dram_alignment();
    f.l1_align = tt::tt_metal::hal::get_l1_alignment();
    f.noc_write_align = kNocWriteAlignment;
    f.max_burst = tt::tt_metal::hal::get_noc_max_burst_size_bytes();
    f.num_banks_dram = allocator->get_num_banks(tt::tt_metal::BufferType::DRAM);
    f.num_banks_l1 = allocator->get_num_banks(tt::tt_metal::BufferType::L1);
    f.l1_base = static_cast<uint32_t>(allocator->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
    f.l1_small_floor = static_cast<uint32_t>(::ttnn::ccl::l1_small_floor_address(mesh_device));
    f.headers_per_risc = headers_per_risc(f.arch);
    for (uint32_t n = 1; n <= kMaxWorkers; ++n) {
        f.mux_map_end[n - 1] = mux_map_end(mesh_device, f.arch, n, f.channel_bytes, f.l1_base);
    }
    f.worker_cores = mesh_device.worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id.value_or(mesh_device.get_sub_device_ids().at(0)));
    if (sub_core_grid.has_value()) {
        f.worker_cores = f.worker_cores.intersection(*sub_core_grid);
    }
    return f;
}

// ---- plan input ----

std::optional<std::string> gather_dim_padding_reason(const tt::tt_metal::TensorSpec& in, int32_t dim_from_end) {
    // Gather-dim padding would need the output's gather-dim extent to be num_devices * the input's padded extent, but
    // it is num_devices * the logical one.
    const auto& logical_shape = in.logical_shape();
    const auto& padded_shape = in.padded_shape();
    if (logical_shape[dim_from_end] == padded_shape[dim_from_end]) {
        return std::nullopt;
    }
    return fmt::format(
        "gather dim {} is padded from {} to {}; size must be a multiple of the tile/shard extent",
        static_cast<int32_t>(logical_shape.rank()) + dim_from_end,
        logical_shape[dim_from_end],
        padded_shape[dim_from_end]);
}

PlanIn make_plan_in(
    const AllGatherParams& args, const tt::tt_metal::TensorSpec& in, const tt::tt_metal::TensorSpec& out) {
    const Facts& f = args.facts;
    PlanIn p;
    p.facts = f;
    p.N = args.num_devices;
    p.shape_reason = shape_reason(args, in, out);
    if (p.shape_reason.empty()) {
        const auto ci = static_cast<uint32_t>(in.compute_page_size_bytes());
        const auto co = static_cast<uint32_t>(out.compute_page_size_bytes());
        const uint32_t ri = in.memory_config().is_dram() ? f.dram_align : f.l1_align;
        const uint32_t ro = out.memory_config().is_dram() ? f.dram_align : f.l1_align;
        // The walk steps output pages L apart: the bank count when interleaved, else the shard layout's.
        const uint32_t banks = out.memory_config().is_dram() ? f.num_banks_dram : f.num_banks_l1;
        uint32_t L = banks;
        uint32_t run_pages = std::numeric_limits<uint32_t>::max();
        const auto sharding = out.compute_buffer_sharding_args();
        const auto& dist = sharding.buffer_distribution_spec();
        if (dist.has_value()) {
            L = dist->contiguous_page_stride();
            run_pages = dist->num_contiguous_pages(0);
        }
        const uint32_t ppsr = pages_per_stripe_row(in, args.dim_from_end, ci);
        p.pages = Pages{
            .in_page_bytes = ci,
            .out_page_bytes = co,
            .in_aligned_page_bytes = tt::align(ci, ri),
            .out_aligned_page_bytes = tt::align(co, ro),
            .in_read_align = ri,
            .out_read_align = ro,
            .in_pages = static_cast<uint32_t>(in.compute_packed_buffer_size_bytes() / ci),
            .pages_per_stripe_row = ppsr,
            .out_page_stride = std::max(L, 1u),
            .out_run_pages = run_pages,
            .stripe_banks =
                stripe_banks(dist, banks, out.compute_packed_buffer_size_bytes() / co, uint64_t{ppsr} * ci, co, p.N)};
    }

    const bool fabric_2d = tt::tt_fabric::is_2d_fabric_config(args.fabric_config);
    for (uint32_t a = 0; a < 2; ++a) {
        p.axis_devices[a] = args.axis_num_devices[a];
        // Two devices are one hop apart both ways, so a 2-device ring is a line.
        p.axis_ring[a] = tt::tt_fabric::is_ring_or_torus(args.axis_topology[a]) && p.axis_devices[a] > 2;
    }
    p.true_2d = args.is_true_2d();
    if (p.true_2d) {
        // One multicast core per link serves both axes, so the axis with fewer links sets the count.
        p.links = std::min(args.axis_num_links[0], args.axis_num_links[1]);
        p.bent_axis = fabric_2d && !(args.axis_is_straight[0] && args.axis_is_straight[1]);
    } else {
        const uint32_t axis = args.get_1d_axis();
        p.ring = p.axis_ring[axis];
        p.links = args.axis_num_links[axis];
        p.bent_axis = fabric_2d && !args.axis_is_straight[axis];
        p.neighbor_exchange = args.fabric_config == tt::tt_fabric::FabricConfig::FABRIC_1D_NEIGHBOR_EXCHANGE;
    }
    return p;
}

// ---- geometry ----

std::optional<Geometry> make_geometry(const PlanIn& in, bool unicast) {
    const Pages& g = in.pages;
    if (g.in_page_bytes == 0 || g.out_page_bytes == 0 || g.in_read_align == 0 || g.out_read_align == 0 ||
        g.pages_per_stripe_row == 0 || in.N == 0) {
        return std::nullopt;
    }
    // Every source is read at unit offsets into a unit-aligned slot, so the unit must be a multiple of each source's
    // read alignment (else BH reads (src & ~63) | (dst & 63)) and of the NOC write alignment. The unicast relay reads
    // its own output.
    const uint32_t A = std::lcm(std::lcm(in.facts.noc_write_align, g.in_read_align), unicast ? g.out_read_align : 1u);
    Geometry geo{};
    geo.unit_bytes = A;
    if (g.in_page_bytes % A == 0 && g.out_page_bytes % A == 0) {
        // Sub-page: cuts land on any unit boundary, also inside a page.
        geo.in_page_units = g.in_page_bytes / A;
        geo.out_page_units = g.out_page_bytes / A;
    } else if (g.in_page_bytes == g.out_page_bytes && tt::align(g.in_page_bytes, A) <= g.out_aligned_page_bytes) {
        // Whole-page: a page moves at pitch align(content, A), and the tail lands in the output page's own padding.
        geo.in_page_units = geo.out_page_units = tt::align(g.in_page_bytes, A) / A;
    } else {
        return std::nullopt;
    }
    geo.max_packet_bytes = std::min(in.facts.payload, in.facts.max_burst) / A * A;
    // Pages that are not whole rows (packed size and page grid disagree) or a count past uint32: composite.
    if (g.in_pages % g.pages_per_stripe_row != 0) {
        return std::nullopt;
    }
    const uint64_t S = uint64_t{g.pages_per_stripe_row} * geo.in_page_units;
    geo.rows = g.in_pages / g.pages_per_stripe_row;
    geo.element_units = (S % geo.out_page_units == 0) ? geo.out_page_units : 1;
    const uint64_t elements = geo.rows * S / geo.element_units;
    if (S * in.N > std::numeric_limits<uint32_t>::max() || elements > std::numeric_limits<uint32_t>::max()) {
        return std::nullopt;
    }
    geo.stripe_units = static_cast<uint32_t>(S);
    geo.elements = static_cast<uint32_t>(elements);
    geo.dst_cross = geo.out_page_units * A == g.out_aligned_page_bytes;
    geo.src_cross = geo.in_page_units * A == g.in_aligned_page_bytes;
    if (geo.max_packet_bytes < A) {
        return geo;  // illegal (verdict says why); no walk to shape
    }

    const uint32_t L = std::max(g.out_page_stride, 1u);
    const ag::WalkGeo wg{
        .unit_bytes = A,
        .out_page_units = geo.out_page_units,
        .stripe_units = geo.stripe_units,
        .num_devices = in.N,
        .element_units = geo.element_units,
        .max_packet_bytes = geo.max_packet_bytes,
        .dst_cross = geo.dst_cross};
    // A segment is also no longer than the destination is contiguous: one page where a page never continues into the
    // next (padded pages, or unit elements on a strided layout), else the output's contiguous run.
    uint64_t contig = uint64_t{geo.out_page_units} * A;
    if (geo.dst_cross && (geo.element_units == geo.out_page_units || L == 1)) {
        contig *= g.out_run_pages;
    }
    geo.packet_cap = static_cast<uint32_t>(
        std::min<uint64_t>(ag::packet_cap(wg, ag::walk_shape(wg, L)), ag::kSegmentsPerPacket * contig));
    return geo;
}

// ---- legality ----

std::string verdict(const PlanIn& in, Algo algo) {
    TT_FATAL(algo != Algo::Composite, "all_gather: composite has no verdict");
    const bool unicast = algo == Algo::Unicast;
    const Facts& f = in.facts;
    if (!in.shape_reason.empty()) {
        return in.shape_reason;
    }
    if (in.N < 2) {
        return "needs at least 2 devices";
    }
    if (in.links == 0) {
        return "has no fabric link along the gathered axis";
    }
    if (unicast && in.true_2d) {
        return "needs a 1D axis";
    }
    if (!unicast && in.neighbor_exchange) {
        return "cannot route on a neighbor-exchange fabric (one hop only)";
    }
    if (!unicast && in.bent_axis) {
        return "cannot route along a mesh axis that turns corners";
    }
    const auto geo = make_geometry(in, unicast);
    if (!geo) {
        return "cannot cut these pages into aligned units";
    }
    if (geo->elements == 0) {
        return "has nothing to move";
    }
    if (f.payload > f.max_burst) {
        return fmt::format("fabric payload {} B exceeds the {} B NOC burst", f.payload, f.max_burst);
    }
    if (geo->max_packet_bytes < geo->unit_bytes) {
        return fmt::format("fabric payload {} B is below the {} B cut quantum", f.payload, geo->unit_bytes);
    }
    const uint32_t footprint = kCbSlots * slot_bytes(1, *geo, f);
    if (footprint > l1_budget(f)) {
        return fmt::format("{} B of slot buffers exceed the {} B of worker L1", footprint, l1_budget(f));
    }
    if (unicast) {
        if (f.mux_map_end[0] > f.l1_small_floor) {
            return fmt::format(
                "the smallest Mux V2 map ends at {:#x}, past the L1_SMALL floor {:#x}",
                f.mux_map_end[0],
                f.l1_small_floor);
        }
        if (!place_unicast(f.worker_cores, 1, 1)) {
            return "needs a mux and a worker on one core row for each direction";
        }
    } else {
        // The writer holds one scatter and one unicast header per connection.
        const uint32_t headers = 2 * mcast_connections(in);
        if (headers > f.headers_per_risc) {
            return fmt::format("needs {} packet headers per RISC, the pool holds {}", headers, f.headers_per_risc);
        }
        if (!place_multicast(f.worker_cores, 1)) {
            return "has no free worker cores";
        }
    }
    return {};
}

// ---- model ----

namespace detail {

uint64_t busiest_link_bytes(const Geometry& geo, uint32_t N, bool ring, uint32_t links) {
    if (links == 0 || N < 2) {
        return 0;
    }
    // A line's end link carries N - 1 stripes one way; a ring splits them over both directions.
    const uint64_t stripe_bytes = uint64_t{geo.rows} * geo.stripe_units * geo.unit_bytes;
    return tt::div_up(stripe_bytes * (N - 1), uint64_t{ring ? 2u : 1u} * links);
}

uint64_t busiest_link_packets(const Geometry& geo, uint32_t N, bool ring, uint32_t links) {
    return geo.packet_cap == 0 ? 0 : tt::div_up(busiest_link_bytes(geo, N, ring, links), uint64_t{geo.packet_cap});
}

double model_us(const PlanIn& in, Algo algo, const Geometry& geo) {
    const Tuning& t = tuning(in.facts.arch);
    const size_t a = algo_index(algo);
    const size_t topo = in.ring ? 1 : 0;
    const uint32_t N = in.N;
    // GB/s is bytes per ns, so bytes / (GB/s * 1e3) is microseconds.
    double us = static_cast<double>(busiest_link_bytes(geo, N, in.ring, in.links)) / (t.bw_gbps[a][topo] * 1e3);
    if (algo == Algo::Unicast && in.pages.stripe_banks != 0 && N >= 2) {
        // Every device writes N - 1 stripes and re-reads the ones it relays (line N - 2, ring N - 3), each direction
        // one stripe at a time; the two directions work on different stripes.
        const double stripe_bytes = static_cast<double>(geo.rows) * geo.stripe_units * geo.unit_bytes;
        const double relays = in.ring ? std::max(double(N) - 3, 0.0) : double(N) - 2;
        const double bank_us = stripe_bytes * ((N - 1) + relays) / (2.0 * in.pages.stripe_banks * t.bank_gbps * 1e3);
        us = std::max(us, bank_us);
    }
    return t.fixed_us[a][topo] + us;
}

uint32_t model_workers(const Tuning& t, bool ring, uint64_t busiest_packets) {
    return ((ring && t.ring_n3) || busiest_packets >= t.n3_min_packets) ? 3 : 2;
}

}  // namespace detail

// ---- slices and placement ----

std::vector<Slice> make_slices(uint32_t elements, uint32_t count) {
    TT_FATAL(count > 0, "all_gather: cannot cut {} elements into 0 slices", elements);
    const uint32_t base = elements / count;
    const uint32_t rem = elements % count;
    std::vector<Slice> slices(count);
    for (uint32_t i = 0; i < count; ++i) {
        slices[i] = Slice{.e0 = (i * base) + std::min(i, rem), .ne = base + (i < rem ? 1 : 0)};
    }
    return slices;
}

std::optional<std::vector<Group>> place_unicast(const CoreRangeSet& cores, uint32_t links, uint32_t n) {
    if (links == 0 || n == 0) {
        return std::nullopt;
    }
    const size_t groups = 2 * size_t{links};
    const size_t per = size_t{n} + 1;
    std::map<size_t, std::vector<size_t>> free;  // row y -> free x, ascending
    for (const auto& c : corerange_to_cores(cores, std::nullopt, /*row_wise=*/true)) {
        free[c.y].push_back(c.x);
    }
    for (auto& [y, xs] : free) {
        std::sort(xs.begin(), xs.end());
    }
    // A row at a time: each group on its own row first, then rows double up.
    std::vector<std::vector<CoreCoord>> placed;
    for (bool progress = true; progress && placed.size() < groups;) {
        progress = false;
        for (auto& [y, xs] : free) {
            if (placed.size() == groups) {
                break;
            }
            if (xs.size() >= per) {
                std::vector<CoreCoord> g;
                for (size_t i = 0; i < per; ++i) {
                    g.emplace_back(xs[i], y);
                }
                xs.erase(xs.begin(), xs.begin() + static_cast<std::ptrdiff_t>(per));
                placed.push_back(std::move(g));
                progress = true;
            }
        }
    }
    if (placed.size() < groups) {
        return std::nullopt;
    }
    std::vector<Group> out(groups);
    for (size_t g = 0; g < groups; ++g) {
        out[g] = Group{
            .link = static_cast<uint32_t>(g / 2),
            .dir = static_cast<uint32_t>(g % 2),
            .mux = placed[g][0],
            .workers = std::vector<CoreCoord>(placed[g].begin() + 1, placed[g].end())};
    }
    return out;
}

std::optional<std::vector<CoreCoord>> place_multicast(const CoreRangeSet& cores, uint32_t links) {
    if (links == 0 || cores.num_cores() < links) {
        return std::nullopt;
    }
    return corerange_to_cores(cores, links, /*row_wise=*/true);
}

// ---- the plan ----

Plan expand(const PlanIn& in, const Choice& choice) {
    const Facts& f = in.facts;
    const bool unicast = choice.algo == Algo::Unicast;
    TT_FATAL(
        unicast || choice.algo == Algo::Multicast,
        "all_gather: expand needs a native algorithm, got {}",
        algo_name(choice.algo));
    const std::string why = verdict(in, choice.algo);
    TT_FATAL(why.empty(), "all_gather: {} is illegal here: {}", algo_name(choice.algo), why);
    TT_FATAL(
        choice.links >= 1 && choice.links <= in.links,
        "all_gather: {} links requested, {} available",
        choice.links,
        in.links);
    TT_FATAL(
        choice.workers >= 1 && choice.workers <= (unicast ? kMaxWorkers : 1u),
        "all_gather: {} cannot run {} workers per link-direction",
        algo_name(choice.algo),
        choice.workers);
    TT_FATAL(
        !unicast || f.mux_map_end[choice.workers - 1] <= f.l1_small_floor,
        "all_gather: a {}-channel Mux V2 map ends at {:#x}, past the L1_SMALL floor {:#x}",
        choice.workers,
        f.mux_map_end[choice.workers - 1],
        f.l1_small_floor);
    Plan plan;
    plan.choice = choice;
    plan.N = in.N;
    plan.ring = in.ring;
    plan.arch = f.arch;
    plan.geo = make_geometry(in, unicast).value();
    plan.slot_bytes = slot_bytes(choice.packets_per_slot, plan.geo, f);
    TT_FATAL(
        choice.packets_per_slot >= 1 && kCbSlots * plan.slot_bytes <= l1_budget(f),
        "all_gather: {} packets per slot need {} B of slot buffers, the worker L1 holds {} B",
        choice.packets_per_slot,
        kCbSlots * plan.slot_bytes,
        l1_budget(f));
    if (unicast) {
        auto groups = place_unicast(f.worker_cores, choice.links, choice.workers);
        TT_FATAL(
            groups.has_value(),
            "all_gather: {} links x (mux + {} workers) do not fit the worker cores",
            choice.links,
            choice.workers);
        plan.groups = std::move(*groups);
        plan.slices = make_slices(plan.geo.elements, choice.links * choice.workers);
    } else {
        auto cores = place_multicast(f.worker_cores, choice.links);
        TT_FATAL(cores.has_value(), "all_gather: {} multicast cores do not fit the worker cores", choice.links);
        plan.mcast_cores = std::move(*cores);
        plan.slices = make_slices(plan.geo.elements, choice.links);
    }
    return plan;
}

Plan expand(const AllGatherParams& params, const tt::tt_metal::TensorSpec& in, const tt::tt_metal::TensorSpec& out) {
    return expand(make_plan_in(params, in, out), params.choice);
}

Plan make_plan(const PlanIn& in) {
    std::array<std::string, 2> why;
    std::array<std::optional<Geometry>, 2> geo;
    std::array<double, 2> t_us{};
    std::vector<Algo> order;
    for (const Algo a : kNativeAlgos) {
        const size_t i = algo_index(a);
        why[i] = verdict(in, a);
        if (why[i].empty()) {
            geo[i] = make_geometry(in, a == Algo::Unicast);
            t_us[i] = detail::model_us(in, a, *geo[i]);
            order.push_back(a);
        }
    }
    if (!order.empty()) {
        warn_if_uncalibrated(in);
    }
    // Lowest modelled time first; a tie keeps unicast.
    std::stable_sort(
        order.begin(), order.end(), [&](Algo x, Algo y) { return t_us[algo_index(x)] < t_us[algo_index(y)]; });
    log_debug(
        tt::LogOp,
        "all_gather plan: model unicast {:.1f} us, multicast {:.1f} us (0 = illegal)",
        t_us[algo_index(Algo::Unicast)],
        t_us[algo_index(Algo::Multicast)]);

    for (const Algo a : order) {
        if (const auto choice = fit(in, a, *geo[algo_index(a)])) {
            return expand(in, *choice);
        }
        warn_once(fmt::format(
            "all_gather {} does not fit the worker cores; trying the next algorithm. This may lead to performance "
            "loss.",
            algo_name(a)));
    }
    if (!in.shape_reason.empty()) {
        return composite(in, in.shape_reason);
    }
    const auto reason = [&](Algo a) {
        const std::string& w = why[algo_index(a)];
        return w.empty() ? std::string("does not fit the worker cores") : w;
    };
    return composite(in, fmt::format("unicast {}; multicast {}", reason(Algo::Unicast), reason(Algo::Multicast)));
}

bool dir_active(const Plan& plan, uint32_t dev, uint32_t dir) {
    return plan.ring || (dir == kFwd ? dev + 1 < plan.N : dev > 0);
}

WorkerSched schedule(const Plan& plan, uint32_t dev, uint32_t link, uint32_t dir, uint32_t w) {
    const uint32_t N = plan.N;
    const bool unicast = plan.choice.algo == Algo::Unicast;
    const uint32_t workers = plan.choice.workers;
    TT_FATAL(
        (unicast || plan.choice.algo == Algo::Multicast) && dev < N && link < plan.choice.links &&
            dir < (unicast ? 2u : 1u) && w < workers && (link * workers) + w < plan.slices.size(),
        "all_gather: no {} worker (dev {}, link {}, dir {}, w {})",
        algo_name(plan.choice.algo),
        dev,
        link,
        dir,
        w);
    const Slice s = plan.slices[(link * workers) + w];
    const uint32_t E = plan.geo.element_units;
    WorkerSched ws;
    ws.e0 = s.e0;
    ws.ne = s.ne;
    ws.iter_units = s.ne * E;
    ws.stripe0 = dev;
    if (!unicast) {
        // One pass over the own stripe, multicast to every other device, plus the local copy. Init and done each reach
        // every other device once; done follows the data on its route.
        ws.iters = 1;
        ws.last_take = s.ne;
        ws.local_copy = true;
        ws.init_send = true;
        ws.init_wait = N - 1;
        ws.end_wait = N - 1;
        return ws;
    }

    const bool fwd = dir == kFwd;
    // Stripes a direction sends: ring N/2; line forward d+1, backward N-d; none from a line end's dead side.
    auto iters = [&](bool forward) -> uint32_t {
        if (plan.ring) {
            return N / 2;
        }
        return forward ? (dev + 1 < N ? dev + 1 : 0) : (dev > 0 ? N - dev : 0);
    };
    ws.iters = iters(fwd);
    // Iteration i sends stripe dev - i (forward) or dev + i (backward): what upstream sent in iteration i - 1.
    ws.stripe_step = fwd ? N - 1 : 1;
    // Even ring: the antipode is reached both ways, so each direction takes half of that stripe, split by walk seqno.
    const bool even_split = plan.ring && N % 2 == 0;
    ws.last_skip = (even_split && !fwd) ? s.ne / 2 : 0;
    ws.last_take = ws.iters == 0 ? 0 : (even_split ? (fwd ? s.ne / 2 : s.ne - (s.ne / 2)) : s.ne);
    const uint32_t last_units = ws.last_take * E;
    // Units the same-direction upstream worker sends here; it has the same slice, so the same last_units.
    if (plan.ring) {
        ws.end_wait = even_split ? ((N / 2 - 1) * ws.iter_units) + last_units : ((N - 1) / 2) * ws.iter_units;
    } else {
        ws.end_wait = (fwd ? dev : N - 1 - dev) * ws.iter_units;
    }
    // Iteration 0 carries this device's own stripe; the forward worker copies it, unless it has nothing to send.
    ws.local_copy = fwd ? iters(true) > 0 : iters(true) == 0;
    // Init pairing: a writer with a neighbor tells that neighbor's opposite-direction worker w it is alive; that worker
    // is the one writing back into this device's output, so it is the one whose writer waits here.
    const bool active = dir_active(plan, dev, dir);
    ws.init_send = active;
    ws.init_wait = active ? 1 : 0;
    const Tuning& t = tuning(plan.arch);
    const uint64_t iter_packets =
        plan.geo.packet_cap == 0
            ? 0
            : tt::div_up(uint64_t{ws.iter_units} * plan.geo.unit_bytes, uint64_t{plan.geo.packet_cap});
    const uint64_t cadence_packets = std::clamp<uint64_t>(
        tt::div_up(iter_packets, uint64_t{t.signals_per_iter}), t.cadence_min_packets, t.cadence_max_packets);
    ws.cadence_slots = static_cast<uint32_t>(tt::div_up(cadence_packets, uint64_t{plan.choice.packets_per_slot}));
    return ws;
}

std::vector<McastConn> mcast_routes(
    uint32_t N,
    std::array<uint32_t, 2> mesh_shape,
    std::array<uint32_t, 2> my_coord,
    std::optional<uint32_t> axis,
    std::array<bool, 2> axis_ring,
    std::array<uint32_t, 4> phys_dir) {
    // Hops per logical direction. Axis 1 runs E (forward) / W, axis 0 runs S (forward) / N. A ring splits the other
    // devices between its two directions, the forward one taking the extra device of an even ring.
    std::array<uint32_t, 4> hops{};  // indexed by Dir
    uint32_t devices = 1;
    for (uint32_t a = 0; a < 2; ++a) {
        const uint32_t size = mesh_shape[a];
        if (size <= 1 || axis.value_or(a) != a) {
            continue;
        }
        TT_FATAL(size < 256 && my_coord[a] < size, "all_gather: bad mesh axis {}: {} of {}", a, my_coord[a], size);
        devices *= size;
        const uint32_t fwd = axis_ring[a] ? size / 2 : size - 1 - my_coord[a];
        hops[static_cast<size_t>(a == 1 ? Dir::East : Dir::South)] = fwd;
        hops[static_cast<size_t>(a == 1 ? Dir::West : Dir::North)] = axis_ring[a] ? size - 1 - fwd : my_coord[a];
    }
    TT_FATAL(devices == N, "all_gather: mesh axes hold {} devices, the plan {}", devices, N);

    // E/W connections are lines; N/S connections are spines that branch E and W at every hop, so the connections cover
    // every other device exactly once.
    auto range = [&](Dir d) {
        uint32_t r = 0;
        auto put = [&](Dir logical) {
            const auto i = static_cast<size_t>(logical);
            if (hops[i] > 0) {  // an absent branch must not clobber a live slot
                r |= hops[i] << (8 * phys_dir[i]);
            }
        };
        put(d);
        if (d == Dir::North || d == Dir::South) {
            put(Dir::East);
            put(Dir::West);
        }
        return r;
    };
    std::vector<McastConn> conns;
    for (const Dir d : {Dir::East, Dir::West, Dir::North, Dir::South}) {
        if (hops[static_cast<size_t>(d)] > 0) {
            conns.push_back(McastConn{.dir = d, .range = range(d)});
        }
    }
    return conns;
}

}  // namespace ttnn::operations::ccl::all_gather_plan
