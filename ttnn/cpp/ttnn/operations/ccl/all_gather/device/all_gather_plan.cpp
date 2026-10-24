// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_plan.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <tuple>

#include <fmt/format.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "ttnn/operations/ccl/all_gather/device/kernels/ag_walk.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::ccl::all_gather_plan {

namespace {

using tt::tt_metal::NOC;

// NOC_{L1,DRAM}_WRITE_ALIGNMENT_BYTES on Wormhole and Blackhole; no public host API exposes it.
constexpr uint32_t kNocWriteAlignment = 16;

// ---- tuning ----
// Measured on BH 8x p150b (mesh (1, 8), 2 links, payload 15168) with the cmp5 sweep: bf16 tile [1, 1, R, 8192]
// gathered on dim 3 at 64 KB - 42 MB, row-major page sizes 256 B - 16 KB, plus held-out dtypes and layouts.
constexpr Tuning kBlackhole{
    .arch = tt::ARCH::BLACKHOLE,
    .payload = 15168,
    .links = 2,
    // The least T - B / BW over 64 KB - 4 MB (unicast at the n model_workers picks); every size of the fit then picks
    // the faster algorithm. Includes the init handshake, about 6 us of cross-device launch skew. Too low an F pulls
    // small sizes toward that algorithm.
    .fixed_us = {{{20.5, 15.0}, {16.0, 12.0}}},
    // The 12-42 MB slope (unicast at n = 3: 465 / 537 cycles per full packet). Too high a BW pulls large sizes toward
    // that algorithm.
    .bw_gbps = {{{44.0, 38.0}, {37.0, 25.5}}},
    // One DRAM bank caps at ~64 GB/s. ND-sharded DRAM output with one bank per stripe: unicast 175 / 114 us, multicast
    // 128 / 91 us (line / ring). L1 banks are charged the same (unmeasured); they bind only for a stripe on few cores.
    .bank_gbps = 64.0,
    // Ring: n = 3 is 1.4-7% faster than 2 at 256 KB - 9 MB (a worker's ~626 cycles per packet exceed the ring's ~539
    // cycle budget at n = 2).
    .ring_n3 = true,
    // Line: the relay chain is N - 1 iterations; at 121 packets n = 2 is 10% faster, from 242 n = 3 is as fast or
    // faster.
    .n3_min_packets = 160,
    // Two full packets per read barrier: k = 2 costs a worker 24% less time per packet than k = 1.
    .slot_read_bytes = 2 * 15168,
    // Each signal is a packet of its own: a long cadence saves link time, a short one starts the downstream relay
    // sooner. 1 packet is up to 1.5x slower at 8-26 MB; at 2 MB (9 packets per iteration) 2-3 is best and 8 is 20%
    // slower; at 26 MB a 16-packet cap is 1.7% faster than 8 on the line at n = 3.
    .signals_per_iter = 4,
    .cadence_min_packets = 2,
    .cadence_max_packets = 16,
    // n = 1 at 2 buffers costs 12-14%; at n >= 2, 2 is within 3% of 4 and keeps the mux's L1 map smallest.
    .mux_buffers = {4, 2, 2},
    // The forwarder writes every packet into an ethernet core: NOC1 is 1.8-2.3x slower there.
    .mux_forwarder_noc = NOC::NOC_0,
    // Multicast writer on NOC0 (it writes every packet into the ethernet cores): 1.42x faster on a ring, 1.13x on a
    // line. Unicast writer on NOC0: 5% (n = 3) / 10% (n = 2) faster on a ring at 8-42 MB, 8% / 3% slower on a line.
    .writer_noc = {{{NOC::NOC_1, NOC::NOC_0}, {NOC::NOC_0, NOC::NOC_0}}},
};
constexpr std::array kTunings = {kBlackhole};

const char* algo_name(AgAlgo algo) {
    switch (algo) {
        case AgAlgo::Auto: return "auto";
        case AgAlgo::Unicast: return "unicast";
        case AgAlgo::Multicast: return "multicast";
        case AgAlgo::Composite: return "composite";
    }
    return "?";
}

uint64_t div_up(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

// make_plan runs on every call, so a given shortfall is warned about once per process, not once per call.
void warn_once(const std::string& message) {
    static std::mutex mutex;
    static std::set<std::string> warned;
    const std::lock_guard lock(mutex);
    if (warned.insert(message).second) {
        log_warning(tt::LogOp, "{}", message);
    }
}

// ---- facts ----

// Fabric packet-header pool entries per data-movement RISC: NUM_PACKET_HEADERS / MaxDMProcessorsPerCoreType in the
// arch's dev_mem_map.h, one per direction (WH 4, BH 6) and connection convention (2); no public host API exposes it.
// An unknown arch gets the smallest known pool, so the multicast header rule errs toward illegal.
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
    static std::mutex mutex;
    static std::map<Key, uint32_t> memo;
    const uint32_t buffers = mux_buffers(arch, n);
    const Key key{arch, tt::tt_fabric::GetFabricConfig(), n, buffers, channel_bytes, l1_base};
    const std::lock_guard lock(mutex);
    if (const auto it = memo.find(key); it != memo.end()) {
        return it->second;
    }
    const tt::tt_fabric::FabricMuxV2Config config(
        mesh_device, static_cast<uint8_t>(n), static_cast<uint8_t>(buffers), channel_bytes, l1_base);
    return memo.emplace(key, static_cast<uint32_t>(config.get_memory_map_end_address())).first->second;
}

// ---- geometry helpers ----

// L1 a worker's slot CB may use: everything between the allocator base and the L1_SMALL floor (static limits only;
// live allocator state would break the program cache).
uint32_t l1_budget(const AgFacts& f) { return f.l1_small_floor > f.l1_base ? f.l1_small_floor - f.l1_base : 0; }

uint32_t slot_bytes(uint32_t k, const Geometry& geo, const AgFacts& f) {
    return ag::slot_offsets(k, geo.A, f.header_bytes, geo.cap).slot_bytes;
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
    if (input_page_size > output_page_size && output_page_size % kNocWriteAlignment != 0) {
        return fmt::format(
            "output is sharded finer than the input; its rows ({} B) must be a multiple of {} B",
            output_page_size,
            kNocWriteAlignment);
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

// k: about slot_read_bytes of packets (at the packet cap) per read barrier, at least kCbSlots slots per worker
// iteration so both slots work, and at most what fits the L1 budget (at least 1).
uint32_t packets_per_slot(const Tuning& t, const AgFacts& f, const Geometry& geo, uint32_t workers) {
    const uint64_t iter_packets = div_up(div_up(geo.elements, workers) * geo.E * geo.A, geo.cap);
    uint32_t k = static_cast<uint32_t>(std::clamp<uint64_t>(
        std::min(div_up(t.slot_read_bytes, geo.cap), iter_packets / kCbSlots), 1, kMaxPacketsPerSlot));
    while (k > 1 && kCbSlots * slot_bytes(k, geo, f) > l1_budget(f)) {
        --k;
    }
    return k;
}

// The largest (links, n) of one legal algorithm that fits: n shrinks first (3 -> 2 -> 1), then links. Forced values
// do not shrink.
std::optional<AgChoice> fit(const PlanIn& in, uint32_t algo, const Geometry& geo) {
    const bool unicast = algo == kUc;
    const AgFacts& f = in.facts;
    const Tuning& t = tuning(f.arch);
    const uint32_t links_want = in.force.links != 0 ? in.force.links : in.links;
    uint32_t n_want = 1;
    if (unicast) {
        n_want = in.force.workers != 0
                     ? in.force.workers
                     : model_workers(t, in.ring, busiest_link_packets(geo, in.N, in.ring, links_want));
    }
    const uint32_t links_min = in.force.links != 0 ? links_want : 1;
    const uint32_t n_min = (!unicast || in.force.workers != 0) ? n_want : 1;
    for (uint32_t links = links_want; links >= links_min; --links) {
        for (uint32_t n = n_want; n >= n_min; --n) {
            if (unicast && f.mux_map_end[n - 1] > f.l1_small_floor) {
                continue;
            }
            const bool placed = unicast ? place_unicast(f.worker_cores, links, n).has_value()
                                        : place_multicast(f.worker_cores, links).has_value();
            if (!placed) {
                continue;
            }
            const AgAlgo a = unicast ? AgAlgo::Unicast : AgAlgo::Multicast;
            if (n < n_want || links < links_want) {
                warn_once(fmt::format(
                    "all_gather {} uses {} worker(s) x {} link(s), wanted {} x {}: the worker cores (or L1 below "
                    "L1_SMALL for the mux) do not fit more. This may lead to performance loss.",
                    algo_name(a),
                    n,
                    links,
                    n_want,
                    links_want));
            }
            return AgChoice{
                .algo = a,
                .n = static_cast<uint8_t>(n),
                .links = static_cast<uint8_t>(links),
                .k = static_cast<uint8_t>(packets_per_slot(t, f, geo, links * n))};
        }
    }
    return std::nullopt;
}

// Composite gathers through all_broadcast, which multicasts every device's input to every other device: it needs a
// second device and, past 2 devices, routes beyond the neighbor.
AgPlan composite(const PlanIn& in, std::string reason) {
    TT_FATAL(in.N >= 2, "all_gather needs at least 2 devices on the gathered axis, got {}", in.N);
    TT_FATAL(
        !in.neighbor_exchange || in.N <= 2,
        "all_gather cannot run across {} devices on a neighbor-exchange fabric: {}; and the composite fallback needs "
        "multi-hop multicast, which that fabric does not provide",
        in.N,
        reason);
    AgPlan plan;
    plan.choice = AgChoice{.algo = AgAlgo::Composite};
    plan.N = in.N;
    plan.ring = in.ring;
    plan.arch = in.facts.arch;
    plan.composite_reason = std::move(reason);
    return plan;
}

std::vector<CoreCoord> ordered_cores(const CoreRangeSet& cores) {
    std::vector<CoreCoord> out;
    for (const auto& r : cores.ranges()) {
        for (size_t y = r.start_coord.y; y <= r.end_coord.y; ++y) {
            for (size_t x = r.start_coord.x; x <= r.end_coord.x; ++x) {
                out.emplace_back(x, y);
            }
        }
    }
    return out;
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

// Pages::stripe_banks of a sharded output, from the distribution spec's page map. Memoized, as the map walks every
// page and the plan runs on every call; the memo is cleared when it reaches kMemoEntries layouts.
uint32_t sharded_stripe_banks(
    const tt::tt_metal::BufferDistributionSpec& dist, uint64_t out_pages, uint64_t W, uint64_t page_bytes, uint32_t N) {
    constexpr size_t kMemoEntries = 256;
    using Key = std::
        tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<CoreCoord>, int, uint64_t, uint64_t, uint32_t>;
    static std::mutex mutex;
    static std::map<Key, uint32_t> memo;
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
    const std::lock_guard lock(mutex);
    if (const auto it = memo.find(key); it != memo.end()) {
        return it->second;
    }
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
    if (memo.size() >= kMemoEntries) {
        memo.clear();
    }
    return memo.emplace(std::move(key), fewest).first->second;
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

// ---- override ----

AgForce parse_force(std::string_view spec) {
    const size_t colon = spec.find(':');
    const std::string_view name = spec.substr(0, colon);
    constexpr std::array<std::pair<std::string_view, AgAlgo>, 4> kAlgos = {{
        {"auto", AgAlgo::Auto},
        {"unicast", AgAlgo::Unicast},
        {"multicast", AgAlgo::Multicast},
        {"composite", AgAlgo::Composite},
    }};
    const auto algo = std::find_if(kAlgos.begin(), kAlgos.end(), [&](const auto& a) { return a.first == name; });
    if (algo == kAlgos.end()) {
        TT_THROW("{}='{}': unknown algorithm '{}' (auto, unicast, multicast, composite)", kForceEnvVar, spec, name);
    }
    AgForce force{.algo = algo->second};
    if (colon == std::string_view::npos) {
        return force;
    }
    TT_FATAL(
        force.algo != AgAlgo::Composite,
        "{}='{}': composite takes no options (it runs no all_gather kernels)",
        kForceEnvVar,
        spec);

    // The whole key grammar: name, range, whether it applies to unicast only.
    struct Key {
        std::string_view name;
        uint32_t lo, hi;
        bool unicast_only;
    };
    constexpr std::array<Key, 4> kKeys = {{
        {"n", 1, kAgMaxWorkers, true},
        {"links", 1, 255, false},
        {"rdelay", 0, 0xFFFF, false},
        {"wdelay", 0, 0xFFFF, false},
    }};
    std::array<std::optional<uint32_t>, kKeys.size()> values;
    std::string_view rest = spec.substr(colon + 1);
    while (true) {
        const size_t comma = rest.find(',');
        const std::string_view kv = rest.substr(0, comma);
        const size_t eq = kv.find('=');
        TT_FATAL(eq != std::string_view::npos, "{}='{}': '{}' is not key=value", kForceEnvVar, spec, kv);
        const std::string_view key = kv.substr(0, eq);
        const std::string_view text = kv.substr(eq + 1);
        const auto it = std::find_if(kKeys.begin(), kKeys.end(), [&](const Key& k) { return k.name == key; });
        TT_FATAL(it != kKeys.end(), "{}='{}': unknown key '{}' (n, links, rdelay, wdelay)", kForceEnvVar, spec, key);
        auto& value = values[static_cast<size_t>(it - kKeys.begin())];
        TT_FATAL(!value.has_value(), "{}='{}': '{}' given twice", kForceEnvVar, spec, key);
        uint32_t v = 0;
        const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), v);
        TT_FATAL(
            !text.empty() && ec == std::errc{} && end == text.data() + text.size(),
            "{}='{}': '{}' is not a decimal number",
            kForceEnvVar,
            spec,
            text);
        TT_FATAL(v >= it->lo && v <= it->hi, "{}='{}': {} must be {}-{}", kForceEnvVar, spec, key, it->lo, it->hi);
        TT_FATAL(
            !it->unicast_only || force.algo == AgAlgo::Unicast,
            "{}='{}': {} applies to unicast only",
            kForceEnvVar,
            spec,
            key);
        value = v;
        if (comma == std::string_view::npos) {
            break;
        }
        rest = rest.substr(comma + 1);
    }
    force.workers = static_cast<uint8_t>(values[0].value_or(0));
    force.links = static_cast<uint8_t>(values[1].value_or(0));
    force.reader_delay = static_cast<uint16_t>(values[2].value_or(0));
    force.writer_delay = static_cast<uint16_t>(values[3].value_or(0));
    return force;
}

// ---- facts ----

AgFacts make_facts(
    const tt::tt_metal::distributed::MeshDevice& mesh_device,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grid) {
    const auto& allocator = mesh_device.allocator();
    AgFacts f;
    f.arch = tt::tt_metal::hal::get_arch();
    f.payload = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());
    f.header_bytes = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes());
    f.channel_bytes = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes());
    f.dram_align = tt::tt_metal::hal::get_dram_alignment();
    f.l1_align = tt::tt_metal::hal::get_l1_alignment();
    f.max_burst = tt::tt_metal::hal::get_noc_max_burst_size_bytes();
    f.num_banks_dram = allocator->get_num_banks(tt::tt_metal::BufferType::DRAM);
    f.num_banks_l1 = allocator->get_num_banks(tt::tt_metal::BufferType::L1);
    f.l1_base = static_cast<uint32_t>(allocator->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
    f.l1_small_floor = static_cast<uint32_t>(::ttnn::ccl::l1_small_floor_address(mesh_device));
    f.headers_per_risc = headers_per_risc(f.arch);
    for (uint32_t n = 1; n <= kAgMaxWorkers; ++n) {
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
    const AgFacts& f = args.facts;
    PlanIn p;
    p.facts = f;
    p.force = args.force;
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
            .ci = ci,
            .co = co,
            .ai = tt::align(ci, ri),
            .ao = tt::align(co, ro),
            .ri = ri,
            .ro = ro,
            .in_pages = static_cast<uint32_t>(in.compute_packed_buffer_size_bytes() / ci),
            .pages_per_stripe_row = ppsr,
            .L = std::max(L, 1u),
            .run_pages = run_pages,
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
    if (g.ci == 0 || g.co == 0 || g.ri == 0 || g.ro == 0 || g.pages_per_stripe_row == 0 || in.N == 0) {
        return std::nullopt;
    }
    // Every source is read at unit offsets into a unit-aligned slot, so the unit must be a multiple of each source's
    // read alignment (else BH reads (src & ~63) | (dst & 63)) and of the NOC write alignment. The unicast relay reads
    // its own output.
    const uint32_t A = std::lcm(std::lcm(kNocWriteAlignment, g.ri), unicast ? g.ro : 1u);
    Geometry geo{};
    geo.A = A;
    if (g.ci % A == 0 && g.co % A == 0) {
        // Sub-page: cuts land on any unit boundary, also inside a page.
        geo.Ui = g.ci / A;
        geo.Uo = g.co / A;
    } else if (g.ci == g.co && tt::align(g.ci, A) <= g.ao) {
        // Whole-page: a page moves at pitch align(c, A), and the tail past c lands in the output page's own padding.
        geo.Ui = geo.Uo = tt::align(g.ci, A) / A;
    } else {
        return std::nullopt;
    }
    geo.P_A = std::min(in.facts.payload, in.facts.max_burst) / A * A;
    // Pages that are not whole rows (packed size and page grid disagree) or a count past uint32: composite.
    if (g.in_pages % g.pages_per_stripe_row != 0) {
        return std::nullopt;
    }
    const uint64_t S = uint64_t{g.pages_per_stripe_row} * geo.Ui;
    geo.R = g.in_pages / g.pages_per_stripe_row;
    geo.E = (S % geo.Uo == 0) ? geo.Uo : 1;
    const uint64_t elements = geo.R * S / geo.E;
    if (S * in.N > std::numeric_limits<uint32_t>::max() || elements > std::numeric_limits<uint32_t>::max()) {
        return std::nullopt;
    }
    geo.S = static_cast<uint32_t>(S);
    geo.elements = static_cast<uint32_t>(elements);
    geo.dst_cross = geo.Uo * A == g.ao;
    geo.src_cross = geo.Ui * A == g.ai;
    if (geo.P_A < A) {
        return geo;  // illegal (verdict says why); no walk to shape
    }

    const uint32_t L = std::max(g.L, 1u);
    const ag::WalkGeo wg{
        .A = A, .Uo = geo.Uo, .S = geo.S, .N = in.N, .E = geo.E, .P_A = geo.P_A, .dst_cross = geo.dst_cross};
    // A segment is also no longer than the destination is contiguous: one page where a page never continues into the
    // next (padded pages, or unit elements on a strided layout), else the output's contiguous run.
    uint64_t contig = uint64_t{geo.Uo} * A;
    if (geo.dst_cross && (geo.E == geo.Uo || L == 1)) {
        contig *= g.run_pages;
    }
    geo.cap = static_cast<uint32_t>(
        std::min<uint64_t>(ag::packet_cap(wg, ag::walk_shape(wg, L)), ag::kSegmentsPerPacket * contig));
    return geo;
}

// ---- legality ----

std::string verdict(const PlanIn& in, uint32_t algo) {
    const bool unicast = algo == kUc;
    const AgFacts& f = in.facts;
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
    if (geo->P_A < geo->A) {
        return fmt::format("fabric payload {} B is below the {} B cut quantum", f.payload, geo->A);
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

uint64_t busiest_link_bytes(const Geometry& geo, uint32_t N, bool ring, uint32_t links) {
    if (links == 0 || N < 2) {
        return 0;
    }
    // A line's end link carries N - 1 stripes one way; a ring splits them over both directions.
    const uint64_t stripe_bytes = uint64_t{geo.R} * geo.S * geo.A;
    return div_up(stripe_bytes * (N - 1), uint64_t{ring ? 2u : 1u} * links);
}

uint64_t busiest_link_packets(const Geometry& geo, uint32_t N, bool ring, uint32_t links) {
    return geo.cap == 0 ? 0 : div_up(busiest_link_bytes(geo, N, ring, links), geo.cap);
}

double model_us(const PlanIn& in, uint32_t algo, const Geometry& geo) {
    const Tuning& t = tuning(in.facts.arch);
    const uint32_t topo = in.ring ? 1 : 0;
    const uint32_t N = in.N;
    // GB/s is bytes per ns, so bytes / (GB/s * 1e3) is microseconds.
    double us = static_cast<double>(busiest_link_bytes(geo, N, in.ring, in.links)) / (t.bw_gbps[algo][topo] * 1e3);
    if (algo == kUc && in.pages.stripe_banks != 0 && N >= 2) {
        // Every device writes N - 1 stripes and re-reads the ones it relays (line N - 2, ring N - 3), each direction
        // one stripe at a time; the two directions work on different stripes.
        const double stripe_bytes = static_cast<double>(geo.R) * geo.S * geo.A;
        const double relays = in.ring ? std::max(double(N) - 3, 0.0) : double(N) - 2;
        const double bank_us = stripe_bytes * ((N - 1) + relays) / (2.0 * in.pages.stripe_banks * t.bank_gbps * 1e3);
        us = std::max(us, bank_us);
    }
    return t.fixed_us[algo][topo] + us;
}

uint32_t model_workers(const Tuning& t, bool ring, uint64_t busiest_packets) {
    return ((ring && t.ring_n3) || busiest_packets >= t.n3_min_packets) ? 3 : 2;
}

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
    for (const auto& c : ordered_cores(cores)) {
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
    auto all = ordered_cores(cores);
    if (links == 0 || all.size() < links) {
        return std::nullopt;
    }
    all.resize(links);
    return all;
}

// ---- the plan ----

AgPlan expand(const PlanIn& in, const AgChoice& choice) {
    const AgFacts& f = in.facts;
    const bool unicast = choice.algo == AgAlgo::Unicast;
    TT_FATAL(
        unicast || choice.algo == AgAlgo::Multicast,
        "all_gather: expand needs a native algorithm, got {}",
        algo_name(choice.algo));
    const std::string why = verdict(in, unicast ? kUc : kMc);
    TT_FATAL(why.empty(), "all_gather: {} is illegal here: {}", algo_name(choice.algo), why);
    AgPlan plan;
    plan.choice = choice;
    plan.N = in.N;
    plan.ring = in.ring;
    plan.arch = f.arch;
    plan.geo = make_geometry(in, unicast).value();
    plan.slot_bytes = slot_bytes(choice.k, plan.geo, f);
    if (unicast) {
        auto groups = place_unicast(f.worker_cores, choice.links, choice.n);
        TT_FATAL(
            groups.has_value(),
            "all_gather: {} links x (mux + {} workers) do not fit the worker cores",
            choice.links,
            choice.n);
        plan.groups = std::move(*groups);
        plan.slices = make_slices(plan.geo.elements, uint32_t{choice.links} * choice.n);
    } else {
        auto cores = place_multicast(f.worker_cores, choice.links);
        TT_FATAL(cores.has_value(), "all_gather: {} multicast cores do not fit the worker cores", choice.links);
        plan.mcast_cores = std::move(*cores);
        plan.slices = make_slices(plan.geo.elements, choice.links);
    }
    return plan;
}

AgPlan expand(const AllGatherParams& params, const tt::tt_metal::TensorSpec& in, const tt::tt_metal::TensorSpec& out) {
    return expand(make_plan_in(params, in, out), params.choice);
}

AgPlan make_plan(const PlanIn& in) {
    const AgForce& force = in.force;
    if (force.algo == AgAlgo::Composite) {
        return composite(in, fmt::format("forced by {}", kForceEnvVar));
    }
    const std::array<std::string, 2> why = {verdict(in, kUc), verdict(in, kMc)};
    std::array<std::optional<Geometry>, 2> geo;
    std::array<double, 2> t_us{};
    std::vector<uint32_t> order;
    for (const uint32_t a : {kUc, kMc}) {
        if (why[a].empty()) {
            geo[a] = make_geometry(in, a == kUc);
            t_us[a] = model_us(in, a, *geo[a]);
            order.push_back(a);
        }
    }
    if (!order.empty()) {
        warn_if_uncalibrated(in);
    }
    // Lowest modelled time first; a tie keeps unicast.
    std::stable_sort(order.begin(), order.end(), [&](uint32_t x, uint32_t y) { return t_us[x] < t_us[y]; });
    const bool forced = force.algo != AgAlgo::Auto;
    if (forced) {
        const uint32_t a = force.algo == AgAlgo::Unicast ? kUc : kMc;
        TT_FATAL(
            why[a].empty(),
            "all_gather: {} forces {}, which is illegal here: {}",
            kForceEnvVar,
            algo_name(force.algo),
            why[a]);
        TT_FATAL(
            force.links <= in.links,
            "all_gather: {} forces {} links, only {} are available",
            kForceEnvVar,
            force.links,
            in.links);
        order = {a};
    }
    log_debug(
        tt::LogOp, "all_gather plan: model unicast {:.1f} us, multicast {:.1f} us (0 = illegal)", t_us[kUc], t_us[kMc]);

    for (const uint32_t a : order) {
        if (const auto choice = fit(in, a, *geo[a])) {
            return expand(in, *choice);
        }
        TT_FATAL(
            !forced,
            "all_gather: {} forces {}, which does not fit the worker cores or L1 with the forced workers/links",
            kForceEnvVar,
            algo_name(force.algo));
        warn_once(fmt::format(
            "all_gather {} does not fit the worker cores; trying the next algorithm. This may lead to performance "
            "loss.",
            algo_name(a == kUc ? AgAlgo::Unicast : AgAlgo::Multicast)));
    }
    if (!in.shape_reason.empty()) {
        return composite(in, in.shape_reason);
    }
    const auto reason = [&](uint32_t a) {
        return why[a].empty() ? std::string("does not fit the worker cores") : why[a];
    };
    return composite(in, fmt::format("unicast {}; multicast {}", reason(kUc), reason(kMc)));
}

bool dir_active(const AgPlan& plan, uint32_t dev, uint32_t dir) {
    return plan.ring || (dir == kFwd ? dev + 1 < plan.N : dev > 0);
}

WorkerSched schedule(const AgPlan& plan, uint32_t dev, uint32_t link, uint32_t dir, uint32_t w) {
    const uint32_t N = plan.N;
    const bool unicast = plan.choice.algo == AgAlgo::Unicast;
    TT_FATAL(
        (unicast || plan.choice.algo == AgAlgo::Multicast) && dev < N && link < plan.choice.links &&
            dir < (unicast ? 2u : 1u) && w < plan.choice.n,
        "all_gather: no {} worker (dev {}, link {}, dir {}, w {})",
        algo_name(plan.choice.algo),
        dev,
        link,
        dir,
        w);
    const Slice s = plan.slices.at((link * plan.choice.n) + w);
    const uint32_t E = plan.geo.E;
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
    const uint64_t iter_packets = plan.geo.cap == 0 ? 0 : div_up(uint64_t{ws.iter_units} * plan.geo.A, plan.geo.cap);
    const uint64_t cadence_packets =
        std::clamp<uint64_t>(div_up(iter_packets, t.signals_per_iter), t.cadence_min_packets, t.cadence_max_packets);
    ws.cadence_slots = static_cast<uint32_t>(div_up(cadence_packets, plan.choice.k));
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
    std::array<uint32_t, 4> hops{};
    uint32_t devices = 1;
    for (uint32_t a = 0; a < 2; ++a) {
        const uint32_t size = mesh_shape[a];
        if (size <= 1 || axis.value_or(a) != a) {
            continue;
        }
        TT_FATAL(size < 256 && my_coord[a] < size, "all_gather: bad mesh axis {}: {} of {}", a, my_coord[a], size);
        devices *= size;
        const uint32_t fwd = axis_ring[a] ? size / 2 : size - 1 - my_coord[a];
        hops[a == 1 ? kEast : kSouth] = fwd;
        hops[a == 1 ? kWest : kNorth] = axis_ring[a] ? size - 1 - fwd : my_coord[a];
    }
    TT_FATAL(devices == N, "all_gather: mesh axes hold {} devices, the plan {}", devices, N);

    // E/W connections are lines; N/S connections are spines that branch E and W at every hop, so the connections cover
    // every other device exactly once.
    auto range = [&](uint32_t d) {
        uint32_t r = 0;
        auto put = [&](uint32_t logical) {
            if (hops[logical] > 0) {  // an absent branch must not clobber a live slot
                r |= hops[logical] << (8 * phys_dir[logical]);
            }
        };
        put(d);
        if (d == kNorth || d == kSouth) {
            put(kEast);
            put(kWest);
        }
        return r;
    };
    std::vector<McastConn> conns;
    for (uint32_t d = kEast; d <= kSouth; ++d) {
        if (hops[d] > 0) {
            conns.push_back(McastConn{.dir = d, .range = range(d)});
        }
    }
    return conns;
}

}  // namespace ttnn::operations::ccl::all_gather_plan
