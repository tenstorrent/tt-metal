// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-logger/tt-logger.hpp>
#include "ttnn/per_device_program_merge.hpp"
#include "sparse_kv_dispatch_program_factory.hpp"
#include "sparse_kv_dispatch_device_operation.hpp"
#include "sparse_kv_dispatch_layout.hpp"
#include "kernels/kvd3_common.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <cstdlib>
#include <string>
#include <vector>

#include <internal/fabric.hpp>
#include <internal/mesh_device.hpp>
#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/workload/legacy/host_api.hpp>
#include <tt-metalium/utils/math.hpp>
#include <tt-metalium/workload/legacy/tensor_accessor_args.hpp>
#include <tt-metalium/utils/work_split.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/experimental/sparse_kv_push/common/sparse_kv_push_common.hpp"
#include "ttnn/operations/experimental/sparse_kv_push/common/kernels/sparse_push_constants.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

constexpr tt::tt_metal::KernelHandle kSenderKernel = 0;
constexpr tt::tt_metal::KernelHandle kUnpackerKernel = 1;
constexpr tt::tt_metal::KernelHandle kPlannerKernel = 2;

// Port-core CBs (the landing ring first: the same L1 address on every chip, which remote senders rely on).
constexpr uint32_t kCbLand = 0, kCbSend = 1, kCbQin = 2, kCbMisc = 3, kCbUstage = 4;
// Planner-core CBs.
constexpr uint32_t kCbInbox = 0, kCbStage = 1, kCbRow = 2;
// Program semaphores: on the port cores, then on the planner cores.
constexpr uint32_t kSemOwnReady = 0, kSemUnpackDone = 1, kSemCreditOut = 2, kSemQTail0 = 3;  // + stream
// the mux termination handshake: one id on every port core, so a lane's increment lands on its master's word
constexpr uint32_t kSemMuxTermination = 6;
constexpr uint32_t kSemGo = 0;
constexpr uint8_t kMuxBuffersPerChannel = 8;

constexpr const char* kKernelDir = "ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/dispatch/device/kernels/";

std::string join(const std::vector<uint8_t>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) {
        s += (i ? "," : "") + std::to_string(v[i]);
    }
    return s;
}

CoreRangeSet to_range_set(const std::vector<CoreCoord>& cores) {
    std::set<CoreRange> ranges;
    for (const auto& c : cores) {
        ranges.insert(CoreRange(c, c));
    }
    return CoreRangeSet(ranges);
}

tt::tt_metal::KernelDescriptor make_kernel(
    const std::string& file,
    const CoreRangeSet& cores,
    std::vector<uint32_t> ct,
    tt::tt_metal::KernelDescriptor::Defines defines,
    tt::tt_metal::DataMovementProcessor processor,
    tt::tt_metal::NOC noc) {
    tt::tt_metal::KernelDescriptor k;
    k.kernel_source = std::string(kKernelDir) + file;
    k.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    k.core_ranges = cores;
    k.compile_time_args = std::move(ct);
    k.defines = std::move(defines);
    k.config = tt::tt_metal::DataMovementConfigDescriptor{.processor = processor, .noc = noc};
    return k;
}

// The kernels use a CB as a raw L1 region (its base from get_write_ptr, aligned up to 64 B): 64 B pages, one of slack.
void add_cb(
    tt::tt_metal::ProgramDescriptor& desc, const CoreRangeSet& cores, uint32_t index, uint32_t page, uint32_t pages) {
    const uint64_t bytes = uint64_t{page} * pages;
    TT_FATAL(bytes < (1u << 24), "sparse_kv_dispatch: a {} B CB does not fit L1", bytes);
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = static_cast<uint32_t>((bytes + 63) / 64 * 64 + 64),
        .core_ranges = cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index), .data_format = tt::DataFormat::RawUInt8, .page_size = 64}}},
    });
}

void common_args(tt::tt_metal::KernelDescriptor& k, const SparseKvDispatchParams& p, const SparseKvDispatchInputs& t) {
    tt::tt_metal::KernelDescriptor::RTArgList c;
    c.push_back(t.input_tensor.buffer());
    c.push_back(t.output_tensor.buffer());
    c.push_back(t.inbox.buffer());
    c.push_back(t.scratch.buffer());
    if (t.batch_index_tensor.has_value()) {
        c.push_back(t.batch_index_tensor->buffer());
    } else {
        c.push_back(0u);
    }
    c.push_back(p.batch_slot_num_layers);
    c.push_back(p.batch_slot_layer_idx);
    c.push_back(p.batch_index);
    c.push_back(static_cast<uint32_t>(p.fence_semaphore->address()));    // smuggled-rta-ok: semaphore
    c.push_back(static_cast<uint32_t>(p.arrived_semaphore->address()));  // smuggled-rta-ok: semaphore
    c.push_back(static_cast<uint32_t>(p.credit_semaphore->address()));   // smuggled-rta-ok: semaphore
    c.push_back(static_cast<uint32_t>(p.inbox_semaphore->address()));    // smuggled-rta-ok: semaphore
    c.push_back(t.table.buffer());
    if (t.rank_table.has_value()) {  // 13: the rank table (compact landing)
        c.push_back(t.rank_table->buffer());
    } else {
        c.push_back(0u);
    }
    k.emplace_common_runtime_args(c);
}

template <typename Args>
void patch_common(Args& c, const SparseKvDispatchParams& p, const SparseKvDispatchInputs& t) {
    c[0] = t.input_tensor.buffer()->address();
    c[1] = t.output_tensor.buffer()->address();
    c[2] = t.inbox.buffer()->address();
    c[3] = t.scratch.buffer()->address();
    c[4] = t.batch_index_tensor.has_value() ? t.batch_index_tensor->buffer()->address() : 0u;
    c[5] = p.batch_slot_num_layers;
    c[6] = p.batch_slot_layer_idx;
    c[7] = p.batch_index;
    c[8] = p.fence_semaphore->address();
    c[9] = p.arrived_semaphore->address();
    c[10] = p.credit_semaphore->address();
    c[11] = p.inbox_semaphore->address();
    c[12] = t.table.buffer()->address();
    c[13] = t.rank_table.has_value() ? t.rank_table->buffer()->address() : 0u;
}

// Ablation knobs (perf experiments only, results may not be exact): KVD_ABL=NAME[=VALUE],... defines KVD_ABL_NAME in
// every kernel (v2 and v3).
void add_abl_defines(tt::tt_metal::KernelDescriptor::Defines& defines) {
    if (const char* abl = std::getenv("KVD_ABL")) {
        std::string s(abl);
        for (size_t at = 0; at < s.size();) {
            size_t end = s.find(',', at);
            end = end == std::string::npos ? s.size() : end;
            if (end > at) {  // NAME or NAME=VALUE
                const std::string item = s.substr(at, end - at);
                const size_t eq = item.find('=');
                const std::string name = item.substr(0, eq);
                TT_FATAL(!name.empty(), "sparse_kv_dispatch: KVD_ABL has an empty knob name ('{}')", item);
                for (char ch : name) {
                    TT_FATAL(
                        (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_',
                        "sparse_kv_dispatch: KVD_ABL knob name '{}' must be [A-Z0-9_]",
                        name);
                }
                defines.emplace_back(
                    "KVD_ABL_" + item.substr(0, eq), eq == std::string::npos ? "1" : item.substr(eq + 1));
            }
            at = end + 1;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------- v3
constexpr uint32_t kSemWgo = 5;  // v3 worker cores: BRISC releases NCRISC after the chip barrier (WSPLIT=2)
// KVD_DESIGN=3 (experiment): kernels/kvd3_*.cpp, KV_DISPATCH_2D_PLAN.md 10.6. Knobs from
// KVD3="K=4,POOL=2,STAGE=2,IDLE=64".
struct V3Knobs {
    // (2026-11-22 sweep, plan 10.15). pstage 0: automatic, 2 (the old default) up to 8 while the stage CB stays at its
    // 4 G stage slots (8 on the 8x4 torus: K3.1 1M 1081 -> 1052 / 942 -> 890 us, GLM flat; fused plan 11).
    uint32_t K = 3, pool = 5, stage = 2, idle = 64, pstage = 0, lanes = 1, wsplit = 2;
};
V3Knobs v3_knobs() {
    V3Knobs k;
    if (const char* e = std::getenv("KVD3")) {
        std::string s(e);
        for (size_t at = 0; at < s.size();) {
            size_t end = s.find(',', at);
            end = end == std::string::npos ? s.size() : end;
            const std::string item = s.substr(at, end - at);
            const size_t eq = item.find('=');
            if (eq != std::string::npos) {
                const std::string key = item.substr(0, eq);
                const unsigned long raw = std::stoul(item.substr(eq + 1));
                TT_FATAL(raw <= 4096, "sparse_kv_dispatch v3: KVD3 {} out of range", item);
                const uint32_t v = static_cast<uint32_t>(raw);
                if (key == "K") {
                    k.K = v;
                } else if (key == "POOL") {
                    k.pool = v;
                } else if (key == "STAGE") {
                    k.stage = v;
                } else if (key == "IDLE") {
                    k.idle = v;
                } else if (key == "PSTAGE") {
                    TT_FATAL(v >= 1 && v <= 16, "sparse_kv_dispatch: KVD3=PSTAGE= must be in 1..16, got {}", v);
                    k.pstage = v;
                } else if (key == "LANES") {
                    k.lanes = v;
                } else if (key == "WSPLIT") {
                    k.wsplit = v;
                }
            }
            at = end + 1;
        }
    }
    return k;
}

tt::tt_metal::ProgramDescriptor create_descriptor_v3(
    const SparseKvDispatchParams& p, const SparseKvDispatchInputs& t, const ttnn::MeshCoordinate& coord) {
    using tt::tt_metal::distributed::MeshCoordinate;
    tt::tt_metal::ProgramDescriptor desc;
    auto* mesh_device = t.input_tensor.device();
    const auto& view = mesh_device->get_view();
    const uint32_t N = view.shape().mesh_size();
    const auto mesh = kv_dispatch_mesh(mesh_device, p.num_links);
    const uint32_t R = mesh.rows, C = mesh.cols;
    const uint32_t me = ttnn::operations::ccl::common::get_linearized_index(coord, view);
    // Classes: a row's hop generation is below (its first leg's hops) + (its turn's hops) <= the deepest span on X plus
    // the deepest on Y (the span tables' largest forward or backward hop count; a line's is n - 1). Tighter than
    // gmax + 1 (8 vs 10 on an 8x4 torus): fewer pools and staging classes in L1.
    auto deepest = [](uint32_t n, bool wrap) {
        if (!wrap) {
            return n - 1;
        }
        uint32_t m = 0;
        for (uint8_t e : ring_span_table(n, wrap)) {
            m = std::max<uint32_t>(m, std::max<uint32_t>(e & 0xF, e >> 4));
        }
        return m;
    };
    const uint32_t G = std::getenv("KVD3_GMAX_CLASSES")
                           ? mesh.gmax() + 1
                           : std::min(mesh.gmax() + 1, deepest(C, mesh.wrap_x) + deepest(R, mesh.wrap_y));
    const uint32_t L = p.num_links;
    V3Knobs kn = v3_knobs();
    if (kn.pstage == 0) {  // automatic: 2 (the old default), up to 8 while the stage CB keeps its 4 G stage slots
        kn.pstage = std::clamp<uint32_t>(G * kn.stage / kvd::kMaxLinks, 2u, 8u);
    }
    const uint32_t K = kn.K, P = p.planners;
    TT_FATAL(K >= 1 && K <= kvd3::kMaxK, "sparse_kv_dispatch v3: K {} out of range", K);
    TT_FATAL(
        G <= kvd3::kMaxClasses && P + 4 * K <= kvd3::kMaxProducers,
        "sparse_kv_dispatch v3: too many classes/producers");
    TT_FATAL(
        (kn.lanes == 1 || (kn.lanes == 2 && K % 2 == 0 && P % 2 == 0)) && (kn.wsplit == 1 || kn.wsplit == 2) &&
            L <= kvd::kMaxLinks && kn.stage >= 1 && kn.stage <= 16 && kn.pool >= 1 && kn.pool <= 16 && kn.pstage >= 1 &&
            kn.pstage <= 16 && kn.idle >= 1,
        "sparse_kv_dispatch v3: bad links/stage/pool/idle");

    auto neighbour_of = [&](const MeshCoordinate& at, uint32_t d) -> std::optional<MeshCoordinate> {
        int32_t r = static_cast<int32_t>(at[0]), c = static_cast<int32_t>(at[1]);
        const bool wrap = d < 2 ? mesh.wrap_x : mesh.wrap_y;
        const int32_t n = static_cast<int32_t>(d < 2 ? C : R);
        int32_t& v = d < 2 ? c : r;
        v += (d % 2 == 0) ? 1 : -1;
        if (!wrap && (v < 0 || v >= n)) {
            return std::nullopt;
        }
        v = (v % n + n) % n;
        return MeshCoordinate(static_cast<uint32_t>(r), static_cast<uint32_t>(c));
    };
    // per chip: TX(d, l), W(d, l, k) (the workers receiving from the neighbour in d on link l), planners
    struct Chip {
        std::array<std::optional<FabricDirection>, 4> dir;
        std::array<std::array<CoreCoord, kvd::kMaxLinks>, 4> tx{};
        std::array<std::array<std::array<CoreCoord, kvd3::kMaxK>, kvd::kMaxLinks>, 4> w{};
        std::array<std::array<CoreCoord, kvd::kMaxLinks>, 4> hlp{};  // lane-1 reader of TX(d, l) (LANES=2)
        std::vector<CoreCoord> planners;
    };
    const auto cores = corerange_to_cores(p.core_grid, std::nullopt, true);
    // The fabric directions of chip `at`, by mesh direction d (0 +X, 1 -X, 2 +Y, 3 -Y); none where it has no neighbour.
    auto dirs_of = [&](const MeshCoordinate& at) {
        std::array<std::optional<FabricDirection>, 4> dirs;
        const auto at_dirs = fabric_directions(mesh_device, at, L);
        for (uint32_t d = 0; d < 4; ++d) {
            const auto tc = neighbour_of(at, d);
            for (const auto& x : at_dirs) {
                if (tc && x.neighbor == *tc) {
                    dirs[d] = x;
                }
            }
        }
        return dirs;
    };
    // Placement, one program per mesh (as high_bw_all_gather): each chip puts its port cores next to its own Ethernet
    // cores, and the program runs each role's kernel on the union of that role's cores over the chips; a core that is
    // another chip's TX (or worker, or helper) idles on this one (present = 0: it only takes part in the chip barrier).
    // Roles get disjoint cores so that no core carries two kernels: TX cores first (no chip's worker or helper takes
    // a core any chip's TX took), then workers, then helpers; planners on cores of no role, the same on every chip.
    // When the unions do not fit the grid (a small sub-device grid), every chip uses one placement: each direction's
    // cores next to the Ethernet cores of the first chip that has it (uniform; slower on chips whose Ethernet cores
    // are elsewhere: GLM 1M real top-k up to +38%).
    using CoreKey = std::pair<size_t, size_t>;
    auto key_of = [](const CoreCoord& c) { return CoreKey{c.x, c.y}; };
    const auto mesh_coords = tt::tt_metal::distributed::MeshCoordinateRange(view.shape());
    std::map<std::pair<uint32_t, uint32_t>, Chip> placed;
    std::array<std::set<CoreKey>, 3> role_cores;  // 0 TX, 1 workers, 2 helpers: the union over the chips
    auto nearest_free = [&](tt::tt_metal::IDevice* dev,
                            const CoreCoord& eth,
                            const std::set<CoreKey>& taken,
                            const std::set<CoreKey>& blocked) -> std::optional<CoreCoord> {
        const auto closest =
            tt::tt_metal::experimental::Device::get_closest_worker_to_eth_core(*dev, eth, tt::tt_metal::NOC::NOC_1);
        std::optional<CoreCoord> best;
        uint32_t best_hops = std::numeric_limits<uint32_t>::max();
        for (const auto& cand : cores) {
            const auto k = key_of(cand);
            if (taken.contains(k) || blocked.contains(k) || cand == cores.back()) {  // (cores.back(): planner 0)
                continue;
            }
            const uint32_t h =
                (cand == closest.logical_coord ? 0u
                                               : tt::tt_metal::experimental::Device::get_worker_noc_hop_distance(
                                                     dev, cand, closest.logical_coord, tt::tt_metal::NOC::NOC_1)) +
                closest.distance_in_noc_hops;
            if (h < best_hops) {
                best_hops = h;
                best = cand;
            }
        }
        return best;
    };
    // a chip's Ethernet core of (d, l), or none when the chip has no direction d
    auto eth_of = [&](const MeshCoordinate& at, const Chip& ch, uint32_t d, uint32_t l) -> std::optional<CoreCoord> {
        if (!ch.dir[d]) {
            return std::nullopt;
        }
        return tt::tt_fabric::get_forwarding_eth_core(
            *mesh_device,
            mesh_device->get_fabric_node_id(at),
            mesh_device->get_fabric_node_id(*neighbour_of(at, d)),
            ch.dir[d]->links[l]);
    };
    auto choose_planners = [&](const std::set<CoreKey>& used) -> std::optional<std::vector<CoreCoord>> {
        std::vector<CoreCoord> pl{cores.back()};
        for (const auto& c : cores) {
            if (pl.size() == P) {
                break;
            }
            if (c != cores.back() && !used.contains(key_of(c))) {
                pl.push_back(c);
            }
        }
        return pl.size() == P ? std::optional(pl) : std::nullopt;
    };
    auto place_per_chip = [&]() -> bool {
        placed.clear();
        for (auto& r : role_cores) {
            r.clear();
        }
        for (const auto& at : mesh_coords) {
            placed[{at[0], at[1]}].dir = dirs_of(at);
        }
        std::map<std::pair<uint32_t, uint32_t>, std::set<CoreKey>> taken;  // per chip, every role
        for (uint32_t role = 0; role < (kn.lanes == 2 ? 3u : 2u); ++role) {
            std::set<CoreKey> blocked;  // the earlier roles' cores on any chip
            for (uint32_t r = 0; r < role; ++r) {
                blocked.insert(role_cores[r].begin(), role_cores[r].end());
            }
            for (const auto& at : mesh_coords) {
                Chip& ch = placed[{at[0], at[1]}];
                auto& tk = taken[{at[0], at[1]}];
                auto* dev = tt::tt_metal::internal::get_device(*mesh_device, at);
                for (uint32_t d = 0; d < 4; ++d) {
                    for (uint32_t l = 0; l < L; ++l) {
                        const auto eth = eth_of(at, ch, d, l);
                        if (!eth) {
                            continue;
                        }
                        const uint32_t n = role == 1 ? K : 1;
                        for (uint32_t k = 0; k < n; ++k) {
                            const auto c = nearest_free(dev, *eth, tk, blocked);
                            if (!c) {
                                return false;
                            }
                            tk.insert(key_of(*c));
                            role_cores[role].insert(key_of(*c));
                            (role == 0 ? ch.tx[d][l] : role == 1 ? ch.w[d][l][k] : ch.hlp[d][l]) = *c;
                        }
                    }
                }
            }
        }
        std::set<CoreKey> used;
        for (const auto& r : role_cores) {
            used.insert(r.begin(), r.end());
        }
        const auto pl = choose_planners(used);
        if (!pl) {
            return false;
        }
        for (auto& [_, ch] : placed) {
            ch.planners = *pl;
        }
        return true;
    };
    auto place_uniform = [&]() {
        placed.clear();
        for (auto& r : role_cores) {
            r.clear();
        }
        Chip uniform;
        std::set<CoreKey> taken;
        for (uint32_t d = 0; d < 4; ++d) {
            for (const auto& at : mesh_coords) {
                Chip probe;
                probe.dir = dirs_of(at);
                if (!probe.dir[d]) {
                    continue;
                }
                auto* dev = tt::tt_metal::internal::get_device(*mesh_device, at);
                for (uint32_t l = 0; l < L; ++l) {
                    const auto eth = *eth_of(at, probe, d, l);
                    auto pick = [&](uint32_t role) {
                        const auto c = nearest_free(dev, eth, taken, {});
                        TT_FATAL(
                            c.has_value(),
                            "sparse_kv_dispatch v3: too few grid cores for the port cores (directions x links x (1 + "
                            "K, + 1 helper with LANES=2), plus planner 0; K = {}): lower K (KVD3=K=..)",
                            K);
                        taken.insert(key_of(*c));
                        role_cores[role].insert(key_of(*c));
                        return *c;
                    };
                    uniform.tx[d][l] = pick(0);
                    for (uint32_t k = 0; k < K; ++k) {
                        uniform.w[d][l][k] = pick(1);
                    }
                    if (kn.lanes == 2) {
                        uniform.hlp[d][l] = pick(2);
                    }
                }
                break;
            }
        }
        const auto pl = choose_planners(taken);
        TT_FATAL(pl.has_value(), "sparse_kv_dispatch v3: too few cores for {} planners", P);
        uniform.planners = *pl;
        for (const auto& at : mesh_coords) {
            Chip ch = uniform;
            ch.dir = dirs_of(at);
            placed[{at[0], at[1]}] = std::move(ch);
        }
    };
    const bool per_chip = place_per_chip();
    log_info(
        tt::LogOp,
        "sparse_kv_dispatch placement on {} cores, {} planners: per-chip {} (TX {} / workers {} / helpers {} cores "
        "over "
        "the chips)",
        cores.size(),
        P,
        per_chip ? "fits" : "does not fit, uniform",
        role_cores[0].size(),
        role_cores[1].size(),
        role_cores[2].size());
    if (!per_chip) {
        place_uniform();
    }
    auto place = [&](const MeshCoordinate& at) -> const Chip& { return placed.at({at[0], at[1]}); };
    const Chip& mine = place(coord);
    auto virt = [&](const CoreCoord& c) { return mesh_device->worker_core_from_logical_core(c); };

    // every role's cores over the chips (the program runs each role's kernel on all of them), then the planners
    auto sorted_cores = [](const std::set<CoreKey>& keys) {
        std::vector<CoreCoord> v;
        for (const auto& [x, y] : keys) {
            v.emplace_back(x, y);
        }
        return v;
    };
    std::vector<CoreCoord> tx_cores = sorted_cores(role_cores[0]), w_cores = sorted_cores(role_cores[1]),
                           hlp_cores = sorted_cores(role_cores[2]), all_cores;
    all_cores = tx_cores;
    all_cores.insert(all_cores.end(), w_cores.begin(), w_cores.end());
    all_cores.insert(all_cores.end(), hlp_cores.begin(), hlp_cores.end());
    all_cores.insert(all_cores.end(), mine.planners.begin(), mine.planners.end());
    const auto tx_range = to_range_set(tx_cores);
    const auto w_range = to_range_set(w_cores);
    auto send_cores = tx_cores;
    send_cores.insert(send_cores.end(), hlp_cores.begin(), hlp_cores.end());
    const auto send_range = to_range_set(send_cores);  // TX and helpers: ring buffers at the same address
    const auto planner_range = to_range_set(mine.planners);
    const auto all_range = to_range_set(all_cores);
    const uint32_t n_cores = static_cast<uint32_t>(all_cores.size());

    // geometry
    const uint32_t row_bytes = t.input_tensor.buffer()->aligned_page_size();
    const auto max_payload = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());
    uint32_t B = 0;
    while (kvd::slot_rows_offset(B + 1) + (B + 1) * row_bytes <= max_payload) {
        ++B;
    }
    if (p.rows_per_packet) {
        B = std::min(B, p.rows_per_packet);
    }
    TT_FATAL(B >= 1, "sparse_kv_dispatch v3: a {} B row does not fit the {} B payload", row_bytes, max_payload);
    uint32_t sbytes = kvd::slot_bytes(B, row_bytes);  // (the L1 fit below may take fewer rows a packet)
    // compact landing (a rank table given): rows land at compact_page(run_base[owner], rank) (sparse_push_plan)
    const bool compact = t.rank_table.has_value();
    const uint32_t rank_page_words = std::max(2u * p.stripe_chunk / 32u, 32u);
    const uint32_t rank_page_bytes = rank_page_words * 4;
    const uint32_t owner_slot_rows = t.output_tensor.padded_shape()[2] / N;
    const uint32_t inbox_slot = inbox_slot_bytes(p.max_rows, p.stripe_chunk, N);
    const uint32_t num_slots = t.input_tensor.padded_shape()[0];
    const uint32_t pages_per_slot = t.input_tensor.buffer()->num_pages() / num_slots;
    const bool has_batch = t.batch_index_tensor.has_value();
    const uint32_t owner_blocks = tt::div_up(p.max_rows / p.stripe_chunk, N);

    // CBs: ctrl, stage and pool on every core, first and in this order (the same L1 address on every core of every
    // chip: remote cores address them); then each role's own.
    // the reader ring's depth (kvd3_common.hpp kRing: 8, or KVD_ABL=RING=n), for the send CB
    // (from the same parsed defines the kernels compile with; one RING at most, canonical decimal, so the host and the
    // kernel's KVD_ABL_RING literal cannot read it differently)
    uint32_t ring_depth = 8;
    tt::tt_metal::KernelDescriptor::Defines abl_defs;  // parsed once: sizing here and the kernels' defines below
    add_abl_defines(abl_defs);
    {
        // Stamped arrivals (KVD_SEQ, kvd3_common.hpp): packets land without the router's per-packet write flush,
        // which serialises every link at ~0.25 us a packet (K3.1 1M, 7168 B: 1051 -> 838 us). The stamp sits in row
        // padding, so it needs 16 B of it; else, or with KVD_ABL=NOSEQ, packets flush as before (PIGGY,
        // BATCH_ARRIVE and NO_STATEFUL keep their own arrival path; NOFLUSH_ARRIVE drops both: timing only).
        // (both tensors: the stamp also reaches the output's rows, in their padding)
        const uint32_t logical = std::max(
            t.input_tensor.logical_shape()[-1] * t.input_tensor.element_size(),
            t.output_tensor.logical_shape()[-1] * t.output_tensor.element_size());
        // A packet must be one NoC burst (Blackhole: 256 x 64 B, noc_parameters.h): the router's local write is then a
        // single NoC packet, and we rely on its flits landing in order, so the trailer stamp is written last. Blackhole
        // only (Wormhole's burst is 8 KB).
        constexpr uint32_t kMaxNocBurstBlackhole = 16384;
        bool seq = t.input_tensor.device()->arch() == tt::ARCH::BLACKHOLE && max_payload <= kMaxNocBurstBlackhole &&
                   row_bytes >= logical + 16 && t.output_tensor.buffer()->aligned_page_size() == row_bytes;
        for (const auto& [name, value] : abl_defs) {
            if (name == "KVD_ABL_NOSEQ" || name == "KVD_ABL_PIGGY" || name == "KVD_ABL_BATCH_ARRIVE" ||
                name == "KVD_ABL_NOFLUSH_ARRIVE" || name == "KVD_ABL_NO_STATEFUL") {
                seq = false;
            }
        }
        if (seq) {
            abl_defs.emplace_back("KVD_SEQ", "1");
        }
    }
    // KVD_ABL=PSPLIT=S: the planners' units are 1/S of a 2-word group, so a planner reads up to S/2 x the lines
    uint32_t psplit = 1, pdeal = 2;
    for (const auto& [name, value] : abl_defs) {
        const bool num = !value.empty() && value.find_first_not_of("0123456789") == std::string::npos;
        if (name == "KVD_ABL_PSPLIT" || name == "KVD_ABL_HSPLIT" || name == "KVD_ABL_PDEAL") {
            TT_FATAL(
                num && value.size() <= 2, "sparse_kv_dispatch v3: KVD_ABL {}={} must be a small decimal", name, value);
            const uint32_t v = static_cast<uint32_t>(std::stoul(value));
            TT_FATAL(
                (name == "KVD_ABL_PSPLIT" ? v <= 32 : v >= 1 && v <= 32) && (v == 0 || 32 % v == 0),
                "sparse_kv_dispatch v3: KVD_ABL {}={} must divide 32",
                name,
                value);
            if (name == "KVD_ABL_PSPLIT") {
                psplit = std::max(v, 1u);
            } else if (name == "KVD_ABL_PDEAL") {
                pdeal = v;
            }
        }
        if (name == "KVD_ABL_TX_DELAY" || name == "KVD_ABL_P_DELAY") {
            TT_FATAL(
                num && value.size() <= 9, "sparse_kv_dispatch v3: KVD_ABL {}={} must be < 1e9 cycles", name, value);
        }
    }
    const uint32_t psplit_lines = std::max(1u, psplit / std::min(pdeal, psplit));
    // The planner's line sets (one a dealt unit; kvd3_planner.cpp): the whole groups' share, times PSPLIT / 2, plus
    // 4 for the adaptive cold split (at most ~2 units a planner) and the split hot groups. The kernel splits only
    // within this.
    const uint32_t planner_units = tt::div_up(owner_blocks * (p.stripe_chunk / 32) * psplit_lines, p.planners) + 4;
    {
        uint32_t rings_seen = 0;
        for (const auto& [name, value] : abl_defs) {
            if (name == "KVD_ABL_RING") {
                TT_FATAL(++rings_seen == 1, "sparse_kv_dispatch v3: KVD_ABL sets RING more than once");
                TT_FATAL(
                    !value.empty() && value[0] >= '1' && value[0] <= '9',
                    "sparse_kv_dispatch v3: KVD_ABL RING={} must be a decimal without leading zeros or sign",
                    value);
                size_t used = 0;
                unsigned long v = 0;
                try {
                    v = std::stoul(value, &used, 10);
                } catch (const std::exception&) {
                    used = 0;
                }
                TT_FATAL(
                    used == value.size() && v >= 1 && v <= kvd3::kMaxRing,
                    "sparse_kv_dispatch v3: KVD_ABL RING={} must be a decimal 1..{}",
                    value,
                    kvd3::kMaxRing);
                ring_depth = static_cast<uint32_t>(v);
            }
        }
    }
    constexpr uint32_t kCtrl = 0, kStage = 1, kPool = 2, kSend = 3, kInbox = 4, kRow = 5, kRank = 6;
    // (on the whole grid, not just this chip's cores: a CB's base is the L1 high-water mark of the cores it is on, and
    // the placed cores differ by chip; on the grid it is the same on every chip)
    const auto& grid_range = p.core_grid;
    // Fit the landing and staging CBs to the free L1, [allocator base, lowest live L1 buffer) as sparse_sdpa_msa
    // computes it: both scale with the classes G (more on lines and partial tori) and the slot size. Cheapest first
    // (GLM 1M real top-k, 8x4): pool depth to 2 (~4%), then fewer rows a packet, halving to 4 (8 rows: ~20% on the
    // full grid, faster on 40 cores), and staging depth last (1 is ~6x slower: TX and workers stop overlapping).
    {
        bool pool0 = false;
        for (const auto& [name, value] : abl_defs) {
            pool0 = pool0 || name == "KVD_ABL_POOL0";
        }
        const uint64_t align = 64;
        auto cb = [&](uint64_t bytes) { return tt::round_up(bytes, align); };
        const uint64_t l1_base = mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
        const uint64_t l1_end = l1_base + mesh_device->allocator()->get_bank_size(tt::tt_metal::BufferType::L1);
        const uint64_t l1_top =
            std::min<uint64_t>(mesh_device->lowest_occupied_compute_l1_address().value_or(l1_end), l1_end);
        const uint64_t free_l1 = l1_top - l1_base;
        const uint64_t ctrl = cb(4 * kvd3::kCtrlWords);
        const uint64_t send = cb(
            std::max<uint64_t>(kvd3::send_cb_bytes(sbytes, ring_depth, kn.lanes), tt::round_up(max_payload, 64) + 64));
        const uint64_t inbox = cb(std::max(inbox_slot, N * 64 + planner_units * N * 64));
        // (the copy batch, plus KVD_ABL=PDEFER's deferred lists, counted always)
        const uint64_t rows =
            cb(static_cast<uint64_t>(std::max(row_bytes, 64u)) * 8 + 4 * kvd::kMaxLinks * 32 * 16 + row_bytes);
        auto need = [&]() {
            const uint64_t stage = cb(uint64_t{sbytes} * std::max(4 * G * kn.stage, 4 * kvd::kMaxLinks * kn.pstage));
            const uint64_t rank = compact ? cb(uint64_t{rank_page_bytes} * (3 + 2 * std::max(B, 8u))) : 0;
            const uint64_t port = ctrl + rank + stage + cb(uint64_t{sbytes} * G * kn.pool) + send;
            const uint64_t planner = ctrl + rank + stage + inbox + rows + 64;
            return std::max(port, planner);
        };
        const uint32_t pool_in = kn.pool, stage_in = kn.stage, b_in = B;
        while (!pool0 && need() > free_l1 && (kn.pool > 2 || B > 4 || kn.stage > 1)) {
            if (kn.pool > 2) {
                --kn.pool;
            } else if (B > 4) {
                B = (B + 1) / 2;
                sbytes = kvd::slot_bytes(B, row_bytes);
            } else {
                --kn.stage;
                kn.pstage = std::clamp<uint32_t>(G * kn.stage / kvd::kMaxLinks, 2u, 8u);
            }
        }
        if (kn.pool != pool_in || kn.stage != stage_in || B != b_in) {
            log_info(
                tt::LogOp,
                "sparse_kv_dispatch L1 fit: G {} stage {} pool {} pstage {} rows/packet {}: need {} B of {} B free (L1 "
                "base "
                "{}, top {})",
                G,
                kn.stage,
                kn.pool,
                kn.pstage,
                B,
                need(),
                free_l1,
                l1_base,
                l1_top);
        }
        TT_FATAL(
            pool0 || need() <= free_l1,
            "sparse_kv_dispatch: {} classes do not fit L1 ({} B needed at stage 1, pool 2; {} B free): fewer planners "
            "or rows_per_packet",
            G,
            need(),
            free_l1);
    }
    add_cb(desc, grid_range, kCtrl, 4 * kvd3::kCtrlWords, 1);
    if (compact) {  // compact landing: the run bases (page 0), then the rank pages of a slot's (or copy batch's) rows
        add_cb(desc, grid_range, kRank, rank_page_bytes, 3 + 2 * std::max(B, 8u));  // (2 RISCs, + 64 B slack)
    }
    add_cb(desc, grid_range, kStage, sbytes, std::max(4 * G * kn.stage, 4 * kvd::kMaxLinks * kn.pstage));
    uint32_t pool_pages = G * kn.pool;  // (KVD_ABL=POOL0=n: class 0 has n slots)
    for (const auto& [name, value] : abl_defs) {
        if (name == "KVD_ABL_POOL0") {
            // canonical decimal 1..32 (the kernels read the literal: no leading zero, which C++ reads as octal)
            const bool num = !value.empty() && value.size() <= 2 && value[0] != '0' &&
                             value.find_first_not_of("0123456789") == std::string::npos;
            TT_FATAL(num && std::stoul(value) <= 32, "KVD_ABL=POOL0 must be 1..32 in canonical decimal");
            pool_pages = G * kn.pool - kn.pool + std::stoul(value);
        }
    }
    // Landing pools on the port cores only: planners never land packets, and their L1 holds the inbox copy (which
    // grows as planners get fewer). The pools still start at the same address on every chip: kCtrl and kStage, the
    // only CBs below them, cover the whole grid.
    std::vector<CoreCoord> port_cores = tx_cores;
    port_cores.insert(port_cores.end(), w_cores.begin(), w_cores.end());
    port_cores.insert(port_cores.end(), hlp_cores.begin(), hlp_cores.end());
    add_cb(desc, to_range_set(port_cores), kPool, sbytes, pool_pages);
    add_cb(
        desc,
        send_range,
        kSend,
        std::max(kvd3::send_cb_bytes(sbytes, ring_depth, kn.lanes), tt::round_up(max_payload, 64) + 64),
        1);  // the reader->writer ring, credits, descriptors
    add_cb(desc, planner_range, kInbox, std::max(inbox_slot, N * 64 + planner_units * N * 64), 1);
    {
        // the planner's copy batch (8 rows); KVD_ABL=PDEFER: then its deferred rows, 32 a port of 16 B
        uint32_t row_pages = 8;
        for (const auto& [name, value] : abl_defs) {
            if (name == "KVD_ABL_PDEFER") {
                row_pages += (4 * kvd::kMaxLinks * 32 * 16 + row_bytes - 1) / row_bytes;
            }
        }
        add_cb(desc, planner_range, kRow, std::max(row_bytes, 64u), row_pages);
    }
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = kSemGo, .core_type = tt::CoreType::WORKER, .core_ranges = planner_range, .initial_value = 0});
    constexpr uint32_t kSemTxGo = 1, kSemFilled = 2, kSemDrained = 3, kSemRdone = 4;  // TX cores: writer <-> reader
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = kSemWgo, .core_type = tt::CoreType::WORKER, .core_ranges = w_range, .initial_value = 0});
    for (uint32_t id : {kSemTxGo, kSemFilled, kSemDrained, kSemRdone}) {
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = id, .core_type = tt::CoreType::WORKER, .core_ranges = tx_range, .initial_value = 0});
    }

    // defines (as v2)
    tt::tt_metal::KernelDescriptor::Defines ring_defines = {
        {"KVD_XRING", join(ring_span_table(C, mesh.wrap_x))},
        {"KVD_YRING", join(ring_span_table(R, mesh.wrap_y))},
    };
    if (std::getenv("KVD_STATS") != nullptr) {
        ring_defines.emplace_back("KVD_STATS", "1");
    }
    ring_defines.insert(ring_defines.end(), abl_defs.begin(), abl_defs.end());
    // which rows of each requester this call sends (planners: kvd3_planner.cpp route_word)
    ring_defines.emplace_back("KVD_PART", std::to_string(p.part));
    ring_defines.emplace_back("KVD_SKIP_XCHG", p.skip_exchange ? "1" : "0");
    if (compact) {
        const auto shape = packet_shape(mesh_device, row_bytes);  // (the plan's compact layout: its packet shape)
        ring_defines.emplace_back("KVD_COMPACT", "1");
        ring_defines.emplace_back("KVD_PPP", std::to_string(shape.pages_per_packet));
        ring_defines.emplace_back("KVD_BANKS", std::to_string(shape.banks));
        ring_defines.emplace_back("KVD_RANK_PAGE_WORDS", std::to_string(rank_page_words));
        ring_defines.emplace_back("KVD_CB_RANK", std::to_string(kRank));
    }
    if (p.active_from != 0) {
        // every v3 kernel: return when the chunk starts before active_from (kvd3_common.hpp skip_chunk)
        TT_FATAL(kn.lanes == 1, "sparse_kv_dispatch active_from: LANES=1 only (the helper readers have no gate)");
        TT_FATAL(!t.plan_indices.has_value(), "sparse_kv_dispatch active_from: not with the fused plan");
        ring_defines.emplace_back("KVD_ACTIVE_FROM_BLOCK", std::to_string(p.active_from / p.stripe_chunk));
    }
    ring_defines.emplace_back("KVD3_RINGS", std::to_string(kn.lanes));  // (send CB layout: one ring per lane)
    ring_defines.emplace_back("KVD3_PLANNER_UNITS", std::to_string(planner_units));
    auto tx_defines = ring_defines;
    {
        std::vector<uint32_t> chip_ids, mesh_ids;
        for (const auto& c : tt::tt_metal::distributed::MeshCoordinateRange(view.shape())) {
            const auto node = mesh_device->get_fabric_node_id(c);
            chip_ids.push_back(static_cast<uint32_t>(node.chip_id));
            mesh_ids.push_back(static_cast<uint32_t>(*node.mesh_id));
        }
        tx_defines.emplace_back("DEST_CHIP_ID", ttnn::operations::ccl::common::stringify(chip_ids));
        tx_defines.emplace_back("DEST_MESH_ID", ttnn::operations::ccl::common::stringify(mesh_ids));
        for (const auto& d :
             tt::tt_metal::internal::get_fabric_kernel_defines(*mesh_device, tt::tt_fabric::FabricApiType::Mesh)) {
            tx_defines.push_back(d);
        }
    }
    auto acc = [](std::vector<uint32_t>& ct, const Tensor& x) {
        tt::tt_metal::TensorAccessorArgs(x.buffer()).append_to(ct);
    };
    const Tensor& batch_or_dummy = has_batch ? *t.batch_index_tensor : t.scratch;

    std::vector<uint32_t> tx_ct = {
        N,
        L,
        K,
        P,
        G,
        B,
        kn.pool,
        kn.stage,
        row_bytes,
        p.stripe_chunk,
        inbox_slot,
        max_payload,
        owner_blocks,
        kCtrl,
        kStage,
        kPool,
        kSend,
        n_cores,
        kSemTxGo,
        kSemFilled,
        kSemDrained,
        kSemRdone,
        kn.lanes,
        kn.wsplit};
    acc(tx_ct, t.table);
    std::vector<uint32_t> txr_ct = {L,        K,          P,           G,         B,         kn.pool,
                                    kn.stage, row_bytes,  kCtrl,       kStage,    kPool,     kSend,
                                    kSemTxGo, kSemFilled, kSemDrained, kSemRdone, kn.pstage, kn.lanes,
                                    0u,  // lane
                                    0u,  // push
                                    n_cores};
    auto hlp_ct = txr_ct;
    hlp_ct[18] = 1;  // lane 1
    hlp_ct[19] = 1;  // push into ring B
    acc(tx_ct, t.inbox);
    std::vector<uint32_t> w_ct = {
        N,        R,         C,         mesh.wrap_x ? 1u : 0u, mesh.wrap_y ? 1u : 0u, K,     P,      G,     B,
        kn.pool,  kn.stage,  row_bytes, p.stripe_chunk,        owner_slot_rows,       kCtrl, kStage, kPool, kn.idle,
        kn.lanes, kn.wsplit,
        0u,  // class parity (BRISC)
        kSemWgo};
    acc(w_ct, t.output_tensor);
    acc(w_ct, compact ? *t.rank_table : t.output_tensor);  // (the rank table; any valid accessor without one)
    std::vector<uint32_t> planner_ct = {
        N,
        R,
        C,
        mesh.wrap_x ? 1u : 0u,
        mesh.wrap_y ? 1u : 0u,
        L,
        p.stripe_chunk,
        owner_slot_rows,
        inbox_slot,
        row_bytes,
        P,
        kn.pstage,
        pages_per_slot,
        has_batch ? 1u : 0u,
        num_slots,
        owner_blocks,
        kInbox,
        kStage,
        kRow,
        G,
        B,
        kCtrl,
        n_cores,
        kn.lanes};
    acc(planner_ct, t.input_tensor);
    acc(planner_ct, t.output_tensor);
    acc(planner_ct, t.inbox);
    acc(planner_ct, t.scratch);
    acc(planner_ct, t.table);
    acc(planner_ct, batch_or_dummy);
    acc(planner_ct, compact ? *t.rank_table : t.output_tensor);

    auto tx = make_kernel(
        "kvd3_tx.cpp",
        tx_range,
        tx_ct,
        tx_defines,
        tt::tt_metal::DataMovementProcessor::RISCV_0,
        tt::tt_metal::NOC::RISCV_1_default);
    auto wk = make_kernel(
        "kvd3_worker.cpp",
        w_range,
        w_ct,
        ring_defines,
        tt::tt_metal::DataMovementProcessor::RISCV_0,
        tt::tt_metal::NOC::RISCV_0_default);
    auto w_ct1 = w_ct;
    w_ct1[20] = 1;  // the NCRISC serves the odd classes (WSPLIT=2)
    auto wk1 = make_kernel(
        "kvd3_worker.cpp",
        w_range,
        w_ct1,
        ring_defines,
        tt::tt_metal::DataMovementProcessor::RISCV_1,
        tt::tt_metal::NOC::RISCV_1_default);
    auto planner = make_kernel(
        "kvd3_planner.cpp",
        planner_range,
        planner_ct,
        ring_defines,
        tt::tt_metal::DataMovementProcessor::RISCV_0,
        tt::tt_metal::NOC::RISCV_0_default);
    auto txr = make_kernel(
        "kvd3_txr.cpp",
        tx_range,
        txr_ct,
        ring_defines,
        tt::tt_metal::DataMovementProcessor::RISCV_1,
        tt::tt_metal::NOC::RISCV_0_default);
    auto hlp = make_kernel(
        "kvd3_txr.cpp",
        to_range_set(hlp_cores.empty() ? tx_cores : hlp_cores),  // (no helpers: a kernel with no runtime args)
        hlp_ct,
        ring_defines,
        tt::tt_metal::DataMovementProcessor::RISCV_0,
        tt::tt_metal::NOC::RISCV_0_default);
    common_args(hlp, p, t);
    common_args(tx, p, t);
    common_args(wk, p, t);
    common_args(planner, p, t);

    const auto bar = virt(mine.planners[0]);
    const auto sync = bar;  // planner 0 is the same logical core on every chip
    // Per core: every core of a role's union starts idle (another chip's TX, worker or helper), then this chip's own
    // cores get their real args; each core's args are set once, below.
    std::map<CoreKey, std::vector<uint32_t>> tx_rt, txr_rt, hlp_rt, w_rt;
    {
        const std::vector<uint32_t> idle = {0u, static_cast<uint32_t>(bar.x), static_cast<uint32_t>(bar.y)};
        for (const auto& c : tx_cores) {
            tx_rt[key_of(c)] = idle;
            txr_rt[key_of(c)] = {0u};
        }
        for (const auto& c : w_cores) {
            w_rt[key_of(c)] = idle;
        }
        for (const auto& c : hlp_cores) {  // (the helper reads its TX core before the present check)
            hlp_rt[key_of(c)] = {0u, static_cast<uint32_t>(bar.x), static_cast<uint32_t>(bar.y), 0u, 0u};
        }
    }
    for (uint32_t d = 0; d < 4; ++d) {
        if (!mine.dir[d]) {
            continue;
        }
        const auto nb = *neighbour_of(coord, d);
        const Chip& nbc = place(nb);
        const auto nb_node = mesh_device->get_fabric_node_id(nb);
        uint32_t faces = 0;
        for (uint32_t f = 0; f < 4; ++f) {
            faces += mine.dir[f] ? 1 : 0;
        }
        for (uint32_t l = 0; l < L; ++l) {
            // TX(d, l)
            std::vector<uint32_t> a = {
                1u,
                static_cast<uint32_t>(bar.x),
                static_cast<uint32_t>(bar.y),
                d,
                l,
                me,
                mine.dir[d]->eth_direction,
                static_cast<uint32_t>(nb_node.chip_id),
                static_cast<uint32_t>(*nb_node.mesh_id)};
            const auto nbtx = virt(nbc.tx[d ^ 1][l]);
            a.push_back(nbtx.x);
            a.push_back(nbtx.y);
            a.push_back(sync.x);
            a.push_back(sync.y);
            for (uint32_t k = 0; k < K; ++k) {
                const auto c = virt(nbc.w[d ^ 1][l][k]);
                a.push_back(c.x);
                a.push_back(c.y);
            }
            for (uint32_t i = 0; i < P; ++i) {
                const auto c = virt(mine.planners[i]);
                a.push_back(c.x);
                a.push_back(c.y);
            }
            for (uint32_t f = 0; f < 4; ++f) {
                for (uint32_t k = 0; k < K; ++k) {
                    const auto c = mine.dir[f] ? virt(mine.w[f][l][k]) : CoreCoord{0, 0};
                    a.push_back(c.x);
                    a.push_back(c.y);
                }
            }
            a.push_back(faces * K);
            {  // the reader: present, d, l, producers, workers; the helper: present, barrier, TX core, then the same
                std::vector<uint32_t> r = {1u, d, l};
                r.insert(r.end(), a.end() - (2 * (P + 4 * K) + 1), a.end());
                txr_rt[key_of(mine.tx[d][l])] = r;
                if (kn.lanes == 2) {
                    const auto txc = virt(mine.tx[d][l]);
                    std::vector<uint32_t> h = {
                        1u, static_cast<uint32_t>(bar.x), static_cast<uint32_t>(bar.y), txc.x, txc.y, d, l};
                    h.insert(h.end(), r.begin() + 3, r.end());
                    hlp_rt[key_of(mine.hlp[d][l])] = h;
                }
            }
            {  // my helper, and the neighbour TX's helper (credit copies)
                const auto mh = kn.lanes == 2 ? virt(mine.hlp[d][l]) : CoreCoord{0, 0};
                const auto nh = kn.lanes == 2 ? virt(nbc.hlp[d ^ 1][l]) : CoreCoord{0, 0};
                a.push_back(mh.x);
                a.push_back(mh.y);
                a.push_back(nh.x);
                a.push_back(nh.y);
            }
            tt::tt_fabric::append_fabric_connection_rt_args(
                *mesh_device,
                mesh_device->get_fabric_node_id(coord),
                nb_node,
                mine.dir[d]->links[l],
                desc,
                mine.tx[d][l],
                a);
            tx_rt[key_of(mine.tx[d][l])] = a;
            // W(d, l, k)
            uint32_t tx_present = 0;
            for (uint32_t td = 0; td < 4; ++td) {
                tx_present |= (mine.dir[td] ? 1u : 0u) << td;
            }
            for (uint32_t k = 0; k < K; ++k) {
                std::vector<uint32_t> w = {
                    1u,
                    static_cast<uint32_t>(bar.x),
                    static_cast<uint32_t>(bar.y),
                    d,
                    l,
                    k,
                    me,
                    kvd3::worker_producer(P, K, d, k),
                    tx_present};
                for (uint32_t td = 0; td < 4; ++td) {
                    const auto c = mine.dir[td] ? virt(mine.tx[td][l]) : CoreCoord{0, 0};
                    w.push_back(c.x);
                    w.push_back(c.y);
                }
                // this worker's lane reader of each TX(td, l): lane = producer index parity (P even: k parity)
                const bool lane1 = kn.lanes == 2 && (kvd3::worker_producer(P, K, d, k) & 1) != 0;
                for (uint32_t td = 0; td < 4; ++td) {
                    const auto c = !mine.dir[td] ? CoreCoord{0, 0}
                                   : lane1       ? virt(mine.hlp[td][l])
                                                 : virt(mine.tx[td][l]);
                    w.push_back(c.x);
                    w.push_back(c.y);
                }
                w_rt[key_of(mine.w[d][l][k])] = w;
            }
        }
    }
    {
        auto emit = [](tt::tt_metal::KernelDescriptor& k, const std::map<CoreKey, std::vector<uint32_t>>& rt_of) {
            for (const auto& [c, v] : rt_of) {
                tt::tt_metal::KernelDescriptor::RTArgList rt;
                rt.append(v);
                k.emplace_runtime_args(CoreCoord(c.first, c.second), rt);
            }
        };
        emit(tx, tx_rt);
        emit(txr, txr_rt);
        emit(wk, w_rt);
        if (kn.wsplit == 2) {
            emit(wk1, w_rt);
        }
        if (kn.lanes == 2) {
            emit(hlp, hlp_rt);
        }
    }
    uint32_t ports_present = 0;
    for (uint32_t d = 0; d < 4; ++d) {
        for (uint32_t l = 0; l < L; ++l) {
            ports_present |= (mine.dir[d] ? 1u : 0u) << (d * L + l);
        }
    }
    for (uint32_t i = 0; i < P; ++i) {
        std::vector<uint32_t> a = {i, me, kSemGo, ports_present};
        for (uint32_t q = 0; q < 4 * kvd::kMaxLinks; ++q) {
            const uint32_t d = q / L, l = q % L;
            const auto c = (q < 4 * L && mine.dir[d]) ? virt(mine.tx[d][l]) : CoreCoord{0, 0};
            a.push_back(c.x);
            a.push_back(c.y);
        }
        for (uint32_t q = 0; q < 4 * kvd::kMaxLinks; ++q) {  // this planner's lane reader (lane = index parity)
            const uint32_t d = q / L, l = q % L;
            const bool lane1 = kn.lanes == 2 && (i & 1) != 0;
            const auto c = !(q < 4 * L && mine.dir[d]) ? CoreCoord{0, 0}
                           : lane1                     ? virt(mine.hlp[d][l])
                                                       : virt(mine.tx[d][l]);
            a.push_back(c.x);
            a.push_back(c.y);
        }
        a.push_back(i == 0 ? 1u : 0u);
        a.push_back(bar.x);
        a.push_back(bar.y);
        for (uint32_t j = 0; j < P; ++j) {
            const auto c = virt(mine.planners[j]);
            a.push_back(c.x);
            a.push_back(c.y);
        }
        if (i == 0) {
            for (const auto& core : all_cores) {
                const auto c = virt(core);
                a.push_back(c.x);
                a.push_back(c.y);
            }
        }
        tt::tt_metal::KernelDescriptor::RTArgList rt;
        rt.append(a);
        planner.emplace_runtime_args(mine.planners[i], rt);
    }
    // ---- Fused plan (kvd4: plan_indices given). Every core of the grid runs the dispatch-only plan on its five RISCs
    // first (sparse_push_plan.cpp built with PLAN_FUSED as plan_prologue(), then the dispatch kernel's own work):
    // the dispatch kernels call it first; the spare cores and the planners' NCRISCs run a plan-only kernel; every
    // core's TRISCs run sparse_push_plan_trisc.cpp. Its buffers overlay the stage + pool CBs (grid-wide, contiguous):
    // no traffic writes a chip's stage or pool before every chip has its slices, which exist only after its plan.
    // The slot builders count in on the TX cores and planner 0 (kPlanSemReady) once the table holds their slots.
    std::vector<tt::tt_metal::KernelDescriptor> plan_kernels;
    const bool fused = t.plan_indices.has_value();
    if (fused) {
        TT_FATAL(kn.lanes == 1 && kn.wsplit == 2, "sparse_kv_dispatch: the fused plan needs LANES=1, WSPLIT=2");
        TT_FATAL(
            p.core_grid.ranges().size() == 1,
            "sparse_kv_dispatch fused plan: the core grid must be one rectangle (the plan's exchanges multicast to "
            "it)");
        using namespace ::sparse_push_plan;
        auto* idx_buf = t.plan_indices->buffer();
        auto* table_buf = t.table.buffer();
        const auto layout = PlanTableLayout::make(p.max_rows, p.stripe_chunk);
        const uint32_t rows = t.plan_indices->padded_shape()[2], topk = t.plan_indices->padded_shape()[3];
        const auto shape = packet_shape(mesh_device, t.output_tensor.buffer()->aligned_page_size());
        const uint32_t pc = static_cast<uint32_t>(cores.size());
        TT_FATAL(pc >= N, "sparse_kv_dispatch fused plan: {} cores, a slot builder per chip ({}) needed", pc, N);
        const uint32_t total_idx = rows * topk;
        const uint32_t per_core = tt::round_up(tt::div_up(total_idx, pc), kPlanIdxAlign);
        const uint32_t chunks = layout.bitmap_words / kChunkWords;
        const uint32_t max_chunks = tt::div_up(chunks, pc);
        TT_FATAL(max_chunks <= kMaxChunksPerCore, "sparse_kv_dispatch fused plan: {} reduce chunks a core", max_chunks);
        constexpr uint32_t kW = 5;
        const uint32_t bucket_cap = pad_words(tt::div_up(per_core, kW) + 1);
        uint32_t range_blocks = 4 * N;
        while (range_blocks % 16 != 0) {
            range_blocks += N;
        }
        const uint32_t max_ranges = tt::div_up(layout.num_blocks, range_blocks);
        const uint32_t slot_bytes = inbox_slot_bytes(p.max_rows, p.stripe_chunk, N);
        const uint32_t array_words = dirty_array_words(pc);
        // the plan's buffers (its CB ids) at 64 B aligned offsets in the overlay
        std::array<uint32_t, 10> sizes{};
        sizes[kTableImage] = layout.image_words * 4;
        sizes[kIdx] = per_core * 4;
        sizes[kDirtyChunks] = chunks;
        sizes[kDirtyMaps] = kNumDirtyMaps * array_words * 4;
        sizes[kScratch] = std::max(max_chunks * pc * kChunkWords * 4, slot_bytes);
        sizes[kPartials] = max_ranges * owner_stride(N) * 4;
        sizes[kMisc] = misc_words(max_chunks, kW) * 4;
        sizes[kBuckets] = std::max(kW * kW * bucket_cap, tt::div_up(max_chunks * pc * 2, 4u)) * 4;
        sizes[kWordRanks] = 64;  // (unused by the dispatch-only plan)
        sizes[kIdx2] = 0;
        std::string offsets;
        uint32_t off = 0;
        for (uint32_t i = 0; i < sizes.size(); ++i) {
            offsets += (i ? "," : "") + std::to_string(off);
            off += tt::round_up(sizes[i], 64u);
        }
        const uint32_t stage_pages = std::max(4 * G * kn.stage, 4 * kvd::kMaxLinks * kn.pstage);
        const uint64_t overlay = uint64_t{sbytes} * stage_pages + uint64_t{sbytes} * pool_pages;  // (minus slack)
        TT_FATAL(
            off <= overlay,
            "sparse_kv_dispatch fused plan: {} B of plan buffers exceed the {} B stage + pool overlay",
            off,
            overlay);
        constexpr uint32_t kPlanSemBase = 8, kPlanReleaseCb = 16;
        for (uint32_t i = 0; i < 3; ++i) {  // arrive, go, table ready
            desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
                .id = kPlanSemBase + i,
                .core_type = tt::CoreType::WORKER,
                .core_ranges = grid_range,
                .initial_value = 0});
        }
        add_cb(desc, grid_range, kPlanReleaseCb, 64, 1);
        const auto split_src = t.plan_chunk_start_tensor.has_value() ? kSplitTensor
                               : p.plan_chunk_start.has_value()      ? kSplitValue
                                                                     : kNoSplit;
        auto plan_ct = [&](uint32_t role) {
            std::vector<uint32_t> ct = {
                N,
                p.stripe_chunk,
                shape.pages_per_packet,
                shape.banks,
                pc,
                chunks,
                static_cast<uint32_t>(split_src),
                layout.bitmap_words,
                bucket_cap,
                range_blocks,
                topk,
                role,
                0u,
                0u,
                0u,
                0u};
            tt::tt_metal::TensorAccessorArgs(idx_buf).append_to(ct);
            tt::tt_metal::TensorAccessorArgs(table_buf).append_to(ct);
            tt::tt_metal::TensorAccessorArgs(table_buf).append_to(ct);  // (compact: unused)
            tt::tt_metal::TensorAccessorArgs(table_buf).append_to(ct);  // (compact2: unused)
            if (split_src == kSplitTensor) {
                tt::tt_metal::TensorAccessorArgs(t.plan_chunk_start_tensor->buffer()).append_to(ct);
            }
            return ct;
        };
        tt::tt_metal::KernelDescriptor::RTArgList plan_common;
        plan_common.push_back(idx_buf);
        plan_common.push_back(table_buf);
        if (split_src == kSplitTensor) {
            plan_common.push_back(t.plan_chunk_start_tensor->buffer());
        } else {
            plan_common.push_back(p.plan_chunk_start.value_or(0));
        }
        for (const auto& c : cores) {
            const auto v = virt(c);
            plan_common.push_back(static_cast<uint32_t>(v.x | v.y << 16));
        }
        const auto r0 = virt(cores.front()), r1 = virt(cores.back());
        plan_common.push_back(static_cast<uint32_t>(r0.x | r0.y << 16));
        plan_common.push_back(static_cast<uint32_t>(r1.x | r1.y << 16));
        plan_common.push_back(static_cast<uint32_t>(tx_cores.size() + 1));  // table-ready targets
        for (const auto& c : tx_cores) {
            const auto v = virt(c);
            plan_common.push_back(static_cast<uint32_t>(v.x | v.y << 16));
        }
        plan_common.push_back(static_cast<uint32_t>(virt(mine.planners[0]).x | virt(mine.planners[0]).y << 16));
        tt::tt_metal::KernelDescriptor::Defines plan_defs = {
            {"PLAN_FUSED", "1"},
            {"PLAN_ABL_DISPATCH_ONLY", "1"},
            {"PLAN_WORKERS", "5"},
            {"PLAN_OWNER_MAJOR", "1"},
            {"PLAN_SLOT_BYTES", std::to_string(slot_bytes)},
            {"PLAN_SEM_BASE", std::to_string(kPlanSemBase)},
            {"PLAN_OVERLAY_CB", std::to_string(kStage)},
            {"PLAN_RELEASE_CB", std::to_string(kPlanReleaseCb)},
            {"PLAN_BUF_OFFSETS", offsets},
            {"PLAN_TOTAL_IDX", std::to_string(total_idx)},
            {"PLAN_PER_CORE", std::to_string(per_core)},
        };
        if (std::getenv("PLAN_STATS") != nullptr) {
            plan_defs.emplace_back("PLAN_STATS", "1");
        }
        for (const auto& d : abl_defs) {  // (the plan-only and TRISC kernels see the KVD_ABL knobs too)
            if (d.first == "KVD_ABL_FUSED_NOPLAN") {
                plan_defs.push_back(d);
            }
        }
        // a dispatch kernel: the plan's compile-time and common args after its own, the plan's defines
        auto fuse = [&](tt::tt_metal::KernelDescriptor& k, uint32_t role) {
            const uint32_t ct_base = k.compile_time_args.size();
            const auto ct = plan_ct(role);
            k.compile_time_args.insert(k.compile_time_args.end(), ct.begin(), ct.end());
            const uint32_t common_base = k.common_runtime_args.size();
            k.emplace_common_runtime_args(plan_common);
            k.defines.insert(k.defines.end(), plan_defs.begin(), plan_defs.end());
            k.defines.emplace_back("KVD_FUSED_PLAN", "1");
            k.defines.emplace_back("PLAN_CT_BASE", std::to_string(ct_base));
            k.defines.emplace_back("PLAN_COMMON_BASE", std::to_string(common_base));
            k.defines.emplace_back("KVD_PLAN_READY", std::to_string(kPlanSemBase + 2));
        };
        fuse(tx, 0);
        fuse(wk, 0);
        fuse(planner, 0);
        fuse(txr, 1);
        common_args(wk1, p, t);  // (pushed below without common_args again)
        fuse(wk1, 1);
        // the plan-only kernels: spare cores (BRISC + NCRISC), planner cores (NCRISC)
        std::vector<CoreCoord> spare;
        {
            std::set<std::pair<size_t, size_t>> used;
            for (const auto& c : all_cores) {
                used.insert({c.x, c.y});
            }
            for (const auto& c : cores) {
                if (!used.contains({c.x, c.y})) {
                    spare.push_back(c);
                }
            }
        }
        std::vector<CoreCoord> ncrisc_only = spare;
        ncrisc_only.insert(ncrisc_only.end(), mine.planners.begin(), mine.planners.end());
        auto plan_only = [&](const std::vector<CoreCoord>& on, uint32_t role) {
            auto k = make_kernel(
                "kvd4_plan_only.cpp",
                to_range_set(on),
                {},
                {},
                role == 0 ? tt::tt_metal::DataMovementProcessor::RISCV_0 : tt::tt_metal::DataMovementProcessor::RISCV_1,
                role == 0 ? tt::tt_metal::NOC::RISCV_0_default : tt::tt_metal::NOC::RISCV_1_default);
            fuse(k, role);
            k.opt_level = tt::tt_metal::KernelBuildOptLevel::O3;
            return k;
        };
        tt::tt_metal::KernelDescriptor trisc;
        trisc.kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/plan/device/kernels/sparse_push_plan_trisc.cpp";
        trisc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        trisc.core_ranges = grid_range;
        trisc.compile_time_args = {layout.bitmap_words, pc, chunks, bucket_cap, N, p.stripe_chunk / 32};
        trisc.defines = plan_defs;
        trisc.opt_level = tt::tt_metal::KernelBuildOptLevel::O3;
        trisc.config = tt::tt_metal::ComputeConfigDescriptor{};
        // handles (override_runtime_arguments): 5 the TRISCs, 6 the NCRISC plan-only kernel, 7 the spare BRISCs
        plan_kernels.push_back(std::move(trisc));
        TT_FATAL(
            !spare.empty(), "sparse_kv_dispatch fused plan: needs grid cores beyond the dispatch's (spare BRISCs)");
        plan_kernels.push_back(plan_only(ncrisc_only, 1));
        plan_kernels.push_back(plan_only(spare, 0));
    }
    if (std::getenv("KVD3_OS") == nullptr) {  // O3 unless asked for the default (size) build
        for (auto* k : {&tx, &wk, &planner, &txr, &hlp}) {
            k->opt_level = tt::tt_metal::KernelBuildOptLevel::O3;
        }
    }
    desc.kernels.push_back(std::move(tx));
    desc.kernels.push_back(std::move(wk));
    desc.kernels.push_back(std::move(planner));
    desc.kernels.push_back(std::move(txr));  // (handle 3: no common args to patch)
    if (kn.lanes == 2) {
        desc.kernels.push_back(std::move(hlp));  // (handle 4: common args 9, 10 are the persistent barrier words)
    }
    if (kn.wsplit == 2) {
        if (!fused) {
            common_args(wk1, p, t);
        }
        if (std::getenv("KVD3_OS") == nullptr) {
            wk1.opt_level = tt::tt_metal::KernelBuildOptLevel::O3;
        }
        desc.kernels.push_back(std::move(wk1));  // (handle 4, or 5 after the helpers: common args patched)
    }
    for (auto& k : plan_kernels) {  // (fused: after the dispatch kernels, so their handles stay)
        desc.kernels.push_back(std::move(k));
    }
    return desc;
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

using namespace CMAKE_UNIQUE_NAMESPACE;

tt::tt_metal::ProgramDescriptor SparseKvDispatchProgramFactory::create_descriptor(
    const SparseKvDispatchParams& p,
    const SparseKvDispatchInputs& t,
    Tensor& /*output_tensor*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    using tt::tt_metal::distributed::MeshCoordinate;
    TT_FATAL(mesh_dispatch_coordinate.has_value(), "sparse_kv_dispatch programs are per device");
    const auto& coord = *mesh_dispatch_coordinate;
    if (const char* design = std::getenv("KVD_DESIGN"); design != nullptr && std::string(design) == "3") {
        return create_descriptor_v3(p, t, coord);
    }
    TT_FATAL(!t.plan_indices.has_value(), "sparse_kv_dispatch: plan_indices (the fused plan) needs KVD_DESIGN=3");
    tt::tt_metal::ProgramDescriptor desc;
    auto* mesh_device = t.input_tensor.device();
    const auto& view = mesh_device->get_view();
    const uint32_t N = view.shape().mesh_size();
    const auto mesh = kv_dispatch_mesh(mesh_device, p.num_links);
    const uint32_t R = mesh.rows, C = mesh.cols;
    const uint32_t me = ttnn::operations::ccl::common::get_linearized_index(coord, view);
    const uint32_t my_r = coord[0], my_c = coord[1];
    const uint32_t gmax = mesh.gmax();
    // Lanes: independent instances of the protocol; lane l rides fabric link l % links. More than one lane a link
    // shares the link through a fabric mux core.
    const uint32_t K = p.lanes_per_link;
    const uint32_t lanes = p.num_links * K;
    const bool use_mux = K > 1;
    const uint32_t nports = kvd::kDirections * lanes;

    // Ports: model direction d (0 X+, 1 X-, 2 Y+, 3 Y-) -> the fabric direction whose neighbour is there.
    const auto dirs = fabric_directions(mesh_device, coord, p.num_links);
    struct Port {
        bool present = false;
        uint32_t eth_dir = 0, link_idx = 0;
        MeshCoordinate neighbor{0, 0};
    };
    std::array<Port, kvd::kMaxPorts> ports{};
    auto neighbour_of = [&](const MeshCoordinate& at, uint32_t d) -> std::optional<MeshCoordinate> {
        int32_t r = static_cast<int32_t>(at[0]), c = static_cast<int32_t>(at[1]);
        const bool wrap = d < 2 ? mesh.wrap_x : mesh.wrap_y;
        const int32_t n = static_cast<int32_t>(d < 2 ? C : R);
        int32_t& v = d < 2 ? c : r;
        v += (d % 2 == 0) ? 1 : -1;
        if (!wrap && (v < 0 || v >= n)) {
            return std::nullopt;
        }
        v = (v % n + n) % n;
        return MeshCoordinate(static_cast<uint32_t>(r), static_cast<uint32_t>(c));
    };
    for (uint32_t d = 0; d < kvd::kDirections; ++d) {
        const auto tc = neighbour_of(coord, d);
        if (!tc) {
            continue;
        }
        for (const auto& fd : dirs) {
            if (fd.neighbor == *tc) {
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                    ports[kvd::port_index(lane, d)] = Port{
                        .present = true,
                        .eth_dir = fd.eth_direction,
                        .link_idx = fd.links[lane % p.num_links],
                        .neighbor = *tc};
                }
            }
        }
    }
    uint32_t present_mask = 0;
    for (uint32_t i = 0; i < kvd::kMaxPorts; ++i) {
        present_mask |= (ports[i].present ? 1u : 0u) << i;
    }
    auto has = [&](uint32_t l, uint32_t d) { return ports[kvd::port_index(l, d)].present; };

    // Cores, per chip (as sparse_dispatch_2d and high_bw_all_gather place their link workers): every port core, and
    // every mux core, is the free grid core fewest NoC1 hops from its link's Ethernet core (greedy per direction, then
    // link: the mux first, then its lanes' ports). Planner 0 (the slice-exchange sync core, which remote chips address)
    // is the grid's last core on every chip and never a port; the other planners are the grid's remaining cores in
    // order. Port placement only changes performance; neighbours' cores come from the same function.
    // Placing a neighbour chip's cores needs its device (internal::get_device): single-host meshes only (a neighbour
    // owned by another rank would need its placement from that rank).
    const auto cores = corerange_to_cores(p.core_grid, std::nullopt, true);
    const uint32_t mux_count = use_mux ? kvd::kDirections * p.num_links : 0;
    TT_FATAL(
        cores.size() >= nports + mux_count + p.planners,
        "sparse_kv_dispatch: {} cores in the grid, {} needed",
        cores.size(),
        nports + mux_count + p.planners);
    auto ports_at = [&](const MeshCoordinate& at) {
        std::array<bool, kvd::kMaxPorts> pr{};
        const auto at_dirs = fabric_directions(mesh_device, at, p.num_links);
        for (uint32_t d = 0; d < kvd::kDirections; ++d) {
            const auto tc = neighbour_of(at, d);
            for (const auto& x : at_dirs) {
                if (tc && x.neighbor == *tc) {
                    for (uint32_t lane = 0; lane < lanes; ++lane) {
                        pr[kvd::port_index(lane, d)] = true;
                    }
                }
            }
        }
        return pr;
    };
    struct ChipCores {
        std::vector<CoreCoord> ports, muxes, planners;  // ports[port_index], muxes[link * 4 + d]
    };
    auto place = [&](const MeshCoordinate& at) {
        ChipCores cc;
        cc.ports.assign(nports, cores.front());
        cc.muxes.assign(kvd::kDirections * p.num_links, cores.front());
        const auto at_dirs = fabric_directions(mesh_device, at, p.num_links);
        auto* dev = tt::tt_metal::internal::get_device(*mesh_device, at);
        std::set<std::pair<size_t, size_t>> taken{{cores.back().x, cores.back().y}};
        const auto at_node = mesh_device->get_fabric_node_id(at);
        auto nearest = [&](const tt::tt_metal::experimental::Device::CoreAtNocHops& closest) {
            std::optional<CoreCoord> best;
            uint32_t best_hops = std::numeric_limits<uint32_t>::max();
            for (const auto& cand : cores) {
                if (taken.contains({cand.x, cand.y})) {
                    continue;
                }
                const uint32_t h =
                    (cand == closest.logical_coord ? 0u
                                                   : tt::tt_metal::experimental::Device::get_worker_noc_hop_distance(
                                                         dev, cand, closest.logical_coord, tt::tt_metal::NOC::NOC_1)) +
                    closest.distance_in_noc_hops;
                if (h < best_hops) {
                    best_hops = h;
                    best = cand;
                }
            }
            TT_FATAL(best.has_value(), "sparse_kv_dispatch: too few cores for the ports");
            taken.insert({best->x, best->y});
            return *best;
        };
        for (uint32_t d = 0; d < kvd::kDirections; ++d) {
            const auto tc = neighbour_of(at, d);
            const FabricDirection* fd = nullptr;
            for (const auto& x : at_dirs) {
                if (tc && x.neighbor == *tc) {
                    fd = &x;
                }
            }
            if (fd == nullptr) {
                continue;
            }
            const auto nb_node = mesh_device->get_fabric_node_id(*tc);
            for (uint32_t link = 0; link < p.num_links; ++link) {
                const auto eth =
                    tt::tt_fabric::get_forwarding_eth_core(*mesh_device, at_node, nb_node, fd->links[link]);
                const auto closest = tt::tt_metal::experimental::Device::get_closest_worker_to_eth_core(
                    *dev, eth, tt::tt_metal::NOC::NOC_1);
                if (use_mux) {
                    cc.muxes[link * kvd::kDirections + d] = nearest(closest);
                }
                for (uint32_t lane = link; lane < lanes; lane += p.num_links) {
                    cc.ports[kvd::port_index(lane, d)] = nearest(closest);
                }
            }
        }
        // ports of absent directions, then planners: the free cores in order
        for (uint32_t i = 0; i < nports; ++i) {
            if (!ports_at(at)[i]) {
                for (const auto& c : cores) {
                    if (!taken.contains({c.x, c.y})) {
                        cc.ports[i] = c;
                        taken.insert({c.x, c.y});
                        break;
                    }
                }
            }
        }
        cc.planners.push_back(cores.back());
        for (const auto& c : cores) {
            if (cc.planners.size() == p.planners) {
                break;
            }
            if (!taken.contains({c.x, c.y})) {
                cc.planners.push_back(c);
                taken.insert({c.x, c.y});
            }
        }
        return cc;
    };
    const ChipCores mine = place(coord);
    const std::vector<CoreCoord>& port_cores = mine.ports;
    const std::vector<CoreCoord>& planner_cores = mine.planners;
    const auto port_range = to_range_set(port_cores);
    const auto planner_range = to_range_set(planner_cores);
    auto virt = [&](const CoreCoord& c) { return mesh_device->worker_core_from_logical_core(c); };

    // Geometry.
    const uint32_t row_bytes = t.input_tensor.buffer()->aligned_page_size();
    const auto max_payload = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());
    // The most rows whose packet (header, entries, then 64 B aligned rows) fits the payload.
    uint32_t B = 0;
    while (kvd::slot_rows_offset(B + 1) + (B + 1) * row_bytes <= max_payload) {
        ++B;
    }
    if (p.rows_per_packet) {
        B = std::min(B, p.rows_per_packet);
    }
    TT_FATAL(B >= 1, "sparse_kv_dispatch: a {} B row does not fit the {} B fabric payload", row_bytes, max_payload);
    TT_FATAL(kvd::slot_rows_offset(B) + B * row_bytes <= max_payload, "sparse_kv_dispatch: packet exceeds the payload");
    const uint32_t sbytes = kvd::slot_bytes(B, row_bytes);
    const uint32_t owner_slot_rows = t.output_tensor.padded_shape()[2] / N;
    const uint32_t inbox_slot = inbox_slot_bytes(p.max_rows, p.stripe_chunk, N);
    const auto scratch = KvDispatchScratch::make(p.max_rows, p.stripe_chunk, N, p.planners, gmax, lanes);
    const uint32_t num_slots = t.input_tensor.padded_shape()[0];
    const uint32_t pages_per_slot = t.input_tensor.buffer()->num_pages() / num_slots;
    const bool has_batch = t.batch_index_tensor.has_value();
    const uint32_t owner_blocks = tt::div_up(p.max_rows / p.stripe_chunk, N);

    // CBs.
    add_cb(desc, port_range, kCbLand, sbytes, p.landing_slots);
    add_cb(desc, port_range, kCbSend, std::max(sbytes * kvd::kRoundPackets, tt::round_up(max_payload, 64)), 1);
    add_cb(desc, port_range, kCbQin, kvd::kScratchPageBytes, 4);
    add_cb(desc, port_range, kCbMisc, 64 + 64 * p.planners, 1);
    add_cb(desc, port_range, kCbUstage, kvd::kScratchPageBytes, 3);
    add_cb(
        desc,
        planner_range,
        kCbInbox,
        std::max(inbox_slot, N * 64 + tt::div_up(owner_blocks * (p.stripe_chunk / 32), p.planners) * N * 64),
        1);
    add_cb(desc, planner_range, kCbStage, kvd::kScratchPageBytes, nports);
    add_cb(desc, planner_range, kCbRow, std::max(row_bytes, 64u), 8);  // kCopyBatch rows

    // Program semaphores.
    for (uint32_t id :
         {kSemOwnReady,
          kSemUnpackDone,
          kSemCreditOut,
          kSemQTail0,
          kSemQTail0 + 1,
          kSemQTail0 + 2,
          kSemMuxTermination}) {
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = id, .core_type = tt::CoreType::WORKER, .core_ranges = port_range, .initial_value = 0});
    }
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = kSemGo, .core_type = tt::CoreType::WORKER, .core_ranges = planner_range, .initial_value = 0});

    // Defines.
    tt::tt_metal::KernelDescriptor::Defines ring_defines = {
        {"KVD_XRING", join(ring_span_table(C, mesh.wrap_x))},
        {"KVD_YRING", join(ring_span_table(R, mesh.wrap_y))},
    };
    if (std::getenv("KVD_STATS") != nullptr) {
        ring_defines.emplace_back("KVD_STATS", "1");  // per-core counters, profiler zones, one DPRINT a core
    }
    // Ablation knobs (perf experiments only, results are not exact): KVD_ABL=NO_ROW_READ,HDR_ONLY,... defines
    // KVD_ABL_NO_ROW_READ etc. in every kernel.
    add_abl_defines(ring_defines);
    auto sender_defines = ring_defines;
    {
        std::vector<uint32_t> chip_ids, mesh_ids;
        for (const auto& c : tt::tt_metal::distributed::MeshCoordinateRange(view.shape())) {
            const auto node = mesh_device->get_fabric_node_id(c);
            chip_ids.push_back(static_cast<uint32_t>(node.chip_id));
            mesh_ids.push_back(static_cast<uint32_t>(*node.mesh_id));
        }
        sender_defines.emplace_back("DEST_CHIP_ID", ttnn::operations::ccl::common::stringify(chip_ids));
        sender_defines.emplace_back("DEST_MESH_ID", ttnn::operations::ccl::common::stringify(mesh_ids));
        for (const auto& d :
             tt::tt_metal::internal::get_fabric_kernel_defines(*mesh_device, tt::tt_fabric::FabricApiType::Mesh)) {
            sender_defines.push_back(d);
        }
    }

    // Accessor args (a missing batch tensor gets a placeholder; has_batch keeps it unused).
    auto acc = [](std::vector<uint32_t>& ct, const Tensor& x) {
        tt::tt_metal::TensorAccessorArgs(x.buffer()).append_to(ct);
    };
    const Tensor& batch_or_dummy = has_batch ? *t.batch_index_tensor : t.scratch;

    std::vector<uint32_t> sender_ct = {
        N,
        lanes,
        p.stripe_chunk,
        owner_slot_rows,
        inbox_slot,
        row_bytes,
        B,
        p.landing_slots,
        gmax,
        p.planners,
        scratch.pages_own,
        pages_per_slot,
        has_batch ? 1u : 0u,
        max_payload,
        kCbLand,
        kCbSend,
        kCbQin,
        kCbMisc,
        num_slots,
        owner_blocks};  // the sender's compile-time args 18, 19; its accessors start at 20
    // With a mux, its connection's compile-time args follow (5 words, ttnn::ccl::fabric_mux_connection_ct_args); the
    // sender's accessors then start at 25 instead of 20 (KVD_MUX).
    std::optional<tt::tt_fabric::FabricMuxConfig> mux;
    CoreRangeSet mux_range;
    if (use_mux) {
        std::vector<CoreCoord> present_mux;
        for (uint32_t l = 0; l < p.num_links; ++l) {
            for (uint32_t d = 0; d < kvd::kDirections; ++d) {
                if (ports[kvd::port_index(l, d)].present) {
                    present_mux.push_back(mine.muxes[l * kvd::kDirections + d]);
                }
            }
        }
        mux_range = to_range_set(present_mux);
        mux.emplace(
            *mesh_device,
            /*num_full_size_channels=*/K,
            /*num_header_only_channels=*/0,
            /*num_buffers_full_size_channel=*/kMuxBuffersPerChannel,
            /*num_buffers_header_only_channel=*/0,
            /*buffer_size_bytes_full_size_channel=*/
            tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(*mesh_device),
            /*base_l1_address=*/0);
        mux->place_in_program_local_l1(desc, mux_range);
        ttnn::ccl::fabric_mux_connection_ct_args(
            K, tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL, *mux, sender_ct);
        sender_defines.emplace_back("KVD_MUX", "1");
    }
    acc(sender_ct, t.input_tensor);
    acc(sender_ct, t.output_tensor);
    acc(sender_ct, t.table);
    acc(sender_ct, t.inbox);
    acc(sender_ct, t.scratch);
    acc(sender_ct, batch_or_dummy);
    std::vector<uint32_t> unpacker_ct = {
        N,
        R,
        C,
        mesh.wrap_x ? 1u : 0u,
        mesh.wrap_y ? 1u : 0u,
        p.stripe_chunk,
        owner_slot_rows,
        row_bytes,
        B,
        p.landing_slots,
        gmax,
        kCbLand,
        kCbUstage};
    acc(unpacker_ct, t.output_tensor);
    acc(unpacker_ct, t.scratch);
    std::vector<uint32_t> planner_ct = {
        N,
        R,
        C,
        mesh.wrap_x ? 1u : 0u,
        mesh.wrap_y ? 1u : 0u,
        lanes,
        p.stripe_chunk,
        owner_slot_rows,
        inbox_slot,
        row_bytes,
        p.planners,
        scratch.pages_own,
        pages_per_slot,
        has_batch ? 1u : 0u,
        num_slots,
        owner_blocks,
        kCbInbox,
        kCbStage,
        kCbRow};
    acc(planner_ct, t.input_tensor);
    acc(planner_ct, t.output_tensor);
    acc(planner_ct, t.inbox);
    acc(planner_ct, t.scratch);
    acc(planner_ct, t.table);
    acc(planner_ct, batch_or_dummy);

    auto sender = make_kernel(
        "kvd_port_sender.cpp",
        port_range,
        sender_ct,
        sender_defines,
        tt::tt_metal::DataMovementProcessor::RISCV_0,
        tt::tt_metal::NOC::RISCV_1_default);
    auto unpacker = make_kernel(
        "kvd_port_unpacker.cpp",
        port_range,
        unpacker_ct,
        ring_defines,
        tt::tt_metal::DataMovementProcessor::RISCV_1,
        tt::tt_metal::NOC::RISCV_0_default);
    auto planner = make_kernel(
        "kvd_planner.cpp",
        planner_range,
        planner_ct,
        ring_defines,
        tt::tt_metal::DataMovementProcessor::RISCV_0,
        tt::tt_metal::NOC::RISCV_0_default);
    common_args(sender, p, t);
    common_args(unpacker, p, t);
    common_args(planner, p, t);

    const auto sync = virt(planner_cores[0]);
    for (uint32_t l = 0; l < lanes; ++l) {  // l: the lane
        for (uint32_t d = 0; d < kvd::kDirections; ++d) {
            const uint32_t pi = kvd::port_index(l, d);
            const auto& core = port_cores[pi];
            const Port& port = ports[pi];
            // sender
            std::vector<uint32_t> s;
            if (!port.present) {
                s = {0u};
            } else {
                const auto nb_node = mesh_device->get_fabric_node_id(port.neighbor);
                const auto nb_core = virt(place(port.neighbor).ports[kvd::port_index(l, d ^ 1)]);
                s = {
                    1u,
                    d,
                    l,
                    me,
                    port.eth_dir,
                    static_cast<uint32_t>(nb_node.chip_id),
                    static_cast<uint32_t>(*nb_node.mesh_id),
                    static_cast<uint32_t>(nb_core.x),
                    static_cast<uint32_t>(nb_core.y),
                    static_cast<uint32_t>(sync.x),
                    static_cast<uint32_t>(sync.y),
                    kSemOwnReady,
                    kSemUnpackDone,
                    kSemCreditOut};
                const uint32_t other[2] = {d < 2 ? 2u : 0u, d < 2 ? 3u : 1u};
                const bool stream_on[3] = {has(l, d ^ 1), has(l, other[1]), has(l, other[0])};
                for (uint32_t k = 0; k < 3; ++k) {
                    s.push_back(stream_on[k] ? 1u : 0u);
                    s.push_back(scratch.queue_page(pi, k));
                    s.push_back(kSemQTail0 + k);
                }
                s.push_back(scratch.own_base(pi));
                s.push_back(scratch.count_base + pi);
                if (use_mux) {
                    // lanes of link `link`: lane = sub * links + link; sub 0 is the mux's termination master
                    const uint32_t link = l % p.num_links, sub = l / p.num_links;
                    const auto mux_core = mine.muxes[link * kvd::kDirections + d];
                    ttnn::ccl::fabric_mux_connection_rt_args(
                        /*mux_connection_valid=*/true,
                        /*is_termination_master=*/sub == 0,
                        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                        virt(mux_core),
                        /*worker_id=*/sub,
                        core,
                        *mux,
                        desc,
                        virt(port_cores[kvd::port_index(link, d)]),
                        s,
                        kSemMuxTermination);
                } else {
                    tt::tt_fabric::append_fabric_connection_rt_args(
                        *mesh_device, mesh_device->get_fabric_node_id(coord), nb_node, port.link_idx, desc, core, s);
                }
            }
            tt::tt_metal::KernelDescriptor::RTArgList srt;
            srt.append(s);
            sender.emplace_runtime_args(core, srt);
            // unpacker: packets from the neighbour in d (moving d ^ 1)
            std::vector<uint32_t> u = {port.present ? 1u : 0u, d, me, kSemUnpackDone, kSemCreditOut};
            const uint32_t move = d ^ 1;
            const uint32_t turn_plus = move < 2 ? 2u : 0u, turn_minus = move < 2 ? 3u : 1u;
            const uint32_t turn_stream = 1 + (move & 1);
            const struct {
                uint32_t dir, stream;
            } outs[3] = {{move, 0}, {turn_plus, turn_stream}, {turn_minus, turn_stream}};
            for (const auto& o : outs) {
                const bool on = port.present && has(l, o.dir);
                const auto consumer = virt(port_cores[kvd::port_index(l, o.dir)]);
                u.push_back(on ? 1u : 0u);
                u.push_back(scratch.queue_page(kvd::port_index(l, o.dir), o.stream));
                u.push_back(static_cast<uint32_t>(consumer.x));
                u.push_back(static_cast<uint32_t>(consumer.y));
                u.push_back(kSemQTail0 + o.stream);
            }
            tt::tt_metal::KernelDescriptor::RTArgList urt;
            urt.append(u);
            unpacker.emplace_runtime_args(core, urt);
        }
    }
    for (uint32_t i = 0; i < p.planners; ++i) {
        std::vector<uint32_t> a = {i, me, kSemGo, kSemOwnReady, present_mask};
        for (uint32_t pi = 0; pi < kvd::kMaxPorts; ++pi) {  // kMaxPorts pairs; ports past 4 x lanes are absent
            const auto c = pi < nports ? virt(port_cores[pi]) : CoreCoord{0, 0};
            a.push_back(static_cast<uint32_t>(c.x));
            a.push_back(static_cast<uint32_t>(c.y));
        }
        a.push_back(scratch.count_base);
        a.push_back(i == 0 ? 1u : 0u);
        for (uint32_t j = 0; j < p.planners; ++j) {
            const auto c = virt(planner_cores[j]);
            a.push_back(static_cast<uint32_t>(c.x));
            a.push_back(static_cast<uint32_t>(c.y));
        }
        tt::tt_metal::KernelDescriptor::RTArgList prt;
        prt.append(a);
        planner.emplace_runtime_args(planner_cores[i], prt);
    }
    (void)my_r;
    (void)my_c;

    // In the order of kSenderKernel, kUnpackerKernel, kPlannerKernel (then the mux, whose args are never patched).
    desc.kernels.push_back(std::move(sender));
    desc.kernels.push_back(std::move(unpacker));
    desc.kernels.push_back(std::move(planner));
    if (use_mux) {
        tt::tt_metal::KernelDescriptor mux_kernel;
        mux_kernel.kernel_source = "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp";
        mux_kernel.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        mux_kernel.core_ranges = mux_range;
        mux_kernel.compile_time_args = mux->get_fabric_mux_compile_time_args();
        mux_kernel.config = tt::tt_metal::DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default};
        mux_kernel.opt_level = tt::tt_metal::KernelBuildOptLevel::O3;
        for (uint32_t l = 0; l < p.num_links; ++l) {
            for (uint32_t d = 0; d < kvd::kDirections; ++d) {
                const Port& port = ports[kvd::port_index(l, d)];
                if (!port.present) {
                    continue;
                }
                const auto mux_core = mine.muxes[l * kvd::kDirections + d];
                tt::tt_metal::KernelDescriptor::RTArgList rt;
                rt.append(mux->get_fabric_mux_run_time_args(
                    *mesh_device,
                    mesh_device->get_fabric_node_id(coord),
                    mesh_device->get_fabric_node_id(port.neighbor),
                    port.link_idx,
                    desc,
                    mux_core));
                mux_kernel.emplace_runtime_args(mux_core, rt);
            }
        }
        desc.kernels.push_back(std::move(mux_kernel));
    }
    return desc;
}

void SparseKvDispatchProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const SparseKvDispatchParams& p,
    const SparseKvDispatchInputs& t,
    Tensor& /*output_tensor*/,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    for (auto k : {kSenderKernel, kUnpackerKernel, kPlannerKernel}) {
        auto& c = tt::tt_metal::GetCommonRuntimeArgs(program, k);
        patch_common(c, p, t);
    }
    // v3 with LANES=2: the helper readers (handle 4) read the barrier words from their common args too
    const char* design = std::getenv("KVD_DESIGN");
    if (design != nullptr && std::string(design) == "3") {
        const auto kn = v3_knobs();
        uint32_t h = 4;
        if (kn.lanes == 2) {
            patch_common(tt::tt_metal::GetCommonRuntimeArgs(program, h++), p, t);
        }
        if (kn.wsplit == 2) {  // the NCRISC workers
            patch_common(tt::tt_metal::GetCommonRuntimeArgs(program, h++), p, t);
        }
        // the fused plan's common args 0-2 (indices, table, chunk start: tensor address or value) after each kernel's
        // own (13 or 0): tx, wk, planner, txr, wk1, then the plan-only kernels at 6 and 7; 5 is the TRISC kernel
        if (t.plan_indices.has_value()) {
            auto patch = [&](tt::tt_metal::KernelHandle k, uint32_t base) {
                auto& c = tt::tt_metal::GetCommonRuntimeArgs(program, k);
                c[base] = t.plan_indices->buffer()->address();
                c[base + 1] = t.table.buffer()->address();
                c[base + 2] = t.plan_chunk_start_tensor.has_value() ? t.plan_chunk_start_tensor->buffer()->address()
                                                                    : p.plan_chunk_start.value_or(0);
            };
            const std::array<uint32_t, 5> base = {14, 14, 14, 0, 14};
            for (uint32_t k = 0; k < base.size(); ++k) {
                patch(k, base[k]);
            }
            patch(6, 0);
            patch(7, 0);
        }
    }
}

SparseKvDispatchMeshWorkloadFactory::cached_mesh_workload_t SparseKvDispatchMeshWorkloadFactory::create_mesh_workload(
    const SparseKvDispatchParams& p,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const SparseKvDispatchInputs& t,
    Tensor& output_tensor) {
    auto* mesh_device = t.input_tensor.device();
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& range : tensor_coords.ranges()) {
        std::vector<tt::tt_metal::ProgramDescriptor> descriptors;
        for (const auto& coord : range) {
            descriptors.push_back(SparseKvDispatchProgramFactory::create_descriptor(p, t, output_tensor, coord));
        }
        if (range.shape().mesh_size() < 2) {
            workload.add_program(range, tt::tt_metal::Program{*mesh_device, descriptors.front()});
            shared_variables[range] = {};
            continue;
        }
        TT_FATAL(
            tt::tt_metal::distributed::ArePerDeviceRuntimeArgsSupported(*mesh_device, range),
            "sparse_kv_dispatch runs one program per mesh, which needs per-device runtime args on {}",
            range);
        // A direction a chip lacks has shorter (idle) runtime args on its cores: zero padded, never read.
        ttnn::ProgramMergeOptions options;
        options.pad_runtime_args = true;
        auto merged = ttnn::merge_per_device_program_descriptors(descriptors, options);
        TT_FATAL(
            merged.has_value(),
            "sparse_kv_dispatch: the per-chip programs of {} differ beyond runtime-arg values (kernels, compile-time "
            "args, CBs or semaphores must be the same on every chip)",
            range);
        workload.add_program(range, tt::tt_metal::Program{*mesh_device, merged->descriptor});
        ttnn::apply_per_device_runtime_args(workload, range, merged->per_device_runtime_args);
        shared_variables[range] = {};
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

void SparseKvDispatchMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const SparseKvDispatchParams& p,
    const SparseKvDispatchInputs& t,
    Tensor& output_tensor) {
    // (common runtime args only: the same on every chip, so set on the merged program as a whole)
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        SparseKvDispatchProgramFactory::override_runtime_arguments(program, p, t, output_tensor, std::nullopt);
    }
}

}  // namespace ttnn::operations::experimental::sparse_kv_push
