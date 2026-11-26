// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_matmul_planner.hpp"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/indestructible.hpp>

#include "kernels/all_to_all_matmul_args.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::all_to_all_matmul {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshDevice;

namespace {

// CoreRangeSet of the given cells as merged rectangles.
CoreRangeSet planner_rects(const std::vector<CoreCoord>& cells) {
    std::vector<CoreRange> ranges;
    ranges.reserve(cells.size());
    for (const auto& cell : cells) {
        ranges.emplace_back(cell, cell);
    }
    return CoreRangeSet(ranges).merge_ranges();
}

std::vector<uint32_t> planner_mesh_dims(const MeshDevice* mesh_device) {
    const auto& shape = mesh_device->shape();
    std::vector<uint32_t> dims;
    dims.reserve(shape.dims());
    for (size_t i = 0; i < shape.dims(); ++i) {
        dims.push_back(shape[i]);
    }
    return dims;
}

// kFwd / kBwd: the ring port whose connection carries the first hop of the fabric's route src -> dst.
std::optional<uint32_t> planner_first_hop_dir(
    MeshDevice* mesh_device,
    const MeshCoordinate& src,
    const MeshCoordinate& dst,
    const MeshCoordinate& nxt,
    const MeshCoordinate& prv) {
    const auto src_node = mesh_device->get_fabric_node_id(src);
    const auto dir = tt::tt_fabric::get_eth_forwarding_direction(src_node, mesh_device->get_fabric_node_id(dst));
    if (!dir.has_value()) {
        return std::nullopt;
    }
    if (dir == tt::tt_fabric::get_eth_forwarding_direction(src_node, mesh_device->get_fabric_node_id(nxt))) {
        return kFwd;
    }
    if (dir == tt::tt_fabric::get_eth_forwarding_direction(src_node, mesh_device->get_fabric_node_id(prv))) {
        return kBwd;
    }
    return std::nullopt;
}

// Port placement: per (l outer, d inner) the worker with the fewest NoC1 hops to the link's Ethernet core, ties by
// logical (y, x), skipping cores already taken. Throws on any failure (the caller falls back).
std::vector<CoreCoord> planner_place_ports(
    MeshDevice* mesh_device, const MeshCoordinate& coord, const ChipRoute& ch, uint32_t num_links, CoreCoord grid) {
    tt::tt_metal::IDevice* device = mesh_device->get_device(coord);
    if (device == nullptr) {
        throw std::runtime_error("all_to_all_matmul: no local device");
    }
    const auto src = mesh_device->get_fabric_node_id(coord);
    std::set<CoreCoord> taken;
    std::vector<CoreCoord> ports(2 * num_links);
    for (uint32_t l = 0; l < num_links; ++l) {
        for (uint32_t d : {kFwd, kBwd}) {
            const MeshCoordinate& peer = d == kFwd ? ch.next : ch.prev;
            if (l >= ch.links[d].size()) {
                throw std::out_of_range("all_to_all_matmul: missing link");
            }
            const CoreCoord eth =
                tt::tt_fabric::get_forwarding_eth_core(src, mesh_device->get_fabric_node_id(peer), ch.links[d][l]);
            const auto closest = tt::tt_metal::experimental::Device::get_closest_worker_to_eth_core(
                *device, eth, tt::tt_metal::NOC::NOC_1);
            std::optional<std::tuple<uint32_t, size_t, size_t>> best;
            CoreCoord best_core;
            for (size_t y = 0; y < grid.y; ++y) {
                for (size_t x = 0; x < grid.x; ++x) {
                    const CoreCoord worker(x, y);
                    if (taken.contains(worker)) {
                        continue;
                    }
                    const uint32_t hops =
                        closest.distance_in_noc_hops +
                        tt::tt_metal::experimental::Device::get_worker_noc_hop_distance(
                            mesh_device, coord, worker, closest.logical_coord, tt::tt_metal::NOC::NOC_1);
                    const auto score = std::make_tuple(hops, y, x);
                    if (!best.has_value() || score < *best) {
                        best = score;
                        best_core = worker;
                    }
                }
            }
            if (!best.has_value()) {
                throw std::runtime_error("all_to_all_matmul: no free worker for a port");
            }
            ports[port_index(d, l)] = best_core;
            taken.insert(best_core);
        }
    }
    return ports;
}

// 2 * num_links cores of logical row 0, centred.
std::vector<CoreCoord> planner_fallback_ports(CoreCoord grid, uint32_t num_links) {
    const size_t half = grid.x / 2;
    const size_t x0 = half > num_links ? half - num_links : 0;
    std::vector<CoreCoord> ports(2 * num_links);
    for (uint32_t l = 0; l < num_links; ++l) {
        for (uint32_t d : {kFwd, kBwd}) {
            ports[port_index(d, l)] = CoreCoord(x0 + (2 * l) + d, 0);
        }
    }
    return ports;
}

// (out_subblock_h, out_subblock_w): largest product within the DEST limit, h | block_m, w | block_n, wide first.
std::pair<uint32_t, uint32_t> planner_subblock(uint32_t block_m, uint32_t block_n, bool fp32_dest_acc_en) {
    const uint32_t limit = fp32_dest_acc_en ? 4 : 8;
    uint32_t best_h = 1;
    uint32_t best_w = 1;
    for (uint32_t h = 1; h <= std::min(block_m, limit); ++h) {
        if (block_m % h != 0) {
            continue;
        }
        for (uint32_t w = 1; w <= std::min(block_n, limit); ++w) {
            if (block_n % w != 0 || h * w > limit) {
                continue;
            }
            if (std::make_pair(h * w, w) > std::make_pair(best_h * best_w, best_w)) {
                best_h = h;
                best_w = w;
            }
        }
    }
    return {best_h, best_w};
}

uint32_t planner_block_n(uint32_t Kt, uint32_t Nt, uint32_t w_tile_bytes) {
    const uint64_t col_bytes = static_cast<uint64_t>(Kt) * w_tile_bytes;
    TT_FATAL(
        col_bytes <= kWeightBlockBytes,
        "all_to_all_matmul: Kt * weight_tile_bytes = {} > WEIGHT_BLOCK_BYTES ({}): regime R2 (K-blocked weight) is not "
        "built",
        col_bytes,
        kWeightBlockBytes);
    TT_FATAL(Nt >= 1, "all_to_all_matmul: N must hold at least one tile");
    uint32_t best = 0;
    for (uint32_t d = 1; d <= Nt; ++d) {
        if (Nt % d == 0 && col_bytes * d <= kWeightBlockBytes) {
            best = d;
        }
    }
    return best;
}

struct SlotEntry {
    uint32_t h = 0;
    bool wait = false;
    uint32_t nb0 = 0, nb1 = 0;
    std::optional<CoreCoord> helper;
    uint32_t m0 = 0, m1 = 0;
    int32_t piece = -1;
};

using SendKey = std::tuple<MeshCoordinate, uint32_t, uint32_t>;  // (coord, d, l)
using Entry = std::pair<uint32_t, uint32_t>;                     // (j, t)
using ChipCore = std::pair<MeshCoordinate, CoreCoord>;
using ChipHead = std::pair<MeshCoordinate, uint32_t>;

Plan build_plan(const MeshTopology& topo, const PlanInputs& in) {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;
    Plan plan;
    const uint32_t G = topo.G;
    const uint32_t L = topo.num_links;
    const uint32_t B = in.B;
    const uint32_t Hl = in.Hl;
    const uint32_t H = in.H;
    TT_FATAL(G >= 2 && L >= 1 && Hl >= 1, "all_to_all_matmul: empty ring, links or heads");

    // ---- geometry ----
    const uint32_t Kt = in.K / TILE_WIDTH;
    uint32_t Nt = in.N / TILE_WIDTH;
    const uint32_t Mq = in.S / G;
    const uint32_t Mq_t = tt::div_up(Mq, TILE_HEIGHT);
    const uint32_t St = tt::div_up(in.S, TILE_HEIGHT);
    const uint32_t block_m = B * Mq_t;
    const uint32_t slot_tiles = block_m * Kt;
    const uint32_t in_tile = in.in_tile_bytes;
    const uint32_t w_tile = in.w_tile_bytes;
    plan.Kt = Kt;
    plan.Nt = Nt;
    plan.Mq = Mq;
    plan.Mq_t = Mq_t;
    plan.St = St;
    plan.retile = Mq % TILE_HEIGHT != 0;
    plan.stage_tiles = (Mq_t + 1) * Kt;
    plan.block_m = block_m;
    plan.slot_tiles = slot_tiles;
    plan.slot_bytes = slot_tiles * in_tile;
    uint32_t block_n = planner_block_n(Kt, Nt, w_tile);
    uint32_t num_n_blocks = Nt / block_n;
    auto [sbh, sbw] = planner_subblock(block_m, block_n, in.fp32_dest_acc_en);
    TT_FATAL(in_tile > 0, "all_to_all_matmul: input page size is zero");
    const uint32_t packet_tiles = in.payload_bytes / in_tile;
    TT_FATAL(
        packet_tiles >= 1, "all_to_all_matmul: fabric max payload {} B is smaller than one tile", in.payload_bytes);
    const uint32_t packet_bytes = packet_tiles * in_tile;

    // ---- transport ----
    const uint64_t in_bytes = static_cast<uint64_t>(B) * Hl * St * Kt * in_tile;
    uint64_t dram_bytes = in_bytes + (static_cast<uint64_t>(H) * Kt * Nt * w_tile);
    if (!in.output_in_l1) {
        dram_bytes += static_cast<uint64_t>(B) * H * Mq_t * Nt * in_tile;
    }
    TT_FATAL(in_bytes > 0, "all_to_all_matmul: empty input");
    const double local_ratio = static_cast<double>(dram_bytes) / static_cast<double>(in_bytes);
    const bool grouped = local_ratio <= kSendRunMaxLocalRatio;
    const bool pack = packet_tiles <= slot_tiles;
    const uint32_t batch_chunks = std::max(1u, kPortBatchBytes / packet_bytes);
    const uint32_t port_window =
        std::min(kMaxPortWindow, std::max(1u, tt::div_up(kPortReadInflightBytes, packet_bytes)));
    const uint32_t tail_max = grouped ? kTailSplitMaxHeadsGrouped : kTailSplitMaxHeads;
    const uint32_t ms_heads = grouped ? kTailMsplitHeads : 0;
    const uint32_t run = grouped ? kSendRunEntries : 1;

    // Chips in the source's group order.
    std::vector<MeshCoordinate> chip_order;
    for (const auto& [coord, ch] : topo.chips) {
        if (ch.p == 0) {
            chip_order.insert(chip_order.end(), ch.grp.begin(), ch.grp.end());
        }
    }

    // ---- sends: (coord, d, l) -> [(j, t)] ----
    std::map<SendKey, std::vector<Entry>> sends;
    std::vector<SendKey> send_order;
    for (const auto& coord : chip_order) {
        const ChipRoute& ch = topo.chips.at(coord);
        const uint32_t p = ch.p;
        for (uint32_t d : {kFwd, kBwd}) {
            std::vector<uint32_t> dests;
            for (uint32_t t = 0; t < G; ++t) {
                if (ch.route[t] >= 0 && static_cast<uint32_t>(ch.route[t]) == d) {
                    dests.push_back(t);
                }
            }
            auto dest_key = [&](uint32_t t) {
                const uint32_t hops = std::min((t + G - p) % G, (p + G - t) % G);
                return std::make_pair(hops, t);
            };
            std::sort(dests.begin(), dests.end(), [&](uint32_t a, uint32_t b) { return dest_key(a) < dest_key(b); });
            for (uint32_t l = 0; l < L; ++l) {
                std::vector<uint32_t> heads;
                for (uint32_t j = l; j < Hl; j += L) {
                    heads.push_back(j);
                }
                std::vector<Entry> ents;
                for (size_t r0 = 0; r0 < heads.size(); r0 += run) {
                    const size_t r1 = std::min(heads.size(), r0 + run);
                    for (uint32_t t : dests) {
                        for (size_t i = r0; i < r1; ++i) {
                            ents.emplace_back(heads[i], t);
                        }
                    }
                }
                const SendKey key{coord, d, l};
                sends[key] = std::move(ents);
                send_order.push_back(key);
            }
        }
    }

    // ---- cores ----
    const std::vector<CoreCoord>& compute_cells = topo.compute_cells;
    TT_FATAL(!compute_cells.empty(), "all_to_all_matmul: no compute cores");
    std::vector<std::pair<CoreCoord, std::vector<uint32_t>>> assign;
    {
        const auto C = static_cast<uint32_t>(compute_cells.size());
        const uint32_t q = H / C;
        const uint32_t rem = H % C;
        uint32_t h = 0;
        for (uint32_t i = 0; i < C; ++i) {
            const uint32_t n = q + (i < rem ? 1 : 0);
            if (n == 0) {
                break;
            }
            std::vector<uint32_t> hs(n);
            for (uint32_t k = 0; k < n; ++k) {
                hs[k] = h + k;
            }
            assign.emplace_back(compute_cells[i], std::move(hs));
            h += n;
        }
    }
    std::map<uint32_t, CoreCoord> core_of;
    std::map<CoreCoord, std::vector<uint32_t>> heads_of;
    for (const auto& [xy, hs] : assign) {
        for (uint32_t h : hs) {
            core_of[h] = xy;
        }
        heads_of[xy] = hs;
    }
    auto slot_order = [&](uint32_t q, const std::vector<uint32_t>& hs) {
        std::vector<uint32_t> local;
        std::vector<uint32_t> remote;
        for (uint32_t h : hs) {
            (h / Hl == q ? local : remote).push_back(h);
        }
        std::stable_sort(remote.begin(), remote.end(), [&](uint32_t a, uint32_t b) {
            return std::make_pair(a % Hl, a / Hl) < std::make_pair(b % Hl, b / Hl);
        });
        local.insert(local.end(), remote.begin(), remote.end());
        return local;
    };
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> slot_of;  // (q, h) -> slot
    for (uint32_t q = 0; q < G; ++q) {
        for (const auto& [xy, hs] : assign) {
            const auto order = slot_order(q, hs);
            for (uint32_t s = 0; s < order.size(); ++s) {
                slot_of[{q, order[s]}] = s;
            }
        }
    }

    auto send_index = [&](const MeshCoordinate& coord, uint32_t h, size_t& count) {
        const ChipRoute& ch = topo.chips.at(coord);
        const uint32_t q = ch.p;
        const uint32_t j = h % Hl;
        const MeshCoordinate& src = ch.grp[h / Hl];
        const int32_t dir = topo.chips.at(src).route[q];
        TT_FATAL(dir >= 0, "all_to_all_matmul: no route to chip position {}", q);
        const auto& port_sends = sends.at(SendKey{src, static_cast<uint32_t>(dir), j % L});
        const auto it = std::find(port_sends.begin(), port_sends.end(), Entry{j, q});
        TT_FATAL(it != port_sends.end(), "all_to_all_matmul: head {} is not sent to chip position {}", h, q);
        count = port_sends.size();
        return static_cast<size_t>(it - port_sends.begin());
    };
    auto landing_share = [&](const MeshCoordinate& coord, uint32_t h) {
        if (h / Hl == topo.chips.at(coord).p) {
            return 0.0;
        }
        size_t count = 0;
        const size_t idx = send_index(coord, h, count);
        return static_cast<double>(idx + 1) / static_cast<double>(count);
    };

    // ---- M-split row pieces ----
    const uint32_t bn_split = (num_n_blocks == 1 && block_n % 2 == 0) ? block_n / 2 : block_n;
    const uint32_t ms_sbh = planner_subblock(block_m, bn_split, in.fp32_dest_acc_en).first;
    std::vector<std::pair<uint32_t, uint32_t>> m_rows;
    if (ms_heads > 0 && kTailMsplitPieces >= 2 && pack) {
        const uint32_t units = block_m / ms_sbh;
        const uint32_t pieces = std::min(kTailMsplitPieces, units);
        uint32_t u0 = 0;
        for (uint32_t i = 0; i < pieces; ++i) {
            const uint32_t u1 = u0 + (units / pieces) + (i < units % pieces ? 1 : 0);
            m_rows.emplace_back(u0 * ms_sbh, u1 * ms_sbh);
            u0 = u1;
        }
        for (const auto& [r0, r1] : m_rows) {
            if ((r1 - r0) * Kt < packet_tiles) {
                m_rows.clear();
                break;
            }
        }
    }

    // ---- tail split plan per chip ----
    using SplitPlan = std::vector<std::pair<uint32_t, CoreCoord>>;                               // h -> helper
    using MsplitPlan = std::vector<std::pair<uint32_t, std::vector<std::optional<CoreCoord>>>>;  // h -> pieces
    std::map<MeshCoordinate, SplitPlan> split_of;
    std::map<MeshCoordinate, MsplitPlan> msplit_of;
    for (const auto& coord : chip_order) {
        const uint32_t q = topo.chips.at(coord).p;
        std::vector<std::tuple<double, uint32_t, uint32_t>> est;
        for (const auto& [xy, hs] : assign) {
            for (uint32_t h : hs) {
                if (h / Hl == q) {
                    continue;
                }
                est.emplace_back(landing_share(coord, h), static_cast<uint32_t>(hs.size()), h);
            }
        }
        std::sort(est.begin(), est.end(), std::greater<>());
        std::vector<CoreCoord> helpers;
        for (const auto& xy : compute_cells) {
            if (!heads_of.contains(xy)) {
                helpers.push_back(xy);
            }
        }
        std::vector<CoreCoord> local_only;
        for (const auto& [xy, hs] : assign) {
            const bool all_local = std::all_of(hs.begin(), hs.end(), [&](uint32_t h) { return h / Hl == q; });
            const bool fits = static_cast<uint64_t>(hs.size() + 1) * slot_tiles * in_tile <= kASlotBudgetBytes &&
                              hs.size() < kMaxHeadsPerCore;
            if (all_local && fits) {
                local_only.push_back(xy);
            }
        }
        std::stable_sort(local_only.begin(), local_only.end(), [&](const CoreCoord& a, const CoreCoord& b) {
            return heads_of.at(a).size() < heads_of.at(b).size();
        });
        helpers.insert(helpers.end(), local_only.begin(), local_only.end());
        MsplitPlan mplan;
        if (!m_rows.empty()) {
            const size_t n_ms = std::min<size_t>(ms_heads, est.size());
            for (size_t i = 0; i < n_ms; ++i) {
                const uint32_t h = std::get<2>(est[i]);
                const size_t need = m_rows.size() - 1;
                if (helpers.size() < need) {
                    break;
                }
                std::vector<std::optional<CoreCoord>> pieces{std::nullopt};  // piece 0 stays on the primary core
                pieces.insert(pieces.end(), helpers.begin(), helpers.begin() + static_cast<std::ptrdiff_t>(need));
                helpers.erase(helpers.begin(), helpers.begin() + static_cast<std::ptrdiff_t>(need));
                mplan.emplace_back(h, std::move(pieces));
            }
        }
        std::erase_if(est, [&](const auto& e) {
            return std::any_of(mplan.begin(), mplan.end(), [&](const auto& m) { return m.first == std::get<2>(e); });
        });
        const size_t n = std::min<size_t>(helpers.size(), tail_max);
        SplitPlan sp;
        for (size_t i = 0; i < n && i < est.size(); ++i) {
            sp.emplace_back(std::get<2>(est[i]), helpers[i]);
        }
        split_of[coord] = std::move(sp);
        msplit_of[coord] = std::move(mplan);
    }
    bool tail_split = std::any_of(split_of.begin(), split_of.end(), [](const auto& e) { return !e.second.empty(); }) ||
                      std::any_of(msplit_of.begin(), msplit_of.end(), [](const auto& e) { return !e.second.empty(); });
    if (tail_split && num_n_blocks == 1) {
        if (block_n % 2 != 0) {
            tail_split = false;  // nothing to halve
        } else {
            block_n /= 2;
            num_n_blocks = 2;
            std::tie(sbh, sbw) = planner_subblock(block_m, block_n, in.fp32_dest_acc_en);
        }
    }
    if (!tail_split) {
        for (auto& [coord, sp] : split_of) {
            sp.clear();
        }
        for (auto& [coord, mp] : msplit_of) {
            mp.clear();
        }
    }
    const bool any_msplit =
        std::any_of(msplit_of.begin(), msplit_of.end(), [](const auto& e) { return !e.second.empty(); });
    TT_FATAL(!any_msplit || sbh == ms_sbh, "all_to_all_matmul: M-split pieces do not follow the subblock height");
    const uint32_t nb_split = num_n_blocks / 2;

    // ---- slots per (coord, core) ----
    std::map<ChipCore, std::vector<SlotEntry>> slots;
    for (const auto& coord : chip_order) {
        const uint32_t q = topo.chips.at(coord).p;
        const SplitPlan& sp = split_of.at(coord);
        const MsplitPlan& mp = msplit_of.at(coord);
        for (const auto& xy : compute_cells) {
            std::vector<SlotEntry> own;
            const auto hit = heads_of.find(xy);
            const std::vector<uint32_t> hs =
                hit != heads_of.end() ? slot_order(q, hit->second) : std::vector<uint32_t>{};
            for (uint32_t h : hs) {
                const bool in_mp = std::any_of(mp.begin(), mp.end(), [&](const auto& m) { return m.first == h; });
                if (in_mp) {
                    own.push_back(
                        SlotEntry{h, true, 0, num_n_blocks, std::nullopt, m_rows[0].first, m_rows[0].second, 0});
                } else {
                    const auto sit = std::find_if(sp.begin(), sp.end(), [&](const auto& e) { return e.first == h; });
                    std::optional<CoreCoord> helper;
                    if (sit != sp.end()) {
                        helper = sit->second;
                    }
                    own.push_back(SlotEntry{
                        h, h / Hl != q, 0, helper.has_value() ? nb_split : num_n_blocks, helper, 0, block_m, -1});
                }
            }
            for (const auto& [h, hx] : sp) {
                if (hx == xy) {
                    own.push_back(SlotEntry{h, true, nb_split, num_n_blocks, std::nullopt, 0, block_m, -1});
                }
            }
            for (const auto& [h, hxs] : mp) {
                for (size_t i = 1; i < hxs.size(); ++i) {
                    if (hxs[i] == xy) {
                        own.push_back(SlotEntry{
                            h,
                            true,
                            0,
                            num_n_blocks,
                            std::nullopt,
                            m_rows[i].first,
                            m_rows[i].second,
                            static_cast<int32_t>(i)});
                    }
                }
            }
            if (!own.empty()) {
                slots[{coord, xy}] = std::move(own);
            }
        }
    }
    TT_FATAL(!slots.empty(), "all_to_all_matmul: no head slots");
    std::map<ChipHead, uint32_t> helper_slot;
    for (const auto& [coord, sp] : split_of) {
        for (const auto& [h, hx] : sp) {
            const auto& v = slots.at({coord, hx});
            const auto it = std::find_if(
                v.begin(), v.end(), [&](const SlotEntry& e) { return e.h == h && e.nb0 == nb_split && e.piece < 0; });
            TT_FATAL(it != v.end(), "all_to_all_matmul: helper slot of head {} not found", h);
            helper_slot[{coord, h}] = static_cast<uint32_t>(it - v.begin());
        }
    }
    std::map<ChipHead, std::vector<std::tuple<CoreCoord, uint32_t, uint32_t>>> pieces_of;
    for (const auto& [coord, mp] : msplit_of) {
        for (const auto& [h, hxs] : mp) {
            std::vector<std::tuple<CoreCoord, uint32_t, uint32_t>> lst;
            for (size_t i = 0; i < hxs.size(); ++i) {
                const CoreCoord xy = hxs[i].has_value() ? *hxs[i] : core_of.at(h);
                const auto& v = slots.at({coord, xy});
                const auto it = std::find_if(v.begin(), v.end(), [&](const SlotEntry& e) {
                    return e.h == h && e.piece == static_cast<int32_t>(i);
                });
                TT_FATAL(it != v.end(), "all_to_all_matmul: M-split piece {} of head {} not found", i, h);
                lst.emplace_back(xy, static_cast<uint32_t>(it - v.begin()), (m_rows[i].second - m_rows[i].first) * Kt);
            }
            pieces_of[{coord, h}] = std::move(lst);
        }
    }
    size_t slots_per_core_max = 0;
    uint32_t max_blocks = 0;
    for (const auto& [key, v] : slots) {
        slots_per_core_max = std::max(slots_per_core_max, v.size());
        uint32_t blocks = 0;
        for (const auto& e : v) {
            blocks += e.nb1 - e.nb0;
        }
        max_blocks = std::max(max_blocks, blocks);
    }
    TT_FATAL(
        slots_per_core_max <= kMaxHeadsPerCore,
        "all_to_all_matmul: {} head slots per core > {}",
        slots_per_core_max,
        kMaxHeadsPerCore);
    const uint32_t weight_depth = std::min(kWeightDepth, max_blocks);
    const uint32_t w_prefetch = std::max(1u, std::min(weight_depth, kWeightPrefetchBlocks));
    const uint32_t cap_tiles = static_cast<uint32_t>(slots_per_core_max) * slot_tiles;
    TT_FATAL(
        static_cast<uint64_t>(cap_tiles) * in_tile <= kASlotBudgetBytes,
        "all_to_all_matmul: A_RESIDENT fails ({} B of landed rows per core > {}): regime R3 (DRAM landing) is not "
        "built",
        static_cast<uint64_t>(cap_tiles) * in_tile,
        kASlotBudgetBytes);
    std::map<SendKey, std::vector<uint32_t>> data_dests;
    size_t n_ready = 1;
    for (const auto& key : send_order) {
        std::set<uint32_t> ts;
        for (const auto& [j, t] : sends.at(key)) {
            ts.insert(t);
        }
        data_dests[key] = std::vector<uint32_t>(ts.begin(), ts.end());
        n_ready = std::max(n_ready, ts.size());
    }

    plan.block_n = block_n;
    plan.num_n_blocks = num_n_blocks;
    plan.out_subblock_h = sbh;
    plan.out_subblock_w = sbw;
    plan.packet_tiles = packet_tiles;
    plan.packet_bytes = packet_bytes;
    plan.batch_chunks = batch_chunks;
    plan.port_window = port_window;
    plan.grouped = grouped;
    plan.pack = pack;
    plan.tail_split = tail_split;
    plan.weight_depth = weight_depth;
    plan.weight_chunk_tiles = std::max(1u, kWeightReadChunkBytes / w_tile);
    plan.w_prefetch = w_prefetch;
    plan.slots_per_core_max = static_cast<uint32_t>(slots_per_core_max);
    plan.cap_tiles = cap_tiles;
    plan.n_ready = static_cast<uint32_t>(n_ready);
    if (tail_split) {
        plan.active_cells = compute_cells;
    } else {
        for (const auto& [xy, hs] : assign) {
            plan.active_cells.push_back(xy);
        }
    }
    plan.active_cores = planner_rects(plan.active_cells);
    plan.all_port_cores = planner_rects(topo.port_cells);

    // ---- readies: (target coord, back, l) -> [(source coord, (d, l), g)] ----
    std::map<SendKey, std::vector<std::tuple<MeshCoordinate, uint32_t, uint32_t, uint32_t>>> ready_out;
    for (const auto& key : send_order) {
        ready_out[key];
    }
    for (const auto& key : send_order) {
        const auto& [coord, d, l] = key;
        const ChipRoute& ch = topo.chips.at(coord);
        const auto& dests = data_dests.at(key);
        for (uint32_t g = 0; g < dests.size(); ++g) {
            const MeshCoordinate& tc = ch.grp[dests[g]];
            const int32_t back = topo.chips.at(tc).route[ch.p];
            TT_FATAL(back >= 0, "all_to_all_matmul: no route back to chip position {}", ch.p);
            ready_out.at(SendKey{tc, static_cast<uint32_t>(back), l}).emplace_back(coord, d, l, g);
        }
    }

    // ---- per chip ----
    for (const auto& coord : chip_order) {
        const ChipRoute& ch = topo.chips.at(coord);
        const uint32_t q = ch.p;
        ChipPlan cp;
        cp.q = q;
        for (const auto& xy : plan.active_cells) {
            CoreWork cw;
            cw.core = xy;
            const auto sit = slots.find({coord, xy});
            if (sit != slots.end()) {
                for (const auto& e : sit->second) {
                    TT_FATAL(
                        std::max({e.nb0, e.nb1, e.m0, e.m1}) <= ::all_to_all_matmul::nbw_bits::FIELD_MASK,
                        "all_to_all_matmul: n-block / row range of head {} does not fit 8 bits",
                        e.h);
                    SlotWork sw;
                    sw.h = e.h;
                    sw.wait = e.wait;
                    sw.pre = e.piece >= 0 ? std::max(1u, std::min(weight_depth, kTailMsplitPrefetch)) : w_prefetch;
                    sw.nb0 = e.nb0;
                    sw.nb1 = e.nb1;
                    sw.m0 = e.m0;
                    sw.m1 = e.m1;
                    sw.helper = e.helper;
                    sw.helper_slot = e.helper.has_value() ? helper_slot.at({coord, e.h}) : 0;
                    sw.pace = 0;  // the source's weight pace was always 0
                    cw.slots.push_back(sw);
                }
            }
            cp.cores.push_back(std::move(cw));
        }

        cp.ports.resize(2 * static_cast<size_t>(L));
        std::vector<CoreCoord> chip_port_cells;
        for (uint32_t l = 0; l < L; ++l) {
            for (uint32_t d : {kFwd, kBwd}) {
                const SendKey key{coord, d, l};
                const auto& ents = sends.at(key);
                const auto& readies = ready_out.at(key);
                PortWork pw;
                pw.core = ch.ports[port_index(d, l)];
                chip_port_cells.push_back(pw.core);
                pw.peer = d == kFwd ? ch.next : ch.prev;
                pw.has_conn = !ents.empty() || !readies.empty();
                if (l < ch.links[d].size()) {
                    pw.link = ch.links[d][l];
                } else {
                    TT_FATAL(!pw.has_conn, "all_to_all_matmul: port ({}, {}) has no forwarding link", d, l);
                }
                for (const auto& [j, t] : ents) {
                    pw.entries.push_back(j | (t << ::all_to_all_matmul::port_reader_rt::ENTRY_DEST_SHIFT));
                }
                std::set<MeshCoordinate> dest_set;
                for (const auto& [j, t] : ents) {
                    dest_set.insert(ch.grp[t]);
                }
                for (const auto& r : readies) {
                    dest_set.insert(std::get<0>(r));
                }
                const std::vector<MeshCoordinate> dest_coords(dest_set.begin(), dest_set.end());
                const auto n_dest = static_cast<uint32_t>(dest_coords.size());
                pw.hdrs_per_dest = std::max(1u, std::min(batch_chunks, kMaxPacketHeaders / std::max(1u, n_dest)));
                TT_FATAL(
                    n_dest <= kMaxPacketHeaders,
                    "all_to_all_matmul: a port with {} destinations exceeds the header pool",
                    n_dest);
                std::vector<MeshCoordinate> data_coords;
                for (uint32_t t : data_dests.at(key)) {
                    data_coords.push_back(ch.grp[t]);
                }
                auto dest_index_of = [&](const MeshCoordinate& c) {
                    return static_cast<uint32_t>(
                        std::find(dest_coords.begin(), dest_coords.end(), c) - dest_coords.begin());
                };
                for (const auto& dc : dest_coords) {
                    PortDest pd;
                    pd.coord = dc;
                    const auto it = std::find(data_coords.begin(), data_coords.end(), dc);
                    if (it != data_coords.end()) {
                        pd.ready_index = static_cast<uint32_t>(it - data_coords.begin());
                    }
                    pw.dests.push_back(pd);
                }
                for (const auto& [tc, td, tl, g] : readies) {
                    PortReady pr;
                    pr.dest_index = dest_index_of(tc);
                    pr.target_chip = tc;
                    pr.target_port_core = topo.chips.at(tc).ports[port_index(td, tl)];
                    pr.ready_index = g;
                    pw.readies.push_back(pr);
                }
                for (const auto& [j, t] : ents) {
                    const uint32_t h = (q * Hl) + j;
                    const MeshCoordinate& tcoord = ch.grp[t];
                    const uint32_t di = dest_index_of(tcoord);
                    const auto pit = pieces_of.find({tcoord, h});
                    if (pit != pieces_of.end()) {
                        for (const auto& [pxy, s, nt] : pit->second) {
                            pw.subs.push_back(PortSub{di, nt, pxy, s});
                        }
                    } else {
                        pw.subs.push_back(PortSub{di, slot_tiles, core_of.at(h), slot_of.at({t, h})});
                    }
                }
                cp.ports[port_index(d, l)] = std::move(pw);
            }
        }
        cp.port_cores = planner_rects(chip_port_cells);
        plan.chips.emplace(coord, std::move(cp));
    }
    return plan;
}

}  // namespace

const MeshTopology& mesh_topology(MeshDevice* mesh_device, uint32_t cluster_axis, uint32_t num_links) {
    // Keyed by the mesh's id and shape (a reshape keeps the id), not by its fabric node ids: reading those walks
    // every mesh coordinate, and this lookup runs on every call.
    using Key = std::tuple<int64_t, std::vector<uint32_t>, uint32_t, uint32_t>;
    static ttsl::Indestructible<std::map<Key, MeshTopology>> cache;

    const std::vector<uint32_t> dims = planner_mesh_dims(mesh_device);
    TT_FATAL(dims.size() == 2, "all_to_all_matmul: expected a 2D mesh, got {} dims", dims.size());
    const uint32_t rows = dims[0];
    const uint32_t cols = dims[1];
    Key key{static_cast<int64_t>(mesh_device->id()), dims, cluster_axis, num_links};
    auto& map = cache.get();
    if (const auto it = map.find(key); it != map.end()) {
        return it->second;
    }

    std::vector<std::vector<MeshCoordinate>> groups;
    if (cluster_axis == 1) {
        for (uint32_t r = 0; r < rows; ++r) {
            auto& grp = groups.emplace_back();
            for (uint32_t c = 0; c < cols; ++c) {
                grp.emplace_back(r, c);
            }
        }
    } else {
        for (uint32_t c = 0; c < cols; ++c) {
            auto& grp = groups.emplace_back();
            for (uint32_t r = 0; r < rows; ++r) {
                grp.emplace_back(r, c);
            }
        }
    }

    MeshTopology topo;
    topo.num_links = num_links;
    topo.payload_bytes = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());
    for (const auto& grp : groups) {
        const auto G = static_cast<uint32_t>(grp.size());
        topo.G = G;
        for (uint32_t p = 0; p < G; ++p) {
            const MeshCoordinate& coord = grp[p];
            ChipRoute ch;
            ch.p = p;
            ch.grp = grp;
            ch.next = grp[(p + 1) % G];
            ch.prev = grp[(p + G - 1) % G];
            ch.route.assign(G, -1);
            for (uint32_t t = 0; t < G; ++t) {
                if (t == p) {
                    continue;
                }
                const auto d = planner_first_hop_dir(mesh_device, coord, grp[t], ch.next, ch.prev);
                if (!d.has_value()) {
                    TT_THROW(
                        "all_to_all_matmul: the fabric route {} -> {} leaves through neither ring neighbour",
                        coord,
                        grp[t]);
                }
                ch.route[t] = static_cast<int32_t>(*d);
            }
            topo.chips.emplace(coord, std::move(ch));
        }
    }

    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    std::set<CoreCoord> port_set;
    for (auto& [coord, ch] : topo.chips) {
        const auto src = mesh_device->get_fabric_node_id(coord);
        for (uint32_t d : {kFwd, kBwd}) {
            const MeshCoordinate& peer = d == kFwd ? ch.next : ch.prev;
            auto li = tt::tt_fabric::get_forwarding_link_indices(src, mesh_device->get_fabric_node_id(peer));
            if (li.size() > num_links) {
                li.resize(num_links);
            }
            ch.links[d] = std::move(li);
        }
        try {
            ch.ports = planner_place_ports(mesh_device, coord, ch, num_links, grid);
        } catch (const std::exception&) {  // placement is a perf choice; fall back to the simple layout
            ch.ports.clear();
        }
        if (ch.ports.empty()) {
            ch.ports = planner_fallback_ports(grid, num_links);
        }
        port_set.insert(ch.ports.begin(), ch.ports.end());
    }
    topo.port_cells.assign(port_set.begin(), port_set.end());
    for (size_t y = 0; y < grid.y; ++y) {
        for (size_t x = 0; x < grid.x; ++x) {
            const CoreCoord cell(x, y);
            if (!port_set.contains(cell)) {
                topo.compute_cells.push_back(cell);
            }
        }
    }
    topo.compute_cores = planner_rects(topo.compute_cells);
    return map.emplace(std::move(key), std::move(topo)).first->second;
}

const Plan& solve_plan(const MeshTopology& topology, const PlanInputs& inputs) {
    static ttsl::Indestructible<std::map<std::pair<const MeshTopology*, PlanInputs>, Plan>> cache;
    auto& map = cache.get();
    const std::pair<const MeshTopology*, PlanInputs> key{&topology, inputs};
    if (const auto it = map.find(key); it != map.end()) {
        return it->second;
    }
    // build_plan throws before anything is inserted, so a refused input leaves no entry.
    return map.emplace(key, build_plan(topology, inputs)).first->second;
}

tt::tt_metal::TensorSpec scratch_spec(const MeshTopology& topology, const Plan& plan, tt::tt_metal::DataType dtype) {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;
    const auto n = static_cast<uint32_t>(topology.compute_cells.size());
    const tt::tt_metal::MemoryConfig mc(
        tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        tt::tt_metal::BufferType::L1,
        tt::tt_metal::ShardSpec(
            topology.compute_cores,
            {plan.cap_tiles * TILE_HEIGHT, TILE_WIDTH},
            tt::tt_metal::ShardOrientation::ROW_MAJOR));
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({n * plan.cap_tiles * TILE_HEIGHT, TILE_WIDTH}),
        tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), mc));
}

const MeshResources& mesh_resources(MeshDevice* mesh_device, const MeshTopology& topology, const Plan& plan) {
    // Keyed by the memoized topology (one per mesh id, shape, cluster_axis and num_links; its entries are never
    // erased), so a call copies no core list to look its semaphores up.
    static ttsl::Indestructible<std::map<const MeshTopology*, MeshResources>> cache;
    MeshResources& res = cache.get()[&topology];
    bool changed = false;
    while (res.ready.size() < plan.n_ready) {  // one ready counter per destination chip of a port
        res.ready.push_back(
            ttnn::global_semaphore::create_global_semaphore(mesh_device, planner_rects(topology.port_cells), 0));
        changed = true;
    }
    while (res.arrival.size() < plan.slots_per_core_max) {
        res.arrival.push_back(ttnn::global_semaphore::create_global_semaphore(mesh_device, topology.compute_cores, 0));
        changed = true;
    }
    if (changed) {
        res.arrival_addrs.clear();
        res.ready_addrs.clear();
        for (const auto& sem : res.arrival) {
            res.arrival_addrs.push_back(
                static_cast<uint32_t>(ttnn::global_semaphore::get_global_semaphore_address(sem)));
        }
        for (const auto& sem : res.ready) {
            res.ready_addrs.push_back(static_cast<uint32_t>(ttnn::global_semaphore::get_global_semaphore_address(sem)));
        }
        // every chip's counters are zero before any chip's first increment can arrive
        tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt);
    }
    return res;
}

}  // namespace ttnn::operations::all_to_all_matmul
