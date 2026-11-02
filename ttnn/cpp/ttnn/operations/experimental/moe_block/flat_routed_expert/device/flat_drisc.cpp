// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_drisc.hpp"

#include <array>
#include <chrono>
#include <map>
#include <mutex>
#include <string>

#include <fmt/format.h>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

namespace drisc_detail {
using tt::tt_metal::CoreCoord;
namespace mx = tt::tt_metal::experimental;

// se_drisc.hpp
constexpr uint32_t STOP = 16, READY = 32, READY_MAGIC = 0xD815C0DE;
constexpr uint32_t PROF = 1792, PROF_WORDS = 64;  // (se_drisc.hpp SE_DRISC_PROF)
constexpr const char* SENDER_KERNEL =
    "ttnn/cpp/ttnn/operations/experimental/moe_block/flat_routed_expert/device/kernels/se_drisc_gu.cpp";
constexpr auto READY_TIMEOUT = std::chrono::seconds(10);
constexpr auto STOP_TIMEOUT = std::chrono::seconds(5);

// Tells every sender to exit and waits for each to acknowledge; false if one did not in time.
bool signal_stop(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const std::vector<CoreCoord>& cores, uint32_t base) {
    const std::array<uint32_t, 1> one{1};
    for (const auto& c : cores) {
        mx::WriteDramCoreL1(mesh_device, c, base + STOP, one);
    }
    bool exited = true;
    for (const auto& c : cores) {
        if (!mx::WaitForDramCoreL1Word(mesh_device, c, base + READY, 0, STOP_TIMEOUT)) {
            log_error(tt::LogOp, "flat DRISC sender on DRAM core {} did not stop", c.str());
            exited = false;
        }
    }
    return exited;
}

// One sender set per mesh device: they own the DRAM cores' working region.
std::mutex registry_mutex;
std::map<const tt::tt_metal::distributed::MeshDevice*, std::weak_ptr<FlatDriscSenders>> registry;
}  // namespace drisc_detail
using namespace drisc_detail;

tt::tt_metal::distributed::MeshCoordinate flat_reference_coord(tt::tt_metal::distributed::MeshDevice& mesh_device) {
    const auto ids = mesh_device.get_device_ids();
    TT_FATAL(!ids.empty(), "flat_routed_expert: the mesh device has no local chip");
    return mesh_device.get_view().find_device(ids.front());
}

CoreCoord flat_worker_noc0(tt::tt_metal::distributed::MeshDevice& mesh_device, const CoreCoord& logical) {
    return mx::GetWorkerNoc0Core(mesh_device, flat_reference_coord(mesh_device), logical);
}

FlatDriscTopology FlatDriscTopology::of(tt::tt_metal::distributed::MeshDevice& mesh_device, FlatDriscTiles tiles) {
    TT_FATAL(
        mx::AreDramCoreKernelsSupported(mesh_device),
        "flat_routed_expert: DRISC gate/up weight senders only, and this device has no programmable DRAM cores");
    FlatDriscTopology t;
    t.tiles = tiles;
    t.coord = flat_reference_coord(mesh_device);
    t.banks = mesh_device.dram_grid_size().x;
    const auto region = mx::GetDramCoreKernelL1Region(mesh_device);
    t.base = region.base;
    t.size = region.size;
    for (uint32_t b = 0; b < t.banks; ++b) {
        t.cores.push_back(mx::GetDramCores(mesh_device, t.coord, b));
    }
    return t;
}

const mx::DramCoreInfo& FlatDriscTopology::info(const CoreCoord& logical) const {
    return cores.at(logical.x).at(logical.y);
}

// the bank's DRISC tile that is no view's endpoint (both its NIUs in stream mode: it may send on either NoC)
uint32_t FlatDriscTopology::free_tile(uint32_t bank) const {
    const auto& c = cores.at(bank);
    for (uint32_t y = 1; y < c.size(); ++y) {
        if (c[y].endpoint_noc_mask == 0) {
            return y;
        }
    }
    TT_THROW("flat DRISC: bank {} has no DRAM tile outside its endpoints", bank);
}

uint32_t FlatDriscTopology::home_tile(uint32_t bank) const {
    return tiles == FlatDriscTiles::noc1_endpoint_only ? noc1_tile(bank) : free_tile(bank);
}

CoreCoord FlatDriscTopology::logical(uint32_t r) const { return CoreCoord(r % banks, home_tile(r % banks)); }

uint32_t FlatDriscTopology::noc(uint32_t r) const {
    if (tiles == FlatDriscTiles::noc1_endpoint_only) {
        return 0;  // (its NOC1 NIU is the bank's NOC1 endpoint: NOC2AXI)
    }
    return job(r, banks) == 0 ? 1 : 0;
}

// the free tile: job k on stream k. The NOC1-endpoint tile: its first job on stream 1 (as the alternate's one job in
// switch mode), its second on stream 0
uint32_t FlatDriscTopology::stream(uint32_t k) const {
    return tiles == FlatDriscTiles::noc1_endpoint_only ? (1 + k) & 1 : k & 1;
}

// switch mode: a bank's second slot is sent by its other Metal tile (the bank's NOC1 endpoint, on NOC0, DMA stream 1)
// on every call, the free tile's second job skipped (both tiles on every call: one tile per bank for the whole call
// starved its small experts - K3 2 x 4 overlap, Zipf spread / packed fused 3369 / 6148 -> 2589 / 5906 us)
// (one-tile choices: none, the home tile sends both of the bank's slots)
std::pair<CoreCoord, uint32_t> FlatDriscTopology::alt(uint32_t r) const {
    if (r < banks || tiles != FlatDriscTiles::both) {
        return {CoreCoord(0, 0), ~0u};
    }
    const uint32_t b = r % banks;
    return {CoreCoord(b, noc1_tile(b)), 0};
}

std::pair<CoreCoord, uint32_t> FlatDriscTopology::sender(uint32_t r) const {
    const auto [a, j] = alt(r);
    if (j != ~0u) {
        return {info(a).noc0_core, 0u};
    }
    return {info(logical(r)).noc0_core, noc(r)};
}

uint32_t FlatDriscTopology::virtual_xy(const CoreCoord& logical) const {
    const CoreCoord v = info(logical).virtual_core;
    return static_cast<uint32_t>((v.x << 16) | v.y);
}

std::shared_ptr<FlatDriscSenders> FlatDriscSenders::get(
    tt::tt_metal::distributed::MeshDevice& mesh_device, uint32_t n_slots, FlatDriscTiles tiles, FlatDriscProbe probe) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    if (auto it = registry.find(&mesh_device); it != registry.end()) {
        if (auto live = it->second.lock(); live && live->running()) {
            TT_FATAL(
                live->n_slots() == n_slots,
                "flat DRISC senders: running for {} slots, {} requested",
                live->n_slots(),
                n_slots);
            if (live->tiles() == tiles && live->probe() == probe) {
                return live;
            }
            // another tile choice (or probe): the running set owns the DRAM cores (and runs other kernels' tiles), so
            // it stops (its holders' flat calls then fail the running check) and the requested set launches
            log_info(
                tt::LogOp,
                "flat DRISC senders: stopping the {} / {} set for a {} / {} one",
                to_string(live->tiles()),
                to_string(live->probe()),
                to_string(tiles),
                to_string(probe));
            live->stop();
        }
    }
    std::shared_ptr<FlatDriscSenders> senders(new FlatDriscSenders(mesh_device, n_slots, tiles, probe));
    registry[&mesh_device] = senders;
    return senders;
}

bool FlatDriscSenders::running_on(
    const tt::tt_metal::distributed::MeshDevice& mesh_device,
    uint32_t n_slots,
    FlatDriscTiles tiles,
    FlatDriscProbe probe) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto it = registry.find(&mesh_device);
    if (it == registry.end()) {
        return false;
    }
    auto live = it->second.lock();
    return live && live->running() && live->n_slots() == n_slots && live->tiles() == tiles && live->probe() == probe;
}

FlatDriscSenders::FlatDriscSenders(
    tt::tt_metal::distributed::MeshDevice& mesh_device, uint32_t n_slots, FlatDriscTiles tiles, FlatDriscProbe probe) :
    mesh_device_(&mesh_device),
    n_slots_(n_slots),
    tiles_(tiles),
    probe_(probe),
    topo_(FlatDriscTopology::of(mesh_device, tiles)) {
    const uint32_t banks = topo_.banks;
    TT_FATAL(n_slots <= 2 * banks, "flat DRISC: {} reader slots > 2 DRISCs x {} banks", n_slots, banks);
    // per tile: its jobs' (NoC, DMA stream) by job index
    std::map<std::pair<uint32_t, uint32_t>, std::vector<std::pair<uint32_t, uint32_t>>> tile_jobs;
    for (uint32_t r = 0; r < n_slots; ++r) {
        const CoreCoord c = topo_.logical(r);
        auto& v = tile_jobs[{c.x, c.y}];
        const uint32_t k = FlatDriscTopology::job(r, banks);
        if (v.size() <= k) {
            v.resize(k + 1, {0, 0});
        }
        v[k] = {topo_.noc(r), topo_.stream(k)};
    }
    // (switch mode: the alternate tiles' one job on NOC0, DMA stream 1 - a bank's two tiles both on stream 0, each
    // with ~7 reads in flight, next to heavy NOC1 DRAM reads through the bank's NOC1 endpoint wedged those NoC reads)
    for (uint32_t r = 0; r < n_slots; ++r) {
        const auto [ac, aj] = topo_.alt(r);
        if (aj != ~0u) {
            tile_jobs[{ac.x, ac.y}] = {{0, 1}};
        }
    }
    for (const auto& [xy, jobs] : tile_jobs) {
        cores_.emplace_back(xy.first, xy.second);
    }
    // the kernels' stop / ready words zero before they start (each sender zeroes its other words itself)
    const std::array<uint32_t, 1> zero{0};
    for (const auto& c : cores_) {
        mx::WriteDramCoreL1(mesh_device, c, topo_.base + STOP, zero);
        mx::WriteDramCoreL1(mesh_device, c, topo_.base + READY, zero);
    }
    const uint32_t base = topo_.base, size = topo_.size;
    std::string tiles_log;
    for (const auto& c : cores_) {
        const auto& jobs = tile_jobs.at({c.x, c.y});
        tiles_log += fmt::format(" {}:", c.str());
        for (const auto& [noc, stream] : jobs) {
            tiles_log += fmt::format(" noc{}/s{}", noc, stream);
        }
    }
    log_debug(tt::LogOp, "flat DRISC senders ({}, {} slots):{}", to_string(tiles), n_slots, tiles_log);
    kernels_ = mx::LaunchDramCoreKernels(
        mesh_device,
        [&](const tt::tt_metal::distributed::MeshCoordinate& coord) {
            std::vector<mx::DramCoreKernelConfig> out;
            for (const auto& c : cores_) {
                const auto& jobs = tile_jobs.at({c.x, c.y});
                const auto bank_cores = mx::GetDramCores(mesh_device, coord, c.x);
                mx::DramCoreKernelConfig k{
                    .kernel_file = SENDER_KERNEL,
                    .core = c,
                    .noc = jobs[0].first ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0};
                if (probe != FlatDriscProbe::none) {  // (se_drisc_gu.cpp: counters; probe variants)
                    // (call_time: the call cycles only; the others count every step)
                    k.defines["SE_DRISC_PROF"] = probe == FlatDriscProbe::call_time ? "1" : "2";
                    k.defines["SE_DRISC_PROBE"] = std::to_string(static_cast<uint32_t>(probe));
                }
                // (se_drisc_gu.cpp RT) a tile's jobs never share a stream; nor do a bank's two tiles in switch mode
                k.runtime_args = {base, size, static_cast<uint32_t>(jobs.size())};
                for (const auto& j : jobs) {
                    k.runtime_args.push_back(j.first);
                }
                for (const auto& j : jobs) {
                    k.runtime_args.push_back(j.second);
                }
                // MPFE ports: this tile's, the bank's NOC0 / NOC1 endpoints'
                k.runtime_args.push_back(bank_cores.at(c.y).mpfe_port);
                k.runtime_args.push_back(bank_cores.at(0).mpfe_port);
                k.runtime_args.push_back(bank_cores.at(1).mpfe_port);
                out.push_back(std::move(k));
            }
            return out;
        },
        // the mesh device closes with the senders running: its queues are finished, tell them to exit
        [mesh = &mesh_device, cores = cores_, base] { signal_stop(*mesh, cores, base); });
    // ready: every sender has zeroed its words (else the first call's grants could be wiped)
    for (const auto& c : cores_) {
        TT_FATAL(
            mx::WaitForDramCoreL1Word(mesh_device, c, base + READY, READY_MAGIC, READY_TIMEOUT),
            "flat DRISC sender on DRAM core {} did not start",
            c.str());
    }
}

std::vector<std::pair<CoreCoord, std::vector<uint32_t>>> FlatDriscSenders::read_prof() const {
    std::vector<std::pair<CoreCoord, std::vector<uint32_t>>> out;
    if (!running()) {
        return out;
    }
    for (const auto& c : cores_) {
        out.emplace_back(c, mx::ReadDramCoreL1(*mesh_device_, topo_.coord, c, topo_.base + PROF, PROF_WORDS));
    }
    return out;
}

bool FlatDriscSenders::running() const { return kernels_ && kernels_->running(); }

void FlatDriscSenders::stop() {
    if (!running()) {
        return;
    }
    tt::tt_metal::distributed::Finish(mesh_device_->mesh_command_queue());
    if (signal_stop(*mesh_device_, cores_, topo_.base)) {
        kernels_->wait_done();
    }
    kernels_.reset();
}

FlatDriscSenders::~FlatDriscSenders() {
    stop();
    std::lock_guard<std::mutex> lock(registry_mutex);
    if (auto it = registry.find(mesh_device_); it != registry.end() && it->second.expired()) {
        registry.erase(it);
    }
}

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
