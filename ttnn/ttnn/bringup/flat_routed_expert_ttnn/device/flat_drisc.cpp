// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "flat_drisc.hpp"

#include <chrono>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>

#include "tt_metal/impl/context/metal_context.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/impl/program/program_impl.hpp"
#include "tt_metal/llrt/hal.hpp"

namespace tt::tt_metal::slow_dispatch {
void WriteRuntimeArgsToDevice(IDevice& device, Program& program, bool force_slow_dispatch);
void LaunchProgramAsync(IDevice& device, Program& program, bool force_slow_dispatch);
void WaitProgramDone(IDevice& device, const Program& program);
}  // namespace tt::tt_metal::slow_dispatch

namespace ttnn::operations::bringup::flat_routed_expert {

using namespace tt::tt_metal;

namespace {
// se_drisc.hpp
constexpr uint32_t STOP = 16, READY = 32, READY_MAGIC = 0xD815C0DE, SENDER_ZONE = 1024;

struct Running {
    std::vector<IDevice*> devices;
    std::vector<std::unique_ptr<Program>> programs;
    std::vector<CoreCoord> cores;  // logical, one per tile
    uint32_t n_slots = 0;
};
std::mutex mu;
std::map<const distributed::MeshDevice*, Running> running;

const Hal& hal() { return MetalContext::instance().hal(); }
}  // namespace

bool flat_drisc_enabled() {
    const char* v = std::getenv("MIMO_FL_DRISC");
    return v != nullptr && std::atoi(v) != 0;
}

// (above the DRISC L1 arena's sender-state zone, as the tensor prefetcher's kernel)
uint32_t flat_drisc_base() {
    return static_cast<uint32_t>(hal().get_dev_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED)) +
           SENDER_ZONE;
}

uint32_t flat_drisc_size() {
    return hal().get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::UNRESERVED) - SENDER_ZONE;
}

namespace {
IDevice* chip(IDevice* device) {
    if (auto* mesh = dynamic_cast<distributed::MeshDevice*>(device)) {
        return mesh->get_devices().at(0);
    }
    return device;
}
uint32_t env_or(const char* name, uint32_t dflt) {
    const char* v = std::getenv(name);
    return v != nullptr ? static_cast<uint32_t>(std::atoi(v)) : dflt;
}
// the bank's DRISC tile that is no view's endpoint (both its NIUs in stream mode: it may send on either NoC)
uint32_t free_tile(IDevice* device, uint32_t bank) {
    device = chip(device);
    const auto& soc = MetalContext::instance().get_cluster().get_soc_desc(device->id());
    for (uint32_t y = 1; y < soc.dram_bank_endpoint_coords.at(bank).size(); ++y) {
        const CoreCoord v = device->virtual_core_from_logical_core(CoreCoord(bank, y), tt::CoreType::DRAM);
        if (soc.get_dram_endpoint_noc_mask(v) == 0) {
            return y;
        }
    }
    TT_THROW("MIMO_FL_DRISC: bank {} has no DRAM tile outside its endpoints", bank);
}
}  // namespace

// MIMO_FL_DRISC_TILES: 2 (default) - both Metal tiles of a bank, a slot each (tile 1, its bank's NOC1 endpoint, sends on
// NOC0); 1 - the bank's tile that is no endpoint sends both slots, one per NoC (capped at ~21 GB/s: small M is slow)
uint32_t flat_drisc_tiles() { return env_or("MIMO_FL_DRISC_TILES", 2); }

CoreCoord flat_drisc_logical(IDevice* device, uint32_t r, uint32_t banks) {
    return flat_drisc_tiles() == 1 ? CoreCoord(r % banks, free_tile(device, r % banks))
                                   : CoreCoord(r % banks, 1 + r / banks);
}

uint32_t flat_drisc_job(uint32_t r, uint32_t banks) { return flat_drisc_tiles() == 1 ? r / banks : 0; }

namespace {
CoreCoord noc0_of(IDevice* device, const CoreCoord& logical, tt::CoreType type) {
    device = chip(device);
    const auto& soc = MetalContext::instance().get_cluster().get_soc_desc(device->id());
    const CoreCoord v = device->virtual_core_from_logical_core(logical, type);
    const tt::CoreType umd_type = type == tt::CoreType::WORKER ? tt::CoreType::TENSIX : type;
    const auto q =
        soc.translate_coord_to(tt::umd::CoreCoord(v.x, v.y, umd_type, tt::CoordSystem::TRANSLATED), tt::CoordSystem::NOC0);
    return CoreCoord(q.x, q.y);
}
}  // namespace

CoreCoord flat_drisc_physical(IDevice* device, uint32_t r, uint32_t banks) {
    return noc0_of(device, flat_drisc_logical(device, r, banks), tt::CoreType::DRAM);
}

uint32_t flat_drisc_noc(IDevice* device, uint32_t r, uint32_t banks) {
    // one tile per bank: its first slot on NOC0, the second on NOC1; two tiles: the non-endpoint tile's slot on NOC1
    // (M 256 / 512 55.0 / 81.5 -> 50.6 / 77.2 us / expert); MIMO_FL_DRISC_NOC1=0: all on NOC0
    if (flat_drisc_tiles() == 1) {
        return env_or("MIMO_FL_DRISC_NOC1", 1) ? r / banks : 0;
    }
    if (!env_or("MIMO_FL_DRISC_NOC1", 1)) {  // (default: tile 2's half of the weights off the x multicast's NOC0)
        return 0;
    }
    device = chip(device);
    const auto& soc = MetalContext::instance().get_cluster().get_soc_desc(device->id());
    const CoreCoord v =
        device->virtual_core_from_logical_core(flat_drisc_logical(device, r, banks), tt::CoreType::DRAM);
    return (soc.get_dram_endpoint_noc_mask(v) & 0x2) ? 0 : 1;
}

CoreCoord flat_worker_noc0(IDevice* device, const CoreCoord& logical) {
    return noc0_of(device, logical, tt::CoreType::WORKER);
}

void flat_drisc_ensure(distributed::MeshDevice* mesh, uint32_t n_slots, uint32_t banks) {
    std::lock_guard<std::mutex> lk(mu);
    if (running.contains(mesh)) {
        TT_FATAL(running.at(mesh).n_slots == n_slots, "flat DRISC senders: running for another slot count");
        return;
    }
    TT_FATAL(hal().has_programmable_core_type(HalProgrammableCoreType::DRAM), "MIMO_FL_DRISC: no DRAM cores");
    TT_FATAL(n_slots <= 2 * banks, "MIMO_FL_DRISC: {} reader slots > 2 DRISCs x {} banks", n_slots, banks);
    auto& cluster = MetalContext::instance().get_cluster();
    const uint64_t noc_off = hal().get_l1_noc_offset(HalProgrammableCoreType::DRAM);
    const uint32_t base = flat_drisc_base(), size = flat_drisc_size();
    Running run;
    run.n_slots = n_slots;
    IDevice* dev0 = mesh->get_devices().at(0);
    // per tile: its slots' NoCs by job index
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> tile_nocs;
    for (uint32_t r = 0; r < n_slots; ++r) {
        const CoreCoord c = flat_drisc_logical(dev0, r, banks);
        auto& v = tile_nocs[{c.x, c.y}];
        const uint32_t k = flat_drisc_job(r, banks);
        if (v.size() <= k) {
            v.resize(k + 1, 0);
        }
        v[k] = flat_drisc_noc(dev0, r, banks);
    }
    for (const auto& [xy, nocs] : tile_nocs) {
        run.cores.emplace_back(xy.first, xy.second);
    }
    std::map<std::string, std::string> defs;
    // perf knob: MIMO_FL_DRISC_MPFE=w[,o] - the GDDR arbitration weights (0-7) of the DRISC tiles' ports (w) and of the
    // bank's ordinary port (o: the NOC0 endpoint's, the Tensix cores' DRAM traffic)
    if (const char* w = std::getenv("MIMO_FL_DRISC_MPFE")) {
        const std::string ws(w);
        const auto comma = ws.find(',');
        defs["SE_DRISC_MPFE"] = ws.substr(0, comma);
        if (comma != std::string::npos) {
            defs["SE_DRISC_MPFE_ORD"] = ws.substr(comma + 1);
        }
    }
    if (const char* b = std::getenv("MIMO_FL_DRISC_BURST")) {  // (probe: DMA AXI burst beats; 0 = untouched)
        defs["SE_DRISC_BURST"] = b;
    }
    if (const char* m = std::getenv("MIMO_FL_DRISC_DMA_MAX")) {  // (probe: DMA reads in flight per stream)
        defs["SE_DRISC_DMA_MAX"] = m;
    }
    if (std::getenv("MIMO_FL_W_NOREAD")) {  // (perf probe: no weight DMA, garbage)
        defs["SE_W_NOREAD"] = "1";
    }
    if (const char* vc = std::getenv("MIMO_FL_FWD_VC")) {  // (perf probe: the weight writes' unicast VC)
        defs["SE_FWD_VC"] = vc;
    }
    for (IDevice* dev : mesh->get_devices()) {
        auto program = std::make_unique<Program>();
        // one kernel per tile: its MPFE port is P(1 + endpoint index), as test_drisc_flat_bench.cpp
        for (const auto& c : run.cores) {
            auto d = defs;
            if (d.contains("SE_DRISC_MPFE")) {
                d["SE_DRISC_MPFE_PORT"] = std::to_string(1 + c.y);
            }
            const auto& nocs = tile_nocs.at({c.x, c.y});
            // a bank's two tiles DMA on different streams: both on stream 0 with ~7 reads in flight each, next to heavy
            // NOC1 DRAM reads through the bank's NOC1 endpoint (MIMO_FL_DN_NOC0=1), wedged those NoC reads (3 of 3
            // runs; split streams 2 of 2 clean, as is a 2-read cap). MIMO_FL_DRISC_STREAM_SPLIT=0 restores the hang.
            if (env_or("MIMO_FL_DRISC_STREAM_SPLIT", 1) && flat_drisc_tiles() == 2 && c.y == 2) {
                d["SE_DRISC_STREAM0"] = "1";
            }
            const auto k = CreateKernel(
                *program,
                "ttnn/ttnn/bringup/flat_routed_expert_ttnn/device/kernels/se_drisc_gu.cpp",
                c,
                DramConfig{.noc = nocs[0] ? NOC::NOC_1 : NOC::NOC_0, .defines = d});
            std::vector<uint32_t> rt{base, size, static_cast<uint32_t>(nocs.size())};
            rt.insert(rt.end(), nocs.begin(), nocs.end());
            SetRuntimeArgs(*program, k, c, rt);
        }
        const uint32_t zero = 0;
        for (const auto& c : run.cores) {
            const CoreCoord v = dev->virtual_core_from_logical_core(c, tt::CoreType::DRAM);
            cluster.write_core(&zero, 4, tt_cxy_pair(dev->id(), v), noc_off + base + STOP);
            cluster.write_core(&zero, 4, tt_cxy_pair(dev->id(), v), noc_off + base + READY);
        }
        cluster.l1_barrier(dev->id());
        program->impl().compile(dev, /*force_slow_dispatch=*/true);
        slow_dispatch::LaunchProgramAsync(*dev, *program, /*force_slow_dispatch=*/true);
        run.devices.push_back(dev);
        run.programs.push_back(std::move(program));
    }
    // ready: every sender has zeroed its words (else the first call's grants could be wiped)
    const auto t0 = std::chrono::steady_clock::now();
    for (IDevice* dev : run.devices) {
        for (const auto& c : run.cores) {
            const CoreCoord v = dev->virtual_core_from_logical_core(c, tt::CoreType::DRAM);
            uint32_t w = 0;
            while (true) {
                cluster.read_core(&w, 4, tt_cxy_pair(dev->id(), v), noc_off + base + READY);
                if (w == READY_MAGIC) {
                    break;
                }
                TT_FATAL(
                    std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10),
                    "flat DRISC sender on chip {} DRAM core {} did not start",
                    dev->id(),
                    c.str());
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
    }
    running.emplace(mesh, std::move(run));
}

void flat_drisc_dump(distributed::MeshDevice* mesh) {
    std::lock_guard<std::mutex> lk(mu);
    auto it = running.find(mesh);
    if (it == running.end()) {
        return;
    }
    auto& cluster = MetalContext::instance().get_cluster();
    const uint64_t noc_off = hal().get_l1_noc_offset(HalProgrammableCoreType::DRAM);
    for (IDevice* dev : it->second.devices) {
        for (const auto& c : it->second.cores) {
            const CoreCoord v = dev->virtual_core_from_logical_core(c, tt::CoreType::DRAM);
            std::vector<uint32_t> w(2048 / 4);
            cluster.read_core(w.data(), w.size() * 4, tt_cxy_pair(dev->id(), v), noc_off + flat_drisc_base());
            std::string jobs;
            for (uint32_t k = 0; k < 2; ++k) {
                const uint32_t* st = &w[(384 + 32 * k) / 4];
                const uint32_t* d = &w[(512 + 640 * k) / 4];
                std::string cr;
                for (uint32_t j = 0; j < 4; ++j) {
                    cr += std::to_string(w[(128 + 128 * k) / 4 + 4 * j]) + " ";
                }
                jobs += fmt::format(
                    " | job {}: posted {} grants {}state call {} units {} rd {} wr {} fl {} loads {} stream {} hw reads "
                    "{} ours {} desc gddr {:#x} blk {} R {} G {} land {:#x} n_load {} rxy0 {:#x}",
                    k,
                    w[(k ? 64 : 0) / 4],
                    cr,
                    st[0],
                    st[1],
                    st[2],
                    st[3],
                    st[4],
                    st[5],
                    k,
                    st[6],
                    st[7],
                    d[0],
                    d[2],
                    d[4],
                    d[5],
                    d[7],
                    d[9],
                    d[11]);
            }
            log_info(
                tt::LogOp,
                "flat DRISC chip {} {} (virt {}): stop {} ready {:#x} done {}{}",
                dev->id(),
                c.str(),
                v.str(),
                w[4],
                w[8],
                w[12],
                jobs);
        }
    }
}

void flat_drisc_stop(distributed::MeshDevice* mesh) {
    std::lock_guard<std::mutex> lk(mu);
    auto it = running.find(mesh);
    if (it == running.end()) {
        return;
    }
    auto& run = it->second;
    auto& cluster = MetalContext::instance().get_cluster();
    const uint64_t noc_off = hal().get_l1_noc_offset(HalProgrammableCoreType::DRAM);
    const uint32_t one = 1;
    for (uint32_t d = 0; d < run.devices.size(); ++d) {
        IDevice* dev = run.devices[d];
        for (const auto& c : run.cores) {
            const CoreCoord v = dev->virtual_core_from_logical_core(c, tt::CoreType::DRAM);
            cluster.write_core(&one, 4, tt_cxy_pair(dev->id(), v), noc_off + flat_drisc_base() + STOP);
        }
        cluster.l1_barrier(dev->id());
    }
    const auto t0 = std::chrono::steady_clock::now();
    bool exited = true;
    for (IDevice* dev : run.devices) {
        for (const auto& c : run.cores) {
            const CoreCoord v = dev->virtual_core_from_logical_core(c, tt::CoreType::DRAM);
            uint32_t w = 1;
            while (exited) {
                cluster.read_core(&w, 4, tt_cxy_pair(dev->id(), v), noc_off + flat_drisc_base() + READY);
                if (w == 0) {
                    break;
                }
                if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(5)) {
                    log_error(tt::LogOp, "flat DRISC sender chip {} {} did not stop", dev->id(), c.str());
                    exited = false;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
    }
    if (exited) {
        for (uint32_t d = 0; d < run.devices.size(); ++d) {
            slow_dispatch::WaitProgramDone(*run.devices[d], *run.programs[d]);
        }
    }
    running.erase(it);
}

}  // namespace ttnn::operations::bringup::flat_routed_expert
