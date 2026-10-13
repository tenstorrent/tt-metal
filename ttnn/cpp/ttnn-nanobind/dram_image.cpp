// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Prototype DRAM-image weight loading (unit mesh only).
//
// A DRAM image is the raw content of one DRAM address range [address, address + bytes_per_bank) in every bank,
// stored in interleaved page order: page j of an interleaved view buffer at `address` with `page_size` maps to
// bank j % num_banks at address + (j / num_banks) * page_size. read_region dumps the range; pull_region streams an
// image file back into it with Tensix cores pulling from pinned 1 GB hugepages over PCIe (no host push, no ttnn
// host tensors). Allocations replayed afterwards at the same addresses find their bytes already in place.

#include "dram_image.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <thread>
#include <chrono>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_event.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::dram_image {
namespace {

using namespace tt::tt_metal;
using Clock = std::chrono::steady_clock;
constexpr uint64_t GB = 1ull << 30;
constexpr uint32_t BURST = 16384;
constexpr const char* KERNEL = "tests/tt_metal/tt_metal/perf_microbenchmark/dram_image/kernels/pcie_pull_to_dram.cpp";

double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

std::shared_ptr<distributed::MeshBuffer> view(
    distributed::MeshDevice& mesh, uint64_t address, uint64_t bytes_per_bank, uint32_t page_size) {
    TT_FATAL(mesh.num_devices() == 1, "dram_image prototype supports a unit mesh only");
    const uint32_t banks = mesh.get_device(0, 0)->num_dram_channels();
    TT_FATAL(bytes_per_bank % page_size == 0, "bytes_per_bank must be a multiple of page_size");
    distributed::DeviceLocalBufferConfig local{.page_size = page_size, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig global{.size = bytes_per_bank * banks};
    return distributed::MeshBuffer::create(global, local, &mesh, address);
}

uint8_t* map_hugepage_1g(int idx) {
    std::string path = fmt::format("/dev/hugepages-1G/dram_image_{}_{}", getpid(), idx);
    int fd = open(path.c_str(), O_CREAT | O_RDWR, 0600);
    TT_FATAL(fd >= 0, "open {} failed", path);
    void* p = mmap(nullptr, GB, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
    close(fd);
    unlink(path.c_str());
    TT_FATAL(p != MAP_FAILED, "mmap of a 1 GB hugepage failed (free 1G hugepages?)");
    return static_cast<uint8_t*>(p);
}

// raw region -> file
std::map<std::string, double> read_region(
    const std::shared_ptr<distributed::MeshDevice>& mesh,
    uint64_t address,
    uint64_t bytes_per_bank,
    uint32_t page_size,
    const std::string& path) {
    auto t0 = Clock::now();
    auto buf = view(*mesh, address, bytes_per_bank, page_size);
    std::vector<uint8_t> host;
    {
        nb::gil_scoped_release nogil;
        distributed::EnqueueReadMeshBuffer(mesh->mesh_command_queue(), host, buf, true);
    }
    double t_read = since(t0);
    FILE* f = fopen(path.c_str(), "wb");
    TT_FATAL(f, "cannot write {}", path);
    TT_FATAL(fwrite(host.data(), 1, host.size(), f) == host.size(), "short write to {}", path);
    fclose(f);
    return {{"bytes", double(host.size())}, {"read_s", t_read}, {"total_s", since(t0)}};
}

// file -> raw region, Tensix cores pulling over PCIe from a ring of pinned 1 GB hugepages
std::map<std::string, double> pull_region(
    const std::shared_ptr<distributed::MeshDevice>& mesh,
    uint64_t address,
    uint64_t bytes_per_bank,
    uint32_t page_size,
    const std::string& path,
    uint32_t slots,
    uint32_t num_cores,
    uint32_t read_threads) {
    auto t0 = Clock::now();
    TT_FATAL(GB % page_size == 0 && page_size % BURST == 0, "page_size must divide 1 GB and be a multiple of 16 KB");
    auto buf = view(*mesh, address, bytes_per_bank, page_size);
    const uint64_t total = buf->size();
    const uint32_t n_chunks = (total + GB - 1) / GB;
    const uint32_t chunk_pages = GB / page_size;
    slots = std::max(1u, std::min(slots, n_chunks));
    read_threads = std::max(1u, read_threads);

    const distributed::MeshCoordinate coord(0, 0);
    const distributed::MeshCoordinateRangeSet here(distributed::MeshCoordinateRange(coord, coord));
    IDevice* dev = mesh->get_device(0, 0);
    auto& cq = mesh->mesh_command_queue();

    std::vector<uint8_t*> slot_ptr(slots);
    std::vector<std::shared_ptr<experimental::PinnedMemory>> slot_pin(slots);
    for (uint32_t s = 0; s < slots; s++) {
        slot_ptr[s] = map_hugepage_1g(s);
        HostBuffer hb(ttsl::Span<uint8_t>(slot_ptr[s], GB), MemoryPin(std::shared_ptr<void>()));
        slot_pin[s] = experimental::PinnedMemory::Create(*mesh, here, hb, true);
    }
    double t_setup = since(t0);

    auto grid = dev->compute_with_storage_grid_size();
    num_cores = std::min<uint32_t>(num_cores, grid.x * grid.y);
    std::vector<CoreCoord> cores;
    std::set<CoreRange> ranges;
    for (uint32_t i = 0; i < num_cores; i++) {
        cores.push_back({i % grid.x, i / grid.x});
        ranges.insert(CoreRange(cores.back(), cores.back()));
    }
    const CoreRangeSet core_set(ranges);

    int fd = open(path.c_str(), O_RDONLY);
    TT_FATAL(fd >= 0, "cannot open {}", path);
    double t_file = 0, t_wait = 0;
    std::vector<std::optional<distributed::MeshEvent>> done(slots);
    {
        nb::gil_scoped_release nogil;
        for (uint32_t c = 0; c < n_chunks; c++) {
            const uint32_t s = c % slots;
            if (done[s]) {
                auto tw = Clock::now();
                distributed::EventSynchronize(*done[s]);  // the slot's previous chunk has been pulled
                t_wait += since(tw);
            }
            const uint64_t off = uint64_t(c) * GB, len = std::min<uint64_t>(GB, total - off);
            auto tf = Clock::now();
            {
                // page-cache copies are CPU bound: split the chunk across threads
                std::vector<std::thread> th;
                std::atomic<bool> ok{true};
                for (uint32_t k = 0; k < read_threads; k++) {
                    th.emplace_back([&, k] {
                        uint64_t lo = len * k / read_threads, hi = len * (k + 1) / read_threads;
                        for (uint64_t got = lo; got < hi;) {
                            ssize_t r = pread(fd, slot_ptr[s] + got, hi - got, off + got);
                            if (r <= 0) {
                                ok = false;
                                return;
                            }
                            got += r;
                        }
                    });
                }
                for (auto& t : th) {
                    t.join();
                }
                TT_FATAL(ok, "short read of {} in chunk {}", path, c);
            }
            t_file += since(tf);

            auto na = slot_pin[s]->get_noc_addr(dev->id());
            TT_FATAL(na.has_value() && na->device_id == dev->id(), "pinned slot not NOC-mapped for this chip");
            Program prog = CreateProgram();
            CreateCircularBuffer(
                prog,
                core_set,
                CircularBufferConfig(2 * page_size, {{0, tt::DataFormat::RawUInt32}}).set_page_size(0, page_size));
            std::vector<uint32_t> ct = {na->pcie_xy_enc, page_size, BURST};
            TensorAccessorArgs(*buf->get_reference_buffer()).append_to(ct);
            auto k = CreateKernel(
                prog,
                KERNEL,
                core_set,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ct});
            const uint32_t pages = len / page_size, page_off = c * chunk_pages;
            for (uint32_t i = 0; i < num_cores; i++) {
                uint32_t mine = i < pages ? (pages - i + num_cores - 1) / num_cores : 0;
                SetRuntimeArgs(
                    prog,
                    k,
                    cores[i],
                    {uint32_t(na->addr & 0xffffffffu),
                     uint32_t(na->addr >> 32),
                     uint32_t(buf->address()),
                     page_off + i,
                     num_cores,
                     mine,
                     0u,
                     page_off});
            }
            distributed::MeshWorkload wl;
            wl.add_program(distributed::MeshCoordinateRange(coord, coord), std::move(prog));
            distributed::EnqueueMeshWorkload(cq, wl, false);
            done[s] = cq.enqueue_record_event_to_host();
        }
        distributed::Finish(cq);
    }
    close(fd);
    double t_total = since(t0);
    slot_pin.clear();
    for (auto* p : slot_ptr) {
        munmap(p, GB);
    }
    return {
        {"bytes", double(total)},
        {"chunks", double(n_chunks)},
        {"setup_s", t_setup},
        {"file_read_s", t_file},
        {"wait_s", t_wait},
        {"total_s", t_total},
        {"gb_per_s", total / t_total / 1e9}};
}

}  // namespace

void py_module(nb::module_& mod) {
    mod.def(
        "read_region",
        &read_region,
        nb::arg("mesh_device"),
        nb::arg("address"),
        nb::arg("bytes_per_bank"),
        nb::arg("page_size"),
        nb::arg("path"),
        "Dump DRAM [address, address + bytes_per_bank) of every bank to `path`, in interleaved page order.");
    mod.def(
        "pull_region",
        &pull_region,
        nb::arg("mesh_device"),
        nb::arg("address"),
        nb::arg("bytes_per_bank"),
        nb::arg("page_size"),
        nb::arg("path"),
        nb::arg("slots") = 2,
        nb::arg("num_cores") = 8,
        nb::arg("read_threads") = 16,
        "Stream an image written by read_region back into DRAM: Tensix cores pull 1 GB pinned hugepage chunks over "
        "PCIe while the host reads the next chunk from `path`.");
}

}  // namespace ttnn::dram_image
