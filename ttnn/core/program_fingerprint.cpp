// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/program_fingerprint.hpp"

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/program_fingerprint.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::program_fingerprint {

namespace {

std::vector<uint32_t> coords(const tt::tt_metal::distributed::MeshCoordinate& c) {
    return std::vector<uint32_t>(c.coords().begin(), c.coords().end());
}

// A tensor's buffer address (the same on every device of the mesh), or 0 for one with no device buffer.
uint64_t address(const ttnn::Tensor& t) {
    if (t.storage_type() != ttnn::StorageType::DEVICE || !t.is_allocated()) {
        return 0;
    }
    return t.mesh_buffer().address();
}

nlohmann::json addresses(const std::vector<const ttnn::Tensor*>& tensors) {
    nlohmann::json out = nlohmann::json::array();
    for (const auto* t : tensors) {
        out.push_back(address(*t));
    }
    return out;
}

// The device buffers alive at the call, sorted by address: what lets a reader tell an argument that is an address of a
// buffer (an internal staging buffer or semaphore of the operation's own) from a plain number.
nlohmann::json live_buffers(tt::tt_metal::distributed::MeshDevice& mesh_device) {
    std::vector<std::tuple<uint64_t, uint64_t, std::string>> found;
    for (const auto* b : mesh_device.allocator()->get_allocated_buffers()) {
        found.emplace_back(b->address(), b->size(), std::string(enchantum::to_string(b->buffer_type())));
    }
    std::sort(found.begin(), found.end());
    nlohmann::json out = nlohmann::json::array();
    for (const auto& [addr, size, type] : found) {
        out.push_back({{"address", addr}, {"size", size}, {"type", type}});
    }
    return out;
}

nlohmann::json statistics(tt::tt_metal::distributed::MeshDevice& mesh_device, tt::tt_metal::BufferType type) {
    const auto s = mesh_device.allocator()->get_statistics(type);
    return {{"allocated", s.total_allocated_bytes}, {"largest_free", s.largest_free_block_bytes}};
}

}  // namespace

void record(
    std::string_view operation,
    bool program_cache_hit,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    tt::tt_metal::distributed::MeshWorkload& workload,
    const std::vector<const ttnn::Tensor*>& args,
    const std::vector<const ttnn::Tensor*>& returns) {
    static std::mutex mutex;
    static uint64_t seq = 0;

    std::vector<std::pair<std::vector<uint32_t>, nlohmann::json>> programs;
    for (auto& [range, program] : workload.get_programs()) {
        nlohmann::json p;
        p["start"] = coords(range.start_coord());
        p["end"] = coords(range.end_coord());
        p["program"] = nlohmann::json::parse(tt::tt_metal::experimental::program_fingerprint_json(program));
        programs.emplace_back(coords(range.start_coord()), std::move(p));
    }
    std::sort(programs.begin(), programs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    nlohmann::json record;
    record["kind"] = "workload";
    record["op"] = std::string(operation);
    record["hit"] = program_cache_hit;
    const auto& shape = mesh_device.shape();
    std::vector<uint32_t> dims;
    for (size_t i = 0; i < shape.dims(); ++i) {
        dims.push_back(shape[i]);
    }
    record["mesh_shape"] = dims;
    record["mesh_id"] = mesh_device.id();
    record["tensors"] = {{"args", addresses(args)}, {"returns", addresses(returns)}};
    record["buffers"] = live_buffers(mesh_device);
    record["memory"] = {
        {"L1", statistics(mesh_device, tt::tt_metal::BufferType::L1)},
        {"DRAM", statistics(mesh_device, tt::tt_metal::BufferType::DRAM)}};
    nlohmann::json list = nlohmann::json::array();
    for (auto& [_, p] : programs) {
        list.push_back(std::move(p));
    }
    record["programs"] = std::move(list);

    std::lock_guard<std::mutex> lock(mutex);
    record["seq"] = seq++;
    const std::filesystem::path dir(std::getenv(ENV));
    std::filesystem::create_directories(dir);
    std::ofstream out(dir / ("fingerprint-" + std::to_string(::getpid()) + ".jsonl"), std::ios::app);
    out << record.dump() << '\n';
}

}  // namespace ttnn::program_fingerprint
