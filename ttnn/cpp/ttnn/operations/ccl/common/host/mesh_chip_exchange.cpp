// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/common/host/mesh_chip_exchange.hpp"

#include <algorithm>
#include <map>
#include <exception>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::ccl::common {

namespace {

using tt::tt_metal::distributed::multihost::DistributedContext;

// Each chip's record is preceded by the number of ranks that own it, so the sum can be checked.
constexpr uint32_t OWNER_WORDS = 1;

std::map<std::pair<int, std::string>, ChipRecords>& cache() {
    static std::map<std::pair<int, std::string>, ChipRecords> records;
    return records;
}

std::mutex& cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

// reshape() changes a mesh's coordinate-to-chip mapping without changing its id, so the cache key carries the shape,
// the fabric node behind every coordinate, and the record width alongside the caller's configuration.
std::string cache_signature(const ttnn::MeshDevice& mesh, uint32_t words_per_chip, const std::string& config_key) {
    if (config_key.empty()) {
        return {};
    }
    const auto& shape = mesh.shape();
    std::ostringstream signature;
    signature << config_key << "|shape=" << shape << "|words=" << words_per_chip << "|nodes=";
    for (const auto& coord : ttnn::MeshCoordinateRange(shape)) {
        signature << mesh.get_fabric_node_id(coord) << ',';
    }
    return signature.str();
}

}  // namespace

MeshContext mesh_distributed_context(const ttnn::MeshDevice& mesh) {
    return mesh.coowner_context();
}

ChipRecords exchange_chip_records(
    const ttnn::MeshDevice& mesh,
    const MeshContext& context,
    uint32_t words_per_chip,
    const std::function<std::vector<uint32_t>(const ttnn::MeshCoordinate&)>& local_record,
    const std::string& config_key) {
    const auto cache_key =
        std::make_pair(static_cast<int>(mesh.id()), cache_signature(mesh, words_per_chip, config_key));
    if (!config_key.empty()) {
        const std::lock_guard lock(cache_mutex());
        const auto hit = cache().find(cache_key);
        if (hit != cache().end()) {
            return hit->second;
        }
    }

    const auto& shape = mesh.shape();
    const std::size_t stride = OWNER_WORDS + words_per_chip;
    // The last word counts the ranks whose `local_record` failed, so a failure on one rank reaches all of them
    // through the collective instead of leaving the others waiting in it.
    const std::size_t failure_word = shape.mesh_size() * stride;
    std::vector<uint32_t> mine(failure_word + 1, 0);
    std::optional<std::string> local_failure;
    try {
        for (const auto& coord : ttnn::MeshCoordinateRange(shape)) {
            if (!mesh.is_local(coord)) {
                continue;
            }
            const auto record = local_record(coord);
            TT_FATAL(
                record.size() == words_per_chip,
                "mesh chip exchange: the record for {} has {} words, expected {}",
                coord,
                record.size(),
                words_per_chip);
            const std::size_t base = coord.to_linear_index(shape) * stride;
            mine[base] = 1;
            std::copy(record.begin(), record.end(), mine.begin() + base + OWNER_WORDS);
        }
    } catch (const std::exception& error) {
        local_failure = error.what();
        mine[failure_word] = 1;
    }

    std::vector<uint32_t> all = mine;
    if (context.has_value()) {
        context->get().all_reduce(
            std::span<uint32_t>(mine), std::span<uint32_t>(all), tt::tt_metal::distributed::multihost::ReduceOp::SUM);
    }

    if (all[failure_word] != 0) {
        TT_FATAL(
            false,
            "mesh chip exchange: local_record failed on {} rank(s); {}",
            all[failure_word],
            local_failure.has_value() ? "this rank's error: " + *local_failure
                                      : std::string("this rank succeeded, see the failing rank's log"));
    }

    ChipRecords records(shape.mesh_size());
    for (const auto& coord : ttnn::MeshCoordinateRange(shape)) {
        const std::size_t index = coord.to_linear_index(shape);
        const std::size_t base = index * stride;
        TT_FATAL(
            all[base] == 1, "mesh chip exchange: {} ranks own coordinate {}, expected exactly one", all[base], coord);
        records[index].assign(all.begin() + base + OWNER_WORDS, all.begin() + base + stride);
    }

    if (!config_key.empty()) {
        const std::lock_guard lock(cache_mutex());
        cache().emplace(cache_key, records);
    }
    return records;
}

void RecordWriter::put(uint32_t word) {
    TT_FATAL(written_ < words_.size(), "mesh chip exchange: record is full at {} words", words_.size());
    words_[written_++] = word;
}

void RecordWriter::put_core(const tt::tt_metal::CoreCoord& core) {
    put(static_cast<uint32_t>(core.x));
    put(static_cast<uint32_t>(core.y));
}

void RecordWriter::put_coord(const ttnn::MeshCoordinate& coord) {
    for (const uint32_t value : coord.coords()) {
        put(value);
    }
}

std::vector<uint32_t> RecordWriter::finish() { return std::move(words_); }

uint32_t RecordReader::get() {
    TT_FATAL(read_ < words_.size(), "mesh chip exchange: record has only {} words", words_.size());
    return words_[read_++];
}

tt::tt_metal::CoreCoord RecordReader::get_core() {
    const uint32_t x = get();
    const uint32_t y = get();
    return {x, y};
}

ttnn::MeshCoordinate RecordReader::get_coord(std::size_t dims) {
    std::vector<uint32_t> values(dims);
    std::generate(values.begin(), values.end(), [this] { return get(); });
    return ttnn::MeshCoordinate(values);
}

}  // namespace ttnn::operations::ccl::common
