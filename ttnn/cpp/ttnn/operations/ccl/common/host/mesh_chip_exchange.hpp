// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <tt-metalium/distributed_context.hpp>

#include <tt-metalium/core_coord.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::ccl::common {

// One record of uint32 words per mesh coordinate, indexed by the coordinate's linear index in the mesh shape.
using ChipRecords = std::vector<std::vector<uint32_t>>;

// Computes a record for every coordinate of `mesh`, including those on other hosts.
//
// Mesh workloads are SPMD: every rank builds programs for every global coordinate, but a rank can only read the
// physical layout (harvesting, translated NOC coordinates) of the chips it owns. `local_record` is called on the
// owning rank only, with `words_per_chip` words each; the ranks then sum their records over `context`, so every
// rank returns the same full set. A mesh whose coordinates are all local (or a null/one-rank `context`) makes no
// collective call.
//
// Every rank of the mesh must call this with the same arguments in the same order, like any collective.
//
// A non-empty `config_key` caches the result per (mesh, config_key): a later call with the same pair returns the
// cached records without calling `local_record` or the collective. `config_key` must name every input the records
// depend on.
ChipRecords exchange_chip_records(
    const ttnn::MeshDevice& mesh,
    const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>& context,
    uint32_t words_per_chip,
    const std::function<std::vector<uint32_t>(const ttnn::MeshCoordinate&)>& local_record,
    const std::string& config_key = {});

// The context the mesh's ranks share, from the mesh's view. Null on a single-host mesh.
std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> mesh_distributed_context(
    const ttnn::MeshDevice& mesh);

// Appends words to a record, for records that mix coordinates, indices and flags.
class RecordWriter {
public:
    explicit RecordWriter(uint32_t words_per_chip) : words_(words_per_chip, 0) {}
    void put(uint32_t word);
    void put_core(const tt::tt_metal::CoreCoord& core);
    void put_coord(const ttnn::MeshCoordinate& coord);
    // The record, zero-padded to the size given at construction.
    std::vector<uint32_t> finish();

private:
    std::vector<uint32_t> words_;
    uint32_t written_ = 0;
};

class RecordReader {
public:
    explicit RecordReader(std::span<const uint32_t> words) : words_(words) {}
    uint32_t get();
    tt::tt_metal::CoreCoord get_core();
    ttnn::MeshCoordinate get_coord(std::size_t dims);

private:
    std::span<const uint32_t> words_;
    std::size_t read_ = 0;
};

}  // namespace ttnn::operations::ccl::common
