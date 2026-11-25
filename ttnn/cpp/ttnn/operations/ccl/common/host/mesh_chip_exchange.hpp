// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <tt_stl/optional_reference.hpp>
#include <span>
#include <string>
#include <vector>

#include <tt-metalium/mesh/multihost/distributed_context.hpp>

#include <tt-metalium/vocabulary/core_coord.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::ccl::common {

// One record of uint32 words per mesh coordinate, indexed by the coordinate's linear index in the mesh shape.
using ChipRecords = std::vector<std::vector<uint32_t>>;

// A co-owner context borrowed from a mesh; empty when the mesh has no co-owners.
using MeshContext = ttsl::optional_reference<const tt::tt_metal::distributed::multihost::DistributedContext>;

// Computes a record for every coordinate of `mesh`, including those on other hosts.
//
// Mesh workloads are SPMD: every rank builds programs for every global coordinate, but a rank can only read the
// physical layout (harvesting, translated NOC coordinates) of the chips it owns. `local_record` is called on the
// owning rank only, with `words_per_chip` words each; the ranks then sum their records over `context`, so every
// rank returns the same full set. A mesh whose coordinates are all local (or an empty `context`) makes no
// collective call.
//
// Every rank of the mesh must call this with the same arguments in the same order, like any collective.
//
// If `local_record` throws (or returns the wrong number of words) on any rank, every rank throws after the collective,
// so a failure on one host shows up as an error on all of them rather than a hang on the others. Only failures
// inside `local_record` are covered: an exception a rank raises before calling this still leaves the others waiting.
//
// A non-empty `config_key` on a mesh shared between hosts (non-empty `context`) caches the result per (mesh, mesh
// shape and chip layout, record width, config_key): a later call with the same key returns the cached records without
// calling `local_record` or the collective. A single-host mesh is never cached and recomputes on every call.
// `config_key` must name every other input the records depend on.
ChipRecords exchange_chip_records(
    const ttnn::MeshDevice& mesh,
    MeshContext context,
    uint32_t words_per_chip,
    const std::function<std::vector<uint32_t>(const ttnn::MeshCoordinate&)>& local_record,
    const std::string& config_key = {});

// The context spanning the ranks that co-own `mesh`, borrowed from the mesh. Empty when this rank drives every device
// of the mesh.
MeshContext mesh_distributed_context(const ttnn::MeshDevice& mesh);

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
