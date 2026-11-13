// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Helpers of MeshWorkloadSpecFactoryAdapter for factories with per-device run args (`uses_per_device_run_args`):
// one program per range, base run args for every device, and per-device values of a few named kernel run args.

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/distributed/types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::device_operation::per_device_run_args {

// A named kernel run arg on one node (nullopt: a common run arg).
struct Key {
    std::string kernel;
    std::string arg;
    std::optional<tt::tt_metal::CoreCoord> node;
    bool operator==(const Key&) const = default;
};
struct KeyHash {
    size_t operator()(const Key& key) const noexcept;
};
using KeySet = std::unordered_set<Key, KeyHash>;

// The named run args each kernel of a program declares, and the nodes it runs on: what per-device values may set.
struct KernelArgs {
    std::unordered_set<std::string> runtime;
    std::unordered_set<std::string> common;
    std::unordered_map<std::string, uint32_t> runtime_arrays;  // name -> length
    std::unordered_map<std::string, uint32_t> common_arrays;
    tt::tt_metal::CoreRangeSet nodes;
};
using ArgSchema = std::unordered_map<std::string, KernelArgs>;  // by kernel name
ArgSchema arg_schema_of(const tt::tt_metal::experimental::ProgramSpec& spec);

// The per-device state of one range a factory returned. Shared by the range's cached programs: one program when the
// range runs merged (one program, per-device run args), one per coordinate when it runs split (the fallback).
struct RangeState {
    ttnn::MeshCoordinateRange factory_range;
    bool merged = false;
    // The cached programs of the range: {factory_range} when merged, its coordinates in range order when split.
    std::vector<ttnn::MeshCoordinateRange> program_ranges;
    // The keys that carried a per-device value after the last call. The next call's base must list them, so its
    // UpdateProgramRunArgs drops every one of those values (see check_base_covers).
    KeySet prev_keys;
    // The program's named run args, to check per-device values before anything is set.
    ArgSchema schema;
};

// Checks the per-device entries of `range` against the program's `schema` (devices unique and inside it; named
// kernel run args only, each declared by its kernel with the kind and length given, on a node the kernel runs on)
// and returns the keys they set. Metal accepts every value that passes, so a call cannot fail part-way on them.
KeySet validate_and_collect_keys(
    const ttnn::MeshCoordinateRange& range, std::span<const DeviceRunArgs> per_device, const ArgSchema& schema);

// Fails unless `base` lists every key of `prev` and of `current`. A program-level update drops the per-device values
// of exactly the args (and nodes) it lists, so this makes each call fully determine the per-device state: a device,
// core or arg the call no longer sets per device falls back to the base value instead of keeping a stale one.
void check_base_covers(
    const tt::tt_metal::experimental::ProgramRunArgs& base,
    const KeySet& prev,
    const KeySet& current,
    const ttnn::MeshCoordinateRange& range);

// `base` with the named kernel run args of `values` in place of base's: node by node for per-node args, whole arrays.
tt::tt_metal::experimental::ProgramRunArgs overlay(
    const tt::tt_metal::experimental::ProgramRunArgs& base, const tt::tt_metal::experimental::ProgramRunArgs& values);

// The values of `device` among `per_device`, or null.
const tt::tt_metal::experimental::ProgramRunArgs* values_of(
    std::span<const DeviceRunArgs> per_device, const ttnn::MeshCoordinate& device);

// Test-only: while an instance lives on this thread, factories with per-device run args build every range split
// (one program per coordinate, each with its own run args), as when per-device run args are not supported. The
// program cache of `mesh_device` is cleared on construction and destruction, since merged and split entries of an op
// share a hash. Nests.
class ScopedForceSplit {
public:
    explicit ScopedForceSplit(tt::tt_metal::distributed::MeshDevice& mesh_device);
    ~ScopedForceSplit();
    ScopedForceSplit(const ScopedForceSplit&) = delete;
    ScopedForceSplit& operator=(const ScopedForceSplit&) = delete;

private:
    tt::tt_metal::distributed::MeshDevice& mesh_device_;
    bool previous_;
};
bool force_split();

}  // namespace ttnn::device_operation::per_device_run_args
