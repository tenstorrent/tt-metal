// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Merging the per-device programs of an op into one program with per-device runtime args.
//
// Ops whose devices need different runtime-arg values (a ring rank, a peer chip, a fabric channel) build one
// ProgramDescriptor per mesh coordinate. When those descriptors differ only in runtime-arg values (and, with an
// idle-args callback, in which cores a kernel runs on), they can run as one program on the whole range: the words
// that differ become per-device runtime args (tt::tt_metal::distributed::SetRuntimeArgsPerDevice), which saves the
// host the per-program dispatch work of every other device on every call.

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn {

// Words [first_word, first_word + num_words) of a merged program's kernel's runtime args on `core` (common runtime
// args when nullopt), one value block per coordinate of the range in its iteration order. Each run of words that
// differ between devices (or that the op declared per-device) is one field; words between runs stay the program's
// own.
struct PerDeviceRuntimeArgsField {
    uint32_t kernel_index = 0;
    std::optional<tt::tt_metal::CoreCoord> core;
    uint32_t first_word = 0;
    uint32_t num_words = 0;
    std::vector<uint32_t> values_in_range_order;
};

struct MergedProgramDescriptor {
    tt::tt_metal::ProgramDescriptor descriptor;
    std::vector<PerDeviceRuntimeArgsField> per_device_runtime_args;
};

// The runtime args of kernel `kernel_index` on `core` for a device whose own program does not run that kernel there
// (another device's does), such that the kernel does nothing; nullopt if the kernel cannot idle there.
using IdleRuntimeArgsFn =
    std::function<std::optional<std::vector<uint32_t>>(size_t kernel_index, const tt::tt_metal::CoreCoord& core)>;

// Words of a kernel's runtime args (common runtime args when `core` is nullopt) that are per-device even where every
// device has the same value when the program is built, because the op updates them per device on a cache hit.
struct PerDeviceRuntimeArgsWords {
    uint32_t kernel_index = 0;
    std::optional<tt::tt_metal::CoreCoord> core;
    uint32_t first_word = 0;
    uint32_t num_words = 0;
};

struct ProgramMergeOptions {
    // Kernels may run on different cores per device; the merged program runs each kernel on the union of its cores.
    IdleRuntimeArgsFn idle_runtime_args;
    // A kernel's runtime args on a core may differ in length per device: shorter ones are padded with zeros (the
    // kernel must read only what each device's args describe).
    bool pad_runtime_args = false;
    // Words that are per-device fields whatever their values (see PerDeviceRuntimeArgsWords).
    std::vector<PerDeviceRuntimeArgsWords> per_device_words;
};

// Merges `descriptors` (one per coordinate of a range, in the range's iteration order) into one program. Returns
// nullopt when they differ in anything besides runtime-arg values (and, with options.idle_runtime_args, kernel core
// sets): kernels, compile-time args, defines, configs, circular buffers, semaphores, buffer bindings, or a per-device
// word that a buffer binding patches. Also nullopt when a kernel has named (Blaze) args, or when the union of core sets
// would put two kernels, or two circular buffers of one index, on a core where no device has both.
std::optional<MergedProgramDescriptor> merge_per_device_program_descriptors(
    const std::vector<tt::tt_metal::ProgramDescriptor>& descriptors, const ProgramMergeOptions& options = {});

// Sets a merged program's per-device runtime args on the program `workload` holds at `range` (kernel handles are
// the descriptor's kernel indices).
void apply_per_device_runtime_args(
    tt::tt_metal::distributed::MeshWorkload& workload,
    const tt::tt_metal::distributed::MeshCoordinateRange& range,
    const std::vector<PerDeviceRuntimeArgsField>& per_device_runtime_args);

}  // namespace ttnn
