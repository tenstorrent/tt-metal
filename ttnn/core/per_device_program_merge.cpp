// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/per_device_program_merge.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn {

namespace {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::ProgramDescriptor;

std::vector<CoreCoord> cores_of(const CoreRangeSet& set) {
    std::vector<CoreCoord> cores;
    cores.reserve(set.num_cores());
    for (const CoreRange& range : set.ranges()) {
        for (auto y = range.start_coord.y; y <= range.end_coord.y; ++y) {
            for (auto x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                cores.emplace_back(x, y);
            }
        }
    }
    return cores;
}

CoreRangeSet union_of(const CoreRangeSet& a, const CoreRangeSet& b) { return a.merge(b); }

// Everything but runtime-arg values, kernel / CB core sets, buffer bindings and semaphores: the parts that must be
// identical on every device. std::hash<ProgramDescriptor> excludes runtime-arg values and hashes their counts.
size_t structure_hash(const ProgramDescriptor& descriptor) {
    ProgramDescriptor structure = descriptor;
    for (auto& kernel : structure.kernels) {
        kernel.core_ranges = CoreRangeSet{};
        kernel.runtime_args.clear();
        kernel.buffer_bindings.clear();
    }
    for (auto& cb : structure.cbs) {
        cb.core_ranges = CoreRangeSet{};
    }
    structure.semaphores.clear();
    structure.custom_program_hash.reset();  // std::hash would return it instead of hashing the structure
    return std::hash<ProgramDescriptor>{}(structure);
}

bool same_cb_resources(const tt::tt_metal::CBDescriptor& a, const tt::tt_metal::CBDescriptor& b) {
    return a.total_size == b.total_size && a.buffer == b.buffer && a.tensor == b.tensor &&
           a.address_offset == b.address_offset && a.global_circular_buffer == b.global_circular_buffer &&
           a.format_descriptors.size() == b.format_descriptors.size() &&
           a.remote_format_descriptors.size() == b.remote_format_descriptors.size();
}

// The per-device fields of one slot: each run of words where any device differs from the first, or that the op
// declared per-device (`declared`, exact bounds). Words between runs keep the program's own args, so cache-hit updates
// to them reach every device.
std::vector<PerDeviceRuntimeArgsField> fields_of(
    uint32_t kernel_index,
    std::optional<CoreCoord> core,
    const std::vector<const std::vector<uint32_t>*>& per_device_args,
    const std::vector<uint32_t>& declared) {
    const auto& first = *per_device_args.front();
    // Declared words form their own fields with exactly the declared bounds (the op sets them by those bounds); other
    // per-device words form runs between them.
    auto segment = [&](uint32_t i) { return i < declared.size() ? declared[i] : 0u; };
    std::vector<bool> per_device(first.size(), false);
    for (uint32_t i = 0; i < first.size(); ++i) {
        per_device[i] = segment(i) != 0;
        for (const auto* args : per_device_args) {
            per_device[i] = per_device[i] || (*args)[i] != first[i];
        }
    }
    std::vector<PerDeviceRuntimeArgsField> fields;
    for (uint32_t i = 0; i < first.size();) {
        if (!per_device[i]) {
            ++i;
            continue;
        }
        uint32_t end = i;
        while (end < first.size() && per_device[end] && segment(end) == segment(i)) {
            ++end;
        }
        PerDeviceRuntimeArgsField field{
            .kernel_index = kernel_index, .core = core, .first_word = i, .num_words = end - i};
        field.values_in_range_order.reserve(per_device_args.size() * field.num_words);
        for (const auto* args : per_device_args) {
            field.values_in_range_order.insert(
                field.values_in_range_order.end(), args->begin() + i, args->begin() + end);
        }
        fields.push_back(std::move(field));
        i = end;
    }
    return fields;
}

// Two sets that are disjoint on every device must stay disjoint after the union (two kernels on one processor, two
// CBs on one index); sets that already share cores on some device may share more.
bool union_keeps_disjoint(
    const std::vector<const CoreRangeSet*>& a,
    const std::vector<const CoreRangeSet*>& b,
    const CoreRangeSet& merged_a,
    const CoreRangeSet& merged_b) {
    for (size_t d = 0; d < a.size(); ++d) {
        if (a[d]->intersects(*b[d])) {
            return true;
        }
    }
    return !merged_a.intersects(merged_b);
}

bool share_buffer_index(const tt::tt_metal::CBDescriptor& a, const tt::tt_metal::CBDescriptor& b) {
    // local and remote indices share one index space on a core
    auto indices = [](const tt::tt_metal::CBDescriptor& cb) {
        std::set<uint8_t> out;
        for (const auto& f : cb.format_descriptors) {
            out.insert(f.buffer_index);
        }
        for (const auto& f : cb.remote_format_descriptors) {
            out.insert(f.buffer_index);
        }
        return out;
    };
    const auto ia = indices(a);
    for (const uint8_t index : indices(b)) {
        if (ia.contains(index)) {
            return true;
        }
    }
    return false;
}

// Per word of one slot: 0, or 1 + the index of the op-declared range that covers it.
std::vector<uint32_t> forced_words(
    const std::vector<PerDeviceRuntimeArgsWords>& declared, uint32_t kernel_index, std::optional<CoreCoord> core) {
    std::vector<uint32_t> segments;
    for (size_t r = 0; r < declared.size(); ++r) {
        const auto& words = declared[r];
        if (words.kernel_index != kernel_index || words.core != core) {
            continue;
        }
        if (segments.size() < words.first_word + words.num_words) {
            segments.resize(words.first_word + words.num_words, 0);
        }
        std::fill(
            segments.begin() + words.first_word,
            segments.begin() + words.first_word + words.num_words,
            static_cast<uint32_t>(r + 1));
    }
    return segments;
}

}  // namespace

std::optional<MergedProgramDescriptor> merge_per_device_program_descriptors(
    const std::vector<ProgramDescriptor>& descriptors, const ProgramMergeOptions& options) {
    TT_FATAL(!descriptors.empty(), "merge_per_device_program_descriptors: no descriptors");
    MergedProgramDescriptor merged{.descriptor = descriptors.front(), .per_device_runtime_args = {}};
    if (descriptors.size() == 1) {
        return merged;
    }
    auto refuse = [](const char* why) -> std::optional<MergedProgramDescriptor> {
        log_debug(tt::LogOp, "Per-device programs not merged: {}", why);
        return std::nullopt;
    };

    // ---- structure ----
    const auto& first = descriptors.front();
    const size_t first_hash = structure_hash(first);
    for (const auto& descriptor : descriptors) {
        if (descriptor.reload_table.has_value()) {
            return refuse("a descriptor reloads binaries");
        }
        for (const auto& kernel : descriptor.kernels) {
            if (!kernel.blaze_named_args.empty()) {
                return refuse("a kernel has named args (only positional runtime args become per-device)");
            }
        }
        if (descriptor.kernels.size() != first.kernels.size() || descriptor.cbs.size() != first.cbs.size() ||
            structure_hash(descriptor) != first_hash) {
            return refuse("kernels, compile-time args, configs or circular buffers differ");
        }
        for (size_t k = 0; k < first.kernels.size(); ++k) {
            const auto& a = descriptor.kernels[k].common_buffer_bindings;
            const auto& b = first.kernels[k].common_buffer_bindings;
            if (a.size() != b.size()) {
                return refuse("common buffer bindings differ");
            }
            for (size_t i = 0; i < a.size(); ++i) {
                if (a[i].arg_idx != b[i].arg_idx || a[i].buffer != b[i].buffer) {
                    return refuse("common buffer bindings differ");
                }
            }
            if (descriptor.kernels[k].common_runtime_args.size() != first.kernels[k].common_runtime_args.size()) {
                return refuse("common runtime-arg counts differ");
            }
            if (!options.idle_runtime_args && descriptor.kernels[k].core_ranges != first.kernels[k].core_ranges) {
                return refuse("kernel core sets differ and the op gives no idle args");
            }
        }
        for (size_t c = 0; c < first.cbs.size(); ++c) {
            if (!same_cb_resources(descriptor.cbs[c], first.cbs[c])) {
                return refuse("circular buffer resources differ");
            }
        }
    }

    // ---- circular buffers: union of cores ----
    for (size_t c = 0; c < first.cbs.size(); ++c) {
        for (const auto& descriptor : descriptors) {
            merged.descriptor.cbs[c].core_ranges =
                union_of(merged.descriptor.cbs[c].core_ranges, descriptor.cbs[c].core_ranges);
        }
    }

    for (size_t i = 0; i < first.cbs.size(); ++i) {
        for (size_t j = i + 1; j < first.cbs.size(); ++j) {
            if (!share_buffer_index(first.cbs[i], first.cbs[j])) {
                continue;
            }
            std::vector<const CoreRangeSet*> a, b;
            for (const auto& descriptor : descriptors) {
                a.push_back(&descriptor.cbs[i].core_ranges);
                b.push_back(&descriptor.cbs[j].core_ranges);
            }
            if (!union_keeps_disjoint(
                    a, b, merged.descriptor.cbs[i].core_ranges, merged.descriptor.cbs[j].core_ranges)) {
                return refuse("the union of cores puts two circular buffers of one index on a core");
            }
        }
    }

    // ---- semaphores: one per (id, core type, initial value), on the union of its cores ----
    {
        std::map<std::tuple<uint32_t, int, uint32_t>, CoreRangeSet> groups;
        std::vector<std::tuple<uint32_t, int, uint32_t>> order;
        for (const auto& descriptor : descriptors) {
            for (const auto& sem : descriptor.semaphores) {
                const auto key = std::make_tuple(sem.id, static_cast<int>(sem.core_type), sem.initial_value);
                auto [it, inserted] = groups.emplace(key, sem.core_ranges);
                if (inserted) {
                    order.push_back(key);
                } else {
                    it->second = union_of(it->second, sem.core_ranges);
                }
            }
        }
        // the same id with two initial values on one core cannot be one program
        std::set<std::tuple<uint32_t, int, size_t, size_t>> taken;
        merged.descriptor.semaphores.clear();
        for (const auto& key : order) {
            const auto& [id, core_type, initial_value] = key;
            for (const auto& core : cores_of(groups.at(key))) {
                if (!taken.emplace(id, core_type, core.x, core.y).second) {
                    return refuse("a semaphore id has different initial values on one core");
                }
            }
            merged.descriptor.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
                .id = id,
                .core_type = static_cast<tt::CoreType>(core_type),
                .core_ranges = groups.at(key),
                .initial_value = initial_value});
        }
    }

    // ---- kernels ----
    for (size_t k = 0; k < first.kernels.size(); ++k) {
        for (const auto& descriptor : descriptors) {
            merged.descriptor.kernels[k].core_ranges =
                union_of(merged.descriptor.kernels[k].core_ranges, descriptor.kernels[k].core_ranges);
        }
    }
    for (size_t i = 0; i < first.kernels.size(); ++i) {
        for (size_t j = i + 1; j < first.kernels.size(); ++j) {
            std::vector<const CoreRangeSet*> a, b;
            for (const auto& descriptor : descriptors) {
                a.push_back(&descriptor.kernels[i].core_ranges);
                b.push_back(&descriptor.kernels[j].core_ranges);
            }
            if (!union_keeps_disjoint(
                    a, b, merged.descriptor.kernels[i].core_ranges, merged.descriptor.kernels[j].core_ranges)) {
                return refuse("the union of cores puts two kernels that never share a core on one core");
            }
        }
    }
    for (size_t k = 0; k < first.kernels.size(); ++k) {
        auto& kernel = merged.descriptor.kernels[k];

        // buffer bindings: union; one buffer per (core, arg), bound on every device that runs the kernel there (a
        // cache hit patches the word on every device of the merged program)
        std::map<std::tuple<size_t, size_t, uint32_t>, tt::tt_metal::Buffer*> bindings;
        std::vector<std::set<std::tuple<size_t, size_t, uint32_t>>> bound(descriptors.size());
        for (size_t d = 0; d < descriptors.size(); ++d) {
            for (const auto& binding : descriptors[d].kernels[k].buffer_bindings) {
                const auto key = std::make_tuple(binding.core.x, binding.core.y, binding.arg_idx);
                bound[d].insert(key);
                auto [it, inserted] = bindings.emplace(key, binding.buffer);
                if (!inserted && it->second != binding.buffer) {
                    return refuse("a buffer binding differs per device");
                }
            }
        }
        for (const auto& [key, buffer] : bindings) {
            const CoreCoord core(std::get<0>(key), std::get<1>(key));
            for (size_t d = 0; d < descriptors.size(); ++d) {
                if (!descriptors[d].kernels[k].core_ranges.contains(core)) {
                    // the binding would overwrite a word of this device's idle args
                    return refuse("a buffer binding is on a core where a device's kernel idles");
                }
                if (!bound[d].contains(key)) {
                    return refuse("a buffer binding is missing on a device that runs the kernel on its core");
                }
            }
        }
        kernel.buffer_bindings.clear();
        for (const auto& [key, buffer] : bindings) {
            kernel.buffer_bindings.push_back(tt::tt_metal::BufferBinding{
                .core = CoreCoord(std::get<0>(key), std::get<1>(key)), .arg_idx = std::get<2>(key), .buffer = buffer});
        }

        // unique runtime args, per core of the union
        std::vector<std::map<std::pair<size_t, size_t>, const std::vector<uint32_t>*>> args_by_core(descriptors.size());
        for (size_t d = 0; d < descriptors.size(); ++d) {
            for (const auto& [core, args] : descriptors[d].kernels[k].runtime_args) {
                args_by_core[d][{core.x, core.y}] = &args;
            }
        }
        kernel.runtime_args.clear();
        std::vector<std::vector<uint32_t>> owned(descriptors.size());
        const std::vector<uint32_t> no_args;
        for (const auto& core : cores_of(kernel.core_ranges)) {
            std::vector<const std::vector<uint32_t>*> per_device(descriptors.size());
            bool any_args = false;
            size_t length = 0;
            for (size_t d = 0; d < descriptors.size(); ++d) {
                const auto it = args_by_core[d].find({core.x, core.y});
                if (it != args_by_core[d].end()) {
                    per_device[d] = it->second;
                } else if (descriptors[d].kernels[k].core_ranges.contains(core)) {
                    per_device[d] = &no_args;
                } else {
                    auto idle = options.idle_runtime_args ? options.idle_runtime_args(k, core) : std::nullopt;
                    if (!idle.has_value()) {
                        return refuse("a kernel cannot idle on a core another device runs it on");
                    }
                    owned[d] = std::move(*idle);
                    per_device[d] = &owned[d];
                }
                any_args = any_args || !per_device[d]->empty();
                length = std::max(length, per_device[d]->size());
            }
            if (!any_args) {
                continue;
            }
            std::vector<std::vector<uint32_t>> padded;
            if (std::any_of(per_device.begin(), per_device.end(), [&](const auto* a) { return a->size() != length; })) {
                if (!options.pad_runtime_args) {
                    return refuse("runtime-arg counts differ per device");
                }
                padded.reserve(per_device.size());
                for (auto*& args : per_device) {
                    auto& p = padded.emplace_back(*args);
                    p.resize(length, 0);
                    args = &p;
                }
            }
            kernel.runtime_args.emplace_back(core, *per_device.front());
            for (auto& field : fields_of(
                     static_cast<uint32_t>(k),
                     core,
                     per_device,
                     forced_words(options.per_device_words, static_cast<uint32_t>(k), core))) {
                for (const auto& [key, buffer] : bindings) {
                    if (std::get<0>(key) == core.x && std::get<1>(key) == core.y &&
                        std::get<2>(key) >= field.first_word && std::get<2>(key) < field.first_word + field.num_words) {
                        return refuse("a per-device runtime-arg word is a buffer binding");
                    }
                }
                merged.per_device_runtime_args.push_back(std::move(field));
            }
            for (auto& o : owned) {
                o.clear();
            }
        }

        // common runtime args
        std::vector<const std::vector<uint32_t>*> per_device_common;
        per_device_common.reserve(descriptors.size());
        for (const auto& descriptor : descriptors) {
            per_device_common.push_back(&descriptor.kernels[k].common_runtime_args);
        }
        if (!first.kernels[k].common_runtime_args.empty()) {
            for (auto& field : fields_of(
                     static_cast<uint32_t>(k),
                     std::nullopt,
                     per_device_common,
                     forced_words(options.per_device_words, static_cast<uint32_t>(k), std::nullopt))) {
                for (const auto& binding : first.kernels[k].common_buffer_bindings) {
                    if (binding.arg_idx >= field.first_word && binding.arg_idx < field.first_word + field.num_words) {
                        return refuse("a per-device common runtime-arg word is a buffer binding");
                    }
                }
                merged.per_device_runtime_args.push_back(std::move(field));
            }
        }
    }
    return merged;
}

void apply_per_device_runtime_args(
    tt::tt_metal::distributed::MeshWorkload& workload,
    const tt::tt_metal::distributed::MeshCoordinateRange& range,
    const std::vector<PerDeviceRuntimeArgsField>& per_device_runtime_args) {
    for (const auto& field : per_device_runtime_args) {
        tt::tt_metal::distributed::SetRuntimeArgsPerDevice(
            workload,
            range,
            static_cast<tt::tt_metal::KernelHandle>(field.kernel_index),
            field.core,
            field.first_word,
            field.num_words,
            field.values_in_range_order);
    }
}

}  // namespace ttnn
