// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "chip_program.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>
#include <variant>

#include <fmt/format.h>
#include <tracy/Tracy.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::moe_block {

namespace chip_program_detail {

using tt::tt_metal::CBDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::DataMovementConfigDescriptor;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ProgramDescriptor;
using tt::tt_metal::SemaphoreDescriptor;

bool same_config(const KernelDescriptor::ConfigDescriptor& a, const KernelDescriptor::ConfigDescriptor& b) {
    if (a.index() != b.index()) {
        return false;
    }
    if (const auto* x = std::get_if<DataMovementConfigDescriptor>(&a)) {
        const auto& y = std::get<DataMovementConfigDescriptor>(b);
        return x->processor == y.processor && x->noc == y.noc && x->noc_mode == y.noc_mode;
    }
    if (const auto* x = std::get_if<ComputeConfigDescriptor>(&a)) {
        const auto& y = std::get<ComputeConfigDescriptor>(b);
        return x->math_fidelity == y.math_fidelity && x->fp32_dest_acc_en == y.fp32_dest_acc_en &&
               x->dst_full_sync_en == y.dst_full_sync_en && x->unpack_to_dest_mode == y.unpack_to_dest_mode &&
               x->bfp8_pack_precise == y.bfp8_pack_precise && x->math_approx_mode == y.math_approx_mode &&
               x->enable_trisc2_rvv == y.enable_trisc2_rvv;
    }
    return true;  // (reader / writer: no fields)
}

bool same_formats(const auto& a, const auto& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].buffer_index != b[i].buffer_index || a[i].data_format != b[i].data_format ||
            a[i].page_size != b[i].page_size || a[i].tile.has_value() != b[i].tile.has_value() ||
            a[i].face_geometry.has_value() != b[i].face_geometry.has_value()) {
            return false;
        }
    }
    return true;
}

bool same_cb(const CBDescriptor& a, const CBDescriptor& b) {
    return a.total_size == b.total_size && a.core_ranges == b.core_ranges &&
           same_formats(a.format_descriptors, b.format_descriptors) &&
           same_formats(a.remote_format_descriptors, b.remote_format_descriptors) && a.buffer == b.buffer &&
           a.tensor == b.tensor && a.address_offset == b.address_offset &&
           a.global_circular_buffer == b.global_circular_buffer;
}

using CoreKey = std::pair<uint32_t, uint32_t>;
CoreKey key(const CoreCoord& c) { return {static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y)}; }

// What differs between two chips' kernel i (nullptr: nothing). The runtime args are compared core by core (a program
// may list its cores in any order).
const char* kernel_difference(const KernelDescriptor& a, const KernelDescriptor& b) {
    if (a.kernel_source != b.kernel_source || a.source_type != b.source_type) {
        return "source";
    }
    if (a.core_ranges != b.core_ranges) {
        return "cores";
    }
    if (a.compile_time_args != b.compile_time_args || a.named_compile_time_args != b.named_compile_time_args) {
        return "compile-time args";
    }
    if (a.defines != b.defines) {
        return "defines";
    }
    if (a.opt_level != b.opt_level || a.compiler_include_paths != b.compiler_include_paths ||
        !same_config(a.config, b.config)) {
        return "config";
    }
    if (a.common_runtime_args != b.common_runtime_args) {
        return "common args";
    }
    auto args = [](const KernelDescriptor& k) {
        std::map<CoreKey, std::vector<uint32_t>> out;
        for (const auto& [core, v] : k.runtime_args) {
            out[key(core)].assign(v.begin(), v.end());
        }
        return out;
    };
    if (args(a) != args(b)) {
        return "runtime args";
    }
    auto bound = [](const KernelDescriptor& k) {
        std::set<std::tuple<CoreKey, uint32_t, const void*>> out;
        for (const auto& x : k.buffer_bindings) {
            out.emplace(key(x.core), x.arg_idx, x.buffer);
        }
        return out;
    };
    auto common_bound = [](const KernelDescriptor& k) {
        std::set<std::pair<uint32_t, const void*>> out;
        for (const auto& x : k.common_buffer_bindings) {
            out.emplace(x.arg_idx, x.buffer);
        }
        return out;
    };
    if (bound(a) != bound(b) || common_bound(a) != common_bound(b)) {
        return "buffer bindings";
    }
    return nullptr;
}

// The chips' programs (each chip's coordinate and program) as one program, or nullopt with `why` set.
std::optional<tt::tt_metal::ProgramDescriptor> merge_chip_programs(
    const std::vector<std::pair<tt::tt_metal::distributed::MeshCoordinate, tt::tt_metal::ProgramDescriptor>>& chips,
    std::string& why) {
    TT_FATAL(!chips.empty(), "merge_chip_programs: no chips");
    ProgramDescriptor one = chips.front().second;
    const auto& base = chips.front().second;
    for (size_t c = 1; c < chips.size(); ++c) {
        const auto& p = chips[c].second;
        if (p.kernels.size() != base.kernels.size() || p.cbs.size() != base.cbs.size()) {
            why = fmt::format(
                "chip {}: {} kernels / {} CBs, chip 0 {} / {}",
                c,
                p.kernels.size(),
                p.cbs.size(),
                base.kernels.size(),
                base.cbs.size());
            return std::nullopt;
        }
        for (size_t i = 0; i < p.cbs.size(); ++i) {
            if (!same_cb(p.cbs[i], base.cbs[i])) {
                why = fmt::format("chip {}: CB {} differs", c, i);
                return std::nullopt;
            }
        }
        for (size_t i = 0; i < p.kernels.size(); ++i) {
            if (const char* what = kernel_difference(base.kernels[i], p.kernels[i])) {
                why = fmt::format("chip {}: kernel {} ({}): {} differ", c, i, base.kernels[i].kernel_source, what);
                return std::nullopt;
            }
        }
    }
    // (semaphores: each id on the union of its cores; an id starts at one value everywhere)
    std::map<std::tuple<uint32_t, int>, SemaphoreDescriptor> sems;
    for (const auto& [coord, p] : chips) {
        for (const auto& s : p.semaphores) {
            auto [it, fresh] = sems.emplace(std::make_tuple(s.id, static_cast<int>(s.core_type)), s);
            if (!fresh) {
                if (it->second.initial_value != s.initial_value) {
                    why = fmt::format(
                        "semaphore {} starts at {} and {}", s.id, it->second.initial_value, s.initial_value);
                    return std::nullopt;
                }
                it->second.core_ranges = it->second.core_ranges.merge(s.core_ranges);
            }
        }
    }
    one.semaphores.clear();
    for (auto& [k, s] : sems) {
        one.semaphores.push_back(std::move(s));
    }
    return one;
}

}  // namespace chip_program_detail

tt::tt_metal::WorkloadDescriptor merge_chip_workload(
    tt::tt_metal::WorkloadDescriptor w, const char* op, std::string* why_out) {
    ZoneScopedN("moe_block: merge chip workload");
    using tt::tt_metal::distributed::MeshCoordinate;
    std::string why;
    auto refuse = [&](std::string reason) {
        log_debug(tt::LogOp, "{}: one program per chip ({})", op, reason);
        if (why_out != nullptr) {
            *why_out = std::move(reason);
        }
        return std::move(w);
    };
    if (w.programs.size() < 2) {
        return w;
    }
    std::vector<std::pair<MeshCoordinate, tt::tt_metal::ProgramDescriptor>> chips;
    std::vector<uint32_t> lo, hi;
    for (const auto& prog : w.programs) {
        if (prog.range.shape().mesh_size() != 1) {
            return refuse("a program spans more than one chip");
        }
        const auto& c = prog.range.start_coord();
        if (lo.empty()) {
            lo.assign(c.coords().begin(), c.coords().end());
            hi = lo;
        }
        for (size_t d = 0; d < c.dims(); ++d) {
            lo[d] = std::min(lo[d], c[d]);
            hi[d] = std::max(hi[d], c[d]);
        }
        chips.emplace_back(c, prog.descriptor);
    }
    const tt::tt_metal::distributed::MeshCoordinateRange range{MeshCoordinate(lo), MeshCoordinate(hi)};
    if (range.shape().mesh_size() != chips.size()) {
        return refuse("the chips are not one rectangle");
    }
    auto merged = chip_program_detail::merge_chip_programs(chips, why);
    if (!merged.has_value()) {
        return refuse(why);
    }
    log_debug(tt::LogOp, "{}: one program for {} chips", op, range.shape().mesh_size());
    tt::tt_metal::WorkloadDescriptor one;
    one.programs.push_back({range, std::move(*merged)});
    return one;
}

}  // namespace ttnn::operations::experimental::moe_block
