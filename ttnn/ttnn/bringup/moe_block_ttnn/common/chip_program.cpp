// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "chip_program.hpp"

#include <algorithm>
#include <map>
#include <optional>

#include <tracy/Tracy.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::bringup::moe_block {

ChipWorkload merge_chip_workload(tt::tt_metal::WorkloadDescriptor w, const char* op, std::string* why_out) {
    ZoneScopedN("moe_block: merge chip workload");
    using tt::tt_metal::distributed::MeshCoordinate;
    auto refuse = [&](std::string reason) {
        log_debug(tt::LogOp, "{}: one program per chip ({})", op, reason);
        if (why_out != nullptr) {
            *why_out = std::move(reason);
        }
        ChipWorkload out;
        out.per_device_runtime_args.resize(w.programs.size());
        out.workload = std::move(w);
        return out;
    };
    if (w.programs.size() < 2) {
        return refuse("one chip");
    }
    std::map<MeshCoordinate, size_t> index_of;
    std::vector<uint32_t> lo, hi;
    for (size_t i = 0; i < w.programs.size(); ++i) {
        const auto& prog = w.programs[i];
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
        index_of.emplace(c, i);
    }
    const tt::tt_metal::distributed::MeshCoordinateRange range{MeshCoordinate(lo), MeshCoordinate(hi)};
    if (range.shape().mesh_size() != w.programs.size() || index_of.size() != w.programs.size()) {
        return refuse("the chips are not one rectangle");
    }
    // (the merge takes the chips in the range's iteration order; a sender core of some chips only has runtime args on
    // those: the others' are padded, the kernel reads them only where its block says it sends)
    std::vector<tt::tt_metal::ProgramDescriptor> in_order;
    in_order.reserve(w.programs.size());
    for (const auto& c : range) {
        in_order.push_back(w.programs[index_of.at(c)].descriptor);
    }
    auto merged =
        ttnn::merge_per_device_program_descriptors(in_order, ttnn::ProgramMergeOptions{.pad_runtime_args = true});
    if (!merged.has_value()) {
        return refuse(
            "the chips' programs differ beyond runtime-arg values (ttnn::merge_per_device_program_descriptors)");
    }
    log_debug(
        tt::LogOp,
        "{}: one program for {} chips, {} per-device runtime-arg fields",
        op,
        range.shape().mesh_size(),
        merged->per_device_runtime_args.size());
    ChipWorkload out;
    out.workload.programs.push_back({range, std::move(merged->descriptor)});
    out.per_device_runtime_args.push_back(std::move(merged->per_device_runtime_args));
    return out;
}

void add_chip_program(
    tt::tt_metal::distributed::MeshWorkload& workload,
    const ChipWorkload& chips,
    size_t index,
    tt::tt_metal::Program program) {
    const auto& range = chips.workload.programs.at(index).range;
    workload.add_program(range, std::move(program));
    if (!chips.per_device_runtime_args.at(index).empty()) {
        ttnn::apply_per_device_runtime_args(workload, range, chips.per_device_runtime_args[index]);
    }
}

}  // namespace ttnn::operations::bringup::moe_block
