// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "patched_generic_op_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "device/patched_generic_op_device_operation.hpp"
#include "tools/profiler/host_dispatch_microbench.hpp"
#include <tt-metalium/mesh_coord.hpp>

#include <cstdint>
#include <unordered_map>

namespace nb = nanobind;

namespace ttnn::operations::experimental::generic::detail {

namespace {

/// Per-descriptor snapshot of io_tensor buffer addresses from the previous dispatch.
/// Keyed by ProgramDescriptor pointer identity (callers reuse the same object).
std::unordered_map<const void*, std::vector<std::uint32_t>> g_prev_io_addresses;

/// Collect device buffer addresses for all io_tensors (0 for null buffers).
std::vector<std::uint32_t> collect_addresses(const std::vector<Tensor>& io_tensors) {
    std::vector<std::uint32_t> addrs;
    addrs.reserve(io_tensors.size());
    for (const auto& t : io_tensors) {
        auto* buf = t.buffer();
        addrs.push_back(buf != nullptr ? buf->address() : 0u);
    }
    return addrs;
}

/// Diff current vs previous addresses; return indices that changed.
std::vector<std::uint32_t> diff_addresses(
    const std::vector<std::uint32_t>& cur, const std::vector<std::uint32_t>& prev) {
    std::vector<std::uint32_t> changed;
    if (prev.size() != cur.size()) {
        // Size mismatch (first call or layout change) — all indices changed.
        changed.resize(cur.size());
        for (std::uint32_t i = 0; i < cur.size(); ++i) {
            changed[i] = i;
        }
    } else {
        for (std::uint32_t i = 0; i < cur.size(); ++i) {
            if (cur[i] != prev[i]) {
                changed.push_back(i);
            }
        }
    }
    return changed;
}

}  // namespace

void bind_patched_generic_op(nb::module_& mod) {
    mod.def(
        "dump_host_dispatch_microbench_report",
        []() { return tt::tt_metal::host_dispatch_microbench::format_report(); },
        R"doc(
        Return a string of host dispatch microbench stats. Enable collection with
        environment variable ``TTNN_HOST_DISPATCH_MICROBENCH=1`` before import, then call
        after running workload. Use ``reset_host_dispatch_microbench`` between trials.
        )doc");
    mod.def(
        "reset_host_dispatch_microbench",
        []() { tt::tt_metal::host_dispatch_microbench::reset_stats(); },
        R"doc(Zero all TTNN_HOST_DISPATCH_MICROBENCH counters.)doc");

    mod.def(
        "patched_generic_op",
        [](const std::vector<Tensor>& io_tensors,
           tt::tt_metal::ProgramDescriptor& program_descriptor) -> std::tuple<Tensor, std::vector<std::uint32_t>> {
            TT_FATAL(!io_tensors.empty(), "io_tensors must not be empty");
            auto* mesh_device = io_tensors.front().device();
            TT_FATAL(mesh_device != nullptr, "Tensor must be on a device");

            // Snapshot current addresses before dispatch.
            auto cur_addrs = collect_addresses(io_tensors);

            tt::tt_metal::experimental::MeshProgramDescriptor mesh_program_descriptor;
            {
                tt::tt_metal::host_dispatch_microbench::ScopedTimer _nanobind_mesh_timer(
                    tt::tt_metal::host_dispatch_microbench::Slot::PatchedNanobindMeshSetup);
                mesh_program_descriptor.mesh_programs.emplace_back(
                    ttnn::MeshCoordinateRange(mesh_device->shape()), program_descriptor);
            }

            auto result = ttnn::prim::patched_generic_op(io_tensors, mesh_program_descriptor);

            // Diff against previous snapshot for this descriptor.
            const void* key = &program_descriptor;
            auto& prev = g_prev_io_addresses[key];
            auto changed = diff_addresses(cur_addrs, prev);
            prev = std::move(cur_addrs);

            return {std::move(result), std::move(changed)};
        },
        nb::arg("io_tensors"),
        nb::arg("program_descriptor"),
        R"doc(
        Dispatch like ``generic_op`` but with a slot-based program-cache override.

        On program cache miss, builds the program and records which per-core /
        common runtime args and which CBs hold ``io_tensors`` buffer addresses.
        On cache hit, updates only those words and dynamic CB bindings — no
        memcpy of full per-core runtime-arg vectors (unlike ``generic_op``).

        Returns ``(output_tensor, changed_io_indices)`` where ``changed_io_indices``
        lists the ``io_tensors`` positions whose device buffer address differs from
        the previous dispatch of the same ``program_descriptor``.
        )doc");
}

}  // namespace ttnn::operations::experimental::generic::detail
