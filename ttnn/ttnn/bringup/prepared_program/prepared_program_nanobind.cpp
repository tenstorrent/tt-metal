// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/experimental/mesh_program_descriptor.hpp>

#include "prepared_program.hpp"

namespace nb = nanobind;

namespace ttnn::operations::bringup::prepared_program::detail {

void bind_prepared_program(nb::module_& mod) {
    nb::class_<Prepared>(mod, "PreparedProgram", R"doc(
        A generic_op program prepared once: the descriptor is kept, and each call patches only the runtime-arg slots
        holding an io tensor's address and the CBs an io tensor backs (prepared_program.hpp has the contract: every
        per-call tensor a kernel addresses is in io_tensors, at the same positions every call).

            p = ttnn.bringup.prepare_program(mesh_program_descriptor, io_tensors)   # once
            ttnn.bringup.run_prepared(p, io_tensors)                                  # every call
        )doc")
        .def_prop_ro("patch_count", &Prepared::patch_count)
        .def_prop_ro("n_io", [](const Prepared& p) { return p.n_io; })
        .def_prop_ro("n_values", [](const Prepared& p) { return p.n_values; })
        .def_prop_ro("hash", [](const Prepared& p) { return static_cast<uint64_t>(p.hash); })
        // (the launch payload of the first mesh program: runtime-arg words over its cores, kernel-cores with args,
        // common runtime-arg words, CBs)
        // (per kernel of the first mesh program: source (head of), kernel-cores, runtime-arg words, distinct per-core
        // arg vectors, words in positions equal on every core)
        .def_prop_ro(
            "kernel_stats",
            [](const Prepared& p) {
                const auto& pd = p.desc.mesh_programs.at(0).second;
                std::vector<std::tuple<std::string, size_t, size_t, size_t, size_t>> out;
                for (const auto& k : pd.kernels) {
                    std::set<std::vector<uint32_t>> distinct;
                    size_t words = 0, cores = 0, mx = 0;
                    for (const auto& [core, args] : k.runtime_args) {
                        words += args.size();
                        cores += !args.empty();
                        mx = std::max(mx, args.size());
                        distinct.insert(std::vector<uint32_t>(args.begin(), args.end()));
                    }
                    std::string src = k.kernel_source;
                    const auto slash = src.find_last_of('/');
                    if (src.size() < 200 && slash != std::string::npos) {
                        src = src.substr(slash + 1);
                    } else {
                        src = src.substr(0, 60);
                    }
                    // (words in arg positions holding one value on every core of the kernel: common-arg material)
                    size_t same = 0;
                    if (cores > 1) {
                        for (size_t i = 0; i < mx; ++i) {
                            bool eq = true, first = true;
                            uint32_t v = 0;
                            for (const auto& [core, args] : k.runtime_args) {
                                if (args.size() <= i) {
                                    eq = false;
                                    break;
                                }
                                if (first) {
                                    v = args[i];
                                    first = false;
                                } else if (args[i] != v) {
                                    eq = false;
                                    break;
                                }
                            }
                            same += eq ? cores : 0;
                        }
                    }
                    out.emplace_back(src, cores, words, distinct.size(), same);
                }
                return out;
            })
        // (per kernel of the first mesh program: the arg positions equal on every core, as [start, end) runs, and
        // each core's arg count range)
        .def_prop_ro(
            "invariant_runs",
            [](const Prepared& p) {
                const auto& pd = p.desc.mesh_programs.at(0).second;
                std::vector<std::tuple<size_t, size_t, std::vector<std::pair<size_t, size_t>>>> out;
                for (const auto& k : pd.kernels) {
                    size_t mn = SIZE_MAX, mx = 0, cores = 0;
                    for (const auto& [core, args] : k.runtime_args) {
                        if (!args.empty()) {
                            mn = std::min(mn, args.size());
                            mx = std::max(mx, args.size());
                            ++cores;
                        }
                    }
                    std::vector<std::pair<size_t, size_t>> runs;
                    for (size_t i = 0; cores > 1 && i < mn; ++i) {
                        bool eq = true, first = true;
                        uint32_t v = 0;
                        for (const auto& [core, args] : k.runtime_args) {
                            if (args.empty()) {
                                continue;
                            }
                            if (first) {
                                v = args[i];
                                first = false;
                            } else if (args[i] != v) {
                                eq = false;
                                break;
                            }
                        }
                        if (eq) {
                            if (!runs.empty() && runs.back().second == i) {
                                runs.back().second = i + 1;
                            } else {
                                runs.emplace_back(i, i + 1);
                            }
                        }
                    }
                    out.emplace_back(mn == SIZE_MAX ? 0 : mn, mx, runs);
                }
                return out;
            })
        .def_prop_ro("payload", [](const Prepared& p) {
            const auto& pd = p.desc.mesh_programs.at(0).second;
            size_t words = 0, cores = 0, common = 0;
            for (const auto& k : pd.kernels) {
                for (const auto& [core, args] : k.runtime_args) {
                    words += args.size();
                    cores += !args.empty();
                }
                common += k.common_runtime_args.size();
            }
            return std::make_tuple(words, cores, common, pd.cbs.size(), pd.kernels.size(), p.desc.mesh_programs.size());
        });

    mod.def(
        "prepare_program",
        [](const tt::tt_metal::experimental::MeshProgramDescriptor& desc,
           const std::vector<Tensor>& io_tensors,
           const std::vector<std::tuple<uint32_t, uint32_t>>& value_slots) {
            std::vector<ValueSlot> vs;
            for (uint32_t s = 0; s < value_slots.size(); ++s) {
                vs.push_back(
                    ValueSlot{.kernel = std::get<0>(value_slots[s]), .arg = std::get<1>(value_slots[s]), .slot = s});
            }
            return std::const_pointer_cast<Prepared>(prepare(desc, io_tensors, vs));
        },
        nb::arg("mesh_program_descriptor"),
        nb::arg("io_tensors"),
        nb::arg("value_slots") = std::vector<std::tuple<uint32_t, uint32_t>>{},
        "Prepare a MeshProgramDescriptor for patched dispatch (copies it once; records the patch slots). "
        "value_slots: [(kernel index, runtime-arg index)], one per-run value each (run_prepared values=, in this "
        "order), patched on every core of that kernel, left out of the cache key; default none.");
    mod.def(
        "run_prepared",
        [](const std::shared_ptr<Prepared>& p,
           const std::vector<Tensor>& io_tensors,
           const std::vector<uint32_t>& values) { return run(p, io_tensors, values); },
        nb::arg("prepared"),
        nb::arg("io_tensors"),
        nb::arg("values") = std::vector<uint32_t>{},
        "Dispatch a prepared program on these io tensors (the output last); returns the output. values: one per "
        "value slot of the prepare (default none).");
}

}  // namespace ttnn::operations::bringup::prepared_program::detail
