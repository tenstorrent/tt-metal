// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <type_traits>
#include <utility>

#include <tt_stl/assert.hpp>
#include <tt_stl/small_vector.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/runtime_arg_name.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>

namespace ttnn::spec {

using tt::tt_metal::experimental::KernelRunArgs;
using tt::tt_metal::experimental::KernelSpecName;
using tt::tt_metal::experimental::NodeCoord;
using tt::tt_metal::experimental::RtaName;

// Builds a kernel's per-node run-arg values: declare names once, then emit(node, v0, v1, ...) per node.
// Fills each Table via append_unchecked (names unique by schema), so it is cleaner AND O(N) per node.
class KernelRunArgsBuilder {
public:
    KernelRunArgsBuilder(KernelSpecName kernel, std::initializer_list<RtaName> names) : names_(names) {
        result_.kernel = std::move(kernel);
    }

    // Reserve space for `n` nodes up front (one emit() per node).
    KernelRunArgsBuilder& reserve(std::size_t n) {
        result_.runtime_arg_values.reserve(n);
        return *this;
    }

    // Append one node's values, matched positionally to the declared names.
    template <typename... Vs>
    KernelRunArgsBuilder& emit(const NodeCoord& node, Vs... values) {
        static_assert(
            (std::is_convertible_v<Vs, uint32_t> && ...), "KernelRunArgsBuilder::emit values must convert to uint32_t");
        TT_FATAL(
            sizeof...(values) == names_.size(),
            "KernelRunArgsBuilder::emit received {} values but {} names were declared",
            sizeof...(values),
            names_.size());
        KernelRunArgs::RuntimeArgValues args;
        args.reserve(names_.size());
        std::size_t i = 0;
        ((args.append_unchecked(names_[i++], static_cast<uint32_t>(values))), ...);
        result_.runtime_arg_values.push_back({node, std::move(args)});
        return *this;
    }

    // Hand off the built KernelRunArgs. The builder is empty afterward.
    KernelRunArgs take() { return std::move(result_); }

private:
    ttsl::SmallVector<RtaName, 8> names_;
    KernelRunArgs result_;
};

// Builds a whole program's run-args: declare each kernel once, emit per node, take() the lot.
// take() MOVES every kernel in -- authors never write `kernel_run_args = {…}` (which silently copies).
using tt::tt_metal::experimental::ProgramRunArgs;

class ProgramRunArgsBuilder {
public:
    // Declare a kernel and its RTA names; returns a stable emitter (deque-backed, never invalidated).
    KernelRunArgsBuilder& kernel(KernelSpecName name, std::initializer_list<RtaName> rta_names) {
        return kernels_.emplace_back(std::move(name), rta_names);
    }

    // Move every kernel's run-args into one ProgramRunArgs. No copy: each KernelRunArgs is moved in.
    ProgramRunArgs take() {
        ProgramRunArgs out;
        out.kernel_run_args.reserve(kernels_.size());
        for (auto& k : kernels_) {
            out.kernel_run_args.push_back(k.take());
        }
        return out;
    }

private:
    std::deque<KernelRunArgsBuilder> kernels_;
};

}  // namespace ttnn::spec
