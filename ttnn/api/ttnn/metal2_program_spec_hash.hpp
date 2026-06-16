// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Content hash of a Metal 2.0 ProgramSpec, for use as a program-cache key.
//
// The ProgramSpec fully defines a compiled Program, so hashing it keys the cache
// at program identity: specs collide iff the programs are identical. This is the
// correct-by-construction replacement for hand-maintained per-op compute_program_hash.
//
// Two properties the hash must hold:
//   - Order-insensitive. ProgramSpec's Group<T> members are unordered; element
//     hashes are folded in sorted order so ordering can't change the key.
//   - Relaxation-aware. When a TensorParameter declares dynamic_tensor_shape /
//     match_padded_shape_only, the program is invariant to (part of) the tensor
//     shape, so the key drops it -- otherwise volume-equivalent shapes (e.g. [2,3]
//     and [3,2], both one tile) fragment the cache needlessly.
//
// Not yet the live cache key; today it backs the cache-reuse measurement. A
// reflection-based implementation belongs in tt_metal once promoted.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt_stl/reflection.hpp>

namespace ttnn::device_operation::metal2 {

namespace detail {

// Boost-style hash combiner.
inline void hash_combine(std::size_t& seed, std::size_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

// Hash one or more values through ttnn's reflection-aware hasher (handles
// strings, enums, TensorSpec, CoreRangeSet, MemoryConfig, ... uniformly).
template <typename... Ts>
std::size_t hash_of(const Ts&... values) {
    return ttsl::hash::hash_objects_with_default_seed(values...);
}

// Fold a collection of per-element hashes order-insensitively: sort, then combine.
inline std::size_t combine_unordered(std::vector<std::size_t> element_hashes) {
    std::sort(element_hashes.begin(), element_hashes.end());
    std::size_t seed = 0;
    for (std::size_t h : element_hashes) {
        hash_combine(seed, h);
    }
    return seed;
}

}  // namespace detail

// How the tensor-parameter shape is folded into the key.
enum class ShapeKeyPolicy {
    Strict,             // hash the full TensorSpec (logical + padded shape) -- ignores relaxations
    HonorRelaxations,   // respect each TensorParameter's dynamic_tensor_shape / match_padded_shape_only
    ForceShapeAgnostic  // treat every TensorParameter as volume-only (measurement upper bound on reuse)
};

namespace detail {

inline std::size_t hash_dataflow_buffer(const tt::tt_metal::experimental::DataflowBufferSpec& dfb) {
    std::size_t seed = hash_of(dfb.unique_id.get(), dfb.entry_size, dfb.num_entries);
    if (dfb.data_format_metadata) {
        hash_combine(seed, hash_of(static_cast<uint32_t>(*dfb.data_format_metadata)));
    }
    return seed;
}

inline std::size_t hash_kernel_spec(const tt::tt_metal::experimental::KernelSpec& k) {
    namespace m2 = tt::tt_metal::experimental;
    std::size_t seed = hash_of(k.unique_id.get(), k.num_threads, static_cast<int>(k.hw_config.index()));

    std::visit(
        [&](const auto& src) {
            using S = std::decay_t<decltype(src)>;
            if constexpr (std::is_same_v<S, std::filesystem::path>) {
                hash_combine(seed, hash_of(src.string()));
            } else {
                hash_combine(seed, hash_of(src.code));
            }
        },
        k.source);

    // Defines and compile-time args are maps -> hash each entry, fold order-insensitively.
    std::vector<std::size_t> defines;
    defines.reserve(k.compiler_options.defines.size());
    for (const auto& [name, value] : k.compiler_options.defines) {
        defines.push_back(hash_of(name, value));
    }
    hash_combine(seed, combine_unordered(std::move(defines)));

    std::vector<std::size_t> cta;
    cta.reserve(k.compile_time_args.size());
    for (const auto& [name, value] : k.compile_time_args) {
        cta.push_back(hash_of(name, value));
    }
    hash_combine(seed, combine_unordered(std::move(cta)));

    // Runtime-arg *schema* (names) is part of the program; values are not (they live in ProgramRunArgs).
    std::vector<std::size_t> arg_names;
    for (const auto& name : k.runtime_arg_schema.runtime_arg_names) {
        arg_names.push_back(hash_of(name, /*is_common=*/0));
    }
    for (const auto& name : k.runtime_arg_schema.common_runtime_arg_names) {
        arg_names.push_back(hash_of(name, /*is_common=*/1));
    }
    hash_combine(seed, combine_unordered(std::move(arg_names)));

    std::vector<std::size_t> dfb_bindings;
    dfb_bindings.reserve(k.dfb_bindings.size());
    for (const auto& b : k.dfb_bindings) {
        dfb_bindings.push_back(hash_of(
            b.dfb_spec_name.get(),
            b.accessor_name,
            static_cast<int>(b.endpoint_type),
            static_cast<int>(b.access_pattern)));
    }
    hash_combine(seed, combine_unordered(std::move(dfb_bindings)));

    std::vector<std::size_t> tensor_bindings;
    tensor_bindings.reserve(k.tensor_bindings.size());
    for (const auto& b : k.tensor_bindings) {
        tensor_bindings.push_back(hash_of(b.tensor_parameter_name.get(), b.accessor_name));
    }
    hash_combine(seed, combine_unordered(std::move(tensor_bindings)));

    return seed;
}

inline std::size_t hash_tensor_parameter(const tt::tt_metal::experimental::TensorParameter& p, ShapeKeyPolicy policy) {
    const auto& spec = p.spec;
    std::size_t seed =
        hash_of(p.unique_id.get(), static_cast<uint32_t>(spec.data_type()), static_cast<int>(spec.layout()));
    hash_combine(seed, hash_of(spec.memory_config()));

    const bool drop_logical_shape =
        policy == ShapeKeyPolicy::ForceShapeAgnostic ||
        (policy == ShapeKeyPolicy::HonorRelaxations &&
         (p.advanced_options.dynamic_tensor_shape || p.advanced_options.match_padded_shape_only));
    const bool drop_padded_shape =
        policy == ShapeKeyPolicy::ForceShapeAgnostic ||
        (policy == ShapeKeyPolicy::HonorRelaxations && p.advanced_options.dynamic_tensor_shape);

    if (drop_padded_shape) {
        // Program depends only on element/page count, not on tensor shape.
        hash_combine(seed, hash_of(static_cast<uint64_t>(spec.padded_shape().volume())));
    } else if (drop_logical_shape) {
        // Logical shape may vary; padded shape (and thus the access pattern) is fixed.
        hash_combine(seed, hash_of(spec.padded_shape()));
    } else {
        hash_combine(seed, hash_of(spec.logical_shape(), spec.padded_shape()));
    }
    return seed;
}

inline std::size_t hash_work_unit(const tt::tt_metal::experimental::WorkUnitSpec& w) {
    std::vector<std::size_t> kernels;
    kernels.reserve(w.kernels.size());
    for (const auto& name : w.kernels) {
        kernels.push_back(hash_of(name.get()));
    }
    std::size_t seed = combine_unordered(std::move(kernels));
    hash_combine(seed, hash_of(static_cast<int>(w.target_nodes.index())));
    std::visit([&](const auto& nodes) { hash_combine(seed, hash_of(nodes)); }, w.target_nodes);
    return seed;
}

}  // namespace detail

// Content hash of `spec`, suitable as a program-cache key. Combine with the op's
// type hash at the call site so distinct ops with coincidentally-identical specs
// don't collide.
inline std::size_t program_spec_cache_key(
    const tt::tt_metal::experimental::ProgramSpec& spec, ShapeKeyPolicy policy = ShapeKeyPolicy::HonorRelaxations) {
    using namespace detail;
    std::size_t seed = hash_of(spec.name);

    auto fold = [&](const auto& group, auto&& per_element) {
        std::vector<std::size_t> hashes;
        hashes.reserve(group.size());
        for (const auto& element : group) {
            hashes.push_back(per_element(element));
        }
        hash_combine(seed, combine_unordered(std::move(hashes)));
    };

    fold(spec.dataflow_buffers, [](const auto& d) { return hash_dataflow_buffer(d); });
    fold(spec.kernels, [](const auto& k) { return hash_kernel_spec(k); });
    fold(spec.tensor_parameters, [&](const auto& p) { return hash_tensor_parameter(p, policy); });
    fold(spec.work_units, [](const auto& w) { return hash_work_unit(w); });
    return seed;
}

}  // namespace ttnn::device_operation::metal2
