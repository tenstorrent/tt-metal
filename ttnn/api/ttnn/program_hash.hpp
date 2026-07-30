// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <ranges>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <variant>
#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp>
#include <tracy/Tracy.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/concepts.hpp>
#include <tt_stl/reflection.hpp>
#include <tt_stl/small_vector.hpp>

#include "ttnn/operation_concepts.hpp"
#include "ttnn/tensor/tensor.hpp"

// The one definition of an op's program-cache hash: cache key, canonical key and profiler all call it.
namespace ttnn::device_operation::detail {

namespace hash_traits {

template <typename T>
concept HasAttributeNames = requires { std::decay_t<T>::attribute_names; };

}  // namespace hash_traits

// Mirrors hash_object's branches so tensor_args keeps its structure in the key: a flat tensor fold
// would let {bias, no residual} collide with {no bias, residual}. Only the Tensor leaf is relaxed.
class TensorArgsHasher {
public:
    explicit TensorArgsHasher(std::span<const tt::tt_metal::experimental::TensorSpecRelaxations> relaxations) :
        relaxations_(relaxations) {}

    template <typename T>
    ttsl::hash::hash_t operator()(const T& object) {
        using decayed_t = std::decay_t<T>;
        if constexpr (std::same_as<decayed_t, ::ttnn::Tensor>) {
            return hash_tensor(object);
        } else if constexpr (ttsl::is_specialization_v<decayed_t, std::reference_wrapper>) {
            return (*this)(object.get());
        } else if constexpr (ttsl::is_specialization_v<decayed_t, std::optional>) {
            return object.has_value() ? (*this)(*object) : 0;
        } else if constexpr (std::ranges::input_range<decayed_t>) {
            // Any container of elements, so a Tensor cannot hide in a SmallVector, span or set and skip
            // its relaxation.
            ttsl::hash::hash_t hash = 0;
            for (const auto& element : object) {
                hash = ttsl::hash::hash_objects(hash, (*this)(element));
            }
            return hash;
        } else if constexpr (ttsl::is_specialization_v<decayed_t, std::pair>) {
            return ttsl::hash::hash_objects((*this)(object.first), (*this)(object.second));
        } else if constexpr (ttsl::is_specialization_v<decayed_t, std::variant>) {
            return std::visit(
                [&](const auto& value) { return ttsl::hash::hash_objects(object.index(), (*this)(value)); }, object);
        } else if constexpr (ttsl::is_specialization_v<decayed_t, std::tuple>) {
            ttsl::hash::hash_t hash = 0;
            std::apply(
                [&](const auto&... elements) { ((hash = ttsl::hash::hash_objects(hash, (*this)(elements))), ...); },
                object);
            return hash;
        } else if constexpr (hash_traits::HasAttributeNames<decayed_t>) {
            return (*this)(object.attribute_values());
        } else if constexpr (ttsl::concepts::Reflectable<decayed_t>) {
            ttsl::hash::hash_t hash = 0;
            reflect::for_each(
                [&](auto I) { hash = ttsl::hash::hash_objects(hash, (*this)(reflect::get<I>(object))); }, object);
            return hash;
        } else {
            return ttsl::hash::hash_objects_with_default_seed(object);
        }
    }

    std::size_t tensors_hashed() const { return next_relaxation_; }

private:
    ttsl::hash::hash_t hash_tensor(const ::ttnn::Tensor& tensor) {
        TT_FATAL(
            next_relaxation_ < relaxations_.size(),
            "tensor_args_relaxations() returned {} relaxation(s), fewer than the tensors in tensor_args. Return one "
            "per Tensor, counting only engaged std::optionals.",
            relaxations_.size());
        const auto& relaxation = relaxations_[next_relaxation_++];
        return ttsl::hash::hash_objects(
            static_cast<ttsl::hash::hash_t>(tensor.storage_type()),
            tt::tt_metal::experimental::hash_tensorspec_with_relaxation(tensor.tensor_spec(), relaxation));
    }

    std::span<const tt::tt_metal::experimental::TensorSpecRelaxations> relaxations_;
    std::size_t next_relaxation_ = 0;
};

template <typename TensorArgs>
ttsl::hash::hash_t hash_tensor_args_with_relaxations(
    const TensorArgs& tensor_args, std::span<const tt::tt_metal::experimental::TensorSpecRelaxations> relaxations) {
    TensorArgsHasher hasher(relaxations);
    const auto hash = hasher(tensor_args);
    TT_FATAL(
        hasher.tensors_hashed() == relaxations.size(),
        "tensor_args_relaxations() returned {} relaxation(s) but {} tensor(s) were reached. Return one per Tensor, "
        "counting only engaged std::optionals.",
        relaxations.size(),
        hasher.tensors_hashed());
    return hash;
}

// Folds the attributes exactly as ttsl::hash::hash_object's Reflectable branch does, skipping the fields
// named in attributes_excluded_from_key. Flat by design: a struct member is keyed whole or excluded whole.
template <typename attributes_t>
ttsl::hash::hash_t hash_declared_attributes(const attributes_t& attributes) {
    if constexpr (!HasExcludedAttributes<attributes_t>) {
        return ttsl::hash::hash_objects_with_default_seed(attributes);
    } else {
        ttsl::hash::hash_t hash = 0;
        reflect::for_each(
            [&](auto I) {
                if constexpr (!field_is_excluded<attributes_t, I>()) {
                    hash = ttsl::hash::hash_objects(hash, reflect::get<I>(attributes));
                }
            },
            attributes);
        return hash;
    }
}

// The exact-key companion to hash_declared_attributes: same traversal, same skips, so two keys that differ
// only in an excluded attribute encode identically and two that differ anywhere else do not.
template <typename attributes_t>
std::string canonical_declared_attributes(const attributes_t& attributes) {
    if constexpr (!HasExcludedAttributes<attributes_t>) {
        return ttsl::hash::canonical_key(attributes);
    } else {
        std::string key;
        reflect::for_each(
            [&](auto I) {
                if constexpr (!field_is_excluded<attributes_t, I>()) {
                    key += ttsl::hash::canonical_key(reflect::get<I>(attributes));
                }
            },
            attributes);
        return key;
    }
}

template <typename device_operation_t>
ttsl::hash::hash_t compute_op_hash(
    const typename device_operation_t::operation_attributes_t& operation_attributes,
    const typename device_operation_t::tensor_args_t& tensor_args) {
    if constexpr (HasLegacyProgramHash<device_operation_t>) {
        ZoneScopedN("Compute custom program hash");
        return device_operation_t::compute_program_hash(operation_attributes, tensor_args);
    } else if constexpr (
        !HasExcludedAttributes<typename device_operation_t::operation_attributes_t> &&
        !HasTensorArgsRelaxations<device_operation_t>) {
        ZoneScopedN("Compute default program hash");
        return ttsl::hash::hash_objects_with_default_seed(
            ttsl::hash::type_hash<device_operation_t>, operation_attributes, tensor_args);
    } else {
        ZoneScopedN("Compute declared program hash");
        const ttsl::hash::hash_t attributes_hash = hash_declared_attributes(operation_attributes);

        ttsl::hash::hash_t tensors_hash = 0;
        if constexpr (HasTensorArgsRelaxations<device_operation_t>) {
            const ttsl::SmallVector<tt::tt_metal::experimental::TensorSpecRelaxations> relaxations =
                device_operation_t::tensor_args_relaxations(tensor_args);
            tensors_hash = hash_tensor_args_with_relaxations(tensor_args, relaxations);
        } else {
            tensors_hash = ttsl::hash::hash_objects_with_default_seed(tensor_args);
        }

        return ttsl::hash::hash_objects_with_default_seed(
            ttsl::hash::type_hash<device_operation_t>, attributes_hash, tensors_hash);
    }
}

}  // namespace ttnn::device_operation::detail
