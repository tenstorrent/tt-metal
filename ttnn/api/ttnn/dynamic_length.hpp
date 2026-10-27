// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>

#include <tt_stl/reflection.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::dynamic_length {

// The valid (unpadded) extent of one tensor axis: a host scalar, or a single-valued UINT32/INT32 device tensor
// that the kernels read at run time, so the value can change between replays of one captured trace.
using ValidLength = std::variant<std::size_t, Tensor>;

struct SplitLength {
    // The host value; equals the capacity when the length lives in `tensor`, so host-side sizing sees the worst case.
    std::size_t value = 0;
    std::optional<Tensor> tensor;
};

SplitLength split(const ValidLength& length, std::size_t capacity);

// An axis whose valid length may differ from the extent the program was built for. The program is sized for
// `capacity`, so a host length above it is refused rather than run.
struct DynamicAxis {
    static constexpr std::size_t kUnbounded = std::numeric_limits<std::size_t>::max();

    std::string_view name;
    std::size_t capacity = kUnbounded;
    // Host value. Always checked against `capacity` and `alignment`, including when `tensor` is set: `split()`
    // stores the capacity there, and a direct caller must not pass a larger placeholder. The word stored in
    // `tensor` is not read on the host. Keeping that live value within capacity and alignment is the caller's
    // contract (or the kernel's, if it clamps). A trace capture additionally requires a scalar axis's value to
    // equal its capacity; see the mesh adapter.
    std::size_t value = 0;
    const Tensor* tensor = nullptr;
    std::size_t alignment = 1;
};

// Checks every host placeholder against its capacity and alignment, whether or not a length tensor is set, and
// checks that a length tensor is a single allocated UINT32/INT32 value on device. The tensor's stored word is not
// read. Throws naming `operation_name` and the first offending axis.
void validate_axes(std::span<const DynamicAxis> axes, std::string_view operation_name);

namespace detail {
class KeyEncoder;
}  // namespace detail

// What an op's program does not depend on, for the framework to leave out of the program hash and the exact cache
// key, and what it depends on that its arguments no longer key once those are left out. Every reference must be
// into the attributes or tensor_args the op was handed: they are matched by address.
class DynamicKey {
public:
    // A dimension whose extent the program does not depend on, e.g. the rows of a packed sequence; `dim` may be
    // negative. The padded extent goes with it. A sharded tensor is rejected: its shard spec still carries the
    // extent.
    DynamicKey& dim(const Tensor& tensor, int dim);
    // Nothing when `tensor` is absent.
    DynamicKey& dim(const std::optional<Tensor>& tensor, int dim);

    // An attribute the program does not depend on, e.g. a length the kernels read at run time. It must be reached
    // by reference when the attributes are walked (a member, or a std::cref in attribute_values()).
    template <typename T>
    DynamicKey& value(const T& attribute) {
        values_.push_back({&attribute, ttsl::hash::type_hash<T>});
        return *this;
    }

    // Values the program does depend on that are derived from a left-out extent or value, e.g. a block size picked
    // from the row count.
    template <typename... Types>
    DynamicKey& depends_on(const Types&... values) {
        derived_ += ttsl::hash::canonical_key(values...);
        return *this;
    }

private:
    friend class detail::KeyEncoder;

    struct Dim {
        const Tensor* tensor = nullptr;
        int dim = 0;
    };
    struct Value {
        const void* address = nullptr;
        int type = 0;
    };

    std::vector<Dim> dims_;
    std::vector<Value> values_;
    std::string derived_;
};

namespace detail {

// Encodes an op's attributes and tensor arguments as the exact program-cache key, leaving out what a DynamicKey
// declares. Aggregates are walked down to each Tensor and attribute; any other value is encoded by
// ttsl::hash::canonical_key, so a type's own to_hash or std::hash is still honored.
class KeyEncoder {
public:
    KeyEncoder(const DynamicKey& dynamic_key, std::string_view operation_name);

    // Branch for branch the walk of ttsl::hash::canonical_key, descending wherever it descends, so every value
    // it encodes can be reached; only a Tensor and a left-out value are encoded differently.
    template <typename T>
    void append(const T& object) {
        if (is_left_out(&object, ttsl::hash::type_hash<T>)) {
            // canonical_key starts every value with '\x1f', so this cannot collide with a value.
            key_ += '\x1e';
            return;
        }
        if constexpr (std::is_same_v<T, Tensor>) {
            append_tensor(object);
        } else if constexpr (
            std::numeric_limits<T>::is_integer || std::is_floating_point_v<T> || std::is_enum_v<T> ||
            std::is_same_v<T, std::string>) {
            key_ += ttsl::hash::canonical_key(object);
        } else if constexpr (ttsl::reflection::detail::supports_compile_time_attributes_v<T>) {
            std::apply(
                [this](const auto&... attributes) { (this->append(attributes), ...); }, object.attribute_values());
        } else if constexpr (ttsl::is_specialization_v<T, std::tuple> || ttsl::is_specialization_v<T, std::pair>) {
            std::apply([this](const auto&... elements) { (this->append(elements), ...); }, object);
        } else if constexpr (ttsl::is_specialization_v<T, std::optional>) {
            key_ += ttsl::hash::canonical_key(object.has_value());
            if (object.has_value()) {
                append(*object);
            }
        } else if constexpr (ttsl::is_specialization_v<T, std::variant>) {
            key_ += ttsl::hash::canonical_key(std::uint64_t{object.index()});
            std::visit([this](const auto& alternative) { this->append(alternative); }, object);
        } else if constexpr (ttsl::is_specialization_v<T, std::reference_wrapper>) {
            append(object.get());
        } else if constexpr (ttsl::reflection::detail::is_reflective_shared_ptr_v<T>) {
            key_ += ttsl::hash::canonical_key(object != nullptr);
            if (object != nullptr) {
                append(*object);
            }
        } else if constexpr (
            !std::is_same_v<T, std::vector<bool>> &&
            (ttsl::is_specialization_v<T, std::vector> || ttsl::is_specialization_v<T, std::set> ||
             ttsl::is_span_v<T> || is_std_array_v<T>)) {
            key_ += ttsl::hash::canonical_key(std::uint64_t{std::size(object)});
            std::ranges::for_each(object, [this](const auto& element) { this->append(element); });
        } else if constexpr (ttsl::is_specialization_v<T, std::map>) {
            key_ += ttsl::hash::canonical_key(std::uint64_t{object.size()});
            std::ranges::for_each(object, [this](const auto& entry) {
                this->append(entry.first);
                this->append(entry.second);
            });
        } else if constexpr (ttsl::is_specialization_v<T, std::unordered_map>) {
            // Sorted by key, as canonical_key does, so the encoding does not depend on bucket order.
            std::vector<const typename T::value_type*> entries;
            entries.reserve(object.size());
            std::ranges::transform(object, std::back_inserter(entries), [](const auto& entry) { return &entry; });
            std::ranges::sort(entries, {}, [](const auto* entry) -> const auto& { return entry->first; });
            key_ += ttsl::hash::canonical_key(std::uint64_t{object.size()});
            std::ranges::for_each(entries, [this](const auto* entry) {
                this->append(entry->first);
                this->append(entry->second);
            });
        } else if constexpr (!ttsl::reflection::detail::supports_to_hash_v<T> && ttsl::concepts::Reflectable<T>) {
            reflect::for_each([this, &object](auto I) { this->append(reflect::get<I>(object)); }, object);
        } else {
            // vector<bool>, to_hash-only and std::hash-only types: leaves, as in canonical_key.
            key_ += ttsl::hash::canonical_key(object);
        }
    }

    // Throws if a declared dim or value was not reached.
    std::string finish() &&;

private:
    template <typename T>
    static constexpr bool is_std_array_v = false;
    template <typename T, std::size_t N>
    static constexpr bool is_std_array_v<std::array<T, N>> = true;

    bool is_left_out(const void* address, int type);
    void append_tensor(const Tensor& tensor);

    const DynamicKey& dynamic_key_;
    std::string_view operation_name_;
    std::vector<bool> dim_matched_;
    std::vector<bool> value_matched_;
    std::string key_;
};

}  // namespace detail

// The exact program-cache key of an op that declares `dynamic_key`: two invocations that differ only in what it
// leaves out share a key, and so a program.
template <typename Attributes, typename TensorArgs>
std::string program_key(
    const Attributes& attributes,
    const TensorArgs& tensor_args,
    const DynamicKey& dynamic_key,
    std::string_view operation_name) {
    detail::KeyEncoder key(dynamic_key, operation_name);
    key.append(attributes);
    key.append(tensor_args);
    return std::move(key).finish();
}

}  // namespace ttnn::dynamic_length
