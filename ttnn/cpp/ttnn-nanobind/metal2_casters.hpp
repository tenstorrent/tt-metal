// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/stl/detail/nb_dict.h>

#include <tt-metalium/experimental/metal2_host_api/arg_value.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/scratchpad_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/experimental/metal2_host_api/utility/table.hpp>
#include <tt_stl/strong_type.hpp>

namespace nanobind::detail {

// Table<K, V> <-> dict. Table exposes clear()/emplace(k, v) and iterates as
// pair<const K, V>, which is what dict_caster needs.
template <typename K, typename V>
struct type_caster<tt::tt_metal::experimental::Table<K, V>>
    : dict_caster<tt::tt_metal::experimental::Table<K, V>, K, V> {};

// The Metal 2.0 spec names are string StrongTypes; expose them as plain Python strings.
// Specialized one type at a time rather than for StrongType<T, Tag> generally: a catch-all
// would also intercept StrongTypes that are bound as classes elsewhere (ttnn.QueueId).
template <typename SpecName>
struct spec_name_caster {
    using StringCaster = make_caster<std::string>;

    NB_TYPE_CASTER(SpecName, StringCaster::Name)

    bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
        StringCaster caster;
        if (!caster.from_python(src, flags_for_local_caster<std::string>(flags), cleanup) ||
            !StringCaster::template can_cast<std::string>()) {
            return false;
        }
        value = SpecName(caster.operator cast_t<std::string>());
        return true;
    }

    static handle from_cpp(const SpecName& src, rv_policy policy, cleanup_list* cleanup) {
        return StringCaster::from_cpp(*src, policy, cleanup);
    }
};

#define TTNN_SPEC_NAME_CASTER(NAME) \
    template <>                     \
    struct type_caster<tt::tt_metal::experimental::NAME> : spec_name_caster<tt::tt_metal::experimental::NAME> {}

TTNN_SPEC_NAME_CASTER(DFBSpecName);
TTNN_SPEC_NAME_CASTER(KernelSpecName);
TTNN_SPEC_NAME_CASTER(SemaphoreSpecName);
TTNN_SPEC_NAME_CASTER(ScratchpadSpecName);
TTNN_SPEC_NAME_CASTER(TensorParamName);

#undef TTNN_SPEC_NAME_CASTER

// Kernel argument values (compile-time and runtime args) are ArgValue; expose them as plain Python ints.
template <>
struct type_caster<tt::tt_metal::experimental::ArgValue> {
    using U32Caster = make_caster<std::uint32_t>;

    NB_TYPE_CASTER(tt::tt_metal::experimental::ArgValue, U32Caster::Name)

    bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept {
        U32Caster caster;
        if (!caster.from_python(src, flags_for_local_caster<std::uint32_t>(flags), cleanup)) {
            return false;
        }
        value = caster.operator cast_t<std::uint32_t>();
        return true;
    }

    static handle from_cpp(const tt::tt_metal::experimental::ArgValue& src, rv_policy policy, cleanup_list* cleanup) {
        return U32Caster::from_cpp(src.to_uint32(), policy, cleanup);
    }
};

}  // namespace nanobind::detail
