// SPDX-FileCopyrightText: © 2026 Tenstorrent AI Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Serialization helpers for graph argument tracking.
//
// These templates use symbols from <reflect> and tt_stl/reflection.hpp, which
// are intentionally kept OUT of the public tt-metalium API header
// (graph_tracking.hpp).  Only code that needs to *instantiate*
// GraphTracker::track_function_start – i.e. ttnn operation implementations –
// should include this header.

#pragma once

#include <sstream>
#include <any>
#include <functional>

#include <tt_stl/reflection.hpp>
#include <tt-metalium/graph_tracking.hpp>

namespace tt::tt_metal {
namespace graph_detail {

template <typename MemberT>
void serialize_member(std::ostringstream& oss, const MemberT& member) {
    if constexpr (std::is_enum_v<MemberT>) {
        ttsl::reflection::operator<<(oss, member);
    } else if constexpr (ttsl::is_specialization_v<MemberT, std::vector>) {
        oss << "{";
        for (size_t i = 0; i < member.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            serialize_member(oss, member[i]);
        }
        oss << "}";
    } else if constexpr (ttsl::is_specialization_v<MemberT, std::optional>) {
        if (member.has_value()) {
            serialize_member(oss, member.value());
        } else {
            oss << "std::nullopt";
        }
    } else if constexpr (ttsl::is_specialization_v<MemberT, std::reference_wrapper>) {
        serialize_member(oss, member.get());
    } else if constexpr (ttsl::is_specialization_v<MemberT, std::pair>) {
        oss << "{";
        serialize_member(oss, member.first);
        oss << ", ";
        serialize_member(oss, member.second);
        oss << "}";
    } else if constexpr (requires { oss << member; }) {
        oss << member;
    } else if constexpr (ttsl::concepts::Reflectable<MemberT>) {
        oss << reflect::type_name<MemberT>() << "(";
        reflect::for_each(
            [&oss, &member](auto I) {
                if constexpr (I > 0) {
                    oss << ", ";
                }
                serialize_member(oss, reflect::get<I>(member));
            },
            member);
        oss << ")";
    } else {
        oss << "<" << reflect::type_name<MemberT>() << ">";
    }
}

}  // namespace graph_detail

template <typename T>
std::string serialize_tracked_arg(const std::any& a) {
    std::ostringstream oss;
    const auto& ref = std::any_cast<const std::reference_wrapper<T>&>(a);
    const auto& val = ref.get();
    using CleanT = std::remove_cv_t<T>;

    // Guard against incomplete types (e.g. forward-declared MeshCommandQueue)
    // so that type traits below are never evaluated on them.
    if constexpr (!requires { sizeof(CleanT); }) {
        oss << "<incomplete type>";
    } else if constexpr (ttsl::is_specialization_v<CleanT, std::vector>) {
        ttsl::reflection::operator<<(oss, val);
    } else if constexpr (requires { oss << val; }) {
        oss << val;
    } else if constexpr (ttsl::concepts::Reflectable<CleanT>) {
        oss << reflect::type_name<CleanT>() << "(";
        reflect::for_each(
            [&oss, &val](auto I) {
                if constexpr (I > 0) {
                    oss << ", ";
                }
                graph_detail::serialize_member(oss, reflect::get<I>(val));
            },
            val);
        oss << ")";
    } else {
        oss << "<" << reflect::type_name<T>() << ">";
    }
    return oss.str();
}

}  // namespace tt::tt_metal
