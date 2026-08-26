// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace ttnn::graph {

// Marks the legacy stateless constraint-query path on the calling thread.
// Operations whose runtime selection can change the effective recipe use this
// to reject an exact On-mode hit instead of reporting fallback-only resources.
class ScopedStatelessConstraintQuery {
public:
    ScopedStatelessConstraintQuery() noexcept { ++depth(); }
    ~ScopedStatelessConstraintQuery() { --depth(); }

    ScopedStatelessConstraintQuery(const ScopedStatelessConstraintQuery&) = delete;
    ScopedStatelessConstraintQuery& operator=(const ScopedStatelessConstraintQuery&) = delete;

private:
    friend bool is_stateless_constraint_query_active() noexcept;

    static std::size_t& depth() noexcept {
        static thread_local std::size_t value = 0;
        return value;
    }
};

inline bool is_stateless_constraint_query_active() noexcept { return ScopedStatelessConstraintQuery::depth() != 0; }

}  // namespace ttnn::graph
