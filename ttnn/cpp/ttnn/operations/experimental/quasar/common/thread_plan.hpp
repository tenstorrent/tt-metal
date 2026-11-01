// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <tt_stl/assert.hpp>

namespace ttnn::prim::qsr {

// DM / NEO thread plan of a Quasar-native op: R reader harts, C NEOs, W writer harts on one node.
// Quasar reserves DM0 (DFB ISR) and DM1 (remapper), leaving six DM harts, and has four NEOs. Readers feed
// NEOs p, p+R, ... and writers drain NEOs w, w+W, ... on STRIDED rings, so C must be a multiple of both.
// Part of the op attributes (and so of the program-cache key).
struct ThreadPlan {
    uint32_t reader = 4;
    uint32_t compute = 4;
    uint32_t writer = 2;
};

inline void validate_thread_plan(const ThreadPlan& plan, const char* op_name) {
    constexpr uint32_t free_dm_harts = 6;
    constexpr uint32_t neos = 4;
    TT_FATAL(
        plan.reader > 0 && plan.compute > 0 && plan.writer > 0,
        "{}: thread plan counts must be positive, got R={} C={} W={}",
        op_name,
        plan.reader,
        plan.compute,
        plan.writer);
    TT_FATAL(
        plan.compute <= neos, "{}: compute threads ({}) exceed the {} NEOs of a node", op_name, plan.compute, neos);
    TT_FATAL(
        plan.reader + plan.writer <= free_dm_harts,
        "{}: reader ({}) + writer ({}) threads exceed the {} free DM harts of a node",
        op_name,
        plan.reader,
        plan.writer,
        free_dm_harts);
    TT_FATAL(
        plan.compute % plan.reader == 0,
        "{}: compute threads ({}) must be a multiple of reader threads ({}) for STRIDED rings",
        op_name,
        plan.compute,
        plan.reader);
    TT_FATAL(
        plan.compute % plan.writer == 0,
        "{}: compute threads ({}) must be a multiple of writer threads ({}) for STRIDED rings",
        op_name,
        plan.compute,
        plan.writer);
}

// The default plan, or the "R,C,W" override in the environment variable `env_name` (experiments only).
// The whole string must parse; validated before use.
inline ThreadPlan thread_plan_from_env(const char* env_name, const char* op_name, ThreadPlan plan = {}) {
    if (const char* env = std::getenv(env_name)) {
        unsigned r = 0, c = 0, w = 0;
        int consumed = 0;
        const bool parsed = std::strchr(env, '-') == nullptr &&
                            std::sscanf(env, "%u,%u,%u%n", &r, &c, &w, &consumed) == 3 &&
                            static_cast<size_t>(consumed) == std::strlen(env);
        TT_FATAL(parsed, "{} must be \"R,C,W\" with non-negative counts, got '{}'", env_name, env);
        plan = {.reader = r, .compute = c, .writer = w};
    }
    validate_thread_plan(plan, op_name);
    return plan;
}

}  // namespace ttnn::prim::qsr
