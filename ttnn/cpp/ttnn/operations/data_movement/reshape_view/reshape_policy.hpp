// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <ostream>
#include <string_view>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::detail {

// Which implementation `ttnn::reshape` runs.
//   New     -- `reshape_new` (data_movement/reshape_new), the default.
//   Legacy  -- the frozen routed implementation in this directory.
//   Compare -- run both, compare the outputs bit for bit, throw on a mismatch, return the new result.
enum class ReshapeExecutionPolicy { Legacy, New, Compare };

// Case-insensitive "legacy" / "new" / "compare"; anything else is fatal.
ReshapeExecutionPolicy parse_reshape_execution_policy(std::string_view value);

// The policy in effect on the calling thread, by precedence:
//   1. the innermost ScopedReshapeExecutionPolicy on this thread;
//   2. the process-wide override (set_reshape_execution_policy_override);
//   3. TT_METAL_EXECUTION_POLICY_RESHAPE, read once per process;
//   4. New.
ReshapeExecutionPolicy reshape_execution_policy();

// Process-wide override, for tests and benchmarks that switch policy without touching the environment.
// std::nullopt restores the environment-derived policy.
void set_reshape_execution_policy_override(std::optional<ReshapeExecutionPolicy> policy);

// Pins the policy for the calling thread while in scope. The switch point installs one around each
// implementation it runs, so a reshape nested inside an implementation (legacy's own building blocks call
// ttnn::reshape) stays on that implementation instead of re-dispatching or recursing in compare mode.
class [[nodiscard]] ScopedReshapeExecutionPolicy {
public:
    explicit ScopedReshapeExecutionPolicy(ReshapeExecutionPolicy policy);
    ~ScopedReshapeExecutionPolicy();

    ScopedReshapeExecutionPolicy(const ScopedReshapeExecutionPolicy&) = delete;
    ScopedReshapeExecutionPolicy(ScopedReshapeExecutionPolicy&&) = delete;
    ScopedReshapeExecutionPolicy& operator=(const ScopedReshapeExecutionPolicy&) = delete;
    ScopedReshapeExecutionPolicy& operator=(ScopedReshapeExecutionPolicy&&) = delete;

private:
    std::optional<ReshapeExecutionPolicy> previous_;
};

// The arguments of one ttnn::reshape call, after the Span overload resolved inferred dims.
struct ReshapeRequest {
    const ttnn::Tensor& input;
    const ttnn::Shape& logical_shape;
    const ttnn::Shape& padded_shape;
    const std::optional<MemoryConfig>& memory_config;
    const std::optional<PadValue>& pad_value;
    const std::optional<CoreRangeSet>& sub_core_grid;
    bool skip_padding_fill;
};

// The graph-capture argument recorded for every ttnn::reshape call, including the views and no-ops that
// never reach a device operation. It streams as one line of JSON carrying everything needed to rebuild the
// call: the input's logical and padded shapes, dtype, layout, tile, memory config (lossless, via
// ttsl::json::to_json) and topology, and the request. tests/ttnn/benchmark/reshape harvests these.
struct ReshapeCallRecord {
    const ReshapeRequest& request;
};

std::ostream& operator<<(std::ostream& os, const ReshapeCallRecord& record);

// False during trace capture and NO_DISPATCH graph capture, where compare mode cannot read outputs back and so
// runs only the new implementation.
bool compare_readback_allowed(const ttnn::Tensor& tensor);

// Compare-mode check. Throws when the logical shape, dtype or any logical element differs bit for bit;
// the message names the first mismatches by N-D index together with both specs and the request. Spec,
// padded-shape and topology differences are logged, not thrown: those are where the new implementation
// intentionally departs from legacy. Skipped (with a one-time warning) when the outputs cannot be read
// back: during trace capture and NO_DISPATCH graph capture.
void compare_reshape_results(const ReshapeRequest& request, const ttnn::Tensor& legacy, const ttnn::Tensor& candidate);

}  // namespace ttnn::operations::data_movement::detail
