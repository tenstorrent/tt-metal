// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <any>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

#include <tt-metalium/graph_tracking.hpp>

#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::graph {

// PoC: capture the program config that ttnn auto-selects for a matmul, without
// touching matmul.cpp / bound_matmul.
//
// When the matmul op runs, device_operation::launch announces its attributes to the
// graph-capture layer via GraphTracker::track_function_start(name, attributes, inputs).
// The attributes (ttnn::prim::MatmulParams, with program_config already filled at that
// point) are carried as a std::any holding std::reference_wrapper<const MatmulParams> --
// a typed value, not a string. This passive processor watches that stream, recognizes
// the matmul attributes purely by type, and copies the finalized config out.
class MatmulProgramConfigCaptureProcessor : public tt::tt_metal::IGraphProcessor {
public:
    // Passive observer: must not flip GraphTracker::is_enabled() on its own. The real
    // capture processor installed by the query keeps capture enabled.
    bool is_capture_processor() const override { return false; }

    void track_function_start(
        std::string_view /*function_name*/, std::span<tt::tt_metal::TrackedArgument> input_parameters) override {
        for (auto& param : input_parameters) {
            if (const auto* attrs =
                    std::any_cast<std::reference_wrapper<const ttnn::prim::MatmulParams>>(&param.value)) {
                // Copy out now: the std::any only references a stack object that does not
                // outlive this call. The prim matmul fills program_config before launch,
                // so it is present here.
                if (attrs->get().program_config.has_value()) {
                    captured_ = attrs->get().program_config;
                }
            }
        }
    }

    const std::optional<ttnn::operations::matmul::MatmulProgramConfig>& result() const { return captured_; }

private:
    std::optional<ttnn::operations::matmul::MatmulProgramConfig> captured_;
};

// Result of a matmul-aware constraints query: the usual QueryOutput plus the captured
// auto-selected program config (nullopt if the queried op was not a matmul).
struct MatmulQueryOutput {
    QueryOutput query;
    std::optional<ttnn::operations::matmul::MatmulProgramConfig> program_config;
};

// Runs the stateful constraints query and additionally captures the matmul program
// config ttnn selected during the real (NORMAL-phase) run.
//
// The listener is pushed BEFORE the query so it sits at the bottom of the GraphTracker
// processor stack: it observes every track_function_start the query produces, while the
// query's own GraphProcessor push/pop (which uses .back()) is left undisturbed.
template <typename Op, typename... Args>
MatmulQueryOutput query_op_constraints_with_initial_state_capturing_matmul_config(
    Op op,
    tt::tt_metal::distributed::MeshDevice* device,
    const tt::tt_metal::experimental::MockAllocatorState& initial_state,
    Args&&... args) {
    auto listener = std::make_shared<MatmulProgramConfigCaptureProcessor>();
    tt::tt_metal::GraphTracker::instance().push_processor(listener);

    // Pop the listener even if the query throws; the query balances its own captures, so
    // after it returns the listener is back on top of the stack.
    struct PopGuard {
        ~PopGuard() { tt::tt_metal::GraphTracker::instance().pop_processor(); }
    } pop_guard;

    auto query =
        query_op_constraints_with_initial_state(std::move(op), device, initial_state, std::forward<Args>(args)...);

    return MatmulQueryOutput{std::move(query), listener->result()};
}

}  // namespace ttnn::graph
