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

namespace ttnn::graph {

// An extractor recognizes one op's operation_attributes_t by its concrete type in the
// type-erased graph-capture attribute stream and copies out an owned program config.
// Returns nullopt when `attr` is not the attribute type this extractor handles.
using ProgramConfigExtractor = std::function<std::optional<std::any>(const std::any& attr)>;

// Passive graph-capture listener: is_capture_processor() == false, so it sits at the bottom of the
// GraphTracker stack and observes every op the query runs without disturbing the query's own capture
// push/pop. On each call it runs the registered extractors and keeps the last match. The copy
// happens here, in-callback, because the std::any holds a reference_wrapper to a stack object (the
// op's operation_attributes_t) that does not outlive the call.
class ProgramConfigCaptureProcessor : public tt::tt_metal::IGraphProcessor {
public:
    explicit ProgramConfigCaptureProcessor(std::span<const ProgramConfigExtractor> extractors) :
        extractors_(extractors) {}

    bool is_capture_processor() const override { return false; }

    void track_function_start(
        std::string_view /*function_name*/, std::span<tt::tt_metal::TrackedArgument> input_parameters) override {
        for (auto& param : input_parameters) {
            for (const auto& extractor : extractors_) {
                if (auto captured = extractor(param.value)) {
                    captured_ = std::move(captured);
                    break;  // at most one extractor matches a given argument's concrete type
                }
            }
        }
    }

    const std::optional<std::any>& result() const& { return captured_; }

    // Move the owned config out just before this processor is popped and destroyed, so QueryOutput
    // takes ownership without a redundant deep copy of the config.
    std::optional<std::any> take_result() { return std::move(captured_); }

private:
    std::span<const ProgramConfigExtractor> extractors_;
    std::optional<std::any> captured_;
};

// The registered extractors. Defined out of line in the registry hub
// (cpp/ttnn/graph/capture_program_config_registry.cpp) so this header stays free of op types.
std::span<const ProgramConfigExtractor> program_config_extractors();

}  // namespace ttnn::graph
