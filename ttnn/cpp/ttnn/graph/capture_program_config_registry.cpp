// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/graph/capture_program_config.hpp"

#include <any>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

// The single place that knows which ops expose a capturable program config. Add an op by
// including its operation_attributes_t header and appending one make_extractor<> line below.
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"

namespace ttnn::graph {
namespace {

// Build an extractor for attribute type AttrsT. `select` maps the attributes to the optional config
// to capture; on a type match with a config present, that config is copied into an owned std::any.
template <typename AttrsT, typename Select>
ProgramConfigExtractor make_extractor(Select select) {
    return [select = std::move(select)](const std::any& attr) -> std::optional<std::any> {
        if (const auto* ref = std::any_cast<std::reference_wrapper<const AttrsT>>(&attr)) {
            if (auto cfg = select(ref->get()); cfg.has_value()) {
                return std::any(std::move(*cfg));
            }
        }
        return std::nullopt;
    };
}

}  // namespace

std::span<const ProgramConfigExtractor> program_config_extractors() {
    // Explicit registry (not static self-registration, which is fragile under static-lib dead-code
    // stripping and static-init ordering). The extractor must copy out a config that is finalized at
    // the point the op announces its attributes via device_operation::launch; validate that per op
    // before adding it here.
    static const std::vector<ProgramConfigExtractor> extractors = {
        // Matmul: prim::matmul fills normalized_attributes.program_config (the auto-selected config)
        // before launch, so it is present and final when the attributes are announced.
        make_extractor<ttnn::prim::MatmulParams>(
            [](const ttnn::prim::MatmulParams& params) { return params.program_config; }),
    };
    return extractors;
}

}  // namespace ttnn::graph
