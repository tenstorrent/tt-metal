// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/experimental/lazy/compile/passes/passes.hpp"

namespace ttnn::experimental::lazy {

namespace compile {
class UnaryOperationsFusionPass : public compile::Pass {
public:
    std::string name() const override;
    void run(const tt::tt_metal::Tensor& tensor) override;
};
}  // namespace compile
}  // namespace ttnn::experimental::lazy
