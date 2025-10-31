#pragma once

#include "ttnn/experimental/jit/evaluation_manager.hpp"

namespace ttnn::experimental::jit {

class UnaryOperationsFusionPass : public PassManager::Pass {
public:
    std::string name() const override;
    void run(const tt::tt_metal::Tensor& tensor) override;
};
}  // namespace ttnn::experimental::jit
