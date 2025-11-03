#pragma once

#include "ttnn/experimental/lazy/evaluation_manager.hpp"

namespace ttnn::experimental::lazy {

class UnaryOperationsFusionPass : public PassManager::Pass {
public:
    std::string name() const override;
    void run(const tt::tt_metal::Tensor& tensor) override;
};
}  // namespace ttnn::experimental::lazy
