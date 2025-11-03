#pragma once

#include "ttnn/experimental/lazy/evaluation_manager.hpp"

namespace ttnn::experimental::lazy {

namespace compile {
class UnaryOperationsFusionPass : public compile::Pass {
public:
    std::string name() const override;
    void run(const tt::tt_metal::Tensor& tensor) override;
};
}  // namespace compile
}  // namespace ttnn::experimental::lazy
