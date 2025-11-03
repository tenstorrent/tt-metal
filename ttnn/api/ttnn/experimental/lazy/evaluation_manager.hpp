#pragma once

#include <memory>

namespace tt::tt_metal {
class Tensor;
}

namespace ttnn::experimental::lazy {
class LazyTensor;

void evaluate(const std::shared_ptr<LazyTensor>& lazy_tensor);

namespace compile {

struct Pass {
    virtual ~Pass() = default;
    virtual std::string name() const = 0;
    virtual void run(const tt::tt_metal::Tensor& lazy_tensor) = 0;
};

class PassManager {
public:
    PassManager() = default;
    ~PassManager() = default;
    void add_pass(std::unique_ptr<Pass>&& pass);
    void run(const tt::tt_metal::Tensor& lazy_tensor);

private:
    std::vector<std::unique_ptr<Pass>> passes_;
};
}  // namespace compile
}  // namespace ttnn::experimental::lazy
