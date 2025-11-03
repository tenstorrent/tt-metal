#include "ttnn/experimental/lazy/compile/passes/passes.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::lazy::compile {
void PassManager::add_pass(std::unique_ptr<Pass>&& pass) { passes_.push_back(std::move(pass)); }

void PassManager::run(const ttnn::Tensor& lazy_tensor) {
    for (auto& pass : passes_) {
        pass->run(lazy_tensor);
    }
}

}  // namespace ttnn::experimental::lazy::compile
