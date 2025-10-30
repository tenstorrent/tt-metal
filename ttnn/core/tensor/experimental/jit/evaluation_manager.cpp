#include "ttnn/experimental/jit/evaluation_manager.hpp"
#include "ttnn/experimental/jit/graph_utils.hpp"
#include "ttnn/experimental/jit/lazy_tensor.hpp"
#include "ttnn/experimental/jit/lazy_operation.hpp"

namespace ttnn::experimental::jit {

void evaluate(const std::shared_ptr<LazyTensor>& lazy_tensor) {
    auto sorted_tensors = GraphUtils::topological_sort(lazy_tensor);
    for (auto& tensor : sorted_tensors) {
        log_info(
            tt::LogTest,
            "Evaluating tensor {} with op {}",
            tensor->id(),
            tensor->op() ? tensor->op()->name() : "Unknown");
        tensor->materialize();
    }
}

}  // namespace ttnn::experimental::jit
