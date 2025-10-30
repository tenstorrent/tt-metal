#include "ttnn/experimental/jit/evaluation_manager.hpp"
#include "ttnn/tensor/metal_tensor.hpp"
#include "ttnn/experimental/jit/graph_utils.hpp"

namespace ttnn::experimental::jit {

void evaluate(const LazyTensor& lazy_tensor) {
    auto sorted_tensors = GraphUtils::topological_sort(lazy_tensor);
    for (auto& tensor : sorted_tensors) {
        tensor.materialize();
    }
}

}  // namespace ttnn::experimental::jit
