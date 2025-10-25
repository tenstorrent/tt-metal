#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/graph_tracking.hpp>
#include "ttnn/core.hpp"

// namespace ttnn {

// Tensor set_tensor_id(const Tensor& tensor) {
//     if (not tt::tt_metal::GraphTracker::instance().is_enabled()) {
//         return tensor;
//     }
//     auto output = tensor;
//     output.tensor_id = ttnn::CoreIDs::instance().fetch_and_increment_tensor_id();
//     return output;
// };

// }  // namespace ttnn
