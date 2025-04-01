#include "cumsum_pybind.hpp"
#include <optional>
#include "pybind11/decorators.hpp"
#include <pybind11/stl.h>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/experimental/reduction/cumsum/device/cumsum_device_operation.hpp"
#include "ttnn/operations/experimental/reduction/cumsum/cumsum.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental::reduction::detail {
namespace py = pybind11;

void bind_cumsum_operation(py::module& module) {
    using OperationType = decltype(ttnn::experimental::cumsum);
    bind_registered_operation(
        module,
        ttnn::experimental::cumsum,
        "ttnn.experimental.cumsum()",
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const int64_t dim,
               std::optional<tt::tt_metal::DataType>& dtype,
               std::optional<Tensor> preallocated_tensor,
               QueueId queue_id) { return self(queue_id, input_tensor, dim, dtype, preallocated_tensor); },
            py::arg("input").noconvert(),
            py::arg("dim"),
            py::kw_only(),
            py::arg("dtype") = std::nullopt,
            py::arg("output") = std::nullopt,
            py::arg("queueId") = DefaultQueueId});
}

}  // namespace ttnn::operations::experimental::reduction::detail
