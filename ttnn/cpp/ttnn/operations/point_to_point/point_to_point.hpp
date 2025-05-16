#pragma once

#include "ttnn/operations/ccl/ccl_host_types.hpp"

namespace ttnn {
namespace operations::point_to_point {

struct ExecutePointToPoint {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const tt::tt_metal::distributed::MeshCoordinate& receive_coord,
        const ccl::Topology topology,
        MeshDevice& mesh_device,
        const GlobalSemaphore& receiver_semaphore);
};
}  // namespace operations::point_to_point

constexpr auto point_to_point =
    ttnn::register_operation<"ttnn::point_to_point", ttnn::operations::point_to_point::ExecutePointToPoint>();

}  // namespace ttnn
