
#include "ttnn/decorators.hpp"

#include "device/host/point_to_point_device_op.hpp"
#include "point_to_point.hpp"

namespace ttnn::operations::point_to_point {

ttnn::Tensor ExecutePointToPoint::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const tt::tt_metal::distributed::MeshCoordinate& receive_coord,
    const ccl::Topology topology,
    MeshDevice& mesh_device,
    const GlobalSemaphore& receiver_semaphore) {
    // first output tensor in list is intermediate and is discarded
    return ttnn::prim::point_to_point(input_tensor, &mesh_device, topology, receive_coord, receiver_semaphore).at(1);
}

}  // namespace ttnn::operations::point_to_point
