#include "isocket.hpp"
#include "socket_enums.hpp"
#include "tt-metalium/mesh_socket.hpp"

namespace ttnn::distributed {

std::unique_ptr<ISocket> create_socket(
    SocketType socket_type,
    EndpointSocketType endpoint_socket_type,
    tt::tt_metal::distributed::MeshDevice& mesh_device,
    const tt::tt_metal::distributed::SocketConfig& socket_config);

}
