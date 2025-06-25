#include "tt-metalium/mesh_socket.hpp"

namespace ttnn::distributed {

class ISocket {
public:
    virtual ~ISocket() = default;

    virtual void send(const ttnn::Tensor& tensor) = 0;
    virtual void recv(ttnn::Tensor& tensor) = 0;
};

}  // namespace ttnn::distributed
