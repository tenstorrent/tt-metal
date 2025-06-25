#include "isocket.hpp"

namespace ttnn::distributed {

class MPISocket : public ISocket {
public:
    MPISocket(const tt::tt_metal::distributed::MeshSocket& mesh_socket);
    ~MPISocket() override = default;

    void send(const ttnn::Tensor& tensor) override;
    void recv(ttnn::Tensor& tensor) override;

private:
    tt::tt_metal::distributed::MeshSocket mesh_socket_;
};

}  // namespace ttnn::distributed
