#pragma once

#include <cstdint>

namespace ttnn::distributed {

enum class SocketType : uint8_t { MPI, FABRIC };

enum class EndpointSocketType : uint8_t { SENDER, RECEIVER, BIDIRECTIONAL };

}  // namespace ttnn::distributed
