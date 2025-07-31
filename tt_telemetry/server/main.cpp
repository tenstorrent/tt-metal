#include <iostream>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include "impl/context/metal_context.hpp"

int main() {
    const tt::tt_metal::MetalContext &instance = tt::tt_metal::MetalContext::instance();
    const tt::Cluster &cluster = instance.get_cluster();
    
    std::cout << "hello from tt_telemetry" << std::endl;
    return 0;
}