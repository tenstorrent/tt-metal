// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/config.hpp>
#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary_composite.hpp>
#include <ttnn/operations/rand/rand.hpp>

#include <array>
#include <cstdint>
#include <iostream>

using namespace ttnn;

int main(int /*argc*/, char** /*argv*/) {
    // Enable logging for TTNN Visualizer BEFORE opening device
    // This must be done before any TTNN operations
    std::cout << "Enabling TTNN Visualizer logging..." << std::endl;
    ttnn::CONFIG.set<"enable_fast_runtime_mode">(false);
    ttnn::CONFIG.set<"enable_logging">(true);
    ttnn::CONFIG.set<"report_name">(std::filesystem::path("test_glu_cpp"));
    ttnn::CONFIG.set<"enable_detailed_buffer_report">(true);

    std::cout << "Config: " << ttnn::CONFIG << std::endl;

    // Open device
    auto device = open_mesh_device(/*device_id=*/0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);

    // Define shape (1, 1024, 1024, 1024)
    std::array<uint32_t, 4> dimensions = {1, 1024, 1024, 1024};
    Shape shape(dimensions);

    std::cout << "Creating random tensors with shape (1, 1024, 1024, 1024)..." << std::endl;

    // Create random tensors a and b
    const auto a = ttnn::rand(shape, *device, DataType::BFLOAT16, Layout::TILE, DRAM_MEMORY_CONFIG, 0.0f, 1.0f, 42);
    const auto b = ttnn::rand(shape, *device, DataType::BFLOAT16, Layout::TILE, DRAM_MEMORY_CONFIG, 0.0f, 1.0f, 42);

    std::cout << "Performing element-wise multiplication..." << std::endl;

    // Perform element-wise multiplication: c = a * b
    const auto c = ttnn::multiply(a, b);

    std::cout << "Applying GLU operation..." << std::endl;

    // Apply GLU operation with default dim=-1 (which is 3 in 0-indexed)
    const auto ttnn_glu_out = ttnn::glu(c, -1);

    std::cout << "GLU operation completed successfully!" << std::endl;

    // Print some tensor information
    std::cout << "Input tensor shape: " << c.logical_shape() << std::endl;
    std::cout << "Output tensor shape: " << ttnn_glu_out.logical_shape() << std::endl;

    // For verification, we could copy a small sample back to host and compare
    // with torch.nn.functional.glu results, but for now we just verify
    // the operation completes without errors

    // Note: In the Python version, the comparison is:
    // ttnn_glu_out = ttnn.glu(c)
    // torch_glu_out = torch.nn.functional.glu(torch_c)
    // The difference would be calculated as:
    // torch.abs(torch_glu_out - ttnn_glu_out_torch).max()

    // For a complete test, you would:
    // 1. Copy tensors to host using ttnn::from_device or memcpy
    // 2. Compare with torch reference implementation
    // 3. Calculate absolute and relative differences

    std::cout << "Test completed successfully!" << std::endl;

    // Print report path info
    auto report_path = ttnn::CONFIG.get<"report_path">();
    if (report_path.has_value()) {
        std::cout << "TTNN Visualizer report saved to: " << report_path.value() << std::endl;
    }

    return 0;
}
