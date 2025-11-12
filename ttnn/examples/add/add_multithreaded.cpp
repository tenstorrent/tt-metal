// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

using namespace ttnn;

int main(int /*argc*/, char** /*argv*/) {
    // Open device
    auto device = open_mesh_device(/*device_id=*/0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);

    // Enable program cache if environment variable is set
    bool program_cache_enabled = false;
    const char* enable_cache_env = std::getenv("ENABLE_PROGRAM_CACHE");
    if (enable_cache_env != nullptr && std::string(enable_cache_env) == "1") {
        std::cout << "Enabling program cache..." << std::endl;
        device.get()->enable_program_cache();
        program_cache_enabled = true;
    }

    // Create shape for 32x32 tensors
    uint32_t h = 32;
    uint32_t w = 32;
    std::array<uint32_t, 2> dimensions = {h, w};
    std::array<uint32_t, 2> dimensions2 = {2*h, 2*w};

    Shape shape(dimensions);
    Shape shape2(dimensions2);

    std::cout << "Creating two host tensors..." << std::endl;

    // Create two tensors on host with ROW_MAJOR layout, fill with different values
    auto host_tensor1 = full(shape, 1.0f, DataType::BFLOAT16, ROW_MAJOR_LAYOUT);
    auto host_tensor2 = full(shape2, 2.0f, DataType::BFLOAT16, ROW_MAJOR_LAYOUT);

    std::cout << "Converting to TILE layout on host, then moving to device..." << std::endl;

    // Convert to TILE layout on host first, then move to device
    auto device_tensor1 = host_tensor1.to_layout(TILE_LAYOUT).to_device(device.get());
    auto device_tensor2 = host_tensor2.to_layout(TILE_LAYOUT).to_device(device.get());

    std::cout << "Tensors are now on device with TILE layout" << std::endl;
    std::cout << "Spawning two threads to move them back to host simultaneously..." << std::endl;

    // Variables to store results from each thread
    Tensor result1, result2;

    // Thread 1: Move tensor1 back to host, then convert layout
    std::thread thread1([&]() {
        std::cout << "  Thread 1: Moving device_tensor1 to host and converting (TILE -> ROW_MAJOR)..." << std::endl;
        // result1 = device_tensor1.cpu(); // pure tx to CPU -- ok
        // result1 = device_tensor1.cpu().to_layout(ROW_MAJOR_LAYOUT); // tx to CPU, untilize on host -- ok
        result1 = to_layout(device_tensor1, ROW_MAJOR_LAYOUT).cpu(); // untilize on device, tx to cpu -- RACE
        std::cout << "  Thread 1: Conversion complete!" << std::endl;
    });

    // Thread 2: Move tensor2 back to host, then convert layout
    std::thread thread2([&]() {
        std::cout << "  Thread 2: Moving device_tensor2 to host and converting (TILE -> ROW_MAJOR)..." << std::endl;
        // result2 = device_tensor2.cpu().to_layout(ROW_MAJOR_LAYOUT); // tx to CPU, untilize on host -- ok
        result2 = to_layout(device_tensor2, ROW_MAJOR_LAYOUT).cpu(); // untilize on device, tx to cpu -- RACE
        std::cout << "  Thread 2: Conversion complete!" << std::endl;
    });

    // Wait for both threads to complete
    thread1.join();
    thread2.join();

    std::cout << "Both threads completed successfully!" << std::endl;

    // Print some values to verify
    auto vec1 = result1.to_vector<::bfloat16>();
    auto vec2 = result2.to_vector<::bfloat16>();

    std::cout << "\nVerifying results:" << std::endl;
    std::cout << "  Tensor 1 first value: " << static_cast<float>(vec1[0]) << " (expected: 1.0)" << std::endl;
    std::cout << "  Tensor 2 first value: " << static_cast<float>(vec2[0]) << " (expected: 2.0)" << std::endl;

    // Close device
    if (program_cache_enabled) {
        std::cout << "Clearing program cache..." << std::endl;
        device.get()->clear_program_cache();
    }
    close_device(*device);
    std::cout << "\nDevice closed. Example complete!" << std::endl;

    return 0;
}
