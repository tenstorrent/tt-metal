// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/types.hpp>
#include <ttnn/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>

#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "=== TT-NN + OpenCV Example ===" << std::endl;

    // Demonstrate OpenCV functionality
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    // Create a simple OpenCV matrix
    cv::Mat image = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::rectangle(image, cv::Point(10, 10), cv::Point(90, 90), cv::Scalar(0, 255, 0), 2);

    std::cout << "Created OpenCV image with dimensions: " << image.rows << "x" << image.cols << std::endl;
    std::cout << "Image channels: " << image.channels() << std::endl;

    // Try to create TT-NN device (this may fail if no hardware is available)
    try {
        std::cout << "Attempting to create TT-NN device..." << std::endl;
        auto device = ttnn::MeshDevice::create_unit_mesh(
            0,
            /*l1_small_size=*/24576,
            /*trace_region_size=*/6434816,
            /*num_command_queues=*/2,
            /*dispatch_core_config=*/tt::tt_metal::DispatchCoreConfig(tt::tt_metal::DispatchCoreType::ETH));
        std::cout << "TT-NN Device created successfully!" << std::endl;
        device->close();
    } catch (const std::exception& e) {
        std::cout << "Failed to create TT-NN device: " << e.what() << std::endl;
        std::cout << "This is expected if no Tenstorrent hardware is available." << std::endl;
    }

    std::cout << "Example completed successfully!" << std::endl;
    return 0;
}
