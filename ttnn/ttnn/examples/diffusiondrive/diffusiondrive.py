# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import preprocess_model
from model import create_model


def diffusiondrive_model(features, *, parameters):
    # Use preprocessed model with optimizations
    # For conv ops, use sharded memory
    camera = features["camera_feature"]
    lidar = features["lidar_feature"]
    
    # Assume parameters have the layers
    # For example, conv1
    conv1_out = ttnn.conv2d(
        camera,
        parameters['backbone.image_encoder.0.weight'],
        parameters['backbone.image_encoder.0.bias'],
        kernel_size=7,
        stride=2,
        padding=3,
        memory_config=ttnn.L1_MEMORY_CONFIG,  # Use L1
        # Add sharding if possible
    )
    # Continue for other layers
    # For simplicity, call the preprocessed model
    return parameters.model(**features)


def main():
    model = create_model()
    features = {
        "camera_feature": torch.randn(1, 3, 256, 256),
        "lidar_feature": torch.randn(1, 1, 256, 256),
    }

    device_id = 0
    device = ttnn.open_device(device_id=device_id)

    parameters = preprocess_model(
        model_name="diffusiondrive",
        initialize_model=lambda: model,
        run_model=lambda model: model(features),
        reader_patterns_cache={},
        device=device,
    )

    with ttnn.tracer.trace():
        ttnn_features = {
            "camera_feature": ttnn.from_torch(features["camera_feature"], device=device),
            "lidar_feature": ttnn.from_torch(features["lidar_feature"], device=device),
        }
        ttnn_output = diffusiondrive_model(ttnn_features, parameters=parameters)
        # Convert outputs
        output_as_torch = {}
        for k, v in ttnn_output.items():
            output_as_torch[k] = ttnn.to_torch(v)

    ttnn.tracer.visualize(output_as_torch, file_name="diffusiondrive_model_trace.svg")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()