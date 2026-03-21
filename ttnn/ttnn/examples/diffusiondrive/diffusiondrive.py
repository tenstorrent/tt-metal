# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import preprocess_model
from model import create_model


def diffusiondrive_model(features, *, parameters, device):
    camera = features["camera_feature"]
    lidar = features["lidar_feature"]

    # 1. IMAGE ENCODER (ResNet-34)
    # Note: In a real bring-up, you'd iterate through parameters.backbone.image_encoder
    # For Stage 1, we can use the pre-defined ttnn ResNet if available, 
    # or map layers manually:
    img_feat = camera
    for layer in parameters.backbone.image_encoder:
        # This is a simplification; you'd call ttnn.conv2d for each layer
        img_feat = ttnn.conv2d(img_feat, layer.weight, ...) 

    # 2. LIDAR ENCODER
    # Replicating the Sequential(Conv -> BN -> ReLU) in TTNN
    lidar_feat = lidar
    # Layer 0: Conv
    lidar_feat = ttnn.conv2d(
        input_tensor=lidar_feat,
        weight=parameters.backbone.lidar_encoder[0].weight,
        bias=parameters.backbone.lidar_encoder[0].bias,
        in_channels=1,
        out_channels=64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        device=device
    )
    lidar_feat = ttnn.relu(lidar_feat)
    
    # 3. FUSION (Concatenate and 1x1 Conv)
    # Ensure tensors are in the same memory config before concat
    fused = ttnn.concat([img_feat, lidar_feat], dim=1)
    fused = ttnn.conv2d(
        fused, 
        parameters.backbone.fusion.weight, 
        bias=parameters.backbone.fusion.bias,
        kernel_size=(1, 1)
    )

    # 4. DECODER (Transformer)
    # Flatten BEV: [B, C, H, W] -> [B, H*W, C]
    batch, channels, height, width = fused.shape
    bev_flat = ttnn.reshape(fused, (batch, channels, height * width))
    bev_flat = ttnn.transpose(bev_flat, 1, 2) # [B, HW, C]

    # Use TTNN's optimized Transformer layers for Stage 2/3
    # For Stage 1, verify the logic with a simple MatMul/Softmax loop
    
    return {"trajectory": traj_output, "agent_states": agent_output}

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