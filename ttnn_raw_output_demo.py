#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, "/home/ubuntu/pose/tt-metal")

import cv2
import numpy as np
import torch
import ttnn
from loguru import logger

from models.demos.yolov11.reference.yolov11_pose_correct import YoloV11Pose
from models.demos.yolov11.tt.model_preprocessing_pose import create_yolov11_pose_model_parameters
from models.demos.yolov11.tt.ttnn_yolov11_pose_model import TtnnYoloV11Pose


def preprocess_image(image_path, target_size=(640, 640)):
    """Preprocess image for pose estimation"""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    orig_h, orig_w = img.shape[:2]

    # Calculate scaling and padding
    scale = min(target_size[0] / orig_w, target_size[1] / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    # Resize image
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded image
    padded_img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    padded_img[:new_h, :new_w] = resized_img

    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(padded_img).float().permute(2, 0, 1) / 255.0

    return img_tensor, orig_w, orig_h, scale, (target_size[0] - new_w) // 2, (target_size[1] - new_h) // 2


def apply_pytorch_postprocessing(predictions, anchors_per_stride):
    """
    Apply the same post-processing as the PyTorch YoloV11PoseRaw model does
    TTNN returns raw concatenated tensor: [bbox_raw(64) + conf_raw(1) + keypoints_raw(51)]
    """
    # Handle TTNN tensor layout: [1, 1, anchors, channels] -> [batch, channels, anchors]
    print(f"Input predictions shape: {predictions.shape}")
    if len(predictions.shape) == 4 and predictions.shape[0] == 1 and predictions.shape[1] == 1:
        # TTNN output: [1, 1, anchors, channels] -> [anchors, channels]
        predictions = predictions.squeeze(0).squeeze(0)
        # Convert to PyTorch expected format: [channels, anchors] -> [batch, channels, anchors]
        predictions = predictions.permute(1, 0).unsqueeze(0)  # [1, channels, anchors]

    batch_size = predictions.shape[0]
    num_channels = predictions.shape[1]
    num_anchors = predictions.shape[2]
    print(f"After processing: batch_size={batch_size}, channels={num_channels}, num_anchors={num_anchors}")

    # Split predictions into components (TTNN returns raw concatenated format)
    # predictions: [batch, 116, num_anchors]
    # Format: [bbox_raw(64) + conf_raw(1) + keypoints_raw(51)] = 116

    bbox_raw = predictions[:, :64, :]  # [batch, 64, anchors] - raw DFL features
    conf_raw = predictions[:, 64:65, :]  # [batch, 1, anchors] - raw confidence (needs sigmoid)
    kpt_raw = predictions[:, 65:116, :]  # [batch, 51, anchors] - raw keypoint features

    # ===== Process bounding boxes with DFL =====
    # Reshape bbox_raw for DFL processing: [batch, 64, anchors] -> [batch, 4, 16, anchors]
    bbox_raw = bbox_raw.reshape(batch_size, 4, 16, -1)
    bbox_raw = torch.permute(bbox_raw, (0, 2, 1, 3))  # [batch, 16, 4, anchors]

    # Apply softmax for DFL
    bbox_raw = torch.softmax(bbox_raw, dim=1)

    # Apply DFL weights (simplified version)
    dfl_weights = torch.arange(16, dtype=torch.float32, device=bbox_raw.device).reshape(1, 16, 1, 1)
    bbox_decoded = torch.sum(bbox_raw * dfl_weights, dim=1)  # [batch, 4, anchors]

    # Split into left-top and right-bottom coordinates
    lt = bbox_decoded[:, :2, :]  # [batch, 2, anchors]
    rb = bbox_decoded[:, 2:4, :]  # [batch, 2, anchors]

    # Create anchors like YOLOv11 does: grid-based anchor points
    # Each spatial location in each feature map gets anchor coordinates
    anchor_points = []
    stride_tensor = []

    # Scale 0 (stride 8): 80x80 = 6400 locations
    h0, w0 = 80, 80
    sx0 = torch.arange(end=w0, device=predictions.device, dtype=predictions.dtype) + 0.5
    sy0 = torch.arange(end=h0, device=predictions.device, dtype=predictions.dtype) + 0.5
    sy0, sx0 = torch.meshgrid(sy0, sx0, indexing="ij")
    anchor_points.append(torch.stack((sx0, sy0), -1).view(-1, 2))
    stride_tensor.append(torch.full((h0 * w0, 1), 8.0, dtype=predictions.dtype, device=predictions.device))

    # Scale 1 (stride 16): 40x40 = 1600 locations
    h1, w1 = 40, 40
    sx1 = torch.arange(end=w1, device=predictions.device, dtype=predictions.dtype) + 0.5
    sy1 = torch.arange(end=h1, device=predictions.device, dtype=predictions.dtype) + 0.5
    sy1, sx1 = torch.meshgrid(sy1, sx1, indexing="ij")
    anchor_points.append(torch.stack((sx1, sy1), -1).view(-1, 2))
    stride_tensor.append(torch.full((h1 * w1, 1), 16.0, dtype=predictions.dtype, device=predictions.device))

    # Scale 2 (stride 32): 20x20 = 400 locations
    h2, w2 = 20, 20
    sx2 = torch.arange(end=w2, device=predictions.device, dtype=predictions.dtype) + 0.5
    sy2 = torch.arange(end=h2, device=predictions.device, dtype=predictions.dtype) + 0.5
    sy2, sx2 = torch.meshgrid(sy2, sx2, indexing="ij")
    anchor_points.append(torch.stack((sx2, sy2), -1).view(-1, 2))
    stride_tensor.append(torch.full((h2 * w2, 1), 32.0, dtype=predictions.dtype, device=predictions.device))

    # Concatenate all scales
    anchors_expanded = torch.cat(anchor_points, 0).t()  # [2, 8400]
    strides_expanded = torch.cat(stride_tensor, 0).t()  # [1, 8400]

    print(f"Anchors expanded shape: {anchors_expanded.shape}, Strides expanded shape: {strides_expanded.shape}")
    print(f"Total predictions: {num_anchors}")

    # Decode bbox: (lt + rb) / 2 = center, rb - lt = width/height
    center = (lt + rb) / 2
    size = rb - lt
    bbox = torch.concat((center, size), dim=1) * strides_expanded.unsqueeze(0)  # [batch, 4, anchors]

    # ===== Process confidence =====
    # Apply sigmoid to raw confidence scores
    conf = torch.sigmoid(conf_raw)

    # ===== Process keypoints =====
    # Reshape to [batch, 17, 3, anchors] for processing
    kpt_raw = kpt_raw.reshape(batch_size, 17, 3, -1)

    # Extract x, y, visibility
    kpt_x = kpt_raw[:, :, 0, :]  # [batch, 17, anchors]
    kpt_y = kpt_raw[:, :, 1, :]  # [batch, 17, anchors]
    kpt_v = kpt_raw[:, :, 2, :]  # [batch, 17, anchors]

    # Apply sigmoid to visibility
    kpt_v = torch.sigmoid(kpt_v)

    # Decode keypoints: expand anchors for each of the 17 keypoints
    # anchors_expanded: [2, 8400], we need [2, 17, 8400] for keypoints
    anchor_x_expanded = anchors_expanded[0, :].unsqueeze(0).repeat(17, 1).unsqueeze(0)  # [1, 17, 8400]
    anchor_y_expanded = anchors_expanded[1, :].unsqueeze(0).repeat(17, 1).unsqueeze(0)  # [1, 17, 8400]
    strides_expanded_kpt = strides_expanded.unsqueeze(1).repeat(1, 17, 1)  # [1, 17, 8400]

    kpt_x_decoded = (kpt_x * 2.0 - 0.5 + anchor_x_expanded) * strides_expanded_kpt
    kpt_y_decoded = (kpt_y * 2.0 - 0.5 + anchor_y_expanded) * strides_expanded_kpt

    # Stack back to [batch, 17, 3, anchors]
    keypoints_decoded = torch.stack([kpt_x_decoded, kpt_y_decoded, kpt_v], dim=2)

    # Reshape back to [batch, 51, anchors]
    keypoints_decoded = keypoints_decoded.reshape(batch_size, 51, -1)

    # Final output: [bbox(4) + conf(1) + keypoints(51)] = 56 channels
    output = torch.concat((bbox, conf, keypoints_decoded), 1)

    return output


def run_ttnn_raw_demo(image_path):
    """Run TTNN model, get raw output, apply PyTorch post-processing, and visualize"""

    logger.info(f"Running TTNN raw output demo on: {image_path}")

    # Load and preprocess image
    img_tensor, orig_w, orig_h, scale, pad_left, pad_top = preprocess_image(image_path)
    logger.info(f"Image preprocessed: {img_tensor.shape}")

    # Load PyTorch model to get anchors and strides for post-processing
    logger.info("Loading PyTorch model for reference...")
    torch_model = YoloV11Pose()
    weights_path = "models/demos/yolov11/reference/yolov11_pose_pretrained_correct.pth"
    if os.path.exists(weights_path):
        torch_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        logger.info("Loaded pretrained weights")
    else:
        logger.warning(f"Pretrained weights not found at {weights_path}")
    torch_model.eval()

    # Get anchors and strides for YOLOv11 pose
    # YOLOv11 has 3 detection scales with strides 8, 16, 32
    # Anchors are predefined for each scale
    strides = torch.tensor([[8.0, 16.0, 32.0]])  # [1, 3]

    # Standard YOLOv11 anchors (these are typical values)
    # For each stride, there are multiple anchor boxes
    anchors_per_stride = [
        [[12, 16], [19, 36], [40, 28]],  # stride 8
        [[36, 75], [76, 55], [72, 146]],  # stride 16
        [[142, 110], [192, 243], [459, 401]],  # stride 32
    ]

    # Flatten anchors and repeat for each stride
    all_anchors = []
    for stride_idx, stride_anchors in enumerate(anchors_per_stride):
        for anchor in stride_anchors:
            all_anchors.append(anchor)

    anchors = torch.tensor(all_anchors).t()  # [2, num_anchors]
    logger.info(f"Anchors shape: {anchors.shape}, Strides shape: {strides.shape}")
    logger.info(f"Anchors: {anchors}")
    logger.info(f"Strides: {strides}")

    # Initialize TTNN with minimal memory (to avoid the allocation issue)
    logger.info("Setting up TTNN model...")

    # Try to run with minimal device setup
    device = None
    try:
        # Initialize device with minimal memory
        device = ttnn.open_device(device_id=0, l1_small_size=32768)  # 32KB

        # Create model parameters (need batch dimension for preprocessing)
        img_tensor_batched = img_tensor.unsqueeze(0)  # Add batch dimension
        parameters = create_yolov11_pose_model_parameters(torch_model, img_tensor_batched, device=device)

        # Create TTNN model (parameters are moved to device during creation)
        ttnn_model = TtnnYoloV11Pose(device=device, parameters=parameters)

        # Convert input to TTNN format and move to device (add batch dimension)
        img_tensor_batched = img_tensor.unsqueeze(0)  # Add batch dimension
        tt_input = ttnn.from_torch(img_tensor_batched, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_input = ttnn.to_device(tt_input, device)

        # Run TTNN inference (core model only - no memory-intensive post-processing)
        logger.info("Running TTNN inference (core model only)...")
        try:
            tt_output = ttnn_model(tt_input)

            # Convert back to torch (output is in TILE_LAYOUT from DRAM)
            raw_output = ttnn.to_torch(tt_output)
            logger.info(f"TTNN raw output shape: {raw_output.shape}")
            logger.info(f"TTNN raw output range: [{raw_output.min():.3f}, {raw_output.max():.3f}]")
        except Exception as model_error:
            logger.error(f"TTNN model execution failed: {model_error}")
            logger.error("TTNN inference engine failed - check memory configuration")
            return

        # Move memory-intensive post-processing to PyTorch CPU
        logger.info("Moving to PyTorch CPU for memory-intensive post-processing...")
        processed_output = apply_pytorch_postprocessing(raw_output, anchors_per_stride)
        logger.info(f"Processed output shape: {processed_output.shape}")
        logger.info(f"Processed output range: [{processed_output.min():.3f}, {processed_output.max():.3f}]")

        # Visualize results
        visualize_pose_results(image_path, processed_output, orig_w, orig_h, scale, pad_left, pad_top)

    except Exception as e:
        logger.error(f"Error during TTNN processing: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if device:
            ttnn.close_device(device)


def visualize_pose_results(image_path, predictions, orig_w, orig_h, scale, pad_left, pad_top):
    """Visualize pose estimation results"""

    # Load original image
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Could not load image for visualization: {image_path}")
        return

    # Process predictions (PyTorch post-processed format)
    batch_size, num_features, num_anchors = predictions.shape
    logger.info(f"Processing {num_anchors} anchor predictions")

    # Extract components
    bbox = predictions[0, :4, :]  # [4, num_anchors]
    conf = predictions[0, 4:5, :]  # [1, num_anchors]
    keypoints = predictions[0, 5:56, :]  # [51, num_anchors] - 17 keypoints * 3 values each

    # Find detections above confidence threshold
    conf_threshold = 0.3
    valid_indices = torch.where(conf[0] > conf_threshold)[0]

    logger.info(f"Found {len(valid_indices)} detections above confidence threshold {conf_threshold}")

    # Colors for keypoints (COCO format)
    colors = [
        (255, 0, 0),  # nose - red
        (0, 255, 0),  # eyes - green
        (0, 255, 0),
        (0, 0, 255),  # ears - blue
        (0, 0, 255),
        (255, 255, 0),  # shoulders - cyan
        (255, 255, 0),
        (255, 0, 255),  # elbows - magenta
        (255, 0, 255),
        (0, 255, 255),  # wrists - yellow
        (0, 255, 255),
        (128, 0, 128),  # hips - purple
        (128, 0, 128),
        (0, 128, 128),  # knees - teal
        (0, 128, 128),
        (128, 128, 0),  # ankles - olive
        (128, 128, 0),
    ]

    # COCO pose skeleton connections
    skeleton = [
        [15, 13],
        [13, 11],
        [16, 14],
        [14, 12],
        [11, 12],
        [5, 11],
        [6, 12],
        [5, 6],
        [5, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [1, 2],
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
    ]

    detections_count = 0

    for j in valid_indices[:5]:  # Limit to first 5 detections
        detections_count += 1

        # Get bounding box (already decoded by post-processing)
        x, y, w, h = bbox[:, j]

        # Transform bbox to original image space
        x = (x - pad_left) / scale
        y = (y - pad_top) / scale
        w = w / scale
        h = h / scale

        x1 = int(max(0, x - w / 2))
        y1 = int(max(0, y - h / 2))
        x2 = int(min(orig_w, x + w / 2))
        y2 = int(min(orig_h, y + h / 2))

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"Person {detections_count}: {conf[0, j]:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Process keypoints (already decoded by post-processing)
        kpts = keypoints[:, j].reshape(17, 3)
        kpts_transformed = []

        visible_count = 0
        for kpt_idx, kpt in enumerate(kpts):
            kx, ky, kv = kpt

            # Transform to original image coordinates
            kx_final = (kx - pad_left) / scale
            ky_final = (ky - pad_top) / scale

            # Ensure within bounds
            kx_final = max(0, min(orig_w, kx_final))
            ky_final = max(0, min(orig_h, ky_final))

            if kv > 0.3:  # visible
                visible_count += 1
                color = colors[kpt_idx % len(colors)]
                cv2.circle(img, (int(kx_final), int(ky_final)), 6, color, -1)  # filled circle
                cv2.circle(img, (int(kx_final), int(ky_final)), 8, (255, 255, 255), 2)  # white outline

            kpts_transformed.append([kx_final, ky_final, kv])

        # Draw skeleton connections
        for connection in skeleton:
            kpt1_idx = connection[0]
            kpt2_idx = connection[1]

            if (
                kpt1_idx < len(kpts_transformed)
                and kpt2_idx < len(kpts_transformed)
                and kpts_transformed[kpt1_idx][2] > 0.3
                and kpts_transformed[kpt2_idx][2] > 0.3
            ):
                pt1 = (int(kpts_transformed[kpt1_idx][0]), int(kpts_transformed[kpt1_idx][1]))
                pt2 = (int(kpts_transformed[kpt2_idx][0]), int(kpts_transformed[kpt2_idx][1]))
                cv2.line(img, pt1, pt2, (255, 255, 0), 3)  # yellow lines

        logger.info(f"Person {detections_count}: {visible_count}/17 keypoints visible")

    # Save result
    os.makedirs("models/demos/yolov11/demo/runs/pose", exist_ok=True)
    base_name = os.path.basename(image_path).rsplit(".", 1)[0]
    output_path = f"models/demos/yolov11/demo/runs/pose/{base_name}_pose_ttnn_raw.jpg"
    cv2.imwrite(output_path, img)
    logger.info(f"TTNN raw output pose estimation result saved to: {output_path}")


if __name__ == "__main__":
    # Test on dog.jpg
    image_path = "models/demos/yolov11/demo/images/dog.jpg"
    if os.path.exists(image_path):
        run_ttnn_raw_demo(image_path)
        logger.info("TTNN raw output demo completed!")
    else:
        logger.error(f"Image not found: {image_path}")
