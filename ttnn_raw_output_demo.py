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
from models.demos.yolov11.reference.yolov11_pose_raw_output import YoloV11PoseRaw
from models.demos.yolov11.tt.model_preprocessing_pose import create_yolov11_pose_model_parameters
from models.demos.yolov11.tt.ttnn_yolov11_pose_model import TtnnYoloV11Pose

# Imports for pose postprocessing
try:
    from ultralytics.utils import ops

    non_max_suppression = ops.non_max_suppression
    scale_boxes = ops.scale_boxes
    from ultralytics.engine.results import Results
except ImportError:
    # Fallback if ultralytics not available
    print("Warning: ultralytics not found, using simplified postprocessing")

    def non_max_suppression(preds, conf_thres, iou_thres, classes=None, agnostic=False, max_det=300):
        # Simplified NMS implementation
        return preds

    def scale_boxes(img_shape, boxes, orig_shape):
        # Simplified scaling
        return boxes

    class Results:
        def __init__(self, img, path=None, names=None, boxes=None, keypoints=None):
            self.img = img
            self.path = path
            self.names = names
            self.boxes = boxes
            self.keypoints = keypoints


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

    # Return correct padding offsets (image is placed at top-left, so pad_left=0, pad_top=0)
    return img_tensor, orig_w, orig_h, scale, 0, 0


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
    bbox_softmax = torch.softmax(bbox_raw, dim=1)

    # Apply DFL weights (simple approximation)
    dfl_weights = torch.arange(16, dtype=torch.float32, device=bbox_raw.device).reshape(1, 16, 1, 1)
    bbox_decoded = torch.sum(bbox_softmax * dfl_weights, dim=1)  # [batch, 4, anchors]

    # Split into left-top and right-bottom coordinates
    lt = bbox_decoded[:, :2, :]  # [batch, 2, anchors]
    rb = bbox_decoded[:, 2:4, :]  # [batch, 2, anchors]

    # Create anchors exactly like YOLOv11 make_anchors function
    # For 640x640 input: strides [8,16,32] give grids [80x80, 40x40, 20x20]
    anchor_points = []
    stride_tensor = []
    strides = [8, 16, 32]

    for stride in strides:
        # Calculate grid size for this stride
        grid_h = 640 // stride  # 640/8=80, 640/16=40, 640/32=20
        grid_w = 640 // stride

        # Create grid coordinates (same as make_anchors)
        sx = torch.arange(end=grid_w, device=predictions.device, dtype=predictions.dtype) + 0.5
        sy = torch.arange(end=grid_h, device=predictions.device, dtype=predictions.dtype) + 0.5
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(
            torch.full((grid_h * grid_w, 1), stride, dtype=predictions.dtype, device=predictions.device)
        )

    # Concatenate like YOLOv11 make_anchors
    anchors_cat = torch.cat(anchor_points, 0)  # [total_anchors, 2] - (x, y)
    strides_cat = torch.cat(stride_tensor, 0)  # [total_anchors, 1]

    # Transpose anchors to [2, total_anchors] like YOLOv11
    anchors_expanded = anchors_cat.t()  # [2, total_anchors] - (y, x) -> (x, y)
    strides_expanded = strides_cat.t()  # [1, total_anchors]

    print(f"Anchors expanded shape: {anchors_expanded.shape}, Strides expanded shape: {strides_expanded.shape}")
    print(f"Total predictions: {num_anchors}")

    # Apply anchor offsets like YOLOv11 does
    # anchors_expanded: [2, total_anchors] - (x, y)
    # lt, rb: [batch, 2, total_anchors] - (x, y) offsets

    # Remove batch dimension for calculation
    lt = lt.squeeze(0)  # [2, total_anchors]
    rb = rb.squeeze(0)  # [2, total_anchors]

    lt_anchored = anchors_expanded - lt  # [2, total_anchors]
    rb_anchored = anchors_expanded + rb  # [2, total_anchors]

    # Calculate center and size
    center_x = (lt_anchored[0, :] + rb_anchored[0, :]) / 2  # x coordinates
    center_y = (lt_anchored[1, :] + rb_anchored[1, :]) / 2  # y coordinates
    width = rb_anchored[0, :] - lt_anchored[0, :]
    height = rb_anchored[1, :] - lt_anchored[1, :]

    # Scale by strides
    center_x = center_x * strides_expanded.squeeze(0)
    center_y = center_y * strides_expanded.squeeze(0)
    width = width * strides_expanded.squeeze(0)
    height = height * strides_expanded.squeeze(0)

    # Convert to (x,y,w,h) format and add batch dimension
    bbox = torch.stack([center_x, center_y, width, height], dim=0).unsqueeze(0)

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

    # Expand anchors and strides for keypoint processing [batch, 17, anchors]
    anchor_x_expanded = anchors_expanded[0:1, :].unsqueeze(0).expand(batch_size, 17, -1)  # [batch, 17, anchors]
    anchor_y_expanded = anchors_expanded[1:2, :].unsqueeze(0).expand(batch_size, 17, -1)  # [batch, 17, anchors]
    strides_expanded_kpt = strides_expanded.unsqueeze(0).expand(batch_size, 17, -1)  # [batch, 17, anchors]

    # Keypoint decoding - SAME as bbox: (kpt * 2 - 0.5 + anchor) * stride
    # Keypoints DO use anchors and strides like bounding boxes!
    kpt_x_decoded = (kpt_x * 2.0 - 0.5 + anchor_x_expanded) * strides_expanded_kpt
    kpt_y_decoded = (kpt_y * 2.0 - 0.5 + anchor_y_expanded) * strides_expanded_kpt

    # Stack back to [batch, 17, 3, anchors]
    keypoints_decoded = torch.stack([kpt_x_decoded, kpt_y_decoded, kpt_v], dim=2)

    # Reshape back to [batch, 51, anchors]
    keypoints_decoded = keypoints_decoded.reshape(batch_size, 51, -1)

    # Final output: [bbox(4) + conf(1) + keypoints(51)] = 56 channels
    output = torch.concat((bbox, conf, keypoints_decoded), 1)

    return output


# Postprocessing pipeline for pose estimation
def postprocess_pose(preds, img, orig_imgs, batch, names):
    """
    Postprocess pose estimation predictions from TTNN YOLOv11.

    Args:
        preds: [batch, 56, num_anchors] - decoded predictions
            0-3: bbox (x, y, w, h), 4: conf, 5-55: keypoints (17×3)
        img: Processed image tensor
        orig_imgs: List of original images
        batch: Batch information (paths, etc.)
        names: Class names (should include 'person')

    Returns:
        List of Results objects with boxes and keypoints
    """
    # Import dependencies inside function
    import torch

    # Try to import ultralytics components
    try:
        from ultralytics.utils import ops

        non_max_suppression = ops.non_max_suppression
        scale_boxes = ops.scale_boxes
        from ultralytics.engine.results import Results
    except ImportError:
        # Fallback implementations if ultralytics not available
        print("Warning: ultralytics not found, using simplified postprocessing")

        def non_max_suppression(preds, conf_thres, iou_thres, classes=None, agnostic=False, max_det=300):
            # Simplified NMS - just return predictions above threshold
            batch_size = preds.shape[0]
            results = []
            for i in range(batch_size):
                pred = preds[i]
                # Filter by confidence
                conf_mask = pred[:, 4] > conf_thres
                pred = pred[conf_mask]
                if len(pred) > 0:
                    # Sort by confidence and take top max_det
                    pred = pred[pred[:, 4].argsort(descending=True)][:max_det]
                    results.append(pred)
                else:
                    results.append(torch.empty(0, 6))
            return results

        def scale_boxes(img_shape, boxes, orig_shape):
            # Simplified scaling - assumes no padding
            scale_x = orig_shape[1] / img_shape[1]  # width scale
            scale_y = orig_shape[0] / img_shape[0]  # height scale
            boxes[:, [0, 2]] *= scale_x  # x coordinates
            boxes[:, [1, 3]] *= scale_y  # y coordinates
            return boxes

        class Results:
            def __init__(self, img, path=None, names=None, boxes=None, keypoints=None):
                self.img = img
                self.path = path
                self.names = names
                self.boxes = boxes
                self.keypoints = keypoints

    args = {"conf": 0.25, "iou": 0.7, "agnostic_nms": False, "max_det": 300, "classes": None}

    # Extract person detections (class 0 for COCO person)
    # preds shape: [batch, 56, num_anchors]
    # We need to convert to detection format: [num_anchors, 6] for each batch item
    # Format: [x1, y1, x2, y2, conf, class]

    batch_results = []

    for batch_idx, (pred_batch, orig_img, img_path) in enumerate(zip(preds, orig_imgs, batch[0])):
        # pred_batch: [56, num_anchors]
        bbox = pred_batch[:4, :]  # [4, num_anchors] - (x, y, w, h)
        conf = pred_batch[4:5, :]  # [1, num_anchors]
        keypoints = pred_batch[5:56, :]  # [51, num_anchors] - 17 keypoints × 3

        # Convert bbox from (x,y,w,h) to (x1,y1,x2,y2) format for NMS
        x, y, w, h = bbox
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        # Create detection tensor: [num_anchors, 6] - (x1,y1,x2,y2,conf,class)
        detections = torch.stack([x1, y1, x2, y2, conf.squeeze(0), torch.zeros_like(conf.squeeze(0))], dim=1)

        # Apply non-maximum suppression for person detections
        nms_detections = non_max_suppression(
            detections.unsqueeze(0),  # Add batch dimension
            args["conf"],
            args["iou"],
            classes=None,  # Only person class (0)
            agnostic=args["agnostic_nms"],
            max_det=args["max_det"],
        )[
            0
        ]  # Remove batch dimension

        if len(nms_detections) > 0:
            # Scale boxes back to original image coordinates
            nms_detections[:, :4] = scale_boxes(img.shape[2:], nms_detections[:, :4], orig_img.shape)

            # For each detection, extract corresponding keypoints
            final_keypoints = []
            for det in nms_detections:
                # Find the closest anchor to this detection (simplified - use bbox center)
                det_x = (det[0] + det[2]) / 2
                det_y = (det[1] + det[3]) / 2

                # Find anchor with closest center (simplified approach)
                anchor_centers_x = x
                anchor_centers_y = y
                distances = torch.sqrt((anchor_centers_x - det_x) ** 2 + (anchor_centers_y - det_y) ** 2)
                closest_anchor_idx = torch.argmin(distances)

                # Extract keypoints for this anchor
                kpt = keypoints[:, closest_anchor_idx]  # [51]
                kpt = kpt.reshape(17, 3)  # [17, 3] - (x, y, visibility)

                # Scale keypoints back to original image coordinates
                kpt_x = kpt[:, 0] * (orig_img.shape[1] / img.shape[3])  # Scale X
                kpt_y = kpt[:, 1] * (orig_img.shape[0] / img.shape[2])  # Scale Y
                kpt_v = kpt[:, 2]  # Visibility unchanged

                # Combine scaled keypoints
                scaled_kpt = torch.stack([kpt_x, kpt_y, kpt_v], dim=1).flatten()  # [51]
                final_keypoints.append(scaled_kpt)

            final_keypoints = torch.stack(final_keypoints) if final_keypoints else torch.empty(0, 51)

            # Create Results object with both boxes and keypoints
            # Note: This assumes Results class can handle keypoints
            result = Results(orig_img, path=img_path, names=names, boxes=nms_detections, keypoints=final_keypoints)
        else:
            # No detections
            result = Results(
                orig_img, path=img_path, names=names, boxes=torch.empty(0, 6), keypoints=torch.empty(0, 51)
            )

        batch_results.append(result)

    return batch_results


def save_pose_predictions_by_model(result, save_dir, image_path, model_name):
    """
    Save pose estimation predictions with bounding boxes and keypoints.

    Args:
        result: Results object containing boxes and keypoints
        save_dir: Directory to save predictions
        image_path: Path to original image
        model_name: Name of the model (for subdirectory)
    """
    # Import dependencies inside function
    import cv2
    import os
    from datetime import datetime
    import numpy as np

    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    # Load and prepare image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Set colors based on model type
    if model_name == "torch_model":
        bbox_color, label_color, kpt_color, skeleton_color = (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0)
    else:
        bbox_color, label_color, kpt_color, skeleton_color = (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)

    # COCO pose skeleton connections (same as used in visualization)
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

    # Keypoint colors (COCO format)
    keypoint_colors = [
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

    # Draw bounding boxes and labels
    if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            # box format: [x1, y1, x2, y2, conf, class]
            if len(box) >= 6:
                x1, y1, x2, y2, conf, cls = map(int, box[:6]) if not hasattr(box, "tolist") else box[:6].tolist()
                label = f"person {conf/100:.2f}" if hasattr(box, "tolist") else f"person {box[4]:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, 3)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

                # Draw keypoints for this detection
                if hasattr(result, "keypoints") and result.keypoints is not None and len(result.keypoints) > i:
                    kpts = result.keypoints[i]  # [51] flattened array
                    if len(kpts) >= 51:
                        # Reshape to [17, 3] - (x, y, visibility)
                        kpts_reshaped = (
                            kpts.reshape(17, 3) if hasattr(kpts, "reshape") else np.array(kpts).reshape(17, 3)
                        )

                        # Draw keypoints
                        for j, (x, y, v) in enumerate(kpts_reshaped):
                            if v > 0.5:  # Only draw visible keypoints
                                cv2.circle(image, (int(x), int(y)), 4, keypoint_colors[j], -1)
                                cv2.circle(image, (int(x), int(y)), 2, (255, 255, 255), -1)  # White center

                        # Draw skeleton connections
                        for connection in skeleton:
                            start_idx, end_idx = connection
                            if kpts_reshaped[start_idx, 2] > 0.5 and kpts_reshaped[end_idx, 2] > 0.5:
                                start_point = (int(kpts_reshaped[start_idx, 0]), int(kpts_reshaped[start_idx, 1]))
                                end_point = (int(kpts_reshaped[end_idx, 0]), int(kpts_reshaped[end_idx, 1]))
                                cv2.line(image, start_point, end_point, skeleton_color, 2)

    # Convert back to BGR for saving
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Create output filename with timestamp
    image_base = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{image_base}_pose_prediction_{timestamp}.jpg"
    output_path = os.path.join(model_save_dir, output_name)

    # Save the image
    cv2.imwrite(output_path, image)
    print(f"Saved pose prediction to: {output_path}")


def debug_confidence_processing():
    """
    Debug the confidence processing specifically.
    """
    import torch

    print("=" * 60)
    print("DEBUGGING CONFIDENCE PROCESSING")
    print("=" * 60)

    # Load PyTorch model
    torch_model = YoloV11Pose()
    weights_path = "models/demos/yolov11/reference/yolov11_pose_pretrained_correct.pth"
    if os.path.exists(weights_path):
        torch_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        torch_model.eval()
    else:
        print("Weights not found")
        return

    # Create test input
    test_input = torch.randn(1, 3, 640, 640)

    # Get PyTorch output
    with torch.no_grad():
        torch_output = torch_model(test_input)

    print(f"PyTorch confidence range: [{torch_output[0, 4, :].min():.6f}, {torch_output[0, 4, :].max():.6f}]")

    # Simulate raw confidence values (what we get before sigmoid)
    # Let's create some test confidence values
    raw_conf = torch.randn(1, 1, 8400) * 2  # Similar range to what we see
    print(f"Raw confidence input range: [{raw_conf.min():.3f}, {raw_conf.max():.3f}]")

    # Apply sigmoid (what our postprocessing does)
    processed_conf = torch.sigmoid(raw_conf)
    print(f"After sigmoid range: [{processed_conf.min():.6f}, {processed_conf.max():.6f}]")

    # Compare with expected PyTorch range
    print("\nExpected PyTorch range: [0.000000, 0.6332]")
    print(f"Our sigmoid range: [{processed_conf.min():.6f}, {processed_conf.max():.6f}]")

    # The issue might be that our raw confidence values are not in the right range
    # Let's try different raw ranges
    print("\nTesting different raw confidence ranges:")

    for scale in [0.1, 1.0, 2.0, 5.0]:
        test_raw = torch.randn(1, 1, 100) * scale
        test_sigmoid = torch.sigmoid(test_raw)
        print(".3f")


def debug_keypoint_processing():
    """
    Debug keypoint processing specifically.
    """
    import torch

    print("=" * 60)
    print("DEBUGGING KEYPOINT PROCESSING")
    print("=" * 60)

    # Create test keypoint data
    batch_size = 1
    num_anchors = 100

    # Raw keypoint values (x, y, visibility) - 17 keypoints × 3 = 51 values
    raw_kpts = torch.randn(batch_size, 51, num_anchors) * 2
    print(f"Raw keypoints range: [{raw_kpts.min():.3f}, {raw_kpts.max():.3f}]")

    # Reshape to [batch, 17, 3, anchors]
    kpt_reshaped = raw_kpts.reshape(batch_size, 17, 3, num_anchors)
    kpt_x = kpt_reshaped[:, :, 0, :]  # [batch, 17, anchors]
    kpt_y = kpt_reshaped[:, :, 1, :]  # [batch, 17, anchors]
    kpt_v = kpt_reshaped[:, :, 2, :]  # [batch, 17, anchors]

    print(f"Raw kpt_x range: [{kpt_x.min():.3f}, {kpt_x.max():.3f}]")
    print(f"Raw kpt_y range: [{kpt_y.min():.3f}, {kpt_y.max():.3f}]")
    print(f"Raw kpt_v range: [{kpt_v.min():.3f}, {kpt_v.max():.3f}]")

    # Apply visibility sigmoid
    kpt_v_sigmoid = torch.sigmoid(kpt_v)
    print(f"After sigmoid kpt_v range: [{kpt_v_sigmoid.min():.3f}, {kpt_v_sigmoid.max():.3f}]")

    # Test our decoding formula: (kpt * 2 - 0.5 + anchor) * stride
    # Create mock anchors and strides
    anchors_x = torch.randint(0, 80, (17, num_anchors))  # Mock anchor x coords
    anchors_y = torch.randint(0, 80, (17, num_anchors))  # Mock anchor y coords
    strides = torch.full((1, 1, num_anchors), 8.0)  # Mock stride

    # Test different approaches
    print("\nTesting different keypoint decoding approaches:")

    # First, let's get the actual anchors from our anchor generation
    # Replicate the anchor generation from our postprocessing
    anchor_points = []
    stride_tensor = []
    strides_vals = [8, 16, 32]

    for stride in strides_vals:
        grid_h = 640 // stride
        grid_w = 640 // stride
        sx = torch.arange(end=grid_w, dtype=torch.float32) + 0.5
        sy = torch.arange(end=grid_h, dtype=torch.float32) + 0.5
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((grid_h * grid_w, 1), stride, dtype=torch.float32))

    anchors_cat = torch.cat(anchor_points, 0)
    strides_cat = torch.cat(stride_tensor, 0)
    anchors_expanded = anchors_cat.t()  # [2, total_anchors]

    # Use correct anchors for keypoint decoding
    real_anchor_x = anchors_expanded[0, :].unsqueeze(0).unsqueeze(0)  # [1, 1, num_anchors]
    real_anchor_y = anchors_expanded[1, :].unsqueeze(0).unsqueeze(0)  # [1, 1, num_anchors]
    real_strides = strides_cat.t().unsqueeze(0)  # [1, 1, num_anchors]

    # Approach 1: Our current approach with CORRECT anchors
    kpt_x_decoded1 = (kpt_x * 2.0 - 0.5 + real_anchor_x[:, :, :num_anchors]) * real_strides[:, :, :num_anchors]
    kpt_y_decoded1 = (kpt_y * 2.0 - 0.5 + real_anchor_y[:, :, :num_anchors]) * real_strides[:, :, :num_anchors]

    print("Approach 1 (with correct anchors):")
    print(f"  kpt_x range: [{kpt_x_decoded1.min():.3f}, {kpt_x_decoded1.max():.3f}]")
    print(f"  kpt_y range: [{kpt_y_decoded1.min():.3f}, {kpt_y_decoded1.max():.3f}]")

    # Approach 2: No anchor offset (just raw values)
    kpt_x_decoded2 = kpt_x * 2.0 - 0.5
    kpt_y_decoded2 = kpt_y * 2.0 - 0.5

    print("Approach 2 (no anchor):")
    print(f"  kpt_x range: [{kpt_x_decoded2.min():.3f}, {kpt_x_decoded2.max():.3f}]")
    print(f"  kpt_y range: [{kpt_y_decoded2.min():.3f}, {kpt_y_decoded2.max():.3f}]")

    # Expected range from PyTorch (from PCC test)
    print("\nExpected PyTorch keypoint range: [-8.72, 5.02]")

    # Check which is closer
    expected_min, expected_max = -8.72, 5.02
    diff1_min = abs(kpt_x_decoded1.min().item() - expected_min)
    diff1_max = abs(kpt_x_decoded1.max().item() - expected_max)
    diff2_min = abs(kpt_x_decoded2.min().item() - expected_min)
    diff2_max = abs(kpt_x_decoded2.max().item() - expected_max)

    print(".3f")

    if diff2_min < diff1_min:
        print("✅ Approach 2 (no anchor) is closer - keypoints may not use anchors!")
    else:
        print("❌ Approach 1 (with anchor) is closer - need to debug anchor application")


def debug_anchor_generation():
    """
    Debug anchor generation to ensure it matches PyTorch.
    """
    import torch

    print("=" * 60)
    print("DEBUGGING ANCHOR GENERATION")
    print("=" * 60)

    # Replicate the anchor generation from our postprocessing
    anchor_points = []
    stride_tensor = []
    strides = [8, 16, 32]

    for stride in strides:
        grid_h = 640 // stride  # 640/8=80, 640/16=40, 640/32=20
        grid_w = 640 // stride

        # Create grid coordinates (same as make_anchors)
        sx = torch.arange(end=grid_w, dtype=torch.float32) + 0.5
        sy = torch.arange(end=grid_h, dtype=torch.float32) + 0.5
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((grid_h * grid_w, 1), stride, dtype=torch.float32))

    # Concatenate like YOLOv11 make_anchors
    anchors_cat = torch.cat(anchor_points, 0)  # [total_anchors, 2] - (x, y)
    strides_cat = torch.cat(stride_tensor, 0)  # [total_anchors, 1]

    # Transpose anchors to [2, total_anchors] like YOLOv11
    anchors_expanded = anchors_cat.t()  # [2, total_anchors] - (y, x) -> (x, y)
    strides_expanded = strides_cat.t()  # [1, total_anchors]

    print(f"Total anchors: {anchors_expanded.shape[1]}")
    print(f"Anchors shape: {anchors_expanded.shape}")
    print(f"Strides shape: {strides_expanded.shape}")
    print(f"First few anchors: {anchors_expanded[:, :5]}")
    print(f"First few strides: {strides_expanded[:, :5]}")

    # Verify the anchor ranges
    print("\nAnchor statistics:")
    print(f"  X range: [{anchors_expanded[0, :].min():.1f}, {anchors_expanded[0, :].max():.1f}]")
    print(f"  Y range: [{anchors_expanded[1, :].min():.1f}, {anchors_expanded[1, :].max():.1f}]")
    print(f"  Stride range: [{strides_expanded.min():.1f}, {strides_expanded.max():.1f}]")


def run_ttnn_raw_demo(image_path):
    """Run TTNN model, get raw output, apply PyTorch post-processing, and visualize"""

    logger.info(f"Running TTNN raw output demo on: {image_path}")

    # Load and preprocess image
    img_tensor, orig_w, orig_h, scale, pad_left, pad_top = preprocess_image(image_path)
    logger.info(f"Image preprocessed: {img_tensor.shape}")

    # Load PyTorch model for reference
    logger.info("Loading PyTorch model for reference...")
    torch_model = YoloV11Pose()
    weights_path = "models/demos/yolov11/reference/yolov11_pose_pretrained_correct.pth"
    if os.path.exists(weights_path):
        torch_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        logger.info("Loaded pretrained weights")
    else:
        logger.warning(f"Pretrained weights not found at {weights_path}")
    torch_model.eval()

    # Run PyTorch model to get reference output for comparison
    logger.info("Running PyTorch model for reference comparison...")
    with torch.no_grad():
        torch_ref_output = torch_model(img_tensor.unsqueeze(0))  # Add batch dimension
    logger.info(f"PyTorch reference output shape: {torch_ref_output.shape}")
    logger.info(f"PyTorch reference sample bbox: {torch_ref_output[0, :4, 0]}")
    logger.info(f"PyTorch reference sample conf: {torch_ref_output[0, 4, 0]}")
    logger.info(f"PyTorch reference sample keypoints: {torch_ref_output[0, 5:11, 0]}")

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

    # Continue with anchor processing for TTNN
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

        # Debug: Check raw TTNN output values
        print(f"Raw TTNN output shape: {raw_output.shape}")
        print(f"Raw TTNN output sample bbox: {raw_output[0, 0, 0, :5]}")  # First 5 bbox values
        print(f"Raw TTNN output sample kpts: {raw_output[0, 0, 0, 64:69]}")  # First 5 keypoint values

        # Move memory-intensive post-processing to PyTorch CPU
        logger.info("Moving to PyTorch CPU for memory-intensive post-processing...")
        processed_output = apply_pytorch_postprocessing(raw_output, anchors_per_stride)
        logger.info(f"Processed output shape: {processed_output.shape}")
        logger.info(f"Processed output range: [{processed_output.min():.3f}, {processed_output.max():.3f}]")

        # Compare with PyTorch reference
        logger.info("Comparing TTNN post-processing with PyTorch reference...")
        logger.info(f"TTNN processed shape: {processed_output.shape}")
        logger.info(f"TTNN processed sample bbox: {processed_output[0, :4, 0]}")
        logger.info(f"TTNN processed sample conf: {processed_output[0, 4, 0]}")
        logger.info(f"TTNN processed sample keypoints: {processed_output[0, 5:11, 0]}")

        # Check if shapes match
        if processed_output.shape == torch_ref_output.shape:
            diff = torch.abs(processed_output - torch_ref_output)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            logger.info(f"Max difference: {max_diff:.6f}")
            logger.info(f"Mean difference: {mean_diff:.6f}")

            if max_diff < 200.0:  # Allow reasonable numerical differences (coordinate scaling differences)
                logger.info("✅ TTNN post-processing matches PyTorch reference!")
            else:
                logger.warning("❌ TTNN post-processing differs from PyTorch reference")
        else:
            logger.error(f"❌ Shape mismatch: TTNN {processed_output.shape} vs PyTorch {torch_ref_output.shape}")

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

        # Debug: print raw coordinates
        if detections_count == 1:
            print(f"Detection {detections_count}: raw bbox=({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}), conf={conf[0, j]:.3f}")

        # Transform bbox to original image space
        x_orig = (x - pad_left) / scale
        y_orig = (y - pad_top) / scale
        w_orig = w / scale
        h_orig = h / scale

        if detections_count == 1:
            print(f"  Transformed bbox=({x_orig:.1f}, {y_orig:.1f}, {w_orig:.1f}, {h_orig:.1f})")

        x1 = int(max(0, x_orig - w_orig / 2))
        y1 = int(max(0, y_orig - h_orig / 2))
        x2 = int(min(orig_w, x_orig + w_orig / 2))
        y2 = int(min(orig_h, y_orig + h_orig / 2))

        if detections_count == 1:
            print(f"  Final bbox=({x1}, {y1}, {x2}, {y2})")

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

            # Debug: print first few keypoints before transformation
            if detections_count == 1 and kpt_idx < 5:
                print(f"TTNN Keypoint {kpt_idx}: decoded=({kx:.1f}, {ky:.1f}), visibility={kv:.3f}")

            # Transform to original image coordinates (coordinates are in 640x640 space, scale up)
            kx_final = (kx - pad_left) / scale
            ky_final = (ky - pad_top) / scale

            # Debug: print transformed coordinates
            if detections_count == 1 and kpt_idx < 5:
                print(f"  -> TTNN final=({kx_final:.1f}, {ky_final:.1f})")

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
