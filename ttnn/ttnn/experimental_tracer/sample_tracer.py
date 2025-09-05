# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from datetime import datetime
from tracer_backend import trace_torch_model
from generate_pytorch_unittest_graph import (
    PytorchLayerUnitTestGraph,
    PytorchLayerUnitTestGraphConfig,
)
from generate_pytorch_graph import PytorchGraph
from generate_pytorch_excel_graph import PytorchExcelGraph
from torchinfo import summary

allowed_modes = [
    "yolov4",
    "yolov7",
    "yolov8x",
    "yolov8s",
    "yolov8s_world",
    "yolov11",
    "yolov12n",
    "yolov12x",
    "rtdetr",
    "resnet50",
    "mobilenetv2",
    "efficientnetb0",
    "efficientnetb3",
    "efficientnetb4",
    "sentence_bert",
    "vgg_unet",
    "ufld_v2",
    "segformer_classification",
    "vit_base_patch16_224",
    "vit_so400m_patch14_siglip_224",
    "swin_transformer",
    "swin_transformer_v2",
]

allowed_dtypes = ["float32", "float64", "int32", "int64"]


def generate_folder_name(model, input_shapes, input_dtypes):
    """Generate folder name based on model, input shapes, and dtypes with datetime."""
    # Format input shapes as strings
    shape_strs = []
    for shape in input_shapes:
        shape_str = "x".join(map(str, shape))
        shape_strs.append(shape_str)

    # Join all shapes with underscore
    shapes_part = "_".join(shape_strs)

    # Convert dtypes to strings and join with underscore
    dtypes_part = "_".join(str(dtype) for dtype in input_dtypes)

    # Get current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create folder name: model_shapes_dtypes_timestamp
    folder_name = f"{model}_{shapes_part}_{dtypes_part}_{timestamp}"

    return folder_name


def generate_file_name(prefix, model, input_shapes, input_dtypes):
    """Generate file name with prefix based on model and input shapes only."""
    # Format input shapes as strings
    shape_strs = []
    for shape in input_shapes:
        shape_str = "x".join(map(str, shape))
        shape_strs.append(shape_str)

    # Join all shapes with underscore
    shapes_part = "_".join(shape_strs)

    # Create file name: prefix_model_shapes (no dtypes or date)
    file_name = f"{prefix}_{model}_{shapes_part}"

    return file_name


def get_parser():
    """Creates and returns the argument parser."""
    parser = argparse.ArgumentParser(description="Trace YOLO model operations.")
    parser.add_argument(
        "--model",
        type=str,
        choices=allowed_modes,
        required=True,
        help=f"Model type to trace: {','.join(allowed_modes)}",
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs="+",
        action="append",
        required=True,
        help="List of input tensor shapes as space-separated integers (e.g., --input-shape 1 3 640 640 --input-shape 1 3 320 320)",
    )
    parser.add_argument(
        "--input-dtype",
        type=str,
        nargs="*",
        choices=allowed_dtypes,
        default=["float32"],
        help=f"Optional list of data types for the input tensors (default: float32). Allowed types: {', '.join(allowed_dtypes)}",
    )
    parser.add_argument(
        "--disable-torch-summary",
        action="store_true",
        help="Disable torch summary output. Useful for models that do not support torch summary.",
    )
    parser.add_argument(
        "--no-infer",
        action="store_true",
        help="Disable inference during tracing.",
    )
    return parser


def main(args_dict):
    """Main function to trace the model."""
    args = argparse.Namespace(**args_dict)

    if args.model == "yolov8s":
        from ultralytics import YOLO

        torch_model = YOLO("yolov8s.pt").model
    elif args.model == "yolov8x":
        from ultralytics import YOLO

        torch_model = YOLO("yolov8x.pt").model

    elif args.model == "yolov12n":
        from ultralytics import YOLO

        torch_model = YOLO("yolo12n.pt").model

    elif args.model == "yolov12x":
        from ultralytics import YOLO

        torch_model = YOLO("yolo12x.pt").model

    elif args.model == "yolov8s_world":
        from models.demos.yolov8s_world.reference.yolov8s_world import YOLOWorld

        torch_model = YOLOWorld()
    elif args.model == "yolov11":
        from models.demos.yolov11.reference.yolov11 import YoloV11

        torch_model = YoloV11()
    elif args.model == "yolov4":
        from models.demos.yolov4.reference.yolov4 import Yolov4

        torch_model = Yolov4()
    elif args.model == "yolov7":
        from models.demos.yolov7.reference.model import Yolov7_model

        torch_model = Yolov7_model()
    elif args.model == "rtdetr":
        from ultralytics import RTDETR

        torch_model = RTDETR("rtdetr-l.pt").model
    elif args.model == "resnet50":
        from torchvision.models import resnet50

        torch_model = resnet50(pretrained=True)
    elif args.model == "mobilenetv2":
        from torchvision.models import mobilenet_v2
        from models.demos.mobilenetv2.reference.mobilenetv2 import Mobilenetv2

        torch_model = Mobilenetv2()
    elif args.model == "efficientnetb0":
        from torchvision.models import efficientnet_b0

        torch_model = efficientnet_b0(pretrained=True)
    elif args.model == "efficientnetb3":
        from torchvision.models import efficientnet_b3

        torch_model = efficientnet_b3(pretrained=True)
    elif args.model == "efficientnetb4":
        from torchvision.models import efficientnet_b4

        torch_model = efficientnet_b4(pretrained=True)
    elif args.model == "sentence_bert":
        import transformers

        torch_model = transformers.AutoModel.from_pretrained("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
    elif args.model == "vgg_unet":
        from models.demos.vgg_unet.reference.vgg_unet import UNetVGG19

        torch_model = UNetVGG19()
    elif args.model == "ufld_v2":
        from models.demos.ufld_v2.reference.ufld_v2_model import TuSimple34

        torch_model = TuSimple34(input_height=args.input_shape[0][2], input_width=args.input_shape[0][3])
    elif args.model == "segformer_classification":
        from transformers import SegformerForImageClassification

        torch_model = SegformerForImageClassification.from_pretrained("nvidia/mit-b0")
    elif args.model == "vit_base_patch16_224":
        from transformers import ViTForImageClassification

        torch_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
    elif args.model == "swin_transformer":
        from transformers import SwinForImageClassification

        torch_model = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224")
    elif args.model == "swin_transformer_v2":
        from transformers import AutoModelForImageClassification

        # Use a pre-trained Swin Transformer V2 model
        torch_model = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window8-256")
    elif args.model == "vit_so400m_patch14_siglip_224":
        from timm.models import create_model

        try:
            torch_model = create_model("vit_so400m_patch14_siglip_224", pretrained=True)
        except Exception as e:
            raise RuntimeError(
                "Failed to load vit_so400m_patch14_siglip_224 model. Ensure timm==0.9.10 is installed and the model is available."
            )

    torch_model.eval()
    if not args.model == "sentence_bert" and not args.disable_torch_summary:
        print("Started torch summary: ")
        summary(torch_model, input_size=args.input_shape)
        print("Finished torch summary.\n\n\n")
    print("Started info tracing: ")

    # Generate folder and file names
    folder_name = generate_folder_name(args.model, args.input_shape, args.input_dtype)
    base_dir = "output_folder"
    output_dir = os.path.join(base_dir, folder_name)

    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)

    operation_graph = trace_torch_model(
        torch_model,
        args.input_shape,
        input_dtypes=args.input_dtype,
        dump_visualization=True,
        save_original_tensors=not args.no_infer,
    )

    # Generate files with new naming convention
    pytorch_graph = PytorchGraph(operation_graph)
    graph_filename = generate_file_name("graph", args.model, args.input_shape, args.input_dtype) + ".py"
    graph_filepath = os.path.join(output_dir, graph_filename)
    pytorch_graph.dump_to_python_file(graph_filepath, True)

    pytorch_excel_graph = PytorchExcelGraph(operation_graph)
    excel_filename = generate_file_name("graph", args.model, args.input_shape, args.input_dtype) + ".xlsx"
    excel_filepath = os.path.join(output_dir, excel_filename)
    pytorch_excel_graph.dump_to_excel_file(excel_filepath)

    graph = PytorchLayerUnitTestGraph(
        PytorchLayerUnitTestGraphConfig(
            operation_graph,
        )
    )
    test_filename = generate_file_name("test", args.model, args.input_shape, args.input_dtype) + ".py"
    test_filepath = os.path.join(output_dir, test_filename)
    graph.dump_to_python_file(test_filepath, True)

    # Handle the operation_graph_viz.json file
    viz_filename = generate_file_name("operation_graph_viz", args.model, args.input_shape, args.input_dtype) + ".json"
    viz_filepath = os.path.join(output_dir, viz_filename)

    # Check if operation_graph_viz.json was generated and move it
    if os.path.exists("operation_graph_viz.json"):
        os.rename("operation_graph_viz.json", viz_filepath)
        viz_generated = True
    else:
        viz_generated = False

    print(f"Files generated in: {output_dir}")
    print(f"- {graph_filename}")
    print(f"- {excel_filename}")
    print(f"- {test_filename}")
    if viz_generated:
        print(f"- {viz_filename}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(vars(args))
