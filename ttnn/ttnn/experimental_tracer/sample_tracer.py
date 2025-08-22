# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import torch
from torchinfo import summary
from tracer_backend import trace_torch_model
from generate_pytorch_unittest_graph import (
    PytorchLayerUnitTestGraph,
    PytorchLayerUnitTestGraphConfig,
)
from generate_pytorch_graph import PytorchGraph, CompositePytorchGraph
from generate_pytorch_excel_graph import PytorchExcelGraph
from find_repeated_subgraphs import dump_graph_patterns

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
    "vit_large_patch14_reg4_dinov2",
    "vit_b_16_siglip2",
    "swin_transformer",
    "swin_transformer_v2",
    "open_vla",
]

allowed_dtypes = ["float32", "float64", "int32", "int64", "bfloat16"]


class CustomClass(torch.nn.Module):
    def forward(self, x):
        x = x - 1
        for i in range(6):
            x = x * 2
            x = x + 1
            x = x - 1
            x = x + 1
            x = x - 1
        res = x * 2
        return res


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
    elif args.model == "vit_large_patch14_reg4_dinov2":
        from timm.models import create_model

        try:
            torch_model = create_model(
                "vit_large_patch14_reg4_dinov2.lvd142m", pretrained=True, img_size=args.input_shape[0][2]
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to load vit_large_patch14_reg4_dinov2 model. Ensure timm==0.9.10 is installed and the model is available."
            )
    elif args.model == "vit_b_16_siglip2":
        from open_clip import (
            create_model_from_pretrained,
            get_tokenizer,
        )  # works on open-clip-torch >= 2.31.0, timm >= 1.0.15

        try:
            torch_model, _ = create_model_from_pretrained("hf-hub:timm/ViT-B-16-SigLIP2", pretrained=False)
        except Exception as e:
            raise RuntimeError(
                "Failed to load vit_b_16_siglip2 model. Ensure timm==0.9.10 is installed and the model is available."
            )
    elif args.model == "open_vla":
        from transformers import AutoModelForVision2Seq

        class OpenVLA(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.vla = AutoModelForVision2Seq.from_pretrained(
                    "openvla/openvla-7b", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                )
                self.vla.eval()

            def forward(self, pixel_values, attention_mask, input_ids):
                return self.vla.predict_action(
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    input_ids=input_ids,
                    unnorm_key="bridge_orig",
                    do_sample=False,
                )

        torch_model = OpenVLA()

    # torch_model = CustomClass()
    torch_model.eval()
    if not args.model == "sentence_bert" and not args.disable_torch_summary:
        print("Started torch summary: ")
        summary(torch_model, input_size=args.input_shape)
        print("Finished torch summary.\n\n\n")
    print("Started info tracing: ")
    operation_graph = trace_torch_model(
        torch_model,
        args.input_shape,
        input_dtypes=args.input_dtype,
        dump_visualization=True,
        save_original_tensors=not args.no_infer,
    )
    pytorch_graph = PytorchGraph(operation_graph)
    if not args.no_infer:
        pytorch_graph = CompositePytorchGraph(operation_graph)

    pytorch_graph.dump_to_python_file("graph.py", True)
    pytorch_excel_graph = PytorchExcelGraph(operation_graph)
    pytorch_excel_graph.dump_to_excel_file("graph.xlsx")
    graph = PytorchLayerUnitTestGraph(
        PytorchLayerUnitTestGraphConfig(
            operation_graph,
        )
    )
    graph.dump_to_python_file("test.py", True)
    dump_graph_patterns(operation_graph, "graph_patterns.py")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(vars(args))
