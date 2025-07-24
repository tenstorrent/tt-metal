import argparse
from tracer_backend import trace_torch_model
from generate_pytorch_unittest_graph import (
    PytorchLayerUnitTestGraph,
    ConvolutionUnittest,
    AddmUnittest,
    Maxpool2dUnittest,
    AddmCombiner,
    PytorchLayerUnitTestGraphConfig,
    ConvolutionCombiner,
    Maxpool2dCombiner,
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
    "swin_transformer",
    "swin_transformer_v2",
]

allowed_dtypes = ["float32", "float64", "int32", "int64"]


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

    torch_model.eval()
    if not args.model == "sentence_bert":
        print("Started torch summary: ")
        summary(torch_model, input_size=args.input_shape)
        print("Finished torch summary.\n\n\n")
    print("Started info tracing: ")
    operation_graph = trace_torch_model(
        torch_model, args.input_shape, input_dtypes=args.input_dtype, dump_visualization=True
    )
    pytorch_graph = PytorchGraph(operation_graph)
    pytorch_graph.dump_to_python_file("graph.py", True)
    pytorch_excel_graph = PytorchExcelGraph(operation_graph)
    pytorch_excel_graph.dump_to_excel_file("graph.xlsx")
    graph = PytorchLayerUnitTestGraph(
        PytorchLayerUnitTestGraphConfig(
            operation_graph,
            [ConvolutionUnittest, AddmUnittest, Maxpool2dUnittest],
            {
                AddmUnittest: AddmCombiner,
                ConvolutionUnittest: ConvolutionCombiner,
                Maxpool2dUnittest: Maxpool2dCombiner,
            },
        )
    )
    graph.dump_to_python_file("test.py", True)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(vars(args))
