import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from loguru import logger
import tt_lib

from tt_models.utility_functions import torch2tt_tensor, tt2torch_tensor
from tt_models.mnist.tt.mnist_model import mnist_model


def test_mnist_inference():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    # Data preprocessing/loading
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=None, download=True
    )

    # Load model
    tt_model, _ = mnist_model(device)

    with torch.no_grad():
        input_img, _ = test_dataset[0]

        input_img_path = str(ROOT / "input_img.jpg")
        input_img.save(input_img_path)
        logger.info(f"Input image saved to {input_img_path}")

        test_input = transforms.ToTensor()(input_img)
        test_input = torch2tt_tensor(
            test_input, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )

        tt_output = tt_model(test_input)
        tt_output = tt2torch_tensor(tt_output)
        tt_output = tt_output.squeeze()
        tt_lib.device.CloseDevice(device)

    logger.info(f"Tt prediction: {tt_output.topk(10).indices[0]}")
