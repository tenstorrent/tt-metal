from tt_models.EfficientNet.demo.demo_utils import run_gs_demo
from tt_models.EfficientNet.tt.efficientnet_model import efficientnet_lite0


def test_gs_demo_lite0():
    run_gs_demo(efficientnet_lite0)
