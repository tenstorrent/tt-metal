# Single-chip zone profile of one domain case (ZS_T / ZS_C / ZS_MODE / ZS_NORM); MHC_WS_VARIANT=ws|base.
# TT_METAL_DEVICE_PROFILER=1 MHC_POST_PERF_ZONES=reader scripts/run_safe_pytest.sh <this> -s --import-mode=prepend
import importlib, os, sys
from pathlib import Path

import ttnn

sys.path.insert(0, str(Path(__file__).parent))
_m = importlib.import_module("mhc_post_" + os.environ.get("MHC_WS_VARIANT", "ws"))


def test_zones_single(device):
    from eval.golden_tests.mhc_post.feature_spec import _case
    from eval.golden_tests.mhc_post.helpers import call_mhc_post, make_case

    T, C = int(os.environ.get("ZS_T", "160")), int(os.environ.get("ZS_C", "5120"))
    mode, norm = os.environ.get("ZS_MODE", "collapse"), os.environ.get("ZS_NORM", "on")
    g = device.compute_with_storage_grid_size()
    print("grid", g.x, g.y)
    case = make_case(
        _case((1, 1, T, 4 * C)),
        mode=mode,
        coef_source="packed",
        dtype=ttnn.bfloat16,
        sublayer_dtype=ttnn.bfloat16,
        norm=norm,
    )
    for _ in range(2):
        call_mhc_post(
            case,
            mode=mode,
            coef_source="packed",
            dtype=ttnn.bfloat16,
            sublayer_dtype=ttnn.bfloat16,
            device=device,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            op=_m.mhc_post,
        )
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
