"""w_share_early single-chip domain sweep (run under scripts/run_safe_pytest.sh --profile --import-mode=prepend;
MHC_WS_VARIANT=ws|base): the W paths (stats / collapse / collapse + norm) over T (-> R = 5, 10, 20, ...) and C.
Read with domain_report.py <ops_perf_results.csv> (REPS calls per case, test order)."""
import importlib, os, sys
from pathlib import Path

import pytest
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
mhc_post = importlib.import_module("mhc_post_" + os.environ.get("MHC_WS_VARIANT", "ws")).mhc_post
N = 4
REPS = 2
CASES = [
    (160, 5120, "stats", "off", "packed"),
    (160, 5120, "collapse", "on", "packed"),
    (320, 5120, "collapse", "on", "packed"),
    (640, 5120, "stats", "off", "packed"),
    (640, 1792, "collapse", "on", "packed"),
    (1280, 5120, "stats", "off", "packed"),
    (1280, 5120, "collapse", "on", "packed"),
    (1280, 5120, "collapse", "on", "separate"),
    (2048, 5120, "collapse", "on", "packed"),
    (160, 1792, "collapse", "off", "packed"),
]


@pytest.mark.parametrize("T,C,mode,norm,coef_source", CASES)
def test_domain(device, T, C, mode, norm, coef_source):
    from eval.golden_tests.mhc_post.feature_spec import _case
    from eval.golden_tests.mhc_post.helpers import call_mhc_post, make_case

    case = make_case(
        _case((1, 1, T, N * C)),
        mode=mode,
        coef_source=coef_source,
        dtype=ttnn.bfloat16,
        sublayer_dtype=ttnn.bfloat16,
        norm=norm,
    )
    for _ in range(REPS):
        call_mhc_post(
            case,
            mode=mode,
            coef_source=coef_source,
            dtype=ttnn.bfloat16,
            sublayer_dtype=ttnn.bfloat16,
            device=device,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            op=mhc_post,
        )
    ttnn.synchronize_device(device)
