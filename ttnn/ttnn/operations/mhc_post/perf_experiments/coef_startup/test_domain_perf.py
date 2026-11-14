"""Single-chip domain sweep (run under scripts/run_safe_pytest.sh --profile; MHC_CS_VARIANT=cs|base): DS-V4.1 call
forms (bf16, packed, HiFi2) at T160 / T1280 C5120, separate-coefficient collapse + norm, plain fp32."""
import importlib, os, sys
from pathlib import Path

import pytest
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
mhc_post = importlib.import_module(
    "mhc_post_cs" if os.environ.get("MHC_CS_VARIANT", "cs") == "cs" else "mhc_post_base"
).mhc_post
N = 4


def test_plain_fp32(device):
    import torch

    torch.manual_seed(0)
    dev = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    f, x = dev(torch.randn(1, 1, 160, 5120)), dev(torch.randn(1, 1, 160, N * 5120))
    post, comb = dev(torch.rand(1, 1, 160, N)), dev(torch.rand(1, 1, 160, N * N))
    for _ in range(2):
        mhc_post(f, x, post, comb)
    ttnn.synchronize_device(device)


@pytest.mark.parametrize("coef_source", ["packed", "separate"])
@pytest.mark.parametrize("mode,norm", [("plain", "off"), ("stats", "off"), ("collapse", "off"), ("collapse", "on")])
@pytest.mark.parametrize("T", [160, 1280])
def test_ds41(device, T, mode, norm, coef_source):
    from eval.golden_tests.mhc_post.feature_spec import _case
    from eval.golden_tests.mhc_post.helpers import call_mhc_post, make_case

    case = make_case(
        _case((1, 1, T, N * 5120)),
        mode=mode,
        coef_source=coef_source,
        dtype=ttnn.bfloat16,
        sublayer_dtype=ttnn.bfloat16,
        norm=norm,
    )
    for _ in range(2):
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
