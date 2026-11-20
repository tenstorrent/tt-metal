"""sha256 of conv2d_3x3's raw output bytes on every Panoptic LOOSE case (run under scripts/tt-probe.sh).

PANOPTIC_HASH_OUT=<json path>. Two builds giving the same json produced bit-identical outputs."""

import hashlib
import json
import os


def main():
    import torch
    import ttnn

    from eval.golden_tests.conv2d_3x3 import feature_spec
    from eval.golden_tests.conv2d_3x3.helpers import (
        build_input_memory_config,
        create_ttnn_input_tensor,
        make_activation,
        make_bias,
        make_compute_kernel_config,
        make_weight,
    )
    from ttnn.operations.conv2d_3x3 import conv2d_3x3, prepare_conv2d_bias, prepare_conv2d_weights

    device = ttnn.open_device(device_id=0)
    hashes = {}
    try:
        for case in feature_spec.LOOSE_CASES:
            extras = case["extras"]
            if "FLUX" in extras["layer"]:
                continue
            (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
            torch.manual_seed(0)
            c_in_real = extras.get("c_in_real")
            w = make_weight(c_out, c_in, dtype=torch.bfloat16, c_in_real=c_in_real)
            b = make_bias(c_out, dtype=torch.bfloat16) if case["bias_mode"] == "with_bias" else None
            x = make_activation(H, W, c_in, dtype=torch.bfloat16, c_in_real=c_in_real)
            mc = build_input_memory_config(
                (1, 1, H * W, c_in),
                case["memory_layout"],
                layout=case["layout"],
                dtype=case["dtype"],
                device=device,
                extras=extras,
            )
            prep = dict(
                stride=stride,
                padding=padding,
                dilation=dilation,
                input_dtype=case["dtype"],
                input_layout=case["layout"],
                weight_dtype=case["weight_dtype"],
                input_memory_config=mc,
                device=device,
            )
            tw = prepare_conv2d_weights(w, **prep)
            tb = prepare_conv2d_bias(b, **prep) if b is not None else None
            tx = create_ttnn_input_tensor(x, device, dtype=case["dtype"], layout=case["layout"], memory_config=mc)
            out = conv2d_3x3(
                tx,
                tw,
                input_height=H,
                input_width=W,
                bias=tb,
                padding=padding,
                stride=stride,
                dilation=dilation,
                activation=None if case["activation"] == "none" else case["activation"],
                compute_kernel_config=make_compute_kernel_config(
                    case["fp32_dest_acc_en"], extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
                ),
            )
            raw = ttnn.to_torch(out).contiguous().view(torch.int16).numpy().tobytes()
            hashes[extras["layer"]] = hashlib.sha256(raw).hexdigest()
            print("HASH", extras["layer"], hashes[extras["layer"]][:16], flush=True)
            for t in (out, tx, tw) + ((tb,) if tb is not None else ()):
                ttnn.deallocate(t)
    finally:
        ttnn.close_device(device)
    with open(os.environ["PANOPTIC_HASH_OUT"], "w") as f:
        json.dump(hashes, f, indent=1, sort_keys=True)
    print("CASES", len(hashes))


if __name__ == "__main__":
    main()
