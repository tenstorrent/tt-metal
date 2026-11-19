#!/usr/bin/env bash
# Ablation-hooked twins (op_base_abl, op_copy_abl) of op_base / op_copy: ../ablation_r12/ablation_hooks.patch applied,
# the descriptor hunks the copy's changes moved re-applied by hand, and two zone lines shifted so no zone location
# hashes onto another (zone_hash_check.py). Run with CONV2D_3X3_ABLATE=<list> KWP_NOCHECK=1, variants *_abl.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
for v in op_base op_copy; do
    rm -rf "$HERE/${v}_abl"
    cp -r "$HERE/$v" "$HERE/${v}_abl"
    rm -rf "$HERE/${v}_abl/__pycache__"
    (cd "$HERE/${v}_abl" && patch -p5 --no-backup-if-mismatch -s < "$HERE/../ablation_r12/ablation_hooks.patch" || true)
    rm -f "$HERE/${v}_abl/"*.rej
done
python3 - "$HERE/op_copy_abl/conv2d_3x3_program_descriptor.py" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
for a, b in [
    ("_async_bias_defines + slot_ack_defines,", "_async_bias_defines + slot_ack_defines + _abl_defines,"),
    ("                else []),", "                else []) + _abl_defines,"),
    ("                defines=slot_ack_defines,", "                defines=slot_ack_defines + _abl_defines,"),
]:
    if b not in s:
        s = s.replace(a, b)
s = s.replace(
    """                defines=[("KWF_INTERLEAVE_NG", str(g.tensor_n_tiles // planes.n_p))]
                if planes is not None and planes.stream
                else [],""",
    """                defines=([("KWF_INTERLEAVE_NG", str(g.tensor_n_tiles // planes.n_p))]
                if planes is not None and planes.stream
                else []) + _abl_defines,""",
)
open(p, "w").write(s)
PY
sed -i '/MaybeDeviceZoneScope("reader_center_wait");/i\        // (zone-hash shift)' "$HERE/op_base_abl/kernels/conv2d_3x3_reader.cpp"
sed -i '/MaybeDeviceZoneScope("compute_kwf_final");/i\                // (zone-hash shift)' "$HERE/op_base_abl/kernels/conv2d_3x3_kw_fold.hpp"
python3 "$HERE/zone_hash_check.py" | tail -1
