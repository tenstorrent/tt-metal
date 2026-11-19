#!/usr/bin/env bash
# Regenerate the scratch copies this experiment ran with (deleted to keep the dir small):
#   op_abl/  op_base + ../ablation_r13/ablation_hooks.patch (the abl_* variants of test_ov.py; OV_NOCHECK=1)
#   unit/    the unit dir's tests with the op imports pointed at grad/ (the graduation's correctness run:
#            scripts/run_safe_pytest.sh --run-all <dir>/unit/<file> --import-mode=prepend)
#   *_nz     zone-stripped twins for unzoned numbers: mk_nz.sh
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../../../.." && pwd)"
rm -rf "$HERE/op_abl" && cp -r "$HERE/op_base" "$HERE/op_abl"
sed -e 's#a/ttnn/ttnn/operations/conv2d_3x3/#a/#g; s#b/ttnn/ttnn/operations/conv2d_3x3/#b/#g' \
    "$HERE/../ablation_r13/ablation_hooks.patch" | (cd "$HERE/op_abl" && patch -p1 -s)
mkdir -p "$HERE/unit" && touch "$HERE/unit/__init__.py"
cp "$REPO/tests/ttnn/unit_tests/operations/conv2d_3x3/conftest.py" "$HERE/unit/"
for f in "$REPO"/tests/ttnn/unit_tests/operations/conv2d_3x3/test_*.py; do
    b=$(basename "$f")
    [[ $b == test_conv2d_3x3_perf.py ]] && continue
    sed -E 's#ttnn\.operations\.conv2d_3x3([^.a-zA-Z_]|\.conv2d_3x3|\.conv2d_3x3_program_descriptor)#ttnn.operations.conv2d_3x3.perf_experiments.s2_build_walk_overlap.grad\1#g' \
        "$f" > "$HERE/unit/$b"
done
