#!/bin/bash
# usage: run.sh <label> <node...>
#   node = a focus id (res3_{1,2,3}_conv2, semantic0 ...), or H:<height id>, I:<interleaved id>, B:<block id> of
#   tests/.../test_conv2d_3x3_perf.py. Runs against the op copy picked by HWR_COPY (hwr_copy.py; default: the
#   candidate op_copy/), with the HWR_* variant switches passed through. Profiles in one process, prints DEVICE KERNEL
#   DURATION per op (us), keeps the device CSV in runs/ (gzipped) and appends to results.jsonl.
cd /localdev/pmilojevic/workspaces/opgen
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate
# Private profiler artifacts and JIT cache (persistent across this experiment's runs, so the zone-location log matches
# the cached kernels): the device profiler throws on a 16-bit zone-hash collision with ANY location in its log, and the
# shared log grows with every sibling's op copy.
export TT_METAL_PROFILER_DIR=/tmp/hwr/prof TT_METAL_CACHE=/tmp/hwr/cache
mkdir -p $TT_METAL_PROFILER_DIR $TT_METAL_CACHE
E=ttnn/ttnn/operations/conv2d_3x3/perf_experiments/height_weight_rate
label=$1; shift
nodes=()
for id in "$@"; do
    case $id in
        H:*) t=test_conv2d_3x3_height_sharded_perf; id=${id#H:} ;;
        I:*) t=test_conv2d_3x3_perf; id=${id#I:} ;;
        B:*) t=test_conv2d_3x3_block_sharded_perf; id=${id#B:} ;;
        *) t=test_conv2d_3x3_perf_focus ;;
    esac
    nodes+=("$E/test_hwr_op.py::$t[$id]")
done
env | grep "^HWR_" | sort | tr '\n' ' ' > $E/runs/env_$label.txt
d=$(python3 -c "import sys; sys.path.insert(0, '$E'); import hwr_copy; print(hwr_copy.copy_dir())")
chk=$(python3 $E/zone_hash_check.py $d | head -1)
case $chk in *" 0 collisions"*) ;; *) echo "zone hash collision in $d: $chk"; exit 1 ;; esac
out=$(timeout 2400 scripts/run_safe_pytest.sh --profile --run-all --import-mode=prepend "${nodes[@]}" 2>&1)
echo "$out" | grep -E " passed| failed|error:|SAFE_PYTEST_RESULT|HANG|triage|AssertionError|PCC|Error" | tail -8
csv=$(ls -t $TT_METAL_PROFILER_DIR/reports/*/ops_perf_results*.csv 2>/dev/null | head -1)
[ -n "$csv" ] && [ "$csv" -ot $E/runs/env_$label.txt ] && csv=""
echo "$out" > $E/runs/log_$label.txt
if [ -n "$csv" ]; then cp "$(dirname $csv)/profile_log_device.csv" $E/runs/dev_$label.csv && gzip -f $E/runs/dev_$label.csv; fi
python3 - "$csv" "$label" "$E" "$@" <<'PY'
import sys, csv, json, os
path, label, E, ids = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
if not path:
    print("no profiler csv"); sys.exit()
rows = list(csv.DictReader(open(path)))
v = [round(float(r["DEVICE KERNEL DURATION [ns]"]) / 1000, 3) for r in rows if r.get("DEVICE KERNEL DURATION [ns]")]
env = {k: v for k, v in os.environ.items() if k.startswith("HWR_")}
print("DKD_us", label, env, v)
with open(f"{E}/results.jsonl", "a") as f:
    f.write(json.dumps({"label": label, "env": env, "ids": ids, "us": v}) + "\n")
PY
