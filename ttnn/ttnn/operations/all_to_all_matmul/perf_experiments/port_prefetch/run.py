# port_prefetch driver: the tail_chain runner (run_tail_chain.py) pointed at this experiment's base/ and cand/ copies.
# echo "import os; os.environ.update(VARS='base||;cand||pdepth=all,pinfl=2'); exec(open('<this file>').read())" | scripts/tt-probe.sh all_to_all_matmul
import os

os.environ["TAIL_CHAIN_DIR"] = "ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/port_prefetch"
os.environ.setdefault("OUTDIR", os.environ["TAIL_CHAIN_DIR"] + "/out")
os.environ["TAIL_CHAIN_RUN"] = "1"
exec(open("ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/tail_chain/run_tail_chain.py").read())
