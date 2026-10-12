#!/bin/bash
# vrun3.sh <name> <port> <test ids, separated by |> [ENV=VAL ...]: like vrun2.sh, but the VCD goes through a FIFO straight into ext2.py
# (no VCD on disk); keeps log, CSVs, INIT dump and the extracted signals in /proj_sw/user_dev/nstojic/versim-runs/<name>/
name=$1; port=$2; id=$3; shift 3; IFS="|" read -ra IDS <<< "$id"
w=/tmp/vw_$name; c=$(git -C /tmp/mvt rev-parse HEAD)
[ -d $w ] || git -C /home/nstojic/tt-metal worktree add -q --detach $w $c
git -C $w checkout -q --detach $c
cd $w/tt_metal/tt-llk/tests && ln -sfn /tmp/mvt/tt_metal/tt-llk/tests/.venv .venv; ln -sfn /home/nstojic/tt-metal/tt_metal/tt-llk/tests/sfpi sfpi; ln -sfn ../../sfpi-info.sh sfpi-info.sh; ln -sfn ../../sfpi-version sfpi-version
cd python_tests && . ../.venv/bin/activate
A=/proj_sw/user_dev/nstojic/versim-runs/$name; mkdir -p $A
rm -f 1-1-core_dump.vcd versim_*.log; mkfifo 1-1-core_dump.vcd
python3 /tmp/ext2.py 1-1-core_dump.vcd $A/$name.txt 2>$A/ext.err &
EXT=$!
env CHIP_ARCH=wormhole LLK_HOME=$w/tt_metal/tt-llk RUNNER_TEMP=/tmp/build-v_$name \
  TT_METAL_SIMULATOR=${VSIM:-/tmp/vsim} NNG_SOCKET_ADDR=tcp://$(hostname):$port NNG_SOCKET_LOCAL_PORT=$port \
  LLK_SIM_BARRIER=1 LLK_SIM_TIMEOUT=30000 LLK_SIM_INPROC=1 LLK_SIM_LOG=1 LLK_EXP_INIT_DUMP=$A/init.jsonl "$@" \
  python -u -m pytest --run-simulator -p no:randomly -v $PYTEST_EXTRA "${IDS[@]}" > /tmp/v_$name.log 2>&1 < /dev/null
sleep 5; kill $EXT 2>/dev/null; rm -f 1-1-core_dump.vcd
cp /tmp/v_$name.log $A/; p=$(grep -o "Wrote run Parquet batch: [^ ]*" /tmp/v_$name.log | tail -1 | awk '{print $NF}'); [ -n "$p" ] && cp $(dirname $p)/*/*.csv $A/ 2>/dev/null
tail -c 20000 versim_*.log > $A/versim_tail.log 2>/dev/null; rm -f versim_*.log
echo done > /tmp/v_$name.done
