#!/bin/bash
# archive finished Versim runs of the y* batch: log, CSVs, ext.py signals, zstd VCD to /proj_sw; then free /tmp
while true; do
  for d in /tmp/v_y*.done; do
    n=$(basename $d .done); n=${n#v_}; [ -f /tmp/varch_$n.ok ] && continue
    A=/proj_sw/user_dev/nstojic/versim-runs/$n; mkdir -p $A
    cp /tmp/v_$n.log /tmp/v_$n.tl $A/ 2>/dev/null
    p=$(grep -o "Wrote run Parquet batch: [^ ]*" /tmp/v_$n.log | tail -1 | awk '{print $NF}'); [ -n "$p" ] && cp $(dirname $p)/*/*.csv $A/ 2>/dev/null
    if [ -f /tmp/vcd/$n.vcd ]; then python3 /tmp/ext.py /tmp/vcd/$n.vcd $A/$n.txt 2>$A/ext.err; zstd -q -T4 /tmp/vcd/$n.vcd -o $A/$n.vcd.zst && rm -f /tmp/vcd/$n.vcd; fi
    rm -rf /tmp/vw_$n/tt_metal/tt-llk/tests/python_tests/1-1-core_dump.vcd
    touch /tmp/varch_$n.ok; echo "archived $n $(date -u +%H:%M)"
  done
  [ -f /tmp/v2simy.done ] && ! ls /tmp/vcd/y*.vcd >/dev/null 2>&1 && break
  sleep 60
done
