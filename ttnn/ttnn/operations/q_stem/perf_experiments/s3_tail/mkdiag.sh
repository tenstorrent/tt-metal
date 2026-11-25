#!/bin/bash
# q_stem_vdiag = q_stem_v + per-row zones (reader hand-over, compute row, writer row); zone runs only
set -e
cd "$(dirname "$0")"
rm -rf q_stem_vdiag; cp -r q_stem_v q_stem_vdiag; rm -rf q_stem_vdiag/__pycache__
sed -i 's#s3_tail/q_stem_v/kernels#s3_tail/q_stem_vdiag/kernels#' q_stem_vdiag/kernels/*
python3 - <<'PY'
import re
d='q_stem_vdiag/kernels/'
p=d+'q_stem_reader.cpp'; s=open(p).read()
old='''        const uint32_t c = order.index(u, t);'''
assert old in s; s=s.replace(old,'''        { DeviceZoneScopedN("RW_ROW"); }
'''+old); open(p,'w').write(s)
p=d+'q_stem_compute.cpp'; s=open(p).read()
old='''        for (uint32_t i = 0; i < order.total(); ++i) {
            cb_wait_front(cb_desc, 1);'''
assert old in s; s=s.replace(old,'''        for (uint32_t i = 0; i < order.total(); ++i) {
            DeviceZoneScopedN("CR_ROW");
            cb_wait_front(cb_desc, 1);'''); open(p,'w').write(s)
p=d+'q_stem_writer.cpp'; s=open(p).read()
old='''        store_row(u, t);
    }'''
assert old in s; s=s.replace(old,'''        DeviceZoneScopedN("WR_ROW");
        store_row(u, t);
    }'''); open(p,'w').write(s)
PY
