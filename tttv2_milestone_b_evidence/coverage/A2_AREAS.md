
## Area by area, on silicon

Each row names the log. `runs` is how many fresh processes the claim got; a claim
with one run is *observed*, not qualified, and says so.

### Area 1 — paged KV

@@AREA1@@

### Area 2 — concat-32 physical prefill

@@AREA2@@

### Area 3 — prefix-cached and chunked prefill

@@AREA3@@

### Area 4 — device sampling

@@AREA4@@

### Area 5 — long context

@@AREA5@@

Attempt 1's capacity accounting for these three geometries (blocks per user,
pool size, KV bytes per device, RoPE table size, chunk count) is in area 5 above
this section and was not re-derived; what attempt 2 adds is whether each one
actually runs.

### Repeat and cleanup

@@REPEAT@@
