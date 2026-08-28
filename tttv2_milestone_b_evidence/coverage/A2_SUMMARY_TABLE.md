
## Summary by area, attempt 2

| Area | Host (attempt 1) | Device (attempt 2) | New findings |
| --- | --- | --- | --- |
| 1 Paged KV | PASS, 39 host tests | partial: page-table placement 3/3 PASS; late capacity **FAIL** (D-C4); paged-vs-contiguous stopped as a tautology; cross-slot never reached | **D-C4** |
| 2 Concat-32 | PASS, 34 host tests | **FAIL** on the only case reached, both models, for two different reasons (`a2_g10` Llama L1 address clash, `a2_g22` Qwen L1 *capacity* overflow). The per-length and per-active-batch cases never ran | — |
| 3 Prefix cache / chunked | PASS, 19 host tests | prefix-cached vs uncached PASS, 1 run per model. Chunked-vs-uncached and the mixed-slot batch never ran | — |
| 4 Device sampling | PASS, 26 host tests | **FAIL** on the only case reached, both models, for two different reasons (`a2_g11` Llama L1, `a2_g23` Qwen D-C5). Greedy-vs-host-argmax, padded vocabulary, seeded slots and heterogeneous controls never ran | **F-C1 superseded** |
| 5 Long context | PASS, 32 host tests (accounting) | PASS, 4K/32K/128K, both models, 1 run each | — |
| Repeat and cleanup | PASS, 12 host tests | Llama **FAIL** 2/2 (L1, deterministic); Qwen **PASS** 3/3 | — |
| Regression gates | PASS, boundaries clean | re-measured, unchanged | **D-C3** |
