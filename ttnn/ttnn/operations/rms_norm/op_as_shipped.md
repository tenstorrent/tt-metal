# rms_norm — as shipped

> **This artifact is derived from the op's CODE, not from its planning artifacts.** Every number below was read
> from source or observed in a built `ProgramDescriptor`. Nothing here is quoted from `op_design.md`,
> `verification_report.md` or `changelog.md` as fact — those appear only in §10, as claims, where they
> contradict what shipped.

## §0 Provenance

| field | value |
|---|---|
| op | rms_norm |
| op_dir | ttnn/ttnn/operations/rms_norm |
| git_sha | 09305879fec |
| analyzed_at | 2026-09-02 |
| mode | hybrid |
| device | Blackhole p150b, 11x10 = 110-core compute grid, 1531904 B worker L1 unreserved |
| descriptor builds | 25 |
| golden suite | .claude/eval/golden_tests/rms_norm/feature_spec.py |
| design record location | the planner's module docstring (D1-D15, D20-D28) PLUS kernel-site comments (D16-D19, D26). NOT one place -- see design_record[].source. |
| planning artifacts present | op_design.md, op_requirements.md, verification_report.md, changelog.md |
| planning artifacts ABSENT | l1_ledger.md |
| machine-readable companion | op_as_shipped.json |

> **Corrected 2026-09-02, post-audit.** Post-audit correction pass. Two independent fidelity audits were run against this artifact (interface/kernel scope and numeric scope) and their confirmed findings applied. Fixed: 3 semantic errors (compute_kernel_config's stage set, cb_normalized's live set, the dest_fold predicate's missing kernel term), 4 numeric/boundary errors (the DEST_ACC_SQUARE_MAX_WT bracket, the flat-ring bound, the ROW_MAJOR stick CBs' page-format set, two case counts in audit prose), 7 omissions (a 17th bypass, keyword-only-ness, input_tensor's compute:C10, SCHEME_ROWS' HEIGHT-shard fallback, a PASS_B_BLK knob + symbol row, and three regimes promoted from prose to their own families), and every file:line citation in the artifact, each re-derived by reading the cited line back.
>
> provenance.source_digest was NOT re-stamped: no source file changed. All 8 sha256 prefixes still reproduce, so this artifact describes the same revision it always did.  G8 (citation resolution) was added to .claude/skills/analyze-op/SKILL.md as a result of this pass.

**Source digest** — hash these files to know in one command whether this analysis still describes the code.

| file | sha256:12 |
|---|---|
| rms_norm.py | sha256:390998746915 |
| __init__.py | sha256:e163962865d9 |
| rms_norm_program_descriptor.py | sha256:d0c3d2a59c1e |
| kernels/rms_norm_reader.cpp | sha256:70aa6140f8ae |
| kernels/rms_norm_compute.cpp | sha256:3f2bc5ee891a |
| kernels/rms_norm_writer.cpp | sha256:203f03f41bae |
| kernels/perf_instrumentation.hpp | sha256:13071adecb59 |
| .claude/eval/golden_tests/rms_norm/feature_spec.py | sha256:3b3ca051dbf3 |

**Entry points**

| role | symbol |
|---|---|
| op | ttnn/ttnn/operations/rms_norm/rms_norm.py::rms_norm |
| validate | ttnn/ttnn/operations/rms_norm/rms_norm.py::validate |
| planner | ttnn/ttnn/operations/rms_norm/rms_norm_program_descriptor.py::create_program_descriptor |
| solver | ttnn/ttnn/operations/rms_norm/rms_norm_program_descriptor.py::create_program_descriptor._solve_blocking (line 1883) |
| kernels | ttnn/ttnn/operations/rms_norm/kernels/rms_norm_reader.cpp, ttnn/ttnn/operations/rms_norm/kernels/rms_norm_compute.cpp, ttnn/ttnn/operations/rms_norm/kernels/rms_norm_writer.cpp |

## Hard gates

Run against the extracted facts before writing, and re-run against what was written.

| gate | name | result | detail |
|---|---|---|---|
| G1 | CB coverage is the union over regimes | pass | ledger indices {0..18} == the static CB_* name table (19 entries, rms_norm_program_descriptor.py:978-1010) == the union of every buffer index carrying a per-case evaluation across all 25 descriptor builds; names join 1:1. No conditional CB is missing. |
| G2a | signature parity | pass | all 6 public parameters of rms_norm() appear in section 2, checked against the AST of the def (input_tensor, gamma, epsilon, compute_kernel_config, memory_config, program_config). |
| G2b | every parameter has a non-empty enters_at | **FAIL** | program_config has an EMPTY enters_at. This is a defect in the OP, not an omission in the analysis: the parameter is accepted by rms_norm() and validate() and read by neither. Recorded in signature[].note and contradictions[5]. Re-run after the post-audit correction pass: still FAIL, and honestly so. |
| G2c | keyword-only-ness recorded and correct | pass | the AST says gamma, epsilon, compute_kernel_config, memory_config and program_config are keyword-only (rms_norm.py:259 places `*,` immediately after input_tensor); signature[].keyword_only agrees for all 6 parameters. ADDED in the post-audit correction pass -- the first version recorded keyword-only-ness nowhere, so a consumer generating bindings from the table would have emitted a wrong signature. |
| G3 | rectangle parity | pass | SUPPORTED declares 9 axes; validate() builds exactly those 9 (7 in its axes dict literal + 2 from INPUT_TAGGERS), checked against the AST; every axis names whether its value comes from a tensor property, a tagger, or a kwarg. |
| G4 | every size is an expression or an honest solved | pass | 19 capacities, 19 page sizes and the footprint are all expressions; every symbol named by a capacity appears in the 13-entry symbol table with a bound and the predicate establishing it; six symbols are marked as produced by a bounded search (_solve_blocking at rms_norm_program_descriptor.py:1883, _combine_tree_arity at :1013) and carry a 25-case evaluation table. No bare numbers. The post-audit re-run caught one gap the first pass missed: cb_scaler's capacity names kernel_partial_w, which had no symbol row; it now has one, with its bound and its 25-case evaluation confirmed against compute ct[4]. |
| G5 | optional-input cost is a number with evidence | pass | gamma: absent_bytes = 0, evidence = descriptor, from two ablation pairs (c01/c02 and c19/c25). Nothing is still sized for it when absent. |
| G6 | no fact is 'claimed' | pass | zero occurrences of evidence == 'claimed' anywhere in the artifact. Planning artifacts appear only in contradictions[] and claimed_not_built[]. |
| G7 | the two files agree, and both are stamped | pass | op_as_shipped.md is GENERATED from op_as_shipped.json, so the invariant is structural rather than checked. Verified anyway after regeneration: all 8 source digests, the git sha and the mode appear in the .md header, and the .md contains no 4-or-more-digit number absent from the .json. |
| G8 | every file:line citation resolves to the construct it cites | pass | 265 file:line citations across the artifact were re-derived by reading the cited line(s) back from source. Every one now lands on the construct it names -- a definition, a predicate, an allocation call, a CT/RT arg list, an instrumentation zone, or (for the 12 that cite prose deliberately: the design record, the quoted kernel head comments, a claimed-not-built rationale and the compute kernel's regime doc block) the comment being quoted. Mechanically re-checkable: extract every `<file>:<line>` from the JSON and assert the line range is in bounds and contains the named construct. This gate was ADDED to the skill as a result of the post-audit correction pass -- the previous version of this artifact had every citation present and none resolving: all 14 buffer, 19 knob and 9 regime source lines missed by -22 to +44, and every rms_norm.py reference missed by -9 to +8. |

## §1 Registry rectangle

### SUPPORTED — 9 axes

| axis | values | obtained from | source |
|---|---|---|---|
| dtype | ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b | tensor property (input_tensor.dtype) | rms_norm.py:103 |
| fp32_dest_acc_en | True, False | kwarg (compute_kernel_config.fp32_dest_acc_en, defaulted to True by default_compute_kernel_config()) | rms_norm.py:105 |
| layout | ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT | tensor property (input_tensor.layout) | rms_norm.py:106 |
| alignment | tile_aligned, w_non_aligned, h_non_aligned | tagger (tag_alignment) | rms_norm.py:107 |
| rank | 2, 3, 4 | tagger (tag_rank) | rms_norm.py:108 |
| gamma_mode | gamma, no_gamma | kwarg presence (gamma is not None) | rms_norm.py:109 |
| gamma_dtype | ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, none | tensor property (gamma.dtype), or the sentinel 'none' | rms_norm.py:110 |
| gamma_layout | ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, none | tensor property (gamma.layout), or the sentinel 'none' | rms_norm.py:111 |
| memory_layout | INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED | tensor property (input_tensor.memory_config().memory_layout) | rms_norm.py:125-130 |

### EXCLUSIONS

| cell | why | raises | source |
|---|---|---|---|
| {"dtype": "ttnn.float32", "fp32_dest_acc_en": false} | a float32 CALLER asked for fp32 and would get a 16-bit accumulator for the sum of squares; refused natively rather than answered wrongly | ExcludedCell | rms_norm.py:147-149 |

### INPUT_TAGGERS — stated as predicates

| axis | predicate | source |
|---|---|---|
| alignment | shape[-1] % 32 != 0 -> 'w_non_aligned'; elif shape[-2] % 32 != 0 -> 'h_non_aligned'; else 'tile_aligned' | rms_norm.py:63-75 |
| rank | len(shape) | rms_norm.py:78-79 |

- **alignment** — w_non_aligned DOMINATES h_non_aligned: a non-tile-aligned W drives the masked-reduce path (partial scaler / 0-1 mask + logical-W divisor), which is a different kernel path from H row padding.

### PROPERTIES

| property | value | source | confirmed by |
|---|---|---|---|
| multi_core | true | declared | descriptor: 110 cores at c10/c11/c12, 99 at c23 |
| bounded_cb | true | declared | descriptor: every CB page count is a function of the solved knobs; see buffers[] and footprint |
| math_fidelity | ["LoFi", "HiFi2", "HiFi3", "HiFi4"] | declared | never gated in validate(); compute_kernel_config is passed through unmodified to the KernelDescriptor |

### What `validate()` actually gates on

Not the same list as `SUPPORTED`. Three structural pre-checks run **before** the axis loop, and one gate
applies to an axis value that does not come from the input tensor at all.

| gate | raises | order | why | source |
|---|---|---|---|---|
| len(input_tensor.shape) >= 2 | ValueError | structural, BEFORE the axis loop | the reduction needs a row axis and a width axis, and the alignment tagger indexes shape[-2] | rms_norm.py:205-209 |
| gamma is None or gamma.shape[-1] == input_tensor.shape[-1] | ValueError | structural | gamma is a per-channel scale over the reduced axis | rms_norm.py:210-213 |
| epsilon > 0.0 | ValueError | structural | epsilon guards rsqrt(0) | rms_norm.py:214-215 |
| axes[axis] in SUPPORTED[axis], for all 9 axes | UnsupportedAxisValue | the axis loop |  | rms_norm.py:233-235 |
| no EXCLUSIONS cell matches | ExcludedCell | after the axis loop |  | rms_norm.py:238-240 |
| memory_config is None or memory_config.memory_layout in SUPPORTED['memory_layout'] | UnsupportedAxisValue | after EXCLUSIONS; this is the one gate on an axis value NOT derived from the input tensor | the requested OUTPUT placement must also be one the op implements | rms_norm.py:243-247 |

**Axis parity.** SUPPORTED declares 9 axes; validate() builds exactly those 9 (7 from tensor properties/kwargs at rms_norm.py:220-228, plus alignment and rank from INPUT_TAGGERS at rms_norm.py:229-230). No mismatch.

**INVALID** is not here: .claude/eval/golden_tests/rms_norm/feature_spec.py (INVALID) -- not the op's to declare; referenced, not copied

## §2 Signature and argument semantics

| param | type | default | optional | keyword-only | enters at | mathematical role | feeds | source |
|---|---|---|---|---|---|---|---|---|
| input_tensor | ttnn.Tensor | null | False | no (positional) | gate, plan, size, bake, dispatch, read, compute:C2, compute:C3, compute:C10, compute:C12, write | the values normalized; its last dim is the reduced axis | axes: dtype / layout / memory_layout / alignment / rank<br>plan: _plan_placement reads its memory_config and shard spec<br>size: bt = tile_size(dtype) scales cb_input_tiles / cb_x_squared / cb_normalized / cb_output_tiles<br>bake: reader ct[0,1,8,10,11], writer ct[0,1,6,7], compute ct[0]<br>bake: TensorAccessorArgs(input_tensor)<br>dispatch: reader rt[0] x_addr<br>cb:cb_input_tiles | rms_norm.py:258 |
| gamma | Optional[ttnn.Tensor] | null | True | **yes** | gate, size, bake, dispatch, read, compute:C1, compute:C11, compute:C13 | per-channel multiplicative scale applied after the rsqrt | axes: gamma_mode / gamma_dtype / gamma_layout<br>size: gt = tile_size(gamma.dtype) scales cb_gamma_tiles / cb_gamma_sticks; has_gamma adds cb_normalized to _cb_block_mult<br>bake: reader ct[6,7,9,19], compute ct[5]<br>bake: TensorAccessorArgs(gamma)<br>dispatch: reader rt[1] g_addr<br>cb:cb_gamma_tiles<br>cb:cb_gamma_sticks<br>cb:cb_normalized | rms_norm.py:260 |
| epsilon | float | 1e-06 | True | **yes** | gate, bake, compute:C6, compute:C8 | added to the mean square before the rsqrt | gate: epsilon > 0.0<br>bake: compute ct[8] EPS as raw fp32 bits (_f32_bits)<br>the finalize's rms_stat_scale_body: dst = dst * (1/W) + eps | rms_norm.py:261 |
| compute_kernel_config | Optional[ttnn.ComputeConfigDescriptor] | null | True | **yes** | gate, bake, compute:C3, compute:C9, compute:C12, compute:C13 | math_fidelity / fp32_dest_acc_en / math_approx_mode -- the DEST accumulation width and FPU fidelity | axes: fp32_dest_acc_en (defaulted through default_compute_kernel_config())<br>bake: passed through UNMODIFIED as KernelDescriptor.config on the compute kernel (rms_norm_program_descriptor.py:2559)<br>compute:C3 compute_square -- the DEST fold (SQ_OUT.dest_accumulation, kernels/rms_norm_compute.cpp:881) accumulates the chunk's width tiles SERIALLY in a DEST register that is 16-bit at fp32_dest_acc_en=False and 32-bit at True<br>compute:C9 compute_recv_unpack -- COMBINE_DEST_BATCH is clamped to ckl::DEST_AUTO_LIMIT (kernels/rms_norm_compute.cpp:802-805, used at :1249)<br>compute:C12 compute_scale / compute:C13 compute_gamma_mul -- .block_size(PASS_B_BLK), and PASS_B_BLK = pass_b_blk(WT_CHUNK, ckl::DEST_AUTO_LIMIT) (kernels/rms_norm_compute.cpp:707, used at :1289 and :1307) | rms_norm.py:262 |
| memory_config | Optional[ttnn.MemoryConfig] | null | True | **yes** | gate, plan, size, bake, dispatch, write | output placement; defaults to the input's | gate: memory_config.memory_layout must be in SUPPORTED['memory_layout']<br>plan: native_out (a zero-copy output CB is only meaningful when the output shard spec matches the input's), plan.l1_reserved, band_out_local<br>size: out_shard_pages, or cb_out_depth * block_rows * wt_chunk<br>bake: writer ct[8,12,14], TensorAccessorArgs(output_tensor)<br>dispatch: writer rt[0] out_addr<br>cb:cb_output_tiles | rms_norm.py:263 |
| program_config | Optional[Any] | null | True | **yes** | **(none)** | none -- the parameter is accepted by both rms_norm() and validate() and read by neither | — | rms_norm.py:264 |

**Keyword-only.** rms_norm() places `*,` immediately after input_tensor (rms_norm.py:259), and validate() does the same (rms_norm.py:196). input_tensor is the ONLY positional parameter; all five of gamma, epsilon, compute_kernel_config, memory_config and program_config are KEYWORD-ONLY and cannot be passed positionally. Recorded per-parameter in signature[].keyword_only.

- **`compute_kernel_config`** — Does NOT enter the `size` stage. The planner accepts compute_kernel_config and never reads an attribute of it: the only live occurrence in the whole 2585-line planner is `config=compute_kernel_config,  # passed through unmodified` (rms_norm_program_descriptor.py:2559). No CB page count, page size, depth, block-rows solve or footprint term is a function of the config, and fp32_dest_acc_en changes no CB's page format -- every CB derives its format from the dtype of the tensor it carries, and cb_row_stat is fp32 in both modes (see audits.page_format_vs_dest_width). The DEST-width consumers (PASS_B_BLK, COMBINE_DEST_BATCH) are computed IN THE KERNEL from the device-side constant ckl::DEST_AUTO_LIMIT (8 lanes at fp32_dest_acc_en=False, 4 at True), not by any host sizing computation.

- **`program_config`** — G2 FAILURE, recorded rather than papered over. validate() takes program_config in its signature and never references it; rms_norm() forwards it to validate() and does not pass it to create_program_descriptor(). A non-None program_config is therefore silently ignored rather than refused, which the docstring ('reserved; ignored when None') does not say.

## §3 Blocking model as shipped

10 **orthogonal regime families**. A flat cross product would invent rows no case reaches, so each family
gets its own table and the reachable combinations are listed separately.

### Family: `placement_scheme`  (rms_norm_program_descriptor.py:1614 (_plan_placement))

| regime | selected when | observed at | block | data movement (vs. minimum) | what a bigger block buys |
|---|---|---|---|---|---|
| SCHEME_ROWS | input memory_layout == INTERLEAVED, or layout == ROW_MAJOR with a HEIGHT shard / interleaved, or in_ml == HEIGHT_SHARDED and shard_w_tiles != Wt (the shard does not span the full padded width), or the L1 bail-out (force_rows) from a shard plan; also the ragged-width + partial-W bail-out | c01, c02, c03, c04, c05, c06, c07, c08, c10, c11, c12, c14, c15, c16, c17, c24 | block_rows tile-rows x wt_chunk width tiles, split off the independent `row` axis by split_work_to_cores(full grid, row_wise=True) | x once from DRAM (twice under STREAM), out once to DRAM, gamma once per core (a broadcast operand replicated across num_cores DRAM reads). Above the DRAM minimum by (num_cores-1) whole-gamma reads. | fewer LLK inits / format reconfigs / CB handshakes per tile-row; fewer reader barriers (transaction granularity is wt_chunk tiles) |
| SCHEME_SHARD_H | layout == TILE and input memory_layout == HEIGHT_SHARDED and shard_w_tiles == Wt | c18 | the whole resident shard row-slice: block_rows x Wt, block_rows solved against L1 | x: ZERO DRAM crossings (cb_input_tiles aliases the resident L1 shard); out: zero when the output shard spec matches the input's; gamma once per core. Below the stated DRAM minimum because the input never crosses DRAM inside the op. | same fixed costs as SCHEME_ROWS; no regime-added term (no cross-core traffic) |
| SCHEME_SHARD_W | layout == TILE and memory_layout in (WIDTH_SHARDED, BLOCK_SHARDED) and not (ragged width tail and W non-tile-aligned); OR memory_layout == INTERLEAVED, layout == TILE, Wt > 1 and _resolve_width_split returns gw > 1 | c09, c13, c19, c20, c21, c22, c25 | block_rows x wt_per_core (the shard's width slice, or Wt // gw), block_rows capped at TILE_DIM by the compact partial's column count | sharded: x and out ZERO DRAM crossings; interleaved split: x once, out once. Both add cross-core traffic: per row-block round, (group_size-1) unicast 4 kB fp32 tiles into the root plus one multicast of one 4 kB fp32 tile to group_size-1 receivers (tree path adds one hop). | fewer combine ROUNDS (rounds = ceil(rows_per_core / block_rows)); each round is a full gather + fold + finalize + multicast rendezvous |
| SCHEME_SHARD_W+BAND | layout == ROW_MAJOR and input memory_layout in (WIDTH_SHARDED, BLOCK_SHARDED) | c23 | block_rows x wt_chunk, where wt_chunk is the WIDEST global tile span any core's band touches; extents read off the shard spec, not solved | x: ZERO DRAM crossings, staged core-locally out of this core's own resident RM shard into the tilize ring; out written back to the local shard when band_out_local, else through the accessor. Same cross-core combine as SCHEME_SHARD_W. | same as SCHEME_SHARD_W; additionally, a band that fills its tile columns and matches the shard stride collapses a per-stick staging read into one wide transfer |

### Family: `compute_residency`  (rms_norm_program_descriptor.py:1883 (_solve_blocking) and kernels/rms_norm_compute.cpp:28-39)

| regime | selected when | observed at | block | data movement (vs. minimum) | what a bigger block buys |
|---|---|---|---|---|---|
| RESIDENT | x_resident and num_w_chunks == 1 -- the whole per-core width slice fits L1 at a depth from CB_DEPTH_CANDIDATES | c01, c02, c03, c04, c05, c06, c07, c08, c09, c10, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c25 | block_rows x wt_per_core, one chunk | x once from DRAM, gamma once per core | amortizes inits/handshakes across more tile-rows |
| ROW_RESIDENT | not RESIDENT, and one WHOLE tile-row of x plus the whole row of gamma fits (held), with the derived CBs chunked; gated on max_rows_per_core >= ROW_RESIDENT_MIN_ROWS_PER_CORE only when it forces a shallower depth than STREAM would use | c11, c24 | 1 x wt_chunk, with cb_input_tiles / cb_gamma_tiles spanning the whole tile-row (x_hold_wt = wt_per_core) and indexed at a TileOffset | x once, gamma once per core -- identical to RESIDENT; strictly fewer DRAM bytes than STREAM at the same chunk count | block_rows is pinned to 1 here; a coarser wt_chunk reduces the per-chunk helper-call count and the pass-B chunk loop trip count |
| STREAM | not even ONE tile-row of x + gamma fits the budget | c12 | 1 x wt_chunk, nothing held | x TWICE from DRAM (one read per pass, NUM_PASSES == 2), gamma re-read for every pass-B chunk of every row-block, out once. Roughly 2x the input bytes of the other two regimes. | a coarser chunk cuts the pass-B gamma re-read count proportionally |

### Family: `reduce_datapath`  (rms_norm_program_descriptor.py:2208-2215 (reduce_acc_via_add))

| regime | selected when | observed at | block | data movement (vs. minimum) | what a bigger block buys |
|---|---|---|---|---|---|
| ReduceTile | NOT (REDUCE_BULK == 1 and wt_per_core >= REDUCE_ACC_VIA_ADD_MIN_WT and wt_chunk >= REDUCE_ACC_VIA_ADD_MIN_CHUNK_WT and not (num_w_chunks == 1 and x_squared_wt < REDUCE_ACC_VIA_ADD_MIN_CALL_WT)) | c01, c02, c03, c05, c07, c09, c14, c15, c16, c17, c19, c20, c21, c22, c23, c24, c25 | reduce call width = x_squared_wt tiles | no change (a compute-datapath choice) | n/a -- this axis trades DEST accumulation depth and SFPU vs FPU cycles, not movement |
| AccumulateViaAdd | REDUCE_BULK == 1 and wt_per_core >= 4 and wt_chunk >= 2 and not (num_w_chunks == 1 and x_squared_wt < 2) | c04, c06, c08, c10, c11, c12, c13, c18 | reduce call width = x_squared_wt tiles, summed pairwise into DEST | no change | n/a; its partial-W mechanism is a 0/1 mask tile instead of the [full, partial] scaler pair, so scaler_tiles drops to 1 |

### Family: `combine_topology`  (rms_norm_program_descriptor.py:1013 (_combine_tree_arity))

| regime | selected when | observed at | block | data movement (vs. minimum) | what a bigger block buys |
|---|---|---|---|---|---|
| none | plan.combine is False (SCHEME_ROWS, SCHEME_SHARD_H, or a width scheme whose group_size == 1) | c01, c02, c03, c04, c05, c06, c07, c08, c10, c11, c12, c14, c15, c16, c17, c18, c24 | n/a | no cross-core traffic | n/a |
| flat_root | plan.combine and _combine_tree_arity returns None -- i.e. ceil(group_size/4) < 2, or 1*(group_size - f0 - f1) < COMBINE_TREE_MIN_DELETED_FOLD_TILES | c09, c13, c19, c20, c22, c25 | one compact fp32 tile per sender per round; the root folds GATHER_SLOTS = group_size rounded up to even pages | per round: group_size-1 unicast 4096 B writes into the root's cb_partials_gathered + 1 multicast of 4096 B to group_size-1 receivers | a coarser block_rows means fewer rounds; the per-round cost is flat in block_rows since D27 |
| slot_tree | plan.combine and f1 = ceil(group_size/COMBINE_TREE_F0) >= 2 and 1*(group_size - f0 - f1) >= COMBINE_TREE_MIN_DELETED_FOLD_TILES | c21, c23 | two levels: f0 = 4 level-0 slots folded on each of f1 gatherers, then f1 raw sums folded on the root | per round: group_size-1 level-0 unicasts + (f1-1) level-1 unicasts of 4096 B + 1 multicast. ONE extra NoC hop, but the root's ingress and fold both drop from group_size to max(f0,f1). | same as flat_root; the tree also HANDS L1 BACK (rings go group_size -> f0+f1+2 pages) |

### Family: `partial_transport`  (rms_norm_program_descriptor.py:2130 (compact_combine))

| regime | selected when | observed at | block | data movement (vs. minimum) | what a bigger block buys |
|---|---|---|---|---|---|
| n/a | not plan.combine | c01, c02, c03, c04, c05, c06, c07, c08, c10, c11, c12, c14, c15, c16, c17, c18, c24 | n/a | n/a | n/a |
| identity | plan.combine and block_rows == 1 -- both permutation matmuls are the identity and the kernels elide them | c09, c13, c19, c21, c22, c25 | one column-shaped fp32 tile per round; the gather ships GATHER_FACES (2) of its 4 faces | half a tile (2048 B) per member per round on the gather; the multicast still moves a whole tile | n/a (block_rows is 1 by definition of this regime) |
| compact | plan.combine and block_rows > 1 -- a member matmul-permutes its block_rows partials into the COLUMNS of ONE tile before shipping, and un-permutes the multicast stat back | c20, c23 | one fp32 tile per round carrying block_rows stats in columns 0..block_rows-1; requires cb_bank (block_rows bf16 one-hot pages), cb_compact_handoff and cb_mcast_in | one WHOLE 4096 B tile per member per round regardless of block_rows -- this is what removes the group_size x block_rows term from the L1 solve | block_rows up to TILE_DIM (32) rides one round; past 32 the compact tile has no columns left, and both host and kernels assert the bound |

### Family: `square_datapath`  (rms_norm_program_descriptor.py:2161 (square_dest_acc_per_row) and kernels/rms_norm_compute.cpp:615 (SQ_FOLD))

| regime | selected when | observed at | block | data movement (vs. minimum) | what a bigger block buys |
|---|---|---|---|---|---|
| dest_fold | HOST (square_dest_acc_per_row, rms_norm_program_descriptor.py:2161): kernel_partial_w == 0 and wt_chunk <= DEST_ACC_SQUARE_MAX_WT -- this sets x_squared_wt = 1. KERNEL (SQ_FOLD, kernels/rms_norm_compute.cpp:615): (X_SQUARED_WT == 1) && (WT_CHUNK > 1). BOTH terms are required: at wt_chunk == 1 the host predicate is true but the kernel disables the fold (there is nothing to fold), so the build takes the packed path -- which is why c23 and c24 sit under `packed` despite satisfying the host half. | c01, c02, c03, c07, c09, c14, c15, c16, c19, c20, c21, c22, c25 | the chunk's wt_chunk width tiles are squared and accumulated in DEST (DestAccumulation::PerRow); cb_x_squared receives ONE tile per tile-row and x_squared_wt == 1 | no DRAM change; removes wt_chunk-1 packs and the matching unpacks per tile-row | n/a -- this is a CEILING on wt_chunk, because the fold accumulates SERIALLY in a DEST word that is 16-bit at fp32_dest_acc_en == False |
| packed | kernel_partial_w != 0, or wt_chunk > DEST_ACC_SQUARE_MAX_WT, or wt_chunk == 1 (the kernel's SQ_FOLD = (X_SQUARED_WT == 1) && (WT_CHUNK > 1) term) | c04, c05, c06, c08, c10, c11, c12, c13, c17, c18, c23, c24 | every x^2 tile is packed to cb_x_squared; x_squared_wt == wt_chunk | no DRAM change | n/a |

### Family: `gamma_read_granularity`  (rms_norm_program_descriptor.py:1854-1859 (gamma_trim))

| regime | selected when | observed at | block | data movement (vs. minimum) | what a bigger block buys |
|---|---|---|---|---|---|
| face_rows | has_gamma and gamma layout == TILE and gamma tile_size % 4 == 0 and (tile_size//4) % 64 == 0 (every LINEAR tiled format) | c01, c03, c04, c05, c06, c07, c08, c09, c10, c11, c12, c13, c14, c18, c19, c20, c21, c22 | two reads of TILE_DIM * gamma_elem_bytes at page offsets 0 and gt/4 | ~2 face-rows per gamma tile instead of a whole tile | n/a |
| half_page | has_gamma and gamma layout == TILE and the face offset gt/4 is not 64-byte aligned (bfloat8_b: 272-byte face) | c15 | one read of gt/2 from offset 0 | half a gamma tile per tile | n/a |
| whole_tile_or_n/a | gamma absent, or gamma layout == ROW_MAJOR (no tile padding to trim) | c02, c16, c17, c23, c24, c25 | whole tile (TILE gamma) or a single stick (RM gamma) | whole gamma tile per tile, or one stick | n/a |

### Family: `output_residency`  (rms_norm_program_descriptor.py:1642 (native_out = output_tensor.layout == TILE_LAYOUT and _same_shard_spec(input, output)))

| regime | selected when | observed at | block | data movement (vs. minimum) | what a bigger block buys |
|---|---|---|---|---|---|
| native_out (aliased) | the plan is SCHEME_SHARD_H or SCHEME_SHARD_W (a TILE shard) AND output_tensor.layout == TILE_LAYOUT AND _same_shard_spec(input_tensor, output_tensor) -- same grid, same per-core extent, same order | c18, c19, c20, c21, c22, c25 | the output shard itself: out_shard_h_tiles * out_shard_w_tiles pages, aliased through ttnn.cb_descriptor_from_sharded_tensor | ZERO writer NoC traffic for the output -- compute packs straight into the output shard's own L1 and the writer only takes the completion barrier | nothing: the extent is the shard's, not a knob. It removes cb_output_tiles from the arena entirely (out_shard_pages * bt of aliased, not allocated, bytes). |
| allocated_out | not native_out -- an interleaved output, a ROW_MAJOR output, or a sharded output whose spec differs from the input's | c01, c02, c03, c04, c05, c06, c07, c08, c09, c10, c11, c12, c13, c14, c15, c16, c17, c23, c24 | cb_out_depth * block_rows * wt_chunk pages of arena | one DRAM/L1 write per output element, issued by the writer | a coarser block amortises the writer's per-block wait; depth buys compute<->writer overlap |

### Family: `band_write_back`  (rms_norm_program_descriptor.py:1507 (band_out_local) -- reached only on SCHEME_SHARD_W+BAND)

| regime | selected when | observed at | block | data movement (vs. minimum) | what a bigger block buys |
|---|---|---|---|---|---|
| n/a (not the BAND scheme) | plan.band is False | c01, c02, c03, c04, c05, c06, c07, c08, c09, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c24, c25 | n/a | n/a | n/a |
| local_l1 | plan.band and output_tensor.layout == ROW_MAJOR_LAYOUT and _same_shard_spec(input_tensor, output_tensor) -- core i's band IS its own output shard | c23 | one band (w_real_elems elements) per stick, at the out shard's own aligned page stride (writer ct[14] OUT_SHARD_ROW_BYTES != 0) | ZERO DRAM writes: the write is into this core's own resident L1 | nothing -- the band extent is shard-derived |
| accessor_sticks | plan.band and NOT band_out_local, with the output stick-paged (memory_layout in (INTERLEAVED, HEIGHT_SHARDED)). Any other output geometry raises NotImplementedError instead of planning. | **none** | one band per stick, addressed through the output TensorAccessor at byte offset w_off_elems * elem_bytes | one DRAM write per stick per core | nothing -- the band extent is shard-derived |

### Family: `multicast_topology`  (rms_norm_program_descriptor.py:1714-1724 (WIDTH shard), rms_norm_program_descriptor.py:1751-1761 (BLOCK shard), rms_norm_program_descriptor.py:1371-1400 (the interleaved width split))

| regime | selected when | observed at | block | data movement (vs. minimum) | what a bigger block buys |
|---|---|---|---|---|---|
| none | plan.combine is False (group_size == 1) | c01, c02, c03, c04, c05, c06, c07, c08, c10, c11, c12, c14, c15, c16, c17, c18, c24 | n/a | no cross-core traffic | n/a |
| Mcast1D (PerRow) | BLOCK_SHARDED (each grid ROW is a width group, root at column 0), or the interleaved width split when gw <= grid.x (a gw x gh rectangle) | c20 | one group per grid row; every core of the grid is active | one CT block covers all gh groups; the stat multicast runs along each grid row | nothing -- the topology is fixed by the shard/grid rectangle |
| Mcast2D (bounding box) | WIDTH_SHARDED (the whole shard grid is ONE group, root = shard_cores[0]), or the interleaved width split when gw > grid.x (ONE group packed row-major over ceil(gw/grid.x) whole grid rows) | c09, c13, c19, c21, c22, c23, c25 | ONE group over the grid's BOUNDING BOX; in-box cores outside the group join INACTIVE (row_count == 0) so the stat lands in a cb_row_final this program owns | the multicast covers the whole bounding box, so inactive padding cores receive the stat and drop it | nothing -- the topology is fixed by the shard grid / packed group |

### Reachable combinations actually observed

| placement_scheme | compute_residency | reduce_datapath | combine_topology | partial_transport | square_datapath | gamma_read_granularity | output_residency | band_write_back | multicast_topology | cases |
|---|---|---|---|---|---|---|---|---|---|---|
| SCHEME_ROWS | RESIDENT | ReduceTile | none | n/a | dest_fold | face_rows | allocated_out | n/a (not the BAND scheme) | none | c01, c03, c07, c14 |
| SCHEME_ROWS | RESIDENT | ReduceTile | none | n/a | dest_fold | whole_tile_or_n/a | allocated_out | n/a (not the BAND scheme) | none | c02, c16 |
| SCHEME_ROWS | RESIDENT | AccumulateViaAdd | none | n/a | packed | face_rows | allocated_out | n/a (not the BAND scheme) | none | c04, c06, c08, c10 |
| SCHEME_ROWS | RESIDENT | ReduceTile | none | n/a | packed | face_rows | allocated_out | n/a (not the BAND scheme) | none | c05 |
| SCHEME_SHARD_W | RESIDENT | ReduceTile | flat_root | identity | dest_fold | face_rows | allocated_out | n/a (not the BAND scheme) | Mcast2D (bounding box) | c09 |
| SCHEME_ROWS | ROW_RESIDENT | AccumulateViaAdd | none | n/a | packed | face_rows | allocated_out | n/a (not the BAND scheme) | none | c11 |
| SCHEME_ROWS | STREAM | AccumulateViaAdd | none | n/a | packed | face_rows | allocated_out | n/a (not the BAND scheme) | none | c12 |
| SCHEME_SHARD_W | RESIDENT | AccumulateViaAdd | flat_root | identity | packed | face_rows | allocated_out | n/a (not the BAND scheme) | Mcast2D (bounding box) | c13 |
| SCHEME_ROWS | RESIDENT | ReduceTile | none | n/a | dest_fold | half_page | allocated_out | n/a (not the BAND scheme) | none | c15 |
| SCHEME_ROWS | RESIDENT | ReduceTile | none | n/a | packed | whole_tile_or_n/a | allocated_out | n/a (not the BAND scheme) | none | c17 |
| SCHEME_SHARD_H | RESIDENT | AccumulateViaAdd | none | n/a | packed | face_rows | native_out (aliased) | n/a (not the BAND scheme) | none | c18 |
| SCHEME_SHARD_W | RESIDENT | ReduceTile | flat_root | identity | dest_fold | face_rows | native_out (aliased) | n/a (not the BAND scheme) | Mcast2D (bounding box) | c19, c22 |
| SCHEME_SHARD_W | RESIDENT | ReduceTile | flat_root | compact | dest_fold | face_rows | native_out (aliased) | n/a (not the BAND scheme) | Mcast1D (PerRow) | c20 |
| SCHEME_SHARD_W | RESIDENT | ReduceTile | slot_tree | identity | dest_fold | face_rows | native_out (aliased) | n/a (not the BAND scheme) | Mcast2D (bounding box) | c21 |
| SCHEME_SHARD_W+BAND | RESIDENT | ReduceTile | slot_tree | compact | packed | whole_tile_or_n/a | allocated_out | local_l1 | Mcast2D (bounding box) | c23 |
| SCHEME_ROWS | ROW_RESIDENT | ReduceTile | none | n/a | packed | whole_tile_or_n/a | allocated_out | n/a (not the BAND scheme) | none | c24 |
| SCHEME_SHARD_W | RESIDENT | ReduceTile | flat_root | identity | dest_fold | whole_tile_or_n/a | native_out (aliased) | n/a (not the BAND scheme) | Mcast2D (bounding box) | c25 |

### Knobs

| knob | value | selects | kind | source |
|---|---|---|---|---|
| L1_SAFETY_FRACTION | 0.85 | fraction of usable per-core L1 the CBs may occupy; scales every block/chunk solve | perf | rms_norm_program_descriptor.py:635 |
| L1_CB_ARENA_BASE_RESERVE | 70656 | bytes between the worker-L1 unreserved base and the first CB address; subtracted ONLY when a shard is resident | correctness_floor | rms_norm_program_descriptor.py:654 |
| CB_RM_STAGE_DEPTH | 2 | depth of the ROW_MAJOR stick staging CBs (reader<->tilize overlap) | perf | rms_norm_program_descriptor.py:657 |
| CB_DEPTH_CANDIDATES | [2] | ordered depth candidates the RESIDENT search walks for cb_input_tiles / cb_output_tiles, coarsest first | perf | rms_norm_program_descriptor.py:668 |
| GRID_W | 0 | cores along `width` on an INTERLEAVED input: 0 = AUTO policy, >=1 forces the group size (1 = off) | perf | rms_norm_program_descriptor.py:680 |
| WIDTH_SPLIT_MIN_WT_PER_CORE | 4 | smallest number of width TILES the AUTO width split may leave a core | perf | rms_norm_program_descriptor.py:691 |
| WIDTH_SPLIT_MAX_GROUP_CORES | 16 | hard ceiling on cores per width group (the gather fan-in) for the INTERLEAVED split only | perf | rms_norm_program_descriptor.py:695 |
| WIDTH_SPLIT_MIN_GAIN | 4 | minimum ratio of split cores to plain-row-split cores before the width split is taken | perf | rms_norm_program_descriptor.py:700 |
| REDUCE_BULK | 1 | reduce() input policy: 1 = BulkWaitBulkPop, 0 = WaitAndPopPerTile. AccumulateViaAdd is gated on 1. | perf | rms_norm_program_descriptor.py:704 |
| REDUCE_ACC_VIA_ADD_MIN_WT | 4 | smallest wt_per_core (this core's WHOLE reduce dim) at which AccumulateViaAdd is preferred | correctness_floor | rms_norm_program_descriptor.py:715 |
| REDUCE_ACC_VIA_ADD_MIN_CHUNK_WT | 2 | smallest wt_chunk at which AccumulateViaAdd is admitted (a 1-tile chunk has nothing to pair) | correctness_floor | rms_norm_program_descriptor.py:739 |
| REDUCE_ACC_VIA_ADD_MIN_CALL_WT | 2 | smallest x_squared_wt (the reduce's PER-CALL width) below which D20 carves AccumulateViaAdd out | perf | rms_norm_program_descriptor.py:763 |
| CB_ROW_STAT_DEPTH | 2 | ring depth of cb_row_stat / cb_sum_handoff / cb_row_final in units of block_rows | correctness_floor | rms_norm_program_descriptor.py:767 |
| DEST_ACC_SQUARE_MAX_WT | 8 | largest wt_chunk at which pass A's square folds width tiles into DEST instead of packing to cb_x_squared | correctness_floor | rms_norm_program_descriptor.py:777 |
| GATHER_FACES | 2 | faces per fp32 partial tile the IDENTITY-path gather ships member -> root | perf | rms_norm_program_descriptor.py:790 |
| CB_COMBINE_FLAT_DEPTH | 2 | page depth of the combine CBs that carry ONE compact tile per round (block_rows-independent) | correctness_floor | rms_norm_program_descriptor.py:800 |
| COMBINE_TREE_F0 | 4 | level-0 fan-in of the combine's slot tree | perf | rms_norm_program_descriptor.py:875 |
| COMBINE_TREE_MIN_DELETED_FOLD_TILES | 18 | crossover on rows_per_round*(group_size - f0 - f1), the root fold-tiles the tree deletes per round | perf | rms_norm_program_descriptor.py:876 |
| ROW_RESIDENT_MIN_ROWS_PER_CORE | 2 | tile-rows a core must own before ROW_RESIDENT is taken at a SHALLOWER depth than STREAM would use | perf | rms_norm_program_descriptor.py:902 |
| PASS_B_BLK (kernel-derived, not a host constant) | "pass_b_blk(WT_CHUNK, ckl::DEST_AUTO_LIMIT) -- the largest divisor of WT_CHUNK that is <= DEST_AUTO_LIMIT" | pass B's DEST-LANE BLOCK: the block_size of compute_scale (C12) and compute_gamma_mul (C13), and hence the PerBlockSize reserve/push granularity of cb_normalized / cb_output_tiles. It is the ONLY block quantity in the op that changes with fp32_dest_acc_en -- ckl::DEST_AUTO_LIMIT is 8 lanes at False and 4 at True. | perf | kernels/rms_norm_compute.cpp:707 (definition), :674-706 (D21's measured justification), used at :1289 and :1307 |

- **PASS_B_BLK (kernel-derived, not a host constant)** — Listed here even though it is not a module-level constant of the planner: it is a block factor a consumer sizing DEST windows has to know about, it is coupled to the PerBlockSize pack lifecycle (a PerTile reserve at block_size > 1 corrupts the CB ring and HANGS -- observed), and it is the op's only fp32_dest_acc_en-dependent block.
  - measured: 1.28x at rows=1/wt=4, 1.49x at rows=1/wt=32, 1.62x at rows=8/wt=16, 1.65x at rows=32/wt=4, 1.66x with HAS_GAMMA=0; the block_size curve at 128 tiles is monotone 13266/9229/8804/8209 ns at B = 1/2/4/8 (descriptor D21)

### Numeric thresholds between regimes

| quantity measured | knob | value | bracket | below | at or above | straddling pair | source |
|---|---|---|---|---|---|---|---|
| wt_per_core (this core's whole reduce dim) | REDUCE_ACC_VIA_ADD_MIN_WT | 4 | >=  (a floor) | ReduceTile | AccumulateViaAdd | c05 (Wt=3)  /  c06 (Wt=4) | rms_norm_program_descriptor.py:2210 |
| wt_chunk | REDUCE_ACC_VIA_ADD_MIN_CHUNK_WT | 2 | >=  (a floor) | ReduceTile | AccumulateViaAdd eligible | c24 (wt_chunk=1)  /  c04 (wt_chunk=9) | rms_norm_program_descriptor.py:2211 |
| x_squared_wt (the reduce's per-call width), only when num_w_chunks == 1 | REDUCE_ACC_VIA_ADD_MIN_CALL_WT | 2 | >=  (a floor) | ReduceTile (D20 carve-out) | AccumulateViaAdd | c03 (x_squared_wt=1 under the fold)  /  c04 (x_squared_wt=9) | rms_norm_program_descriptor.py:2214 |
| wt_chunk | DEST_ACC_SQUARE_MAX_WT | 8 | <=  (a CEILING; every other row in this table is a >= floor, so this row's `below` is the only inclusive one) | dest_fold (x_squared_wt = 1) -- INCLUSIVE: the predicate is `wt_chunk <= DEST_ACC_SQUARE_MAX_WT`, so wt_chunk == 8 folds | packed (x_squared_wt = wt_chunk) -- strictly above, i.e. wt_chunk >= 9 | c03 (wt_chunk=8)  /  c04 (wt_chunk=9) | rms_norm_program_descriptor.py:2161 and kernels/rms_norm_compute.cpp:615 |
| rows_per_round * (group_size - f0 - f1), with rows_per_round identically 1 | COMBINE_TREE_MIN_DELETED_FOLD_TILES | 18 | >=  (a floor) | flat_root | slot_tree | c22 (group_size 28, deleted 17)  /  c21 (group_size 32, deleted 20) | rms_norm_program_descriptor.py:1027 |
| f1 = ceil(group_size / COMBINE_TREE_F0) | COMBINE_TREE_F0 | 4 | >=  (a floor) | flat_root (f1 < 2 is an expressibility floor: a level that gathers one member is a hop, not a fold) | slot_tree eligible | c19 (group_size 8, f1=2 but deleted 2 < 18)  /  c21 (group_size 32, f1=8) | rms_norm_program_descriptor.py:1022-1024 |
| max_rows_per_core, applied only when ROW_RESIDENT would force a shallower depth than STREAM | ROW_RESIDENT_MIN_ROWS_PER_CORE | 2 | >=  (a floor) | STREAM | ROW_RESIDENT | unobserved  /  c11 (max_rows 3, depth 2 -> 1) | rms_norm_program_descriptor.py:2063 |
| block_rows on any combine path | TILE_DIM (hard cap, not a tunable) | 32 | >=  (a floor) | legal | clamped -- the compact partial packs one tile-row's sum into one COLUMN of a single tile | c20 (block_rows 20, max_rows 32)  /  unobserved (the L1 solve bound it first) | rms_norm_program_descriptor.py:1925 and rms_norm_program_descriptor.py:2131 |
| total cores at work under the AUTO width split, vs the plain row split's core count | WIDTH_SPLIT_MIN_GAIN | 4 | >=  (a floor) | no split (SCHEME_ROWS) | SCHEME_SHARD_W on an interleaved input | c08 ((1024,1024): row split already uses 32 cores)  /  c13 ((1,1,32,7168): 1 -> 16 cores) | rms_norm_program_descriptor.py:1327 |

### Compile-time vs runtime — what a caller cannot change without a recompile

#### `rms_norm_reader` compile-time args  (rms_norm_program_descriptor.py:2389)

| idx | name | c01 | c05 | c11 | c12 | c19 | c20 | c21 | c23 | c24 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | IS_TILE | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| 1 | WT | 4 | 3 | 224 | 344 | 32 | 32 | 160 | 96 | 127 |
| 2 | WT_CHUNK | 4 | 3 | 56 | 86 | 4 | 4 | 5 | 1 | 1 |
| 3 | NUM_W_CHUNKS | 1 | 1 | 4 | 4 | 1 | 1 | 1 | 1 | 127 |
| 4 | BLOCK_ROWS | 1 | 1 | 1 | 1 | 1 | 20 | 1 | 7 | 1 |
| 5 | PARTIAL_W | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | HAS_GAMMA | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 7 | GAMMA_IS_RM | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| 8 | ELEM_BYTES | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| 9 | GAMMA_ELEM_BYTES | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| 10 | R_RM | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 224 | 32 |
| 11 | W | 128 | 72 | 7168 | 11008 | 1024 | 1024 | 5120 | 3072 | 4064 |
| 12 | REDUCE_ACC_VIA_ADD | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| 13 | NATIVE_IN | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 |
| 14 | IN_SHARD_PAGES | 0 | 0 | 0 | 0 | 4 | 128 | 5 | 0 | 0 |
| 15 | BAND | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| 16 | SHARD_ROW_BYTES | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 64 | 0 |
| 17 | STAGE_ZERO | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 18 | X_RESIDENT | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
| 19 | GAMMA_TRIM | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 0 | 0 |
| 20 | BANK_PAGES | 0 | 0 | 0 | 0 | 0 | 20 | 0 | 7 | 0 |

_indices 21.. are TensorAccessorArgs(input) followed by TensorAccessorArgs(gamma) (or the empty-accessor form when gamma is absent -- SAME arg count, values zeroed). Total observed 25 (interleaved) / 33 (tile shard) / 43-45 (wide shard) / 63 (64-core block shard) / 79 (96-core RM band)._

#### `rms_norm_writer` compile-time args  (rms_norm_program_descriptor.py:2433)

| idx | name | c01 | c05 | c11 | c12 | c19 | c20 | c21 | c23 | c24 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | IS_TILE | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| 1 | WT | 4 | 3 | 224 | 344 | 32 | 32 | 160 | 96 | 127 |
| 2 | WT_CHUNK | 4 | 3 | 56 | 86 | 4 | 4 | 5 | 1 | 1 |
| 3 | NUM_W_CHUNKS | 1 | 1 | 4 | 4 | 1 | 1 | 1 | 1 | 127 |
| 4 | BLOCK_ROWS | 1 | 1 | 1 | 1 | 1 | 20 | 1 | 7 | 1 |
| 5 | ELEM_BYTES | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| 6 | R_RM | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 224 | 32 |
| 7 | W | 128 | 72 | 7168 | 11008 | 1024 | 1024 | 5120 | 3072 | 4064 |
| 8 | NATIVE_OUT | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 |
| 9 | COMBINE | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 |
| 10 | GATHER_SEM_ID | 0 | 0 | 0 | 0 | 2 | 2 | 2 | 2 | 0 |
| 11 | GROUP_SIZE | 1 | 1 | 1 | 1 | 8 | 8 | 32 | 96 | 1 |
| 12 | OUT_SHARD_PAGES | 0 | 0 | 0 | 0 | 4 | 128 | 5 | 0 | 0 |
| 13 | BAND | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| 14 | OUT_SHARD_ROW_BYTES | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 64 | 0 |
| 15 | GATHER_FACES | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| 16 | TREE_F0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 4 | 0 |
| 17 | TREE_F1 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 24 | 0 |

_indices 18..23 are plan.mcast.compile_time_args() (or six zeros when there is no combine); the rest are TensorAccessorArgs(output). Kernel declares McastArgs<18, 12>()._

#### `rms_norm_compute` compile-time args  (rms_norm_program_descriptor.py:2466)

| idx | name | c01 | c05 | c11 | c12 | c19 | c20 | c21 | c23 | c24 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | IS_TILE | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| 1 | WT_CHUNK | 4 | 3 | 56 | 86 | 4 | 4 | 5 | 1 | 1 |
| 2 | NUM_W_CHUNKS | 1 | 1 | 4 | 4 | 1 | 1 | 1 | 1 | 127 |
| 3 | BLOCK_ROWS | 1 | 1 | 1 | 1 | 1 | 20 | 1 | 7 | 1 |
| 4 | PARTIAL_W | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | HAS_GAMMA | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 6 | GAMMA_IS_RM | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| 7 | INV_W_BITS | 1006632960 | 1013157433 | 957499685 | 952009466 | 981467136 | 981467136 | 961334477 | 967486123 | 964755972 |
| 8 | EPS_BITS | 897988541 | 897988541 | 897988541 | 897988541 | 897988541 | 897988541 | 897988541 | 897988541 | 897988541 |
| 9 | REDUCE_BULK | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 10 | REDUCE_ACC_VIA_ADD | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| 11 | SCALER_TILES | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 12 | COMBINE | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 |
| 13 | GROUP_SIZE | 1 | 1 | 1 | 1 | 8 | 8 | 32 | 96 | 1 |
| 14 | X_SQUARED_WT | 1 | 3 | 56 | 86 | 1 | 1 | 1 | 1 | 1 |
| 15 | X_RESIDENT | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
| 16 | NATIVE_IN | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 |
| 17 | TREE_F0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 4 | 0 |
| 18 | TREE_F1 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 24 | 0 |

_no trailing args -- exactly 19 on every build._

#### Runtime args (per core)

**`rms_norm_reader`** — length 10  (rms_norm_program_descriptor.py:2529)

| idx | name | per-core meaning |
|---|---|---|
| 0 | x_addr | input buffer address (identical on every core) |
| 1 | g_addr | gamma buffer address, 0 when gamma is absent |
| 2 | row_start | first TILE-row this core owns |
| 3 | row_count | tile-rows owned; 0 marks an INACTIVE core, which returns immediately |
| 4 | w_start | first width TILE owned (0 off the width schemes) |
| 5 | w_real | REAL width tiles owned (<= wt_per_core; smaller on a ragged shard tail) |
| 6 | stick_base | first ROW_MAJOR stick owned |
| 7 | stick_count | ROW_MAJOR sticks owned |
| 8 | w_off_elems | first width ELEMENT owned (the BAND scheme's global-tile-frame offset) |
| 9 | w_real_elems | REAL width elements owned |

**`rms_norm_writer`** — length {"no_combine": 12, "combine": 16}  (rms_norm_program_descriptor.py:2530-2537)

| idx | name | per-core meaning |
|---|---|---|
| 0 | out_addr | output buffer address |
| 1 | row_start | first TILE-row this core owns |
| 2 | row_count | tile-rows owned (0 => INACTIVE) |
| 3 | w_start | first width TILE owned |
| 4 | is_root | 1 on the width group's root (gathers, folds, finalizes, multicasts) |
| 5 | slot | index within the width group |
| 6 | stick_base | first ROW_MAJOR stick owned |
| 7 | stick_count | ROW_MAJOR sticks owned |
| 8 | w_off_elems | first width ELEMENT owned |
| 9 | w_real_elems | REAL width elements owned |
| 10 | tree_parent_x | level-0 gatherer's VIRTUAL x (0 off the tree path and on an INACTIVE core) |
| 11 | tree_parent_y | level-0 gatherer's VIRTUAL y |
| 12..15 | mcast runtime args | plan.mcast.runtime_args(core); absent (length 12) when there is no combine |

**`rms_norm_compute`** — length 4  (rms_norm_program_descriptor.py:2538)

| idx | name | per-core meaning |
|---|---|---|
| 0 | row_count | tile-rows owned (0 => INACTIVE, returns before any CB or LLK state is touched) |
| 1 | owns_last_w | 1 only on the core holding the row's LAST width tile -- applies the partial-W scaler / mask |
| 2 | is_root | 1 on the width group's root |
| 3 | slot | index within the width group |

## §4 Work split and placement

| scheme | axis cut | core assignment | grid | ragged tail | locality | source |
|---|---|---|---|---|---|---|
| SCHEME_ROWS | the independent `row` axis (all leading dims folded, including H) | ttnn.split_work_to_cores(full compute grid, Rt, row_wise=True) -- two groups with rows-per-core rpc1 / rpc2, assigned by a prefix sum over _cores_in(g1) then _cores_in(g2). row_wise=True MUST match _cores_in's row_wise=True. | device.compute_with_storage_grid_size() (11 x 10 = 110 on this part); cores used = min(Rt, 110) | split_work_to_cores' own two-group split; a core's LAST row-block is partial whenever block_rows does not divide its assignment, which is exactly what CB_ROW_STAT_DEPTH >= 2 exists to keep correct (D6) | local | rms_norm_program_descriptor.py:1416 |
| SCHEME_SHARD_H | the independent `row` axis, already cut by the input's own HEIGHT shard | the shard's own grid, in _cores_in(shard_grid) order; core i owns tile-rows [i*shard_h_t, i*shard_h_t + shard_h_t) | the shard spec's grid | row_count = max(0, min(shard_h_t, Rt - row_start)) -- a trailing core can own 0 tile-rows and joins INACTIVE | local | rms_norm_program_descriptor.py:1644-1669 |
| SCHEME_SHARD_W | the DEPENDENT `width` axis | WIDTH_SHARDED: the whole shard grid is ONE group, root = shard_cores[0], multicast over the grid's BOUNDING BOX with in-box/out-of-shard cores joining INACTIVE. BLOCK_SHARDED: a rectangle, grid column x owns width slice x and grid row y owns row block y; each grid ROW is a width group with its root at column 0. INTERLEAVED split: gw x gh rectangle (Mcast1D PerRow) when gw <= grid.x, else ONE group packed row-major over ceil(gw/grid.x) whole grid rows (Mcast2D over the bounding box). | the shard spec's grid, or the (gw, gh) rectangle / packed box from _auto_width_split | SHARDED: a ragged width tail is allowed -- the last core's block ends in whole PAD tiles which the reader zeroes once (native) so they add exactly 0; the combination (ragged tail AND non-tile-aligned W) bails out to SCHEME_ROWS because no ReducePartialScaler can put the mask on the last REAL tile. INTERLEAVED: gw is clamped to a DIVISOR of Wt, so there is no ragged tail at all (an interleaved core has no pad storage). | cross-core | rms_norm_program_descriptor.py:1763-1775 / rms_norm_program_descriptor.py:1347 |
| SCHEME_SHARD_W+BAND | the DEPENDENT `width` axis, cut by a ROW_MAJOR shard whose page is a row SEGMENT | read off the RM shard spec; each core owns a BAND of contiguous elements of every row it holds, and stages that band out of its OWN L1 into the tilize ring in the tensor's GLOBAL TILE FRAME | the shard spec's grid (observed 96 cores in a 99-core bounding box) | wt_chunk is the WIDEST global tile span any core's band touches; a core whose band spans fewer tiles stages an all-zero pad tile column, which the reduce adds 0 for and the writer never writes back | cross-core | rms_norm_program_descriptor.py:1460 |

### Cross-core schemes: what crosses and what synchronises it

| scheme | crosses | synchronised by |
|---|---|---|
| SCHEME_SHARD_W | per row-block round: each member's fp32 partial sum-of-squares (ONE 4096-byte tile under the compact layout, or GATHER_FACES=2 faces = 2048 B under the identity layout) member -> root; then the root's finalized 1/rms stat (one 4096-byte fp32 tile) root -> every member by multicast. | the mcast helper's two owned semaphores (base_sem_id 0) plus ONE ARRIVAL SEMAPHORE PER TREE LEVEL, ids consecutive from plan.gather_sem_id = mcast.next_base_sem_id() (observed 2). Flat: 3 semaphores. Tree: 4. Built at rms_norm_program_descriptor.py:2573-2579. |
| SCHEME_SHARD_W+BAND | identical to SCHEME_SHARD_W | identical to SCHEME_SHARD_W |

### §4 hazards  _(prose; referenced from the JSON as `hazard_ref`)_

**Arrival ordering, and why one semaphore per tree level is not negotiable.** A level-1 sender only has to
finish its OWN level-0 chunk before forwarding, and that is a *different* chunk from the root's own level-0
run. So a level-1 sender can legally arrive before one of the root's own level-0 members. A single cumulative
arrival counter would let that early level-1 increment satisfy the root's level-0 `wait_min`, and the root
would fold a slot that has not landed — silent corruption, not a hang. The op allocates ids consecutively from
`plan.gather_sem_id`, one per level.

**No self-signalling on the root.** `Semaphore::up(value)` is a non-atomic local read-modify-write. A local
bump on the root would race the members' remote atomic increments and silently drop one, which surfaces as a
hang. The root therefore never signals itself; it accounts for its own slot by arithmetic.

**An un-written gather slot is folded whole.** The root's fold walks the entire `GATHER_SLOTS` window
pairwise, so any page no sender writes — an odd group's pad slot, or a ragged tree run's tail — must be an
exact `+0.0`. It is boot-zeroed once. Zeroing it per round instead would be a race against an already-landed
member.

**INACTIVE cores are load-bearing.** A width-shard grid that is not a rectangle, and a packed interleaved
width group wider than one grid row, both drag in cores that own no rows. They join the program anyway, with
`row_count == 0`, purely so the stat multicast lands in a CB *this program owns* rather than in whatever else
holds that L1. Every kernel returns immediately on `row_count == 0`, before touching any CB or LLK state.

**The band's staging frame is the tensor's global tile grid, not the band's own byte offset.** A DRAM read
whose source offset is not 64-byte aligned is silently *truncated down* to the alignment. Staging in the global
tile frame keeps every gamma fetch on a tile column — a multiple of 64 bytes for every dtype — which is why
gamma works at both layouts on the BAND scheme and why that refinement added no exclusions.

## §5 Buffer inventory — the as-shipped CB ledger

Schema per `references/l1-footprint-discipline.md`, plus five columns a design-time ledger has no place for:
`Index` (the join key to the descriptor), `Present when` (a shipped inventory is regime- and optional-input-
conditional), `Depth` (a knob the reader of this artifact is choosing), `Aliased / backing` (a zero-copy CB over
a resident tensor is a different fact from CB-to-CB sharing), and `Source`.

| CB | Idx | Holds | Present when | Capacity (pages) | Depth | Page format | Live set | Axis accounting | Producer | Consumer | Lifetime | Aliased / backing | Shares with / why not | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cb_input_sticks | 0 | padded ROW_MAJOR stick staging of x, in the tensor's global tile frame | layout == ROW_MAJOR (IS_TILE == 0) | CB_RM_STAGE_DEPTH * wt_chunk | {"value": 2, "knob": "CB_RM_STAGE_DEPTH", "policy": "double (reader <-> tilize overlap)"} | Float16_b/Float32 | wt_chunk pages (one staged tile-row of the chunk) | row (tile-rows): streams -> TILE_DIM sticks per push; width (width tiles): spans -> wt_chunk | reader (read_sticks_for_tilize, or stage_band on the BAND scheme) | compute tilize (C2 / C10) | pass A tilize; pass B tilize as well under STREAM | allocated | NOT SHARED with cb_output_sticks. No reason is recorded in the source; see audits.disjoint_lifetime. | rms_norm_program_descriptor.py:2239 |
| cb_input_tiles | 1 | x tiles for pass A (square) and pass B (scale) | always | cb_x_depth * block_rows * x_hold_wt   (allocated)  \|  shard_h_tiles * shard_w_tiles  (aliased, native_in) | {"value_evals": {"c01": 2, "c02": 2, "c03": 2, "c04": 2, "c05": 2, "c06": 2, "c07": 2, "c08": 2, "c09": 2, "c10": 2, "c11": 1, "c12": 2, "c13": 2, "c14": 2, "c15": 2, "c16": 1, "c17": 1, "c18": null, "c19": null, "c20": null, "c21": null, "c22": null, "c23": 1, "c24": 1, "c25": null}, "knob": "CB_DEPTH_CANDIDATES (forced to 1 on the ROW_MAJOR path and by the ROW_RESIDENT depth sacrifice)", "policy": "double when a NoC producer feeds a compute consumer; single otherwise"} | Bfp8_b/Float16_b/Float32 | block_rows * x_hold_wt (one block; the whole tile-row under ROW_RESIDENT) | row (tile-rows): spans -> block_rows; width (width tiles): spans -> x_hold_wt | reader noc_async_read_tile (TILE) / compute tilize (ROW_MAJOR) / the resident shard itself (native_in -- the reader only PUBLISHES the pages) | compute pass A square (C3) and pass B scale (C12) | the whole block: pass A through pass B; popped once per row-block under ROW_RESIDENT, per chunk otherwise | allocated, or aliased -> input_tensor's L1 shard when plan.native_in | NOT SHARED: live across both passes, and aliased onto the tensor on the native path. | rms_norm_program_descriptor.py:2252 (allocated) / rms_norm_program_descriptor.py:2247 (aliased) |
| cb_x_squared | 2 | x^2 tiles, pass A's output and the reduce's input | always | block_rows * x_squared_wt | {"value": 1, "knob": "none", "policy": "single"} | Bfp8_b/Float16_b/Float32 | block_rows * x_squared_wt | row (tile-rows): spans -> block_rows; width (width tiles): spans -> x_squared_wt (== 1 under the DEST fold, else wt_chunk) | compute square (C3) | compute accumulate_reduce_block (C4) | pass A only | allocated | NOT SHARED with cb_normalized (same page format, disjoint lifetime). No reason recorded; see audits.disjoint_lifetime. | rms_norm_program_descriptor.py:2254 |
| cb_scaler | 3 | the reduce's constant operand: a 1.0 scaler, a [full, partial] scaler pair, or a 0/1 mask tile | always | scaler_pages = 2 if kernel_partial_w != 0 else 1 | {"value": 1, "knob": "none", "policy": "single"} | Float16_b | scaler_tiles = 1 if reduce_acc_via_add else scaler_pages | row (tile-rows): streams -> nothing (constant); width (width tiles): streams -> nothing (constant) | reader (R1 reader_scaler_boot, once at boot) | compute reduce (C4); popped once at kernel exit | whole kernel | allocated | NOT SHARED: live for the whole kernel. | rms_norm_program_descriptor.py:2255 |
| cb_row_stat | 4 | fp32 accumulator for sum(x^2), transformed in place into 1/rms | not plan.combine | CB_ROW_STAT_DEPTH * block_rows | {"value": 2, "knob": "CB_ROW_STAT_DEPTH", "policy": "double, in units of block_rows"} | Float32 | block_rows | row (tile-rows): spans -> block_rows; width (width tiles): streams -> nothing (a REDUCE_ROW result is width-collapsed) | compute reduce (C4) | compute finalize (C6) in place, then pass B scale (C12) as the Col operand | pass A through pass B of one row-block | allocated | NOT SHARED: live across the finalize and pass B. | rms_norm_program_descriptor.py:2266-2267 |
| cb_gamma_sticks | 5 | ROW_MAJOR gamma stick staging | has_gamma and gamma layout == ROW_MAJOR | wt_chunk | {"value": 1, "knob": "none", "policy": "single"} | Float16_b/Float32 | wt_chunk | row (tile-rows): streams -> nothing (gamma is row-invariant); width (width tiles): spans -> wt_chunk | reader (R4 reader_read_gamma) | compute tilize (C1 at boot when X_RESIDENT, else C11 per pass-B chunk) | boot (X_RESIDENT) or per pass-B chunk (STREAM) | allocated | NOT SHARED. No reason recorded. | rms_norm_program_descriptor.py:2272 |
| cb_gamma_tiles | 6 | gamma tiles; only TILE ROW 0 carries real weights | has_gamma | x_hold_wt | {"value": 1, "knob": "none", "policy": "single"} | Bfp8_b/Float16_b/Float32 | x_hold_wt | row (tile-rows): streams -> nothing; width (width tiles): spans -> x_hold_wt | reader (TILE gamma, trimmed per GAMMA_TRIM) / compute tilize (RM gamma) | compute pass B gamma_mul (C13), BroadcastDim::Row | whole kernel when X_RESIDENT (popped once at exit); per pass-B chunk under STREAM | allocated | NOT SHARED: live for the whole kernel on the resident regimes. | rms_norm_program_descriptor.py:2273 |
| cb_normalized | 7 | x * (1/rms), the intermediate between pass B's two multiplies | has_gamma | block_rows * wt_chunk | {"value": 1, "knob": "none", "policy": "single"} | Bfp8_b/Float16_b/Float32 | block_rows * wt_chunk -- the WHOLE capacity. Pass B runs `scale` (C12) and `gamma_mul` (C13) as two SEQUENTIAL helper calls on one compute thread (kernels/rms_norm_compute.cpp:1286-1314): C12 fills the entire IterationShape::grid(rows, WT_CHUNK) into NORM_OUT (== cb_normalized when HAS_G) and only then does C13 wait Upfront on it and pop AtEnd. Nothing drains the buffer while C12 produces. | row (tile-rows): spans -> block_rows; width (width tiles): spans -> wt_chunk | compute scale (C12) | compute gamma_mul (C13) | pass B only | allocated | NOT SHARED with cb_x_squared (same format, disjoint lifetime). No reason recorded; see audits.disjoint_lifetime. | rms_norm_program_descriptor.py:2274 |
| cb_output_tiles | 8 | output tiles | always | cb_out_depth * block_rows * wt_chunk  (allocated)  \|  out_shard_h_tiles * out_shard_w_tiles (aliased, native_out) | {"value": "cb_out_depth (== cb_x_depth on every observed build)", "knob": "CB_DEPTH_CANDIDATES", "policy": "double / single, same rule as cb_input_tiles"} | Bfp8_b/Float16_b/Float32 | block_rows * wt_chunk | row (tile-rows): spans -> block_rows; width (width tiles): spans -> wt_chunk | compute gamma_mul (C13) or scale (C12) when gamma is absent | writer noc_async_write_tile (TILE) / compute untilize (C14, ROW_MAJOR) / nothing on the native path (the pages ARE the tensor) | pass B only | allocated, or aliased -> output_tensor's L1 shard when plan.native_out | NOT SHARED: aliased onto the tensor on the native path; concurrent with cb_input_tiles otherwise. | rms_norm_program_descriptor.py:2280 (allocated) / rms_norm_program_descriptor.py:2278 (aliased) |
| cb_output_sticks | 9 | untilized ROW_MAJOR stick staging of the output | layout == ROW_MAJOR | CB_RM_STAGE_DEPTH * wt_chunk | {"value": 2, "knob": "CB_RM_STAGE_DEPTH", "policy": "double (reader <-> tilize overlap)"} | Float16_b/Float32 | wt_chunk | row (tile-rows): streams -> TILE_DIM sticks per pop; width (width tiles): spans -> wt_chunk | compute untilize (C14) | writer (write_sticks_after_untilize, or write_band) | pass B only | allocated | NOT SHARED with cb_input_sticks. No reason recorded; see audits.disjoint_lifetime. | rms_norm_program_descriptor.py:2240 |
| cb_sum_handoff | 10 | fp32 raw per-row partial sums; pass A's reduce packs straight into it | plan.combine | CB_ROW_STAT_DEPTH * block_rows | {"value": 2, "knob": "CB_ROW_STAT_DEPTH", "policy": "double, in units of block_rows"} | Float32 | block_rows | row (tile-rows): spans -> block_rows; width (width tiles): streams -> nothing | compute reduce (C4, as CB_REDUCE_ACC) | compute member_pack (C5) on the compact path; the writer's gather (W3/W9/W12) on the identity path | pass A through the gather of one row-block | allocated | NOT SHARED: deliberately distinct from cb_row_stat so a dataflow reader is not a second consumer of the compute accumulator; and cb_row_stat is not even allocated here (D27). | rms_norm_program_descriptor.py:2307 |
| cb_partials_gathered | 11 | fp32 landing slots: one page per sender (flat root), or the LEVEL-0 ring (slot tree) | plan.combine | flat: group_size + group_size % 2   \|   tree: tree_f0 + tree_f0 % 2 | {"value": 1, "knob": "none -- one round", "policy": "single; deepening the gather ring was a measured regression twice over"} | Float32 | group_size (flat) / tree_f0 (tree) | row (tile-rows): streams -> nothing (flat in block_rows since D27); width (width tiles): streams -> nothing | remote members' writer (W3/W9/W12 ship_partial, a raw noc_async_write into this CB's locally-computed address) | compute root fold (C8) / tree level-0 fold (C7) | one combine round | allocated | NOT SHARED: declared on EVERY core (including INACTIVE ones) so its L1 address is identical everywhere -- that is how a sender computes the root's landing address without the host knowing a CB address. | rms_norm_program_descriptor.py:2341 (flat) / rms_norm_program_descriptor.py:2343 (tree) |
| cb_stat_handoff | 12 | fp32 finalized COMPACT stat, the multicast source | plan.combine | CB_COMBINE_FLAT_DEPTH | {"value": 2, "knob": "CB_COMBINE_FLAT_DEPTH", "policy": "double, FLAT in block_rows"} | Float32 | 1 (ONE tile per round) | row (tile-rows): streams -> block_rows (one compact tile carries all block_rows stats); width (width tiles): streams -> nothing | compute root fused fold+finalize (C8) | writer mcast send (W7/W11) | one combine round on the root | allocated | NOT SHARED: D24 publishes the root's own stat copy before the broadcast and needs the other half of the ring untouched. | rms_norm_program_descriptor.py:2348 |
| cb_row_final | 13 | fp32 per-row 1/rms as pass B's Col operand: the un-permute's output (compact) or the multicast landing (identity) | plan.combine | CB_ROW_STAT_DEPTH * block_rows | {"value": 2, "knob": "CB_ROW_STAT_DEPTH", "policy": "double, in units of block_rows"} | Float32 | block_rows | row (tile-rows): spans -> block_rows; width (width tiles): streams -> nothing | compute recv_unpack (C9) on the compact path; the writer's multicast receive (W8/W13) on the identity path | compute pass B scale (C12), as CB_STAT_B | the finalize through pass B of one row-block | allocated | NOT SHARED. No reason recorded. | rms_norm_program_descriptor.py:2351 |
| cb_bank | 14 | bf16 one-hot permutation bank E_r (page r carries a single exact 1.0 at [0][r]) | plan.combine and block_rows > 1 (compact_combine) | block_rows | {"value": 1, "knob": "none", "policy": "single"} | Float16_b | block_rows (all pages live simultaneously -- never popped) | row (tile-rows): spans -> block_rows; width (width tiles): streams -> nothing | reader (R3 reader_bank_boot, hand-rolled L1 stores) | compute member_pack (C5) and recv_unpack (C9), as a matmul operand | whole kernel; never popped | allocated | NOT SHARED: never popped, so it is live for the whole kernel. | rms_norm_program_descriptor.py:2367 |
| cb_compact_handoff | 15 | fp32 this core's COMPACT partial, the gather source | plan.combine and block_rows > 1 | CB_COMBINE_FLAT_DEPTH | {"value": 2, "knob": "CB_COMBINE_FLAT_DEPTH", "policy": "double, FLAT in block_rows"} | Float32 | 1 | row (tile-rows): streams -> block_rows (one tile carries the whole block); width (width tiles): streams -> nothing | compute member_pack (C5) | writer gather ship (W3/W9/W12) | one combine round | allocated | NOT SHARED: single producer, single consumer, across a kernel boundary. | rms_norm_program_descriptor.py:2355 |
| cb_mcast_in | 16 | fp32 multicast landing of the COMPACT stat | plan.combine and block_rows > 1 | CB_COMBINE_FLAT_DEPTH | {"value": 2, "knob": "CB_COMBINE_FLAT_DEPTH", "policy": "double, FLAT in block_rows"} | Float32 | 1 | row (tile-rows): streams -> block_rows; width (width tiles): streams -> nothing | writer mcast send/recv (W7/W8/W11/W13) | compute recv_unpack (C9) | one combine round | allocated | NOT SHARED: declared on ALL cores of the mcast box (including the INACTIVE ones a non-rectangular width-shard grid drags in) so its L1 address is identical everywhere. | rms_norm_program_descriptor.py:2360 |
| cb_gather_l1 | 17 | fp32 the ROOT's LEVEL-1 landing ring | plan.combine and _combine_tree_arity is not None | tree_f1 + tree_f1 % 2 | {"value": 1, "knob": "none -- one round", "policy": "single; deepening the gather ring was a measured regression twice over"} | Float32 | tree_f1 | row (tile-rows): streams -> nothing; width (width tiles): streams -> nothing | level-0 gatherers' writer (W5 writer_tree_forward) | compute root fused fold (C8) | one combine round | allocated | NOT SHARED: same identical-L1-address requirement as cb_partials_gathered. | rms_norm_program_descriptor.py:2344 |
| cb_node_out | 18 | fp32 an interior gatherer's RAW folded sum (no finalize -- only the last level rsqrts) | plan.combine and _combine_tree_arity is not None | CB_COMBINE_FLAT_DEPTH | {"value": 2, "knob": "CB_COMBINE_FLAT_DEPTH", "policy": "double, FLAT in block_rows"} | Float32 | 1 | row (tile-rows): streams -> nothing; width (width tiles): streams -> nothing | compute tree level-0 fold (C7) | writer tree forward (W5) | one combine round | allocated | NOT SHARED. | rms_norm_program_descriptor.py:2345 |

Per-CB notes:

- **cb_input_sticks** — page_size is a TILE of the input dtype even though the buffer holds sticks -- the ring is sized in tile-columns. Page format is whatever the carried tensor's dtype is, so Float32 is reachable: {ROW_MAJOR, float32} is in SUPPORTED and is not in feature_spec.py's INVALID (only bfloat8_b x ROW_MAJOR is). No case in the set builds it, so the Float32 value is static evidence; Float16_b is descriptor-observed. Bfp8_b correctly cannot appear.
- **cb_scaler** — page_format is Float16_b on EVERY build regardless of input dtype -- the scaler value 1.0 and the 0/1 mask are exact in bf16.
- **cb_row_stat** — Capacity is 2x the live set for a CORRECTNESS reason (D6), not overlap: transform_in_place ROTATES the ring, so a PARTIAL final row-block's finalized tiles would straddle the wrap at depth 1.
- **cb_gamma_sticks** — Stays CHUNKED even under ROW_RESIDENT -- the tilize consumes it a chunk at a time into the whole-row cb_gamma_tiles. Page format is whatever the carried tensor's dtype is, so Float32 is reachable: {ROW_MAJOR, float32} is in SUPPORTED and is not in feature_spec.py's INVALID (only bfloat8_b x ROW_MAJOR is). No case in the set builds it, so the Float32 value is static evidence; Float16_b is descriptor-observed. Bfp8_b correctly cannot appear.
- **cb_normalized** — PASS_B_BLK is the DEST-lane block -- the PerBlockSize reserve/push GRANULARITY (kernels/rms_norm_compute.cpp:707, 714-715) -- not the residency. The capacity is exactly the live set, so this CB is NOT an over-allocation and correctly does not appear in audits.capacity_vs_live_set (over).
- **cb_output_sticks** — Page format is whatever the carried tensor's dtype is, so Float32 is reachable: {ROW_MAJOR, float32} is in SUPPORTED and is not in feature_spec.py's INVALID (only bfloat8_b x ROW_MAJOR is). No case in the set builds it, so the Float32 value is static evidence; Float16_b is descriptor-observed. Bfp8_b correctly cannot appear.
- **cb_sum_handoff** — Capacity is 2x the live set so D25's pipelined pass A can pack block blk+1's partial while the writer still ships block blk's.
- **cb_partials_gathered** — Rounded UP TO EVEN so the root's fold is a universal PAIRWISE DEST walk; the pad slot is boot-zeroed and adds an exact +0.0.
- **cb_row_final** — The 2x factor is inherited from CB_ROW_STAT_DEPTH; no reason specific to cb_row_final is recorded. See audits.capacity_vs_live_set.
- **cb_bank** — bf16 because the one-hot is EXACT there; ONE bank serves both directions via matmul's srcB transpose flag.

### Symbol table

| symbol | bound | established by | solved by |
|---|---|---|---|
| block_rows | 1 <= block_rows <= min(max_rows_per_core, brmax); additionally <= TILE_DIM (32) on any combine path | _solve_blocking's L1 fit `brmax` plus `if plan.combine: max_rows = min(max_rows, TILE_DIM)` | rms_norm_program_descriptor.py:1883 |
| wt_chunk | a DIVISOR of wt_per_core (D1); 1 <= wt_chunk <= wt_per_core | _largest_divisor_at_most(wt_per_core, room) -- three helper mechanisms require a uniform chunk | rms_norm_program_descriptor.py:1883 |
| num_w_chunks | wt_per_core // wt_chunk; == 1 on every combine path (asserted on host and static_asserted in the writer) | rms_norm_program_descriptor.py:2117 (assert not (combine and num_w_chunks > 1)) and kernels/rms_norm_writer.cpp:177 (its static_assert mirror) | rms_norm_program_descriptor.py:1883 |
| x_hold_wt | wt_per_core when x_resident, else wt_chunk | rms_norm_program_descriptor.py:2142 (with its assert at :2143) | rms_norm_program_descriptor.py:1883 |
| x_squared_wt | 1 under the DEST fold, else wt_chunk (asserted: `x_squared_wt in (1, wt_chunk)`) | rms_norm_program_descriptor.py:2163 (with its assert at :2487) and kernels/rms_norm_compute.cpp:616-619 | — |
| wt_per_core | Wt on SCHEME_ROWS / SCHEME_SHARD_H; the shard's tile width on SCHEME_SHARD_W; Wt // gw on the interleaved width split; the widest global tile span any band touches on SCHEME_SHARD_W+BAND | rms_norm_program_descriptor.py:1240 (_Plan.wt_per_core) | — |
| cb_x_depth / cb_out_depth | a member of CB_DEPTH_CANDIDATES (shipped: (2,)), forced to 1 on the ROW_MAJOR path and permitted to drop to 1 by the ROW_RESIDENT search | rms_norm_program_descriptor.py:1936 (depth_candidates, forced to (1,) on the ROW_MAJOR path) and rms_norm_program_descriptor.py:2062-2063 (the ROW_RESIDENT depth walk that may drop to 1) | rms_norm_program_descriptor.py:1883 |
| group_size | 1 when there is no combine; <= WIDTH_SPLIT_MAX_GROUP_CORES (16) on an INTERLEAVED input; the shard grid's core count on a sharded input, hence <= the device grid (110 on this part). Observed max 96. | rms_norm_program_descriptor.py:1276 (_width_group_cores, interleaved) / rms_norm_program_descriptor.py:1700 (WIDTH shard: len(shard_cores)) / rms_norm_program_descriptor.py:1731 (BLOCK shard: nx) | — |
| tree_f0 / tree_f1 | 0 (flat) or (COMBINE_TREE_F0, ceil(group_size / COMBINE_TREE_F0)); the tree is taken only when f1 >= 2 and group_size - f0 - f1 >= 18, so a FLAT cb_partials_gathered is bounded BY CONSTRUCTION at group_size <= 29 (30 pages): at 29, f1 = ceil(29/4) = 8 and deleted = 29 - 4 - 8 = 17 < 18, while 30 is the first group where the tree fires. The observed maximum flat group is 28 (c22) | rms_norm_program_descriptor.py:1013 | rms_norm_program_descriptor.py:1013 |
| scaler_pages | 2 when kernel_partial_w != 0, else 1 | rms_norm_program_descriptor.py:1880 | — |
| kernel_partial_w | 0 <= kernel_partial_w < TILE_DIM (32). It is W % 32 on every scheme EXCEPT SCHEME_SHARD_W+BAND, where it is forced to 0 -- the band stages at its REAL element width into a boot-zeroed ring, so its pad lanes contribute an exact 0 and a per-core band boundary is not expressible as one program-wide PARTIAL_W anyway. | rms_norm_program_descriptor.py:1876 (kernel_partial_w = 0 if plan.band else partial_w), with partial_w = W % TILE_DIM at rms_norm_program_descriptor.py:1796 | — |
| shard_h_tiles * shard_w_tiles (in_shard_pages / out_shard_pages) | the resident shard's own tile extent; costs ZERO arena bytes (the CB aliases the tensor's L1) but is charged to the budget through plan.l1_reserved | rms_norm_program_descriptor.py:1094 (_shard_l1_bytes) and rms_norm_program_descriptor.py:2245 / rms_norm_program_descriptor.py:2277 (in_shard_pages / out_shard_pages) | — |
| PASS_B_BLK (kernel-side) | 1 <= PASS_B_BLK <= ckl::DEST_AUTO_LIMIT, and a DIVISOR of WT_CHUNK (so every outer iteration is full). DEST_AUTO_LIMIT is 8 at fp32_dest_acc_en=False and 4 at True, which makes this the op's ONLY fp32_dest_acc_en-dependent block quantity. Clamps to 1 at WT_CHUNK == 1. | kernels/rms_norm_compute.cpp:707 (PASS_B_BLK = pass_b_blk(WT_CHUNK, ckl::DEST_AUTO_LIMIT)) | — |

- **kernel_partial_w** — Named by cb_scaler's capacity (scaler_pages = 2 if kernel_partial_w else 1) and baked as reader ct[5] / compute ct[4] PARTIAL_W. It also gates the DEST square fold (square_datapath).
- **PASS_B_BLK (kernel-side)** — Not host-visible: it is derived in the compute kernel from a build-flag constant, so the descriptor walk cannot read it and every eval is null. It does not enter any CB capacity -- only the reserve/push granularity of cb_normalized / cb_output_tiles under the PerBlockSize lifecycle.

### Evaluation table — capacity in pages, per case

| CB | c01 | c02 | c03 | c04 | c05 | c06 | c07 | c08 | c09 | c10 | c11 | c12 | c13 | c14 | c15 | c16 | c17 | c18 | c19 | c20 | c21 | c22 | c23 | c24 | c25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cb_input_sticks | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 8 | 4 | — | — | — | — | — | 2 | 2 | — |
| cb_input_tiles | 8 | 8 | 16 | 18 | 6 | 8 | 4 | 64 | 16 | 192 | 224 | 172 | 28 | 8 | 8 | 4 | 2 | 16 | 4 | 128 | 5 | 8 | 7 | 127 | 4 |
| cb_x_squared | 1 | 1 | 1 | 9 | 3 | 4 | 1 | 32 | 1 | 96 | 56 | 86 | 14 | 1 | 1 | 1 | 2 | 16 | 1 | 20 | 1 | 1 | 7 | 1 | 1 |
| cb_scaler | 1 | 1 | 1 | 1 | 2 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| cb_row_stat | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | — | 6 | 2 | 2 | — | 2 | 2 | 2 | 2 | 2 | — | — | — | — | — | 2 | — |
| cb_gamma_sticks | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 4 | 2 | — | — | — | — | — | 1 | 1 | — |
| cb_gamma_tiles | 4 | — | 8 | 9 | 3 | 4 | 2 | 32 | 8 | 32 | 224 | 86 | 14 | 4 | 4 | 4 | 2 | 16 | 4 | 4 | 5 | 8 | 1 | 127 | — |
| cb_normalized | 4 | — | 8 | 9 | 3 | 4 | 2 | 32 | 8 | 96 | 56 | 86 | 14 | 4 | 4 | 4 | 2 | 16 | 4 | 80 | 5 | 8 | 7 | 1 | — |
| cb_output_tiles | 8 | 8 | 16 | 18 | 6 | 8 | 4 | 64 | 16 | 192 | 56 | 172 | 28 | 8 | 8 | 4 | 2 | 16 | 4 | 128 | 5 | 8 | 7 | 1 | 4 |
| cb_output_sticks | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 8 | 4 | — | — | — | — | — | 2 | 2 | — |
| cb_sum_handoff | — | — | — | — | — | — | — | — | 2 | — | — | — | 2 | — | — | — | — | — | 2 | 40 | 2 | 2 | 14 | — | 2 |
| cb_partials_gathered | — | — | — | — | — | — | — | — | 16 | — | — | — | 16 | — | — | — | — | — | 8 | 8 | 4 | 28 | 4 | — | 8 |
| cb_stat_handoff | — | — | — | — | — | — | — | — | 2 | — | — | — | 2 | — | — | — | — | — | 2 | 2 | 2 | 2 | 2 | — | 2 |
| cb_row_final | — | — | — | — | — | — | — | — | 2 | — | — | — | 2 | — | — | — | — | — | 2 | 40 | 2 | 2 | 14 | — | 2 |
| cb_bank | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 20 | — | — | 7 | — | — |
| cb_compact_handoff | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2 | — | — | 2 | — | — |
| cb_mcast_in | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2 | — | — | 2 | — | — |
| cb_gather_l1 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 8 | — | 24 | — | — |
| cb_node_out | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 2 | — | 2 | — | — |

### Per-core footprint

```
per-core CB arena bytes = [not is_tile] 2 * CB_RM_STAGE_DEPTH * wt_chunk * bt + [not native_in] cb_x_depth * block_rows * x_hold_wt * bt + block_rows * x_squared_wt * bt + scaler_pages * st + [not combine] CB_ROW_STAT_DEPTH * block_rows * ft + [has_gamma] (x_hold_wt * gt + [gamma_is_rm] wt_chunk * gt + block_rows * wt_chunk * bt) + [not native_out] cb_out_depth * block_rows * wt_chunk * bt + [combine] (CB_ROW_STAT_DEPTH * block_rows * ft   (cb_sum_handoff)            + CB_ROW_STAT_DEPTH * block_rows * ft   (cb_row_final)            + (group_size + group_size%2) * ft      (flat)  OR  ((f0+f0%2) + (f1+f1%2) + CB_COMBINE_FLAT_DEPTH) * ft  (tree)            + CB_COMBINE_FLAT_DEPTH * ft            (cb_stat_handoff)            + [block_rows > 1] (2 * CB_COMBINE_FLAT_DEPTH * ft + block_rows * st))
```

| tile-size symbol | meaning |
|---|---|
| bt | ttnn.tile_size(input dtype) |
| gt | ttnn.tile_size(gamma dtype) |
| st | ttnn.tile_size(bfloat16) == 2048 |
| ft | ttnn.tile_size(float32) == 4096 |

Which terms scale with which knob:

| scales with | CBs |
|---|---|
| block_rows | cb_input_tiles, cb_x_squared, cb_normalized, cb_output_tiles, cb_row_stat, cb_sum_handoff, cb_row_final, cb_bank |
| wt_chunk | cb_input_sticks, cb_output_sticks, cb_gamma_sticks, cb_normalized, cb_output_tiles |
| x_hold_wt | cb_input_tiles, cb_gamma_tiles |
| x_squared_wt | cb_x_squared |
| group_size | cb_partials_gathered (flat path only) |
| tree_f0 / tree_f1 | cb_partials_gathered, cb_gather_l1 (tree path) |
| nothing | cb_scaler, cb_stat_handoff, cb_compact_handoff, cb_mcast_in, cb_node_out |

**Budget** — `int(max(0, get_max_worker_l1_unreserved_size() - (plan.l1_reserved + L1_CB_ARENA_BASE_RESERVE if plan.l1_reserved else 0)) * L1_SAFETY_FRACTION)`  (device worker-L1 unreserved on this part: 1531904 B)

| case | c01 | c02 | c03 | c04 | c05 | c06 | c07 | c08 | c09 | c10 | c11 | c12 | c13 | c14 | c15 | c16 | c17 | c18 | c19 | c20 | c21 | c22 | c23 | c24 | c25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arena bytes ALLOCATED | 61440 | 45056 | 110592 | 139264 | 55296 | 69632 | 36864 | 468992 | 192512 | 1271808 | 1271808 | 1243136 | 292864 | 112640 | 37440 | 86016 | 53248 | 108544 | 77824 | 641024 | 106496 | 176128 | 348160 | 546816 | 61440 |
| incl. aliased CBs | 61440 | 45056 | 110592 | 139264 | 55296 | 69632 | 36864 | 468992 | 192512 | 1271808 | 1271808 | 1243136 | 292864 | 112640 | 37440 | 86016 | 53248 | 174080 | 94208 | 1165312 | 126976 | 208896 | 348160 | 546816 | 77824 |

### The four ledger audits

| audit | result | detail |
|---|---|---|
| capacity_vs_live_set (over) | FINDING | cb_scaler is allocated scaler_pages = 2 whenever kernel_partial_w != 0, but the live set is scaler_tiles = 1 on the AccumulateViaAdd datapath (its partial-W mechanism is a single 0/1 mask tile, not the [full, partial] scaler pair). The host asserts scaler_tiles <= scaler_pages and never shrinks the allocation. Observed at c06 (PARTIAL_W=8, REDUCE_ACC_VIA_ADD=1, cb_scaler 2 pages / 4096 B, SCALER_TILES=1): one 2048-byte page per core is allocated and never pushed. No reason is recorded at the site. |
| capacity_vs_live_set (over) | FINDING | cb_row_final is CB_ROW_STAT_DEPTH * block_rows pages against a live set of block_rows. cb_row_stat's 2x has a recorded CORRECTNESS reason (D6, ring rotation) and cb_sum_handoff's has a recorded PIPELINE reason (D25); cb_row_final reuses the same constant with no reason of its own recorded. At c20 (block_rows 20) that is 20 spare fp32 pages = 81920 B per core on 64 cores. |
| capacity_vs_live_set (under) | PASS | No CB whose axis accounting says `spans` an axis fails to scale with it. The two that look collapsed are documented: cb_x_squared spans width at x_squared_wt (== 1 only under the DEST fold, where the live set genuinely is one tile), and cb_stat_handoff / cb_compact_handoff / cb_mcast_in are FLAT in block_rows because D27's compact tile carries all block_rows stats in its columns. |
| page_format_vs_dest_width (over) | FINDING | cb_row_stat, cb_sum_handoff, cb_partials_gathered, cb_stat_handoff, cb_row_final, cb_gather_l1 and cb_node_out are Float32 on EVERY build, including the ELEVEN cases with fp32_dest_acc_en == False (c10, c11, c12, c13, c19, c20, c21, c22, c23, c24, c25). Reasons ARE recorded -- D5 for cb_row_stat (it is the cross-chunk accumulator that reduce()'s Accumulate reloads, so fp32 keeps the STREAM reload lossless even when DEST is 16-bit) and D11 for the combine chain (the cross-core sum is itself an fp32 elementwise add) -- but the reference's audit-2 `over` direction is stated as a finding no stated reason rescues, so it is recorded as one here rather than adjudicated. |
| page_format_vs_dest_width (under) | FINDING | cb_x_squared and cb_normalized are pure INTERMEDIATES (no user tensor corresponds to them) yet carry the INPUT dtype -- 16-bit at bfloat16/bfloat8_b -- on the FOURTEEN cases with fp32_dest_acc_en == True (c01-c09 and c14-c18; thirteen of them also carry gamma, which is where the earlier count of eleven came from). The audit's `under` direction calls a 16-bit page under fp32 DEST a correctness finding: the x^2 values and the normalized values are truncated to 16 bits at those phase boundaries. The recorded rule (D5) is that every CB declares data_format = the dtype of the tensor it carries; for an intermediate that rule picks the input dtype by convention, not by a precision argument. |
| disjoint_lifetime_no_justification | FINDING | There is no ledger in the tree at all (no l1_ledger.md for any op), so no CB carries a `Shares with / why not` cell. Three pairs have plainly disjoint lifetimes and identical page formats with no recorded sharing decision: (cb_x_squared, cb_normalized) -- pass A vs pass B, both the input dtype; (cb_input_sticks, cb_output_sticks) -- pass A staging vs pass B staging, both a tile of the input/output dtype; (cb_gamma_sticks, cb_input_sticks) on the RM path when X_RESIDENT, since cb_gamma_sticks is consumed once at boot. D25's pipeline (block blk+1's pass A issued before block blk's combine) would foreclose the first pair, but that pipeline is carved out to native_in only, so it does not cover the interleaved builds. |
| bounds_and_closed_form | PASS | Every capacity is closed-form in the 13 symbols of the symbol table plus five named depth constants; every symbol has a bound and the predicate establishing it. Six of them are produced by a bounded search rather than a formula (_solve_blocking at rms_norm_program_descriptor.py:1883, _combine_tree_arity at rms_norm_program_descriptor.py:1013) and are recorded as such with an evaluation table over 25 cases. The one bound that is not stated at its own site is group_size on a SHARDED input: it is the shard grid's core count, bounded only by the device grid, and the flat cb_partials_gathered ring is linear in it -- the slot tree caps the RING at f0 = 4 above deleted >= 18, which bounds a FLAT ring at group_size <= 29 (30 pages) BY CONSTRUCTION rather than by declaration: at group_size 29, f1 = ceil(29/4) = 8 and deleted = 29 - 4 - 8 = 17, one short of the threshold, so the flat path survives; group_size 30 (deleted 18) is the first that takes the tree. 28 (c22) is the observed maximum, not the constructed one. |

## §6 Memory traffic

### SCHEME_ROWS / RESIDENT or ROW_RESIDENT  _(unit: per core, per whole assignment)_

| tensor | DRAM crossings | why that many | cross-core traffic added |
|---|---|---|---|
| x | 1 (once per element) | held in cb_input_tiles across both passes (x_resident); ROW_RESIDENT re-INDEXES the held CB at a TileOffset instead of re-reading | none |
| gamma | 1 per CORE (num_cores * W elements total) | a broadcast operand read from DRAM by every core; the read-once-and-multicast successor is not built | none |
| out | 1 | written once per element | none |

**Minimum:** against DRAM: x once + out once + gamma once. This regime is at the minimum for x and out, and (num_cores - 1) whole-gamma reads above it for gamma.

### SCHEME_ROWS / STREAM  _(unit: per core, per whole assignment)_

| tensor | DRAM crossings | why that many | cross-core traffic added |
|---|---|---|---|
| x | 2 (NUM_PASSES == 2 -- one read per pass) | not enough L1 to hold even one tile-row of x, so pass B re-reads it | none |
| gamma | num_blocks_this_core * NUM_W_CHUNKS chunk reads, i.e. the whole row per row-block per core | gamma is chunked and not held; the reader stages it inside the pass-B chunk loop | none |
| out | 1 | written once | none |

**Minimum:** same DRAM minimum. This regime is roughly 2x the minimum on x and num_blocks x above it on gamma. It is an L1 fallback, not a parallelization.

### SCHEME_SHARD_H  _(unit: per core, per whole assignment)_

| tensor | DRAM crossings | why that many | cross-core traffic added |
|---|---|---|---|
| x | 0 | cb_input_tiles aliases the resident L1 shard; the reader only PUBLISHES the pages so cb_wait_front can see them | none |
| gamma | 1 per core | same broadcast-operand replication | none |
| out | 0 when the output shard spec matches the input's, else 1 | cb_output_tiles aliases the output shard; compute packs straight into the tensor | none |

**Minimum:** below the stated DRAM minimum for x and out -- the tensors are already core-local L1 (tier 1, free).

### SCHEME_SHARD_W (sharded input)  _(unit: per core; the cross-core term is per COMBINE ROUND, and rounds = ceil(rows_per_core / block_rows))_

| tensor | DRAM crossings | why that many | cross-core traffic added |
|---|---|---|---|
| x | 0 | resident shard, zero-copy CB | none for x itself |
| gamma | 1 per core, of this core's width slice only (wt_chunk tiles, trimmed to face-rows or a half page) | each core needs only the weights for the band it owns | none |
| out | 0 when native_out | zero-copy output CB | none |
| the partial sum-of-squares | 0 | never leaves L1 | FLAT: (group_size - 1) unicast writes of one 4096 B fp32 tile into the root's cb_partials_gathered, plus one multicast of one 4096 B tile to group_size - 1 receivers, per round, gated on one arrival semaphore. TREE: (group_size - 1) level-0 unicasts + (f1 - 1) level-1 unicasts + the same multicast, on two semaphores -- one extra hop, but the root's ingress and fold drop from group_size to max(f0, f1). |

**Minimum:** the cross-core term is structurally unavoidable: the reduce axis is cut across cores, so partial sums must be combined. The identity path halves the gather bytes by shipping GATHER_FACES = 2 of a tile's 4 faces.

### SCHEME_SHARD_W (interleaved input, the AUTO width split)  _(unit: per core; cross-core term per round)_

| tensor | DRAM crossings | why that many | cross-core traffic added |
|---|---|---|---|
| x | 1, of this core's width slice only (Wt // gw tiles per row) | an interleaved tensor has no resident per-core slice, so x still arrives through a TensorAccessor | none for x |
| gamma | 1 per core, of this core's slice | as above | none |
| out | 1 | written through the accessor | none |
| the partial sum-of-squares | 0 | never leaves L1 | identical to the sharded SCHEME_SHARD_W row |

**Minimum:** at the DRAM minimum for x and out (each element crosses once), plus the combine. The split is taken only when it puts WIDTH_SPLIT_MIN_GAIN (4) times as many cores to work.

### SCHEME_SHARD_W+BAND  _(unit: per core; cross-core term per round)_

| tensor | DRAM crossings | why that many | cross-core traffic added |
|---|---|---|---|
| x | 0 | staged core-locally out of this core's own resident ROW_MAJOR shard into the tilize ring -- tier 1 (free) rather than DRAM. One wide transfer per tile-row when the band fills its tile columns and the shard stride matches, else one per stick. | none |
| gamma | 1 per core, fetched on a TILE COLUMN of the tensor's global tile frame | an unaligned DRAM read is silently truncated to the alignment, so the staging frame is the global tile grid, not the band's own byte offset | none |
| out | 0 when band_out_local (the out shard matches the in shard), else 1 through the accessor | written back stick-by-stick into the core's own L1 shard | none |
| the partial sum-of-squares | 0 | never leaves L1 | identical to the sharded SCHEME_SHARD_W row |

**Minimum:** below the DRAM minimum for x and out. A core whose band spans fewer tiles than wt_chunk stages an all-zero pad tile column, which the reduce adds 0 for and the writer never writes back.

## §7 Optional-input handling

| input | optionality expressed as | absent: buffers not allocated | absent: buffers still sized for it | absent: bytes | absent: CT/RT args | evidence | ablation pairs |
|---|---|---|---|---|---|---|---|
| gamma | host-side inventory change PLUS compile-time specialisation: has_gamma removes cb_gamma_tiles / cb_gamma_sticks / cb_normalized from the CB list entirely, and HAS_GAMMA (reader ct[6], compute ct[5]) is an `if constexpr` in both kernels | cb_gamma_tiles (6), cb_normalized (7), cb_gamma_sticks (5) when gamma is ROW_MAJOR | **none** | 0 | reader ct[6] HAS_GAMMA 1->0, ct[9] GAMMA_ELEM_BYTES 2->0, ct[19] GAMMA_TRIM 2->0, and the gamma TensorAccessorArgs collapse to the empty-accessor form -- SAME argument count (25->25 interleaved, 33->33 sharded), values zeroed. compute ct[5] HAS_GAMMA 1->0. Writer CT args are byte-identical. | descriptor | c01/c02; c19/c25 |

- **gamma — absent-cost detail.** Zero bytes are allocated for gamma when it is absent, and nothing else is sized for it. The pair (c01 with gamma, c02 without) differs by exactly 16384 B of arena on an otherwise identical cell -- that is the COST OF PRESENCE, not a residue of absence. The same pair on the combine path (c19 / c25) differs by the same 16384 B.
- **gamma — phases elided.** C1 compute_gamma_tilize, C11 compute_gamma_tilize_b, C13 compute_gamma_mul, R4 reader_read_gamma
- **gamma** — With gamma absent, pass B's scale writes straight into cb_output_tiles (NORM_OUT = HAS_G ? cb_normalized : cb_output_tiles), so the intermediate disappears rather than being bypassed.

## §8 Kernel phase sequence

### `rms_norm_reader` — **instrumented** (`MaybeDeviceZoneScope`, source order)

| # | zone | line | branch | loop nesting | guarded_by | waits on | pushes / pops |
|---|---|---|---|---|---|---|---|
| 1 | reader_scaler_boot | 196 | — | once at boot | none (unconditional); the body branches on REDUCE_ACC_VIA_ADD then PARTIAL_W | cb_scaler (reserve, inside prepare_reduce_mask / prepare_reduce_scaler / prepare_partial_reduce_scalers) | cb_scaler (1 tile, or 2 on ReduceTile + partial W) |
| 2 | reader_stage_zero | 220 | — | once at boot | if constexpr (RM && STAGE_ZERO != 0) | — | — |
| 3 | reader_bank_boot | 264 | — | once at boot | if constexpr (BANK_PAGES != 0)  [i.e. the COMPACT combine path] | cb_bank (reserve BANK_PAGES) | cb_bank (BANK_PAGES; never popped by anyone) |
| 4 | reader_read_gamma | 294 | — | TWO call sites: (a) X_RESIDENT -- once at boot, per w-chunk (loop at :381); (b) !X_RESIDENT -- per row-block, per pass-B w-chunk (:505, guarded by pass == 1) | if constexpr (HAS_G); call-site if constexpr (X_RESIDENT) vs if constexpr (!X_RESIDENT) && pass == 1; inner if constexpr (G_RM) and GAMMA_TRIM 2/1/else | cb_gamma_sticks (RM); cb_gamma_tiles (TILE, reserve WT_CHUNK) | cb_gamma_sticks (RM, plus a ragged top-up push); cb_gamma_tiles (TILE) |
| 5 | reader_native_publish | 393 | — | once at boot | if constexpr (NATIVE_X); inner runtime if (w_real < WT_CHUNK) zeroes the shard's pad tiles | cb_input_tiles (reserve IN_SHARD_PAGES) | cb_input_tiles (IN_SHARD_PAGES) |
| 6 | reader_read_x | 460 | — | per row-block, per pass, per w-chunk (NUM_PASSES = X_RESIDENT ? 1 : 2) | none -- the zone opens BEFORE the branch, so it is emitted on every path; the body then branches NATIVE_X (immediate return) / RM -> BAND_X / TILE | cb_input_sticks (RM / BAND); cb_input_tiles (TILE, reserve WT_CHUNK per tile-row) | cb_input_sticks; cb_input_tiles |

- **reader_stage_zero (#2)** — raw whole-ring zero of cb_input_sticks; no push/pop, only write_zeros_l1_barrier
- **reader_native_publish (#5)** — there is no NoC read for x at all -- the reader only PUBLISHES the resident pages so cb_wait_front can see them
- **reader_read_x (#6)** — On the NATIVE_X path the zone opens and the body immediately returns, so the profiler reports reader_read_x once per (block x pass x chunk) with essentially zero payload.

**Uninstrumented phases in this kernel:** CT/RT decode and TensorAccessor construction (:81-182); the inactive-core early exit if (num_rows == 0) return (:178); the resident-gamma driver loop (:380-384) -- only its per-chunk bodies carry a zone; stage_band (:439-457) -- runs inside reader_read_x's zone; the row-block / pass / chunk loop nest (:497-512).

### `rms_norm_writer` — **instrumented** (`MaybeDeviceZoneScope`, source order)

| alias | resolves to |
|---|---|
| CB_GATHER_SRC | COMPACT ? cb_compact_handoff : cb_sum_handoff |
| CB_MCAST_LAND | COMPACT ? cb_mcast_in : cb_row_final |
| COMPACT | CROSS_CORE && BLOCK_ROWS > 1 |

| # | zone | line | branch | loop nesting | guarded_by | waits on | pushes / pops |
|---|---|---|---|---|---|---|---|
| 1 | writer_write | 287 | — | per row-block (called from all four block loops) | none -- the zone opens before the branch; body returns immediately on NATIVE, else RM -> BAND_OUT / accessor, or TILE; inner runtime if (wt < WT) skips ragged pad tiles | cb_output_sticks (RM / BAND); cb_output_tiles (TILE) | pops cb_output_sticks / cb_output_tiles |
| 2 | writer_gather_zero | 477 | — | once at boot; called 0, 1 or 2 times per core | if constexpr (CROSS_CORE); TREE: l0_gatherer with a ragged run, and the root when TREE_F1 < TREE_SL1. FLAT: is_root when GATHER_SLOTS != GROUP_SIZE (an odd group's pad slot). | — | — |
| 3 | writer_gather_ship | 552 | TREE path, level 0 (every core) | per row-block | if constexpr (CROSS_CORE) && if constexpr (TREE); runtime if (l0_gatherer) for the reserve, if (!l0_gatherer) for the semaphore up | CB_GATHER_SRC (wait 1); cb_partials_gathered (reserve TREE_SL0, only if l0_gatherer) | pops CB_GATHER_SRC |
| 4 | writer_gather_wait | 572 | TREE path | per row-block | TREE + runtime early-return if (!l0_gatherer) | gather_sem.wait_min(arrivals) -- a semaphore, not a CB | cb_partials_gathered (TREE_SL0) |
| 5 | writer_tree_forward | 582 | TREE path, level 1 | per row-block | TREE + l0_gatherer; runtime if (is_root) reserve, else semaphore up | cb_node_out (wait 1); cb_gather_l1 (reserve TREE_SL1, root only) | pops cb_node_out |
| 6 | writer_tree_wait | 599 | TREE path | per row-block | TREE + l0_gatherer + runtime if (is_root) | gather_sem_l1.wait_min(arrivals_l1) | cb_gather_l1 (TREE_SL1) |
| 7 | writer_mcast_send | 616 | TREE path, root | per row-block | if constexpr (TREE) + runtime if (is_root); inner if constexpr (mc.active) | cb_stat_handoff (wait 1); CB_MCAST_LAND (reserve 1) | CB_MCAST_LAND (1); pops cb_stat_handoff |
| 8 | writer_mcast_recv | 635 | TREE path, non-root | per row-block | if constexpr (TREE) + runtime else of is_root | CB_MCAST_LAND (reserve 1); receiver.receive() | CB_MCAST_LAND (1) |
| 9 | writer_gather_ship | 652 | FLAT path, root | per row-block | CROSS_CORE + !TREE + runtime is_root | CB_GATHER_SRC (wait 1); cb_partials_gathered (reserve GATHER_SLOTS) | — |
| 10 | writer_gather_wait | 670 | FLAT path, root | per row-block | !TREE + is_root | gather_sem.wait_min(arrivals) | cb_partials_gathered (GATHER_SLOTS) |
| 11 | writer_mcast_send | 685 | FLAT path, root | per row-block | !TREE + is_root; inner if constexpr (mc.active) | cb_stat_handoff (wait 1); CB_MCAST_LAND (reserve 1) | CB_MCAST_LAND (1) -- pushed BEFORE sender.send() (D24); pops cb_stat_handoff |
| 12 | writer_gather_ship | 735 | FLAT path, non-root member | per row-block | CROSS_CORE + !TREE + !is_root | CB_GATHER_SRC (wait 1) | pops CB_GATHER_SRC |
| 13 | writer_mcast_recv | 751 | FLAT path, non-root member | per row-block | !TREE + !is_root | CB_MCAST_LAND (reserve 1); receiver.receive() | CB_MCAST_LAND (1) |

- **writer_gather_zero (#2)** — raw page zeroing on cb_partials_gathered / cb_gather_l1
- **writer_gather_ship (#9)** — the matching cb_pop_front(CB_GATHER_SRC, 1) sits at :666, OUTSIDE the zone -- asymmetric with the tree and member spellings of the same stage name
- **writer_gather_ship (#12)** — a member never reserves cb_partials_gathered -- it writes the ROOT's ring at a locally-computed get_write_ptr

**Uninstrumented phases in this kernel:** CT/RT decode, TensorAccessor construction and all BAND geometry derivation (:124-252); the inactive-core early exit (:233); Noc / Semaphore construction (:332-341); mc.sender / mc.receiver construction (:609, :631, :644, :724); the FLAT-root cb_pop_front(CB_GATHER_SRC, 1) at :666 -- a real CB operation stranded between two zones; write_band (:253-275) -- runs inside writer_write's zone; the FINAL NATIVE completion barrier cb_wait_front(cb_output_tiles, num_rows * WT_CHUNK) (:767-772) -- a real terminal drain, entirely uninstrumented.

**Phantom-stage rule.** The tree body is written as `if constexpr (!TREE) { return; } else { ... }` rather than a runtime if, for two reasons recorded at :542-545: cb_gather_l1 / cb_node_out are not ALLOCATED off the tree path, and an emitted-but-uncalled MaybeDeviceZoneScope would report a PHANTOM STAGE to the profiler. The same shape appears at member_pack in the compute kernel.

### `rms_norm_compute` — **instrumented** (`MaybeDeviceZoneScope`, source order)

| alias | resolves to |
|---|---|
| CB_REDUCE_ACC | CROSS_CORE ? cb_sum_handoff : cb_row_stat |
| CB_STAT_B | CROSS_CORE ? cb_row_final : cb_row_stat |
| NORM_OUT | HAS_G ? cb_normalized : cb_output_tiles |

**Main loop.** kernel_main at :440. Boot: hw startup (:559), policy constexprs (:561-833), gamma tilize (:846-853). Then the PIPE_A prologue pass_a(rows_of(0),0) (:1006) and the BLOCK LOOP (:1010): [PIPE_A] member_pack -> widened wait -> pass_a(blk+1)  |  [else] pass_a(blk) -> member_pack; then either the local finalize (:1042) or the combine branch (:1048-1265); then the PASS B chunk loop (:1268-1324); then the block epilogue pops (:1326-1333). Kernel epilogue pops cb_scaler and cb_gamma_tiles (:1337-1341).

| # | zone | line | branch | loop nesting | guarded_by | waits on | pushes / pops |
|---|---|---|---|---|---|---|---|
| 1 | compute_gamma_tilize | 849 | — | once at boot (the NUM_W_CHUNKS loop is INSIDE the zone) | if constexpr (HAS_G && X_RESIDENT && G_RM) | cb_gamma_sticks | cb_gamma_tiles |
| 2 | compute_tilize_x | 871 | — | per row-block, per w-chunk (inside pass_a; under PIPE_A the call is shifted to blk+1 plus a prologue call) | if constexpr (RM) | cb_input_sticks | cb_input_tiles |
| 3 | compute_square | 878 | — | per row-block, per w-chunk | none; the SHAPE is switched by SQ_FOLD / SQ_OUT (DestAccumulation::PerRow when x_squared_wt == 1 && wt_chunk > 1) | cb_input_tiles (WaitPolicy::Upfront; PopPolicy None when X_RESIDENT, AtEnd otherwise); cb_x_squared (reserve) | cb_x_squared |
| 4 | compute_reduce | 886 | — | per row-block, per w-chunk | none | cb_x_squared; cb_scaler (read, not popped here); CB_REDUCE_ACC (reserve + reload on later chunks) | CB_REDUCE_ACC |
| 5 | compute_member_pack | 977 | — | per row-block | if constexpr (!COMPACT) { return; } else { ... }  [COMPACT = COMBINE && BLOCK_ROWS > 1] | cb_sum_handoff (rows); cb_compact_handoff (reserve 1); cb_bank (read; never waited, never popped) | cb_compact_handoff (1); pops cb_sum_handoff (rows) |
| 6 | compute_finalize | 1044 | — | per row-block (loops `rows` times inside the zone) | if constexpr (!CROSS_CORE) | cb_row_stat (wait 1, then reserve 1 -- pop BEFORE reserve, so one page suffices) | cb_row_stat (in place) |
| 7 | compute_tree_fold_l0 | 1083 | — | per row-block | CROSS_CORE + if constexpr (TREE) + runtime if (my_slot % TREE_F0 == 0) | cb_partials_gathered (TREE_SL0); cb_node_out (reserve 1) | cb_node_out (1); pops cb_partials_gathered |
| 8 | compute_root_fused | 1186 | — | per row-block | CROSS_CORE + runtime if (is_root); inner if constexpr (TREE) picks the ring; the RMS_ABLATE_* defines select a handshake-only body | cb_gather_l1 (TREE_SL1) or cb_partials_gathered (GATHER_SLOTS); cb_stat_handoff (reserve 1) | cb_stat_handoff (1); pops the in-ring |
| 9 | compute_recv_unpack | 1243 | — | per row-block | CROSS_CORE + if constexpr (COMPACT) | cb_mcast_in (1); cb_row_final (reserve rows); cb_bank (read) | cb_row_final (rows); pops cb_mcast_in |
| 10 | compute_tilize_x_b | 1274 | — | per row-block, per w-chunk | if constexpr (RM && !X_RESIDENT) | cb_input_sticks | cb_input_tiles |
| 11 | compute_gamma_tilize_b | 1278 | — | per row-block, per w-chunk | if constexpr (HAS_G && !X_RESIDENT && G_RM) | cb_gamma_sticks | cb_gamma_tiles |
| 12 | compute_scale | 1287 | — | per row-block, per w-chunk | none | cb_input_tiles (Upfront; PopPolicy None under ROW_RESIDENT, AtEnd otherwise); CB_STAT_B (BroadcastDim::Col, OperandKind::Col, PopPolicy::None); NORM_OUT (reserve, PerBlockSize) | NORM_OUT, per PASS_B_BLK |
| 13 | compute_gamma_mul | 1305 | — | per row-block, per w-chunk | if constexpr (HAS_G) | cb_normalized (Upfront, AtEnd); cb_gamma_tiles (BroadcastDim::Row, PopPolicy::None); cb_output_tiles (reserve, PerBlockSize) | cb_output_tiles; pops cb_normalized |
| 14 | compute_untilize | 1317 | — | per row-block, per w-chunk | if constexpr (RM) | cb_output_tiles; cb_output_sticks (reserve) | cb_output_sticks; pops cb_output_tiles |

- **compute_reduce (#4)** — the zone is UNBRACED -- declared at :886 and live to the end of the `for c` body, so it covers the whole accumulate_reduce_block call
- **compute_root_fused (#8)** — ONE DEST window: the group fold accumulates PAIRWISE and the finalize runs in that same window, one pack (D22)

**Uninstrumented phases in this kernel:** CT/RT decode and the whole constexpr derivation block (:441-833); the inactive-core exit (:496-499), placed 'before any CB or LLK state is touched'; compute_kernel_hw_startup(CB_A, cb_scaler, cb_output_tiles) (:559) -- a real boot step with no zone; the PIPE_A prologue pass_a(rows_of(0), 0) (:1006) -- its inner zones fire but are indistinguishable in the profile from an in-loop pass_a; the pipeline's widened blocking wait cb_wait_front(cb_input_tiles, (rows+next_rows)*X_HOLD_WT) (:1029); pass-B cb_pop_front(cb_gamma_tiles) (:1322); block-epilogue pops (:1331, :1333); kernel-epilogue drain pops (:1338, :1340).

**Retired zone.** compute_partial_handoff, documented at :1049-1052 as 'retired with the copy it measured' (D18 deleted the cb_row_stat -> cb_sum_handoff copy).

## §9 Raw-primitive bypasses

| site | bypasses | stated reason | what would close it |
|---|---|---|---|
| kernels/rms_norm_compute.cpp:203-204 | #include "ckernel_sfpu_sqrt.h" and "ckernel_sfpu_binop_with_unary.h" -- direct LLK SFPU headers under #ifdef TRISC_MATH | The finalize's SFPU ops (mul_unary_tile / add_unary_tile / rsqrt_tile) each hard-code VectorMode::RC and expose NO VectorMode seam; the SFPU walks a face as [rg0-even, rg0-odd, ...], so COLUMN PARITY is the inner walk axis and is unreachable through ITERATIONS. | a VectorMode template parameter or runtime argument on rsqrt_tile / mul_unary_tile / add_unary_tile in api/compute/eltwise_unary/ |
| kernels/rms_norm_compute.cpp:231,235 | _llk_math_eltwise_unary_sfpu_params_(rms_stat_scale_body<2,4> / rms_stat_rsqrt_body<2,4>, idst, VectorMode::C, ...) -- the narrow even-parity finalize | Same seam argument; cb_row_stat's only consumer is mul<BroadcastDim::Col> == column 0, which lives in faces 0 and 2 (VectorMode::C) and is EVEN, so an even-parity walk reaches it in 4 vector ops per face instead of 8. Measured 600.7 -> 244.5 ns MATH-thread per finalize call (2.04x on the stage); safety measured, not assumed (columns 1..31 seeded five orders of magnitude wrong -> pcc 0.999992). | same as above |
| kernels/rms_norm_compute.cpp:264,267 | _llk_math_eltwise_unary_sfpu_params_(<1,8>, VectorMode::C) -- stat_scale_col_full / rsqrt_tile_col_full | A CORRECTNESS REQUIREMENT, not a perf choice: D27's compact combine finalizes ONE tile whose columns 0..block_rows-1 each hold a different tile-row's group sum, so the finalize must visit EVERY one of those columns. The <2,4> even-parity pair is SILENTLY WRONG from block_rows = 2 up (measured pcc 0.9972987 at rel-RMS 1036 against a 0.04 bound -- and 1.39x FASTER, i.e. exactly the sort of 'win' that has to be refused). | same as above |
| kernels/rms_norm_compute.cpp:270,273 | _llk_math_eltwise_unary_sfpu_params_(<1,8>, VectorMode::RC) -- stat_scale_all / rsqrt_tile_all | block_rows > 16 needs columns 16..31, which live in faces 1 and 3. Widening C to RC is a measured flat +452 ns/round, so it is taken only where block_rows actually needs those columns. | same as above |
| kernels/rms_norm_compute.cpp:404-437 (combine_fold) | a hand-rolled DEST window (add_tiles_init(acc_to_dest=true) + add_tiles + rsqrt_tile_init + pack_tile) instead of ckl::eltwise_chain | The fusion is INEXPRESSIBLE through eltwise_chain: DestAccumulation::PerRow gives the right DEST window, but EVERY chain element's apply runs on EVERY inner iteration of that row, so a StatFinalize element after the accumulating BinaryFpu would rsqrt a PARTIAL sum group_size/2 times instead of once on the completed one. There is no apply-after-the-accumulation element kind and no per-row tail hook. The helper-expressible split form measures 1.93x against this 2.18x. | an apply-after-accumulation element kind, or a per-row tail hook, in eltwise_chain |
| kernels/rms_norm_compute.cpp:987-999 (member_pack) and kernels/rms_norm_compute.cpp:1246-1263 (compute_recv_unpack) | raw matmul_init / matmul_tiles / reconfig_data_format<SrcOrder::Reverse> / tile_regs_* / pack_tile -- the compact partial's column permutation and its inverse | A COLUMN PERMUTATION has no kernel_lib expression: the eltwise / bcast / reduce families all preserve or collapse the column axis, and transpose_wh transposes the WHOLE tile, which is a different map. The FPU's only horizontal-mixing primitive is the matmul, so matmul_tiles against a one-hot bank IS the operation -- with matmul_init's srcB transpose flag reading E_r as E_r^T so ONE bank serves both directions. | a column-permute (or gather-columns) block operation in compute_kernel_lib |
| kernels/rms_norm_compute.cpp:88-147 (the retired-upstream note, then rms_norm_local::accumulate_reduce_block and transform_in_place) | two kernel_lib helpers reinstated VERBATIM inside the op, with their raw compute-API bodies (cb_wait_front / tile_regs_* / copy_tile / pack_tile / reconfig) | accumulate_reduce_block and transform_in_place lived in kernel_lib/streaming_reduce_helpers.hpp, which upstream retired ('kernel_lib: drop the streaming-reduce wrappers'). Reinstated verbatim so the call sites are unchanged. | restoring the wrappers upstream, or replacing the two call sites with whatever superseded them |
| kernels/rms_norm_reader.cpp:282 (reader_bank_boot) | hand-rolled single-element L1 stores -- *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(bank_base + byte_off) = BF16_ONE -- into a CB page | No kernel_lib helper writes an arbitrary constant at an arbitrary tile position: l1_helpers.hpp offers zero_tile / prepare_zero_tile, and reduce_helpers_dataflow.hpp's prepare_reduce_scaler / prepare_reduce_mask emit a UNIFORM scaler or a row-0 CONTIGUOUS 0/1 mask -- neither can place a single 1.0 at column r. 'Adding a helper for a one-off boot constant would be a worse trade than 8 lines with the tile layout spelled out.' | a dataflow helper that writes a constant at an arbitrary (row, column) of a tile |
| kernels/rms_norm_reader.cpp:336-371 and kernels/rms_norm_reader.cpp:483-490 | raw TensorAccessor + noc_async_read_tile / noc_async_read for whole-tile interleaved x reads and for every gamma read (including D23's face-row sub-reads) | 'TILE staging + gamma reads are TensorAccessor + noc_async_read_tile: the dataflow tilize helper covers neither whole-tile interleaved reads nor the gamma slot (op_design.md section 6.1).' | a dataflow helper for whole-tile interleaved reads, and one for a trimmed broadcast-vector read |
| kernels/rms_norm_reader.cpp:439-457 (stage_band) | raw cb_reserve_back / get_noc_addr / noc_async_read / noc_async_read_barrier / cb_push_back instead of read_sticks_for_tilize | 'Such a shard's page is a row SEGMENT, so no accessor read can reach a row. Instead each core stages the band it already holds out of its OWN L1.' | a dataflow staging helper that reads from local L1 at a shard stride into a tilize ring |
| kernels/rms_norm_reader.cpp:223 and kernels/rms_norm_reader.cpp:401 and kernels/rms_norm_writer.cpp:481-485 | noc.async_write_zeros + write_zeros_l1_barrier -- the raw device-zero API on a CB ring | Reader: the L1 pad lanes of a staged ROW_MAJOR row are never written by a stick read, so L1 garbage would survive into the reduce (inf*0 / nan*0 = NaN poisons the whole row). Writer: a gather-ring page that NO SENDER EVER WRITES is folded WHOLE into the group sum, so it must be an exact +0.0 -- measured catastrophic without it (rel-RMS 1.00 at pcc 0.999672 for 1e30). Whole-page rather than face-subset zeroing is also measured: 'this stage pays per API CALL, not per byte'. | nothing -- async_write_zeros IS the device zero API; the note is that no CB-level helper wraps it |
| kernels/rms_norm_writer.cpp:333,340,564,594,601,672,742 | a hand-rolled arrival handshake on api/dataflow/noc_semaphore.h, while the multicast half uses the mcast_pipe.hpp helper | It lives in the WRITER because NoC1 is idle through pass A. The no-self-signal rule is recorded: 'Semaphore::up(value) is a NON-ATOMIC local read-modify-write, so a local bump on the root would race the members' remote atomic incs and silently drop one -- a hang.' One semaphore PER TREE LEVEL, because a shared counter would let an early level-1 inc satisfy the root's level-0 wait_min. | a gather/fan-in pipe helper alongside mcast_pipe.hpp |
| kernels/rms_norm_writer.cpp:413-419 (ship_partial) | raw noc_async_write into another core's CB page (address computed locally from the identically-placed CB) | Recorded as the mechanism, with measurement: 'a member ships that ONE WHOLE TILE into page my_slot: one transaction, 4 kB, flat in BLOCK_ROWS ... writer_gather_ship on a member 1891 -> 1087 ns/round.' The identity path ships faces 0 and 2 only; widening it was a whole-op regression MONOTONE in group_size. | the same gather pipe helper |
| kernels/rms_norm_writer.cpp:314-324 (TILE output write) | raw cb_wait_front / get_read_ptr / noc_async_write_tile / cb_pop_front for whole-tile accessor writes | none recorded -- the header states what the path does ('the TILE build moves cb_output_tiles -> whole output tiles') but records no helper-coverage reason, unlike its ROW_MAJOR sibling which uses write_sticks_after_untilize. | a dataflow helper for whole-tile interleaved writes |
| kernels/rms_norm_writer.cpp:253-275 (write_band) | raw cb_wait_front / get_read_ptr / get_noc_addr / noc_async_write / noc_async_write_barrier / cb_pop_front instead of dataflow_kernel_lib::write_sticks_after_untilize, which the sibling branch two lines away calls (:306-307) for exactly this consumer contract. The mirror of bypass #10 (stage_band) on the read half. | 'BAND (Ref. 2b): cb_output_sticks -> this core's own resident ROW_MAJOR shard, one band (w_real_elems elements) per stick, at the shard's own L1 stride. Local L1, no DRAM traffic.' (:14-16) and 'the reader's stage_band, mirrored -- Same transaction granularity as the read half (one per tile-row when the band fills its tile columns and the shard stride matches, one per stick otherwise), so neither NoC half is the batched one. The band sits at lane (w_off_elems % 32) of each untilized stick -- the reader's GLOBAL TILE FRAME, mirrored.' (:242-250) | a write_sticks_after_untilize overload that takes a per-stick BYTE OFFSET into the destination (the band's w_off_elems * elem_bytes) and a destination that may be a raw local L1 address rather than an accessor -- the write-side twin of the stage_band overload bypass #10 asks for. |
| kernels/rms_norm_writer.cpp:620,695 | a raw noc_async_write placing the root's own stat copy in front of the mcast_pipe helper's send | 'The root places its OWN copy first, then broadcasts in place (src == dst => EXCLUDE-source). Doing it this way makes the two host emitters behave identically: Mcast1D's per-row sender rect EXCLUDES the sender while Mcast2D's rect contains it.' D24 measures compute_scale 13575 -> 10932 ns on the root. | an include-self option on the mcast helper's sender |
| kernels/rms_norm_compute.cpp:1029, kernels/rms_norm_compute.cpp:1322,1331,1333,1338,1340 | explicit cb_wait_front / cb_pop_front outside every helper's WaitPolicy / PopPolicy | 'Explicit cb_pop_front calls on cb_input_tiles / cb_row_stat / cb_gamma_tiles / cb_scaler are the sanctioned pattern for operands whose lifetime spans more calls than any single PopPolicy can express (op_design.md section 6.1).' The widened wait at :1029 is D25's pipeline: 'cb_input_tiles' front is still block blk, so block blk+1 begins rows * X_HOLD_WT tiles further in.' | a PopPolicy that spans a caller-declared number of helper calls |

**1 of 17 bypass sites record no reason:** kernels/rms_norm_writer.cpp:314-324 (TILE output write).

## §10 Design record and contradictions

A numbered decision record exists — D1..D28 — but it is **not one document**: the planner's module docstring (D1-D15, D20-D28) PLUS kernel-site comments (D16-D19, D26). NOT one place -- see design_record[].source.
It is an emergent convention of this op, mandated nowhere.

| id | subject | what it decided | measurement cited | source |
|---|---|---|---|---|
| D1 | WT_CHUNK constrained to a DIVISOR of Wt | every width chunk is the same size; wt_chunk is still the coarsest admissible divisor, so the knob is not collapsed | n/a -- three mechanism constraints: tilize/untilize's compile-time block_width_tiles, reduce()'s BulkWaitBulkPop assert num_pages % cols == 0, and a multi-page cb_reserve_back must not straddle the ring | rms_norm_program_descriptor.py:78 |
| D2 | the STREAM chunk-size solve counts the ROW_MAJOR staging CBs at WT_CHUNK tiles, not at Wt | count what is actually allocated | — | rms_norm_program_descriptor.py:89 |
| D3 | accumulate_reduce_block now forwards reduce()'s ReduceFp32Mode / ReduceAlgorithm slots | RESOLVED by Refinement 1b; the op still passes ReduceFp32Mode::Fast | — | rms_norm_program_descriptor.py:91 |
| D4 | the regime predicate SEARCHES the depth knob instead of fixing it | walk CB_DEPTH_CANDIDATES coarsest-first, take RESIDENT at the first depth that fits; shipped (2,) is byte-identical to a fixed-depth predicate | depth (2,1) vs (2,): (1,1,32,4032) 38399 -> 46136 ns = 0.83x REGRESSION; three other shapes 0.99x-1.00x | rms_norm_program_descriptor.py:98 |
| D5 | per-CB data_format = the dtype of the tensor it carries | bfloat8_b needed no new format machinery; cb_row_stat stays fp32 in BOTH fp32_dest_acc_en modes; unpack_to_dest_mode left entirely at Default (no CB qualifies) | — | rms_norm_program_descriptor.py:129 |
| D6 | cb_row_stat is CB_ROW_STAT_DEPTH (=2) * BLOCK_ROWS pages | a CORRECTNESS requirement, not a perf depth: transform_in_place ROTATES the ring, so a PARTIAL final row-block's finalized tiles straddle the wrap at depth 1 | symptom without it: PCC 0.55-0.93, the 2nd..last row of each partial block garbage; invisible to Phase 0 because every Phase-0 golden cell had block_rows == 1 | rms_norm_program_descriptor.py:158 |
| D7 | ReduceAlgorithm::AccumulateViaAdd above REDUCE_ACC_VIA_ADD_MIN_WT | a crossover, not unconditional; BOTH datapaths stay live and covered | whole-op 1.00x-1.06x at four shapes; isolated reduce 0.67x at R=1 rising to 5.35x at R=32. The gap is the finding: rms_norm is dataflow-bound at these widths. | rms_norm_program_descriptor.py:183 |
| D8 | D7's crossover is measured against wt_per_core, not wt_chunk | gate on the core's WHOLE reduce dim, since the precision motive scales with the total | a HEIGHT-sharded (1,1,160,11008) squeezed WT_CHUNK to 2, dropped below the threshold and brought rms 0.127 back | rms_norm_program_descriptor.py:236 |
| D9 | Refinement 2's placement layer | the compute kernel's phase sequence, every helper call, the knob set and the L1 predicate are UNCHANGED by sharding; a sharded build differs only in who fills cb_input_tiles, which CB pass B reads the stat from, and whether the finalize is local or on a root | — | rms_norm_program_descriptor.py:607 |
| D10 | the ROW_MAJOR BAND scheme (_plan_band) | an RM width shard's page is a row SEGMENT, but the combine sums per-row PARTIALS elementwise and never needs a whole width tile; each core reduces the band it holds, staged from its own L1 in the tensor's GLOBAL TILE FRAME; PARTIAL_W is passed to the kernels as 0 and the zeroed staging ring REPLACES the reduce mask | (1,1,224,3072) WIDTH TILE 133224 vs RM BAND 136856 ns (+2.7%); BLOCK +21%; (1,1,256,512) +233% -- the last is a placement cost the caller chose (an RM granule cuts W=512 into 64 slices where TILE cuts 16), not the band | rms_norm_program_descriptor.py:257 |
| D11 | GRID_W turned from 1 to an AUTO policy (_auto_width_split) | a decode profile (Rt=1) runs an arbitrarily wide tensor through ONE core; splitting width is the only axis left. Two topologies: a gw x gh rectangle, or a PACKED single group over a bounding box with INACTIVE fillers. | group size sweep on (1,1,32,7168): 41779 / 13926 / 12876 / 14224 / 19338 ns at gw 1/8/16/32/56. Whole-op 1.00x-3.46x over 13 shapes. WIDTH_SPLIT_MIN_GAIN=4 keeps three grid-filling shapes byte-identical; at MIN_GAIN=2 (1024,1024) regressed 0.92x. | rms_norm_program_descriptor.py:313 |
| D12 | pass A's square folds width tiles into DEST (DestAccumulation::PerRow) below DEST_ACC_SQUARE_MAX_WT | a CEILING, not a floor -- the fold accumulates SERIALLY in a DEST word that is 16-bit at fp32_dest_acc_en=False. Gated on PARTIAL_W == 0. | (1,1,8192,1024) BLOCK 64c 102173 -> 98989 ns (1.03x); four WIDTH geometries 1.02x-1.03x | rms_norm_program_descriptor.py:401 |
| D13 | GATHER_FACES -- how many of a partial tile's four faces the gather ships | SCOPED by D27 to the block_rows == 1 branch, not retired | whole-tile ship at block_rows==1 regressed four WIDTH geometries 0.86x-0.98x, MONOTONE in group_size; and the gather was never byte-bound (halving bytes moved the 64-core BLOCK shard ~5%) | rms_norm_program_descriptor.py:427 |
| D14 | ROW_RESIDENT -- the op's THIRD compute regime (X_RESIDENT decoupled from num_w_chunks == 1) | hold ONE tile-row of x and the whole row of gamma, chunk only the DERIVED CBs; no second code path (the held CBs are indexed at a TileOffset). GATED ON THE DEPTH SACRIFICE, not on the regime. | (1,1,8192,5120) 753345 -> 468487 ns (1.61x); (1,1,8192,7168) 1043918 -> 655687 (1.59x). The gate's two halves: (1,1,32,7168) GRID_W=1 went 0.83x and is declined; ROW_MAJOR (1,1,32,4096) went 1.11x and is taken. | rms_norm_program_descriptor.py:484 |
| D15 | the finalize's rsqrt scoped to the tile faces pass B reads (RSQRT_COL_SCOPE) | superseded by D17, which extends the scope to the WHOLE finalize chain and deletes the selector | RC vs C: 1.03x-1.14x over five geometries. THIS is what the sharded geometries were spending time on -- the root runs one transform_in_place per tile-row per round with a group_size-wide fan-out of waiters behind it. | rms_norm_program_descriptor.py:570 |
| D16 | ROOT_FOLD_OUT -- the root fold's packer-L1-accumulation output spec | DELETED by D22 | D22 measured 2.18x AND better rel-RMS (2.42e-3 vs 3.38e-3), which REFUTES D16's recorded reasoning | kernels/rms_norm_compute.cpp:635 and kernels/rms_norm_compute.cpp:1108 |
| D17 | the whole finalize chain scoped to VectorMode::C in two DEST passes instead of three | the only path; the whole-tile spelling and its selector (compute CT arg 16) are gone | 600.7 -> 244.5 ns MATH-thread per finalize call, 2.04x on the stage; and an isolated bench poisoned columns 1..31 five orders of magnitude wrong and still passed the PCC gate | kernels/rms_norm_compute.cpp:149-201 |
| D18 | on the COMBINE path pass A's reduce packs its partial STRAIGHT into cb_sum_handoff | deletes the fp32 tile copy that used to move cb_row_stat -> cb_sum_handoff on EVERY core of EVERY group; legal because the combine path takes its slice in ONE chunk | the compute_partial_handoff zone is 'retired with the copy it measured' | kernels/rms_norm_compute.cpp:809 |
| D19 | a separate finalize chain that unpacked cb_row_stat and packed cb_stat_handoff | DELETED by D22's fused root chain | see D22 | kernels/rms_norm_compute.cpp:1111 |
| D20 | the reduce datapath's THIRD floor, on the reduce's actual PER-CALL width (x_squared_wt) | spelled as a narrow carve-out BELOW D7/D8's two floors, which both stay load-bearing | per-call width 1: ReduceTile 2.75x faster at BETTER rel-RMS. Whole pass A 11551 -> 6079 ns (1.90x), rel-RMS 0.00491 -> 0.00459. Getting the polarity backwards was measured twice: replacing MIN_CHUNK_WT regressed (1,1,160,11008) to 0.04774; vetoing on num_w_chunks>1 regressed (1,1,32,4064) RM to 0.06093. | rms_norm_program_descriptor.py:542 and rms_norm_program_descriptor.py:741-763 (REDUCE_ACC_VIA_ADD_MIN_CALL_WT) |
| D21 | pass B's DEST-LANE BLOCK SIZE (PASS_B_BLK, a divisor of WT_CHUNK capped by DEST_AUTO_LIMIT) plus its PerChunk pack lifecycle | adopted | 14050 -> 8860 ns (1.59x), BITWISE identical. The assigned fusion of pass B's two multiplies was a measured REGRESSION (0.84x) and is NOT here. | rms_norm_program_descriptor.py:547 |
| D22 | the FUSED ROOT CHAIN -- the group fold accumulates PAIRWISE IN DEST and the finalize runs in that same DEST window, one pack | replaces D16 and D19; every COMBINE-path use of cb_row_stat is deleted; needs GATHER_SLOTS (group_size rounded up to even) so the pairwise walk is universal | stage pair 5874 -> 2698 ns (2.18x) and MORE accurate (rel-RMS 2.42e-3 vs 3.38e-3). The pad-free alternative (seed DEST with a copy_tile) was built and LOST: 4442 -> 4610 ns (0.964x). | rms_norm_program_descriptor.py:551 |
| D23 | TILE-gamma read granularity (gamma_trim) | fetch only the face-rows pass B's BroadcastDim::Row consumer reads; bfloat8_b demotes to a half page (272-byte face is not 64-B aligned) -- a FORMAT fact, not a shape guard | interleaved prefill 1.10x-1.18x, BIT-EXACT across 7 geometries x 3 gamma dtypes; captures 83% of gamma's marginal read cost (the hard ceiling, deleting gamma outright, is 82785 ns vs 90338) | rms_norm_program_descriptor.py:558 and rms_norm_program_descriptor.py:1854-1859 (gamma_trim) |
| D24 | the root publishes its OWN stat copy BEFORE the broadcast | adopted; also makes Mcast1D's sender-excluding rect and Mcast2D's sender-including rect behave identically | -2643 ns on the root (compute_scale 13575 -> 10932) | rms_norm_program_descriptor.py:562 |
| D25 | the COMBINE PIPELINE -- block blk+1's pass A issued before block blk's combine, plus cb_sum_handoff at depth 2 | carved out to native_in: on a reader-fed input ring the pipeline is INCORRECT (a tile offset cannot cross a ring wrap) and, once made correct, still 0.894x. The gather ring stays at ONE round. | fills the root's ~3400 ns/round arrival idle; 1.161x with the depth-2 handoff vs 1.135x without. Deepening the gather ring was a regression twice over. | rms_norm_program_descriptor.py:564 |
| D26 | the gather's per-face boot zeroing, DELETED on the identity path | deleted; only the odd-group pad page is still zeroed | a bench seeded faces 1/3 with NINE catastrophic patterns (1e30, -1e30, NaN, +-Inf, fp32 subnormals, an Inf+(-Inf)=NaN mix, a stale-L1 lookalike) at group sizes 4/8/9/28/32 and got output BIT-IDENTICAL (torch.equal) to the zeroed run every time | kernels/rms_norm_writer.cpp:423-432 |
| D27 | THE COMPACT PARTIAL TRANSPOSE -- the combine's per-round unit becomes ONE tile whose columns 0..block_rows-1 are the block's stats | cb_partials_gathered goes GATHER_SLOTS*block_rows -> GATHER_SLOTS; cb_stat_handoff / cb_compact_handoff / cb_mcast_in become flat in block_rows; cb_bank is new; cb_row_stat is NOT ALLOCATED on the combine path. Carved out at block_rows == 1, where both permutes are the identity. | the flat ring was 1152 kB at block_rows 32 / group_size 8 (an L1 OOM). block_rows 8->20 (4 rounds -> 2) measured 25761 -> 24164 ns (1.066x); block_rows 32 (1 round) is L1 INFEASIBLE. The identity carve-out: leaving the permutes uniform regressed four WIDTH geometries 0.76-0.80x. | rms_norm_program_descriptor.py:442 |
| D28 | THE COMBINE'S SLOT TREE, at wide groups only | two levels, full stop (3 and 4 levels lost at 6 of 7 cells). One arrival semaphore PER LEVEL. Gated on the DERIVED quantity (deleted fold tiles), blind to shape/dtype/placement. | isolated 1.40x-1.45x; whole-op bracket ADJACENT: deleted 17 = 0.972x REGRESSION, deleted 18 = 1.028x WIN, deleted 20 = 1.074x. L1 goes DOWN (32 -> 14 fp32 pages at group_size 32). | rms_norm_program_descriptor.py:462 |

### Contradictions with planning artifacts — the code wins, always

| planning artifact's claim | artifact source | shipped reality | shipped source | recorded |
|---|---|---|---|---|
| op_design.md designs the op with 12 named circular buffers (cb_input_sticks, cb_input_tiles, cb_x_squared, cb_scaler, cb_row_stat, cb_gamma_sticks, cb_gamma_tiles, cb_normalized, cb_output_tiles, cb_output_sticks, cb_sum_handoff, cb_partials_gathered). | ttnn/ttnn/operations/rms_norm/op_design.md (whole file; grep for 'cb_') | 19 buffer indices are allocated across the case set (union over 25 descriptor builds = {0..18}). Seven have no mention in op_design.md at all: cb_stat_handoff (12), cb_row_final (13), cb_bank (14), cb_compact_handoff (15), cb_mcast_in (16), cb_gather_l1 (17), cb_node_out (18). | rms_norm_program_descriptor.py:978-1010 (the CB index table) and the descriptor walk in cases[] | shipped |
| op_design.md describes two compute regimes and never names ROW_RESIDENT, the BAND scheme, or the combine's slot tree (0 occurrences of each). | ttnn/ttnn/operations/rms_norm/op_design.md | Three compute regimes (RESIDENT / ROW_RESIDENT / STREAM, all three observed), a fourth placement scheme (SCHEME_SHARD_W+BAND, observed at c23), and a two-level slot-tree combine topology (observed at c21 and c23). | rms_norm_program_descriptor.py:484 (D14), rms_norm_program_descriptor.py:257 (D10), rms_norm_program_descriptor.py:462 (D28) | shipped |
| verification_report.md line 34: 'No mcast_pipe opportunity exists yet (no inter-core communication in Phase 0).' | ttnn/ttnn/operations/rms_norm/verification_report.md:34 (dated 2026-08-04, Phase 0) | Eight of 25 cases run a cross-core combine over ttnn.Mcast1D / ttnn.Mcast2D with 3 or 4 semaphores; the writer owns the whole gather -> root -> multicast handshake. | rms_norm_program_descriptor.py:2573-2579 (semaphores) and kernels/rms_norm_writer.cpp:333-341 | shipped |
| The compute kernel's own head comment: 'Every phase is a kernel_lib helper. The only raw LLK is inside the transform_in_place lambda' and ':201 this kernel's one sanctioned raw-LLK site'. | kernels/rms_norm_compute.cpp:41-45 and kernels/rms_norm_compute.cpp:200-201 | Three further raw-primitive sites exist, each with its own separately-recorded justification further down the same file: combine_fold (:404-437), member_pack (:987-999) and compute_recv_unpack (:1246-1263). The head comment also cites streaming_reduce_helpers.hpp:75-78 as the routing authority for the finalize, but that header is recorded as RETIRED upstream at :88-90 of the same file. | kernels/rms_norm_compute.cpp:404-437, kernels/rms_norm_compute.cpp:987-999, kernels/rms_norm_compute.cpp:1246-1263, kernels/rms_norm_compute.cpp:88-90 | shipped |
| The planner's Step-6 checklist mandates an l1_ledger.md beside op_design.md. | .claude/references/l1-footprint-discipline.md (The ledger section) | No l1_ledger.md exists for rms_norm, or for any op in the tree. | ttnn/ttnn/operations/rms_norm/ (directory listing) | shipped |
| rms_norm()'s own docstring: 'program_config: reserved; ignored when None.' | ttnn/ttnn/operations/rms_norm/rms_norm.py:276 | program_config is ignored ALWAYS, not only when None. rms_norm() forwards it to validate(), validate() accepts it in its signature and never references it, and it is not passed to create_program_descriptor(). A non-None program_config is silently accepted and has no effect -- it is neither honoured nor refused. This is the one hard-gate failure (G2b: a public parameter with an empty `enters_at`). | ttnn/ttnn/operations/rms_norm/rms_norm.py:194-202 (validate signature), :278-285 (the forward), :301-307 (the create_program_descriptor call, which omits it) | shipped |

### Claimed, not built  _(kept OUT of §3 — these are claims, not regimes)_

| claim | status | why | source |
|---|---|---|---|
| a hierarchical/two-stage GATHER beyond the two-level slot tree (3 and 4 levels) | rejected | measured at 7 cells and lost at 6 of them (group_size 32: 3 levels 3870-3991 ns vs 2 levels 3744; 4 levels 4461). 'A deeper tree buys another fold division but pays another hop, and the hop is the expensive half.' | rms_norm_program_descriptor.py:821-824 |
| CB_DEPTH_CANDIDATES = (2, 1) -- offering a shallower depth for the RESIDENT regime | deferred | measured a net loss today (0.83x at (1,1,32,4032)); recorded as a one-line change once Lamp L5 / L1 land, both of which have now landed, so this is re-testable | rms_norm_program_descriptor.py:663-668 |
| read-once-and-multicast for gamma (rather than every core reading the whole gamma row from DRAM) | not recorded as a regime anywhere | No deferral row exists for it. It is the largest above-minimum DRAM term in the SCHEME_ROWS traffic table -- (num_cores - 1) whole-gamma reads, 50 MB on (1,1,8192,7168) at 110 cores by D14's own arithmetic. D23 trimmed the per-tile bytes 83% but did not change the replication factor. | absent -- this row is an analysis finding, not a harvested claim |
| a ragged-tail width chunk (runtime wt_c) to fix the prime-Wt cliff | deferred | D1 forces WT_CHUNK \| Wt, so Wt = 127 collapses to one tile per chunk; L5 removes that shape's pass-B RE-READ but not its 127 one-tile compute phases. Observed at c24 (wt_chunk = 1, num_w_chunks = 127). | rms_norm_program_descriptor.py:533-536 |
| a compact partial handoff carrying a member's REDUCE_ROW partial as 32 floats instead of a 4096-byte tile | superseded | D27 built the compact transpose, which achieves the same round-trip reduction by a different mechanism (one tile per BLOCK, not per tile-row). D13's face-run gather survives at block_rows == 1. | rms_norm_program_descriptor.py:394-400 |

## §11 Unresolved

An empty `unresolved` on an op this complex would mean the analysis was shallow, not clean.

| item | why | what would resolve it |
|---|---|---|
| SCHEME_ROWS reached via the L1 bail-out (force_rows=True) -- a tiled shard whose per-core block does not fit even at block_rows == 1, re-planned on SCHEME_ROWS | no case in the set triggers it; every sharded geometry drawn from feature_spec fits | a HEIGHT or WIDTH shard whose per-core slice exceeds the L1 budget -- e.g. a very wide HEIGHT shard on few cores |
| SCHEME_ROWS reached via the ragged-width + partial-W bail-out (Wt % shard_w_t != 0 AND W % 32 != 0) | not exercised: the auto_shard_config geometries drawn from feature_spec do not produce that pair | a WIDTH/BLOCK shard on a shape with both a ragged tile tail and a non-tile-aligned W, e.g. (1,1,224,1000) WIDTH_SHARDED |
| SCHEME_SHARD_W+BAND with STAGE_ZERO == 1 | c23's bands fill their tile columns exactly (W=3072 over 96 cores = 32 elements each), so the stage_zero predicate is False. The band path with a sub-tile-column band -- the case D10's alignment argument is actually about -- is unobserved. | a ROW_MAJOR WIDTH shard whose granule is not a multiple of 32 elements, e.g. (1,1,256,512) ROW_MAJOR WIDTH_SHARDED (an 8-element bf16 granule) |
| COMBINE_TREE_MIN_DELETED_FOLD_TILES straddled at ADJACENT points (deleted 17 vs 18) | the straddle observed is deleted 17 (c22, group_size 28, flat) vs deleted 20 (c21, group_size 32, tree). group_size 30 (deleted exactly 18) needs a 30-core shard grid, which auto_shard_config did not produce for any shape in the set. | a WIDTH shard pinned to 30 cores, e.g. (1,1,32,4800) with shard [32,160] -- the exact cell D28's own bracket table uses |
| the ROW_RESIDENT_MIN_ROWS_PER_CORE gate DECLINING ROW_RESIDENT (max_rows < 2 with a depth sacrifice, falling through to STREAM) | the only observed depth-sacrificing ROW_RESIDENT build (c11) has max_rows 3 and is ACCEPTED. The declining branch is unobserved. | (1,1,32,7168) with GRID_W forced to 1 -- D14's own measured example (41779 -> 50598 ns, 0.83x, declined) |
| block_rows clamped by the TILE_DIM cap rather than by L1 | the coarsest observed combine block is c20's block_rows 20 out of max_rows 32; L1 bound it first. The kernels and host both assert block_rows <= 32, but no case reaches the clamp. | a combine geometry with > 32 tile-rows per core and enough L1 headroom -- D27 records that this was found by (1,1,3232,96) WIDTH-sharded solving to a 101-row block |
| cb_x_depth != cb_out_depth | every observed build has them equal -- _solve_blocking returns the same `depth` for both on every return path. They are two symbols with one value, and no case separates them. | nothing in the current code: the two are structurally tied. Recorded so a seeded planner does not treat them as independent knobs. |
| REDUCE_BULK == 0 (WaitAndPopPerTile) | the knob is hard-set to 1 and AccumulateViaAdd is gated on it, so the WaitAndPopPerTile reduce input policy is unreachable without editing the constant. It is a live code path in the helper but a dead one in this op. | setting REDUCE_BULK = 0 and re-running the sweep; note that this forces ReduceTile everywhere |
| per-core RUNTIME-arg variation is summarised, not enumerated | the dump records each kernel's rt-arg vector for every core but the artifact reports only the schema plus the per-case distinct-vector count. Enumerating 110 cores x 25 cases x 3 kernels would swamp the artifact. | the raw dump at scratchpad/rms_dump.json, if a consumer needs per-core values |
| the ablation pair holds gamma's absent cost at ZERO bytes on two cells, not across all seven regime families | c01/c02 (SCHEME_ROWS, RESIDENT) and c19/c25 (SCHEME_SHARD_W, flat, identity) are the two pairs built. The gate is satisfied (evidence: descriptor, 0 B), and the mechanism is a host-side inventory change that cannot be regime-dependent -- but STREAM and ROW_RESIDENT, where gamma is chunked and re-read, are not ablated. | adding a no-gamma twin of c11 (ROW_RESIDENT) and c12 (STREAM) |
| data-movement numbers are STRUCTURAL counts (crossings per tensor per unit), not measured bytes/ns | this skill does not measure performance; the counts are read off the reader/writer control flow and confirmed by the CT args, but no profile was taken | /perf-measure, or the D-record measurements already quoted in design_record[] |
| whether cb_row_final's 2x depth is load-bearing | it reuses CB_ROW_STAT_DEPTH with no reason of its own recorded; the audit flags it as over-capacity, but proving the 1x form correct needs a device run, not a read | halving it and running the golden suite plus the resilience group |
| band_write_back == accessor_sticks (a BAND input whose output is NOT the same shard spec, written stick-by-stick through the output TensorAccessor) | the only BAND case in the set (c23) allocates its output with the input's own memory_config, so _same_shard_spec holds and band_out_local is True (writer ct[14] OUT_SHARD_ROW_BYTES = 64). A ROW_MAJOR WIDTH/BLOCK-sharded input with an INTERLEAVED or HEIGHT_SHARDED output would reach the other branch; any other output geometry raises NotImplementedError rather than planning. | build one descriptor for a ROW_MAJOR WIDTH_SHARDED input with an explicit INTERLEAVED memory_config and read back writer ct[13]=1 / ct[14]=0. |
| PASS_B_BLK's value per case | it is derived inside the compute kernel from ckl::DEST_AUTO_LIMIT, a build-flag constant, so no host-side descriptor walk can observe it; the symbol row carries the bound and the formula but a null evaluation column. | a DPRINT of PASS_B_BLK from the compute kernel on one build per fp32_dest_acc_en value, or reading the compiled kernel's constant. |

## Appendix — the case set

Selection rule: the minimal set of cells covering every regime in every family, plus a straddling pair at every
numeric threshold, plus an ablation pair per optional input. Cells are drawn from
`.claude/eval/golden_tests/rms_norm/feature_spec.py`'s INPUTS and LOOSE_CASES; no shape here would be refused by `validate()`.

| id | shape | dtype | layout | memory_layout | fp32acc | gamma | cores (active) | scheme | residency | reduce | combine | transport | square | gamma read | out residency | band write-back | mcast | block_rows | wt_chunk | nwc | group | tree | arena B | why chosen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c01 | [1, 1, 64, 128] | BFLOAT16 | TILE | INTERLEAVED | True | gamma | 2 (2) | SCHEME_ROWS | RESIDENT | ReduceTile | none | n/a | dest_fold | face_rows | allocated_out | n/a (not the BAND scheme) | none | 1 | 4 | 1 | 1 | — | 61440 | baseline ROWS/RESIDENT; gamma-present half of ablation A1 |
| c02 | [1, 1, 64, 128] | BFLOAT16 | TILE | INTERLEAVED | True | no_gamma | 2 (2) | SCHEME_ROWS | RESIDENT | ReduceTile | none | n/a | dest_fold | whole_tile_or_n/a | allocated_out | n/a (not the BAND scheme) | none | 1 | 4 | 1 | 1 | — | 45056 | ablation A1 absent: same cell, gamma omitted |
| c03 | [1, 1, 64, 256] | BFLOAT16 | TILE | INTERLEAVED | True | gamma | 2 (2) | SCHEME_ROWS | RESIDENT | ReduceTile | none | n/a | dest_fold | face_rows | allocated_out | n/a (not the BAND scheme) | none | 1 | 8 | 1 | 1 | — | 110592 | Wt=8 == DEST_ACC_SQUARE_MAX_WT: fold ON (straddle below) |
| c04 | [1, 1, 64, 288] | BFLOAT16 | TILE | INTERLEAVED | True | gamma | 2 (2) | SCHEME_ROWS | RESIDENT | AccumulateViaAdd | none | n/a | packed | face_rows | allocated_out | n/a (not the BAND scheme) | none | 1 | 9 | 1 | 1 | — | 139264 | Wt=9 > DEST_ACC_SQUARE_MAX_WT: fold OFF (straddle above) |
| c05 | [1, 1, 32, 72] | BFLOAT16 | TILE | INTERLEAVED | True | gamma | 1 (1) | SCHEME_ROWS | RESIDENT | ReduceTile | none | n/a | packed | face_rows | allocated_out | n/a (not the BAND scheme) | none | 1 | 3 | 1 | 1 | — | 55296 | Wt=3 < REDUCE_ACC_VIA_ADD_MIN_WT: ReduceTile (straddle below); w_non_aligned |
| c06 | [1, 1, 32, 104] | BFLOAT16 | TILE | INTERLEAVED | True | gamma | 1 (1) | SCHEME_ROWS | RESIDENT | AccumulateViaAdd | none | n/a | packed | face_rows | allocated_out | n/a (not the BAND scheme) | none | 1 | 4 | 1 | 1 | — | 69632 | Wt=4 == REDUCE_ACC_VIA_ADD_MIN_WT, partial_w keeps fold off: AccViaAdd (straddle above) |
| c07 | [1, 1, 17, 64] | BFLOAT16 | TILE | INTERLEAVED | True | gamma | 1 (1) | SCHEME_ROWS | RESIDENT | ReduceTile | none | n/a | dest_fold | face_rows | allocated_out | n/a (not the BAND scheme) | none | 1 | 2 | 1 | 1 | — | 36864 | h_non_aligned, rank 4 |
| c08 | [1024, 1024] | BFLOAT16 | TILE | INTERLEAVED | True | gamma | 32 (32) | SCHEME_ROWS | RESIDENT | AccumulateViaAdd | none | n/a | packed | face_rows | allocated_out | n/a (not the BAND scheme) | none | 1 | 32 | 1 | 1 | — | 468992 | rank 2; WIDTH_SPLIT_MIN_GAIN declines the split |
| c09 | [1, 32, 4096] | BFLOAT16 | TILE | INTERLEAVED | True | gamma | 22 (16) | SCHEME_SHARD_W | RESIDENT | ReduceTile | flat_root | identity | dest_fold | face_rows | allocated_out | n/a (not the BAND scheme) | Mcast2D (bounding box) | 1 | 8 | 1 | 16 | — | 192512 | rank 3 |
| c10 | [1, 1, 8192, 1024] | BFLOAT16 | TILE | INTERLEAVED | False | gamma | 110 (110) | SCHEME_ROWS | RESIDENT | AccumulateViaAdd | none | n/a | packed | face_rows | allocated_out | n/a (not the BAND scheme) | none | 3 | 32 | 1 | 1 | — | 1271808 | prefill RESIDENT interleaved; fp32_dest_acc_en=False |
| c11 | [1, 1, 8192, 7168] | BFLOAT16 | TILE | INTERLEAVED | False | gamma | 110 (110) | SCHEME_ROWS | ROW_RESIDENT | AccumulateViaAdd | none | n/a | packed | face_rows | allocated_out | n/a (not the BAND scheme) | none | 1 | 56 | 4 | 1 | — | 1271808 | prefill wide: expect ROW_RESIDENT (D14) |
| c12 | [1, 1, 4096, 11008] | BFLOAT16 | TILE | INTERLEAVED | False | gamma | 110 (110) | SCHEME_ROWS | STREAM | AccumulateViaAdd | none | n/a | packed | face_rows | allocated_out | n/a (not the BAND scheme) | none | 1 | 86 | 4 | 1 | — | 1243136 | one tile-row of x+gamma over budget: expect STREAM |
| c13 | [1, 1, 32, 7168] | BFLOAT16 | TILE | INTERLEAVED | False | gamma | 22 (16) | SCHEME_SHARD_W | RESIDENT | AccumulateViaAdd | flat_root | identity | packed | face_rows | allocated_out | n/a (not the BAND scheme) | Mcast2D (bounding box) | 1 | 14 | 1 | 16 | — | 292864 | decode wide: interleaved cross-core width split (D11 AUTO) |
| c14 | [1, 1, 64, 128] | FLOAT32 | TILE | INTERLEAVED | True | gamma | 2 (2) | SCHEME_ROWS | RESIDENT | ReduceTile | none | n/a | dest_fold | face_rows | allocated_out | n/a (not the BAND scheme) | none | 1 | 4 | 1 | 1 | — | 112640 | dtype float32 (+fp32_dest_acc_en=True, the only legal cell) |
| c15 | [1, 1, 64, 128] | BFLOAT8_B | TILE | INTERLEAVED | True | gamma | 2 (2) | SCHEME_ROWS | RESIDENT | ReduceTile | none | n/a | dest_fold | half_page | allocated_out | n/a (not the BAND scheme) | none | 1 | 4 | 1 | 1 | — | 37440 | dtype bfloat8_b; gamma_trim demotes to half page (D23) |
| c16 | [1, 1, 64, 128] | BFLOAT16 | ROW_MAJOR | INTERLEAVED | True | gamma | 2 (2) | SCHEME_ROWS | RESIDENT | ReduceTile | none | n/a | dest_fold | whole_tile_or_n/a | allocated_out | n/a (not the BAND scheme) | none | 1 | 4 | 1 | 1 | — | 86016 | ROW_MAJOR x + ROW_MAJOR gamma: stick staging, depth forced to 1 |
| c17 | [1, 1, 32, 50] | BFLOAT16 | ROW_MAJOR | INTERLEAVED | True | gamma | 1 (1) | SCHEME_ROWS | RESIDENT | ReduceTile | none | n/a | packed | whole_tile_or_n/a | allocated_out | n/a (not the BAND scheme) | none | 1 | 2 | 1 | 1 | — | 53248 | ROW_MAJOR + w_non_aligned: stage_zero |
| c18 | [1, 1, 256, 512] | BFLOAT16 | TILE | HEIGHT_SHARDED | True | gamma | 8 (8) | SCHEME_SHARD_H | RESIDENT | AccumulateViaAdd | none | n/a | packed | face_rows | native_out (aliased) | n/a (not the BAND scheme) | none | 1 | 16 | 1 | 1 | — | 108544 | SCHEME_SHARD_H: zero-copy CBs, local reduce |
| c19 | [1, 1, 32, 1024] | BFLOAT16 | TILE | WIDTH_SHARDED | False | gamma | 8 (8) | SCHEME_SHARD_W | RESIDENT | ReduceTile | flat_root | identity | dest_fold | face_rows | native_out (aliased) | n/a (not the BAND scheme) | Mcast2D (bounding box) | 1 | 4 | 1 | 8 | — | 77824 | combine, group 8, block_rows==1 identity path; gamma-present half of ablation A2 |
| c20 | [1, 1, 8192, 1024] | BFLOAT16 | TILE | BLOCK_SHARDED | False | gamma | 64 (64) | SCHEME_SHARD_W | RESIDENT | ReduceTile | flat_root | compact | dest_fold | face_rows | native_out (aliased) | n/a (not the BAND scheme) | Mcast1D (PerRow) | 20 | 4 | 1 | 8 | — | 641024 | combine, group 8, block_rows>1 COMPACT path (D27) |
| c21 | [1, 1, 32, 5120] | BFLOAT16 | TILE | WIDTH_SHARDED | False | gamma | 32 (32) | SCHEME_SHARD_W | RESIDENT | ReduceTile | slot_tree | identity | dest_fold | face_rows | native_out (aliased) | n/a (not the BAND scheme) | Mcast2D (bounding box) | 1 | 5 | 1 | 32 | 4x8 | 106496 | group 32 -> deleted 20 >= 18: SLOT TREE (straddle above) |
| c22 | [1, 1, 32, 7168] | BFLOAT16 | TILE | WIDTH_SHARDED | False | gamma | 28 (28) | SCHEME_SHARD_W | RESIDENT | ReduceTile | flat_root | identity | dest_fold | face_rows | native_out (aliased) | n/a (not the BAND scheme) | Mcast2D (bounding box) | 1 | 8 | 1 | 28 | — | 176128 | group 28 -> deleted 17 < 18: FLAT root (straddle below) |
| c23 | [1, 1, 224, 3072] | BFLOAT16 | ROW_MAJOR | WIDTH_SHARDED | False | gamma | 99 (96) | SCHEME_SHARD_W+BAND | RESIDENT | ReduceTile | slot_tree | compact | packed | whole_tile_or_n/a | allocated_out | local_l1 | Mcast2D (bounding box) | 7 | 1 | 1 | 96 | 4x24 | 348160 | ROW_MAJOR width shard: the BAND scheme (D10) |
| c24 | [1, 1, 32, 4064] | BFLOAT16 | ROW_MAJOR | INTERLEAVED | False | gamma | 1 (1) | SCHEME_ROWS | ROW_RESIDENT | ReduceTile | none | n/a | packed | whole_tile_or_n/a | allocated_out | n/a (not the BAND scheme) | none | 1 | 1 | 127 | 1 | — | 546816 | Wt=127 prime -> wt_chunk==1 < REDUCE_ACC_VIA_ADD_MIN_CHUNK_WT: ReduceTile |
| c25 | [1, 1, 32, 1024] | BFLOAT16 | TILE | WIDTH_SHARDED | False | no_gamma | 8 (8) | SCHEME_SHARD_W | RESIDENT | ReduceTile | flat_root | identity | dest_fold | whole_tile_or_n/a | native_out (aliased) | n/a (not the BAND scheme) | Mcast2D (bounding box) | 1 | 4 | 1 | 8 | — | 61440 | ablation A2 absent: combine path without gamma |

