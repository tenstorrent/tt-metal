# Port Report — nlp_concat_heads_decode

## Outcome

*(pending verification — construction complete, both factories + both kernels converted; awaiting post-port test run)*

## Provenance

- **Recipe docs (this port):** *not pinnable* — `git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/` prints nothing; the `metal_2.0/` docs tree is untracked in the invoking checkout (read from the main checkout by absolute path; it does not exist in this worktree at all).
- **Audit docs (inherited):** *not pinnable* — same condition, recorded verbatim from `METAL2_PORT_BRIEF.md`.
- Port baseline commit: `4a5bfad59c6` (worktree branched at origin/main).
- Baseline test run: `tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_concat_heads_decode.py` — **15 passed** (coordinator-run, main checkout byte-identical to `4a5bfad59c6`; both factories covered: all `test_concat_head` and all `test_concat_head_subcoregrids` variants).

## TTNN ProgramFactory

### Concept realized
`ProgramSpecFactoryConcept`, both factories (`NLPConcatHeadsDecodeProgramFactory`,
`NLPConcatHeadsDecodeSubcoregridsProgramFactory`) — as the audit chose. `create_descriptor` →
`create_program_artifacts` inside the existing `program_factory_t` variant; the device-operation class is untouched.

### Device-op-class edits
- Pybind entry points removed: **none** (nothing bound `create_descriptor`).
- Custom `compute_program_hash`: **none** (default reflection hash; nothing to preserve).

### Open items
- Relaxation candidates: none observed — the kernels bake batch/head geometry into CTAs, so strict `TensorSpec`
  matching is the right cache-equivalence class.

## Handoff points

none — no capitulations, no boundary-rule violations, no kernel-lib gaps, no pybind surface removed. The
environment-limited test `tests/ttnn/distributed/test_multidevice_TG.py::test_galaxy_nlp_concat_heads_decode`
(TG-galaxy-only) could not be run on this box (single 8x8 wormhole_b0); it exercises the default factory through
the same public op. Flagging for whoever next has TG time, alongside the model_traced sweep
(`tests/sweep_framework/sweeps/model_traced/nlp_concat_heads_decode_model_traced.py`), also not run here.

## Successes

- **Two-toucher DFB → 1P+1C (patterns catalog)** fired exactly as written: the brief and the re-derived census
  agree (two role-free raw writers of the borrowed output CB per node — the dual-instance work-split shape). The
  catalog's warning against reaching for `allow_instance_multi_binding` was on point; no flag was set
  (`device/nlp_concat_heads_decode_program_factory.cpp`, READER=PRODUCER / WRITER=CONSUMER bindings).
- **Case 2 raw-pointer bridge (whitelist rule 5)**: `TensorAccessor(tensor::input).get_bank_base_address()` slotted
  in for the legacy `q_start_addr` RTA with the remote-NoC walk byte-identical
  (`device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_decode.cpp:34-35`). The brief's pre-classification
  meant zero mid-port judgment calls.
- **Varargs caution (patterns catalog)**: the two leading scalars were named/bound rather than riding the vararg
  block (trap 1 avoided); the NoC coordinate tables — CTA-driven counts, data-driven cursor (`get_vararg(qkv_x)`) —
  are genuine indexed collections and stayed varargs. The `AddRuntimeArgsForNode` helper kept the legacy node-first
  loop intact.
- **hw_config discipline (recipe §Hardware configuration)**: verified the legacy `ReaderConfigDescriptor{}` /
  `WriterConfigDescriptor{}` resolution in `tt_metal/impl/kernels/kernel_types.cpp:13-43` (reader RISCV_1/NOC_0,
  writer RISCV_0/NOC_1, both DM_DEDICATED_NOC on all Gen1 arches) before adopting the TTNN
  `create_reader/writer_datamovement_config(arch)` helpers — the values match exactly, so the helpers are faithful.

## Friction

- **Worktree bootstrap:** a fresh `git worktree` has no submodules checked out; the first `./build_metal.sh` failed
  at cmake configure ("Missing submodules"). `git submodule update --init --recursive` inside the worktree fixed it.
  Worth a line in `workspace_setup.md` for worktree-based porting sessions.
- **Varargs are per-node duplicated:** the NoC coordinate tables are identical on every node (the host builds them
  once), but the faithful mirror of the legacy per-core RTA lists is `runtime_varargs[core] = noc_coords` for every
  core — a `Table<NodeCoord, vector>` holding N copies of the same vector. `num_common_runtime_varargs` is the
  natural fit but is an RTA→CRTA dispatch-semantics change the recipe defers; see Open items.
- *(minor)* The recipe's self-audit grep commands assume unrestricted `git grep "$BASE"` invocations; a
  worktree-sandboxed agent has to decompose them into per-file `git show | grep -c` equivalents. Same result, more
  steps.

## Open items for downstream

- **Shared kernel touches:** none — both kernel sources are op-owned, single-consumer, converted in place with
  their factories (no `_metal2` fork created anywhere).
- **CRTA candidates (later cleanup pass, not port work):** both kernels' vararg NoC-coordinate blocks are
  node-invariant → `num_common_runtime_varargs` would shrink dispatch traffic; `in_tile_offset_by_head` is
  genuinely per-node and stays an RTA. (Legacy dispatched everything per-core; converting changes dispatch
  semantics, so it was deliberately not done in the port.)
- **Pre-existing observations carried forward from the audit's misc anomalies (not fixed, per scope discipline):**
  dead shadowed `q_write_addr` local in both kernels (outer declaration is dead; preserved verbatim);
  `memory_config` op argument accepted and ignored end-to-end; default factory hardcodes 32x32 tile/face geometry
  (16 / `512 * element_size` / `256 * ELEMENT_SIZE`) while the subcoregrids pair derives it from the tensor's
  tile spec; default factory assumes a dense row-major input grid (that is what the subcoregrids factory exists
  to lift).
- **Test coverage note:** the op's only exercised coverage is
  `tests/tt_eager/python_api_testing/unit_testing/misc/test_nlp_concat_heads_decode.py`; the TG-galaxy test and the
  model_traced sweep are environment-gated (see Handoff points note).
