# Coding Guideline Conformance Audit: `ttnn/` and `tests/ttnn/`

Audit of the tt-metal `ttnn/` and `tests/ttnn/` trees against **Section 2 (Coding Guidelines)** of
*Coding and Architecture Guidelines, Tenstorrent AI-IP Software Stack*, Draft v0.1, 18-Aug-2026
([source document](https://docs.google.com/document/d/1OSo2KuUgN90GvLxZEDO2lzytMXn2VDmPSBtd-sA3lis/edit?tab=t.0)).

| | |
|---|---|
| Repository | `/localdev/fplavec/git_2026_08_19_quasar_topk/tt-metal` |
| Branch | `fplavec/quasar_topk` at `e8424280d58` |
| Audit date | 2026-09-02 |
| Rules assessed | 20 (CG-001 to CG-071, Sections 2.1 to 2.8) |
| Result | **18 violated, 2 clean** |

Section 2.9 defines a log structure rather than a rule, so it is not assessed.

---

## 1. Scope decisions

Every MUST rule in Section 2 is conditioned on code being "safety-relevant". tt-metal has no declared
safety baseline, no ASIL assignment, and no designated safety-relevant path today, so the audit needed
an agreed reading before it could produce anything useful. Four decisions were taken:

**Safety-relevant path: narrow.** Section 2.1 rules (CG-001 to CG-006) are judged only against the
on-target execution path, because 2.1 states it covers "on-target and firmware" code. Inside `ttnn/`
that means the device kernels: 1,385 files containing `void kernel_main()`, or 1,496 files including
kernel headers. The dispatch firmware itself lives in `tt_metal/` and is outside the requested scope.
The host operator library is judged only against the rules outside 2.1.

**Process rules: judged from repo evidence.** CG-041, CG-050, CG-051, CG-070, and CG-071 target program
artifacts (a release manifest, a static-analysis gate, a deviation log, a third-party register) that do
not live in source code. They are answered from what the repository does show, and tagged
*program-level* in the table. The guideline itself names `.clang-tidy` and `.codechecker` directly,
so these rules are not purely external.

**Evidence: inspection only.** CG-005 and CG-010 would normally need a tool run (a complexity scanner,
`mypy --strict`). They are answered here from targeted inspection plus scripted counting, with no
`clang-tidy` or `mypy` execution. Where a count is a lower bound, the table says so.

**Subtrees: all of `ttnn/`.** Includes `api`, `cpp`, `core`, the `ttnn` Python package, and also
`tutorials`, `examples`, and the legacy `tt_lib`. Rows whose only evidence sits in demo or legacy code
would be flagged; none of the findings below fall into that category.

### Trees audited

| Tree | Content |
|---|---|
| `ttnn/` | ~3,500 host C++/hpp files, 1,283 device-kernel `.cpp` across 253 `kernels/` directories, 93 Python |
| `tests/ttnn/` | 858 Python, 147 C++/hpp |

---

## 2. Conformance table

| Rule | Violations (Y/N) | Representative violation |
|---|:---:|---|
| **CG-001** [MUST] No unbounded recursion on-target; depth statically bounded and verified by static analysis | **N** (code), Y (verification clause) | Zero runtime self-recursion in 1,496 kernel files. Two scan candidates were overload forwarding, not recursion. The verification half is unmet: `misc-no-recursion` is off at [.clang-tidy:87](.clang-tidy#L87) and kernels are not in the compile database |
| **CG-002** [MUST] No dynamic allocation on-target after init | **N** | Genuinely clean. Zero `new`/`delete`/`malloc`/`alloca`/STL containers across all 1,496 kernel files. The 69 `allocate_header` hits are fixed-size fabric header slots, not heap |
| **CG-003** [MUST] No goto/setjmp/longjmp; no multiple return points from acquisition blocks; RAII for all resource lifetimes | **Y** | [conv_reader_common.hpp:215-218](ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/conv_reader_common.hpp#L215-L218): `reserve_back(act_cb_tiles)` then a bare `return;` with no `push_back`. Goto clause is clean (zero in ttnn), but 1,241 kernel files use manual reserve/push pairs and only one op ([integral_image common.hpp:77-91](ttnn/cpp/ttnn/operations/experimental/reduction/integral_image/device/kernels/common.hpp#L77-L91)) uses RAII guards |
| **CG-004** [MUST] No raw pointer arithmetic outside a small reviewed DMA helper set; use bounds-checked span/array | **Y** | [writer_deepseek_grouped_gate.cpp:332-360](ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate/device/kernels/dataflow/writer_deepseek_grouped_gate.cpp#L332-L360): four raw `tt_l1_ptr` pointers built by `reinterpret_cast` from runtime addresses, then indexed. 318 kernel files reinterpret_cast to raw device pointers, 162 do arithmetic on them, **zero** use `std::span` or `Span` |
| **CG-005** [SHOULD] Cyclomatic complexity =< 15 | **Y** | [dit_rmsnorm_fused_compute.cpp:45-1236](ttnn/cpp/ttnn/operations/experimental/ccl/dit_fused_distributed_rmsnorm/device/kernels/compute/dit_rmsnorm_fused_compute.cpp#L45-L1236): one 1,192-line `kernel_main` at **CC 165** (48 `if constexpr`, 56 `for`, 26 ternary, 3 `if`), eleven times the limit. **256 of 1,385** kernel entry points exceed 15; 33 exceed 50; 3 exceed 100. Highest runtime branching is [ring_joint_reader.cpp:229-1159](ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_reader.cpp#L229-L1159) at CC 144 (25 `if`, 32 `&&`, 9 `\|\|`). Helper functions are not counted, so all figures are floors |
| **CG-006** [MUST] Single central target macro/enum checked by static_assert; no ad hoc per-file `#ifdef` | **Y** | [conv_bmm_tilize_metal2.cpp:48-455](ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/conv_bmm_tilize_metal2.cpp#L48-L455): 24 separate `#ifdef`/`#ifndef ARCH_QUASAR` blocks in one file. 169 such sites across ttnn (117 QUASAR, 49 BLACKHOLE, 4 WORMHOLE, 3 GRAYSKULL), zero central static_assert |
| **CG-010** [MUST] Python fully type-hinted, mypy --strict as build gate | **Y** | No mypy configuration or gate covers ttnn at all. The only mypy in the repo is [tools/triage/mypy.ini](tools/triage/mypy.ini#L1-L3), a separate tool. Return-annotation coverage: `ttnn/` 232 of 1,062 defs (21.8%), `tests/ttnn/` 270 of 8,558 (3.2%) |
| **CG-011** [MUST] No eval/exec/dynamic import from non-static paths | **Y** | Most egregious: [perf_csv.py:25](tests/ttnn/unit_tests/operations/ccl/perf/perf_csv.py#L25), `eval()` on a string read out of a profiler CSV. In `ttnn/` proper: [__init__.py:483-499](ttnn/ttnn/__init__.py#L483-L499) walks `__path__` with `pkgutil` and imports whatever it finds |
| **CG-012** [SHOULD] Bounded recursion in build-time Python where depth follows graph size | **Y** | [tracer.py:100-110](ttnn/ttnn/tracer.py#L100-L110): `preprocess_arg` recurses through nested containers with no depth limit, in the trace-and-codegen pass. [unsafe_allocation_tracker.py:168-178](ttnn/ttnn/unsafe_allocation_tracker.py#L168-L178) shows the same pattern done right, with an explicit `depth` budget |
| **CG-020** [MUST] `_sm` suffix on safety mechanisms, `_diag` on diagnostic-only tooling | **Y** | Zero identifiers in either tree use either suffix for its intended meaning (the four `_sm` hits mean "sharded", the four `_diag` hits mean matrix diagonal). Sharpest case: [reader_bmm_tile_layout_in0_sender_padding.cpp:78-90](ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp#L78-L90) documents an `ASSERT` as the detection for a contract violation that otherwise deadlocks the device, and that `ASSERT` compiles to nothing without watcher ([assert.h:37-40](tt_metal/hw/inc/api/debug/assert.h#L37-L40)). That is FM-012 exactly: a diagnostic tool serving as the safety mechanism, named as neither |
| **CG-021** [SHOULD] Names match SW Architecture Description element names | **Y** (partial) | No architecture description exists in the repo, so the reference list is missing. The checkable symptom is present: the producer/consumer synchronization buffer has two live names in the same tree, `cb_reserve_back`/`cb_wait_front` (51 files) and `DataflowBuffer` (841 files), with zero files using both. No tool can map one element to one symbol |
| **CG-030** [MUST] Sequence or generation counter checked by consumer on every handoff | **Y** | 1,241 kernel files perform reserve/push/wait/pop handoff and **zero** carry a data-currency counter. Neither API has a field for one. Example: [reader_binary_interleaved_start_id_metal2.cpp:36-70](ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id_metal2.cpp#L36-L70), reserve/read/push with nothing the consumer can check |
| **CG-031** [MUST] Concurrency model documented inline at resource acquisition | **Y** | 111 kernel files acquire NoC semaphores and 134 use cross-core NoC addresses. Zero mention freedom from interference, ISO 26262-11, or any of the four interference categories. Six name the sharing cores in prose only |
| **CG-040** [MUST] Production build statically asserts the debug host-synchronous dispatch flag is unset, failing the build | **Y** | [strided_all_gather_async_op.cpp:74-76](ttnn/cpp/ttnn/operations/experimental/ccl/strided_all_gather_async/device/strided_all_gather_async_op.cpp#L74-L76): the mode is detected at run time with `std::getenv("TT_METAL_SLOW_DISPATCH_MODE")` inside a `TT_FATAL`. Zero `static_assert` or `#error` on that flag anywhere in either tree |
| **CG-041** [MUST] Release records exact compiler, flags, target configuration | **Y** (program-level) | No release manifest exists. The nearest thing is the kernel cache path `<git_hash>/<build_id>` at [kernel_cache.hpp:17](tt_metal/api/tt-metalium/experimental/kernel_cache.hpp#L17), which keys on source revision, not compiler or flags. `compile_commands.json` and `CMakeCache.txt` exist only inside a local build tree |
| **CG-050** [MUST] Named blocking static-analysis gate against this guideline, not the general `.clang-tidy`/`.codechecker` | **Y** (program-level) | The repo has precisely the configuration the rule names as insufficient, and it is tuned away from these rules: [.clang-tidy:60-87](.clang-tidy#L60-L87) disables `cppcoreguidelines-owning-memory` (CG-002), `-pro-bounds-pointer-arithmetic` and `-pro-type-reinterpret-cast` (CG-004), and `misc-no-recursion` (CG-001). Device kernels are not in the compile database, so 2.1 is unchecked by any tool |
| **CG-051** [MUST] Deviation log for MUST-rule deviations | **Y** (program-level) | No deviation log exists in the repo, and no CG-xxx rule ID appears anywhere. Every finding above is an undocumented deviation by definition |
| **CG-060** [SHOULD] End-to-end integrity check on NoC and Ethernet transfers | **Y** | 897 kernel files issue NoC async reads/writes and 72 use the ethernet/fabric path. Zero implement a checksum, CRC, or sequence validation. The six "parity" matches are DEST bank parity and chunk-half indexing, not integrity |
| **CG-061** [MUST] No silently swallowed errors; handle to a defined state or propagate to fault reaction | **Y** | On-target: [assert.h:37-40](tt_metal/hw/inc/api/debug/assert.h#L37-L40) makes `ASSERT` a no-op in a production build, and 58 ttnn kernel files use `ASSERT` as their only contract check, so the failure becomes a silent hang. Host Python: [distributed.py:117-118](ttnn/ttnn/distributed/distributed.py#L117-L118) is a bare `except: continue` that drops a device from the mesh mapping with no signal |
| **CG-070** [MUST] Third-party register entry with a stated qualification path before use | **Y** (program-level) | No register and no qualification-path metadata of any kind. ttnn directly includes six third-party libraries: fmt (49 files), xtensor (23), tracy (13), nlohmann (11), reflect (5), boost (1). The build pulls 89 CPM packages |
| **CG-071** [MUST] OSS with no accountable supplier is an unqualified black box needing explicit evaluation | **Y** (program-level) | Same evidence as CG-070. No evaluation record exists for any dependency, and [.codechecker.skiplist:5-8](.codechecker.skiplist#L5-L8) excludes third-party code from even the general-purpose scan, so those components are unexamined by both routes |

---

## 3. Observations

**The two clean rows are clean for a structural reason, not by choice.** Tensix kernels have no heap and
no stack budget for recursion, so CG-001 and CG-002 hold as a side effect of the hardware model. Neither
is enforced by any tool, so neither would survive a change of habit.

**CG-050 is the multiplier.** The four clang-tidy checks that map to CG-001 through CG-004 are switched
off, and device kernels never enter the compile database. The entire on-target path in this table is
therefore unchecked by any tool today. Enabling those checks for the kernel tree would convert four rows
from prose into a CI signal.

**CG-005 is structural, not a few outliers.** 18% of all ttnn kernel entry points exceed the complexity
limit, and three sit above 100. This is a property of how these kernels are written rather than a short
list of functions to refactor.

**One op concentrates five violations.**
[`ttnn/cpp/ttnn/operations/experimental/quasar/matmul/`](ttnn/cpp/ttnn/operations/experimental/quasar/matmul/)
violates CG-004, CG-006, CG-020, CG-030, and CG-061 in the same operator. Its own comments at
[sparse_matmul_device_operation.cpp:162-169](ttnn/cpp/ttnn/operations/experimental/quasar/matmul/device/sparse/sparse_matmul_device_operation.cpp#L162-L169)
describe the resulting failure mode as a data-dependent device deadlock detectable only under watcher.
If the SW-FMEA needs one worked example, that operator is it.

**Compile-time branching couples CG-005 and CG-006.**
[`conv_bmm_tilize_metal2.cpp`](ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/conv_bmm_tilize_metal2.cpp)
is both the CG-006 worst case (24 ad hoc architecture `#ifdef` blocks) and a CG-005 top-ten entry. The
same per-architecture conditionals drive both numbers.

---

## Appendix A: Cyclomatic complexity

CG-005 sets a limit on cyclomatic complexity, so a short definition is useful when reading that row.

Cyclomatic complexity counts how many independent paths run through a piece of code. Thomas McCabe
defined it in 1976, and it has been the standard complexity measure in safety standards ever since.

Start at 1, then add 1 for every decision point:

| Construct | Adds |
|---|---|
| `if`, `else if` | 1 each |
| `for`, `while`, `do` | 1 each |
| `case` in a switch | 1 each |
| `&&`, `\|\|` | 1 each (each is a hidden branch, because the right side may not run) |
| `? :` | 1 |
| `catch` | 1 |

A function with no branches scores 1: one path in, one path out.

The number matters for two reasons. It is a lower bound on how many test cases are needed to cover every
branch, so a function scoring 73 needs at least 73 tests before every path has been exercised once. It
also tracks how much of a function a reader must hold in mind at once. ISO 26262-6 asks for restricted
size and complexity of software units, and 15 is a common industry threshold, roughly "a function a
reviewer can still reason about completely".

### Compile-time versus runtime paths

Device kernels use `if constexpr` heavily, which affects how the number should be read. Two measurements
apply, and both exceed the limit:

- **As written.** Every branch counts, including `if constexpr`. This is what a reviewer faces reading
  the source and what a static-analysis tool reports by default. The table uses this measure.
- **As compiled.** `if constexpr` selects one side at build time, so any single binary contains fewer
  live paths. For `ring_joint_reader.cpp` the runtime floor is 32 (25 `if` plus 6 `for` plus 1), still
  twice the limit before counting any `&&`, `||`, or ternary that survives into runtime code.

Compile-time branches are not free even though they vanish from the binary. They set how many distinct
kernel variants a build can produce, and each variant is separate work to verify. That is CG-005 and
CG-006 pulling in the same direction on one function.

---

## Appendix B: Method and limitations

Findings come from scripted searches over the two trees plus targeted reading of each cited site. No
build, `clang-tidy` run, or `mypy` run was performed, per the agreed scope.

**Counts are lower bounds.** Complexity figures cover `kernel_main` bodies only and exclude helper
functions in the same files. Rule-specific searches match known API and construct names, so a violation
expressed through an unusual spelling would be missed.

**CG-021 is only partly assessable.** The rule requires names to match a SW Architecture Description
that does not exist for these trees. The reported symptom (two coexisting names for one architectural
element) is a proxy for the rule's stated intent, not a direct measurement against it.

**Program-level rows describe the repository, not the code.** CG-041, CG-050, CG-051, CG-070, and CG-071
report the absence of program artifacts. A record kept outside this repository would change those rows
and would not be visible from here.

**Self-recursion detection was manual at the last step.** The automated scan produced false positives
(overload forwarding, method calls on other objects). Each candidate was read individually; the CG-001
result rests on that reading rather than on the scan output.
