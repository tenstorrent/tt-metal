# TTTv2 2D Modules and WH Galaxy Model Reconstruction Plan

## Status

Planning only. Do not begin Milestone B until Milestone A passes its exit gate, and do not begin
Milestone C until Milestone B passes its exit gate.

## Goal

Build the reusable TTTv2 2D module set for the canonical Wormhole Galaxy mesh `(8, 4)`, prove the
modules in clean TTTv2 reconstructions of:

- `meta-llama/Llama-3.3-70B-Instruct`; and
- `Qwen/Qwen3-32B`,

then integrate both reconstructions with `models/common/llm_runtime` and the TT vLLM plugin. The
finished TTTv2 path must match or improve the existing models' accuracy and remain within 3% of
their performance in paired same-host measurements while also meeting the active absolute product
targets.

This work is also an extension test of TTTv2's modular architecture. A successful design adds a new
mesh topology and two product models primarily through new 2D modules, model packages, and immutable
configuration. It must not require edits to 1D module implementation code, and it should require no
behavioral `llm_runtime` changes beyond narrowly justified generic policy plumbing.

## Fixed Decisions

- The final deliverable is the full TTTv2 product stack: reusable modules, reconstructed tensor
  models, model-owned executors, common runtime integration, and vLLM integration.
- Delivery is split into three gated milestones:
  1. reusable 2D modules and focused tests;
  2. direct TTTv2 Llama and Qwen tensor-model reconstructions;
  3. common-runtime executors and vLLM serving.
- DeepSeek-V3 is out of scope. Its full model requires multi-Galaxy execution and its MLA/MoE
  namespace-style stack does not establish the required dense 2D module contract.
- Blackhole is out of scope.
- The only supported and tested 2D mesh orientation is `(8, 4)`.
- Keep and finish the existing `MLP2D` and `RMSNorm2D`; do not rewrite them without a demonstrated
  blocker.
- Add `Attention2D`, `Embedding2D`, `LMHead2D`, `RotarySetup2D`, `Sampling2D`, and common
  `Prefetcher2D` infrastructure.
- Do not add `Penalties2D`.
- Do not add `from_model_args` to any 2D module. The reconstructed models must build explicit module
  configs. Existing prototype `MLP2D.from_model_args` and `RMSNorm2D.from_model_args` shims are not
  part of the target API and should be removed once their direct-config replacements are covered.
- Do not modify any 1D module implementation file. In particular, files such as `attention_1d.py`,
  `embedding_1d.py`, `lm_head_1d.py`, `mlp_1d.py`, `rmsnorm_1d.py`, `rope_1d.py`, and
  `sampling_1d.py` are outside the change set.
- Treat existing 1D module and model behavior as a regression contract. New tests may exercise it,
  but 2D support must not be implemented by adding 2D branches, fields, or hooks to 1D classes.
- Add an immutable generic batched-prefill policy to `llm_runtime`. Galaxy uses physical batch 32,
  at least 16 active rows, no cached prefixes, and a maximum sequence length of 2048. Qualify
  sequence length 128 first, then expand through 2048.
- Prefer changes limited to `llm_runtime` configuration. If configuration alone cannot express a
  required policy, allow only minimal topology-neutral plumbing that delegates to resolved config.
  Do not add Galaxy, Llama, Qwen, 2D-mesh, or `(8, 4)` conditionals to runtime execution code.
- `RMSNorm2D` owns the WH fused residual-add plus distributed RMSNorm path.
- `Attention2D` owns 2D user placement, per-column batch offsets and prefix-cache masks, and optional
  Qwen Q/K normalization.
- Add common `Prefetcher2D` infrastructure. Production module configs receive its resolved
  `global_cb`, subdevice, and weight-address resources.
- Initially retain a Galaxy-specific, mode-aware CCL implementation at the reconstructed model
  layer and inject it into module configs. Do not expand the existing common `TT_CCL` into the
  Galaxy implementation during this work.
- Add a follow-up TODO to evaluate merging Galaxy CCL and common `TT_CCL` after both models pass.
- Add clean, independent Galaxy packages. They may use generic helpers outside model-named
  directories, but they must not import code from any existing model-named package, including
  `models/demos/llama3_70b_galaxy`, `models/common/models/llama33_70b`, or
  `models/common/models/qwen3_32b`.
- CI registration is out of scope and must be a separate follow-up PR.

## Authoritative Design Constraints

All modules follow the contract in `models/common/modules/README.md`:

- one `<Name>2DConfig` dataclass as the source of truth;
- a simple constructor for the common path;
- `from_config(config)` for complete control;
- lazy and explicit weight materialization through `LazyWeight`;
- straight-line mode-specific compute;
- no static topology, architecture, precision, prefetch, or model-family branch in a hot path;
- direct TTNN operations and explicit ownership;
- model-specific tuning supplied through config rather than imported from model code.

The runtime and model stack follow `models/common/llm_runtime/README.md`:

- the model owns graph orchestration;
- the model-owned executor is the resource and cleanup root;
- the common runtime owns reusable planning, staging, compilation, tracing, KV allocation, output
  reads, and warmup mechanics;
- immutable construction policy is resolved before serving;
- traced execution never silently falls back to eager execution;
- vLLM-specific normalization and eager/traced selection remain at the generator boundary.

### Extension discipline

The implementation must follow this order whenever shared code appears insufficient:

1. Express the requirement with an existing config field or injected collaborator.
2. If that is impossible, add a frozen topology-neutral config value or callable policy.
3. If execution code must consume the new value, make the smallest mechanical delegation to the
   resolved config and preserve the previous default exactly.
4. If more than config plus mechanical delegation is required, stop and write a focused reduction
   showing why the requirement cannot remain module-owned or model-owned before changing runtime.

Forbidden shortcuts:

- modifying a 1D module to understand 2D tensors;
- adding `is_galaxy`, model-name, architecture, or mesh-shape branches to common runtime hot paths;
- changing existing 1D model configs to accommodate the 2D path;
- broad runtime refactors bundled with the 2D work;
- silently changing default planner, warmup, trace, cache, or output semantics.

The default-config path before and after this work must be behaviorally identical for every existing
1D model.

### Modularity scorecard

At each milestone, record:

- new 2D/model files added;
- existing shared files changed and why config alone was insufficient;
- 1D module implementation files changed (required value: zero);
- default runtime behaviors changed (required value: zero);
- 1D regression suites run and their result;
- any topology assumption discovered in common code;
- whether the extension stayed in modules/config/model boundaries or leaked into orchestration.

This scorecard is part of the project evidence. Passing model tests while violating these boundaries
does not count as a successful TTTv2 extension.

## Baseline Findings

The reusable 2D inventory currently contains only:

- `models/common/modules/mlp/mlp_2d.py`;
- `models/common/modules/rmsnorm/rmsnorm_2d.py`;
- basic tests for those modules.

The existing WH Galaxy Llama/Qwen stack also contains model-local implementations of embedding,
RoPE, attention, LM head, sampling, distributed norm, prefetching, subdevice management, and
mode-aware CCL. Those implementations are behavioral references and paired-baseline targets, not
dependencies of the new packages.

The current runtime already models batched requests with active rows, padded physical rows,
canonical page tables, and trace identities based on padded geometry. It hard-codes supported
physical batches `{1, 2, 4, 8, 16, 32}` and therefore needs policy injection, not a separate Galaxy
prefill runtime.

The existing Galaxy concat-32 router is narrower than its README:

- active batch is at least 16;
- physical batch is 32;
- rows have a common padded sequence length;
- the currently routed path is sequence length 128;
- cached prefixes are excluded;
- seeded or greedy device sampling requiring slot-stable logits is excluded.

The new generic policy must represent these constraints explicitly. After 128-token qualification,
the same physical path is expanded and tested through sequence length 2048.

## Target Object Graph

```text
provider checkpoint
  -> model-owned HF adaptor and weight conversion
  -> Llama33_70BGalaxyTransformer2D or Qwen3_32BGalaxyTransformer2D
       -> Embedding2D
       -> RotarySetup2D
       -> TransformerBlock2D x N
            -> RMSNorm2D
            -> Attention2D
            -> RMSNorm2D
            -> MLP2D
       -> RMSNorm2D
       -> LMHead2D
       -> optional Sampling2D
       -> borrowed GalaxyCCL
       -> borrowed Prefetcher2D contexts
  -> model-owned executor
       -> PagedKVCacheManager
       -> PrefillRuntime
       -> DecodeRuntime
       -> ProgramCompiler
       -> EagerExecutor
       -> optional TraceCompiler and TracedExecutor
       -> WarmupCoordinator
  -> model-owned vLLM generator
       -> VLLMAdapter
       -> one full-mesh Galaxy executor
```

Galaxy vLLM DP=4 remains four logical request lanes of eight users over one full `(8, 4)` TP32
model. It must not construct four 70B model replicas or four `LaneGroupExecutor` members. The TT
plugin continues collapsing gathered DP to one process and exposes global capacity 32.

## Shared 2D Contracts

### Mesh and static validation

Every resolved 2D config must fail closed unless:

- architecture is Wormhole;
- logical mesh shape is exactly `(8, 4)`;
- device count is exactly 32;
- all weights and injected resources refer to the same mesh;
- dimensions are divisible by the selected row/column partition;
- required CCL and prefetch contexts are compatible with the selected mode.

Tests may construct configs with mocks, but hardware execution is only qualified on real `(8, 4)`
WH Galaxy.

### Tensor placement

Each module documents and validates:

- host/source weight shape;
- 2D mesh mapper and shard dimensions;
- device weight shape after padding;
- decode input/output placement;
- prefill input/output placement;
- ownership and deallocation of transient tensors;
- the residual-stream placement expected by adjacent modules.

Tensor placement is part of the module contract, not implicit model knowledge.

### CCL injection

Define the smallest structural interface required by 2D modules. The interface covers:

- mode-specific reduce-scatter, all-gather, all-reduce, and all-gather-concat resources;
- cluster axis and topology;
- semaphore cycling;
- persistent/intermediate output buffers where required;
- active subdevice/stall-group identity.

Place the initial implementation outside model-named packages, at the reconstructed model layer
(for example `models/common/models/galaxy/ccl.py`), so Llama and Qwen share it without making common
modules import model code. Module configs receive the interface as an injected collaborator.

Do not merge this implementation into `models/common/modules/tt_ccl.py` in this work. Record the
required API overlap and differences for the follow-up merge evaluation.

### Prefetcher ownership

Add `models/common/modules/prefetcher/prefetcher_2d.py` with:

- `Prefetcher2DConfig`;
- immutable prefill and decode context values;
- explicit subdevice-manager creation and activation;
- global circular-buffer allocation;
- explicit registration/sealing of prefetched device weights and address metadata;
- deterministic, idempotent cleanup.

The model creates one prefetcher owner for its mesh. Modules borrow resolved contexts through their
configs. The executor activates prefill/decode context at operation boundaries. Modules do not
create subdevice managers or discover peer modules.

Weight registration must be explicit and sealed before compilation or trace capture. It must not
scan the model graph or materialize unrelated lazy weights proactively.

## Module Work

### `Embedding2D`

Add:

- `models/common/modules/embedding/embedding_2d.py`;
- `models/common/tests/modules/embedding/test_embedding_2d.py`.

Contract:

- embedding table source shape `[vocab_size, dim]`;
- table sharded across the hidden dimension on Galaxy columns and replicated as required on rows;
- decode and prefill output dtype/memory configs resolved separately;
- optional embedding scale;
- output placement compatible with the first `RMSNorm2D`.

Tests cover Llama and Qwen vocabulary/hidden dimensions, decode batch 32, and representative
prefill lengths.

### `RotarySetup2D`

Add:

- `models/common/modules/rope/rope_2d.py`;
- `models/common/tests/modules/rope/test_rope_2d.py`.

Contract:

- lazy cos/sin tables;
- decode lookup for Galaxy's per-column user groups;
- prefill slicing by start position and sequence length;
- explicit batch-group and core-grid config;
- transformation matrices owned by the module;
- support for Llama scaled RoPE and Qwen RoPE parameters through data/config, not model branches.

### `RMSNorm2D`

Finish:

- `models/common/modules/rmsnorm/rmsnorm_2d.py`;
- `models/common/tests/modules/rmsnorm/test_rmsnorm_2d.py`.

Required changes:

- lock production resolution to WH `(8, 4)`;
- add immutable residual policy and mode-specific residual/output memory configs;
- expose a dynamic optional residual tensor;
- implement fused decode residual-add plus distributed RMSNorm;
- return both normalized output and the residual sum when the fused path owns that sum;
- retain straight-line, separately resolved prefill and decode implementations;
- cover ordinary final norm and Qwen per-head Q/K norm geometries;
- remove dependence on `from_model_args`.

The residual tensor is runtime data; whether the operation is fused is static resolved policy.

### `MLP2D`

Finish:

- `models/common/modules/mlp/mlp_2d.py`;
- `models/common/tests/modules/mlp/test_mlp_2d.py`.

Required changes:

- lock production resolution to WH `(8, 4)`;
- validate Llama `8192 -> 28672 -> 8192` and Qwen `5120 -> 25600 -> 5120`;
- resolve separate decode/prefill program, memory, dtype, and kernel configs;
- inject prefetch and Galaxy CCL resources;
- preserve the 2D W1/W3 reduce-scatter, activation, W2 all-gather/matmul, and final reduction dataflow;
- add sequence-keyed prefill config factories where geometry changes;
- remove dependence on `from_model_args`;
- add representative real-hardware tests instead of importing legacy model configs.

### `Attention2D`

Add:

- `models/common/modules/attention/attention_2d.py`;
- `models/common/tests/modules/attention/test_attention_2d.py`;
- optional profiling tests under `models/common/tests/modules/attention/profiling/`.

`Attention2DConfig` owns:

- fused QKV and WO lazy weights and optional QKV bias;
- optional Q/K `RMSNorm2DConfig` values;
- heads, KV heads, head dimension, scale, maximum batch, and maximum sequence;
- 2D weight and activation placement;
- decode and prefill program/memory/kernel configs;
- paged KV metadata and externally bound cache tensors;
- prefetch and Galaxy CCL collaborators;
- `users_per_column=8`;
- batch-offset tensor construction;
- per-column lower/upper prefix-cache bounds;
- chunked-SDPA alignment and sequence-keyed configs;
- output placement required by residual accumulation.

Execution coverage includes:

- decode QKV, head creation, RoPE, paged-cache update, SDPA, head concat, WO, and reductions;
- single-row prefill;
- physical concat-32 prefill;
- paged KV fill;
- prefix-cached/chunked prefill;
- Llama without Q/K norm;
- Qwen with Q/K norm and QKV bias if required by the checkpoint.

Static Llama/Qwen differences resolve into data or bound callables during construction. The
mode-specific forward path must not branch on model family.

### `LMHead2D`

Add:

- `models/common/modules/lm_head/lm_head_2d.py`;
- `models/common/tests/modules/lm_head/test_lm_head_2d.py`.

Contract:

- padded vocabulary and logical valid vocabulary are distinct;
- 2D weight sharding yields complete per-user vocabulary logits after the configured collective;
- decode and prefill weights/configs may differ but remain lazy;
- large output projection may be split explicitly;
- output shape and shard metadata are stable for host logits and `Sampling2D`;
- invalid padded vocabulary entries cannot be sampled.

Cover Llama vocabulary 128256 and Qwen vocabulary 151936 with their required tile/device padding.

### `Sampling2D`

Add:

- `models/common/modules/sampling/sampling_2d.py`;
- `models/common/tests/modules/sampling/test_sampling_2d.py`.

Contract:

- top-k, top-p, temperature, seed, and forced argmax remain per-call values;
- mutable device state uses `LazyBuffer`;
- sampling understands 2D vocab sharding and Galaxy user placement;
- deterministic seeded requests remain slot-stable;
- greedy, seeded stochastic, and unseeded stochastic paths are covered;
- no penalties functionality is added.

## Configuration-First `llm_runtime` Changes

### Shared-code change budget

The preferred runtime diff is confined to immutable configuration and its tests. The first
implementation attempt must determine whether the Galaxy path can be expressed with existing
planner/runtime hooks plus model-owned input reshaping.

If existing hooks are insufficient, the permitted fallback is:

- define the topology-neutral batched-prefill policy in `prefill/config.py`;
- replace hard-coded batch selection with delegation to that resolved policy;
- thread the policy through the minimum required call sites;
- preserve the current policy as the exact default.

Likely plumbing sites are `prefill/plan.py` and `prefill/runtime.py`; touching them requires a focused
test that fails without the delegation and passes with it. `warmup.py`, signatures, tracing, decode,
KV ownership, output reading, and execution composition should remain unchanged unless a separate
reduction proves a generic contract gap.

No runtime file may import a 2D module or reconstructed model package.

### Immutable batched-prefill policy

Add a frozen policy value owned by `PrefillRuntimeConfig`, with at least:

- supported physical batch sizes;
- minimum active rows;
- maximum physical batch;
- maximum batched sequence length;
- maximum total tokens;
- cached-prefix eligibility;
- requirements imposed by sampling/output extraction.

Default resolution must preserve current non-Galaxy behavior exactly.

Galaxy resolves:

```text
physical_batch_sizes = (32,)
minimum_active_rows = 16
maximum_physical_batch = 32
maximum_sequence_length = 2048
allow_cached_prefix = false
```

The planner continues producing rank-2 `[physical_batch, sequence]` token tensors, source-row
metadata, slot mapping, and padded page tables. The Galaxy model reshapes/concatenates those rows
for its device graph; the generic runtime must not contain a Llama/Qwen-specific flatten.

If a callable policy is used, it receives only generic request/config facts and returns generic
physical planning facts. The Galaxy values are supplied by the model-owned executor while the
runtime remains unaware of Galaxy.

### Planning and trace coverage

Extend focused host tests to prove:

- 15 active rows remain sequential;
- 16, 31, and 32 eligible rows produce one physical-32 request;
- padding rows have zero tokens and `-1` page-table entries;
- source-row and slot order survive assembly;
- cached rows remain sequential/chunked;
- sequence length above 2048 remains sequential;
- slot-stable seeded/greedy sampling selects the qualified safe plan;
- program and trace signatures use physical batch 32, not active row count;
- default 1D policies are unchanged;
- every pre-existing planner test passes without changed expectations.

Qualify physical-32 traces at sequence length 128 first. Then add each supported padded sequence
length through 2048 to warmup and trace coverage.

### Galaxy operation-boundary activation

The model-owned executor activates the required prefetcher/CCL subdevice context before delegating
to prefill or decode runtime execution. A mode transition is an operation-boundary lifecycle action,
not a static branch inside a module hot path.

Compilation, capture, replay, and cleanup tests must include transitions:

- decode to prefill;
- prefill to decode;
- repeated prefill;
- repeated decode;
- failure during transition;
- cleanup from either active mode.

## Clean Model Packages

### Llama

Add:

```text
models/common/models/llama33_70b_galaxy/
  __init__.py
  hf_adaptor.py
  model.py
  executor.py
  generator.py
  demo.py
```

The package:

- targets exactly `meta-llama/Llama-3.3-70B-Instruct`;
- owns provider key/layout conversion and Llama 3 scaled-RoPE preparation;
- builds only the new 2D module configs;
- owns precision/optimization recipes;
- exposes prefill/decode graph methods and runtime input helpers;
- exposes `iter_executor_named_modules`;
- transactionally binds/unbinds paged KV cache;
- exposes per-layer KV metadata;
- never imports the existing 1D or legacy Galaxy model package.

### Qwen

Add:

```text
models/common/models/qwen3_32b_galaxy/
  __init__.py
  hf_adaptor.py
  model.py
  executor.py
  generator.py
  demo.py
```

The package:

- targets exactly `Qwen/Qwen3-32B`;
- independently owns provider key/layout conversion;
- builds `Attention2D` with Q/K norm;
- builds the Qwen MLP, embedding, RoPE, norm, LM head, and sampler configs;
- owns Qwen precision/optimization recipes;
- satisfies the same executor and KV contracts as Llama;
- never imports the existing 1D or legacy Galaxy model package.

### Permitted sharing

Both packages may use:

- reusable `models/common/modules`;
- `models/common/llm_runtime`;
- generic tensor, checkpoint, tokenizer, and mesh helpers not located under a model-named directory;
- the shared reconstructed-model Galaxy CCL implementation outside model-named packages.

Do not extract code from an existing model package merely to avoid writing the new package. Promote
only a genuinely topology/model-neutral helper with tests and at least two immediate callers.

## Milestone A: Reusable 2D Modules

### Sequence

1. Add the injected Galaxy CCL interface and shared model-layer implementation.
2. Add `Prefetcher2D` and lifecycle tests.
3. Add `Embedding2D`, `RotarySetup2D`, and `LMHead2D`.
4. finish `RMSNorm2D` and `MLP2D`.
5. Add `Attention2D`.
6. Add `Sampling2D`.
7. Add the generic batched-prefill policy and host/runtime tests.
8. Update `models/common/modules/README.md` with the final 2D inventory and contracts.
9. Audit the diff and complete the modularity scorecard before opening Milestone B.

### Representative dimensions

Llama:

```text
hidden = 8192
intermediate = 28672
heads = 64
kv_heads = 8
head_dim = 128
vocab = 128256
```

Qwen:

```text
hidden = 5120
intermediate = 25600
head_dim = 128
vocab = 151936
Q/K norm = enabled
```

Use explicit representative configs and tensor sizes. Module tests must not import configuration or
implementation code from the previous model stacks.

### Milestone A exit gate

Every module must:

- pass host-only config validation tests;
- pass real WH `(8, 4)` decode and prefill tests;
- cover representative Llama and Qwen geometry where applicable;
- achieve PCC `>= 0.99` against an independent PyTorch/Hugging Face reference;
- achieve KV-cache PCC `>= 0.99` where applicable;
- pass ownership/cleanup and repeat-invocation tests;
- demonstrate that prefetch/CCL/static strategy is resolved before the hot path;
- have no `from_model_args` dependency;
- leave every 1D module implementation file unchanged;
- pass the existing 1D module test suite;
- preserve every pre-existing default-runtime test and expectation;
- keep any runtime execution-code change to tested, topology-neutral config delegation.

Do not begin model reconstruction until all Milestone A gates pass.

## Milestone B: Direct Tensor-Model Reconstructions

### Sequence

1. Build the Llama provider adaptor and one-layer model.
2. Validate one Llama block in decode and prefill.
3. Scale to the full 80-layer Llama model and direct demo.
4. Build the Qwen provider adaptor and one-layer model.
5. Validate one Qwen block in decode and prefill.
6. Scale to the full Qwen model and direct demo.
7. Add paged KV, prefix-cache, concat-32, device-sampling, and long-context direct-demo coverage.

### Milestone B tests

For both models:

- host-only adaptor and config tests;
- one-layer decode and prefill PCC;
- full-model prefill plus first decode token;
- teacher-forced decode;
- batch 1 and batch 32;
- paged KV;
- prefix-cached/chunked prefill;
- physical-32 prefill at sequence length 128, then through 2048;
- greedy and stochastic device sampling;
- repeated requests and deterministic cleanup.

### Milestone B exit gate

On WH `(8, 4)`:

- Llama teacher-forced accuracy at batch 1, prefill 512/decode 511:
  - top-1 `>= 91%`;
  - top-5 `>= 99%`.
- Qwen teacher-forced accuracy at batch 1, sequence length 512:
  - top-1 `>= 89%`;
  - top-5 `>= 97%`.
- Batch-32 direct demos produce valid output with no cross-slot contamination.
- Batch-1 4K, 32K, and 128K functional smokes pass.
- Prefix-cached output matches uncached execution under the model's numerical acceptance.
- No dependency imports come from an existing model-named implementation package.
- The Milestone B diff still contains zero changes to 1D module implementation files.
- Existing 1D model contract and demo-contract host tests remain green without expectation changes.

Do not begin executor/vLLM integration until both models pass Milestone B.

## Milestone C: Executors, Runtime, Tracing, and vLLM

### Model-owned executors

Each executor composes:

- one full-mesh 2D tensor model;
- one `PagedKVCacheManager`;
- one `OutputReader`;
- resolved prefill/decode configs using the Galaxy batched-prefill policy;
- one `ProgramCompiler`;
- one `EagerExecutor`;
- optional `TraceCompiler` and `TracedExecutor` over that exact eager executor;
- one `WarmupCoordinator`;
- the model's prefetcher and Galaxy CCL lifecycle.

The executor is terminal after cleanup and releases resources in common-runtime order, with
prefetcher/CCL resources released after outstanding work and before mesh teardown.

### Generator and vLLM boundary

Each generator:

- uses `VLLMAdapter` for request normalization and KV-shape validation;
- selects eager/traced execution at the model boundary;
- preserves explicit trace misses rather than hiding them in `TracedExecutor`;
- maps vLLM's logical DP=4/max-num-seqs=8 contract to global capacity 32 on one full Galaxy model;
- supports async decode output;
- exposes device sampling;
- owns no TT tensors.

Update the TT plugin's exact model/version routing so:

- Galaxy Llama selects
  `models.common.models.llama33_70b_galaxy.generator:Llama33_70BGalaxyGenerator`;
- Galaxy Qwen selects
  `models.common.models.qwen3_32b_galaxy.generator:Qwen3_32BGalaxyGenerator`;
- existing non-Galaxy TTTv2 Llama/Qwen mappings remain unchanged;
- existing invalid-version behavior remains fail-closed;
- gathered DP continues collapsing into the existing single-process lane coordinator.

Add host tests for both exact mappings and Galaxy DP conversion. Do not change unrelated vLLM core,
packaging, scheduler, or loader behavior.

### Milestone C functional gate

For both models:

- direct eager prefill/decode;
- eager compilation and warmup;
- traced decode;
- traced eligible prefill;
- explicit eager handling for trace-ineligible requests;
- eager/traced logits PCC `>= 0.999` for the same prepared request;
- identical deterministic sampled tokens between eager and traced execution;
- paged-KV late capacity resolution;
- prefix caching and chunked prefill;
- async decode read/complete;
- DP=4 logical-lane request isolation;
- repeated startup, serving, and cleanup without retained TT resources.

Run vLLM server/offline smoke tests with:

```text
--data_parallel_size 4
--max_num_seqs 8
global capacity = 32
```

### Performance methodology

Use paired TTTv1 and TTTv2 measurements:

- same WH Galaxy;
- same repository commit and firmware/runtime environment;
- same checkpoint, precision recipe, prompt corpus, batch, sequence, trace, sampling, and KV setup;
- one unmeasured warmup;
- three measured runs;
- compare medians;
- retain profiler artifacts and exact commands.

No gated TTTv2 metric may regress by more than 3% from its paired TTTv1 median. TTTv2 must also meet
the active absolute targets:

Llama, batch 32 / sequence length 507:

```text
TTFT <= 99 ms
decode >= 71.5 tokens/s/user
aggregate decode >= 2288 tokens/s
```

Qwen, batch 32 / sequence length 507:

```text
TTFT <= 700 ms
decode >= 60 tokens/s/user
aggregate decode >= 1920 tokens/s
```

If an absolute target and paired baseline disagree materially, stop and document the environment and
baseline discrepancy instead of weakening either gate silently.

### Milestone C exit gate

Milestone C is complete only when:

- all functional gates pass for both models;
- all paired and absolute performance gates pass;
- all accuracy gates remain satisfied in the performance configuration;
- vLLM DP=4 serves 32-request global concurrency correctly;
- no legacy model implementation is imported by either new package;
- no 1D module implementation file has changed;
- existing 1D executor/runtime integration tests pass with their original expected behavior;
- every non-config runtime change has a focused generic reduction and regression test;
- the final modularity scorecard shows zero default-runtime behavior changes;
- cleanup is repeatable and terminal;
- the exact commands, measurements, and revisions are recorded.

## Test Organization

Add or update:

```text
models/common/tests/modules/prefetcher/test_prefetcher_2d.py
models/common/tests/modules/embedding/test_embedding_2d.py
models/common/tests/modules/rope/test_rope_2d.py
models/common/tests/modules/rmsnorm/test_rmsnorm_2d.py
models/common/tests/modules/mlp/test_mlp_2d.py
models/common/tests/modules/attention/test_attention_2d.py
models/common/tests/modules/lm_head/test_lm_head_2d.py
models/common/tests/modules/sampling/test_sampling_2d.py
models/common/tests/llm_runtime/test_prefill_runtime.py
models/common/tests/llm_runtime/test_warmup.py
models/common/tests/llm_runtime/test_executor_integration.py
models/common/tests/models/llama33_70b_galaxy/
models/common/tests/models/qwen3_32b_galaxy/
```

Keep fast host/config tests separate from tests requiring real Galaxy hardware. Hardware tests should
state mesh, checkpoint, mode, batch, sequence, and optimization requirements in their IDs/markers.
CI registration is deliberately deferred.

The regression run must include the existing 1D module tests and the existing common-runtime,
model-contract, and executor-integration host tests. Do not update their expected values merely to
accept the 2D extension.

## Risks and Required Reductions

### Prefetcher and lazy-weight interaction

Risk: global-CB address registration may force eager weight materialization or make cache identity
opaque.

Required response: reduce the smallest registration/materialization case, keep registration explicit,
and preserve lazy construction. Do not hide graph scanning in `Prefetcher2D`.

### CCL/subdevice mode transitions

Risk: prefill/decode semaphore or stall-group state leaks across compilation, trace capture, and replay.

Required response: add transition tests before adding model-local retries or synchronization.

### Concat-32 planning

Risk: padding inactive rows writes KV or returns logits for inactive slots.

Required response: inspect planned tokens/page tables/source rows and test KV/logit isolation for
active batches 16, 31, and 32.

### Qwen Q/K norm

Risk: head-local normalization is accidentally treated as hidden-dimension distributed norm.

Required response: give Q/K norm explicit geometry in `RMSNorm2DConfig` and validate it independently
before enabling it in `Attention2D`.

### Trace identity

Risk: active batch size or runtime slot assignment leaks into program/trace identity and causes trace
explosion or stale replay state.

Required response: keep physical geometry in identity and refresh active-row/slot data as trace input.

### Performance parity

Risk: a functionally correct no-prefetch fallback passes PCC while missing the project goal.

Required response: production acceptance always uses the configured `Prefetcher2D` path and paired
performance gate. A fallback may aid diagnosis but cannot satisfy a milestone.

### Shared-runtime leakage

Risk: Galaxy-specific planning, mode switching, or tensor shaping leaks into common orchestration and
changes 1D behavior.

Required response: first move the decision into 2D module/model config. If generic runtime plumbing
is unavoidable, require a focused reduction, preserve the old default policy byte-for-byte, and run
the full 1D runtime regression set. Reject any solution requiring a Galaxy/model branch in runtime.

### 1D module coupling

Risk: adding a 2D feature appears easiest by extending a 1D class or generalizing a hot-path helper
inside a `*_1d.py` file.

Required response: do not make the change. Put the behavior in the corresponding `*_2d.py` module or
promote a truly dimension-neutral helper into a separate common file with independent tests. The
1D module source remains untouched.

## Explicit Follow-up TODOs

These are not part of this implementation:

1. Add WH Galaxy CI registrations in a separate PR after the complete local hardware evidence exists.
2. Evaluate merging Galaxy CCL with `models/common/modules/tt_ccl.py`; require an API/ownership
   comparison and regression coverage for 1D and 2D users first.
3. Evaluate Blackhole Galaxy support separately.
4. Evaluate DeepSeek-V3 only after the dense 2D module/model stack is complete; scope MLA, MoE, and
   multi-host runtime independently.
5. Evaluate `(4, 8)` orientation only if a product requirement appears.
6. Add `Penalties2D` only when a concrete serving requirement cannot be satisfied by `Sampling2D`.

## Definition of Done

The project is done when the reusable modules, both clean tensor models, both model-owned executors,
and both vLLM paths pass all three milestone gates on WH `(8, 4)`, with recorded accuracy and
performance evidence, no imports from prior model-named implementations, and no more than 3% paired
performance regression while meeting active absolute targets. The final diff must contain zero
changes to 1D module implementation files, zero Galaxy/model branches in common runtime, and no
change to existing default 1D runtime behavior. The modularity scorecard must show that the extension
was achieved through new 2D modules, model-owned composition, and immutable configuration, with only
minimal tested generic runtime delegation where configuration could not stand alone.
