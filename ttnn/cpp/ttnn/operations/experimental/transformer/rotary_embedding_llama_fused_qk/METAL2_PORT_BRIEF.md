# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ✓ · Features ✓

## TTNN factory analysis

The factory concept is selected downstream from these facts (→ `port_op_to_metal2_ttnn_factory.md`); the port does not pick it here. Carry these forward:

- **Op-owned tensors:** none
- **MeshWorkload:** not needed
- **Pybind `create_descriptor`:** none
- **Other risky pybind:** none
- **Custom `override_runtime_arguments`:** none

## Construct — to do

**Tensor bindings:** N/A — both kernels are pure compute kernels. They access tensor data entirely through borrowed-memory CBs; there are no `TensorAccessor` bindings to re-express. The two input CBs (c_0 q_input, c_1 k_input) have genuine producer+consumer FIFO cycles and translate as borrowed-memory DFBs via `DataflowBufferSpec::borrowed_from`. The five fake CBs are handled by the fake-CB workaround (see Watch for).

**Custom hash:** none

## Watch for

- **Notable constructs — borrowed-memory DFBs (7 CBs):** All seven `CBDescriptor`s in the factory have `.buffer` set to a non-null sharded tensor buffer. Port each with `DataflowBufferSpec::borrowed_from = <tensor_parameter_name>`. The seven CBs and their factory sites (`rotary_embedding_llama_fused_qk_program_factory.cpp`):
  - c_0 `q_input_cb` → `borrowed_from` the q_input tensor parameter (line 103)
  - c_1 `k_input_cb` → `borrowed_from` the k_input tensor parameter (line 115)
  - c_2 `cos_cb` → `borrowed_from` cos (line 127) — **fake CB, see below**
  - c_3 `sin_cb` → `borrowed_from` sin (line 139) — **fake CB, see below**
  - c_4 `trans_mat_cb` → `borrowed_from` trans_mat (line 153) — **fake CB, see below**
  - c_16 `q_output_cb` → `borrowed_from` q_output (line 199) — **fake CB, see below**
  - c_17 `k_output_cb` → `borrowed_from` k_output (line 210) — **fake CB, see below**

- **Fake CBs (address-only — 5 CBs, sanctioned workaround applies):** Five of the seven borrowed-memory CBs lack a genuine producer+consumer FIFO cycle and cannot be expressed as Metal 2.0 DFBs directly (spec validator requires ≥1 producer and ≥1 consumer):
  - `cos_cb` (c_2): wrapper object instantiated but never called; used only as raw uint32_t in LLK calls. Both kernel variants.
  - `sin_cb` (c_3): same pattern. Both kernel variants.
  - `trans_mat_cb` (c_4): same pattern. Both kernel variants.
  - `q_output_cb` (c_16): compute kernel calls `.reserve_back` + `.push_back` but no `.wait_front` / consumer. Both kernel variants.
  - `k_output_cb` (c_17): same as q_output_cb. Both kernel variants.

  Apply the fake-CB workaround from the porting recipe for each of these five. The workaround keeps the port unblocked.

- **Cross-op / shared kernels:** none — both kernel files are op-owned; all includes are LLK/HAL tier.

- **RTA varargs:** none
