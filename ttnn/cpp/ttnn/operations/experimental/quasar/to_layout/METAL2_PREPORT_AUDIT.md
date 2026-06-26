# Metal 2.0 Host-Port Pre-Port Audit — `experimental/quasar/to_layout`

## Verdict: **N/A — no device program to port** (pure host dispatcher)

`to_layout` has no `device/` directory, no device operation, and no program factory
(`create_descriptor` / `ProgramDescriptor` / `KernelDescriptor` — none present). `to_layout_op.cpp`
is host orchestration that selects and invokes other quasar ops based on the requested layout / dtype /
memory config:
- `tilize`, `tilize_with_val_padding`
- `untilize`, `untilize_with_unpadding`
- `reshape_view` / experimental `reshape::view`

**Check 1 (ProgramDescriptor prerequisite): N/A** — there is no factory to convert from
`create_descriptor` to `create_program_artifacts`. All remaining audit subjects (Device 2.0, feature
compatibility, TensorAccessor handling, etc.) are likewise N/A: they concern kernel/program artifacts
that this op does not own.

**Downstream dependency (FYI, not a gate):** `to_layout`'s end-to-end Metal 2.0 behavior is entirely
inherited from the sub-ops it dispatches to. Those are ported / tracked independently:
- `untilize` — host-2.0 DONE (8/8 factories).
- `tilize_with_val_padding` — 2/4 factories done; 2 scratchpad-blocked.
- `tilize`, `untilize_with_unpadding`, `reshape_view` — per their own reports.

No host-2.0 work is required on `to_layout` itself.
