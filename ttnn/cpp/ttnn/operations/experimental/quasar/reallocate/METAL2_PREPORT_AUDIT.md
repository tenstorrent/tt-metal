# Metal 2.0 Host-Port Pre-Port Audit — `experimental/quasar/reallocate`

## Verdict: **N/A — no device program to port** (pure host dispatcher)

`reallocate` has no `device/` directory, no device operation, and no program factory. `reallocate.cpp`
is a one-line forwarder:

```cpp
Tensor reallocate(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    return ttnn::operations::experimental::quasar::move(input_tensor, memory_config);
}
```

**Check 1 (ProgramDescriptor prerequisite): N/A** — no factory exists here; all other audit subjects
are N/A for the same reason.

**Downstream dependency (FYI, not a gate):** the actual device work is owned by `quasar::move`
(`move/device/...`, which has its own program factory and its own Metal 2.0 status). No host-2.0 work
is required on `reallocate` itself.
