# Metal 2.0 Host-Port Pre-Port Audit — `experimental/quasar/to_device`

## Verdict: **N/A — no device program to port** (pure host op)

`to_device` has no `device/` directory, no device operation, and no program factory. `to_device.cpp`
forwards to the core tensor transfer method:

```cpp
return tensor.to_device(mesh_device, mem_config, queue_id);
```

`Tensor::to_device` is a core tt-metal host/runtime path (host→device data placement), not a TTNN
device operation built from kernels — there are no compute/dataflow kernels and no ProgramDescriptor.

**Check 1 (ProgramDescriptor prerequisite): N/A**; all other audit subjects are N/A. No host-2.0 work
is required on `to_device`.
