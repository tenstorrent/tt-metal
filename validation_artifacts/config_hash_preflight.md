## Config Hash Preflight

The config hash preflight check computes the config hashes of the reconstructed model traces using the current version of `model_tracer/generic_ops_tracer.py` config hash computation function to compare with config hashes stored in the database. It serves as a preflight check to see if the current method of config hash computation has varied from the time that the trace was added to the database.

| Field | Value |
|---|---|
| Hash source | `model_tracer.generic_ops_tracer.recompute_config_hashes()` |
| Hash method | `_compute_config_hash()` |
| JSON | `model_tracer/traced_operations/ttnn_operations_wan.json` |
| Configs | 249 |
| Changed | 1 (0.4%) |
| Allow partial | `True` |
| Decision | `continue_partial_changed` |

<details><summary>Changed hashes by operation (1 ops)</summary>

| Operation | Changed hashes |
|---|---:|
| `ttnn.multiply` | 1 |

</details>
