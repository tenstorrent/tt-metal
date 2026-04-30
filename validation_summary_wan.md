# Sweep Trace Validation Report

**Total master configs:** 249
**Exact matches:** 227
**With diffs:** 2
**Hash mismatch (args match, hash differs):** 0
**Not exercised by sweep:** 20
**Incidental (non-target ops):** 0
**Coverage:** 92.0%

## Per-operation summary

| Operation | Match | Diff | Hash Mismatch | Missing | Total |
|-----------|------:|-----:|--------------:|--------:|------:|
| `ttnn.add` | 18 | 0 | 0 | 1 | 19 |
| `ttnn.clamp` | 2 | 0 | 0 | 0 | 2 |
| `ttnn.concat` | 34 | 0 | 0 | 0 | 34 |
| `ttnn.cos` | 1 | 0 | 0 | 0 | 1 |
| `ttnn.embedding` | 2 | 0 | 0 | 0 | 2 |
| `ttnn.experimental.all_gather_async` | 12 | 2 | 0 | 0 | 14 |
| `ttnn.experimental.nlp_create_qkv_heads` | 2 | 0 | 0 | 0 | 2 |
| `ttnn.lerp` | 0 | 0 | 0 | 2 | 2 |
| `ttnn.matmul` | 2 | 0 | 0 | 0 | 2 |
| `ttnn.multiply` | 12 | 0 | 0 | 14 | 26 |
| `ttnn.pad` | 15 | 0 | 0 | 0 | 15 |
| `ttnn.permute` | 5 | 0 | 0 | 0 | 5 |
| `ttnn.pow` | 1 | 0 | 0 | 0 | 1 |
| `ttnn.repeat` | 1 | 0 | 0 | 0 | 1 |
| `ttnn.reshape` | 53 | 0 | 0 | 0 | 53 |
| `ttnn.silu` | 2 | 0 | 0 | 0 | 2 |
| `ttnn.sin` | 1 | 0 | 0 | 0 | 1 |
| `ttnn.slice` | 36 | 0 | 0 | 0 | 36 |
| `ttnn.softmax` | 1 | 0 | 0 | 0 | 1 |
| `ttnn.squeeze` | 3 | 0 | 0 | 0 | 3 |
| `ttnn.subtract` | 1 | 0 | 0 | 0 | 1 |
| `ttnn.tanh` | 0 | 0 | 0 | 1 | 1 |
| `ttnn.transformer.concatenate_heads` | 3 | 0 | 0 | 0 | 3 |
| `ttnn.transformer.scaled_dot_product_attention` | 0 | 0 | 0 | 2 | 2 |
| `ttnn.transformer.split_query_key_value_and_split_heads` | 2 | 0 | 0 | 0 | 2 |
| `ttnn.typecast` | 1 | 0 | 0 | 0 | 1 |
| `ttnn.unsqueeze` | 5 | 0 | 0 | 0 | 5 |
| `ttnn.unsqueeze_to_4D` | 6 | 0 | 0 | 0 | 6 |
| `ttnn.upsample` | 6 | 0 | 0 | 0 | 6 |

## Master configs not exercised by sweep

20 configs had no corresponding sweep execution.

<details><summary>Show all</summary>

| Operation | Config ID | Config Hash |
|-----------|----------:|-------------|
| `ttnn.add` | 211 | `eacd0f9b2db280c2...` |
| `ttnn.lerp` | 248 | `22874736914b6f9f...` |
| `ttnn.lerp` | 41 | `4482093c19095b14...` |
| `ttnn.multiply` | 114 | `024a93e53a521bc9...` |
| `ttnn.multiply` | 66 | `0ad40cb91a9ccd17...` |
| `ttnn.multiply` | 126 | `103067643295e4ce...` |
| `ttnn.multiply` | 86 | `28e411ed84c2b157...` |
| `ttnn.multiply` | 81 | `414c36292846bb28...` |
| `ttnn.multiply` | 76 | `4a997a1cea202796...` |
| `ttnn.multiply` | 71 | `58fc1d9a253f62c9...` |
| `ttnn.multiply` | 120 | `759f75e9b797f144...` |
| `ttnn.multiply` | 136 | `76c6a044da4d748c...` |
| `ttnn.multiply` | 147 | `8b5f252cec26fa66...` |
| `ttnn.multiply` | 91 | `987e4cb0e0c89328...` |
| `ttnn.multiply` | 96 | `b576053a6cd36db6...` |
| `ttnn.multiply` | 153 | `cb0636b337080036...` |
| `ttnn.multiply` | 142 | `d64d6a481df55d58...` |
| `ttnn.tanh` | 241 | `0e267f3da3198912...` |
| `ttnn.transformer.scaled_dot_product_attention` | 59 | `687849b40db7e7ad...` |
| `ttnn.transformer.scaled_dot_product_attention` | 24 | `d5137a7395384ee4...` |

</details>

## Diff categories

| Category | Count | Description |
|----------|------:|-------------|
| `extra_key` | 2 | extra or missing key |

## Detailed diffs

### `ttnn.experimental.all_gather_async` config_hash `efe5ef58fac15396...`
master config_id=220, sweep config_id=37

| Path | Category | Master | Sweep |
|------|----------|--------|-------|
| `persistent_output_buffer` | `extra_key` | None | <missing> |

### `ttnn.experimental.all_gather_async` config_hash `f2b296da91b09cb7...`
master config_id=232, sweep config_id=38

| Path | Category | Master | Sweep |
|------|----------|--------|-------|
| `persistent_output_buffer` | `extra_key` | None | <missing> |
