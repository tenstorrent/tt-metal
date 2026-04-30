# Sweep Trace Validation Report

**Total master configs:** 130
**Exact matches:** 129
**With diffs:** 0
**Hash mismatch (args match, hash differs):** 0
**Not exercised by sweep:** 1
**Incidental (non-target ops):** 3
**Coverage:** 99.2%

## Per-operation summary

| Operation | Match | Diff | Hash Mismatch | Missing | Total |
|-----------|------:|-----:|--------------:|--------:|------:|
| `ttnn.add` | 16 | 0 | 0 | 0 | 16 |
| `ttnn.clamp` | 4 | 0 | 0 | 0 | 4 |
| `ttnn.embedding` | 3 | 0 | 0 | 0 | 3 |
| `ttnn.experimental.all_gather_async` | 1 | 0 | 0 | 0 | 1 |
| `ttnn.experimental.fast_reduce_nc` | 1 | 0 | 0 | 0 | 1 |
| `ttnn.experimental.nlp_concat_heads` | 1 | 0 | 0 | 0 | 1 |
| `ttnn.experimental.nlp_concat_heads_decode` | 1 | 0 | 0 | 0 | 1 |
| `ttnn.experimental.nlp_create_qkv_heads` | 1 | 0 | 0 | 0 | 1 |
| `ttnn.experimental.nlp_create_qkv_heads_decode` | 1 | 0 | 0 | 0 | 1 |
| `ttnn.experimental.paged_fill_cache` | 1 | 0 | 0 | 0 | 1 |
| `ttnn.experimental.paged_update_cache` | 1 | 0 | 0 | 0 | 1 |
| `ttnn.experimental.rotary_embedding_llama` | 4 | 0 | 0 | 0 | 4 |
| `ttnn.interleaved_to_sharded` | 0 | 0 | 0 | 1 | 1 |
| `ttnn.linear` | 3 | 0 | 0 | 0 | 3 |
| `ttnn.matmul` | 7 | 0 | 0 | 0 | 7 |
| `ttnn.multiply` | 9 | 0 | 0 | 0 | 9 |
| `ttnn.pad` | 1 | 0 | 0 | 0 | 1 |
| `ttnn.permute` | 3 | 0 | 0 | 0 | 3 |
| `ttnn.plus_one` | 2 | 0 | 0 | 0 | 2 |
| `ttnn.repeat` | 1 | 0 | 0 | 0 | 1 |
| `ttnn.reshape` | 17 | 0 | 0 | 0 | 17 |
| `ttnn.rms_norm` | 5 | 0 | 0 | 0 | 5 |
| `ttnn.scatter` | 2 | 0 | 0 | 0 | 2 |
| `ttnn.sigmoid` | 2 | 0 | 0 | 0 | 2 |
| `ttnn.slice` | 6 | 0 | 0 | 0 | 6 |
| `ttnn.softmax` | 2 | 0 | 0 | 0 | 2 |
| `ttnn.sum` | 1 | 0 | 0 | 0 | 1 |
| `ttnn.tilize_with_zero_padding` | 1 | 0 | 0 | 0 | 1 |
| `ttnn.topk` | 3 | 0 | 0 | 0 | 3 |
| `ttnn.transformer.paged_scaled_dot_product_attention_decode` | 2 | 0 | 0 | 0 | 2 |
| `ttnn.transformer.scaled_dot_product_attention` | 2 | 0 | 0 | 0 | 2 |
| `ttnn.transpose` | 6 | 0 | 0 | 0 | 6 |
| `ttnn.typecast` | 6 | 0 | 0 | 0 | 6 |
| `ttnn.unsqueeze` | 1 | 0 | 0 | 0 | 1 |
| `ttnn.unsqueeze_to_4D` | 8 | 0 | 0 | 0 | 8 |
| `ttnn.untilize` | 2 | 0 | 0 | 0 | 2 |
| `ttnn.zeros_like` | 2 | 0 | 0 | 0 | 2 |

## Master configs not exercised by sweep

1 configs had no corresponding sweep execution.

<details><summary>Show all</summary>

| Operation | Config ID | Config Hash |
|-----------|----------:|-------------|
| `ttnn.interleaved_to_sharded` | 378 | `f57b9423de491eaf...` |

</details>
