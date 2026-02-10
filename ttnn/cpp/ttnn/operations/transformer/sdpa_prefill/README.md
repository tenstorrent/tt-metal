# Prefill Optimized SDPA

This is a placeholder directory for the coming implementation of prefill optimized SDPA multi-device op. Implementaion will be tailored for the DeepSeek V3 model with long sequence lenghts.

## Key difference from existing SDPA implementations

This version of the op will enable causality on sequence distributed MLA SDPA.

## Implementation steps

1. Sequnce distributed causal attention - /wo KV cache communication

2. Ring causal attention

3. Load balancing

## API

TBD
