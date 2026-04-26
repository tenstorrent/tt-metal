import torch
import ttnn

class TtBarkAttention:
    def __init__(self, device, parameters, base_address, config):
        self.device = device
        self.parameters = parameters
        self.base_address = base_address
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        
        # Load weights and shard them if needed
        self.q_proj = parameters[f"{base_address}.attn.q_proj.weight"]
        self.k_proj = parameters[f"{base_address}.attn.k_proj.weight"]
        self.v_proj = parameters[f"{base_address}.attn.v_proj.weight"]
        self.out_proj = parameters[f"{base_address}.attn.out_proj.weight"]

    def forward(self, x, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        # x: [1, 1, seq_len, hidden_size]
        
        # Parallel Projections
        q = ttnn.linear(x, self.q_proj, memory_config=memory_config)
        k = ttnn.linear(x, self.k_proj, memory_config=memory_config)
        v = ttnn.linear(x, self.v_proj, memory_config=memory_config)
        
        # Split heads (Simplified for bring-up)
        # In a real optimized version, we'd use ttnn.transformer.split_query_key_value_and_split_heads
        
        # Compute Attention Scores
        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len)
        attn_weights = ttnn.matmul(q, ttnn.transpose(k, -2, -1), memory_config=memory_config)
        attn_weights = ttnn.softmax(attn_weights, dim=-1)
        
        # Attn Output
        attn_output = ttnn.matmul(attn_weights, v, memory_config=memory_config)
        
        # Final linear
        output = ttnn.linear(attn_output, self.out_proj, memory_config=memory_config)
        
        return output

class TtBarkBlock:
    def __init__(self, device, parameters, base_address, config):
        self.device = device
        self.ln_1_weight = parameters[f"{base_address}.ln_1.weight"]
        self.ln_1_bias = parameters[f"{base_address}.ln_1.bias"]
        self.ln_2_weight = parameters[f"{base_address}.ln_2.weight"]
        self.ln_2_bias = parameters[f"{base_address}.ln_2.bias"]
        
        self.attn = TtBarkAttention(device, parameters, base_address, config)
        
        # MLP weights
        self.mlp_fc = parameters[f"{base_address}.mlp.fc.weight"]
        self.mlp_fc_bias = parameters[f"{base_address}.mlp.fc.bias"]
        self.mlp_out = parameters[f"{base_address}.mlp.out.weight"]
        self.mlp_out_bias = parameters[f"{base_address}.mlp.out.bias"]

    def forward(self, x):
        # LayerNorm 1 + Attention + Residual
        norm_x = ttnn.layer_norm(x, weight=self.ln_1_weight, bias=self.ln_1_bias)
        attn_out = self.attn.forward(norm_x)
        x = ttnn.add(x, attn_out)
        
        # LayerNorm 2 + MLP + Residual
        norm_x = ttnn.layer_norm(x, weight=self.ln_2_weight, bias=self.ln_2_bias)
        mlp_hidden = ttnn.linear(norm_x, self.mlp_fc, bias=self.mlp_fc_bias)
        mlp_hidden = ttnn.gelu(mlp_hidden)
        mlp_out = ttnn.linear(mlp_hidden, self.mlp_out, bias=self.mlp_out_bias)
        x = ttnn.add(x, mlp_out)
        
        return x
