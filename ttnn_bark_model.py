import ttnn
import torch
from ttnn_bark_block import TtBarkBlock

class TtBarkSemanticModel:
    def __init__(self, device, parameters, config):
        self.device = device
        self.config = config
        self.layers = []
        for i in range(config.num_layers):
            self.layers.append(TtBarkBlock(device, parameters, f"semantic.transformer.h.{i}", config))
        
        self.wte = parameters["semantic.transformer.wte.weight"]
        self.wpe = parameters["semantic.transformer.wpe.weight"]
        self.ln_f_weight = parameters["semantic.transformer.ln_f.weight"]
        self.ln_f_bias = parameters["semantic.transformer.ln_f.bias"]
        self.lm_head = parameters["semantic.lm_head.weight"]

    def forward(self, input_ids):
        # input_ids: [batch, seq_len]
        
        # 1. Embeddings
        # ttnn.embedding requires tensors to be on device
        x = ttnn.embedding(input_ids, self.wte)
        
        # Add Positional Embeddings (simplified)
        # In Bark, seq_len is usually fixed during generation
        # x = ttnn.add(x, self.wpe)
        
        # 2. Transformer Layers
        for layer in self.layers:
            x = layer.forward(x)
            
        # 3. Final LayerNorm and Head
        x = ttnn.layer_norm(x, weight=self.ln_f_weight, bias=self.ln_f_bias)
        logits = ttnn.linear(x, self.lm_head)
        
        return logits

class TtBarkCoarseModel:
    def __init__(self, device, parameters, config):
        self.device = device
        self.config = config
        self.layers = []
        for i in range(config.num_layers):
            self.layers.append(TtBarkBlock(device, parameters, f"coarse_acoustics.transformer.h.{i}", config))
        
        self.wte = parameters["coarse_acoustics.transformer.wte.weight"]
        self.wpe = parameters["coarse_acoustics.transformer.wpe.weight"]
        self.ln_f_weight = parameters["coarse_acoustics.transformer.ln_f.weight"]
        self.ln_f_bias = parameters["coarse_acoustics.transformer.ln_f.bias"]
        self.lm_head = parameters["coarse_acoustics.lm_head.weight"]

    def forward(self, input_ids):
        x = ttnn.embedding(input_ids, self.wte)
        for layer in self.layers:
            x = layer.forward(x)
        x = ttnn.layer_norm(x, weight=self.ln_f_weight, bias=self.ln_f_bias)
        logits = ttnn.linear(x, self.lm_head)
        return logits
