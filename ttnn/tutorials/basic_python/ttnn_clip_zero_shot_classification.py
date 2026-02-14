# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import ttnn
import torch
from loguru import logger
import re
import math
from PIL import Image
from transformers import CLIPTokenizer, CLIPModel
import requests
from io import BytesIO
import time
import safetensors.torch
import os

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode


def main():
    def open_ttnn():
        """Initialize TT-NN device with specified L1 cache size."""
        global device
        device = ttnn.open_device(device_id=0, l1_small_size=8192)

    def close_ttnn():
        """Clean up and close the TT-NN device."""
        global device
        if device is not None:
            ttnn.close_device(device)

    def get_device():
        """Get the current TT-NN device handle."""
        global device
        return device

    def convert_model_to_ttnn(state_dict):
        """
        Convert a PyTorch model's state dictionary to TT-NN format.

        Args:
            state_dict: PyTorch model state dictionary containing weights and biases

        Returns:
            dict: State dictionary with tensors converted to TT-NN format
        """
        ttnn_state_dict = {}
        logger.info(f"Converting model to TT-NN format")

        # Convert each tensor in the state dictionary to TT-NN format
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                # Convert PyTorch tensors to TT-NN tensors
                ttnn_state_dict[key] = ttnn.from_torch(value, layout=ttnn.TILE_LAYOUT, device=get_device())
                ttnn_state_dict[key] = ttnn.typecast(ttnn_state_dict[key], dtype=ttnn.bfloat16)
            elif isinstance(value, torch.Size):
                # Convert PyTorch Size objects to TT-NN Size objects
                ttnn_state_dict[key] = ttnn.Size(value)

        return ttnn_state_dict

    class MultiHeadAttention:
        def __init__(self, state_dict, num_heads, attention_mask=None, prefix=""):
            self.attention_mask = attention_mask
            self.prefix = prefix
            self.num_heads = num_heads

            # Scale factor for attention scores: 1/sqrt(head_dim) for numerical stability
            self.scale = 1.0 / math.sqrt(num_heads)

            self.q_proj_weight = state_dict[f"{prefix}.q_proj.weight"]
            self.q_proj_bias = state_dict[f"{prefix}.q_proj.bias"]
            self.k_proj_weight = state_dict[f"{prefix}.k_proj.weight"]
            self.k_proj_bias = state_dict[f"{prefix}.k_proj.bias"]
            self.v_proj_weight = state_dict[f"{prefix}.v_proj.weight"]
            self.v_proj_bias = state_dict[f"{prefix}.v_proj.bias"]
            self.out_proj_weight = state_dict[f"{prefix}.out_proj.weight"]
            self.out_proj_bias = state_dict[f"{prefix}.out_proj.bias"]

        def forward(self, hidden_states):
            sequence_size, batch_size, hidden_size = hidden_states.shape

            head_size = hidden_size // self.num_heads

            # Compute Q, K, V projections from input hidden states
            # Each projection: [sequence_size, batch_size, hidden_size] -> [sequence_size, batch_size, hidden_size]
            q = ttnn.linear(hidden_states, self.q_proj_weight, bias=self.q_proj_bias, transpose_b=True)
            k = ttnn.linear(hidden_states, self.k_proj_weight, bias=self.k_proj_bias, transpose_b=True)
            v = ttnn.linear(hidden_states, self.v_proj_weight, bias=self.v_proj_bias, transpose_b=True)

            # Reshape for multi-head attention: split hidden_size into (num_heads * head_dim)
            # [sequence_size, batch_size, hidden_size] -> [sequence_size, batch_size * num_heads, head_dim]
            q = ttnn.reshape(q, (sequence_size, batch_size * self.num_heads, head_size))
            k = ttnn.reshape(k, (sequence_size, batch_size * self.num_heads, head_size))
            v = ttnn.reshape(v, (sequence_size, batch_size * self.num_heads, head_size))

            # Transpose to bring batch_size * num_heads dimension first for parallel attention computation
            # [sequence_size, batch_size * num_heads, head_dim] -> [batch_size * num_heads, sequence_size, head_size]
            q = ttnn.transpose(q, 0, 1)
            k = ttnn.transpose(k, 0, 1)
            v = ttnn.transpose(v, 0, 1)

            # Compute attention scores: Q @ K^T
            # [batch_size * num_heads, sequence_size, head_size] @ [batch_size * num_heads, head_size, sequence_size]
            #   -> [batch_size * num_heads, sequence_size, sequence_size]
            scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))
            # Scale scores by 1 / sqrt(head_size) to prevent softmax saturation
            scores = scores * self.scale

            # Apply attention mask if provided (for causal attention in text encoder)
            if self.attention_mask is not None:
                # Add mask to scores (mask contains -inf for positions that should be ignored)
                # Mask is broadcastable to [batch_size * num_heads, sequence_size, sequence_size]
                scores = scores + self.attention_mask

            # Apply softmax to get attention weights
            # numeric_stable=True uses the numerically stable softmax: softmax(x) = softmax(x - max(x))
            # This prevents overflow when computing exp(x) for large values
            attn_weights = ttnn.softmax(
                scores,
                dim=-1,
                numeric_stable=True,
            )

            # [batch_size * num_heads, sequence_size, sequence_size] @ [batch_size*heads, sequence_size, head_size]
            #   -> [batch_size * num_heads, sequence_size, head_size]
            attn_output = ttnn.matmul(attn_weights, v)

            # Transpose back to sequence-first format
            # [batch_size * num_heads, sequence_size, head_size] -> [sequence_size, batch_size * num_heads, head_size]
            attn_output = ttnn.transpose(attn_output, 0, 1)

            # Merge heads back into hidden dimension
            # [sequence_size, batch_size * num_heads, head_size] -> [sequence_size, batch_size, hidden_size]
            attn_output = ttnn.reshape(attn_output, (sequence_size, batch_size, hidden_size))

            # Apply output projection
            dense_out = ttnn.linear(
                attn_output,
                self.out_proj_weight,
                bias=self.out_proj_bias,
                transpose_b=True,
            )

            return dense_out

    class MultilayerPerceptron:
        def __init__(self, state_dict, attention_mask=None, prefix=""):
            self.prefix = prefix

            self.mlp_c_fc_weight = state_dict[f"{prefix}.fc1.weight"]
            self.mlp_c_fc_bias = state_dict[f"{prefix}.fc1.bias"]
            self.mlp_c_proj_weight = state_dict[f"{prefix}.fc2.weight"]
            self.mlp_c_proj_bias = state_dict[f"{prefix}.fc2.bias"]

        def forward(self, x):
            x = ttnn.linear(x, self.mlp_c_fc_weight, bias=self.mlp_c_fc_bias, transpose_b=True)
            x = ttnn.gelu(x)
            x = ttnn.linear(x, self.mlp_c_proj_weight, bias=self.mlp_c_proj_bias, transpose_b=True)
            return x

    class ResidualAttentionBlock:
        def __init__(self, state_dict, num_heads, attention_mask=None, prefix=""):
            self.prefix = prefix
            self.num_heads = num_heads

            self.attention = MultiHeadAttention(
                state_dict, num_heads=num_heads, attention_mask=attention_mask, prefix=f"{prefix}.self_attn"
            )
            self.mlp = MultilayerPerceptron(state_dict, prefix=f"{prefix}.mlp")

            self.layer_norm_1_weight = state_dict[f"{prefix}.layer_norm1.weight"]
            self.layer_norm_1_bias = state_dict[f"{prefix}.layer_norm1.bias"]
            self.layer_norm_2_weight = state_dict[f"{prefix}.layer_norm2.weight"]
            self.layer_norm_2_bias = state_dict[f"{prefix}.layer_norm2.bias"]

        def forward(self, x):
            # LayerNorm
            residual = x
            x = ttnn.layer_norm(x, weight=self.layer_norm_1_weight, bias=self.layer_norm_1_bias)

            # Multihead attention / Self-Attention
            # This must be equal to nn.MultiheadAttention(d_model, n_head)(x, x, x, need_weights=False, attn_mask=self.attn_mask)
            x = residual + self.attention.forward(x)

            # LayerNorm
            x_post_layer_norm = ttnn.layer_norm(x, weight=self.layer_norm_2_weight, bias=self.layer_norm_2_bias)

            # Multi-Layer Perceptron
            x = x + self.mlp.forward(x_post_layer_norm)

            return x

    class Transformer:
        def __init__(self, state_dict, num_layers, num_heads, attention_mask=None, prefix=""):
            """
            Initialize a generic Transformer that can be used for both text and vision encoding.

            Args:
                state_dict: Model weights dictionary
                num_heads: Number of attention heads
                attention_mask: Attention mask for causal attention (used for text, None for vision)
                prefix: Prefix for layer names in state_dict (e.g., "text_model.encoder" or "vision_model.encoder")
            """
            self.prefix = prefix
            self.layers = []

            # Initialize each transformer layer with converted weights
            self.layers = [
                ResidualAttentionBlock(
                    state_dict, attention_mask=attention_mask, num_heads=num_heads, prefix=f"{prefix}.layers.{i}"
                )
                for i in range(0, num_layers)
            ]

        def forward(self, x):
            for i in range(len(self.layers)):
                layer = self.layers[i]
                x = layer.forward(x)

            return x

    class VisionTransformer:
        def __init__(self, state_dict, num_vision_layers):
            self.output_dim = 0

            conv2_state_dict_name = "vision_model.embeddings.patch_embedding.weight"
            self.vision_width = state_dict[conv2_state_dict_name].shape[0]
            self.patch_size = state_dict[conv2_state_dict_name].shape[-1]
            self.vision_heads = self.vision_width // 64

            self.class_embedding = state_dict["vision_model.embeddings.class_embedding"]
            self.positional_embedding = state_dict["vision_model.embeddings.position_embedding.weight"]
            self.proj = state_dict["visual_projection.weight"]

            # Weights preparation for convolution (ttnn.conv2d) must be done on host (CPU)
            # To that end, we move convolution weights from device to host and perform its
            # layout to Row-Major, which is the preferred layout for TT-NN convolution kernels.
            self.conv1_weights = ttnn.from_device(state_dict[conv2_state_dict_name])
            self.conv1_weights = ttnn.to_dtype(self.conv1_weights, dtype=ttnn.bfloat16)
            self.conv1_weights = ttnn.to_layout(self.conv1_weights, layout=ttnn.ROW_MAJOR_LAYOUT)

            # Layer normalization applied before transformer layers
            self.ln_pre_weights = state_dict["vision_model.pre_layrnorm.weight"]
            self.ln_pre_bias = state_dict["vision_model.pre_layrnorm.bias"]

            # Layer normalization applied after transformer layers (to class token)
            self.ln_post_weights = state_dict["vision_model.post_layernorm.weight"]
            self.ln_post_bias = state_dict["vision_model.post_layernorm.bias"]

            self.transformer = Transformer(
                state_dict,
                num_layers=num_vision_layers,
                num_heads=self.vision_heads,
                attention_mask=None,
                prefix="vision_model.encoder",
            )

        def forward(self, x):
            (batch_size, in_channels, height, width) = x.shape

            # === Important: TT-NN conv2d differs from PyTorch in tensor layout ===
            #
            # PyTorch Conv2d expects: (N, C_in, H, W) - channels-first "Struct of Arrays"
            # TT-NN conv2d expects:   (N, H, W, C_in) - channels-last "Array of Structs"
            #
            # PyTorch Conv2d output:  (N, C_out, H_out, W_out) - 4D tensor
            # TT-NN conv2d output:    (1, 1, N*H_out*W_out, C_out) - flattened 4D tensor
            #
            # This is why we need to permute and reshape before and after convolution.

            # Step 1: Rearrange from channels-first to channels-last layout
            # [batch_size, in_channels, height, width] -> [batch_size, height, width, in_channels]
            x = ttnn.permute(x, [0, 2, 3, 1])

            # Step 2: Convert to row-major layout (required by ttnn.conv2d)
            # TT-NN convolution kernels are optimized for row-major data access
            x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)

            # Output channels for patch embedding (standard ViT uses 768)
            out_channels = 768

            self.conv1_weights = ttnn.prepare_conv_weights(
                weight_tensor=self.conv1_weights,
                input_memory_config=x.memory_config(),
                input_layout=x.layout,
                weights_format="OIHW",
                in_channels=in_channels,
                out_channels=out_channels,
                batch_size=batch_size,
                input_height=height,
                input_width=width,
                kernel_size=[self.patch_size, self.patch_size],
                stride=[self.patch_size, self.patch_size],
                padding=[0, 0],
                dilation=[1, 1],
                has_bias=False,
                groups=1,
                device=get_device(),
                input_dtype=x.dtype,
            )

            # Step 3: Apply patch embedding convolution
            # This converts the 2D image into a sequence of patch embeddings
            # For patch_size=32 and image 224x224: creates (224/32)^2 = 49 patches
            x = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=self.conv1_weights,
                in_channels=in_channels,  # Input channels (3 for RGB)
                out_channels=out_channels,  # Embedding dimension (768)
                batch_size=batch_size,
                input_height=height,
                input_width=width,
                kernel_size=(self.patch_size, self.patch_size),  # Patch size (e.g., 32x32)
                stride=(self.patch_size, self.patch_size),  # Non-overlapping patches: stride = kernel_size
                padding=(0, 0),  # No padding needed
                dilation=(1, 1),  # Standard convolution (no dilation)
                groups=0,  # Standard convolution (not grouped/depthwise)
                device=get_device(),
                return_weights_and_bias=False,  # We already have weights, don't return them
                return_output_dim=False,  # We know the output dimensions
            )

            # Step 4: Reshape convolution output from flattened to sequence format
            # Convert to tile layout for subsequent operations (TT-NN's optimized 2D tiled format)
            x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
            # Unflatten: [1, 1, N*H_out*W_out, C_out] -> [N, num_patches, embed_dim]
            # where num_patches = H_out * W_out = (224/32)^2 = 49

            x = ttnn.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))

            # Step 5: Prepare the [CLS] token (class embedding)
            # ViT prepends a learnable class token to the sequence of patch embeddings
            # Reshape class token: [embed_dim] -> [batch_size, 1, embed_dim]
            class_embedding = ttnn.reshape(self.class_embedding, (x.shape[0], 1, x.shape[-1]))

            # Step 6: Prepare tensors for concatenation
            # Move to DRAM memory (slower but more capacity than L1) for concatenation operation
            # Note: Concatenation currently requires DRAM memory; future optimizations may use L1 sharded memory
            x = ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            # Ensure class_embedding has matching memory configuration
            class_embedding = ttnn.to_memory_config(class_embedding, memory_config=x.memory_config())

            # Step 7: Prepend class token to patch sequence
            # [batch, 1, embed] + [batch, num_patches, embed] -> [batch, num_patches+1, embed]
            # This creates: [[CLS], patch_1, patch_2, ..., patch_49]
            x = ttnn.concat([class_embedding, x], dim=1, memory_config=None)

            # Step 8: Add positional embeddings
            # Positional embeddings encode the position of each token in the sequence
            # [1, num_patches+1, embed] -> broadcast and add to [batch, num_patches+1, embed]
            positional_embedding = ttnn.reshape(self.positional_embedding, (1, x.shape[1], x.shape[2]))
            x = x + positional_embedding

            # Step 9: Pre-transformer layer normalization
            x = ttnn.layer_norm(x, weight=self.ln_pre_weights, bias=self.ln_pre_bias)

            # Step 10: Permute to sequence-first format for transformer
            # Transformers typically process in sequence-first format
            # [batch_size, sequence_size, hidden_size] -> [sequence_size, batch_size, hidden_size]
            x = ttnn.permute(x, (1, 0, 2))

            # Step 11: Pass through transformer encoder layers
            x = self.transformer.forward(x)

            # Step 12: Permute back to batch-first format
            # [sequence_size, batch_size, hidden_size] -> [batch_size, sequence_size, hidden_size]
            x = ttnn.permute(x, (1, 0, 2))

            # Step 13: Extract [CLS] token and apply post-layer normalization
            # In ViT, the [CLS] token (first token) is used for classification
            # x[:, 0, :] extracts the [CLS] token: [batch, seq_len, embed] -> [batch, embed]
            x = ttnn.layer_norm(x[:, 0, :], weight=self.ln_post_weights, bias=self.ln_post_bias)

            # Step 14: Project to final embedding space (optional projection layer)
            # Maps from hidden dimension to the shared vision-language embedding space
            if self.proj is not None:
                x = ttnn.matmul(x, self.proj, transpose_b=True)

            return x

    class CLIP:
        def __init__(self, state_dict):
            self.token_embedding = state_dict["text_model.embeddings.token_embedding.weight"]
            self.positional_embedding = state_dict["text_model.embeddings.position_embedding.weight"]

            self.text_projection = state_dict["text_projection.weight"]
            self.context_length = self.positional_embedding.shape[0]
            self.vocab_size = self.token_embedding.shape[0]
            self.transformer_width = state_dict["text_model.final_layer_norm.weight"].shape[0]
            transformer_heads = self.transformer_width // 64

            self.ln_final_weights = state_dict["text_model.final_layer_norm.weight"]
            self.ln_final_bias = state_dict["text_model.final_layer_norm.bias"]

            self.logit_scale = state_dict["logit_scale"].item()

            num_vision_layers = 12  # Hardcoded value for CLIP-ViT-base-patch32
            self.visual = VisionTransformer(state_dict, num_vision_layers=num_vision_layers)

            num_text_layers = 12  # Hardcoded value for CLIP-ViT-base-patch32
            self.transformer = Transformer(
                state_dict,
                num_layers=num_text_layers,
                num_heads=transformer_heads,
                attention_mask=self.build_attention_mask(),
                prefix="text_model.encoder",
            )

        def build_attention_mask(self):
            """
            Build causal attention mask for text transformer.

            Causal masking ensures each token can only attend to itself and previous tokens,
            preventing the model from "cheating" by looking at future tokens. This is essential
            for autoregressive language modeling.

            Returns:
                Upper triangular mask [context_length, context_length] with -inf above diagonal
            """
            # Create a square mask filled with -inf (tokens cannot attend to masked positions)
            # Shape: [context_length, context_length]
            mask = ttnn.full(
                shape=[self.context_length, self.context_length],
                fill_value=float("-inf"),
                dtype=ttnn.bfloat16,
                device=get_device(),
                layout=ttnn.TILE_LAYOUT,
            )
            # Keep only upper triangle (excluding diagonal): prevents attending to future tokens
            # diagonal=1 means the diagonal itself is not masked (tokens can attend to themselves)
            mask = ttnn.triu(mask, diagonal=1)
            return mask

        def encode_image(self, image):
            return self.visual.forward(image)

        def encode_text(self, tokens):
            """
            Encode text tokens into feature embeddings.

            Args:
                tokens: Tokenized text input [batch_size, context_length]

            Returns:
                Text embeddings in shared vision-language space [batch_size, embed_dim]
            """
            # Convert token IDs to uint32 for embedding lookup
            tokens = ttnn.typecast(tokens, dtype=ttnn.uint32)

            # Token embedding: [batch, seq_len] -> [batch, seq_len, embed_dim]
            x = ttnn.embedding(tokens, weight=self.token_embedding, dtype=ttnn.bfloat16)

            # Add learned positional embeddings
            # Positional embeddings help the model understand token order
            x = x + self.positional_embedding

            # Permute to sequence-first format for transformer
            # [batch, seq_len, embed] -> [seq_len, batch, embed]
            x = ttnn.permute(x, (1, 0, 2))

            # Pass through text transformer with causal masking
            # Causal masking prevents tokens from attending to future tokens
            x = self.transformer.forward(x)

            # Permute back to batch-first format
            # [seq_len, batch, embed] -> [batch, seq_len, embed]
            x = ttnn.permute(x, (1, 0, 2))

            # Final layer normalization
            x = ttnn.layer_norm(x, weight=self.ln_final_weights, bias=self.ln_final_bias)

            # Text Transformer is auto-regressive. This means that the last token has access to all the information in the sequence.
            # We can thus extract text features from the end-of-text (EOT) token position
            # Note: Using PyTorch for argmax since TT-NN doesn't support advanced indexing yet
            torch_tokens = ttnn.to_torch(tokens).to(torch.int64)
            torch_x = ttnn.to_torch(x)

            eot_indices = torch_tokens.argmax(dim=-1)  # [batch_size]
            torch_selected_features = torch_x[torch.arange(torch_x.shape[0]), eot_indices]  # [batch_size, embed_dim]

            # Move back to TT device and apply text projection
            # Projects from transformer hidden size to shared embedding space
            x = ttnn.from_torch(torch_selected_features, device=get_device(), layout=ttnn.TILE_LAYOUT)
            x = ttnn.matmul(x, self.text_projection, transpose_b=True)

            return x

        def forward(self, image, tokens):
            """
            Compute similarity scores between images and text descriptions.

            Args:
                image: Preprocessed image tensor [batch_size, channels, height, width]
                tokens: Tokenized text tensor [batch_size, context_length]

            Returns:
                logits_per_image: Image-to-text similarity scores [batch_size_image, batch_size_text]
                logits_per_text: Text-to-image similarity scores [batch_size_text, batch_size_image]
            """
            # Encode both modalities into the shared embedding space
            text_features = self.encode_text(tokens)  # [batch_text, embed_dim]
            image_features = self.encode_image(image)  # [batch_image, embed_dim]

            # Normalize features to unit vectors for cosine similarity
            # L2 norm: ||x||_2 = sqrt(sum(x_i^2))
            norm_image_features = ttnn.operations.moreh.norm(image_features, p=2.0, dim=1, keepdim=True)
            norm_text_features = ttnn.operations.moreh.norm(text_features, p=2.0, dim=1, keepdim=True)

            # Normalize: x / ||x|| -> unit vector
            image_features = ttnn.divide(image_features, norm_image_features)
            text_features = ttnn.divide(text_features, norm_text_features)

            # Compute cosine similarity scaled by learned temperature parameter
            # logit_scale is learned during training to control the sharpness of the distribution
            logit_scale = math.exp(self.logit_scale)

            # Compute similarity matrix: scaled dot product of normalized features
            # Result: [batch_image, embed] @ [embed, batch_text] = [batch_image, batch_text]
            logits_per_image = ttnn.matmul(logit_scale * image_features, text_features, transpose_b=True)
            # Transpose for text-to-image direction
            logits_per_text = ttnn.transpose(logits_per_image, 0, 1)

            return logits_per_image, logits_per_text

    def preprocess_image(image, model_resolution):
        def _convert_image_to_rgb(image):
            return image.convert("RGB")

        # Pre-process image on host with torch
        transform_fn = Compose(
            [
                Resize(model_resolution, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(model_resolution),
                _convert_image_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        return transform_fn(image)

    def download_image(url):
        """
        Download an image from a URL and return it as a PIL Image object.

        Args:
            url (str): The URL of the image to download

        Returns:
            PIL.Image: The downloaded image
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Convert the response content to a PIL Image
            image = Image.open(BytesIO(response.content))
            return image
        except requests.RequestException as e:
            raise Exception(f"Failed to download image from {url}: {e}")
        except Exception as e:
            raise Exception(f"Failed to process downloaded image: {e}")

    def download_model(model_name):
        clip_model_location = model_name  # By default, download from Hugging Face

        # If TTNN_TUTORIALS_MODELS_CLIP_PATH is set, use it as the cache directory to avoid requests to Hugging Face
        cache_dir = os.getenv("TTNN_TUTORIALS_MODELS_CLIP_PATH")
        if cache_dir is not None:
            clip_model_location = cache_dir

        # Load model weights (download if cache_dir was not set)
        model = CLIPModel.from_pretrained(clip_model_location)

        return model

    def download_tokenizer(tokenizer_name):
        clip_tokenizer_location = tokenizer_name  # By default, download from Hugging Face

        # If TTNN_TUTORIALS_MODELS_CLIP_PATH is set, use it as the cache directory to avoid requests to Hugging Face
        cache_dir = os.getenv("TTNN_TUTORIALS_MODELS_CLIP_PATH")
        if cache_dir is not None:
            clip_tokenizer_location = cache_dir

        tokenizer = CLIPTokenizer.from_pretrained(clip_tokenizer_location)

        return tokenizer

    # Initialize TT-NN device for hardware acceleration
    open_ttnn()

    # Load pre-trained CLIP model and convert weights to TT-NN format
    logger.info("Loading pre-trained CLIP model...")

    model = download_model("openai/clip-vit-base-patch32")
    state_dict = convert_model_to_ttnn(model.state_dict())

    tokenizer = download_tokenizer("openai/clip-vit-base-patch32")

    # Initialize our TT-NN CLIP implementation
    clip = CLIP(state_dict)

    # Download and preprocess test image
    logger.info("Downloading and preprocessing image...")
    image_url = "https://media.githubusercontent.com/media/tenstorrent/tutorial-assets/refs/heads/main/media/clip_tutorial/CLIP.png"
    image = download_image(image_url)

    # Preprocess image to model requirements (224x224, normalized with ImageNet statistics)
    # unsqueeze(0) adds batch dimension: [C, H, W] -> [1, C, H, W]
    image = preprocess_image(image, 224).unsqueeze(0).to("cpu")

    # Convert PyTorch image tensor to TT-NN tensor with bfloat16 precision
    # bfloat16 provides good balance between precision and memory/compute efficiency
    preferred_dtype = ttnn.bfloat16
    tt_image = ttnn.from_torch(image, device=get_device(), layout=ttnn.TILE_LAYOUT, dtype=preferred_dtype)

    # Define text prompts for zero-shot classification
    # The model will compute similarity between the image and each text description
    prompts = ["a diagram", "a dog", "a cat"]

    # Tokenize text prompts using CLIP's tokenizer
    logger.info("Tokenizing text prompts...")
    # padding="max_length" ensures all sequences are padded to context_length (77 tokens)
    # return_tensors="pt" returns PyTorch tensors
    tokenized_inputs = tokenizer(prompts, padding="max_length", max_length=clip.context_length, return_tensors="pt")
    tokens_pretrained_host = tokenized_inputs["input_ids"]  # Shape: [num_prompts, context_length]
    # Convert tokenized text to TT-NN tensors for device execution
    tokens_pretrained = ttnn.from_torch(tokens_pretrained_host, device=get_device(), layout=ttnn.TILE_LAYOUT)

    # Perform CLIP inference: compute similarity between image and text
    logger.info("Running CLIP inference...")
    time_start = time.time()
    logits_per_image, logits_per_text = clip.forward(tt_image, tokens_pretrained)
    time_end = time.time()
    logger.info(f"Time taken: {time_end - time_start:.3f} seconds")

    # Convert logits (similarity scores) to probabilities using softmax
    # Softmax normalizes scores so they sum to 1.0, representing a probability distribution
    probs = ttnn.softmax(logits_per_image, dim=-1)
    logger.info(f"==== Zero-shot Classification Results ====")
    logger.info(f"Image: {image_url.split('/')[-1]}")
    logger.info(f"Classification probabilities:")

    # Display results sorted by probability (highest first)
    probs_torch = ttnn.to_torch(probs)
    results = [(prompt, probs_torch[0][i].item()) for i, prompt in enumerate(prompts)]
    results.sort(key=lambda x: x[1], reverse=True)

    for prompt, prob in results:
        logger.info(f"  '{prompt}': {prob:.4f} ({prob*100:.2f}%)")

    # Clean up resources
    close_ttnn()


if __name__ == "__main__":
    main()
