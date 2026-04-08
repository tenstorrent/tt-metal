# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import os
from pathlib import Path
import torch
import transformers
import ttnn
from ttnn.tracer import trace, visualize
from models.demos.bert.tt import ttnn_bert
from models.demos.bert.tt import ttnn_optimized_bert
from ttnn.model_preprocessing import preprocess_model_parameters


def main():
    os.environ["TTNN_CONFIG_OVERRIDES"] = '{"enable_fast_runtime_mode": false}'

    transformers.logging.set_verbosity_error()

    with trace():
        # Create a random integer tensor of shape (1, 64) with values between 0-99
        tensor = torch.randint(0, 100, (1, 64), dtype=torch.int32)
        # Apply exponential function element-wise
        # This demonstrates how mathematical operations are captured
        tensor = torch.exp(tensor)

    # Visualize the computational graph of the traced operations
    # This will show the flow from random tensor creation to exp operation
    visualize(tensor)

    with trace():
        # Create a PyTorch tensor with shape (4, 64)
        tensor = torch.randint(0, 100, (4, 64), dtype=torch.int32)

        # Convert PyTorch tensor to TT-NN format
        # This operation moves data to the TT-NN representation
        tensor = ttnn.from_torch(tensor)

        # Reshape the tensor from (4, 64) to (2, 4, 32)
        # This demonstrates how reshape operations are handled in TT-NN
        tensor = ttnn.reshape(tensor, (2, 4, 32))

        # Convert back to PyTorch for visualization
        tensor = ttnn.to_torch(tensor)

    # Visualize the graph showing PyTorch → TT-NN → reshape → PyTorch conversion
    visualize(tensor)

    def download_google_bert_model_and_config(
        model_name: str,
    ) -> tuple[transformers.models.bert.modeling_bert.BertSelfOutput, transformers.BertConfig]:
        model_location = model_name  # By default, download from Hugging Face

        # If TTNN_TUTORIALS_MODELS_TRACER_PATH is set, use it as the cache directory to avoid requests to Hugging Face
        cache_dir = os.getenv("TTNN_TUTORIALS_MODELS_TRACER_PATH")
        if cache_dir is not None:
            model_location = Path(cache_dir) / Path("config_google_bert.json")

        # Load model weights (download if cache_dir was not set)
        config = transformers.BertConfig.from_pretrained(model_location)
        model = transformers.models.bert.modeling_bert.BertSelfOutput(config).eval()

        return model, config

    def download_ttnn_bert_config(model_name: str) -> transformers.BertConfig:
        config_location = model_name  # By default, download from Hugging Face

        # If TTNN_TUTORIALS_MODELS_TRACER_PATH is set, use it as the cache directory to avoid requests to Hugging Face
        cache_dir = os.getenv("TTNN_TUTORIALS_MODELS_TRACER_PATH")
        if cache_dir is not None:
            config_location = Path(cache_dir) / Path("config_ttnn_bert.json")

        # Load config (download if cache_dir was not set)
        config = transformers.BertConfig.from_pretrained(config_location)

        return config

    def download_ttnn_bert_model(
        model_name: str, config: transformers.BertConfig
    ) -> transformers.BertForQuestionAnswering:
        model_location = model_name  # By default, download from Hugging Face

        # If TTNN_TUTORIALS_MODELS_TRACER_PATH is set, use it as the cache directory to avoid requests to Hugging Face
        cache_dir = os.getenv("TTNN_TUTORIALS_MODELS_TRACER_PATH")
        if cache_dir is not None:
            model_location = Path(cache_dir)

        # Load model weights (download if cache_dir was not set)
        model = transformers.BertForQuestionAnswering.from_pretrained(model_location, config=config).eval()

        return model

    model_name = "google/bert_uncased_L-4_H-256_A-4"
    model, config = download_google_bert_model_and_config(model_name)

    # Trace the BERT self-output layer operations
    with trace():
        # Create dummy inputs matching the expected dimensions
        # hidden_states: output from self-attention (batch=1, seq_len=64, hidden_size=256)
        hidden_states = torch.rand((1, 64, config.hidden_size))
        # input_tensor: residual connection input
        input_tensor = torch.rand((1, 64, config.hidden_size))

        # Run the layer forward pass
        output = model(hidden_states, input_tensor)

    # Visualize the BERT layer computation graph
    visualize(output)

    # Configure the dispatch core type based on the architecture
    # ETH cores are used on newer architectures, WORKER cores on Grayskull
    dispatch_core_type = ttnn.device.DispatchCoreType.ETH
    if os.environ.get("ARCH_NAME") and "grayskull" in os.environ.get("ARCH_NAME"):
        dispatch_core_type = ttnn.device.DispatchCoreType.WORKER

    # Open device with custom configuration
    # - l1_small_size: Set L1 memory allocation to 8KB for small tensors
    # - dispatch_core_config: Configure which cores handle dispatch operations
    device = ttnn.open_device(
        device_id=0, l1_small_size=8192, dispatch_core_config=ttnn.device.DispatchCoreConfig(dispatch_core_type)
    )

    def ttnn_bert(bert):
        """
        Run and trace a complete BERT model for question answering.

        Args:
            bert: Either ttnn_bert or ttnn_optimized_bert module
        """
        # Use a larger BERT model fine-tuned for question answering
        model_name = "phiyodr/bert-large-finetuned-squad2"
        config = download_ttnn_bert_config(model_name)

        # Limit to 1 layer for faster execution in this demo
        # Full BERT-large has 24 layers
        config.num_hidden_layers = 1

        # Set batch size and sequence length for input
        batch_size = 8
        sequence_size = 384  # Standard for question answering tasks

        # ===== Model Parameter Preprocessing =====
        # Convert model parameters to TT-NN format and optimize for device
        # This includes weight packing, layout conversion, and memory placement
        model = download_ttnn_bert_model(model_name, config)
        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            custom_preprocessor=bert.custom_preprocessor,
            device=device,
        )

        # ===== Trace BERT Inference =====
        with trace():
            # Create dummy input tensors
            # input_ids: Token IDs from vocabulary
            input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.uint32)

            # token_type_ids: Segment IDs (0 for question, 1 for context in QA)
            torch_token_type_ids = torch.ones((batch_size, sequence_size), dtype=torch.uint32)

            # position_ids: Position embeddings (usually just 0 to sequence_length-1)
            torch_position_ids = torch.ones((batch_size, sequence_size), dtype=torch.uint32)

            # attention_mask: Mask for padding tokens (only for optimized version)
            # Shape differs between regular and optimized BERT implementations
            torch_attention_mask = (
                torch.zeros(1, sequence_size, dtype=torch.bfloat16) if bert == ttnn_optimized_bert else None
            )

            # Preprocess inputs for TT-NN format
            # This converts PyTorch tensors to device tensors with appropriate layout
            ttnn_bert_inputs = bert.preprocess_inputs(
                input_ids,
                torch_token_type_ids,
                torch_position_ids,
                torch_attention_mask,
                device=device,
            )

            # Run BERT model for question answering
            # Returns start and end logits for answer span prediction
            output = bert.bert_for_question_answering(
                config,
                *ttnn_bert_inputs,
                parameters=parameters,
            )

            # Move output back from device to host for visualization
            output = ttnn.from_device(output)

        # Visualize the complete BERT computation graph
        return visualize(output)

    # Run the optimized BERT implementation
    # This version includes TT-NN specific optimizations for better performance
    ttnn_bert(ttnn_optimized_bert)

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
