# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import os
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
        tensor = torch.randint(0, 100, (1, 64))
        tensor = torch.exp(tensor)
    visualize(tensor)

    with trace():
        tensor = torch.randint(0, 100, (4, 64))
        tensor = ttnn.from_torch(tensor)
        tensor = ttnn.reshape(tensor, (2, 4, 32))
        tensor = ttnn.to_torch(tensor)
    visualize(tensor)

    model_name = "google/bert_uncased_L-4_H-256_A-4"
    config = transformers.BertConfig.from_pretrained(model_name)
    model = transformers.models.bert.modeling_bert.BertSelfOutput(config).eval()

    with trace():
        hidden_states = torch.rand((1, 64, config.hidden_size))
        input_tensor = torch.rand((1, 64, config.hidden_size))
        hidden_states = model(hidden_states, input_tensor)
        output = ttnn.from_torch(hidden_states)
    visualize(output)

    dispatch_core_type = ttnn.device.DispatchCoreType.ETH
    if os.environ.get("ARCH_NAME") and "grayskull" in os.environ.get("ARCH_NAME"):
        dispatch_core_type = ttnn.device.DispatchCoreType.WORKER
    device = ttnn.open_device(
        device_id=0, l1_small_size=8192, dispatch_core_config=ttnn.device.DispatchCoreConfig(dispatch_core_type)
    )

    def ttnn_bert(bert):
        model_name = "phiyodr/bert-large-finetuned-squad2"

        config = transformers.BertConfig.from_pretrained(model_name)
        config.num_hidden_layers = 1

        batch_size = 8
        sequence_size = 384

        parameters = preprocess_model_parameters(
            initialize_model=lambda: transformers.BertForQuestionAnswering.from_pretrained(
                model_name, config=config
            ).eval(),
            custom_preprocessor=bert.custom_preprocessor,
            device=device,
        )

        with trace():
            input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
            torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
            torch_position_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
            torch_attention_mask = torch.zeros(1, sequence_size) if bert == ttnn_optimized_bert else None

            ttnn_bert_inputs = bert.preprocess_inputs(
                input_ids,
                torch_token_type_ids,
                torch_position_ids,
                torch_attention_mask,
                device=device,
            )

            output = bert.bert_for_question_answering(
                config,
                *ttnn_bert_inputs,
                parameters=parameters,
            )
            output = ttnn.from_device(output)

        return visualize(output)

    ttnn_bert(ttnn_optimized_bert)

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
