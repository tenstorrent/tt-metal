# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import transformers

import ttnn
from ttnn.model_preprocessing import preprocess_model


def embedding(input_ids, *, parameters):
    output = ttnn.embedding(input_ids, weight=parameters.weight, layout=ttnn.TILE_LAYOUT)
    return output


def bert_embeddings(input_ids, token_type_ids, *, parameters):
    position_ids = torch.randint(0, 2, (1, 128))
    position_ids = ttnn.from_torch(position_ids, device=input_ids.device())

    word_embeddings = embedding(input_ids, parameters=parameters.word_embeddings)
    token_type_embeddings = embedding(token_type_ids, parameters=parameters.token_type_embeddings)
    position_embeddings = embedding(position_ids, parameters=parameters.position_embeddings)
    output = word_embeddings + token_type_embeddings
    output += position_embeddings
    output = ttnn.layer_norm(output, weight=parameters.LayerNorm.weight, bias=parameters.LayerNorm.bias)
    return output


def bert_attention(hidden_states, *, parameters):
    output = hidden_states @ parameters.self.query.weight
    output = output + parameters.self.query.bias
    return output


def bert_layer(hidden_states, *, parameters):
    output = bert_attention(hidden_states, parameters=parameters.attention)
    return output


def bert_encoder(hidden_states, *, parameters):
    output = bert_layer(hidden_states, parameters=parameters.layer[0])
    return output


def bert_model(input_ids, *, parameters):
    token_type_ids = torch.randint(0, 2, (1, 128))
    token_type_ids = ttnn.from_torch(token_type_ids, device=input_ids.device())
    embeddings = bert_embeddings(input_ids, token_type_ids, parameters=parameters.embeddings)
    output = bert_encoder(embeddings, parameters=parameters.encoder)
    return output


def main():
    model_name = "bert-base-uncased"
    config = transformers.BertConfig.from_pretrained(model_name)
    config.num_hidden_layers = 1
    model = transformers.BertModel.from_pretrained(model_name, config=config).eval()
    inputs = torch.randint(0, 1000, (1, 128))
    outputs = model(inputs)

    device_id = 0
    device = ttnn.open_device(device_id=device_id)

    parameters = preprocess_model(
        model_name=model_name,
        initialize_model=lambda: model,
        run_model=lambda model: model(inputs),
        reader_patterns_cache={},
        device=device,
    )

    with ttnn.tracer.trace():
        ttnn_inputs = ttnn.from_torch(inputs, device=device)
        ttnn_output = bert_model(ttnn_inputs, parameters=parameters)
        ttnn_output_as_torch = ttnn.to_torch(ttnn_output)

    ttnn.tracer.visualize(ttnn_output_as_torch, file_name="bert_model_trace.svg")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
