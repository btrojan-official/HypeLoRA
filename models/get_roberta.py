import torch
import torch.nn as nn
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
)

from models.dynamic_lora_layer import DynamicLoRALayer
from models.hypernet import LoRAHyperNet, LoRAHyperNetTransformer


def get_baseline_roberta(
    model_name="roberta-base",
    peft_model_name="",
    use_peft=False,
    lora_r=1,
    lora_alpha=16,
    target_modules=["query", "value"],
    layers_to_transform=list(range(12)),
    layers_pattern="encoder.layer",
    layers_to_freeze=[],
):
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    if use_peft and peft_model_name == "":
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            layers_to_transform=layers_to_transform,
            layers_pattern=layers_pattern,
        )

        model = get_peft_model(model, peft_config=peft_config)

    elif use_peft:
        model = PeftModel.from_pretrained(model, peft_model_name)
        for name, param in model.named_parameters():
            if "lora_" in str(name):
                param.requires_grad = True


    for name, param in model.named_parameters():
        for layer in layers_to_freeze:
            if layer in str(name):
                param.requires_grad = False

    return model, tokenizer


def get_hypernet_on_last_layer_roberta(
    model_name="roberta-base",
    peft_model_name="",
    use_peft=True,
    lora_r=1,
    lora_alpha=16,
    hypernet_use_embedding=True,
    hypernet_use_transformer=True,
    hypernet_transformer_nhead=8,
    hypernet_transformer_num_layers=2,
    hypernet_use_batches=False,
    hypernet_layers=[11],
    hypernet_hidden_dim=16,
    hypernet_embeddings_dim=8,
    hypernet_noise_type_A="replace",
    hypernet_noise_type_B="replace",
    hypernet_noise_alpha=0.5,
    use_on_value_matrix=True,
    hypernet_with_embedding_input_only=False, #! in older versions of the code there is wihtout "_only"
    use_fixed_A=True,
    use_large_model=False,
    target_modules=["query", "value"],
    layers_to_transform=list(range(12)),
    layers_pattern="encoder.layer",
    layers_to_freeze=[],
):
    
    model = RobertaForSequenceClassification.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    base_hidden_size = model.config.hidden_size

    if use_peft and peft_model_name == "":
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            layers_to_transform=layers_to_transform,
            layers_pattern=layers_pattern,
        )

        model = get_peft_model(model, peft_config=peft_config)
    elif use_peft:
        model = PeftModel.from_pretrained(model, peft_model_name)

    if not hypernet_use_transformer:
        hypernet = LoRAHyperNet(
            base_hidden_size,
            hypernet_hidden_dim,
            lora_r,
            use_embedding=hypernet_use_embedding,
            num_of_embeddings=2 * len(hypernet_layers) if use_on_value_matrix else len(hypernet_layers),
            embedding_dim=hypernet_embeddings_dim,
            embedding_input_only=hypernet_with_embedding_input_only,
            large_model=use_large_model,
            use_batches=hypernet_use_batches,
            use_fixed_A=use_fixed_A
        )
    else:
        hypernet = LoRAHyperNetTransformer(
            base_hidden_size,
            hypernet_hidden_dim,
            lora_r,
            use_embedding=hypernet_use_embedding,
            num_of_embeddings=2 * len(hypernet_layers) if use_on_value_matrix else len(hypernet_layers),
            embedding_dim=hypernet_embeddings_dim,
            embedding_input_only=hypernet_with_embedding_input_only,
            nhead=hypernet_transformer_nhead,
            num_layers=hypernet_transformer_num_layers,
            use_batches=hypernet_use_batches,
            use_fixed_A=use_fixed_A
        )

    dynamic_lora_layers = []

    for idx, layer_id in enumerate(hypernet_layers):

        adapter_name = "default"

        initial_A = model.roberta.encoder.layer[layer_id].attention.self.query.lora_A[adapter_name].weight.clone().reshape(base_hidden_size, -1)
        initial_B = model.roberta.encoder.layer[layer_id].attention.self.query.lora_B[adapter_name].weight.clone().reshape(-1, base_hidden_size)

        dynamic_lora_layers.append(DynamicLoRALayer(
            base_hidden_size, lora_r, hypernet, layer_id=idx, hypernet_use_batches=hypernet_use_batches, initial_A=initial_A, initial_B=initial_B, noise_type_A=hypernet_noise_type_A, noise_type_B=hypernet_noise_type_B, noise_alpha=hypernet_noise_alpha
        ))

        model.roberta.encoder.layer[layer_id].attention.self.query.lora_A[
            adapter_name
        ] = dynamic_lora_layers[-1]
        model.roberta.encoder.layer[layer_id].attention.self.query.lora_B[
            adapter_name
        ] = nn.Identity()

        if use_on_value_matrix:
            initial_A = model.roberta.encoder.layer[layer_id].attention.self.value.lora_A[adapter_name].weight.clone().reshape(base_hidden_size, -1)
            initial_B = model.roberta.encoder.layer[layer_id].attention.self.value.lora_B[adapter_name].weight.clone().reshape(-1, base_hidden_size)
            dynamic_lora_layers.append(DynamicLoRALayer(
                base_hidden_size, lora_r, hypernet, layer_id=idx + len(hypernet_layers), hypernet_use_batches=hypernet_use_batches, initial_A=initial_A, initial_B=initial_B, noise_type_A=hypernet_noise_type_A, noise_type_B=hypernet_noise_type_B, noise_alpha=hypernet_noise_alpha
            ))
            model.roberta.encoder.layer[layer_id].attention.self.value.lora_A[
                adapter_name
            ] = dynamic_lora_layers[-1]
            model.roberta.encoder.layer[layer_id].attention.self.value.lora_B[
                adapter_name
            ] = nn.Identity()


    # Freeze all lora_A matrices
    for name, param in model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False

    for name, param in model.named_parameters():
        for layer in layers_to_freeze:
            if layer in str(name):
                param.requires_grad = False

    return model, tokenizer, hypernet, dynamic_lora_layers
