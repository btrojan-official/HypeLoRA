import time

import numpy as np
import torch
import wandb
from transformers import Trainer, TrainingArguments, enable_full_determinism

from data_loading.get_datasets import get_glue_dataset
from utils.metrics import compute_B_mean, compute_B_std, compute_ece
from utils.metrics_trainer_callback import SaveMetricsCallback
from utils.forward_pass_repetition_data_collator import SimpleGradientAccumulationTrainer
from utils.batch_generation_trainer import BatchedHypernetTrainer
from utils.alpha_callback import ReduceAlphaCallback
from models.get_roberta import get_baseline_roberta, get_hypernet_on_last_layer_roberta

import argparse
import importlib.util
import os
import random

def set_global_seed(seed: int):

    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    enable_full_determinism(seed)
    
    print(f"Global seed set to {seed}")

def run_experiment(params, id, device="cpu"):
    print(f"=== Run {id} ==============")

    experiment_id = str(int(time.time()))
    random_seed = params["seed"] + id if params["seed"] else id
    set_global_seed(random_seed)

    wandb.init(
        name=f"{params['results_filename']}_{id}_{experiment_id}",
        settings=wandb.Settings(_disable_stats=True),
        config={
            "fixed_A": params.get("hypernet_use_fixed_A", False),
            "reduce_noise": params.get("hypernet_reduce_noise_alpha", False),
            "lora_r": params.get("lora_r", 1),
            "layers_transformed": params.get("layers_to_transform", []),
        },
        tags=["hypernet" if params["use_hypernet"] else "baseline", params["glue_dataset_name"], f"run_{id}"]
    )

    if params["use_hypernet"]:
        model, tokenizer, hypernet, dynamic_lora_layers = get_hypernet_on_last_layer_roberta(
            model_name=params["model_name"],
            peft_model_name=params["peft_model_name"] if "peft_model_name" in params.keys() else "",
            use_peft=params["use_peft"],
            lora_r=params["lora_r"],
            lora_alpha=params["lora_alpha"],
            hypernet_use_embedding=params["hypernet_use_embedding"] if "hypernet_use_embedding" in params.keys() else True,
            hypernet_use_transformer=params["hypernet_use_transformer"],
            hypernet_transformer_nhead=params["hypernet_transformer_nhead"],
            hypernet_transformer_num_layers=params["hypernet_transformer_num_layers"],
            hypernet_use_batches=params["hypernet_use_batches"],
            hypernet_layers=params["layers_to_use_hypernet"] if "layers_to_use_hypernet" in params.keys() else [11],
            hypernet_hidden_dim=params["hypernet_hidden_dim"],
            hypernet_embeddings_dim=params["hypernet_embeddings_dim"],
            hypernet_noise_type_A=params["hypernet_noise_type_A"],
            hypernet_noise_type_B=params["hypernet_noise_type_B"],
            use_on_value_matrix=params["hypernet_use_on_value_matrix"],
            hypernet_with_embedding_input_only=params[
                "hypernet_with_embedding_input_only"
            ],
            hypernet_noise_alpha=params["hypernet_noise_alpha"],
            use_large_model=params["hypernet_large_model"],
            use_fixed_A=params["hypernet_use_fixed_A"],
            target_modules=params.get("target_modules", ["query", "value"]),
            layers_to_transform=params.get("layers_to_transform", list(range(12))),
            layers_pattern=params.get("layers_pattern", "encoder.layer"),
            layers_to_freeze=params.get("layers_to_freeze", []),
        )
    else:
        model, tokenizer = get_baseline_roberta(
            model_name=params["model_name"],
            peft_model_name=params["peft_model_name"] if "peft_model_name" in params.keys() else "",
            use_peft=params["use_peft"],
            lora_r=params["lora_r"],
            lora_alpha=params["lora_alpha"],
            target_modules=params.get("target_modules", ["query", "value"]),
            layers_to_transform=params.get("layers_to_transform", list(range(12))),
            layers_pattern=params.get("layers_pattern", "encoder.layer"),
            layers_to_freeze=params.get("layers_to_freeze", []),
        )
    encoded_dataset, metric = get_glue_dataset(
        params["glue_dataset_name"], tokenizer, truncation=True, max_length=512
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

        predictions = np.argmax(probs, axis=-1)
        results = metric.compute(predictions=predictions, references=labels)

        results["ece"] = compute_ece(probs, labels)
        if params["use_hypernet"]:
            results["hyper_B_std"] = compute_B_std(hypernet, device=device)
            results["hyper_B_mean"] = compute_B_mean(hypernet, device=device)

        if wandb.run is not None:
            wandb.log({f"eval/{k}": v for k, v in results.items()})

        return results

    training_args = TrainingArguments(
        output_dir=f"{params['output_dir']}_{params['glue_dataset_name']}_{experiment_id}",
        eval_strategy=params["eval_strategy"],
        eval_steps=params["eval_steps"] if params["eval_strategy"] == "steps" else None,
        save_strategy=params["save_strategy"],
        save_steps=params["save_steps"] if params["save_strategy"] == "steps" else None,
        logging_strategy=params["logging_strategy"],
        logging_steps=(
            params["logging_steps"] if params["logging_strategy"] == "steps" else None
        ),
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=params["per_device_train_batch_size"],
        gradient_accumulation_steps=params["gradient_accumulation_steps"],
        per_device_eval_batch_size=params["per_device_eval_batch_size"],
        num_train_epochs=params["num_train_epochs"],
        metric_for_best_model=params["metric_for_best_model"],
        warmup_ratio=params["warmup_ratio"],
        lr_scheduler_type=params["lr_scheduler_type"],
        optim=params["optim"],
        weight_decay=params["weight_decay"],
        disable_tqdm=params["disable_tqdm"],
        report_to=["wandb"],
    )

    callbacks=[
        SaveMetricsCallback(
            params["results_dir"],
            f"{params['results_filename']}_{params['glue_dataset_name']}_{experiment_id}.csv",
        ),
    ]
    if params["use_hypernet"] and params["hypernet_reduce_noise_alpha"]:
        num_of_training_steps = (len(encoded_dataset["train"]) // params["per_device_train_batch_size"]) * params["num_train_epochs"]
        callbacks.append(ReduceAlphaCallback(params["hypernet_noise_alpha"], dynamic_lora_layers, num_of_training_steps))

    if params["forward_pass_reps"] > 1:
        if params["hypernet_use_batches"]:
            print("WARNING!!!! The parameter ['hypernet_use_batches'] is not used!")
        trainer = SimpleGradientAccumulationTrainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["validation"],
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            accumulation_steps=params["forward_pass_reps"],
        )
    elif params["use_hypernet"] and params["hypernet_use_batches"]:
        trainer = BatchedHypernetTrainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["validation"],
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            hypernet=hypernet,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["validation"],
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

    # Printing trainable parameters
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(str(name))

    trainer.evaluate()
    trainer.train()

    wandb.finish()

def load_params_from_file(file_path):
    spec = importlib.util.spec_from_file_location("params_module", file_path)
    params_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params_module)
    return params_module.params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="Path to params .py file"
    )
    args = parser.parse_args()

    # Load params
    if not os.path.exists(args.params):
        raise FileNotFoundError(f"Params file {args.params} does not exist.")

    params = load_params_from_file(args.params)
    print("Loaded params:", params)

    params["results_filename"] = args.params.replace(".py", "").replace(".", "").replace("/", "_").replace("\\", "_")
    params["results_filename"] = params["results_filename"] if params["results_filename"][0] != "_" else params["results_filename"][1:]


    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Hypernet on: {params['glue_dataset_name']}")

    for i in range(params["num_runs"]):
        run_experiment(params, i, device=device)
    
