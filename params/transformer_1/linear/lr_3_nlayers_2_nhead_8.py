params = {
    # general params
    "glue_dataset_name": "cola",
    "model_name": "./pretrained_models/roberta-base-with-classifier_cola_1757839415/checkpoint-21440",
    "peft_model_name": "./pretrained_models/task_1_baseline_1760028520/checkpoint-5360",
    "use_hypernet": True,

    "layers_to_freeze": [
        "embeddings",
        "layer.0.",
        "layer.1.",
        "layer.2.",
        "layer.3",
        "layer.4",
        "layer.5",
        "layer.6",
        "layer.7",
        "layer.8",
        "layer.9",
        "layer.10",
        "model.classifier"
    ],

    # lora params
    "use_peft": True,
    "lora_r": 1,
    "lora_alpha": 16,
    "target_modules": ["query", "value"],
    "layers_to_transform": [11],
    "layers_pattern": "encoder.layer",

    # hypernet params
    "hypernet_use_transformer": True,
    "hypernet_transformer_nhead": 8,       # ! Important
    "hypernet_transformer_num_layers": 2,  # ! Important
    # "replace", "add", "multiply"
    "hypernet_noise_type_A": "add",
    "hypernet_noise_type_B": "add",
    "hypernet_reduce_noise_alpha": False,
    "hypernet_noise_alpha": 0.0,             # TODO Mid
    "hypernet_use_batches": True,
    "hypernet_hidden_dim": 256,            # TODO Mid
    "hypernet_embeddings_dim": 64,
    "layers_to_use_hypernet": [11],
    "hypernet_use_on_value_matrix": True, 
    "hypernet_with_embedding_input_only": False,
    "hypernet_use_fixed_A": True, 
    "hypernet_large_model": False,

    # in most cases this param is 1, it says how many time in a row we should run forward pass on single batch
    "forward_pass_reps": 1,

    # transformers trainer args
    "output_dir": f"./pretrained_models/hypernet",
    "eval_strategy": "epoch",
    "eval_steps": 5,
    "save_strategy": "steps",
    "save_steps": 1000000000,
    "logging_strategy": "epoch",
    "logging_steps": 50,
    "learning_rate": 1e-3,              # ! Important
    "weight_decay": 0.1,                # TODO Mid
    "per_device_train_batch_size": 16,  # TODO Mid
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 20,
    "metric_for_best_model": "matthews_correlation",
    "warmup_ratio": 0.06,               # TODO Mid
    "lr_scheduler_type": "linear",      # ! Important
    "optim": "adamw_torch",
    "disable_tqdm": True,

    # filenames and else
    "results_dir": "./results/transformer_1",
    "num_runs": 1, 
    "seed": 11,
}
