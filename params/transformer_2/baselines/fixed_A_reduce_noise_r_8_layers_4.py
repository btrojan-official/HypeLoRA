params = {
    # general params
    "glue_dataset_name": "cola",
    "model_name": "roberta-base", # here you need to adjust the path to the trained model
    "use_hypernet": False, #hypernet set to False for baseline calculation

    "layers_to_freeze": [
    ],

    # lora params
    "use_peft": True,
    "lora_r": 8,
    "lora_alpha": 16,
    "target_modules": ["query", "value"],
    "layers_to_transform": [8, 9, 10, 11],
    "layers_pattern": "encoder.layer",

    # hypernet params
    "hypernet_use_transformer": True,
    "hypernet_transformer_nhead": 16,       # ! Important
    "hypernet_transformer_num_layers": 4,  # ! Important
    # "replace", "add", "multiply"
    "hypernet_noise_type_A": "add",
    "hypernet_noise_type_B": "add",
    "hypernet_reduce_noise_alpha": True,
    "hypernet_noise_alpha": 0.99,             # TODO Mid
    "hypernet_use_batches": True,
    "hypernet_hidden_dim": 256,            # TODO Mid
    "hypernet_embeddings_dim": 64,
    "layers_to_use_hypernet": [8, 9, 10, 11],
    "hypernet_use_on_value_matrix": True, 
    "hypernet_with_embedding_input_only": True,
    "hypernet_use_fixed_A": True, 
    "hypernet_large_model": True,

    # in most cases this param is 1, it says how many time in a row we should run forward pass on single batch
    "forward_pass_reps": 1,

    # transformers trainer args
    "output_dir": f"./pretrained_models/basic",
    "eval_strategy": "epoch",
    "eval_steps": 5,
    "save_strategy": "steps",
    "save_steps": 1000000000,
    "logging_strategy": "epoch",
    "logging_steps": 50,
    "learning_rate": 4e-4,              # ! Important
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
    "results_dir": "./results/basic",
    "num_runs": 1, 
    "seed": 11,
}
