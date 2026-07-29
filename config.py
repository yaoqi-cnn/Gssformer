from copy import deepcopy


DATASETS = {
    "village": {
        "class_names": [
            "Courtyard",
            "Vegetation",
            "Ancient road",
            "Ancient building",
        ],
        "train_areas": ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5"],
        "test_areas": ["Area_6"],
        "block_size": 20.0,
        "stride": 10.0,
    },
    "sensaturban": {
        "class_names": [
            "Ground",
            "High vegetation",
            "Building",
            "Wall",
            "Bridge",
            "Parking",
            "Rail",
            "Traffic road",
            "Street furniture",
            "Cars",
            "Footpath",
            "Bikes",
            "Water",
        ],
        "train_areas": [
            "birmingham_block_0",
            "birmingham_block_1",
            "birmingham_block_2",
            "birmingham_block_3",
            "birmingham_block_6",
            "birmingham_block_8",
            "birmingham_block_9",
            "cambridge_block_6",
            "cambridge_block_9",
            "cambridge_block_15",
            "cambridge_block_16",
            "cambridge_block_21",
            "cambridge_block_27",
            "cambridge_block_28",
        ],
        "test_areas": [
            "birmingham_block_4",
            "birmingham_block_5",
            "cambridge_block_8",
            "cambridge_block_22",
        ],
        "block_size": 40.0,
        "stride": 10.0,
    },
}


BASE_MODEL = {
    "in_channels": 10,
    "architecture": "gssformer",
    "enable_flash": True,
    "grid_size": 0.01,
    "model_size": "base",
    "use_gssm": True,
    "use_geometry_modulation": True,
    "use_boundary_modulation": True,
    "fusion_mode": "class_aware",
    "num_scan_orders": 2,
    "bidirectional": True,
    "random_scan_orders": True,

    "residual_gate_init": 0.0,
}


MODEL_VARIANTS = {
    "gssformer": {},
    "ptv3_c": {
        "use_gssm": False,
        "fusion_mode": "none",
    },
    "ptv3_c_camp": {
        "use_gssm": False,
        "fusion_mode": "class_aware",
    },
    "ptv3_c_gssm": {
        "use_gssm": True,
        "fusion_mode": "none",
    },
    "ssm_standard": {
        "use_gssm": True,
        "use_geometry_modulation": False,
        "use_boundary_modulation": False,
        "fusion_mode": "class_aware",
    },
    "ssm_geometry": {
        "use_gssm": True,
        "use_geometry_modulation": True,
        "use_boundary_modulation": False,
        "fusion_mode": "class_aware",
    },
    "ssm_boundary": {
        "use_gssm": True,
        "use_geometry_modulation": False,
        "use_boundary_modulation": True,
        "fusion_mode": "class_aware",
    },
    "gssm_main_head": {
        "use_gssm": True,
        "fusion_mode": "none",
    },
    "gssm_shared_fusion": {
        "use_gssm": True,
        "fusion_mode": "shared",
    }

}


TRAINING = {
    "epochs": 300,
    "num_points": -1,
    "gradient_accumulation": 4,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "warmup_epochs": 5,
    "minimum_lr_factor": 0.01,
    "boundary_loss_weight": 0.1,
    "boundary_pos_weight_max": 20.0,
    "gradient_clip": 0.5,
    "scale_min": 0.95,
    "scale_max": 1.05,
    "jitter_std": 0.005,
    "num_workers": 6,
    "minimum_block_points": 2,
    "block_sampling_attempts": 32,
}


EVALUATION = {
    "num_runs": 3,
    "save_predictions": True,
}


def available_variants():
    return tuple(MODEL_VARIANTS.keys())


def get_experiment(name, data_root, variant="gssformer"):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    if variant not in MODEL_VARIANTS:
        raise ValueError(
            f"Unknown model variant: {variant}. "
            f"Available variants: {', '.join(available_variants())}"
        )

    experiment = deepcopy(DATASETS[name])
    model = deepcopy(BASE_MODEL)
    model.update(deepcopy(MODEL_VARIANTS[variant]))


    if not model["use_gssm"]:
        model["use_geometry_modulation"] = False
        model["use_boundary_modulation"] = False
    if model["fusion_mode"] not in {"none", "shared", "class_aware"}:
        raise ValueError(f"Unsupported fusion_mode: {model['fusion_mode']}")
    if int(model["num_scan_orders"]) < 1:
        raise ValueError("num_scan_orders must be at least 1")

    experiment["name"] = name
    experiment["variant"] = variant
    experiment["data_root"] = str(data_root)
    experiment["model"] = model
    experiment["training"] = deepcopy(TRAINING)
    experiment["evaluation"] = deepcopy(EVALUATION)
    return experiment
