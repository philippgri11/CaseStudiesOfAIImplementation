from datetime import datetime

import yaml

import wandb

with open("../params.yaml", "r") as file:
    param = yaml.safe_load(file)
    dataset = param.get("dataset")

executionTime = datetime.now().strftime("%d/%m/%Y, %H:%M")

sweep_config = {
    "method": "bayes",
    "name": "CaseStudiesOfAIImplementation " + executionTime,
}

metric = {"name": "mse_val", "goal": "minimize"}

sweep_config["metric"] = metric
features = {
    "loadCurveOne": {"monthlyCols": {"values": [[]]}, "dailyCols": {"values": [[]]}},
    "loadCurveOneFull": {
        "monthlyCols": {"values": [[]]},
        "dailyCols": {"values": [[]]},
        "columns": {"values": [[]]},
    },
    "loadCurveTwo": {
        "monthlyCols": {
            "values": [
                ["t1", "r1"],
                ["t1"],
                ["r1"],
                ["t2"],
                ["r2"],
                ["t1", "t2"],
                ["t1", "r1", "t2", "r2"],
            ]
        },
        "dailyCols": {
            "values": [
                ["t1", "r1"],
                ["t1"],
                ["r1"],
                ["t2"],
                ["r2"],
                ["t1", "t2"],
                ["t1", "r1", "t2", "r2"],
            ]
        },
        "columns": {
            "values": [
                ["t1", "r1"],
                ["t1"],
                ["r1"],
                ["t2"],
                ["r2"],
                ["t1", "t2"],
                ["t1", "r1", "t2", "r2"],
            ]
        },
    },
    "loadCurveTwoFull": {
        "monthlyCols": {
            "values": [
                ["t1", "r1"],
                ["t1"],
                ["r1"],
                ["t2"],
                ["r2"],
                ["t1", "t2"],
                ["t1", "r1", "t2", "r2"],
            ]
        },
        "dailyCols": {
            "values": [
                ["t1", "r1"],
                ["t1"],
                ["r1"],
                ["t2"],
                ["r2"],
                ["t1", "t2"],
                ["t1", "r1", "t2", "r2"],
            ]
        },
        "columns": {
            "values": [
                ["t1", "r1"],
                ["t1"],
                ["r1"],
                ["t2"],
                ["r2"],
                ["t1", "t2"],
                ["t1", "r1", "t2", "r2"],
            ]
        },
    },
    "loadCurveThreeFull": {
        "monthlyCols": {"values": [["t1", "r1"], ["t1"], ["r1"]]},
        "dailyCols": {"values": [["t1", "r1"], ["t1"], ["r1"]]},
        "columns": {"values": [["t1", "r1"], ["t1"], ["r1"]]},
    },
}
parameters_dict_XGBoost = {
    "max_depth": {"values": [4, 6, 8, 10]},
    "colsample_bytree": {"min": 0.5, "max": 1.0},
    "min_child_weight": {"min": 1, "max": 10},
    "subsample": {"min": 0.1, "max": 1.0},
    "reg_alpha": {"min": 0.0, "max": 5.0},
    "reg_lambda": {"min": 0.0, "max": 5.0},
    "tree_method": {"value": "exact"},
    "eval_metric": {"value": "mphe"},
    "test_size": {"value": 0.2},
    "shifts": {"min": 0, "max": 10},
    "load_lag": {"min": 0, "max": 500},
    "neg_shifts": {"min": -10, "max": 0},
    "enable_daytime_index": {"values": [True, False]},
    "monthly_cols": features[dataset]["monthlyCols"],
    "daily_cols": features[dataset]["dailyCols"],
    "columns": features[dataset]["columns"],
}
parameters_dict_LSTM = {
    "epochs": {"min": 3, "max": 100},
    "batch_size": {"values": [1024,16384]},
    "learning_rate": {"min": 0.0001, "max": 0.1},
    "dropout": {"min": 0.01, "max": 0.8},
    "shifts": {"min": 2, "max": 30},
    "enable_daytime_index": {"values": [True, False]},
    "monthly_cols": features[dataset]["monthlyCols"],
    "daily_cols": features[dataset]["dailyCols"],
    "columns": features[dataset]["columns"],

}
parameters_dict_ARD = {
    "n_iter": {"values": [100, 300, 500, 1000, 1500]},
    "tol": {"values": [1e-3, 1e-4, 1e-5]},
    "alpha_1": {"values": [1e-6, 1e-7, 1e-8]},
    "alpha_2": {"values": [1e-6, 1e-7, 1e-8]},
    "lambda_1": {"values": [1e-6, 1e-7, 1e-8]},
    "lambda_2": {"values": [1e-6, 1e-7, 1e-8]},
    "compute_score": {"values": [True, False]},
    "test_size": {"value": 0.2},
    "shifts": {"min": 0, "max": 10},
    "load_lag": {"min": 0, "max": 500},
    "neg_shifts": {"min": -10, "max": 0},
    "enable_daytime_index": {"values": [True, False]},
    "monthly_cols": features[dataset]["monthlyCols"],
    "daily_cols": features[dataset]["dailyCols"],
    "columns": features[dataset]["columns"],
}
parameters_dict_k_neighbors = {
    "n_neighbors": {"values": [100, 300, 500]},
    "p":  {"values": [1, 2]},
    "algorithm": {"values": ["kd_tree", "ball_tree","auto"]},
    "weights": {"values": ["distance", "uniform"]},
    "leaf_size": {"min": 30, "max": 100},
    "test_size": {"value": 0.2},
    "shifts": {"min": 0, "max": 10},
    "load_lag": {"min": 0, "max": 500},
    "neg_shifts": {"min": -10, "max": 0},
    "enable_daytime_index": {"values": [True, False]},
    "monthly_cols": features[dataset]["monthlyCols"],
    "daily_cols": features[dataset]["dailyCols"],
    "columns": features[dataset]["columns"],
}

def getSweepIDLSTM():
    sweep_config["parameters"] = parameters_dict_LSTM
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="CaseStudiesOfAIImplementation",
        entity="philippgrill",
    )
    return sweep_id


def get_sweep_id_xg_boost():
    sweep_config["parameters"] = parameters_dict_XGBoost
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="CaseStudiesOfAIImplementation",
        entity="philippgrill",
    )
    return sweep_id

def get_sweep_ard():
    sweep_config["parameters"] = parameters_dict_ARD
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="CaseStudiesOfAIImplementation",
        entity="philippgrill",
    )
    return sweep_id

def get_sweep_k_neighbors():
    sweep_config["parameters"] = parameters_dict_k_neighbors
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="CaseStudiesOfAIImplementation",
        entity="philippgrill",
    )
    return sweep_id
