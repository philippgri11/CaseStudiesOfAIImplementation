from datetime import datetime

import yaml

import wandb


def get_dataset():
    with open("../params.yaml", "r") as file:
        param = yaml.safe_load(file)
        dataset = param.get("dataset")
        return dataset


def get_sweep():
    executionTime = datetime.now().strftime("%d/%m/%Y, %H:%M")

    name = str(
        "CaseStudiesOfAIImplementationResults " + get_dataset() + " " + executionTime
    )
    sweep_config = {"method": "bayes", "name": name}

    metric = {"name": "mse_cv", "goal": "minimize"}

    sweep_config["metric"] = metric
    return sweep_config


features = {
    "loadCurveOne": {"monthly_cols": {"values": [[]]}, "daily_cols": {"values": [[]]}},
    "loadCurveOneFull": {
        "monthly_cols": {"values": [[]]},
        "daily_cols": {"values": [[]]},
        "columns": {"values": [[]]},
    },
    "loadCurveTwo": {
        "monthly_cols": {
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
        "daily_cols": {
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
        "monthly_cols": {
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
        "daily_cols": {
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
        "monthly_cols": {"values": [["t1", "r1"], ["t1"], ["r1"]]},
        "daily_cols": {"values": [["t1", "r1"], ["t1"], ["r1"]]},
        "columns": {"values": [["t1", "r1"], ["t1"], ["r1"]]},
    },
}


def get_parameters_dict_XGBoost():
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
        "monthly_cols": features[get_dataset()]["monthly_cols"],
        "daily_cols": features[get_dataset()]["daily_cols"],
        "columns": features[get_dataset()]["columns"],
    }
    return parameters_dict_XGBoost


def get_parameters_dict_LSTM():
    parameters_dict_LSTM = {
        "epochs": {"min": 20, "max": 300},
        "batch_size": {"values": [1024, 16384]},
        "learning_rate": {"min": 0.0001, "max": 0.1},
        "dropout": {"min": 0.01, "max": 0.8},
        "shifts": {"min": 2, "max": 30},
        "enable_daytime_index": {"values": [True, False]},
        "monthly_cols": features[get_dataset()]["monthly_cols"],
        "daily_cols": features[get_dataset()]["daily_cols"],
        "columns": features[get_dataset()]["columns"],
    }
    return parameters_dict_LSTM


def get_parameters_dict_ARD():
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
        "monthly_cols": features[get_dataset()]["monthly_cols"],
        "daily_cols": features[get_dataset()]["daily_cols"],
        "columns": features[get_dataset()]["columns"],
    }
    return parameters_dict_ARD


def get_parameters_dict_k_neighbors():
    parameters_dict_k_neighbors = {
        "n_neighbors": {"values": [100, 300, 500]},
        "p": {"values": [1, 2]},
        "algorithm": {"values": ["kd_tree", "ball_tree", "auto"]},
        "weights": {"values": ["distance", "uniform"]},
        "leaf_size": {"min": 30, "max": 100},
        "test_size": {"value": 0.2},
        "shifts": {"min": 0, "max": 10},
        "load_lag": {"min": 0, "max": 500},
        "neg_shifts": {"min": -10, "max": 0},
        "enable_daytime_index": {"values": [True, False]},
        "monthly_cols": features[get_dataset()]["monthly_cols"],
        "daily_cols": features[get_dataset()]["daily_cols"],
        "columns": features[get_dataset()]["columns"],
    }
    return parameters_dict_k_neighbors


def getSweepIDLSTM():
    sweep_config = get_sweep()
    sweep_config["parameters"] = get_parameters_dict_LSTM()
    sweep_config["name"] = sweep_config["name"] + " LSTM"
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="CaseStudiesOfAIImplementationResults",
        entity="philippgrill",
    )
    return sweep_id


def get_sweep_id_xg_boost():
    sweep_config = get_sweep()
    sweep_config["parameters"] = get_parameters_dict_XGBoost()
    sweep_config["name"] = sweep_config["name"] + " xg_boost"
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="CaseStudiesOfAIImplementationResults",
        entity="philippgrill",
    )
    return sweep_id


def get_sweep_ard():
    sweep_config = get_sweep()
    sweep_config["parameters"] = get_parameters_dict_ARD()
    sweep_config["name"] = sweep_config["name"] + " ard"
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="CaseStudiesOfAIImplementationResults",
        entity="philippgrill",
    )
    return sweep_id


def get_sweep_k_neighbors():
    sweep_config = get_sweep()
    sweep_config["parameters"] = get_parameters_dict_k_neighbors()
    sweep_config["name"] = sweep_config["name"] + " k_neighbors"
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="CaseStudiesOfAIImplementationResults",
        entity="philippgrill",
    )
    return sweep_id
