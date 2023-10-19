from datetime import datetime

import wandb
executionTime = datetime.now().strftime('%d/%m/%Y, %H:%M')

sweep_config = {
    'method': 'bayes',
    'name': 'CaseStudiesOfAIImplementation '+executionTime
}

metric = {
    'name': 'mse',
    'goal': 'minimize'
}

sweep_config['metric'] = metric

parameters_dict = {
        "max_depth": {
            "values": [4, 6, 8, 10]
        },
        "colsample_bytree": {
            "min": 0.1,
            "max": 1.0
        },
        "min_child_weight": {
            "min": 1,
            "max": 10
        },
        "subsample": {
            "min": 0.1,
            "max": 1.0
        },
        "reg_alpha": {
            "min": 0.0,
            "max": 1.0
        },
        "reg_lambda": {
            "min": 0.0,
            "max": 2.0
        },
        "n_estimators": {
            "min": 10,
            "max": 20
        },
        "tree_method": {
            "value": 'exact'
        },
        "eval_metric": {
            "value": 'mphe'
        },
        "test_size": {
            "value": 0.2
        },
        "colums": {
            "values": [['t1', 'r1']]
        },
        "shifts": {
            "min": 0,
            "max": 10
        },
        "negShifts": {
            "min": -10,
            "max": 0
        }
}

sweep_config['parameters'] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project='CaseStudiesOfAIImplementation', entity='philippgrill')
