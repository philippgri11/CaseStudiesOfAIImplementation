from datetime import datetime

import yaml

import wandb

with open('../params.yaml', 'r') as file:
    param = yaml.safe_load(file)
    dataset = param.get('dataset')

executionTime = datetime.now().strftime('%d/%m/%Y, %H:%M')

sweep_config = {
    'method': 'bayes',
    'name': 'CaseStudiesOfAIImplementation ' + executionTime
}

metric = {
    'name': 'mse_val',
    'goal': 'minimize'
}

sweep_config['metric'] = metric
features = {
    'loadCurveOne': {
        "monthlyCols": {
            "values": [[]]
        },
        "dailyCols": {
            "values": [[]]
        }
    },
    'loadCurveOneFull': {
        "monthlyCols": {
            "values": [[]]
        },
        "dailyCols": {
            "values": [[]]
        },
        "colums": {
            "values": [[]]
        }
    },
    'loadCurveTwo': {
        "monthlyCols": {
            "values": [['t1', 'r1'], ['t1'], ['r1'], ['t2'], ['r2'], ['t1', 't2'], ['t1', 'r1', 't2', 'r2']]
        },
        "dailyCols": {
            "values": [['t1', 'r1'], ['t1'], ['r1'], ['t2'], ['r2'], ['t1', 't2'], ['t1', 'r1', 't2', 'r2']]
        },
        "colums": {
            "values": [['t1', 'r1'], ['t1'], ['r1'], ['t2'], ['r2'], ['t1', 't2'], ['t1', 'r1', 't2', 'r2']]
        }
    },
    'loadCurveTwoFull': {
        "monthlyCols": {
            "values": [['t1', 'r1'], ['t1'], ['r1'], ['t2'], ['r2'], ['t1', 't2'], ['t1', 'r1', 't2', 'r2']]
        },
        "dailyCols": {
            "values": [['t1', 'r1'], ['t1'], ['r1'], ['t2'], ['r2'], ['t1', 't2'], ['t1', 'r1', 't2', 'r2']]
        },
        "colums": {
            "values": [['t1', 'r1'], ['t1'], ['r1'], ['t2'], ['r2'], ['t1', 't2'], ['t1', 'r1', 't2', 'r2']]
        }
    },
    'loadCurveThreeFull': {
        "monthlyCols": {
            "values": [['t1', 'r1'], ['t1'], ['r1']]
        },
        "dailyCols": {
            "values": [['t1', 'r1'], ['t1'], ['r1']]
        },
        "colums": {
            "values": [['t1', 'r1'], ['t1'], ['r1']]
        }
    }
}
parameters_dict_XGBoost = {
    "max_depth": {
        "values": [4, 6, 8, 10]
    },
    "colsample_bytree": {
        "min": 0.5,
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
        "max": 5.0
    },
    "reg_lambda": {
        "min": 0.0,
        "max": 5.0
    },
    "n_estimators": {
        "min": 1000,
        "max": 2000
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
    "shifts": {
        "min": 0,
        "max": 10
    },
    "loadLag":{
        "min": 0,
        "max": 500
    },
    "negShifts": {
        "min": -10,
        "max": 0
    },
    "enable_daytime_index": {
        "values": [True, False]
    },
    "monthlyCols": features[dataset]['monthlyCols'],
    "dailyCols": features[dataset]['dailyCols'],
    "colums": features[dataset]['colums'],
}
parameters_dict_LSTM = {
    "epochs": {
        "min": 3,
        "max": 30
    },
    # "epochs": {
    #     "values": [1]
    # },
    "batch_size": {
        "values": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    },
    "learning_rate": {
        "min": 0.0001,
        "max": 0.1
    },
    "dropout": {
        "min": 0.01,
        "max": 0.8
    },
    "shifts": {
        "min": 2,
        "max": 30
    },
    "enable_daytime_index": {
        "values": [True, False]
    },
    "monthlyCols": features[dataset]['monthlyCols'],
    "dailyCols": features[dataset]['dailyCols']
}


def getSweepIDLSTM():
    sweep_config['parameters'] = parameters_dict_LSTM
    sweep_id = wandb.sweep(sweep=sweep_config, project='CaseStudiesOfAIImplementation', entity='philippgrill')
    return sweep_id


def getSweepIDXGBoost():
    sweep_config['parameters'] = parameters_dict_XGBoost
    sweep_id = wandb.sweep(sweep=sweep_config, project='CaseStudiesOfAIImplementation', entity='philippgrill')
    return sweep_id
