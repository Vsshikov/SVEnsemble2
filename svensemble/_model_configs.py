model_configs = {
    'unbiased': {
        'name': 'unbiased',
        'training_params': {
            'function': 'unbiased',
            'custom_cols': [],
            'model_kwargs': {}
        },
        'description': 'Unbiased random forest model for SV call probability estimation.'
    },
    'unbiased_GT': {
        'name': 'unbiased_GT',
        'training_params': {
            'function': 'unbiased',
            'custom_cols': ['SVTYPE'],
            'model_kwargs': {}
        },
        'description': 'Unbiased random forest model for SV call probability estimation with extra genotyping step for all input callsets.'
    },
    'unbiased_d31': {
        'name': 'unbiased_d31',
        'training_params': {
            'function': 'unbiased',
            'custom_cols': [],
            'model_kwargs': {'max_depth': 31}
        },
        'bps_suffix': 'unbiased', # This parameter allows us to use the same preprocessed breakpoints for different models.
        'matched_bps_suffix': 'unbiased',
        'description': 'Unbiased random forest model for SV call probability estimation. Model with max depth 31 so GPUTreeExplainer can be used.'
    },
    'unbiased_GT_d31': {
        'name': 'unbiased_GT_d31',
        'training_params': {
            'function': 'unbiased',
            'custom_cols': ['SVTYPE'],
            'model_kwargs': {'max_depth': 31}
        },
        'bps_suffix': 'unbiased_GT',
        'matched_bps_suffix': 'unbiased_GT',
        'description': 'Unbiased random forest model for SV call probability estimation with extra genotyping step for all input callsets. Model with max depth 31 so GPUTreeExplainer can be used.'
    },
    'unbiased_concat_NA12878': {
        'name': 'unbiased_concat_NA12878',
        'training_params': {
            'function': 'unbiased_concat',
            'function_params': {
                'samples': ['NA12878']
            },
            'custom_cols': [],
            'model_kwargs': {}
        },
        'bps_suffix': 'unbiased',
        'matched_bps_suffix': 'unbiased',
        'feature_matrix_suffix': 'unbiased',
        'description': 'Unbiased random forest model for SV call probability estimation with cross-training on NA12878.'
    },
    'unbiased_concat_HG002': {
        'name': 'unbiased_concat_HG002',
        'training_params': {
            'function': 'unbiased_concat',
            'function_params': {
                'samples': ['HG002']
            },
            'custom_cols': [],
            'model_kwargs': {}
        },
        'bps_suffix': 'unbiased',
        'matched_bps_suffix': 'unbiased',
        'feature_matrix_suffix': 'unbiased',
        'description': 'Unbiased random forest model for SV call probability estimation with cross-training on NA12878.'
    },
    'unbiased_concat_CHM13': {
        'name': 'unbiased_concat_CHM13',
        'training_params': {
            'function': 'unbiased_concat',
            'function_params': {
                'samples': ['CHM13']
            },
            'custom_cols': [],
            'model_kwargs': {}
        },
        'bps_suffix': 'unbiased',
        'matched_bps_suffix': 'unbiased',
        'feature_matrix_suffix': 'unbiased',
        'description': 'Unbiased random forest model for SV call probability estimation with cross-training on NA12878.'
    },
    'unbiased_concat_HG00514': {
        'name': 'unbiased_concat_HG00514',
        'training_params': {
            'function': 'unbiased_concat',
            'function_params': {
                'samples': ['HG00514']
            },
            'custom_cols': [],
            'model_kwargs': {}
        },
        'bps_suffix': 'unbiased',
        'matched_bps_suffix': 'unbiased',
        'feature_matrix_suffix': 'unbiased',
        'description': 'Unbiased random forest model for SV call probability estimation with cross-training on NA12878.'
    },
    'unbiased_concat_NA12878_HG002_CHM13_HG00514': {
        'name': 'unbiased_concat_NA12878_HG002_CHM13_HG00514',
        'training_params': {
            'function': 'unbiased_concat',
            'function_params': {
                'samples': ['NA12878', 'HG002', 'CHM13', 'HG00514']
            },
            'custom_cols': [],
            'model_kwargs': {}
        },
        'bps_suffix': 'unbiased',
        'matched_bps_suffix': 'unbiased',
        'feature_matrix_suffix': 'unbiased',
        'description': 'Unbiased random forest model for SV call probability estimation with cross-training on NA12878, HG002, CHM13 and HG00514.'
    },
    'unbiased_xgb': {
        'name': 'unbiased_xgb',
        'training_params': {
            'function': 'unbiased_xgb',
            'custom_cols': [],
            'model_kwargs': {},
            'fillna': None,
        },
        'bps_suffix': 'unbiased',
        'matched_bps_suffix': 'unbiased',
        'description': 'Unbiased XGBoost model for SV call probability estimation with native NaN handling.'
    },
    'unbiased_xgb_GT': {
        'name': 'unbiased_xgb_GT',
        'training_params': {
            'function': 'unbiased_xgb',
            'custom_cols': ['SVTYPE'],
            'model_kwargs': {},
            'fillna': None,
        },
        'bps_suffix': 'unbiased_GT',
        'matched_bps_suffix': 'unbiased_GT',
        'description': 'Unbiased XGBoost model for SV call probability estimation with extra genotyping step and native NaN handling.'
    },
    'unbiased_xgb_concat_NA12878': {
        'name': 'unbiased_xgb_concat_NA12878',
        'training_params': {
            'function': 'unbiased_xgb_concat',
            'function_params': {
                'samples': ['NA12878']
            },
            'custom_cols': [],
            'model_kwargs': {},
            'fillna': None,
        },
        'bps_suffix': 'unbiased',
        'matched_bps_suffix': 'unbiased',
        'feature_matrix_suffix': 'unbiased_xgb',
        'description': 'Unbiased XGBoost model for SV call probability estimation with cross-training on NA12878.'
    },
    'unbiased_xgb_concat_HG002': {
        'name': 'unbiased_xgb_concat_HG002',
        'training_params': {
            'function': 'unbiased_xgb_concat',
            'function_params': {
                'samples': ['HG002']
            },
            'custom_cols': [],
            'model_kwargs': {},
            'fillna': None,
        },
        'bps_suffix': 'unbiased',
        'matched_bps_suffix': 'unbiased',
        'feature_matrix_suffix': 'unbiased_xgb',
        'description': 'Unbiased XGBoost model for SV call probability estimation with cross-training on HG002.'
    },
    'unbiased_xgb_concat_CHM13': {
        'name': 'unbiased_xgb_concat_CHM13',
        'training_params': {
            'function': 'unbiased_xgb_concat',
            'function_params': {
                'samples': ['CHM13']
            },
            'custom_cols': [],
            'model_kwargs': {},
            'fillna': None,
        },
        'bps_suffix': 'unbiased',
        'matched_bps_suffix': 'unbiased',
        'feature_matrix_suffix': 'unbiased_xgb',
        'description': 'Unbiased XGBoost model for SV call probability estimation with cross-training on CHM13.'
    },
    'unbiased_xgb_concat_HG00514': {
        'name': 'unbiased_xgb_concat_HG00514',
        'training_params': {
            'function': 'unbiased_xgb_concat',
            'function_params': {
                'samples': ['HG00514']
            },
            'custom_cols': [],
            'model_kwargs': {},
            'fillna': None,
        },
        'bps_suffix': 'unbiased',
        'matched_bps_suffix': 'unbiased',
        'feature_matrix_suffix': 'unbiased_xgb',
        'description': 'Unbiased XGBoost model for SV call probability estimation with cross-training on HG00514.'
    },
    'unbiased_xgb_concat_NA12878_HG002_CHM13_HG00514': {
        'name': 'unbiased_xgb_concat_NA12878_HG002_CHM13_HG00514',
        'training_params': {
            'function': 'unbiased_xgb_concat',
            'function_params': {
                'samples': ['NA12878', 'HG002', 'CHM13', 'HG00514']
            },
            'custom_cols': [],
            'model_kwargs': {},
            'fillna': None,
        },
        'bps_suffix': 'unbiased',
        'matched_bps_suffix': 'unbiased',
        'feature_matrix_suffix': 'unbiased_xgb',
        'description': 'Unbiased XGBoost model for SV call probability estimation with cross-training on NA12878, HG002, CHM13 and HG00514.'
    },
}

# Set default values for all configs if some were not set above
for model_name, model_config in model_configs.items():
    # Set the bps_suffix if not set to the model name
    if 'bps_suffix' not in model_config.keys():
        model_config['bps_suffix'] = model_name
    if 'matched_bps_suffix' not in model_config.keys():
        model_config['matched_bps_suffix'] = model_name
    if 'model_pkl_suffix' not in model_config.keys():
        model_config['model_pkl_suffix'] = model_name
    if 'feature_matrix_suffix' not in model_config.keys():
        model_config['feature_matrix_suffix'] = model_name