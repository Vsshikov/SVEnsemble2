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
    'unbiased_d31_nosvlen': {
        'name': 'unbiased_d31_nosvlen',
        'training_params': {
            'function': 'unbiased_nosvlen',
            'custom_cols': [],
            'model_kwargs': {'max_depth': 31},
            'fillna': None,
        },
        'bps_suffix': 'unbiased',
        'matched_bps_suffix': 'unbiased',
        'description': 'Unbiased random forest model for SV call probability estimation. Model with max depth 31 so GPUTreeExplainer can be used and all SVLEN columns removed from the feature matrix prior to training.'
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
    'unbiased_concat': {
        'name': 'unbiased_concat',
        'training_params': {
            'function': 'unbiased_concat',
            'concat': True,
            'custom_cols': [],
            'model_kwargs': {}
        },
        'bps_suffix': 'unbiased',
        'matched_bps_suffix': 'unbiased',
        'feature_matrix_suffix': 'unbiased',
        'description': 'Unbiased random forest model trained on a pooled (concat/cross) training set from one or more samples, supplied via --train_samples/--train_matched_bps.'
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
    'unbiased_xgb_nosvlen': {
        'name': 'unbiased_xgb',
        'training_params': {
            'function': 'unbiased_xgb_nosvlen',
            'custom_cols': [],
            'model_kwargs': {},
            'fillna': None,
        },
        'bps_suffix': 'unbiased',
        'matched_bps_suffix': 'unbiased',
        'description': 'Unbiased XGBoost model for SV call probability estimation with native NaN handling and all SVLEN columns removed from the feature matrix prior to training.'
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
    'unbiased_xgb_concat': {
        'name': 'unbiased_xgb_concat',
        'training_params': {
            'function': 'unbiased_xgb_concat',
            'concat': True,
            'custom_cols': [],
            'model_kwargs': {},
            'fillna': None,
        },
        'bps_suffix': 'unbiased',
        'matched_bps_suffix': 'unbiased',
        'feature_matrix_suffix': 'unbiased_xgb',
        'description': 'Unbiased XGBoost model trained on a pooled (concat/cross) training set from one or more samples, supplied via --train_samples/--train_matched_bps.'
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
    # Non-concat models pool a single sample implicitly; only concat-style
    # models (trained via matched_bps_sources, see models.train_model) opt in.
    if 'concat' not in model_config['training_params'].keys():
        model_config['training_params']['concat'] = False