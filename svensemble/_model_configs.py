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
    }
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