model_configs = {
    'unbiased': {
        'name': 'unbiased',
        'training_params': {
            'function': 'unbiased',
            'custom_cols': []
        },
        'description': 'Unbiased random forest model for SV call probability estimation.'
    },
    'unbiased_GT': {
        'name': 'unbiased_GT',
        'training_params': {
            'function': 'unbiased',
            'custom_cols': ['CUSTOM_SVTYPE']
        },
        'description': 'Unbiased random forest model for SV call probability estimation with extra genotyping step for all input callsets.'
    }
}