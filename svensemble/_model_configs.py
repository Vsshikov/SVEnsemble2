# Model config presets.
#
# `model_configs` is the full preset registry: the three canonical BASE presets
# (see BASE_MODELS) plus retained legacy names kept for backwards compatibility.
# Every entry carries a `base` naming the canonical base preset it derives from;
# `base` is the human-readable prefix of the reconstructible model_id
# ({base}-{hash8}, see _config.model_id). Model identity is the content hash of
# the behavioural training_params, NOT this dict key or `name` (both are labels).
#
# This module does ONLY preset generation and never imports from `_config`; the
# resolver (resolve_model_config), the hasher, and model_id all live in _config.py.

import copy
from typing import TypedDict, NotRequired

# The canonical base presets that new models are built on (base preset +
# freeform overrides). See _config.resolve_model_config.
BASE_MODELS = ('unbiased_rf', 'unbiased_xgb', 'consensus', 'unbiased_rf_topol', 'unbiased_xgb_topol')

class TrainingParams(TypedDict, total=False):
    """Behavioural model parameters that determine model_id hash."""
    function: str
    concat: bool
    custom_cols: list[str]
    model_kwargs: dict
    fillna: int | float | None
    clpr: float | None
    add_topology: bool


class ResolvedModelConfig(TypedDict):
    """An model config with known base preset."""
    name: str
    base: str
    training_params: TrainingParams
    description: NotRequired[str]

model_configs: dict[str, ResolvedModelConfig] = {
    # ---------------------------------------------------------------------
    # Canonical base presets
    # ---------------------------------------------------------------------
    'unbiased_rf': {
        'name': 'unbiased_rf',
        'base': 'unbiased_rf',
        'training_params': {
            'function': 'unbiased',
            'custom_cols': [],
            'model_kwargs': {},
        },
        'description': 'Unbiased random forest model for SV call probability estimation.'
    },
    'unbiased_xgb': {
        'name': 'unbiased_xgb',
        'base': 'unbiased_xgb',
        'training_params': {
            'function': 'unbiased_xgb',
            'custom_cols': [],
            'model_kwargs': {},
            'fillna': None,
        },
        'description': 'Unbiased XGBoost model for SV call probability estimation with native NaN handling.'
    },
    'unbiased_rf_topol': {
        'name': 'unbiased_rf_topol',
        'base': 'unbiased_rf_topol',
        'training_params': {
            'function': 'unbiased',
            'custom_cols': [],
            'model_kwargs': {},
            'add_topology': True,
        },
        'description': 'Unbiased random forest model for SV call probability estimation.'
    },
    'unbiased_xgb_topol': {
        'name': 'unbiased_xgb_topol',
        'base': 'unbiased_xgb_topol',
        'training_params': {
            'function': 'unbiased_xgb',
            'custom_cols': [],
            'model_kwargs': {},
            'fillna': None,
            'add_topology': True,
        },
        'description': 'Unbiased XGBoost model for SV call probability estimation with native NaN handling.'
    },
    'consensus': {
        'name': 'consensus',
        'base': 'consensus',
        'training_params': {
            'function': 'consensus',
            'custom_cols': [],
            'model_kwargs': {'min_matches_mode': 'majority'},
        },
        'description': 'Consensus classifier for SV call probability estimation using majority voting.'
    },

    # ---------------------------------------------------------------------
    # Retained legacy presets (backwards compatibility). Suffixes removed;
    # `base` names the canonical base each derives from.
    # ---------------------------------------------------------------------
    'consensus_majority': {
        'name': 'consensus_majority',
        'base': 'consensus',
        'training_params': {
            'function': 'consensus',
            'custom_cols': [],
            'model_kwargs': {'min_matches_mode': 'majority'},
        },
        'description': 'Consensus classifier for SV call probability estimation using majority voting.'
    },
    'unbiased': {
        'name': 'unbiased',
        'base': 'unbiased_rf',
        'training_params': {
            'function': 'unbiased',
            'custom_cols': [],
            'model_kwargs': {}
        },
        'description': 'Unbiased random forest model for SV call probability estimation.'
    },
    'unbiased_GT': {
        'name': 'unbiased_GT',
        'base': 'unbiased_rf',
        'training_params': {
            'function': 'unbiased',
            'custom_cols': ['SVTYPE'],
            'model_kwargs': {}
        },
        'description': 'Unbiased random forest model for SV call probability estimation with extra genotyping step for all input callsets.'
    },
    'unbiased_d31': {
        'name': 'unbiased_d31',
        'base': 'unbiased_rf',
        'training_params': {
            'function': 'unbiased',
            'custom_cols': [],
            'model_kwargs': {'max_depth': 31}
        },
        'description': 'Unbiased random forest model for SV call probability estimation. Model with max depth 31 so GPUTreeExplainer can be used.'
    },
    'unbiased_d31_topol': {
        'name': 'unbiased_d31_topol',
        'base': 'unbiased_rf',
        'training_params': {
            'function': 'unbiased',
            'custom_cols': [],
            'model_kwargs': {'max_depth': 31},
            'add_topology': True,
        },
        'description': 'Unbiased random forest model for SV call probability estimation. Model with max depth 31 so GPUTreeExplainer can be used.'
    },
    'unbiased_d31_nosvlen': {
        'name': 'unbiased_d31_nosvlen',
        'base': 'unbiased_rf',
        'training_params': {
            'function': 'unbiased_nosvlen',
            'custom_cols': [],
            'model_kwargs': {'max_depth': 31},
            'fillna': None,
        },
        'description': 'Unbiased random forest model for SV call probability estimation. Model with max depth 31 so GPUTreeExplainer can be used and all SVLEN columns removed from the feature matrix prior to training.'
    },
    'unbiased_GT_d31': {
        'name': 'unbiased_GT_d31',
        'base': 'unbiased_rf',
        'training_params': {
            'function': 'unbiased',
            'custom_cols': ['SVTYPE'],
            'model_kwargs': {'max_depth': 31}
        },
        'description': 'Unbiased random forest model for SV call probability estimation with extra genotyping step for all input callsets. Model with max depth 31 so GPUTreeExplainer can be used.'
    },
    'unbiased_concat': {
        'name': 'unbiased_concat',
        'base': 'unbiased_rf',
        'training_params': {
            'function': 'unbiased_concat',
            'concat': True,
            'custom_cols': [],
            'model_kwargs': {}
        },
        'description': 'Unbiased random forest model trained on a pooled (concat/cross) training set from one or more samples, supplied via --train_samples/--train_matched_bps.'
    },
    'unbiased_xgb_nosvlen': {
        'name': 'unbiased_xgb_nosvlen',
        'base': 'unbiased_xgb',
        'training_params': {
            'function': 'unbiased_xgb_nosvlen',
            'custom_cols': [],
            'model_kwargs': {},
            'fillna': None,
        },
        'description': 'Unbiased XGBoost model for SV call probability estimation with native NaN handling and all SVLEN columns removed from the feature matrix prior to training.'
    },
    'unbiased_xgb_GT': {
        'name': 'unbiased_xgb_GT',
        'base': 'unbiased_xgb',
        'training_params': {
            'function': 'unbiased_xgb',
            'custom_cols': ['SVTYPE'],
            'model_kwargs': {},
            'fillna': None,
        },
        'description': 'Unbiased XGBoost model for SV call probability estimation with extra genotyping step and native NaN handling.'
    },
    'unbiased_xgb_concat': {
        'name': 'unbiased_xgb_concat',
        'base': 'unbiased_xgb',
        'training_params': {
            'function': 'unbiased_xgb_concat',
            'concat': True,
            'custom_cols': [],
            'model_kwargs': {},
            'fillna': None,
        },
        'description': 'Unbiased XGBoost model trained on a pooled (concat/cross) training set from one or more samples, supplied via --train_samples/--train_matched_bps.'
    },
}


def _enrich_model_config(model_config: ResolvedModelConfig) -> ResolvedModelConfig:
    """
    Enrich a model config with default values if some were not set above.

    Fills ``base`` (defaults to ``name``) and ``training_params['concat']``
    (defaults to False). Deep-copies the input so the source preset is never
    mutated. Pure preset generation only -- no imports from ``utils``.
    """
    mc = copy.deepcopy(model_config)
    mc.setdefault('base', mc['name'])
    mc['training_params'].setdefault('concat', False)
    return mc


# Fill defaults for all presets in place.
for _name, _cfg in model_configs.items():
    model_configs[_name] = _enrich_model_config(_cfg)
