#! /usr/bin/env python3

#SBATCH --job-name=svensemble2_importances
#SBATCH --output=svensemble2_importances.out
#SBATCH --error=svensemble2_importances.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G

"""
WIP script to compute SHAP values for SVEnsemble2 models. Uses fixed files/variables for now
"""

import os
import sys
import pickle
import shap

# Set working directory and sys.path for local svensemble module
WORKDIR = "/vast/ai_projects/Papenfuss_SVEnsemble2"
os.chdir(WORKDIR)
if WORKDIR not in sys.path:
    sys.path.insert(0, WORKDIR)

import svensemble.importances as importances # noqa: E402


training_sets = {
    'gridss': ['gridss'],
    'manta_candidate': ['manta_candidate'],
    'manta_diploid': ['manta_diploid'],
    'delly': ['delly'],
    'gridss+manta_candidate': ['gridss', 'manta_candidate'],
    'gridss+manta_diploid': ['gridss', 'manta_diploid'],
    'gridss+manta_candidate+delly': ['gridss', 'manta_candidate', 'delly'],
    'gridss+manta_diploid+delly': ['gridss', 'manta_diploid', 'delly'],
    'delly+manta_candidate': ['delly', 'manta_candidate'],
    'delly+manta_diploid': ['delly', 'manta_diploid']
}

# Shap stuff
sample_name = 'NA12878_30x'
training_set_name = 'delly+manta_candidate'
model_name = 'unbiased'

sample_pkl_path = f'/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/{sample_name}/sv_calling/svensemble2/{sample_name}.{model_name}.pkl'
val_dict_pkl_path = f'/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/{sample_name}/sv_calling/svensemble2/{sample_name}.{model_name}.[\'all\'].all_lens.val_dict.pkl'


X, y = importances.load_svensemble2_val_Xy_from_sample_val_pickles(sample_pkl_path, val_dict_pkl_path, training_set_name, verbose=True)


sample = pickle.load(open(sample_pkl_path, 'rb'))
model = sample['pred_dicts'][training_set_name]['RF_fit']


# build a Permutation explainer and explain the model predictions on the given dataset
explainer = shap.explainers.TreeExplainer(model, X)
shap_values = explainer(X)

os.mkdir(f'/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/{sample_name}/sv_calling/svensemble2/shap_values', exist_ok=True)

pickle.dump(shap_values, open(f'/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/{sample_name}/sv_calling/svensemble2/shap_values/{sample_name}.{model_name}.{training_set_name}.[\'all\'].all_lens.shap_values.pkl', 'wb'))