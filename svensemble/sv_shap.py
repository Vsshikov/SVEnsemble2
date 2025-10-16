#! /usr/bin/env python3

#SBATCH --job-name=svensemble2_importances
#SBATCH --output=svensemble2_importances.out
#SBATCH --error=svensemble2_importances.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:P100:4

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

# Shap stuff
sample_name = 'NA12878_30x'
training_set_name = 'gridss+manta_diploid'
model_name = 'unbiased'

samples_pkl_path = '/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/model/saved/samples.pkl'
val_dict_pkl_path = '/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/model/saved/val_dict.pkl'


X, y = importances.load_svensemble2_val_Xy_from_pickles(samples_pkl_path, val_dict_pkl_path, sample_name, training_set_name, verbose=True)


samples = pickle.load(open(samples_pkl_path, 'rb'))
model = samples[sample_name]['pred_dicts'][training_set_name]['RF_fit']


# build a Permutation explainer and explain the model predictions on the given dataset
explainer = shap.explainers.TreeExplainer(model, X)
shap_values = explainer(X)

pickle.dump(shap_values, open('/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/model/temp/shap_values.pkl', 'wb'))