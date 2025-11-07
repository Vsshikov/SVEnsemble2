#! /usr/bin/env python3

#SBATCH --job-name=svensemble2_importances
#SBATCH --output=svensemble2_importances.out
#SBATCH --error=svensemble2_importances.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G

"""
WIP script to compute SHAP values for SVEnsemble2 models.
"""

import os
import sys
import pickle
import shap
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Compute SHAP values for SVEnsemble2 models.")
parser.add_argument("--training_set_name", type=str, default="delly+manta_candidate", help="Name of the training set to use")
parser.add_argument("--sample_name", type=str, default="NA12878_30x", help="Name of the sample to use")
parser.add_argument("--model_name", type=str, default="unbiased", help="Name of the model to use")
args = parser.parse_args()

training_set_name = args.training_set_name
sample_name = args.sample_name
model_name = args.model_name


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
# sample_name = 'NA12878_30x'
# training_set_name = 'delly+manta_candidate'
# model_name = 'unbiased'

sample_pkl_path = f'/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/{sample_name}/sv_calling/svensemble2/{sample_name}.{model_name}.bench_preprocessed.pkl'
# val_dict_pkl_path = f'/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/{sample_name}/sv_calling/svensemble2/{sample_name}.{model_name}.all.50:inf.val_dict.pkl'


X, y = importances.load_svensemble2_val_Xy_from_sample_val_pickles(sample_pkl_path=sample_pkl_path, val_dict_pkl_path=None, training_set_name=training_set_name, all_matches=True, verbose=True)


sample = pickle.load(open(sample_pkl_path, 'rb'))
model = sample['pred_dicts'][training_set_name]['RF_fit']


explainer = shap.explainers.TreeExplainer(model, X)
shap_values = explainer(X)

out_dir = Path(f'/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/{sample_name}/sv_calling/svensemble2/shap_values')
if not out_dir.exists():
    os.mkdir(out_dir)

pickle.dump(shap_values, open(out_dir / f'{sample_name}.{model_name}.{training_set_name}.all.50:inf.shap_values.pkl', 'wb'))

# for sample_name in NA12878 NA12878_30x HG002 CHM13; do for model_name in unbiased unbiased_GT; do for training_set_name in gridss manta_candidate manta_diploid delly gridss+manta_candidate gridss+manta_diploid gridss+manta_candidate+delly gridss+manta_diploid+delly delly+manta_candidate delly+manta_diploid; do sbatch svensemble/sv_shap.py --sample_name $sample_name --training_set_name $training_set_name --model_name $model_name; done; done; done