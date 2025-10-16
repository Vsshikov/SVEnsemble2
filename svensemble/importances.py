#! /usr/bin/env python3

#SBATCH --job-name=svensemble2_importances
#SBATCH --output=svensemble2_importances.out
#SBATCH --error=svensemble2_importances.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G

"""
Functions for computing feature importances for svensemble2 models.
"""

import os
import time
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import sklearn.model_selection as skm

# Set working directory and sys.path for local svensemble module
WORKDIR = "/vast/ai_projects/Papenfuss_SVEnsemble2"
os.chdir(WORKDIR)
if WORKDIR not in sys.path:
    sys.path.insert(0, WORKDIR)

import svensemble.matched_breakpoints as matchedbps  # noqa: E402


def load_svensemble2_val_Xy_from_pickles(samples_pkl_path, val_dict_pkl_path, sample_name, training_set_name, verbose=False):
    samples = pickle.load(open(samples_pkl_path, 'rb'))
    val_dict = pickle.load(open(val_dict_pkl_path, 'rb'))

    matched_bps = samples[sample_name]['matched_bps'][training_set_name].copy()
    matched_bps.loc[:, 'val_INFO_EVENT'] = np.where(val_dict[sample_name][f'svensemble2_{training_set_name}']['is_in_val'], True, np.nan)

    preprocessed_matched_bps = matchedbps.preprocess_matchedbps_for_pu(
        joint_bp_df=matched_bps,
        sample_config=samples[sample_name],
        match_sample_name='val',
        training_set_name=training_set_name,
        add_match_ids_to_index=False,
        verbose=verbose
    )

    return matchedbps.get_X_y_from_training_df(preprocessed_matched_bps, match_col='ref_match')


def svensemble2_permutation_importance(model, X, y, n_repeats=10, random_state=42):

    start_time = time.time()

    # Check if features in the model are the same as the dataframe columns
    # If extra columns present in the dataframe, remove them
    # If impossible to make features/columns the same, raise an error
    if set(model.feature_names_in_) != set(X.columns):
        extra_features = set(X.columns) - set(model.feature_names_in_)
        missing_features = set(model.feature_names_in_) - set(X.columns)
        if len(extra_features) > 0:
            print(f"Extra features in the dataframe: {extra_features}")
            X = X.drop(columns=extra_features)
        if len(missing_features) > 0:
            print(f"Missing features in the dataframe: {missing_features}")
            raise ValueError("Features in the model are not the same as the dataframe columns")

    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    return result

def plot_svensemble2_permutation_importance(importances, stds=None, plot_n=20, output=None, title=None):

    if stds is None:
        stds = np.zeros(len(importances))

    importance_df = pd.DataFrame({'importances': importances, 'stds': stds}).sort_values(by='importances', ascending=False).iloc[:plot_n, :]

    fig, ax = plt.subplots()
    importance_df.plot(ax=ax, kind='barh', yerr='stds', y='importances')
    ax.set_title("Mean accuracy decrease")
    ax.set_ylabel("Feature importances using permutation on full model")
    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    fig.set_size_inches((5,10))
    if output is not None:
        plt.savefig(f'{output}')
    else:
        plt.show()


def svensemble2_feature_importance(samples_pkl_path, val_dict_pkl_path, sample_name, training_set_name, model_name='unbiased', output_dir=None, rerun_calc=False, n_repeats=10, random_state=42, plot_n=20, verbose=False):
    """
    Compute feature importances for the specified training set in the specified sample.
    Loads all the files using sample names and saves the importances and plots to specified output directory.

    Requires pred_dicts to be calculated and saved in data/{sample_name}/sv_calling/svensemble2/unbiased_dict.pkl

    IMPORTANT: WILL NEED TO ADD SUPPORT FOR OTHER MODELS

    Args:
        sample_name (str): Name of the sample to process.
        samples_pkl_path (str): Path to the samples pickle file.
        val_dict_pkl_path (str): Path to the val dictionary pickle file.
        training_set_name (str): Name of the training set to process.
        model_name (str): Name of the model to process.
        output_dir (str): Path to the output directory.
        rerun_calc (bool): Whether to rerun the calculation.
        n_repeats (int): Number of times to repeat the permutation importance calculation.
        random_state (int): Random state to use for the permutation importance calculation.
        plot_n (int): Number of features to plot.
        verbose (bool): Whether to print verbose output.
    """
    # TEMPORARY: DEFAULT OUTPUT DIR
    if output_dir is None:
        output_dir = '/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/model/figs/featureimportance'
        output_png = f'{output_dir}/svensemble2_permutation_importance_{sample_name}_{training_set_name}_{model_name}.png'
        output_csv = f'{output_dir}/svensemble2_permutation_importance_{sample_name}_{training_set_name}_{model_name}.csv'

    if rerun_calc and os.path.exists(output_csv):
        os.remove(output_csv)
        os.remove(output_png)

    if os.path.exists(output_csv):
        importances_df = pd.read_csv(output_csv, index_col=0)
        importances_series = importances_df['importances']
        importances_stds = importances_df['stds']

    else:
        X, y = load_svensemble2_val_Xy_from_pickles(samples_pkl_path, val_dict_pkl_path, sample_name, training_set_name, verbose=verbose)

        # Load the trained model + predictions
        svensemble2_model = pickle.load(open(samples_pkl_path, 'rb'))[sample_name]['pred_dicts'][training_set_name]['RF_fit']

        # Check if features in the model are the same as the X columns
        # If extra columns present in the dataframe, remove them
        # If impossible to make features/columns the same, raise an error
        if set(svensemble2_model.feature_names_in_) != set(X.columns):
            extra_features = set(X.columns) - set(svensemble2_model.feature_names_in_)
            missing_features = set(svensemble2_model.feature_names_in_) - set(X.columns)
            if len(extra_features) > 0:
                print(f"Extra features in the dataframe: {extra_features}, dropping them")
                X = X.drop(columns=extra_features)
            if len(missing_features) > 0:
                print(f"Missing features in the dataframe: {missing_features}")
                raise ValueError("Features in the model are not the same as the dataframe columns")

        importances = svensemble2_permutation_importance(svensemble2_model, X, y, n_repeats=n_repeats, random_state=random_state)

        importances_series = pd.Series(importances.importances_mean, index=X.columns)
        importances_stds = pd.Series(importances.importances_std, index=X.columns)

        pd.DataFrame({'importances': importances_series, 'stds': importances_stds}).to_csv(output_csv)

    plot_svensemble2_permutation_importance(importances_series, stds=importances_stds, plot_n=plot_n, output=output_png, title=f"{sample_name}\n{training_set_name}\n{model_name}")




if __name__ == '__main__':
    import argparse

    # Create the argument parser for command-line execution
    parser = argparse.ArgumentParser(
        description="Compute feature importances for svensemble2 models."
    )
    parser.add_argument(
        '--samples_pkl_path',
        type=str,
        required=False,
        help="Path to the samples pickle file."
    )
    parser.add_argument(
        '--val_dict_pkl_path',
        type=str,
        required=False,
        help="Path to the val dictionary pickle file."
    )
    parser.add_argument(
        '--config_file',
        type=str,
        required=False,
        help="Path to the config file."
    )
    parser.add_argument(
        '--config_line',
        type=int,
        required=False,
        help="Line number of the config file to process."
    )
    parser.add_argument(
        '--sample_name',
        type=str,
        required=False,
        help="Name of the sample to process."
    )
    parser.add_argument(
        '--training_set_name',
        type=str,
        required=False,
        help="Name of the training set to process."
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='unbiased',
        help="Name of the model to process."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help="Directory to save the output."
    )
    parser.add_argument(
        '--rerun_calc',
        type=bool,
        default=False,
        help="Whether to rerun the calculation."
    )
    parser.add_argument(
        '--n_repeats',
        type=int,
        default=10,
        help="Number of times to repeat the permutation importance calculation."
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help="Random state to use for the permutation importance calculation."
    )
    parser.add_argument(
        '--plot_n',
        type=int,
        default=20,
        help="Number of features to plot."
    )

    args = parser.parse_args()

    if args.config_file is not None and args.config_line is not None:
        with open(args.config_file, 'r') as f:
            config_lines = f.readlines()
        k = 0
        for line in config_lines:
            if k == args.config_line+1:
                config_line = line
                break
            k += 1
        sample_name, training_set_name, model_name = config_line.strip().split(',')
    else:
        sample_name = args.sample_name
        training_set_name = args.training_set_name
        model_name = args.model_name

    svensemble2_feature_importance(
        samples_pkl_path=args.samples_pkl_path,
        val_dict_pkl_path=args.val_dict_pkl_path,
        sample_name=sample_name,
        training_set_name=training_set_name,
        model_name=model_name,
        output_dir=args.output_dir,
        rerun_calc=args.rerun_calc,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
        plot_n=args.plot_n
    )
