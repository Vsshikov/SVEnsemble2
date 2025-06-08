import os

workdir = '/vast/ai_projects/Papenfuss_SVEnsemble2/model'
os.chdir(workdir)
os.getcwd()

import copy
import importlib

import numpy as np
import pandas as pd
import matplotlib as mpl
import pyranges as pr
import sklearn

from sklearn.ensemble import RandomForestClassifier as RF
import sklearn.model_selection as skm
from sklearn.metrics import confusion_matrix

# df_path = 'saved/gridss_overlaps.tsv'
# callset_overlapped_df = pd.read_csv(df_path, sep='\t', index_col='ID')

# df_path = 'saved/gridss.overlapped_bps.csv'
df_path = 'saved/training_sets/gridss_ref_matched_final.csv'
callset_overlapped_df = pd.read_csv(df_path, index_col='INFO_EVENT_gridss')
# callset_overlapped_df = callset_overlapped_df.drop(columns=['REF_bnd1_gridss', 'REF_bnd2_gridss', 'POS_bnd1_gridss', 'POS_bnd2_gridss'])
# callset_overlapped_df = callset_overlapped_df.loc[:, ~(callset_overlapped_df.dtypes == 'object')]

df_path_manta = 'saved/training_sets/manta_ref_matched_final.csv'
callset_overlapped_df_manta = pd.read_csv(df_path_manta, index_col='INFO_EVENT_manta')
# callset_overlapped_df_manta = callset_overlapped_df_manta.drop(columns=['REF_bnd1_manta', 'REF_bnd2_manta', 'POS_bnd1_manta', 'POS_bnd2_manta'])
# callset_overlapped_df_manta = callset_overlapped_df_manta.loc[:, ~(callset_overlapped_df_manta.dtypes == 'object')]

df_path_gridss_manta = 'saved/training_sets/gridss_manta_ref_matched_final.csv'
callset_overlapped_df_gridss_manta = pd.read_csv(df_path_gridss_manta, index_col='INFO_EVENT_combined')
callset_overlapped_df_gridss_manta = callset_overlapped_df_gridss_manta.drop(columns=['INFO_EVENT_manta_combined'])
# callset_overlapped_df_gridss_manta = callset_overlapped_df_gridss_manta.drop(columns=['REF_bnd1_combined', 'REF_bnd2_combined', 'POS_bnd1_combined', 'POS_bnd2_combined', 'POS_bnd1_manta_combined', 'POS_bnd2_manta_combined', 'REF_bnd1_manta_combined', 'REF_bnd2_manta_combined'])
# callset_overlapped_df_gridss_manta = callset_overlapped_df_gridss_manta.loc[:, ~(callset_overlapped_df_gridss_manta.dtypes == 'object')]
substrings_for_removal = ['bnd_pos_encoding', '_POS_', 'POS_']

for i, col in enumerate(callset_overlapped_df.columns):
    if any(substring in col for substring in substrings_for_removal):
        callset_overlapped_df = callset_overlapped_df.drop(columns=[col])
        continue
    for col2 in callset_overlapped_df.columns[i+1:]:
        if col.replace('INFO', 'FORMAT') == col2.replace('INFO', 'FORMAT'):
            print(f'{col} and {col2} are identical')
            callset_overlapped_df = callset_overlapped_df.drop(columns=[col])

for i, col in enumerate(callset_overlapped_df_manta.columns):
    if any(substring in col for substring in substrings_for_removal):
        callset_overlapped_df_manta = callset_overlapped_df_manta.drop(columns=[col])
        continue
    for col2 in callset_overlapped_df_manta.columns[i+1:]:
        if col.replace('INFO', 'FORMAT') == col2.replace('INFO', 'FORMAT'):
            print(f'{col} and {col2} are identical')
            callset_overlapped_df_manta = callset_overlapped_df_manta.drop(columns=[col])

for i, col in enumerate(callset_overlapped_df_gridss_manta.columns):
    if any(substring in col for substring in substrings_for_removal):
        callset_overlapped_df_gridss_manta = callset_overlapped_df_gridss_manta.drop(columns=[col])
        continue
    for col2 in callset_overlapped_df_gridss_manta.columns[i+1:]:
        if col.replace('INFO', 'FORMAT') == col2.replace('INFO', 'FORMAT'):
            print(f'{col} and {col2} are identical')
            callset_overlapped_df_gridss_manta = callset_overlapped_df_gridss_manta.drop(columns=[col])
def get_overlapped_xy(overlapped_df, match_col='is_in_training'):
    overlapped_df_copy = overlapped_df.copy().rename(columns={match_col: 'ref_match'})
    X = pd.get_dummies(overlapped_df_copy.drop(columns=['ref_match']))
    # Drop columns with only 1 unique value
    nunique = X.nunique()
    constant_cols = nunique[nunique == 1].index
    X = X.drop(columns=constant_cols)
    y = overlapped_df_copy['ref_match']
    return X, y

X, y = get_overlapped_xy(callset_overlapped_df)
X_manta, y_manta = get_overlapped_xy(callset_overlapped_df_manta)
X_gridss_manta, y_gridss_manta = get_overlapped_xy(callset_overlapped_df_gridss_manta)

X_train, X_val, y_train, y_val = skm.train_test_split(X, y, train_size=0.8, random_state=0)
X_train_manta, X_val_manta, y_train_manta, y_val_manta = skm.train_test_split(X_manta, y_manta, train_size=0.8, random_state=0)
X_train_gridss_manta, X_val_gridss_manta, y_train_gridss_manta, y_val_gridss_manta = skm.train_test_split(X_gridss_manta, y_gridss_manta, train_size=0.8, random_state=0)

RF_nontrad = RF(max_features=int(np.sqrt(X.shape[1])), random_state=0, n_estimators=500)
RF_nontrad_fit = RF_nontrad.fit(X_train, y_train)
y_nontrad = RF_nontrad_fit.predict(X_val)

label_freq = sum(y_nontrad & y_val) / sum(y_val)
label_prob = sum(y) / len(y)
clpr = label_prob / label_freq



RF_nontrad_manta = RF(max_features=int(np.sqrt(X_manta.shape[1])), random_state=0, n_estimators=500)
RF_nontrad_manta_fit = RF_nontrad_manta.fit(X_train_manta, y_train_manta)
y_nontrad_manta = RF_nontrad_manta_fit.predict(X_val_manta)

label_freq_manta = sum(y_nontrad_manta & y_val_manta) / sum(y_val_manta)
label_prob_manta = sum(y_manta) / len(y_manta)
clpr_manta = label_prob_manta / label_freq_manta



RF_nontrad_gridss_manta = RF(max_features=int(np.sqrt(X_gridss_manta.shape[1])), random_state=0, n_estimators=500)
RF_nontrad_gridss_manta_fit = RF_nontrad_gridss_manta.fit(X_train_gridss_manta, y_train_gridss_manta)
y_nontrad_gridss_manta = RF_nontrad_gridss_manta_fit.predict(X_val_gridss_manta)

label_freq_gridss_manta = sum(y_nontrad_gridss_manta & y_val_gridss_manta) / sum(y_val_gridss_manta)
label_prob_gridss_manta = sum(y_gridss_manta) / len(y_gridss_manta)
clpr_gridss_manta = label_prob_gridss_manta / label_freq_gridss_manta


# Define class weights based on label probability and class prior ratio
# Weight for negative class (0) is inversely proportional to probability of negative examples
# Weight for positive class (1) accounts for class prior ratio and label probability
weight_dict = {
    0: 1 / (1 - label_prob),  # Weight for negative class
    1: 2 * clpr / label_prob  # Weight for positive class
}

# Initialize Random Forest classifier with:
# - Number of features per split: sqrt(total features) 
# - Fixed random seed for reproducibility
# - Class weights from above
# - 500 trees in the forest
RF_duplessis = RF(
    max_features=int(np.sqrt(X.shape[1])),
    random_state=0,
    class_weight=weight_dict,
    n_estimators=500
)

# Train the classifier on training data
RF_duplessis_fit = RF_duplessis.fit(X_train, y_train)

# Get probability predictions for all examples
# Extract probability of positive class ([:,1])
y_duplessis = RF_duplessis_fit.predict_proba(X)[:,1]

# Save predictions to CSV file
# pd.Series(y_duplessis).to_csv('saved/y_duplessis.csv')
pd.Series(y_duplessis).to_csv('saved/model_output/y_duplessis_bps.csv')



weight_dict_manta = {
    0: 1 / (1 - label_prob_manta),  # Weight for negative class
    1: 2 * clpr_manta / label_prob_manta  # Weight for positive class
}
RF_duplessis_manta = RF(
    max_features=int(np.sqrt(X_manta.shape[1])),
    random_state=0,
    class_weight=weight_dict_manta,
    n_estimators=500
)
RF_duplessis_manta_fit = RF_duplessis_manta.fit(X_train_manta, y_train_manta)
y_duplessis_manta = RF_duplessis_manta_fit.predict_proba(X_val_manta)[:,1]
pd.Series(y_duplessis_manta).to_csv('saved/model_output/y_duplessis_manta_bps.csv')



weight_dict_gridss_manta = {
    0: 1 / (1 - label_prob_gridss_manta),  # Weight for negative class
    1: 2 * clpr_gridss_manta / label_prob_gridss_manta  # Weight for positive class
}
RF_duplessis_gridss_manta = RF(
    max_features=int(np.sqrt(X_gridss_manta.shape[1])),
    random_state=0,
    class_weight=weight_dict_gridss_manta,
    n_estimators=500
)
RF_duplessis_gridss_manta_fit = RF_duplessis_gridss_manta.fit(X_train_gridss_manta, y_train_gridss_manta)
y_duplessis_gridss_manta = RF_duplessis_gridss_manta_fit.predict_proba(X_val_gridss_manta)[:,1]
pd.Series(y_duplessis_gridss_manta).to_csv('saved/model_output/y_duplessis_gridss_manta_bps.csv')


from sklearn.inspection import permutation_importance
import time

start_time = time.time()
result = permutation_importance(
    RF_duplessis_fit, X_val, y_val, n_repeats=10, random_state=42, n_jobs=2
)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.DataFrame({'importances': result.importances_mean, 'std': result.importances_std}, index=X.columns)

start_time = time.time()
result = permutation_importance(
    RF_duplessis_manta_fit, X_val_manta, y_val_manta, n_repeats=10, random_state=42, n_jobs=2
)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances_manta = pd.DataFrame({'importances': result.importances_mean, 'std': result.importances_std}, index=X_manta.columns)

start_time = time.time()
result = permutation_importance(
    RF_duplessis_gridss_manta_fit, X_val_gridss_manta, y_val_gridss_manta, n_repeats=10, random_state=42, n_jobs=2
)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances_gridss_manta = pd.DataFrame({'importances': result.importances_mean, 'std': result.importances_std}, index=X_gridss_manta.columns)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

importance_list = [importance.sort_values('importances', ascending=True)[-20:] for importance in [forest_importances, forest_importances_manta, forest_importances_gridss_manta]]


for caller, importances in zip(['gridss', 'manta', 'gridss+manta'], importance_list):
    importances['importances'].plot(xerr=importances['std'], ax=ax, kind='barh')
    ax.set_title(f"Feature importances using permutation on full model\nCaller: {caller}")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()

    plt.savefig(f'figs/featureimportance_{caller}.png')


y_proba_nontrad = RF_nontrad.predict_proba(X_train)
pos_weights = y_proba_nontrad[:, 1]
pos_weights[y_train] = 1

y_proba_nontrad_manta = RF_nontrad_manta.predict_proba(X_train_manta)
pos_weights_manta = y_proba_nontrad_manta[:, 1]
pos_weights_manta[y_train_manta] = 1

y_proba_nontrad_gridss_manta = RF_nontrad_gridss_manta.predict_proba(X_train_gridss_manta)
pos_weights_gridss_manta = y_proba_nontrad_gridss_manta[:, 1]
pos_weights_gridss_manta[y_train_gridss_manta] = 1


# We double the unlabeled part of X and y and give negative weights to it
X_train_neg = X_train[y_train == False]
y_train_neg = y_train[y_train == False]
neg_weights = y_proba_nontrad[y_train == False, 0]
X_doubled = pd.concat([X_train, X_train_neg])
y_doubled = np.concatenate([y_train, y_train_neg])
weights_doubled = np.concatenate([pos_weights, neg_weights])
RF_enoto = RF(
    max_features=int(np.sqrt(X.shape[1])), random_state=0, n_estimators=500
)
RF_enoto_fit = RF_enoto.fit(X_doubled, y_doubled, sample_weight=weights_doubled)


X_train_neg_manta = X_train_manta[y_train_manta == False]
y_train_neg_manta = y_train_manta[y_train_manta == False]
neg_weights_manta = y_proba_nontrad_manta[y_train_manta == False, 0]
X_doubled_manta = pd.concat([X_train_manta, X_train_neg_manta])
y_doubled_manta = np.concatenate([y_train_manta, y_train_neg_manta])
weights_doubled_manta = np.concatenate([pos_weights_manta, neg_weights_manta])
RF_enoto_manta = RF(
    max_features=int(np.sqrt(X_manta.shape[1])), random_state=0, n_estimators=500
)
RF_enoto_manta_fit = RF_enoto_manta.fit(X_doubled_manta, y_doubled_manta, sample_weight=weights_doubled_manta)


X_train_neg_gridss_manta = X_train_gridss_manta[y_train_gridss_manta == False]
y_train_neg_gridss_manta = y_train_gridss_manta[y_train_gridss_manta == False]
neg_weights_gridss_manta = y_proba_nontrad_gridss_manta[y_train_gridss_manta == False, 0]
X_doubled_gridss_manta = pd.concat([X_train_gridss_manta, X_train_neg_gridss_manta])
y_doubled_gridss_manta = np.concatenate([y_train_gridss_manta, y_train_neg_gridss_manta])
weights_doubled_gridss_manta = np.concatenate([pos_weights_gridss_manta, neg_weights_gridss_manta])
RF_enoto_gridss_manta = RF(
    max_features=int(np.sqrt(X_gridss_manta.shape[1])), random_state=0, n_estimators=500
)
RF_enoto_gridss_manta_fit = RF_enoto_gridss_manta.fit(X_doubled_gridss_manta, y_doubled_gridss_manta, sample_weight=weights_doubled_gridss_manta)

y_enoto = RF_enoto_fit.predict_proba(X)[:,1]
y_enoto_manta = RF_enoto_manta_fit.predict_proba(X_manta)[:,1]
y_enoto_gridss_manta = RF_enoto_gridss_manta_fit.predict_proba(X_gridss_manta)[:,1]

pd.DataFrame({
    'ID': callset_overlapped_df.index,
    'y_duplessis': y_duplessis,
    'y_enoto': y_enoto
}).to_csv('saved/model_output/y_duplessis_enoto_bps.csv', index=False)

pd.DataFrame({
    'ID': callset_overlapped_df_manta.index,
    'y_duplessis': y_duplessis_manta,
    'y_enoto': y_enoto_manta
}).to_csv('saved/model_output/y_duplessis_enoto_manta_bps.csv', index=False)

pd.DataFrame({
    'ID': callset_overlapped_df_gridss_manta.index,
    'y_duplessis': y_duplessis_gridss_manta,
    'y_enoto': y_enoto_gridss_manta
}).to_csv('saved/model_output/y_duplessis_enoto_gridss_manta_bps.csv', index=False)
