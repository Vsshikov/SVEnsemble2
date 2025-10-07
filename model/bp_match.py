

"""
SV callset matching with Truvari-like algorithm
modified to work with breakpoints
"""

import os
import sys

workdir = "/vast/ai_projects/Papenfuss_SVEnsemble2"
sys.path.append(workdir)
os.chdir(workdir)
os.getcwd()

import importlib
from pathlib import Path

import pandas as pd

import svensemble.vcfparse as vcfp
import svensemble._caller_params as caller_params
import svensemble.breakends as bnds
import svensemble.breakpoints as bps
import svensemble.events as evts
import svensemble.caller_specific as caller_specific
import svensemble.utils as sv_utils
import svensemble.chunk_match as chunk_match

importlib.reload(caller_params)
importlib.reload(bnds)
importlib.reload(bps)
importlib.reload(evts)
importlib.reload(caller_specific)
importlib.reload(sv_utils)
importlib.reload(chunk_match)

pd.set_option('display.max_columns', None)


samples = {}
for sample_name in ['CHM13', 'NA12878', 'NA12878_30x']:
    samples.update(sv_utils.create_config_dict(sample_name))

reference_sample_config = sv_utils.create_config_dict('gnomad', is_ground_truth=True)

for sample_name, sample_config in samples.items():
    bp_df_dict = {}
    if 'gridss' in sample_config['callsets'].keys():
        gridss_bp_df = pd.read_csv(sample_config['sample_dir'] / 'sv_calling' / 'gridss' / 'svensemble2' / 'gridss_bps.csv')
        bp_df_dict['gridss'] = gridss_bp_df
    if 'manta_candidate' in sample_config['callsets'].keys():
        manta_candidate_bp_df = pd.read_csv(sample_config['sample_dir'] / 'sv_calling' / 'manta' / 'svensemble2' / 'manta_candidate_bps.csv')
        bp_df_dict['manta_candidate'] = manta_candidate_bp_df
    if 'manta_diploid' in sample_config['callsets'].keys():
        manta_diploid_bp_df = pd.read_csv(sample_config['sample_dir'] / 'sv_calling' / 'manta' / 'svensemble2' / 'manta_diploid_bps.csv')
        bp_df_dict['manta_diploid'] = manta_diploid_bp_df
    bp_df_dict['gnomad'] = pd.read_csv(reference_sample_config['gnomad']['sample_dir'] / 'svensemble2' / 'gnomad_bps.csv')

    matched_bp_df = chunk_match.outer_join_bps(
        bp_df_dict=bp_df_dict,
        ref_callset='gnomad',
        insert_size_threshold=0.7,
        max_dist=200
    )
    if not (sample_config['sample_dir'] / 'sv_calling' / 'svensemble2').exists():
        (sample_config['sample_dir'] / 'sv_calling' / 'svensemble2').mkdir(parents=True, exist_ok=True)
    matched_bp_df.to_csv(sample_config['sample_dir'] / 'sv_calling' / 'svensemble2' / 'matched_bps.csv')

    del bp_df_dict, matched_bp_df