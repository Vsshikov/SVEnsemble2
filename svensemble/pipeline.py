"""
VCF-to-breakpoints parsing for SVEnsemble2 callers.

parse_preprocess_vcf handles every caller with an entry in
callset_specific.parse_recipes; parse_preprocess_delly handles Delly, whose
BND/TRA records need converting with delly2bnd first. Both return the raw
breakpoints DataFrame behind api.preprocess_raw.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
from pysam import VariantFile

from . import vcfparse as vcfp
from . import breakends as bnds
from . import events as evts
from . import callset_specific as callset_specific
from . import delly2bnd as delly2bnd


def _log(msg: str) -> None:
    """Print a message with an ISO 8601 timestamp prefix."""
    print(f"{datetime.now().astimezone().isoformat()} {msg}")


def parse_preprocess_delly(
    sample_name: str,
    workflow_config: dict,
    verbose: bool = False
) -> pd.DataFrame:
    """Read and preprocess the Delly dataset for a sample.

    Delly requires a custom parser because BND/TRA records are first
    converted to a standard BND VCF via delly2bnd, then parsed separately
    from the event records.

    Args:
        sample_name: Name of the sample.
        workflow_config: Configuration dictionary for the sample.
        verbose: If True, print progress messages.
    """
    callset = workflow_config['callsets']['delly']
    if 'ref_path' not in workflow_config:
        raise ValueError("Reference path not found in workflow config")
    ref_path = workflow_config['ref_path']

    with tempfile.TemporaryDirectory(prefix='svensemble_delly_') as temp_dir:
        bnd_vcf_path = Path(temp_dir) / 'delly_bnd.tmp.vcf'
        if verbose:
            _log("Converting DELLY BND/TRA records via delly2bnd")
        delly2bnd.delly2bnd_noargparse(
            vcf=callset['path'],
            ref=ref_path,
            out=bnd_vcf_path
        )

        if verbose:
            _log(f"Loading DELLY BND VCF from {bnd_vcf_path}")
        with VariantFile(bnd_vcf_path) as delly_bnd_only_vcf:
            delly_bnd_only_df, _ = vcfp.parse_vcf_to_df(
                vcf_file=delly_bnd_only_vcf,
                caller_params=callset['params']
            )
    delly_bnd_only_df = callset_specific.delly_preprocess_bnds_only(
        delly_bnd_only_df, caller_params=callset['params']
    )

    if verbose:
        _log(f"Loading DELLY event VCF from {callset['path']}")
    delly_vcf = VariantFile(callset['path'])
    delly_df, _ = vcfp.parse_vcf_to_df(
        vcf_file=delly_vcf,
        caller_params=callset['params']
    )
    delly_evt_only_df = delly_df[~delly_df['INFO_SVTYPE'].isin(['BND', 'TRA'])]
    delly_evt_only_df = callset_specific.delly_preprocess_evts_only(
        delly_evt_only_df, caller_params=callset['params']
    )
    if verbose:
        _log("Converting DELLY events to breakends")
    delly_evt_bnd_df = evts.event_to_bnd(
        delly_evt_only_df, bnd2_pos=callset['params']['bnd2_pos']
    )

    delly_bnd_df = pd.concat([delly_evt_bnd_df, delly_bnd_only_df], axis=0)

    if verbose:
        _log("Converting DELLY breakends to breakpoints")
    delly_bp_df = bnds.bnd_to_bp(
        delly_bnd_df, force_diff_cols=['INFO_CIPOS_l', 'INFO_CIPOS_r']
    )
    return delly_bp_df


def parse_preprocess_vcf(
    callset_name: str,
    callset: dict,
    verbose: bool = False
) -> pd.DataFrame:
    """Generic VCF-to-breakpoints preprocessor driven by callset_specific.parse_recipes.

    Handles three VCF layouts via the recipe's bnd_svtypes field:
      None  - entire VCF is BND (no event_to_bnd step)
      []    - entire VCF is events (all rows go through event_to_bnd)
      list  - rows with INFO_SVTYPE in the list are BND; the rest are events

    Args:
        callset_name: Name of the callset (must be a key in parse_recipes).
        callset: Callset dict containing 'path' and 'params'.
        verbose: If True, print progress messages.

    Returns:
        DataFrame of breakpoints.
    """
    recipe = callset_specific.parse_recipes[callset_name]

    if verbose:
        _log(f"Loading {callset_name} VCF from {callset['path']}")
    vcf = VariantFile(callset['path'])
    df, _ = vcfp.parse_vcf_to_df(vcf_file=vcf, caller_params=callset['params'])

    if recipe['preprocess_all']:
        if verbose:
            _log(f"Preprocessing all {callset_name} variants")
        df = recipe['preprocess_all'](df, caller_params=callset['params'])

    bnd_svtypes = recipe['bnd_svtypes']

    if bnd_svtypes is None:
        # Entire VCF is BND -- go straight to bnd_to_bp
        bnd_df = df
    else:
        if bnd_svtypes:
            bnd_mask = df['INFO_SVTYPE'].isin(bnd_svtypes)
            bnd_df = df[bnd_mask]
            evt_df = df[~bnd_mask]
        else:
            bnd_df = pd.DataFrame(columns=df.columns)
            evt_df = df

        if recipe['preprocess_bnds'] and len(bnd_df) > 0:
            if verbose:
                _log(f"Preprocessing {callset_name} BND variants")
            bnd_df = recipe['preprocess_bnds'](bnd_df, caller_params=callset['params'])

        if recipe['preprocess_evts'] and len(evt_df) > 0:
            if verbose:
                _log(f"Preprocessing {callset_name} event variants")
            evt_df = recipe['preprocess_evts'](evt_df, caller_params=callset['params'])

        if len(evt_df) > 0:
            if verbose:
                _log(f"Converting {callset_name} events to breakends")
            evt_bnd_df = evts.event_to_bnd(evt_df, bnd2_pos=callset['params']['bnd2_pos'])
            bnd_df = pd.concat([bnd_df, evt_bnd_df], axis=0) if len(bnd_df) > 0 else evt_bnd_df

    if verbose:
        _log(f"Converting {callset_name} breakends to breakpoints")
    force_diff = recipe['force_diff_cols']
    bp_df = bnds.bnd_to_bp(bnd_df, force_diff_cols=force_diff) if force_diff else bnds.bnd_to_bp(bnd_df)

    if recipe['post_bps']:
        bp_df = recipe['post_bps'](bp_df)

    return bp_df
