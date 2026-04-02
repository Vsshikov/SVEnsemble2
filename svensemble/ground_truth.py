"""
Functions for preprocessing ground truth sets for SVEnsemble2.
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from pysam import VariantFile

from . import _caller_params
from . import breakends as bnds
from . import caller_specific
from . import utils as sv_utils
from . import vcfparse as vcfp


def preprocess_gnomad_ground_truth(
    ground_truth_name: str,
    filters: List[str] | None = None,
    evidence: List[str] | None = None,
    svtypes: List[str] | None = None,
    algorithms: List[str] | None = None,
    min_af: float | None = None,
    max_af: float | None = None,
    svlens: tuple[int, int] | None = (50, np.inf),
    bothsides_support_only: bool = False,
    canonical_chroms_only: bool = True,
    force_rerun: bool = False,
    verbose: bool = False,
) -> None:
    """
    Preprocess a gnomAD VCF into a filtered ground truth breakpoint set.

    Paths are resolved from the config system via create_config_dict(ground_truth_name, is_ground_truth=True),
    which reads callset_path_configs and caller_param_dict from _caller_params.

    Args:
        ground_truth_name (str): Identifier for this ground truth version. Must have a matching entry in callset_path_configs.
        filters (list or None): VCF FILTER values to retain. Defaults to ['PASS'].
        evidence (list or None): Required EVIDENCE types (SR, PE, RD, BAF). A variant is kept if it carries at least one of the listed types. None disables this filter.
        svtypes (list or None): SV types to retain. Defaults to ['DEL', 'DUP', 'INV', 'INS'].
        algorithms (list or None): Required ALGORITHMS (depth, manta, melt, rescan, wham). A variant is kept if it was called by at least one of the listed algorithms. None disables this filter.
        min_af (float or None): Minimum allele frequency (inclusive).
        max_af (float or None): Maximum allele frequency (inclusive).
        svlens (tuple or None): (min_len, max_len) SV length range applied to breakpoint INFO_SVLEN. Defaults to (50, inf).
        bothsides_support_only (bool): If True, keep only variants with the BOTHSIDES_SUPPORT flag.
        canonical_chroms_only (bool): If True, restrict to canonical human chromosomes (chr1-22, X, Y, M).
        force_rerun (bool): Re-create the output even if it already exists.
        verbose (bool): If True, print progress messages.

    """
    if ground_truth_name not in _caller_params.callset_path_configs:
        raise ValueError(
            f'No config found for ground truth name {ground_truth_name!r}. '
            f'Add an entry to callset_path_configs in _caller_params.py. '
            f'Available: {list(_caller_params.callset_path_configs)}'
        )

    config = sv_utils.create_config_dict(
        ground_truth_name, is_ground_truth=True
    )[ground_truth_name]

    callset_key = next(iter(config['callsets']))
    callset = config['callsets'][callset_key]
    vcf_path = Path(callset['path'])
    svensemble_dir = Path(callset['svensemble_dir'])
    gnomad_params = callset['params']

    output_path = svensemble_dir / 'bps.csv'

    if output_path.exists() and not force_rerun:
        if verbose:
            print(f"Output already exists at {output_path}, skipping (use force_rerun=True to overwrite)")
        return

    if not svensemble_dir.exists():
        svensemble_dir.mkdir(parents=True, exist_ok=True)

    if not vcf_path.exists():
        raise FileNotFoundError(
            f'VCF not found at {vcf_path} (resolved for ground truth {ground_truth_name!r}). '
            f'Check the path template in callset_path_configs[{ground_truth_name!r}]'
        )

    # Parse VCF
    if verbose:
        print(f"Parsing gnomAD VCF: {vcf_path}")
    gnomad_vcf = VariantFile(str(vcf_path))
    gnomad_df, _ = vcfp.parse_vcf_to_df(
        vcf_file=gnomad_vcf,
        caller_params=gnomad_params
    )
    if verbose:
        print(f"Parsed {len(gnomad_df)} records")

    # Apply filters
    if filters is None:
        filters = ['PASS']

    if svtypes is None:
        svtypes = gnomad_params.get('supported_svtypes', ['DEL', 'DUP', 'INV', 'INS'])

    gnomad_df = _apply_filters(
        gnomad_df,
        filters=filters,
        evidence=evidence,
        svtypes=svtypes,
        algorithms=algorithms,
        min_af=min_af,
        max_af=max_af,
        bothsides_support_only=bothsides_support_only,
        canonical_chroms_only=canonical_chroms_only,
        verbose=verbose
    )

    # Run gnomad_preprocess for CIPOS / CHR2 / EVENT assignment
    gnomad_df = caller_specific.gnomad_preprocess(
        gnomad_df,
        svtypes_retain=svtypes,
        filter_pass_only=False,
        caller_params=gnomad_params,
        verbose=verbose
    )

    # Convert to breakpoints
    if verbose:
        print("Converting events to breakends")
    gnomad_bnd_df = caller_specific.gnomad_event_to_bnd(gnomad_df, caller_params=gnomad_params)

    if verbose:
        print("Converting breakends to breakpoints")
    gnomad_bp_df = bnds.bnd_to_bp(
        gnomad_bnd_df,
        force_diff_cols=['INFO_CIPOS_l', 'INFO_CIPOS_r']
    )

    # SVLEN filter (applied on breakpoints)
    if svlens is not None:
        min_len, max_len = svlens
        if verbose:
            print(f"Filtering breakpoints by SVLEN [{min_len}, {max_len}]")
        gnomad_bp_df = gnomad_bp_df[
            gnomad_bp_df['INFO_SVLEN'].abs().between(min_len, max_len)
        ]

    # Save
    gnomad_bp_df.to_csv(output_path, index=False)
    if verbose:
        print(f"Saved {len(gnomad_bp_df)} breakpoints to {output_path}")

    _write_manifest(
        output_dir=svensemble_dir,
        ground_truth_name=ground_truth_name,
        vcf_path=vcf_path,
        n_breakpoints=len(gnomad_bp_df),
        filters=filters,
        evidence=evidence,
        svtypes=svtypes,
        algorithms=algorithms,
        min_af=min_af,
        max_af=max_af,
        svlens=svlens,
        bothsides_support_only=bothsides_support_only,
        canonical_chroms_only=canonical_chroms_only,
    )
    if verbose:
        print(f"Wrote manifest and params template to {svensemble_dir}")


def _write_manifest(
    output_dir: Path,
    ground_truth_name: str,
    vcf_path: Path,
    n_breakpoints: int,
    filters: List[str],
    evidence: List[str] | None,
    svtypes: List[str],
    algorithms: List[str] | None,
    min_af: float | None,
    max_af: float | None,
    svlens: tuple[int, int] | None,
    bothsides_support_only: bool,
    canonical_chroms_only: bool,
) -> None:
    """Write a human-readable manifest and a _caller_params.py snippet."""
    from datetime import datetime

    # --- manifest.txt ---
    lines = [
        f"ground_truth_name: {ground_truth_name}",
        f"source_vcf: {vcf_path}",
        f"created: {datetime.now().isoformat(timespec='seconds')}",
        f"breakpoints: {n_breakpoints}",
        "",
        "# Filters applied",
        f"filters: {filters}",
        f"svtypes: {svtypes}",
        f"evidence: {evidence}",
        f"algorithms: {algorithms}",
        f"min_af: {min_af}",
        f"max_af: {max_af}",
        f"svlens: {svlens}",
        f"bothsides_support_only: {bothsides_support_only}",
        f"canonical_chroms_only: {canonical_chroms_only}",
    ]

    # --- Reproduction command ---
    cmd_parts = [
        "python -m svensemble.ground_truth",
        f"    --ground_truth_name {ground_truth_name}",
    ]
    if filters and filters != ['PASS']:
        cmd_parts.append(f"    --filters {','.join(filters)}")
    if evidence:
        cmd_parts.append(f"    --evidence {','.join(evidence)}")
    if svtypes:
        cmd_parts.append(f"    --svtypes {','.join(svtypes)}")
    if algorithms:
        cmd_parts.append(f"    --algorithms {','.join(algorithms)}")
    if min_af is not None:
        cmd_parts.append(f"    --min_af {min_af}")
    if max_af is not None:
        cmd_parts.append(f"    --max_af {max_af}")
    if svlens:
        cmd_parts.append(f"    --svlen_lower {svlens[0]}")
        if svlens[1] != np.inf:
            cmd_parts.append(f"    --svlen_upper {svlens[1]}")
    if bothsides_support_only:
        cmd_parts.append("    --bothsides_support_only")
    if not canonical_chroms_only:
        cmd_parts.append("    --no_canonical_chroms")
    cmd_parts.append("    --verbose")
    lines.append("")
    lines.append("# Reproduce with:")
    lines.append(" \\\n".join(cmd_parts))
    (output_dir / 'manifest.txt').write_text('\n'.join(lines) + '\n')


def _apply_filters(
    df: pd.DataFrame,
    *,
    filters: List[str],
    evidence: List[str] | None,
    svtypes: List[str],
    algorithms: List[str] | None,
    min_af: float | None,
    max_af: float | None,
    bothsides_support_only: bool,
    canonical_chroms_only: bool,
    verbose: bool
) -> pd.DataFrame:
    """Apply successive filters to a parsed gnomAD DataFrame."""

    # FILTER
    filter_cols = [f'FILTER_{f}' for f in filters]
    present_cols = [c for c in filter_cols if c in df.columns]
    if present_cols:
        if verbose:
            print(f"Filtering by FILTER values: {filters}")
        mask = df[present_cols].any(axis=1).astype(bool)
        df = df[mask]

    # SVTYPE
    if verbose:
        print(f"Filtering by SVTYPE: {svtypes}")
    df = df[df['INFO_SVTYPE'].isin(svtypes)]

    # EVIDENCE
    if evidence is not None and 'INFO_EVIDENCE' in df.columns:
        if verbose:
            print(f"Filtering by EVIDENCE: {evidence}")
        evidence_set = set(evidence)
        mask = df['INFO_EVIDENCE'].apply(
            lambda x: bool(evidence_set & set(x)) if x is not None else False
        )
        df = df[mask]

    # ALGORITHMS
    if algorithms is not None and 'INFO_ALGORITHMS' in df.columns:
        if verbose:
            print(f"Filtering by ALGORITHMS: {algorithms}")
        algorithms_set = set(algorithms)
        mask = df['INFO_ALGORITHMS'].apply(
            lambda x: bool(algorithms_set & set(x)) if x is not None else False
        )
        df = df[mask]

    # AF — stored as a single-element tuple by pysam
    if (min_af is not None or max_af is not None) and 'INFO_AF' in df.columns:
        if verbose:
            print(f"Filtering by AF: [{min_af}, {max_af}]")
        af_series = df['INFO_AF'].apply(
            lambda x: x[0] if isinstance(x, tuple) and len(x) > 0 else x
        )
        if min_af is not None:
            df = df[af_series >= min_af]
            af_series = af_series.loc[df.index]
        if max_af is not None:
            df = df[af_series <= max_af]

    # BOTHSIDES_SUPPORT
    if bothsides_support_only and 'INFO_BOTHSIDES_SUPPORT' in df.columns:
        if verbose:
            print("Filtering to BOTHSIDES_SUPPORT variants only")
        df = df[df['INFO_BOTHSIDES_SUPPORT'].astype(bool)]

    # Canonical chromosomes
    if canonical_chroms_only:
        if verbose:
            print("Filtering to canonical chromosomes")
        df = df[df['CHROM'].isin(sv_utils.human_chroms)]

    if verbose:
        print(f"{len(df)} variants remaining after filtering")

    return df


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Preprocess a gnomAD VCF into a filtered ground truth breakpoint set for SVEnsemble2."
    )
    parser.add_argument('--ground_truth_name', type=str, required=True,
                        help="Ground truth version name (must match an entry in callset_path_configs).")
    parser.add_argument('--filters', type=str, default=None,
                        help="Comma-separated FILTER values to retain (default: PASS).")
    parser.add_argument('--evidence', type=str, default=None,
                        help="Comma-separated EVIDENCE types to require (e.g. SR,PE).")
    parser.add_argument('--svtypes', type=str, default=None,
                        help="Comma-separated SV types to retain (default: DEL,DUP,INV,INS).")
    parser.add_argument('--algorithms', type=str, default=None,
                        help="Comma-separated ALGORITHMS to require (e.g. manta,depth).")
    parser.add_argument('--min_af', type=float, default=None,
                        help="Minimum allele frequency (inclusive).")
    parser.add_argument('--max_af', type=float, default=None,
                        help="Maximum allele frequency (inclusive).")
    parser.add_argument('--svlen_lower', type=int, default=50,
                        help="Minimum SV length (default: 50).")
    parser.add_argument('--svlen_upper', type=float, default=np.inf,
                        help="Maximum SV length (default: inf).")
    parser.add_argument('--bothsides_support_only', action='store_true', default=False,
                        help="Keep only variants with BOTHSIDES_SUPPORT flag.")
    parser.add_argument('--no_canonical_chroms', action='store_true', default=False,
                        help="Do not restrict to canonical chromosomes.")
    parser.add_argument('--force_rerun', action='store_true', default=False,
                        help="Overwrite existing output.")
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help="Print progress messages.")

    args = parser.parse_args()

    filters = args.filters.split(',') if args.filters else None
    evidence = args.evidence.split(',') if args.evidence else None
    svtypes = args.svtypes.split(',') if args.svtypes else None
    algorithms = args.algorithms.split(',') if args.algorithms else None

    preprocess_gnomad_ground_truth(
        ground_truth_name=args.ground_truth_name,
        filters=filters,
        evidence=evidence,
        svtypes=svtypes,
        algorithms=algorithms,
        min_af=args.min_af,
        max_af=args.max_af,
        svlens=(args.svlen_lower, args.svlen_upper),
        bothsides_support_only=args.bothsides_support_only,
        canonical_chroms_only=not args.no_canonical_chroms,
        force_rerun=args.force_rerun,
        verbose=args.verbose
    )
