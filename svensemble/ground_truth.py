"""
Functions for preprocessing ground truth sets for SVEnsemble2.

The single public entry point, `preprocess_ground_truth`, takes an explicit
`vcf` and `output_dir` and exposes all filtering options
(FILTER values, EVIDENCE, ALGORITHMS, allele-frequency bounds, SV length
range, BOTHSIDES_SUPPORT, canonical-chromosome restriction). It is invoked
either through `svensemble ground-truth` in the CLI or as a plain Python
call from a workflow.

Ground-truth-family-specific processing (CI assignment, EVENT population,
event-to-BND conversion, BND-to-BP collapse) is dispatched via
`callset_specific.ground_truth_recipes`. The recipe key is
`CALLSET_PARAM_DICT[ground_truth_name]['source']`, so every variant of a
family (e.g. `gnomad`, `gnomad_pass`, `gnomad_filtered`) shares the same
recipe and per-variant differences (CIPOS dictionaries, filter defaults,
input filenames) live only in the params entry.
"""

import logging
import shlex
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from pysam import VariantFile

from . import _caller_params
from . import callset_specific
from ._constants import FILTER_PREFIX
from . import io as sv_io
from . import utils as sv_utils
from . import vcfparse as vcfp

logger = logging.getLogger(__name__)


def preprocess_ground_truth(
    vcf: Path,
    output_dir: Path,
    ground_truth_name: str = "gnomad",
    filters: List[str] | None = None,
    evidence: List[str] | None = None,
    svtypes: List[str] | None = None,
    algorithms: List[str] | None = None,
    min_af: float | None = None,
    max_af: float | None = None,
    svlens: tuple[float, float] | None = (50, np.inf),
    bothsides_support_only: bool = False,
    canonical_chroms_only: bool = True,
    overwrite: bool = False,
    write_manifest: bool = True,
) -> Path:
    """
    Preprocess a ground-truth VCF into a filtered breakpoint CSV.

    Reads `vcf`, applies the requested filters, dispatches to the
    ground-truth-specific preprocessing recipe registered in
    `callset_specific.ground_truth_recipes[ground_truth_name]`, converts to
    breakpoints, applies the SVLEN range filter and writes the result to
    `output_dir/{ground_truth_name}.bps.csv` (io.CALLSET_BPS).

    Args:
        vcf: Full path to the input ground-truth VCF.
        output_dir: Output directory (created if missing).
        ground_truth_name: Key into `_caller_params.CALLSET_PARAM_DICT` for
            parsing parameters and into
            `callset_specific.ground_truth_recipes` for the preprocessing
            recipe. Both registries must contain the name.
        filters: VCF FILTER values to retain. When None (the default), falls
            back to `CALLSET_PARAM_DICT[ground_truth_name]['filter_defaults']
            ['filters']` if set, otherwise no FILTER filter is applied
            (all records kept). This is how e.g. `gnomad_pass` and `gnomad`
            differ in behavior from the same input VCF: `gnomad_pass` sets
            `filter_defaults={'filters': ['PASS'], ...}` while `gnomad` sets
            no `filter_defaults` at all.
        evidence: Required EVIDENCE types (SR, PE, RD, BAF). A record is kept
            if it carries at least one of the listed types. None → resolved
            from `filter_defaults['evidence']` if set, else no filter.
        svtypes: SV types to retain. None → resolved from
            `filter_defaults['svtypes']` if set, else the recipe's
            `supported_svtypes` (e.g. ['DEL', 'DUP', 'INV', 'INS']).
        algorithms: Required ALGORITHMS (depth, manta, melt, rescan, wham).
            None → resolved from `filter_defaults['algorithms']` if set,
            else no filter.
        min_af: Minimum allele frequency (inclusive). None → resolved from
            `filter_defaults['min_af']` if set, else no filter.
        max_af: Maximum allele frequency (inclusive). None → resolved from
            `filter_defaults['max_af']` if set, else no filter.
        svlens: (min_len, max_len) breakpoint SVLEN range applied after
            breakpoint conversion. Defaults to (50, inf); pass None to skip.
        bothsides_support_only: If True, keep only variants with the
            BOTHSIDES_SUPPORT flag set.
        canonical_chroms_only: If True, restrict to chr1-22, X, Y, M.
        overwrite: Re-create the output even if it already exists. Without
            it an existing output is reused as is: only its existence is
            checked, not the inputs or filters that produced it.
        write_manifest: If True, write `{ground_truth_name}.manifest.txt`
            (io.GROUND_TRUTH_MANIFEST) next to the output, recording the source VCF, timestamp, filter set and a
            reproduction command.

    Returns:
        The output bps.csv path that was written (or reused when it already
        existed and `overwrite` is False).

    Raises:
        FileNotFoundError: If `vcf` does not exist.
        KeyError: If `ground_truth_name` is not registered in
            `callset_specific.ground_truth_recipes` or
            `_caller_params.CALLSET_PARAM_DICT`.
    """
    vcf_path = Path(vcf)
    output_dir = Path(output_dir)
    output_path = output_dir / sv_io.CALLSET_BPS.format(caller=ground_truth_name)
    manifest_path = output_dir / sv_io.GROUND_TRUTH_MANIFEST.format(ground_truth_name=ground_truth_name)

    if output_path.exists() and not overwrite:
        logger.warning(
            "Ground truth already exists at %s; reusing it "
            "(pass overwrite / --overwrite to rebuild).",
            output_path,
        )
        return output_path

    if not vcf_path.exists():
        raise FileNotFoundError(
            f'Ground-truth VCF not found at {vcf_path}'
        )

    if ground_truth_name not in _caller_params.CALLSET_PARAM_DICT:
        raise KeyError(
            f'No callset_params registered for ground truth {ground_truth_name!r}. '
            f'Add an entry to _caller_params.CALLSET_PARAM_DICT.'
        )
    gt_params = _caller_params.CALLSET_PARAM_DICT[ground_truth_name]

    recipe_key = gt_params.get('source', ground_truth_name)
    if recipe_key not in callset_specific.ground_truth_recipes:
        raise KeyError(
            f'No processing recipe registered for ground-truth family {recipe_key!r} '
            f'(resolved from CALLSET_PARAM_DICT[{ground_truth_name!r}][\"source\"]). '
            f'Add an entry to callset_specific.ground_truth_recipes. '
            f'Available: {list(callset_specific.ground_truth_recipes)}'
        )
    recipe = callset_specific.ground_truth_recipes[recipe_key]

    logger.info("Parsing ground-truth VCF: %s", vcf_path)
    gt_vcf = VariantFile(str(vcf_path))
    gt_df, _ = vcfp.parse_vcf_to_df(vcf_file=gt_vcf, caller_params=gt_params)
    logger.info("Parsed %d records", len(gt_df))

    filter_defaults = gt_params.get('filter_defaults', {})
    if filters is None:
        filters = filter_defaults.get('filters')
    if evidence is None:
        evidence = filter_defaults.get('evidence')
    if algorithms is None:
        algorithms = filter_defaults.get('algorithms')
    if min_af is None:
        min_af = filter_defaults.get('min_af')
    if max_af is None:
        max_af = filter_defaults.get('max_af')
    if svtypes is None:
        svtypes = filter_defaults.get(
            'svtypes',
            gt_params.get('supported_svtypes', ['DEL', 'DUP', 'INV', 'INS']),
        )

    gt_df = _apply_filters(
        gt_df,
        filters=filters,
        evidence=evidence,
        svtypes=svtypes,
        algorithms=algorithms,
        min_af=min_af,
        max_af=max_af,
        bothsides_support_only=bothsides_support_only,
        canonical_chroms_only=canonical_chroms_only,
    )

    gt_bp_df = recipe['process'](
        gt_df,
        caller_params=gt_params,
        svtypes_retain=svtypes,
    )

    if svlens is not None:
        min_len, max_len = svlens
        logger.debug("Filtering breakpoints by SVLEN [%s, %s]", min_len, max_len)
        gt_bp_df = gt_bp_df[gt_bp_df['INFO_SVLEN'].abs().between(min_len, max_len)]

    # The manifest is written first so the bps CSV -- the file the
    # skip-if-exists check above looks at -- is always the last one written.
    if write_manifest:
        _write_manifest(
            manifest_path=manifest_path,
            output_path=output_path,
            output_dir=output_dir,
            ground_truth_name=ground_truth_name,
            vcf_path=vcf_path,
            n_breakpoints=len(gt_bp_df),
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
        logger.info("Wrote manifest to %s", manifest_path)

    with sv_utils.atomic_path(output_path) as tmp:
        gt_bp_df.to_csv(tmp, index=False)
    logger.info("Saved %d breakpoints to %s", len(gt_bp_df), output_path)

    return output_path


def _write_manifest(
    manifest_path: Path,
    output_path: Path,
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
    svlens: tuple[float, float] | None,
    bothsides_support_only: bool,
    canonical_chroms_only: bool,
) -> None:
    """Write a human-readable manifest alongside the output bps CSV."""
    q = shlex.quote
    lines = [
        f"ground_truth_name: {ground_truth_name}",
        f"source_vcf: {vcf_path}",
        f"output: {output_path}",
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

    cmd_parts = [
        "svensemble ground-truth",
        f"    --vcf {q(str(vcf_path))}",
        f"    --output_dir {q(str(output_dir))}",
        f"    --ground_truth_name {ground_truth_name}",
    ]
    if filters:
        cmd_parts.append(f"    --filters {' '.join(filters)}")
    if evidence:
        cmd_parts.append(f"    --evidence {' '.join(evidence)}")
    if svtypes:
        cmd_parts.append(f"    --svtypes {' '.join(svtypes)}")
    if algorithms:
        cmd_parts.append(f"    --algorithms {' '.join(algorithms)}")
    if min_af is not None:
        cmd_parts.append(f"    --min_af {min_af}")
    if max_af is not None:
        cmd_parts.append(f"    --max_af {max_af}")
    if svlens:
        cmd_parts.append(f"    --svlen_min {svlens[0]}")
        if svlens[1] != np.inf:
            cmd_parts.append(f"    --svlen_max {svlens[1]}")
    if bothsides_support_only:
        cmd_parts.append("    --bothsides_support_only")
    if not canonical_chroms_only:
        cmd_parts.append("    --no_canonical_chroms")
    cmd_parts.append("    --verbose")

    lines.append("")
    lines.append("# Reproduce with:")
    lines.append(" \\\n".join(cmd_parts))

    with sv_utils.atomic_path(manifest_path) as tmp:
        tmp.write_text('\n'.join(lines) + '\n')


def _apply_filters(
    df: pd.DataFrame,
    *,
    filters: List[str] | None,
    evidence: List[str] | None,
    svtypes: List[str],
    algorithms: List[str] | None,
    min_af: float | None,
    max_af: float | None,
    bothsides_support_only: bool,
    canonical_chroms_only: bool,
) -> pd.DataFrame:
    """Apply successive record-level filters to a parsed ground-truth DataFrame.

    `filters=None` or `[]` skips the FILTER filter entirely (all records kept).
    """

    if filters:
        filter_cols = [f'{FILTER_PREFIX}{f}' for f in filters]
        present_cols = [c for c in filter_cols if c in df.columns]
        if present_cols:
            logger.debug("Filtering by FILTER values: %s", filters)
            mask = df[present_cols].any(axis=1).astype(bool)
            df = df[mask]
    else:
        logger.debug("No FILTER filter applied (keeping all records)")

    logger.debug("Filtering by SVTYPE: %s", svtypes)
    df = df[df['INFO_SVTYPE'].isin(svtypes)]

    if evidence is not None and 'INFO_EVIDENCE' in df.columns:
        logger.debug("Filtering by EVIDENCE: %s", evidence)
        evidence_set = set(evidence)
        mask = df['INFO_EVIDENCE'].apply(
            lambda x: bool(evidence_set & set(x)) if x is not None else False
        )
        df = df[mask]

    if algorithms is not None and 'INFO_ALGORITHMS' in df.columns:
        logger.debug("Filtering by ALGORITHMS: %s", algorithms)
        algorithms_set = set(algorithms)
        mask = df['INFO_ALGORITHMS'].apply(
            lambda x: bool(algorithms_set & set(x)) if x is not None else False
        )
        df = df[mask]

    if (min_af is not None or max_af is not None) and 'INFO_AF' in df.columns:
        logger.debug("Filtering by AF: [%s, %s]", min_af, max_af)
        af_series = df['INFO_AF'].apply(
            lambda x: x[0] if isinstance(x, tuple) and len(x) > 0 else x
        )
        if min_af is not None:
            df = df[af_series >= min_af]
            af_series = af_series.loc[df.index]
        if max_af is not None:
            df = df[af_series <= max_af]

    if bothsides_support_only and 'INFO_BOTHSIDES_SUPPORT' in df.columns:
        logger.debug("Filtering to BOTHSIDES_SUPPORT variants only")
        df = df[df['INFO_BOTHSIDES_SUPPORT'].astype(bool)]

    if canonical_chroms_only:
        logger.debug("Filtering to canonical chromosomes")
        df = df[df['CHROM'].isin(sv_utils.human_chroms)]

    logger.info("%d variants remaining after filtering", len(df))

    return df
