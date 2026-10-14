"""
Validation (benchmarking truth) set preprocessing.

Each preprocessor turns one published truth set into the breakpoints CSV that
`benchmark-callset --val_bps` consumes. preprocess_validation_set is the entry
point; its `name` is the set's callset_param_dict key and must match the
--val_name used when benchmarking. Parallels ground_truth.py.
"""

from pathlib import Path

import pandas as pd
import pysam

from . import breakends as bnds
from . import events as evts
from . import utils as sv_utils
from . import _caller_params as caller_params
from . import vcfparse as vcfp


def load_truth_variants(ref_path):
    """Load reference variants from Kosugi study from BED file

    Args:
        ref_path: Path to BED file with reference variants

    Returns:
        DataFrame with columns:
        - Chromosome, Start, End: Variant coordinates
        - ALT: Variant type (DEL, INS, etc)
        - length: Length of variant
        - is_in_KosugiSet: Always True
        - ID: String identifier combining coordinates
    """
    ref_df = pd.read_csv(ref_path, sep='\t', names=['Chromosome', 'Start', 'End', 'ALT'])
    ref_df.loc[:, 'length'] = ref_df['End'] - ref_df['Start']
    ref_df.loc[:, 'is_in_KosugiSet'] = True
    ref_df.loc[:, 'ID'] = ref_df['Chromosome'].astype(str) + ':' + ref_df['Start'].astype(str) + '-' + ref_df['End'].astype(str)
    return ref_df

def truth_to_vcf_df(ref_df):
    """Convert truth variant set from original format to VCF-like DataFrame

    Args:
        ref_df_pr: DataFrame with reference variants in Kosugi format

    Returns:
        DataFrame with VCF-like fields including:
        - CHROM, POS: Variant position
        - INFO_END: End position
        - INFO_CHR2: Secondary chromosome
        - INFO_EVENT: Variant ID
        - INFO_SVTYPE: Variant type
        - INFO_SVLEN: Variant length
        - INFO_CIPOS_l: Left confidence interval (0)
        - INFO_CIPOS_r: Right confidence interval (0)
    """
    ref_vcf = ref_df.rename(columns={
        'Chromosome': 'CHROM',
        'Start': 'POS',
        'End': 'INFO_END'
    })

    ref_vcf.loc[:, 'INFO_CHR2'] = ref_vcf['CHROM']
    ref_vcf.loc[:, 'INFO_EVENT'] = ref_vcf['ID']
    ref_vcf.loc[:, 'INFO_SVTYPE'] = ref_vcf['ALT']
    ref_vcf.loc[:, 'INFO_SVLEN'] = ref_vcf['length']
    ref_vcf.loc[:, 'INFO_CIPOS_l'] = 0
    ref_vcf.loc[:, 'INFO_CIPOS_r'] = 0
    ref_vcf.loc[:, 'INFO_CIEND_l'] = 0
    ref_vcf.loc[:, 'INFO_CIEND_r'] = 0

    return ref_vcf

def truth_event_to_bnd(ref_vcf):
    """Convert truth event VCF to breakpoint pairs

    Args:
        ref_vcf: DataFrame with truth variants in VCF format

    Returns:
        DataFrame with breakpoint pairs including:
        - Original VCF fields split into bnd1/bnd2
        - bnd_pos_encoding: Encoding of breakpoint orientations
        - bndtype: Combined orientation encoding for the pair
    """
    ref_vcf_bnds = evts.event_to_bnd(ref_vcf)
    ref_vcf_bps = bnds.bnd_to_bp(ref_vcf_bnds, force_diff_cols=['CHROM', 'INFO_CIPOS_l', 'INFO_CIPOS_r'])
    return ref_vcf_bps


def preprocess_hg002_val_set(raw_path, output_path, force_rerun=False, long_val=False):
    """
    Preprocess the HG002 validation VCF file for benchmarking by parsing, annotating,
    and converting SV events to breakpoints. Results are stored as a CSV.

    Args:
        raw_path (str or Path): Path to the input VCF file for HG002.
        output_path (str or Path): Path of the breakpoints CSV to write.
        force_rerun (bool, optional): If True, forces preprocessing even if the output exists.
        long_val (bool, optional): Only retain variants with SVLEN > 50 if True.
    """
    # Only run if the annotated file does not exist, or if force_rerun is True
    if not Path(output_path).exists() or force_rerun:

        # Parse the input VCF file using the predefined caller parameters
        hg002_val_vcf = pysam.VariantFile(raw_path)
        hg002_val_vcf_df, hg002_val_vcf_meta = vcfp.parse_vcf_to_df(
            hg002_val_vcf, caller_params=caller_params.hg002_params
        )

        # Filter out entries with missing SVTYPE
        hg002_val_vcf_df = hg002_val_vcf_df[hg002_val_vcf_df['INFO_SVTYPE'].notna()]

        # Assign unique event IDs and set them as VCF IDs
        hg002_val_vcf_df.loc[:, 'INFO_EVENT'] = [
            f'HG002_{i}' for i in range(len(hg002_val_vcf_df))
        ]
        hg002_val_vcf_df.loc[:, 'ID'] = hg002_val_vcf_df['INFO_EVENT']

        # Set dummy confidence intervals for positions and ends, and set INFO_CHR2 to chromosome name
        hg002_val_vcf_df.loc[:, 'INFO_CIPOS_l'] = -10
        hg002_val_vcf_df.loc[:, 'INFO_CIPOS_r'] = 10
        hg002_val_vcf_df.loc[:, 'INFO_CIEND_l'] = -10
        hg002_val_vcf_df.loc[:, 'INFO_CIEND_r'] = 10
        hg002_val_vcf_df.loc[:, 'INFO_CHR2'] = hg002_val_vcf_df['CHROM']

        # Transform events to breakend format, then to breakpoints
        hg002_val_vcf_df_bnds = evts.event_to_bnd(hg002_val_vcf_df)
        hg002_val_vcf_df_bps = bnds.bnd_to_bp(
            hg002_val_vcf_df_bnds,
            force_diff_cols=['INFO_CIPOS_l', 'INFO_CIPOS_r']
        )

        # If only 'long' variants are desired, filter for SVLEN > 50
        if long_val:
            hg002_val_vcf_df_bps = hg002_val_vcf_df_bps[
                hg002_val_vcf_df_bps['INFO_SVLEN'] > 50
            ]

        # Save the processed DataFrame as CSV
        hg002_val_vcf_df_bps.to_csv(output_path, index=False)


def preprocess_kosugi_val_set(raw_path, output_path, force_rerun=False, long_val=False):
    """
    Preprocess the Kosugi validation set, converting to VCF-like breakpoints and saving as CSV.

    Args:
        raw_path (str or Path): Path to the Kosugi BED-like truth file.
        output_path (str or Path): Path of the breakpoints CSV to write.
        force_rerun (bool, optional): If True, regenerate even if the file already exists.
        long_val (bool, optional): Only retain variants with SVLEN > 50 if True.
    """
    out_path = Path(output_path)

    if not out_path.exists() or force_rerun:
        val_df = load_truth_variants(
            raw_path
        )
        val_vcf = truth_to_vcf_df(val_df)

        # Convert Kosugi variant DataFrame to breakpoint format
        val_vcf_bps = truth_event_to_bnd(val_vcf)

        if long_val:
            val_vcf_bps = val_vcf_bps[val_vcf_bps['INFO_SVLEN'] > 50]

        val_vcf_bps.to_csv(out_path, index=False)


def preprocess_chm13_val_set(raw_path, output_path, force_rerun=False):
    """
    Preprocess CHM13 validation set: filter, normalize, convert to breakpoints, save as CSV.

    Args:
        raw_path (str or Path): Path to the CHM13 validation annotation CSV.
        output_path (str or Path): Path of the breakpoints CSV to write.
        force_rerun (bool, optional): If True, regenerate even if the file already exists.
    """
    out_path = Path(output_path)

    if not out_path.exists() or force_rerun:
        # Read raw CHM13 validation annotation data
        chm_val_df = pd.read_csv(
            raw_path
        )

        chm_val_vcf_df = chm_val_df.copy()

        # Filter for rows with valid coordinates and correct assembly
        chm_val_vcf_df = chm_val_vcf_df.dropna(subset=['Start', 'End'])
        chm_val_vcf_df = chm_val_vcf_df[
            chm_val_vcf_df['Assembly Name'] == 'GRCh38 (hg38)'
        ]

        # Standardize variant type nomenclature
        chm_rename_dict = {
            'insertion': 'INS',
            'deletion': 'DEL',
            'inversion': 'INV',
        }
        chm_val_vcf_df.loc[:, 'Variant Call type'] = chm_val_vcf_df[
            'Variant Call type'
        ].map(chm_rename_dict)

        # Ensure variant coordinates are integers
        chm_val_vcf_df.loc[:, 'Start'] = chm_val_vcf_df['Start'].astype(int)
        chm_val_vcf_df.loc[:, 'End'] = chm_val_vcf_df['End'].astype(int)

        # Prepare field names and assign standardized VCF-like values
        chm_val_cols_to_keep = [
            'CHROM', 'POS', 'INFO_CHR2', 'REF', 'ALT', 'ID',
            'INFO_EVENT', 'INFO_SVTYPE', 'INFO_SVLEN', 'INFO_END',
            'INFO_CIPOS_l', 'INFO_CIPOS_r', 'INFO_CIEND_l', 'INFO_CIEND_r'
        ]
        chm_val_vcf_df.loc[:, 'CHROM'] = 'chr' + chm_val_vcf_df['Chromosome'].astype(str)
        chm_val_vcf_df.loc[:, 'POS'] = chm_val_vcf_df['Start'] + 1
        chm_val_vcf_df.loc[:, 'INFO_CHR2'] = 'chr' + chm_val_vcf_df['Chromosome'].astype(str)
        chm_val_vcf_df.loc[:, 'INFO_END'] = chm_val_vcf_df['End']
        chm_val_vcf_df.loc[:, 'REF'] = '.'
        chm_val_vcf_df.loc[:, 'ALT'] = '.'
        chm_val_vcf_df.loc[:, 'ID'] = chm_val_vcf_df['Variant ID']
        chm_val_vcf_df.loc[:, 'INFO_EVENT'] = chm_val_vcf_df['Variant ID']
        chm_val_vcf_df.loc[:, 'INFO_SVTYPE'] = chm_val_vcf_df['Variant Call type']
        chm_val_vcf_df.loc[:, 'INFO_SVLEN'] = 0
        chm_val_vcf_df.loc[:, 'INFO_CIPOS_l'] = -10
        chm_val_vcf_df.loc[:, 'INFO_CIPOS_r'] = 10
        chm_val_vcf_df.loc[:, 'INFO_CIEND_l'] = -10
        chm_val_vcf_df.loc[:, 'INFO_CIEND_r'] = 10

        # Subset and retain only canonical chromosomes
        chm_val_vcf_df = chm_val_vcf_df[chm_val_cols_to_keep]
        chm_val_vcf_df = chm_val_vcf_df[
            chm_val_vcf_df['CHROM'].isin(sv_utils.human_chroms)
        ]

        # Convert to breakpoint format. No SVLEN in this reference set for long_val filter.
        result_bps = bnds.bnd_to_bp(
            evts.event_to_bnd(chm_val_vcf_df, bnd2_pos='end'),
            force_diff_cols=['INFO_CIPOS_l', 'INFO_CIPOS_r']
        )

        result_bps.to_csv(out_path, index=False)


def preprocess_hg00514_val_set(raw_path, output_path, force_rerun=False):
    """
    Preprocess the HG00514 validation VCF file for benchmarking by parsing, annotating,
    and converting SV events to breakpoints. Results are stored as a bps CSV.

    Args:
        raw_path (str or Path): Path to nstd152.GRCh38.variant_call.vcf.gz (or uncompressed).
        output_path (str or Path): Path of the breakpoints CSV to write.
        force_rerun (bool, optional): If True, regenerate even if the output CSV already exists.
    """
    out_path = Path(output_path)

    if not out_path.exists() or force_rerun:
        hg00514_vcf = pysam.VariantFile(raw_path)
        hg00514_vcf_df, _ = vcfp.parse_vcf_to_df(
            hg00514_vcf, caller_params=caller_params.hg00514_params
        )

        # Filter to HG00514 sample and canonical SV types only
        hg00514_vcf_df = hg00514_vcf_df[
            (hg00514_vcf_df['INFO_SAMPLE'] == 'HG00514') &
            (hg00514_vcf_df['INFO_SVTYPE'].isin(['INS', 'DEL', 'DUP', 'INV']))
        ]

        # VCF uses bare chromosome numbers ("1") — prepend "chr"
        hg00514_vcf_df.loc[:, 'CHROM'] = 'chr' + hg00514_vcf_df['CHROM'].astype(str)
        hg00514_vcf_df.loc[:, 'INFO_CHR2'] = hg00514_vcf_df['CHROM']

        # SVLEN is declared Type=String in this VCF; pysam returns a tuple of strings.
        # DEL records carry a negative value — take the absolute integer.
        hg00514_vcf_df.loc[:, 'INFO_SVLEN'] = (
            hg00514_vcf_df['INFO_SVLEN']
            .apply(lambda x: abs(int(x[0])) if x is not None else 0)
        )

        # CIPOS/CIEND are Number=2 Integer tuples; split_cipos/split_ciend handle None
        # bounds (from imprecise records with "." in one position) via fillna(0).
        hg00514_vcf_df.loc[:, ['INFO_CIPOS_l', 'INFO_CIPOS_r']] = sv_utils.split_cipos(hg00514_vcf_df['INFO_CIPOS'])
        hg00514_vcf_df.loc[:, ['INFO_CIEND_l', 'INFO_CIEND_r']] = sv_utils.split_ciend(hg00514_vcf_df['INFO_CIEND'])
        hg00514_vcf_df = hg00514_vcf_df.drop(columns=['INFO_CIPOS', 'INFO_CIEND', 'INFO_SAMPLE'])

        # Use the VCF variant ID as the event identifier
        hg00514_vcf_df.loc[:, 'INFO_EVENT'] = hg00514_vcf_df['ID']

        # Convert events to BND representation then to breakpoint pairs
        hg00514_bnd_df = evts.event_to_bnd(hg00514_vcf_df, bnd2_pos='end')
        hg00514_bp_df = bnds.bnd_to_bp(
            hg00514_bnd_df,
            force_diff_cols=['INFO_CIPOS_l', 'INFO_CIPOS_r']
        )
        hg00514_bp_df.to_csv(out_path, index=False)


def preprocess_hgsvc2_val_set(raw_path, output_path, force_rerun=False, sample_name='NA12878'):
    """
    Preprocess the HGSVC2 freeze-4 integrated SV callset for benchmarking.

    The HGSVC2 VCF is a multi-sample file (35 genomes).  This function
    parses it via parse_vcf_to_df, filters for records where
    *sample_name* carries at least one ALT allele, restricts to canonical
    chromosomes and DEL/INS types, then converts events to breakpoints.

    Args:
        raw_path (str or Path): Path to variants_freeze4_sv_insdel_alt.vcf.gz.
        output_path (str or Path): Path of the breakpoints CSV to write.
        force_rerun (bool, optional): If True, regenerate even if the output exists.
        sample_name (str, optional): Sample to extract from the multi-sample VCF.
    """
    out_path = Path(output_path)

    if not out_path.exists() or force_rerun:
        hgsvc2_vcf = pysam.VariantFile(str(raw_path))
        hgsvc2_df, _ = vcfp.parse_vcf_to_df(
            hgsvc2_vcf,
            caller_params=caller_params.hgsvc2_params,
            sample_name=sample_name
        )

        # Filter to records where the sample carries at least one ALT allele
        hgsvc2_df = hgsvc2_df[
            hgsvc2_df['SAMPLE_GT'].apply(
                lambda gt: gt is not None and any(a is not None and a > 0 for a in gt)
            )
        ]

        # SVTYPE and SVLEN are per-allele tuples — unpack first element
        hgsvc2_df.loc[:, 'INFO_SVTYPE'] = hgsvc2_df['INFO_SVTYPE'].apply(
            lambda x: x[0] if isinstance(x, tuple) else x
        )
        hgsvc2_df.loc[:, 'INFO_SVLEN'] = hgsvc2_df['INFO_SVLEN'].apply(
            lambda x: abs(int(x[0])) if isinstance(x, tuple) else abs(int(x))
        )

        # Restrict to canonical chromosomes and supported SV types
        hgsvc2_df = hgsvc2_df[hgsvc2_df['CHROM'].isin(sv_utils.human_chroms)]
        hgsvc2_df = hgsvc2_df[hgsvc2_df['INFO_SVTYPE'].isin(['DEL', 'INS'])]

        hgsvc2_df.loc[:, 'INFO_CHR2'] = hgsvc2_df['CHROM']
        hgsvc2_df.loc[:, 'INFO_EVENT'] = hgsvc2_df['ID']
        hgsvc2_df.loc[:, 'INFO_CIPOS_l'] = -10
        hgsvc2_df.loc[:, 'INFO_CIPOS_r'] = 10
        hgsvc2_df.loc[:, 'INFO_CIEND_l'] = -10
        hgsvc2_df.loc[:, 'INFO_CIEND_r'] = 10

        hgsvc2_bnd_df = evts.event_to_bnd(hgsvc2_df)
        hgsvc2_bp_df = bnds.bnd_to_bp(
            hgsvc2_bnd_df,
            force_diff_cols=['INFO_CIPOS_l', 'INFO_CIPOS_r']
        )

        hgsvc2_bp_df.to_csv(out_path, index=False)


def preprocess_huddleston_val_set(raw_path, output_path, force_rerun=False):
    """
    Preprocess the Huddleston CHM13 validation set, converting to VCF-like breakpoints and saving as CSV.
    """
    out_path = Path(output_path)

    if not out_path.exists() or force_rerun:
        huddleston_vcf_df, huddleston_vcf_df_meta  = vcfp.parse_vcf_to_df(pysam.VariantFile(raw_path), caller_params=caller_params.huddleston_params, sample_name='huddleston')
        huddleston_vcf_df.loc[:, 'INFO_SVLEN'] = huddleston_vcf_df['INFO_SVLEN'].str[0]
        # Replace all strings in INFO_SVTYPE using the mapping dictionary from caller_params.huddleston_params
        mapping = caller_params.huddleston_params['svtype_mapping']
        huddleston_vcf_df.loc[:, 'INFO_SVTYPE'] = huddleston_vcf_df['INFO_SVTYPE'].map(mapping).fillna(huddleston_vcf_df['INFO_SVTYPE'])
        huddleston_vcf_df.loc[:, ['INFO_CIPOS_l', 'INFO_CIPOS_r', 'INFO_CIEND_l', 'INFO_CIEND_r']] = (-10, 10, -10, 10)
        huddleston_vcf_df.loc[:, 'ID'] = 'huddleston' + huddleston_vcf_df.index.astype(str)
        huddleston_vcf_df.loc[:, 'INFO_EVENT'] = 'huddleston' + huddleston_vcf_df.index.astype(str)
        huddleston_vcf_df.loc[:, 'INFO_CHR2'] = huddleston_vcf_df['CHROM']
        huddleston_bnd_df = evts.event_to_bnd(huddleston_vcf_df, bnd2_pos='end')
        huddleston_bp_df = bnds.bnd_to_bp(
            bnd_df=huddleston_bnd_df,
            force_diff_cols=['INFO_CIPOS_l', 'INFO_CIPOS_r']
        )

        huddleston_bp_df.to_csv(out_path, index=False)


_val_set_preprocessors = {
    'kosugi': preprocess_kosugi_val_set,
    'parikh': preprocess_kosugi_val_set,
    'hg002_val': preprocess_hg002_val_set,
    'huddleston': preprocess_huddleston_val_set,
    'chm13_val': preprocess_chm13_val_set,
    'hg00514_val': preprocess_hg00514_val_set,
    'hgsvc2': preprocess_hgsvc2_val_set,
}

VALIDATION_SET_NAMES = tuple(_val_set_preprocessors)


def preprocess_validation_set(
    name: str,
    raw_path: str | Path,
    output_path: str | Path,
    force_rerun: bool = False,
) -> Path:
    """Preprocess a published validation set into a breakpoints CSV.

    Args:
        name: Validation set name (one of VALIDATION_SET_NAMES).
        raw_path: Path to the published truth file.
        output_path: Path of the breakpoints CSV to write.
        force_rerun: Regenerate even if output_path already exists.

    Returns:
        output_path.

    Raises:
        ValueError: If no preprocessor is registered for name.
        FileNotFoundError: If raw_path does not exist.
    """
    if name not in _val_set_preprocessors:
        raise ValueError(
            f"Unknown validation set {name!r}. Available: {list(VALIDATION_SET_NAMES)}"
        )
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw input for validation set {name!r} not found at {raw_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _val_set_preprocessors[name](raw_path, output_path, force_rerun=force_rerun)
    return output_path
