import numpy as np
import pandas as pd
import pyranges as pr
import re
from typing import List

from . import utils






def encode_bnd_alts(alt_col):
    """Encode breakend ALT fields into numeric categories based on their pattern.
    
    Args:
        alt_col: pandas Series containing ALT fields from VCF
        
    Returns:
        pandas Series with numeric encodings:
        - 1: N[PARTNER[ (partner extending after breakpoint)
        - 2: ]PARTNER]N (partner extending before breakpoint) 
        - 3: N]PARTNER] (reverse complement of partner extending after breakpoint)
        - 4: [PARTNER[N (reverse complement of partner extending before breakpoint)
        - 0: No match
    """
    partner_pattern_right = r'\[\.?c?h?r?[0-9XY]{0,2}\:?[0-9]{0,}\['
    partner_pattern_left = r'\]\.?c?h?r?[0-9XY]{0,2}\:?[0-9]{0,}\]'
    nucl_pattern = r'[ATGCNatgcn]+'

    bnd_regex = re.compile(rf'({nucl_pattern}{partner_pattern_right})|({partner_pattern_left}{nucl_pattern})|({nucl_pattern}{partner_pattern_left})|({partner_pattern_right}{nucl_pattern})')

    encoding_df = alt_col.str.extract(bnd_regex)
    encoding_df.columns = [1, 2, 3, 4]
    encoding_df[~encoding_df.isna()] = True
    
    encodings = encoding_df.idxmax(axis=1)
    encodings[encodings.isna()] = 0
    encodings = encodings.astype(int)
    
    return encodings


def annotate_deletions(bps_df):
    """
    Annotate structural variants as deletions based on breakpoint pairs.
    
    This function identifies deletion variants by comparing the length of the deleted
    reference sequence to any inserted sequence at the breakpoints. A variant is classified
    as a deletion if:
    1. It is an intrachromosomal breakpoint pair with forward-forward orientation
    2. The length of deleted reference sequence exceeds the length of any inserted sequences
    
    Args:
        bps_df: DataFrame containing breakpoint pairs with columns:
            - CHROM_bnd1, CHROM_bnd2: Chromosomes for each breakpoint
            - POS_bnd1, POS_bnd2: Positions for each breakpoint  
            - ALT_bnd1, ALT_bnd2: ALT fields containing inserted sequences
            - bndtype: Orientation encoding of the breakpoint pair
    
    Returns:
        pd.DataFrame: DataFrame containing variant type annotations and lengths:
            - INFO_SVTYPE: 'DEL' for deletions, 'not_implemented' for other variant types
            - INFO_SVLEN: Length of the structural variant
    """
    # Initialize all variants as not implemented
    variant_types = pd.Series('not_implemented', index=bps_df.index)
    
    # Calculate length of deleted reference sequence between breakpoints
    deletion_length = bps_df['POS_bnd2'] - bps_df['POS_bnd1'] - 1

    # Filter for intrachromosomal breakpoint pairs with forward-forward orientation
    del_or_ins_mask = (
        (bps_df['bndtype'] == '12') & 
        (bps_df['CHROM_bnd1'] == bps_df['CHROM_bnd2'])
    )
    
    # Extract inserted sequences from breakpoint ALTs using regex
    # Format is: N[chr:pos[ for bnd1 and ]chr:pos]N for bnd2
    # where N is the inserted sequence
    inserted_seq_bnd1 = bps_df['ALT_bnd1'].str.extract(r'([ATGCNatgcn]+)\[')
    inserted_seq_bnd2 = bps_df['ALT_bnd2'].str.extract(r'(\][ATGCNatgcn]+$)')
    
    # Calculate total insertion length (-1 to account for anchor base)
    insertion_length_bnd1 = inserted_seq_bnd1[0].str.len() - 1  
    insertion_length_bnd2 = inserted_seq_bnd2[0].str.len() - 1
    total_insertion_length = insertion_length_bnd1 + insertion_length_bnd2


    # Mark as deletion where deletion length exceeds insertion length
    deletion_mask = del_or_ins_mask & (
        bps_df.index.isin(
            bps_df[deletion_length > total_insertion_length].index
        )
    )
    info_svlen = np.maximum(deletion_length, total_insertion_length)
    
    # Set deletion variants
    variant_types[deletion_mask] = 'DEL'
    
    # Create output dataframe
    result_df = pd.DataFrame({
        'INFO_SVTYPE': variant_types,
        'INFO_SVLEN': info_svlen
    })
    
    return result_df
