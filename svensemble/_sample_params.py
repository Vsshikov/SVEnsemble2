from pathlib import Path

long_suffix = '_long'

sample_val_set_template = {
    'name': '',
    'path': Path('')
}

sample_val_sets = {
    'NA12878': {
        'name': 'kosugi',
        'path': Path(f'data/Kosugi_validation/NA12878_DGV-2016_LR-assembly.hg38.{long_suffix}.bps.csv')
    },
    'NA12878_30x': {
        'name': 'kosugi',
        'path': Path(f'data/Kosugi_validation/NA12878_DGV-2016_LR-assembly.hg38.{long_suffix}.bps.csv')
    },
    'CHM13': {
        'name': 'chm13_val',
        'path': Path(f'data/CHM13_validation/supporting_variants_for_nstd137.{long_suffix}.bps.csv')
    },
    'HG002': {
        'name': 'hg002_val',
        'path': Path(f'data/HG002_validation/HG002_validation.{long_suffix}.bps.csv')
    }
}
