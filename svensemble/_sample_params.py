from pathlib import Path

default_model_name = 'unbiased'

sample_val_set_template = {
    'name': '',
    'path': Path('')
}

sample_val_sets = {
    'NA12878': {
        'name': 'kosugi',
        'path': Path(f'data/Kosugi_validation/NA12878_DGV-2016_LR-assembly.hg38.bps.csv')
    },
    'NA12878_5x': {
        'name': 'kosugi',
        'path': Path(f'data/Kosugi_validation/NA12878_DGV-2016_LR-assembly.hg38.bps.csv')
    },
    'NA12878_8x': {
        'name': 'kosugi',
        'path': Path(f'data/Kosugi_validation/NA12878_DGV-2016_LR-assembly.hg38.bps.csv')
    },
    # 'CHM13': {
    #     'name': 'chm13_val',
    #     'path': Path(f'data/CHM13_validation/supporting_variants_for_nstd137.bps.csv')
    # },
    'CHM13': {
        'name': 'huddleston',
        'path': Path(f'data/CHM13_validation/CHM13_SVs.annotated.bps.csv')
    },
    'HG002': {
        'name': 'hg002_val',
        'path': Path(f'data/HG002_validation/HG002_validation.bps.csv')
    },
    'HG002_30x': {
        'name': 'hg002_val',
        'path': Path(f'data/HG002_validation/HG002_validation.bps.csv')
    },
    'HG00514': {
        'name': 'hg00514_val',
        'path': Path(f'data/HG00514_validation/HG00514_validation.bps.csv')
    }
}
