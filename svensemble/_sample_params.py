from pathlib import Path

default_model_name = 'unbiased'

sample_val_set_template = {
    'name': '',
    'path': Path(''),
    'meta': {},
    'callset_params': {}
}

sample_val_sets = {
    'NA12878': {
        'name': 'kosugi',
        'path': Path(f'data/Kosugi_validation/NA12878_DGV-2016_LR-assembly.hg38.bps.csv'),
        'meta': {
            'svtypes': ['DEL', 'INS', 'DUP', 'INV']
        }
    },
    'NA12878_hg19': {
        'name': 'parikh',
        'path': Path(f'/home/users/allstaff/shikov.v/projects/Papenfuss_SVEnsemble2/data/Parikh_validation/NA12878_metasv_all.hg19.bps.csv'),
        'meta': {
            'svtypes': ['DEL', 'INS', 'DUP', 'INV']
        }
    },
    'NA12878_5x_test': {
        'name': 'kosugi',
        'path': Path(f'data/Kosugi_validation/NA12878_DGV-2016_LR-assembly.hg38.bps.csv'),
        'meta': {
            'svtypes': ['DEL', 'INS', 'DUP', 'INV']
        }
    },
    'NA12878_8x': {
        'name': 'kosugi',
        'path': Path(f'data/Kosugi_validation/NA12878_DGV-2016_LR-assembly.hg38.bps.csv'),
        'meta': {
            'svtypes': ['DEL', 'INS', 'DUP', 'INV']
        }
    },
    # 'CHM13': {
    #     'name': 'chm13_val',
    #     'path': Path(f'data/CHM13_validation/supporting_variants_for_nstd137.bps.csv')
    # },
    'CHM13': {
        'name': 'huddleston',
        'path': Path(f'data/CHM13_validation/CHM13_SVs.annotated.bps.csv'),
        'meta': {
            'svtypes': ['DEL', 'INS', 'INV']
        }
    },
    'HG002': {
        'name': 'hg002_val',
        'path': Path(f'data/HG002_validation/HG002_validation.bps.csv'),
        'meta': {
            'svtypes': ['DEL', 'INS']
        }
    },
    'HG002_30x': {
        'name': 'hg002_val',
        'path': Path(f'data/HG002_validation/HG002_validation.bps.csv'),
        'meta': {
            'svtypes': ['DEL', 'INS']
        }
    },
    'HG00514': {
        'name': 'hg00514_val',
        'path': Path(f'data/HG00514_validation/HG00514_validation.bps.csv'),
        'meta': {
            'svtypes': ['DEL', 'INS', 'DUP', 'INV']
        }
    },
    'NA12878_long': {
        'name': 'kosugi',
        'path': Path(f'data/Kosugi_validation/NA12878_DGV-2016_LR-assembly.hg38.bps.csv'),
        'meta': {
            'svtypes': ['DEL', 'INS', 'DUP', 'INV']
        }
    },
}

# Ensure same structure in all val_set dicts by filling in missing keys from the template
for val_set in sample_val_sets.values():
    for key, default in sample_val_set_template.items():
        val_set.setdefault(key, default)