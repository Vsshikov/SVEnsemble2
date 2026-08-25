from pathlib import Path

default_model_name = 'unbiased_rf'

sample_val_set_template = {
    'name': '',
    'path': Path(''),
    'raw_path': Path(''),
    'meta': {},
    'callset_params': {}
}

sample_val_sets = {
    'NA12878': {
        'name': 'kosugi',
        'path': Path('data/Kosugi_validation/NA12878_DGV-2016_LR-assembly.hg38.bps.csv'),
        'raw_path': Path('data/Kosugi_validation/NA12878_DGV-2016_LR-assembly.hg38.bed'),
        'meta': {
            'svtypes': ['DEL', 'INS', 'DUP', 'INV']
        }
    },
    'NA12878_hg19': {
        'name': 'parikh',
        'path': Path('data/Parikh_validation/NA12878_metasv_all.hg19.bps.csv'),
        'raw_path': Path('data/Parikh_validation/NA12878_metasv_all.hg19.bed'),
        'meta': {
            'svtypes': ['DEL', 'INS', 'DUP', 'INV']
        }
    },
    'NA12878_5x_test': {
        'name': 'kosugi',
        'path': Path('data/Kosugi_validation/NA12878_DGV-2016_LR-assembly.hg38.bps.csv'),
        'raw_path': Path('data/Kosugi_validation/NA12878_DGV-2016_LR-assembly.hg38.bed'),
        'meta': {
            'svtypes': ['DEL', 'INS', 'DUP', 'INV']
        }
    },
    'NA12878_8x': {
        'name': 'kosugi',
        'path': Path('data/Kosugi_validation/NA12878_DGV-2016_LR-assembly.hg38.bps.csv'),
        'raw_path': Path('data/Kosugi_validation/NA12878_DGV-2016_LR-assembly.hg38.bed'),
        'meta': {
            'svtypes': ['DEL', 'INS', 'DUP', 'INV']
        }
    },
    # 'CHM13': {
    #     'name': 'chm13_val',
    #     'path': Path('data/CHM13_validation/supporting_variants_for_nstd137.bps.csv'),
    #     'raw_path': Path('data/CHM13_validation/supporting_variants_for_nstd137.csv'),
    # },
    'CHM13': {
        'name': 'huddleston',
        'path': Path('data/CHM13_validation/CHM13_SVs.annotated.bps.csv'),
        'raw_path': Path('data/CHM13_validation/CHM13_SVs.annotated.vcf'),
        'meta': {
            'svtypes': ['DEL', 'INS', 'INV']
        }
    },
    'HG002': {
        'name': 'hg002_val',
        'path': Path('data/HG002_validation/HG002_validation.bps.csv'),
        'raw_path': Path('data/HG002_validation/HG002_validation.vcf.gz'),
        'meta': {
            'svtypes': ['DEL', 'INS']
        }
    },
    'HG002_30x': {
        'name': 'hg002_val',
        'path': Path('data/HG002_validation/HG002_validation.bps.csv'),
        'raw_path': Path('data/HG002_validation/HG002_validation.vcf.gz'),
        'meta': {
            'svtypes': ['DEL', 'INS']
        }
    },
    'HG00514': {
        'name': 'hg00514_val',
        'path': Path('data/HG00514_validation/HG00514_validation.bps.csv'),
        'raw_path': Path('data/HG00514_validation/downloads/nstd152.GRCh38.variant_call.vcf.gz'),
        'meta': {
            'svtypes': ['DEL', 'INS', 'DUP', 'INV']
        }
    },
    'NA12878_long': {
        'name': 'kosugi',
        'path': Path('data/Kosugi_validation/NA12878_DGV-2016_LR-assembly.hg38.bps.csv'),
        'raw_path': Path('data/Kosugi_validation/NA12878_DGV-2016_LR-assembly.hg38.bed'),
        'meta': {
            'svtypes': ['DEL', 'INS', 'DUP', 'INV']
        }
    },
    'NA12878_hgsvc2': {
        'name': 'hgsvc2',
        'path': Path('data/NA12878_hgsvc2/NA12878_hgsvc2.bps.csv'),
        'raw_path': Path('data/NA12878_hgsvc2/downloads/variants_freeze4_sv_insdel_alt.vcf.gz'),
        'meta': {
            'svtypes': ['DEL', 'INS']
        }
    },
    'HG008-N-D': {
        'name': '',
        'path': Path(''),
        'raw_path': Path(''),
        'meta': {
            'svtypes': []
        }
    },
    'HG008-T': {
        'name': '',
        'path': Path(''),
        'raw_path': Path(''),
        'meta': {
            'svtypes': []
        }
    },
    'COLO829-N': {
        'name': '',
        'path': Path(''),
        'raw_path': Path(''),
        'meta': {
            'svtypes': []
        }
    },
    'COLO829-T': {
        'name': '',
        'path': Path(''),
        'raw_path': Path(''),
        'meta': {
            'svtypes': []
        }
    },
}

# Ensure same structure in all val_set dicts by filling in missing keys from the template
for val_set in sample_val_sets.values():
    for key, default in sample_val_set_template.items():
        val_set.setdefault(key, default)