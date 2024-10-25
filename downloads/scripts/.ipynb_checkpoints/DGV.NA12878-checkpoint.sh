#!/bin/bash

wget -P $(cat utils/workdir.txt)/downloads https://dgv.tcag.ca/dgv/docs/DGV.GS.hg38.gff3
wget -P $(cat utils/workdir.txt)/downloads https://dgv.tcag.ca/dgv/docs/GRCh38_hg38_variants_2020-02-25.txt