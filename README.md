# SVEnsemble2

<!-- TODO: badges once they exist (version, license, CI). Delete this line if none. -->

<!-- TODO: 2-3 sentences. What problem this solves, for whom, and the benefit over a
single caller. -->

SVEnsemble2 is distributed as the `svensemble` Python package, with a command-line tool
(`svensemble`) and a Python API (`svensemble.api`).

## How it works

<!-- TODO: check this section; add a figure if you have one. -->

SVEnsemble2 combines the structural variant (SV) calls of several short-read callers on one
or more samples into a single scored callset. No pretrained model is shipped. A model is
trained on the run's own calls and then scores those same calls.

1. **Parse.** Each caller's VCF is parsed into breakpoint pairs and normalised, one row per
   record and sample (see [Samples](#samples)).
2. **Label.** A population SV reference (gnomAD-SV) is filtered to breakpoints.
3. **Match.** Breakpoints from all callers and the reference are joined by position and
   insert-size similarity, within each sample. A call that matches the reference is
   *labelled*. Every other call is *unlabelled*: it may be real or an artefact.
4. **Train.** A model is fitted to the caller-reported features (FILTER, INFO and quality
   fields) of the matched calls. Training on labelled vs unlabelled examples is a
   positive-unlabelled (PU) learning problem.
5. **Score.** Every candidate SV gets a score in each sample. The calls are written as CSV
   and VCF, and optionally as copies of the input caller VCFs with the score added.

### Models

| `--model` | Description |
|---|---|
| `unbiased_rf` | Random forest trained with PU class weights |
| `unbiased_xgb` | XGBoost trained with PU class weights, with native missing-value handling |
| `consensus` | Simple majority voting. |

The `unbiased_*` models first estimate the label frequency c = p(labelled | real SV) on a
held-out 20% split, using the estimator of Elkan & Noto (2008). They then weight the classes
following the unbiased PU learning of Kiryo et al. (2017), so that the classifier estimates
whether a call is real rather than whether it is in gnomAD.

## Installation

Requires Python >= 3.12 on Linux or macOS.
<!-- TODO: tested Python versions and platforms. -->

<!-- TODO: once released, put Bioconda first, then PyPI and the BioContainers image:
  conda install -c conda-forge -c bioconda <name>
  pip install <name>
  docker pull quay.io/biocontainers/<name>:<version>
  (Singularity/Apptainer: https://depot.galaxyproject.org/singularity/<name>:<version>)
-->

## With Nix

This repository is a flake, so SVEnsemble2 can be run with

```bash
nix run github:Vsshikov/SVEnsemble2
```

### With conda

The repository includes a conda environment with all dependencies. Any conda-compatible
tool works (conda, mamba or micromamba):

```bash
git clone https://github.com/Vsshikov/SVEnsemble2.git
cd SVEnsemble2
conda env create -f environment.yml
conda activate svensemble2-dev
pip install --no-deps .
```

### With pip

In an existing Python >= 3.12 environment:

```bash
pip install git+https://github.com/Vsshikov/SVEnsemble2.git
```

### Check the installation

```bash
svensemble --version
```

## Quick start

<!-- TODO: small bundled test dataset (path or URL), expected runtime and output. -->
```bash
svensemble run \
  --ref GRCh38.fa \
  --callers manta_diploid delly \
  --caller_vcfs manta.diploidSV.vcf.gz delly.bcf \
  --ground_truth_vcf gnomad_sv.sites.vcf.gz \
  --model unbiased_rf \
  --output_dir results/ \
  --annotate_vcfs
```

The scored calls are written to `results/svensemble2_calls.csv` and `.vcf`. The two caller
VCFs must name their sample column the same (see [Samples](#samples)). To get only the calls
VCF at a path of your choice, replace `--output_dir results/` with
`--output_vcf results/calls.svensemble2.vcf.gz`.

```bash
svensemble run \
  --ref GRCh38.fa \
  --callers manta_diploid delly \
  --caller_vcfs manta.diploidSV.vcf.gz delly.bcf \
  --ground_truth_vcf gnomad_sv.sites.vcf.gz \
  --model unbiased_rf \
  --output_vcf results/calls.svensemble2.vcf.gz
```

## Inputs

Inputs to `svensemble run`:

| Option | Required | Description |
|---|---|---|
| `--callers` | yes | Caller names, see [Supported callers](#supported-callers). Repeat a name to give one caller several files, e.g. one per sample |
| `--caller_vcfs` | yes | Caller VCF/BCF files, unmodified caller output, in the same order as `--callers` |
| `--ref` | yes | Reference genome FASTA the reads were aligned to (currently only recorded in the run's provenance) |
| `--ground_truth_vcf` | one of these two | Ground truth SV sites VCF, used to label calls. Currently only gnomAD-SV is supported |
| `--ground_truth_bps` | one of these two | Ground truth already processed by `svensemble ground-truth`, to reuse across samples |
| `--model` | yes | `unbiased_rf`, `unbiased_xgb` or `consensus`, see [Models](#models) |
| `--output_dir` | one or both of these two | Directory for the calls CSV and VCF, the provenance record and any `--save_*` / `--annotate_vcfs` outputs; created if missing |
| `--output_vcf` | one or both of these two | Write the calls VCF to exactly this path (`.vcf` or `.vcf.gz`). With it alone, nothing else is written |
| `--annotate_vcfs` | no | Also write a copy of each caller VCF with the score added, see [Output](#output). Needs `--output_dir` |
| `--save_model`, `--save_feature_matrix`, `--save_matched_bps`, `--save_ground_truth`, `--save_callsets` | no | Also write that intermediate to `--output_dir` |
| `--workdir` | no | Keep intermediates in this directory. By default they go to a temporary directory that is deleted after a successful run |

<!-- TODO: genome build(s), germline vs somatic, gnomAD-SV version and download link. -->

### Supported callers

| Caller | `--callers` value |
|---|---|
| GRIDSS | `gridss` |
| Manta | `manta_diploid` (`diploidSV.vcf.gz`) |
| Delly | `delly` |
| SvABA | `svaba` |
| smoove | `smoove` |

<!-- TODO: tested caller versions. -->

### Samples

Every caller VCF is read per sample: each record gives one row per sample column, keyed by
the sample's name in the `#CHROM` line, so the samples are taken from the VCF headers. One run can score one sample or several, from
joint-called multi-sample VCFs or from one file per sample.

Requirements:

- **Header.** Every caller VCF has a `FORMAT` column and at least one sample column
  (`#CHROM ... INFO FORMAT S1 [S2 ...]`). Sites-only caller VCFs are rejected.
- **Same names.** All callers in a run carry the same set of sample names, compared exactly.
  Callers often name the same sample differently (in our test data Delly writes
  `NA12878.sorted` and Manta `SAMPLE1`), so rename the columns first, e.g. with
  `bcftools reheader --samples`. The Nextflow pipeline does this in its
  `FORCESAMPLENAMES` step.
- **One file per sample.** When a caller is given several files, each sample appears in only
  one of them.
- **Reserved names.** Sample names that read back as missing values (`NA`, `nan`, `None`,
  `null`, ...) and `*` are rejected.
- **FORMAT fields.** The per-sample features of each caller (e.g. Delly `GT`, `GQ`, `RC`,
  `DR`, `DV`; Manta `FT`, `GQ`, `PR`, `SR`) are read from the row's own sample. A field
  missing from a record is a missing value. duphold's `DHFC`, `DHBFC` and `DHFFC` FORMAT
  fields and its `GCF` INFO field are also read per sample when the header defines them.
- **Ground truth.** The ground truth is a sample-less sites set (gnomAD-SV) and applies to
  every sample. It must not have a sample column.

Breakpoints are paired, matched and turned into features within each sample. One model is
trained on the rows of all samples, and every SV gets a score in each sample.

## Output

With `--output_dir`:

| File | Content |
|---|---|
| `svensemble2_calls.csv` | One row per merged SV and sample: consensus breakpoint coordinates, each caller's event ID (`<caller>_INFO_EVENT`), the sample (`FORMAT_SAMPLE_ID`) and the model score |
| `svensemble2_calls.vcf` | One record per merged SV, with the score of each sample in FORMAT `SVENSSC` |
| `svensemble_run.json` | Provenance record: command, parameters, input paths/sizes/dates, output checksums, model config, software versions, timing |

With `--output_vcf PATH`, the calls VCF is written to `PATH` instead. If `--output_dir` is
not also given, it is the only file written.

The score column is named `{training_set}_{model_id}_pred`, where `training_set` is the
callers sorted and joined with `+` (e.g. `delly+manta_diploid`).

The calls VCF:

- **Records.** DEL, DUP, INS and INV are symbolic-allele records (`SVTYPE`, `END`, `SVLEN`,
  and `CIPOS`/`CIEND` when known). Other SVs are BND pairs linked by `MATEID` and `EVENT`.
- **INFO and FORMAT.** INFO is shared by all samples. `SVENSSC` is per sample, and is missing
  (`.`) for a sample without that SV.
- **Header.** It records `##svensemble_version`, `##svensemble_model_id` and
  `##svensemble_command`.

`--annotate_vcfs` also writes, to `--output_dir`, a copy of each input caller VCF/BCF named
`{caller}.{input file name without extension}.svensemble2.{vcf,vcf.gz,bcf}` (same format
as the input):

- **What changes.** A per-sample FORMAT field `SVENSSC` holds the record's score in that
  sample. Records are edited as text, or as bytes for BCF, never re-encoded, so nothing else
  in the file changes.
- **Records without a score** get `.`. These are records the pipeline dropped, e.g.
  single-sided BNDs.
- **Inversions** that SVEnsemble2 splits into two events get the mean of the two scores.
- **Separately.** The same is available on its own with
  `svensemble annotate --calls svensemble2_calls.csv --callers ... --caller_vcfs ... --output_dir ...`.

<!-- TODO: how to choose a score threshold; the key output columns / INFO fields. -->
We recommend

Intermediate files (parsed and normalised breakpoints per caller, matched breakpoints,
trained model and feature matrix) are kept under `--workdir` when it is given, and can be
copied to `--output_dir` with the `--save_*` flags.

## Usage

`svensemble run` runs every step for one sample. The steps are also available as separate
subcommands, which the Nextflow pipeline uses: `preprocess-raw`, `preprocess`,
`ground-truth`, `match`, `train`, `predict` and `annotate`. `benchmark-callset`,
`validation-set` and `shap` evaluate and explain the results.

Every command prints its options and defaults with `--help` (or `-h`):

```bash
svensemble --help          # list subcommands
svensemble run --help      # options for one subcommand
```

The same steps are available from Python. They take and return pandas DataFrames:

```python
from svensemble import api, io

raw = api.preprocess_raw(io.read_vcf("manta.diploidSV.vcf.gz", "manta_diploid"), "manta_diploid")
bps = api.preprocess(raw, "manta_diploid", model="unbiased_rf")
matched = api.match({"manta_diploid": bps}, io.read_bps("gnomad.bps.csv"))
X, model_fit = api.train(matched, ["manta_diploid"], "unbiased_rf")
calls = api.predict(model_fit, X, matched, "unbiased_rf", callers=["manta_diploid"])

# Copies of the caller VCFs with the score as FORMAT SVENSSC (caller -> path(s))
api.annotate(calls, {"manta_diploid": "manta.diploidSV.vcf.gz"}, "results/")
```

`api.run` takes the same options as `svensemble run` as keyword arguments, with
`caller_vcfs` as a mapping from caller name to one path or a list of paths.

## Citation

<!-- TODO: paper/preprint + DOI, and a CITATION.cff file. -->

## References

- Elkan C, Noto K. Learning classifiers from only positive and unlabeled data.
  *Proc. 14th ACM SIGKDD* (2008). doi:10.1145/1401890.1401920
- Kiryo R, Niu G, du Plessis MC, Sugiyama M. Positive-unlabeled learning with non-negative
  risk estimator. *Advances in Neural Information Processing Systems* 30 (2017).
  arXiv:1703.00593

## License

<!-- TODO: add a LICENSE file; there is none yet. -->

## Contact

<!-- TODO: issue tracker; maintainer and affiliation. -->
