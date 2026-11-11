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

1. **Parse.** Each caller's VCF is parsed into breakpoint pairs and normalised.
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
| `consensus` | Simple majority voting |

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

### With Nix

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
  --caller_vcfs manta.diploidSV.vcf.gz delly.bcf \
  --ground_truth_vcf gnomad_sv.sites.vcf.gz \
  --model unbiased_rf \
  --output_dir results/ \
  --annotate_vcfs
```

The scored calls are written to `results/svensemble2_calls.csv` and `.vcf`, and copies of the
two caller VCFs with the score added to `results/`. To write only the calls VCF, to a path of
your choice:

```bash
svensemble run \
  --ref GRCh38.fa \
  --caller_vcfs manta.diploidSV.vcf.gz delly.bcf \
  --ground_truth_vcf gnomad_sv.sites.vcf.gz \
  --model unbiased_rf \
  --output_vcf results/calls.svensemble2.vcf.gz
```

## Inputs

Inputs to `svensemble run`:

| Option | Required | Description |
|---|---|---|
| `--caller_vcfs` | yes | Caller VCF/BCF files, unmodified caller output. Give one caller several files, e.g. one per sample |
| `--callers` | no | Caller name per `--caller_vcfs` file, in the same order, see [Supported callers](#supported-callers). By default each file's caller is read from its header; `.` does that for one file |
| `--ref` | yes | Reference genome FASTA the reads were aligned to (currently only recorded in the run's provenance) |
| `--ground_truth_vcf` | one of these two | Ground truth SV sites VCF, used to label calls. Currently only gnomAD-SV is supported |
| `--ground_truth_bps` | one of these two | Ground truth already processed by `svensemble ground-truth`, to reuse across samples |
| `--model` | yes | `unbiased_rf`, `unbiased_xgb` or `consensus`, see [Models](#models) |
| `--output_dir` | one or both of these two | Directory for the calls CSV and VCF, the provenance record and any `--save_*` / `--annotate_vcfs` outputs; created if missing |
| `--output_vcf` | one or both of these two | Write the calls VCF to exactly this path (`.vcf` or `.vcf.gz`) |
| `--annotate_vcfs` | no | Also write a copy of each caller VCF with the score added, see [Output](#output). Needs `--output_dir` |
| `--save_model`, `--save_feature_matrix`, `--save_matched_bps`, `--save_ground_truth`, `--save_callsets` | no | Also write that intermediate to `--output_dir` |
| `--workdir` | no | Keep intermediates in this directory. By default they go to a temporary directory that is deleted after a successful run |

<!-- TODO: genome build(s), germline vs somatic, gnomAD-SV version and download link. -->

### Supported callers

| Caller | Name (read from the header) |
|---|---|
| GRIDSS | `gridss` |
| Manta | `manta_diploid` (`diploidSV.vcf.gz`) |
| Delly | `delly` |
| SvABA | `svaba` |
| smoove | `smoove` |

The caller is recognised by the INFO, FORMAT and FILTER fields its VCF header declares, so
`--callers` is needed only for a file whose header names no single caller.

<!-- TODO: tested caller versions. -->

### Samples

Samples are read from the caller VCFs. Each sample must be named in the `#CHROM` header line
and have its own column after `FORMAT`. Sample names must be the same in every caller VCF of
a run.

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

In the calls VCF, DEL, DUP, INS and INV are written as symbolic-allele records (`SVTYPE`,
`END`, `SVLEN`, `CIPOS`/`CIEND`), other SVs as BND pairs linked by `MATEID` and `EVENT`. INFO
is shared by all samples; `SVENSSC` is `.` for a sample without the SV. The header records
`##svensemble_version`, `##svensemble_model_id` and `##svensemble_command`.

`--annotate_vcfs` also writes a copy of each caller VCF/BCF,
`{caller}.{input name}.svensemble2.{vcf,vcf.gz,bcf}`, with the score added as FORMAT
`SVENSSC`; nothing else in the file changes. Records SVEnsemble2 dropped (e.g. single-sided
BNDs) get `.`, and inversions split into two events get the mean of their scores.
`svensemble annotate` does the same for an existing calls CSV.

<!-- TODO: how to choose a score threshold; the key output columns / INFO fields. -->
We recommend

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
api.annotate(calls, ["manta.diploidSV.vcf.gz"], "results/")
```

`api.run` takes the same options as `svensemble run` as keyword arguments: `caller_vcfs` is
one path or a list of paths, and the optional `callers` lists a name per file, with `None` for
a file whose caller is read from its header.

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
