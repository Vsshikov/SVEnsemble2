# SVEnsemble2

<!-- TODO: badges once they exist (version, license, CI). Delete this line if none. -->

<!-- TODO: 2-3 sentences. What problem this solves, for whom, and the benefit over a
single caller. -->

SVEnsemble2 is distributed as the `svensemble` Python package, with a command-line tool
(`svensemble`) and a Python API (`svensemble.api`).

## How it works

<!-- TODO: check this section; add a figure if you have one. -->

SVEnsemble2 combines the structural variant (SV) calls of several short-read callers on one
sample into a single scored callset. No pretrained model is shipped. A model is trained on
each sample's own calls and then scores those same calls.

1. **Parse.** Each caller's VCF is parsed into breakpoint pairs and normalised.
2. **Label.** A population SV reference (gnomAD-SV) is filtered to breakpoints.
3. **Match.** Breakpoints from all callers and the reference are joined by position and
   insert-size similarity. A call that matches the reference is *labelled*. Every other
   call is *unlabelled*: it may be real or an artefact.
4. **Train.** A model is fitted to the caller-reported features (FILTER, INFO and quality
   fields) of the matched calls. Training on labelled vs unlabelled examples is a
   positive-unlabelled (PU) learning problem.
5. **Score.** Every candidate SV gets a score. The calls are written as CSV and VCF.

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

Requires Python >= 3.12.
<!-- TODO: tested Python versions and platforms; pick one official install route. -->

```bash
git clone https://github.com/Vsshikov/SVEnsemble2.git
cd SVEnsemble2
micromamba create -f environment.yml
micromamba activate svensemble2-dev
pip install --no-deps .
svensemble --version
```

## Quick start

<!-- TODO: small bundled test dataset (path or URL), expected runtime and output. -->

```bash
svensemble run \
  --sample SAMPLE1 \
  --workdir work/ \
  --ref GRCh38.fa \
  --callers manta_diploid delly \
  --caller_vcfs manta.diploidSV.vcf.gz delly.bcf \
  --ground_truth_vcf gnomad_sv.sites.vcf.gz \
  --model unbiased_rf
```

The scored calls are written to `work/predict/svensemble2_calls.csv` and `.vcf`.

## Inputs

| Input | Description |
|---|---|
| Caller VCFs | One VCF/BCF per caller, unmodified caller output |
| Reference FASTA | The reference the reads were aligned to |
| gnomAD-SV sites VCF | Population SV reference used for labelling |

<!-- TODO(you): genome build(s), germline vs somatic, gnomAD-SV version. -->

### Supported callers

| Caller | `--callers` value |
|---|---|
| GRIDSS | `gridss` |
| Manta | `manta_diploid` (`diploidSV.vcf.gz`) |
| Delly | `delly` |
| SvABA | `svaba` |
| smoove | `smoove` |

<!-- TODO: tested caller versions. -->

## Output

`svensemble2_calls.csv` and `svensemble2_calls.vcf` hold one row per merged SV with
consensus breakpoint coordinates and the model score. The score column is named
`{training_set}_{model_id}_pred`, where `training_set` is the callers sorted and joined
with `+` (e.g. `delly+manta_diploid`).

<!-- TODO: how to choose a score threshold; the key output columns / INFO fields. -->
We recommend

Intermediate files (parsed and normalised breakpoints per caller, matched breakpoints,
trained model and feature matrix) are kept under `--workdir`.

## Usage

`svensemble run` runs every step for one sample. The steps are also available as separate
subcommands, which the Nextflow pipeline uses: `preprocess-raw`, `preprocess`,
`ground-truth`, `match`, `train` and `predict`. See `svensemble <subcommand> --help`.

The same steps are available from Python. They take and return pandas DataFrames:

```python
from svensemble import api, io

raw = api.preprocess_raw("manta.diploidSV.vcf.gz", "manta_diploid", "GRCh38.fa")
bps = api.preprocess(raw, "manta_diploid", model="unbiased_rf")
matched = api.match({"manta_diploid": bps}, io.read_bps("gnomad.bps.csv"))
X, model_fit = api.train(matched, ["manta_diploid"], "unbiased_rf")
calls = api.predict(model_fit, X, matched, "unbiased_rf", callers=["manta_diploid"])
```

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
