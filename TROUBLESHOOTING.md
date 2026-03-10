# Troubleshooting

Common errors and how to resolve them.

---

## Preprocess fails: "VCF not found for \<callset\>"

```
FileNotFoundError: VCF not found for 'survivor' at /vast/.../sv_calling/survivor/NA12878_5x_test.survivor.vcf.
If this caller was not run, remove it from SVE_CALLSETS or the --callsets argument.
```

**Cause**: `SVE_CALLSETS` in the run config lists a callset whose VCF has not
been generated yet. This typically happens when:

- You are running an SVEnsemble2-only job (`RUN_SVENSEMBLE2=true`) without
  first running the SV caller that produces the VCF.
- A previous caller job failed and did not write its output VCF.
- A new callset was added to `SVE_CALLSETS` in `default_run.sh` but the
  corresponding caller has never been run for this sample.

**Fix** — choose one:

1. **Run the caller first.** Submit a run that includes the missing caller
   (e.g. `RUN_SURVIVOR=true`) and wait for it to complete before re-submitting
   the SVEnsemble2 run. You can combine both in a single `master.sh`
   submission — SLURM dependencies ensure the preprocess job waits for the
   caller to finish.

2. **Remove the callset from the run.** If you don't need that callset for
   this particular run, override `SVE_CALLSETS` in your run config:

   ```bash
   # data/<sample>/scripts/config/runs/my_run.sh
   SVE_CALLSETS=(gridss gridss_H manta_candidate manta_diploid delly)
   ```

3. **Check a previous caller job.** If the caller *was* submitted but the VCF
   is still missing, inspect the caller job's error log to find out why it
   failed:

   ```bash
   ls data/<sample>/logs/run/<run_dir>/<caller>/
   cat data/<sample>/logs/run/<run_dir>/<caller>/<job_id>.err
   ```

---

## Preprocess fails: "Parent callset \<name\> raw breakpoints not found"

```
FileNotFoundError: Parent callset 'gridss' raw breakpoints not found at
.../callset_bps/gridss.raw_bps.csv. Cannot derive 'gridss_H'.
```

**Cause**: A derived callset (like `gridss_H` or `sniffles_H`) depends on its
parent callset having been preprocessed first. The parent's VCF was either
missing (see above) or its raw preprocessing failed.

**Fix**: Resolve the parent callset issue first — once the parent's
`raw_bps.csv` is produced, the derived callset will be generated automatically.

---

## SVEnsemble2 job fails: "Raw breakpoints not found"

```
FileNotFoundError: Raw breakpoints not found at
.../callset_bps/gridss.raw_bps.csv. Run parse_preprocess_raw first.
```

**Cause**: The model-specific preprocessing (`parse_preprocess_common`) expects
the shared raw breakpoints to already exist in `callset_bps/`. This means the
upstream preprocess job either did not run or failed.

**Fix**: Check the preprocess job's logs at
`data/<sample>/logs/run/<run_dir>/sve_preprocess/`. If it failed, fix the
underlying issue (usually a missing VCF — see above) and re-run the pipeline.

---

## Benchmark fails: "No such file or directory" for a `.bps.csv`

**Cause**: The benchmark job could not find processed breakpoints for a callset.
This usually means the SVEnsemble2 job that should have written the file either
did not run or failed.

**Fix**: Check the SVEnsemble2 job logs at
`data/<sample>/logs/run/<run_dir>/svensemble2/`. Resolve any errors there first
— once the SVEnsemble2 jobs succeed, re-submit the benchmark.
