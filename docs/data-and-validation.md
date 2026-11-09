# Data and validation

I used two public, anonymized datasets of 4,848 and 1,740 records, and every number stays on the released scale. Two choices shape the results. One is an 80/20 split plus a second split that keeps repeated target profiles together. The other is a special rule for the all-zero target records. Back to the [README](../README.md).

> [!NOTE]
> <sub>physical vapor deposition (PVD): a chipmaking step that deposits thin metal films · target: one of the 17 released measurements to predict · profile: the set of 17 target values for one record · released scale: the anonymized units used in the data release</sub>

## What the data contain

The data come from Infineon's public
[PVD APC/SPC dataset](https://doi.org/10.5281/zenodo.16881338) and its
[associated paper](https://doi.org/10.3233/FAIA251482). Amina Mević, Andreas Laber, and Senka Krivić published it under the
[CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

My four local CSV files are unchanged copies of the release and match its MD5 checksums. Git ignores them, and
[`data/raw/README.md`](../data/raw/README.md) has the download steps and checksums.

| Process | Input columns | Released targets | Rows |
|---|---:|---:|---:|
| AlCu | 97 | 17 | 4,848 |
| WTi | 108 | 17 | 1,740 |

The release has no physical units, inverse scaling, point coordinates, timestamps, equipment identifiers, specifications, or fault labels. So all results stay on the released scale.

The mean index and the reference-profile deviation are statistical summaries I derived for engineering review. They don't measure physical thickness or physical uniformity.

## Audit

The audit found no missing or nonfinite cells, no exact duplicate X rows, no duplicate X columns, and no exact duplicate combined X/Y rows. Inputs (X) and targets (Y) pair up by row position, because the release has no join key.

Feature screening on the training rows drops `Sensor_26` and `Sensor_63` from AlCu. A single value fills at least 98% of the training rows in each. That leaves 95 AlCu inputs.

All 108 WTi inputs stay in.

## Splits

The main validation is a fixed 80/20 random split with seed 42. A second split keeps together all rows that share exactly the same 17-target profile. It is a sensitivity check, not the main result.

I do not assume that equal profiles are duplicate wafers.

Still, 1,203 AlCu rows and 93 WTi rows repeat an earlier profile, so the ordinary split can put near-copies on both sides. The grouped split removes that overlap, and its counts and overlap checks are in
[`results/audit/split_summary.csv`](../results/audit/split_summary.csv).

## All-zero target records

Each process has one record where every target is zero. These are zero-based AlCu index 2,586 and WTi index 936. I don't know what they mean.

They stay in the monitoring analysis, which looks only at the inputs. I leave them out of the main target-based models and summaries, and add them back only for explicit sensitivity checks. The effect of adding them back is in [WTi replication and zero sensitivity](wti-replication.md).

Sources: data/raw/README.md, results/audit/*.csv, data/processed/*_feature_screening.csv, data/processed/*_split_assignments.csv, src/01_data_audit.py, src/02_preprocess_and_quality.py
