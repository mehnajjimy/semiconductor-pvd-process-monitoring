# Published source data

This folder holds unchanged local copies of the public PVD datasets the project uses. They are the published, anonymized release, not raw sensor traces straight from the fab.

Source: [Advanced Process Control and Statistical Process Control Data for Thickness Prediction of AlCu and WTi Metal Layer in Semiconductor Manufacturing](https://doi.org/10.5281/zenodo.16881338)

Authors: Amina Mević, Andreas Laber, and Senka Krivić  
License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

| File | MD5 |
|---|---|
| `X_pvd_AlCu.csv` | `9dd4bfc44ffb6a570e519000e07e9fe8` |
| `Y_pvd_AlCu.csv` | `99b845acd32ae57060e194e407c08f19` |
| `X_pvd_WTi.csv` | `cfc27ea3bdf780bcdf6dfdc5711836c1` |
| `Y_pvd_WTi.csv` | `ab6a3b425d18d6f88f0466a7c40b340a` |

Git ignores the CSV files. That keeps the repository small, and it makes the official DOI the one place to download them. Put the downloaded files in this folder before you run the scripts.
