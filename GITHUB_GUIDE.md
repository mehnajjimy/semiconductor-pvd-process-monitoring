# Git and GitHub Publishing Guide

**Current project status:** pre-audit v0.9. Do not publish it yet. First run the
independent review in `report/independent_audit_prompt.md`, correct every
confirmed issue, rerun the workflow, and change the public-facing status to
v1.0. The corrected, independently audited project should be the first public
release.

This guide is intentionally manual and beginner friendly. Run one command at a
time, inspect its output, and stop if the result differs from the stated
expectation. Nothing in the analysis scripts initializes Git, creates a GitHub
repository, or pushes files.

The four released CSV files should not be published in this repository. They
are ignored by `.gitignore`. The tracked `data/raw/README.md` instead records
the official Zenodo DOI, license, filenames, and checksums.

GitHub recommends creating an empty remote repository when pushing an existing
local project. The official instructions are:
[Adding locally hosted code to GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).

## 1. Confirm the Project Folder

Open Terminal and run:

```bash
cd ~/Documents/semiconductor-pvd-process-monitoring
```

`cd` changes the Terminal's working directory. It does not change project
files.

```bash
pwd
```

The printed path should end with
`/Documents/semiconductor-pvd-process-monitoring`. Stop if it does not.

```bash
ls
```

Confirm that the listing includes `README.md`, `requirements.txt`, `src`,
`data`, `figures`, `results`, and `report`.

## 2. Inspect or Initialize Git

Always inspect first:

```bash
git status
```

If this shows normal repository status, the project is already a Git
repository. Do **not** run `git init` again.

If it says “not a git repository,” initialize it once:

```bash
git init
```

Then confirm:

```bash
git status
```

`git init` creates local Git metadata only. It does not upload anything.

## 3. Inspect the Remote and Commit Identity

Check whether a remote is already configured:

```bash
git remote -v
```

No output means no remote is configured. If an `origin` appears, inspect the
username and repository name carefully. A URL containing an old username must
be replaced before any push.

Check the identity Git would use for this repository:

```bash
git config user.name
```

```bash
git config user.email
```

If either value is missing or wrong, set it for this repository only. Replace
the placeholders:

```bash
git config user.name "YOUR NAME"
```

```bash
git config user.email "YOUR EMAIL"
```

These commands intentionally omit `--global`, so they do not alter unrelated
repositories. Use an email address that matches your GitHub account or your
GitHub-provided private no-reply address if you want the commit attributed to
your account.

## 4. Confirm the Raw Data Are Ignored

Run all four checks:

```bash
git check-ignore data/raw/X_pvd_AlCu.csv
```

```bash
git check-ignore data/raw/Y_pvd_AlCu.csv
```

```bash
git check-ignore data/raw/X_pvd_WTi.csv
```

```bash
git check-ignore data/raw/Y_pvd_WTi.csv
```

Each command should print the path it was given. If any command prints nothing,
stop and inspect `.gitignore` before staging or pushing.

## 5. Create an Empty GitHub Repository

Only do this after the independent audit is complete and the local project is
ready to be labeled v1.0.

1. Sign in to [GitHub](https://github.com/).
2. Choose **New repository**.
3. Name it `semiconductor-pvd-process-monitoring`.
4. Add a short description, such as “Retrospective multivariate process-state
   monitoring and released-scale virtual metrology for public PVD data.”
5. Choose public or private.
6. Do not add a README, `.gitignore`, or license on GitHub. The local project
   already contains the intended files.
7. Create the empty repository.

Never paste a password, token, or other credential into a project file.

## 6. Add or Correct the GitHub Remote

First inspect again:

```bash
git remote -v
```

Replace `NEW-USERNAME` below with your current GitHub username.

If there is **no** remote named `origin`, add it:

```bash
git remote add origin https://github.com/NEW-USERNAME/semiconductor-pvd-process-monitoring.git
```

If `origin` already exists but contains the wrong or old username, replace its
URL:

```bash
git remote set-url origin https://github.com/NEW-USERNAME/semiconductor-pvd-process-monitoring.git
```

Use only the command that matches your situation. Do not run both.

Verify the result:

```bash
git remote -v
```

Both fetch and push URLs should point to your current username and the exact
`semiconductor-pvd-process-monitoring` repository. If they do not, stop before
pushing.

Name the local branch `main`:

```bash
git branch -M main
```

## 7. Review and Commit the Audited Project

Inspect all changes and untracked files:

```bash
git status
```

Stage the nonignored project:

```bash
git add .
```

Review the staged paths:

```bash
git status
```

The four `data/raw/*.csv` files must not appear. If they do, stop and unstage
them before continuing.

Review a compact staged summary:

```bash
git diff --cached --stat
```

If the project has no earlier commit, record the audited public snapshot:

```bash
git commit -m "Initial audited semiconductor PVD project v1.0"
```

If the repository already has local commits, use a truthful message describing
the final audit corrections instead. Do not rewrite or delete history merely to
make it look like a first commit.

## 8. Perform the Final Pre-Push Check

Run these four checks immediately before the first push:

```bash
git status
```

```bash
git remote -v
```

```bash
git config user.name
```

```bash
git config user.email
```

Confirm that:

- the working tree contains only the intended committed state;
- `origin` uses the current GitHub username and correct repository;
- the displayed name and email are the intended commit identity;
- the README and reports say v1.0, not pre-audit v0.9;
- the independent audit has no unresolved critical or important findings.

## 9. Push the Project

This is the first command in the guide that uploads commits:

```bash
git push -u origin main
```

The `-u` option links local `main` to `origin/main`. GitHub may open a browser
sign-in or use a credential helper. Follow GitHub's authentication flow and
never save a credential in this repository.

After a successful push, create the first public release tag:

```bash
git tag -a v1.0 -m "First independently audited public release"
```

Inspect the tag before uploading it:

```bash
git show v1.0
```

Then push the tag:

```bash
git push origin v1.0
```

## 10. Verify GitHub in the Browser

Open the repository page and confirm:

- the README renders and identifies the audited release as v1.0;
- all six numbered scripts are present;
- all expected figures and result tables are present;
- the technical report, career materials, independent-audit prompt, and this
  guide are present;
- `data/raw/README.md` is present;
- none of the four raw CSV files is present;
- no `.venv`, cache, credential, or unrelated file is present;
- the `v1.0` tag points to the intended audited commit.

Finally run:

```bash
git status
```

The expected result is a clean working tree on `main` that is up to date with
`origin/main`.

## 11. Publish Later Changes Safely

For later updates, inspect before staging:

```bash
git status
```

```bash
git diff
```

Stage intended changes:

```bash
git add .
```

Review exactly what will be committed:

```bash
git diff --cached
```

Commit with a truthful description:

```bash
git commit -m "Clarify grouped validation results"
```

Recheck the destination:

```bash
git remote -v
```

Push:

```bash
git push
```

Finish with:

```bash
git status
```

Do not use force-push, destructive reset, or file-deletion commands to work
around an error. If a command produces an unfamiliar result, stop at that
command and inspect it before proceeding. Local commits and files remain
available, so no guess is needed.
