# GitHub Release Checklist

This project directory is prepared to become a normal GitHub repository.

Large artifacts are intentionally ignored by `.gitignore`:

- `dataset/`
- `dataset_hl.zarr/`
- `dataset_ll.zarr/`
- `checkpoint_*.ckpt`
- generated videos, trajectories, run directories, and baseline results

These files can be restored with:

```bash
bash scripts/setup_artifacts.sh
```

## Local Commit

On this server, Git object writes failed directly on the NFS-backed shared
directory. The current repository therefore uses a shared worktree with a local
Git directory:

```text
worktree: /home/gaoj/share4/_piano/pianomime
gitdir:   /home/gaoj/piano_scratch/pianomime_gitdir
```

That setup has already produced local commits, including the baseline fork and
the completed baseline handoff documentation. Check the current hash with
`git log --oneline -1`.

If recreating from scratch, prefer:

```bash
cd /home/gaoj/share4/_piano/pianomime
git init --separate-git-dir=/home/gaoj/piano_scratch/pianomime_gitdir
git branch -M main
git add .
git status --short
git commit -m "Prepare course PianoMime baseline fork"
```

## Push

The GitHub CLI is not available on this server, and the GitHub connector cannot
create a brand-new repository by itself. Create an empty GitHub repository
manually, then push:

```bash
git remote add origin git@github.com:<YOUR_USER_OR_ORG>/<REPO_NAME>.git
git push -u origin main
```

If HTTPS auth is preferred:

```bash
git remote add origin https://github.com/<YOUR_USER_OR_ORG>/<REPO_NAME>.git
git push -u origin main
```

Do not force-add ignored artifacts unless the team intentionally decides to use
Git LFS for checkpoints/datasets.
