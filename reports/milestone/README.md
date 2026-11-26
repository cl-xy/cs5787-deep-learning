# ACM project milestone

The main source is **milestone.tex**, using the unmodified ACM **acmart v2.20**
class with `sigconf,nonacm` options. Author names, Cornell NetIDs, and email
addresses intentionally remain TODOs in **authors.tex**. No ACM conference
acceptance, DOI, ISBN, or publication status is claimed.

## Build or edit

Upload this directory (including `acmart.cls`, the bibliography style,
`references.bib`, `data/`, and `figures/`) to Overleaf and select
`milestone.tex` as the main document. Alternatively, with a full TeX distribution:

```bash
python3 reports/milestone/build.py
```

The resulting short report is copied to `output/pdf/dit_milestone.pdf`.
Bibliography resolution requires the included `.bib` and `.bst` files.
The class was generated from the official `v2.20` source; see
`template/README.md` for provenance. The final verified build is three pages,
including references, in the ACM conference layout.

## Evidence and scope

- All numerical results are from `team_20261117T003651069722Z`, validation,
  20 images per method, ten Imagenette classes, 50 sampling steps, and FP32/MPS.
- Measured damping is fixed alpha=0.5 on replacement calls. The source snapshot
  also implements progressive 1/k damping; no new experimental result is
  claimed for it. On the isolated [4, 7] selection, the scales coincide.
- Real-reference FID uses 5,000 class-balanced Imagenette images. FID-20 is
  explicitly exploratory; the planned 5,000-image-per-method test is not reported
  as completed. No TPIPS or CLIP results are invented.
- `data/` includes the CSVs, environment/config, measured function sources,
  reporting metadata, and the source commit. `figures/` contains the compact
  main-text figure and additional saved-report figures for later expansion.
- The six proposal references were checked against the paper/project pages.
  Additional citations cover LPIPS, FID, small-sample bias, Clean-FID, and data.

## Optional code supplement

The main milestone excludes code listings to preserve its short page count.
`appendix_code.tex` is an optional supplement, containing upstream-code links,
provenance notes, and full listings of all notebook code cells and both tests.
To append it:

```bash
python3 reports/milestone/build.py --appendix
```

This creates a much longer PDF; use it only when a separate code appendix is
requested. `code/DiT_Layer_Looping_GCP.ipynb` is an output-free snapshot;
`code/notebook_code.py` is an archival listing, not a recommended batch runner.
Run tests in the original repository with its initialized DiT submodule.
Third-party libraries, weights and the DiT source are linked rather than
redistributed as project-authored work. Before final submission, fill in the
student contribution record and any additional original Colab/companion-code
URL that was not recorded in the repository.
