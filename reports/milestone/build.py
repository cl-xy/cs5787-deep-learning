"""Build the ACM milestone with a local TeX distribution; no network calls."""
from pathlib import Path
import shutil
import subprocess
import sys


root = Path(__file__).resolve().parent
build = root / "build"
build.mkdir(exist_ok=True)
appendix = "--appendix" in sys.argv
job = "dit_milestone_with_code" if appendix else "dit_milestone"
source = r"\def\IncludeCodeAppendix{1}\input{milestone.tex}" if appendix else "milestone.tex"
if not shutil.which("pdflatex") or not shutil.which("bibtex"):
    raise SystemExit("Install a TeX distribution with pdflatex and bibtex, or upload this folder to Overleaf.")
command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
           "-file-line-error", "-output-directory=build", f"-jobname={job}", source]
subprocess.run(command, cwd=root, check=True)
subprocess.run(["bibtex", f"build/{job}"], cwd=root, check=True)
subprocess.run(command, cwd=root, check=True)
subprocess.run(command, cwd=root, check=True)
destination = root.parents[1] / "output/pdf" / f"{job}.pdf"
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(build / f"{job}.pdf", destination)
print(f"Built: {destination}")
