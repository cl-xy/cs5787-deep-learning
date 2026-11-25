# Block ablation studies

Implement five separately committed tasks on `feature/block-ablation-studies`:

1. Join calibration feature changes to a complete single-block removal sweep by
   original block index; validate matched manifests and plot delta versus LPIPS.
2. Plot removed block index versus paired LPIPS with nearly equal depth thirds
   (28 blocks: 10/9/9) and labeled early/middle/late regions.
3. Combine delta and removal LPIPS into a two-row block heatmap, with row-specific
   color scales and raw values so unlike units remain interpretable.
4. Add a no-loop single-block sweep and A/B/C study using existing sampling and
   LPIPS / CLIP evaluators. A chooses minimum delta; B chooses the successor of
   the maximum delta with an existing successor; C chooses each third's center.
   Permit block zero in removal-only schedules. Save frozen plans and provenance.
5. Freeze descending single-block validation LPIPS ranking (ties by block index),
   run nested top-R removals, and plot all three metrics against R including the
   intact R=0 reference. Export adjacent LPIPS slopes as exploratory evidence of
   sharp rises, not a statistically established threshold.

Keep generation in Python modules and expose opt-in notebook cells plus a CLI.
Existing sampling and strict evaluation enforce identical labels, seeds and noise.
Full DiT inference needs downloaded weights and substantial compute; validate
locally with CPU models, synthetic saved artifacts and the existing test suites.

Review focus: misaligned/missing block rows, nonfinite scores, boundary successors,
removing block zero without reuse, nested rankings and invalid R, and changed plans
or incomplete outputs during resume. Do not select from held-out test scores.
