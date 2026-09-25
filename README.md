# Cross-Method Robustness and Aero-Propulsive Identifiability of Icing-Associated Aerodynamic Degradation

This repository is the reader-facing reproducibility package associated with the Skywalker X8 clean/iced flight-data re-analysis by Murat Metehan Türkoğlu.

## Source flight campaign

The original flight-test data, aircraft/configuration parameters, and authenticated 3-D leading-edge ice geometries are hosted by NTNU Dataverse and are **not redistributed here**:

- DOI: https://doi.org/10.18710/NNHVBP

## Repository scope

The public package separates four roles:

- `code/publication_utilities/`: compact reader-facing scripts that verify selected reported arithmetic and generate the evidence-overview visualization.
- `code/analysis_provenance/`: selected non-proprietary scientific scripts exported from the local analysis tree after extension/path filtering and a conservative secret-pattern scan.
- `derived_evidence/`: selected small machine-readable derived result/summary files; source/raw flight data are excluded.
- `media/`: publication figures and explanatory visualization media.
- `manifests/`: SHA-256 checksums and inventories for released artifacts.

The manuscript uses scientific terminology rather than internal execution labels. Original script filenames in `code/analysis_provenance/` may retain provenance-oriented names because renaming them would break traceability.

## Scientific interpretation

The central result is directional rather than magnitude-based: the held-out reference-state aerodynamic inference, the support-restricted nonlinear comparison, and the matched-pair aero-propulsive force balance retain a positive iced-minus-clean degradation direction while estimating different quantities.

The repository does **not** convert within-campaign methodological convergence into independent experimental replication, does not claim one invariant drag-penalty magnitude, and does not attribute the complete clean/iced configuration contrast uniquely to ice morphology.

## Supplementary visualization

The tagged release v1.0.1 contains `x8_icing_evidence_overview.mp4`, a short explanatory visualization of the manuscript-reported multiplicity, support, cross-method, and propulsion-sensitivity results. The video is explanatory only; it is not an additional inferential evidence layer.

## Version

Manuscript-associated public release: https://github.com/MeteMurat/x8-icing-robustness-identifiability/releases/tag/v1.0.1

File integrity: `manifests/SHA256SUMS.csv`