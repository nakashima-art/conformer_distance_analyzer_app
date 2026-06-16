# Conformer Distance Analyzer

A Streamlit app for evaluating the geometrical compatibility of experimentally observed NOE correlations with conformer ensembles derived from multi-conformer SDF files or Gaussian log files.

## Features

- Read multi-conformer SDF files
- Read multiple Gaussian log files
- Group conformers by candidate stereochemistry or user-defined labels
- Define atom mappings for each candidate
- Define interchangeable proton sets such as H-2′/H-6′
- Evaluate user-defined H···H distance criteria
- Compare summed conformer populations satisfying distance thresholds
- Export candidate comparison tables and per-conformer analysis as CSV

## Supported input types

### 1. Multi-conformer SDF
Use this mode when conformer populations are already estimated by external conformer-search software such as CONFLEX.

### 2. Gaussian log files
Use this mode when each conformer is stored as a separate Gaussian output file.  
Multiple log files can be uploaded together and grouped by candidate relative configuration or any other user-defined label.

The app extracts:
- final geometry
- electronic energy when available
- Gibbs free energy when available

If free energies are available, conformer populations are estimated from Boltzmann weighting.  
If not, electronic energies are used.  
If neither is available, equal populations are assigned within each candidate group.

## Typical workflow

1. Upload SDF ensembles or Gaussian log files
2. Assign each file to a group and candidate
3. Define atom mappings
4. Define interchangeable proton groups if needed
5. Define distance criteria corresponding to observed NOEs
6. Run the analysis
7. Compare summed conformer populations across candidates
8. Export results as CSV

## Example use cases

- NOE-based stereochemical assignment
- Comparing candidate relative configurations
- Evaluating conformer ensembles against experimental distance constraints
- Screening conformers for proximity-based criteria

## Local installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/conformer-distance-analyzer.git
cd conformer-distance-analyzer
