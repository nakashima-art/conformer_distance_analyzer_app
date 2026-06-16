# Conformer Distance Analyzer

**Version 1.2.0**

Conformer Distance Analyzer is a Streamlit application for comparing conformer ensembles using user-defined H···H distance criteria and conformer populations. It supports multi-conformer SDF files and Gaussian log files.

## Main features

- Load one or more multi-conformer SDF files
- Load Gaussian log files grouped by candidate
- Select atoms directly in an interactive 3D molecular viewer
- Register candidate-specific proton labels and atom mappings
- Copy mappings from another candidate
  - labels and atom numbers
  - labels only
  - apply the current mapping to all compatible candidates
- Define H···H distance criteria
- Compare the summed populations of conformers satisfying the selected distance range
- Save the complete analysis as a project file and reopen it later
- Switch between English and Japanese interfaces

## Population handling

### SDF files

The application uses conformer populations stored in the SDF file when available. If populations are not present but conformational-search energies are available, Boltzmann populations are calculated from those energies. If neither populations nor usable energies are found, equal populations are assigned.

### Gaussian log files

For Gaussian log files, the energy source used to calculate conformer populations can be selected:

- **Automatic**: Gibbs free energy is used only when it is available for every conformer; otherwise, electronic energy is used.
- **Electronic energy**: the final SCF electronic energy is used.
- **Gibbs free energy**: Gibbs free energy is used. Every conformer must contain frequency-calculation results with a Gibbs free energy value.

The default setting is **Electronic energy**.

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## File structure

```text
conformer_distance_analyzer_click/
├── app.py
├── requirements.txt
├── README.md
└── atom_picker_component/
    └── index.html
```

The `atom_picker_component` directory must remain in the same directory as `app.py`.

## Streamlit Community Cloud

Upload the complete folder structure to the GitHub repository connected to Streamlit Community Cloud. After replacing files, reboot the app from the Streamlit management screen.

## Project files

A saved project contains the loaded conformer coordinates, energies, populations, candidate-specific atom mappings, distance criteria, analysis settings, and the most recent results. Project files can be downloaded to the local computer and reopened in the application.

## Author

Developed by **Ken-ichi Nakashima**  
Laboratory of Medicinal Resources, School of Pharmacy, Aichi Gakuin University
