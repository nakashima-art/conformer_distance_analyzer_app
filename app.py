import io
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    from rdkit import Chem
except Exception:
    Chem = None


st.set_page_config(page_title="Conformer Distance Analyzer", layout="wide")


# ==============================
# Data models
# ==============================
@dataclass
class ConformerRecord:
    source_type: str  # sdf | gaussian_log
    source_name: str
    group: str
    candidate: str
    conformer_id: str
    atoms: List[str]
    coords: np.ndarray  # shape (n_atoms, 3)
    energy: Optional[float] = None
    free_energy: Optional[float] = None
    population: Optional[float] = None
    metadata: Dict[str, str] = field(default_factory=dict)


# ==============================
# Gaussian log parsing
# ==============================
def _parse_gaussian_standard_orientation(text: str) -> Tuple[List[int], np.ndarray]:
    lines = text.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        if "Standard orientation:" in lines[i] or "Input orientation:" in lines[i]:
            j = i + 5
            atoms = []
            coords = []
            while j < len(lines):
                line = lines[j].strip()
                if line.startswith("----"):
                    break
                parts = line.split()
                if len(parts) >= 6:
                    atoms.append(int(parts[1]))
                    coords.append([float(parts[3]), float(parts[4]), float(parts[5])])
                j += 1
            if atoms and coords:
                blocks.append((atoms, np.array(coords, dtype=float)))
            i = j
        i += 1
    if not blocks:
        raise ValueError("No geometry block found in Gaussian log.")
    return blocks[-1]


def _parse_gaussian_energies(text: str) -> Tuple[Optional[float], Optional[float]]:
    energy = None
    free_energy = None

    scf_matches = re.findall(r"SCF Done:\s+E\([^)]*\)\s+=\s+([-]?[0-9]+\.[0-9]+)", text)
    if scf_matches:
        energy = float(scf_matches[-1])

    g_matches = re.findall(
        r"Sum of electronic and thermal Free Energies=\s+([-]?[0-9]+\.[0-9]+)",
        text,
    )
    if g_matches:
        free_energy = float(g_matches[-1])

    return energy, free_energy


def _atomic_number_to_symbol(z: int) -> str:
    periodic = {
        1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P",
        16: "S", 17: "Cl", 35: "Br", 53: "I"
    }
    return periodic.get(z, f"Z{z}")


def parse_gaussian_log(uploaded_file, group: str, candidate: str) -> ConformerRecord:
    text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    atom_nums, coords = _parse_gaussian_standard_orientation(text)
    energy, free_energy = _parse_gaussian_energies(text)
    atoms = [_atomic_number_to_symbol(z) for z in atom_nums]
    return ConformerRecord(
        source_type="gaussian_log",
        source_name=uploaded_file.name,
        group=group,
        candidate=candidate,
        conformer_id=uploaded_file.name,
        atoms=atoms,
        coords=coords,
        energy=energy,
        free_energy=free_energy,
        metadata={"file_name": uploaded_file.name},
    )


# ==============================
# SDF parsing
# ==============================
def parse_sdf_ensemble(uploaded_file, group: str, candidate: str) -> List[ConformerRecord]:
    if Chem is None:
        raise ImportError("RDKit is required to parse SDF files. Add rdkit to your environment.")

    content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    supplier = Chem.ForwardSDMolSupplier(io.BytesIO(uploaded_file.getvalue()), removeHs=False)

    records: List[ConformerRecord] = []
    for idx, mol in enumerate(supplier):
        if mol is None:
            continue
        conf = mol.GetConformer()
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coords = np.array([
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
        ], dtype=float)

        props = mol.GetPropsAsDict()
        population = None
        energy = None
        free_energy = None
        for key, value in props.items():
            k = str(key).lower()
            try:
                if population is None and ("pop" in k or k == "p"):
                    population = float(value)
                if free_energy is None and ("free" in k or "gibbs" in k):
                    free_energy = float(value)
                if energy is None and (k == "energy" or "mmff" in k or "conflex" in k):
                    energy = float(value)
            except Exception:
                pass

        records.append(
            ConformerRecord(
                source_type="sdf",
                source_name=uploaded_file.name,
                group=group,
                candidate=candidate,
                conformer_id=f"{uploaded_file.name}::conf_{idx+1}",
                atoms=atoms,
                coords=coords,
                energy=energy,
                free_energy=free_energy,
                population=population,
                metadata={k: str(v) for k, v in props.items()},
            )
        )
    return records


# ==============================
# Population utilities
# ==============================
def hartree_to_kcal(delta_hartree: float) -> float:
    return delta_hartree * 627.509474


def compute_boltzmann_populations(records: List[ConformerRecord], temperature: float = 298.15) -> List[ConformerRecord]:
    grouped: Dict[Tuple[str, str], List[ConformerRecord]] = {}
    for rec in records:
        grouped.setdefault((rec.group, rec.candidate), []).append(rec)

    r_kcal = 0.0019872041
    for _, recs in grouped.items():
        values = []
        mode = None
        for rec in recs:
            if rec.free_energy is not None:
                values.append(rec.free_energy)
                mode = "free_energy"
            elif rec.energy is not None:
                values.append(rec.energy)
                mode = "energy"
            else:
                values.append(None)

        if any(v is None for v in values):
            total = len(recs)
            for rec in recs:
                rec.population = 100.0 / total
            continue

        min_e = min(values)
        deltas = [hartree_to_kcal(v - min_e) for v in values]
        weights = [math.exp(-d / (r_kcal * temperature)) for d in deltas]
        denom = sum(weights)
        for rec, w in zip(recs, weights):
            rec.population = 100.0 * w / denom
            rec.metadata["population_mode"] = mode or "unknown"
    return records


# ==============================
# Distance analysis
# ==============================
def atom_index_from_user_number(user_number: int) -> int:
    # User enters 1-based atom numbers
    return user_number - 1


def pair_distance(coords: np.ndarray, idx_a: int, idx_b: int) -> float:
    return float(np.linalg.norm(coords[idx_a] - coords[idx_b]))


def min_group_distance(coords: np.ndarray, idx_a: int, idx_group: List[int]) -> float:
    return float(min(pair_distance(coords, idx_a, j) for j in idx_group))


# ==============================
# App state helpers
# ==============================
def init_state():
    defaults = {
        "records": [],
        "loaded": False,
        "mapping_rows": [],
        "criteria_rows": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

st.title("Conformer Distance Analyzer")
st.caption("Streamlit tool for comparing conformer ensembles from SDF or Gaussian log files using user-defined H···H distance criteria and conformer populations.")


# ==============================
# Sidebar configuration
# ==============================
st.sidebar.header("Analysis settings")
col_thr1, col_thr2 = st.sidebar.columns(2)
threshold_1 = col_thr1.number_input("Threshold 1 (Å)", value=3.0, step=0.1, format="%.2f")
threshold_2 = col_thr2.number_input("Threshold 2 (Å)", value=3.5, step=0.1, format="%.2f")
thresholds = sorted({float(threshold_1), float(threshold_2)})

temperature = st.sidebar.number_input("Boltzmann temperature (K)", value=298.15, step=1.0)


# ==============================
# Step 1: input mode
# ==============================
st.header("1. Input source")
input_mode = st.radio("Choose input type", ["SDF ensemble", "Gaussian log files"], horizontal=True)

if input_mode == "SDF ensemble":
    st.subheader("Upload SDF ensembles")
    st.write("Upload one or more multi-conformer SDF files. Each file is treated as one candidate ensemble.")
    sdf_files = st.file_uploader("SDF files", type=["sdf"], accept_multiple_files=True)
    default_group = st.text_input("Default group name", value="default_group")

    if sdf_files:
        mapping_df = pd.DataFrame(
            [{"file_name": f.name, "group": default_group, "candidate": re.sub(r"\.sdf$", "", f.name, flags=re.I)} for f in sdf_files]
        )
        edited = st.data_editor(mapping_df, num_rows="dynamic", use_container_width=True, key="sdf_map")
        if st.button("Load SDF ensembles"):
            all_records = []
            for f in sdf_files:
                row = edited.loc[edited["file_name"] == f.name].iloc[0]
                recs = parse_sdf_ensemble(f, str(row["group"]), str(row["candidate"]))
                all_records.extend(recs)
            all_records = compute_boltzmann_populations(all_records, temperature=temperature)
            st.session_state.records = all_records
            st.session_state.loaded = True
            st.success(f"Loaded {len(all_records)} conformers from {len(sdf_files)} SDF file(s).")

else:
    st.subheader("Upload Gaussian log files")
    st.write("Upload multiple Gaussian log files and assign each file to a group and candidate.")
    log_files = st.file_uploader("Gaussian log files", type=["log", "out"], accept_multiple_files=True)
    default_group = st.text_input("Default group name", value="default_group_logs")

    if log_files:
        mapping_df = pd.DataFrame(
            [{"file_name": f.name, "group": default_group, "candidate": "candidate_1"} for f in log_files]
        )
        edited = st.data_editor(mapping_df, num_rows="dynamic", use_container_width=True, key="log_map")
        if st.button("Load Gaussian log files"):
            all_records = []
            for f in log_files:
                row = edited.loc[edited["file_name"] == f.name].iloc[0]
                rec = parse_gaussian_log(f, str(row["group"]), str(row["candidate"]))
                all_records.append(rec)
            all_records = compute_boltzmann_populations(all_records, temperature=temperature)
            st.session_state.records = all_records
            st.session_state.loaded = True
            st.success(f"Loaded {len(all_records)} conformers from {len(log_files)} Gaussian log file(s).")


# ==============================
# Loaded summary
# ==============================
if st.session_state.loaded and st.session_state.records:
    st.header("2. Loaded conformers")
    recs = st.session_state.records
    summary_df = pd.DataFrame([
        {
            "group": r.group,
            "candidate": r.candidate,
            "source": r.source_name,
            "conformer_id": r.conformer_id,
            "n_atoms": len(r.atoms),
            "energy": r.energy,
            "free_energy": r.free_energy,
            "population": r.population,
        }
        for r in recs
    ])
    st.dataframe(summary_df, use_container_width=True, height=280)

    # ==============================
    # Step 3: atom mapping
    # ==============================
    st.header("3. Atom mapping")
    unique_candidates = sorted({(r.group, r.candidate) for r in recs})
    candidate_options = [f"{g} :: {c}" for g, c in unique_candidates]
    selected_candidate = st.selectbox("Candidate for atom mapping", candidate_options)
    sel_group, sel_cand = selected_candidate.split(" :: ", 1)
    candidate_records = [r for r in recs if r.group == sel_group and r.candidate == sel_cand]
    example = candidate_records[0]

    st.write("Enter atom labels and 1-based atom numbers. Use comma-separated values for interchangeable proton groups.")
    atom_table = pd.DataFrame({
        "label": ["H3", "H4", "H3pp", "H4pp", "H2p_or_H6p", "H2ppp_or_H6ppp", "H5pp"],
        "atom_numbers": ["", "", "", "", "", "", ""],
    })
    atom_table = st.data_editor(atom_table, num_rows="dynamic", use_container_width=True, key=f"mapping_{selected_candidate}")

    mapping: Dict[str, List[int]] = {}
    for _, row in atom_table.iterrows():
        label = str(row["label"]).strip()
        numbers_raw = str(row["atom_numbers"]).strip()
        if label and numbers_raw:
            nums = [atom_index_from_user_number(int(x.strip())) for x in numbers_raw.split(",") if x.strip()]
            mapping[label] = nums

    with st.expander("Show atom list for selected candidate"):
        atom_df = pd.DataFrame({
            "atom_number_1_based": list(range(1, len(example.atoms) + 1)),
            "element": example.atoms,
        })
        st.dataframe(atom_df, use_container_width=True, height=240)

    # ==============================
    # Step 4: criteria
    # ==============================
    st.header("4. Distance criteria")
    st.write("Define criteria as either pair distances or group-min distances.")
    criteria_df = pd.DataFrame({
        "label": ["NOE1", "NOE2"],
        "atom_a_label": ["H3", "H3"],
        "atom_b_label_or_group": ["H2ppp_or_H6ppp", "H2p_or_H6p"],
    })
    criteria_df = st.data_editor(criteria_df, num_rows="dynamic", use_container_width=True, key=f"criteria_{selected_candidate}")

    # ==============================
    # Step 5: analysis
    # ==============================
    st.header("5. Analysis")
    if st.button("Run distance analysis"):
        analysis_rows = []
        detail_rows = []

        # Evaluate each candidate across all loaded records using current mapping/criteria.
        candidate_groups = sorted({(r.group, r.candidate) for r in recs})
        for group, cand in candidate_groups:
            subset = [r for r in recs if r.group == group and r.candidate == cand]
            total_pop = sum((r.population or 0.0) for r in subset)
            if total_pop == 0:
                total_pop = float(len(subset))

            for thr in thresholds:
                criterion_pop_sums = {}
                all_satisfied_pop = 0.0

                for _, crow in criteria_df.iterrows():
                    crit_label = str(crow["label"]).strip()
                    a_label = str(crow["atom_a_label"]).strip()
                    b_label = str(crow["atom_b_label_or_group"]).strip()
                    if not crit_label or not a_label or not b_label:
                        continue
                    if a_label not in mapping or b_label not in mapping:
                        continue

                    a_idx = mapping[a_label][0]
                    b_idxs = mapping[b_label]
                    sat_pop = 0.0
                    sat_ids = []
                    for rec in subset:
                        if a_idx >= len(rec.atoms) or max(b_idxs) >= len(rec.atoms):
                            continue
                        d = min_group_distance(rec.coords, a_idx, b_idxs)
                        pop = rec.population if rec.population is not None else (100.0 / len(subset))
                        detail_rows.append({
                            "group": group,
                            "candidate": cand,
                            "conformer_id": rec.conformer_id,
                            "threshold": thr,
                            "criterion": crit_label,
                            "distance": d,
                            "population": pop,
                            "satisfies": d <= thr,
                        })
                        if d <= thr:
                            sat_pop += pop
                            sat_ids.append(rec.conformer_id)
                    criterion_pop_sums[crit_label] = sat_pop

                # all-criteria satisfied in same conformer
                active_criteria = []
                for _, crow in criteria_df.iterrows():
                    crit_label = str(crow["label"]).strip()
                    a_label = str(crow["atom_a_label"]).strip()
                    b_label = str(crow["atom_b_label_or_group"]).strip()
                    if crit_label and a_label in mapping and b_label in mapping:
                        active_criteria.append((crit_label, mapping[a_label][0], mapping[b_label]))

                for rec in subset:
                    satisfied_all = True
                    for _, a_idx, b_idxs in active_criteria:
                        if a_idx >= len(rec.atoms) or max(b_idxs) >= len(rec.atoms):
                            satisfied_all = False
                            break
                        d = min_group_distance(rec.coords, a_idx, b_idxs)
                        if d > thr:
                            satisfied_all = False
                            break
                    if satisfied_all:
                        all_satisfied_pop += rec.population if rec.population is not None else (100.0 / len(subset))

                row = {
                    "group": group,
                    "candidate": cand,
                    "threshold_A": thr,
                    "all_criteria_same_conformer_population_sum": all_satisfied_pop,
                }
                row.update({f"population_sum::{k}": v for k, v in criterion_pop_sums.items()})
                analysis_rows.append(row)

        if analysis_rows:
            result_df = pd.DataFrame(analysis_rows).sort_values(
                by=["threshold_A", "all_criteria_same_conformer_population_sum"], ascending=[True, False]
            )
            detail_df = pd.DataFrame(detail_rows)

            st.subheader("Candidate comparison")
            st.dataframe(result_df, use_container_width=True)

            st.subheader("Per-conformer detail")
            st.dataframe(detail_df, use_container_width=True, height=320)

            csv_result = result_df.to_csv(index=False).encode("utf-8")
            csv_detail = detail_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download candidate comparison CSV", csv_result, "candidate_comparison.csv", "text/csv")
            st.download_button("Download per-conformer detail CSV", csv_detail, "per_conformer_detail.csv", "text/csv")
        else:
            st.warning("No analysis rows were generated. Check atom mapping and criteria.")


st.markdown("---")
st.caption("Tips: Use Gaussian logs when you want populations computed from energies/free energies. Use SDF when population values already come from conformer search software such as CONFLEX.")
