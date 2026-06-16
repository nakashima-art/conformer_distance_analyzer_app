import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    from rdkit import Chem
except Exception:
    Chem = None


st.set_page_config(page_title="Conformer Distance Analyzer", layout="wide")


# ==============================
# 3D atom-picker component
# ==============================
_COMPONENT_DIR = Path(__file__).parent / "atom_picker_component"
_atom_picker_component = components.declare_component(
    "atom_picker_component",
    path=str(_COMPONENT_DIR),
)


def atom_picker(
    atoms: List[str],
    coords: np.ndarray,
    selected_atoms: Optional[List[int]] = None,
    height: int = 520,
    key: Optional[str] = None,
) -> List[int]:
    """Return selected atom numbers as a 1-based integer list."""
    xyz_lines = [str(len(atoms)), "Conformer Distance Analyzer"]
    for symbol, (x, y, z) in zip(atoms, coords):
        xyz_lines.append(f"{symbol} {x:.8f} {y:.8f} {z:.8f}")

    default_selection = sorted(set(int(x) for x in (selected_atoms or [])))
    value = _atom_picker_component(
        xyz="\n".join(xyz_lines),
        selected_atoms=default_selection,
        height=int(height),
        key=key,
        default=default_selection,
    )
    if not isinstance(value, list):
        return default_selection

    cleaned: List[int] = []
    for item in value:
        try:
            number = int(item)
            if 1 <= number <= len(atoms):
                cleaned.append(number)
        except (TypeError, ValueError):
            pass
    return sorted(set(cleaned))


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
        for prop_key, value in props.items():
            k = str(prop_key).lower()
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


def compute_boltzmann_populations(
    records: List[ConformerRecord],
    temperature: float = 298.15,
) -> List[ConformerRecord]:
    grouped: Dict[Tuple[str, str], List[ConformerRecord]] = {}
    for rec in records:
        grouped.setdefault((rec.group, rec.candidate), []).append(rec)

    r_kcal = 0.0019872041
    for _, recs in grouped.items():
        # Preserve populations already supplied by the SDF when all are available.
        if all(rec.population is not None for rec in recs):
            total = sum(float(rec.population) for rec in recs)
            if total > 0:
                for rec in recs:
                    rec.population = 100.0 * float(rec.population) / total
                    rec.metadata["population_mode"] = "supplied_population"
                continue

        values: List[Optional[float]] = []
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
                rec.metadata["population_mode"] = "equal_weight"
            continue

        numeric_values = [float(v) for v in values if v is not None]
        min_e = min(numeric_values)

        # Gaussian energies are in hartree. SDF search energies are commonly in kcal/mol.
        if all(rec.source_type == "gaussian_log" for rec in recs):
            deltas = [hartree_to_kcal(float(v) - min_e) for v in values if v is not None]
        else:
            deltas = [float(v) - min_e for v in values if v is not None]

        weights = [math.exp(-d / (r_kcal * temperature)) for d in deltas]
        denom = sum(weights)
        for rec, weight in zip(recs, weights):
            rec.population = 100.0 * weight / denom
            rec.metadata["population_mode"] = mode or "unknown"
    return records


# ==============================
# Distance analysis
# ==============================
def atom_index_from_user_number(user_number: int) -> int:
    return user_number - 1


def pair_distance(coords: np.ndarray, idx_a: int, idx_b: int) -> float:
    return float(np.linalg.norm(coords[idx_a] - coords[idx_b]))


def min_between_groups(coords: np.ndarray, group_a: List[int], group_b: List[int]) -> float:
    distances = [
        pair_distance(coords, i, j)
        for i in group_a
        for j in group_b
        if i != j
    ]
    if not distances:
        raise ValueError("No valid atom pair was available for this criterion.")
    return float(min(distances))


# ==============================
# App state helpers
# ==============================
def init_state():
    defaults = {
        "records": [],
        "loaded": False,
        "atom_mappings": {},
        "mapping_selections": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

st.title("Conformer Distance Analyzer")
st.caption(
    "Compare conformer ensembles from SDF or Gaussian log files using "
    "user-defined H···H distance criteria and conformer populations."
)


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
            [
                {
                    "file_name": f.name,
                    "group": default_group,
                    "candidate": re.sub(r"\.sdf$", "", f.name, flags=re.I),
                }
                for f in sdf_files
            ]
        )
        edited = st.data_editor(mapping_df, num_rows="fixed", use_container_width=True, key="sdf_map")
        if st.button("Load SDF ensembles"):
            all_records: List[ConformerRecord] = []
            for f in sdf_files:
                matching = edited.loc[edited["file_name"] == f.name]
                if matching.empty:
                    continue
                row = matching.iloc[0]
                recs = parse_sdf_ensemble(f, str(row["group"]), str(row["candidate"]))
                all_records.extend(recs)
            all_records = compute_boltzmann_populations(all_records, temperature=temperature)
            st.session_state.records = all_records
            st.session_state.loaded = True
            st.session_state.atom_mappings = {}
            st.session_state.mapping_selections = {}
            st.success(f"Loaded {len(all_records)} conformers from {len(sdf_files)} SDF file(s).")
else:
    st.subheader("Upload Gaussian log files")
    st.write(
        "Upload Gaussian log files in groups. After files are added to one group, "
        "a new upload area can be added for the next group."
    )

    if "gaussian_group_count" not in st.session_state:
        st.session_state.gaussian_group_count = 1
    if "gaussian_group_specs" not in st.session_state:
        st.session_state.gaussian_group_specs = {}

    group_specs = []

    for i in range(st.session_state.gaussian_group_count):
        with st.container(border=True):
            group_name = st.text_input(
                f"Group {i + 1} name",
                value=st.session_state.gaussian_group_specs.get(i, {}).get("group", f"Group {i + 1}"),
                key=f"gaussian_group_name_{i}",
            )
            files = st.file_uploader(
                f"Gaussian log files for Group {i + 1}",
                type=["log", "out"],
                accept_multiple_files=True,
                key=f"gaussian_group_files_{i}",
            )

            st.session_state.gaussian_group_specs[i] = {"group": group_name}
            group_specs.append(
                {
                    "group": group_name,
                    "candidate": group_name,
                    "files": files or [],
                }
            )

    last_group_has_files = bool(group_specs and group_specs[-1]["files"])
    if last_group_has_files and st.button("Add another Gaussian log group"):
        st.session_state.gaussian_group_count += 1
        st.rerun()

    populated_groups = [g for g in group_specs if g["files"]]
    if populated_groups:
        summary_rows = []
        for group_spec in populated_groups:
            for f in group_spec["files"]:
                summary_rows.append(
                    {
                        "group": group_spec["group"],
                        "candidate": group_spec["candidate"],
                        "file_name": f.name,
                    }
                )
        st.markdown("#### Current assignments")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    if st.button("Load Gaussian log files"):
        if not populated_groups:
            st.error("Please upload at least one Gaussian log group.")
        else:
            all_records: List[ConformerRecord] = []
            total_files = 0
            for group_spec in populated_groups:
                for f in group_spec["files"]:
                    rec = parse_gaussian_log(f, group_spec["group"], group_spec["candidate"])
                    all_records.append(rec)
                    total_files += 1
            all_records = compute_boltzmann_populations(all_records, temperature=temperature)
            st.session_state.records = all_records
            st.session_state.loaded = True
            st.session_state.atom_mappings = {}
            st.session_state.mapping_selections = {}
            st.success(
                f"Loaded {len(all_records)} conformers from {total_files} Gaussian log file(s) "
                f"across {len(populated_groups)} group(s)."
            )


# ==============================
# Loaded summary
# ==============================
if st.session_state.loaded and st.session_state.records:
    st.header("2. Loaded conformers")
    recs: List[ConformerRecord] = st.session_state.records
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
            "population_mode": r.metadata.get("population_mode", ""),
        }
        for r in recs
    ])
    st.dataframe(summary_df, use_container_width=True, height=280)

    # ==============================
    # Step 3: intuitive atom mapping
    # ==============================
    st.header("3. Atom mapping")
    st.write(
        "Choose a candidate, click one or more atoms in the 3D structure, assign a proton label, "
        "and save the mapping. Multiple selected atoms are treated as an interchangeable group."
    )

    unique_candidates = sorted({(r.group, r.candidate) for r in recs})
    candidate_options = [f"{g} :: {c}" for g, c in unique_candidates]
    selected_candidate = st.selectbox("Candidate for atom mapping", candidate_options)
    sel_group, sel_cand = selected_candidate.split(" :: ", 1)
    candidate_key = selected_candidate
    candidate_records = [r for r in recs if r.group == sel_group and r.candidate == sel_cand]
    example = candidate_records[0]

    mappings_for_candidate: List[Dict[str, object]] = st.session_state.atom_mappings.setdefault(
        candidate_key, []
    )
    current_selection = st.session_state.mapping_selections.get(candidate_key, [])

    left, right = st.columns([1.55, 1.0], gap="large")

    with left:
        st.markdown("#### Click atoms in the 3D structure")
        picked_atoms = atom_picker(
            example.atoms,
            example.coords,
            selected_atoms=current_selection,
            key=f"atom_picker_{candidate_key}",
        )
        st.session_state.mapping_selections[candidate_key] = picked_atoms
        current_selection = picked_atoms

        st.caption(
            "Hydrogen atom numbers are shown by default. Use the checkboxes above the viewer "
            "to display all atom numbers or to hide labels."
        )

    with right:
        st.markdown("#### New proton mapping")

        if current_selection:
            selected_details = [
                f"{example.atoms[number - 1]} {number}"
                for number in current_selection
                if 1 <= number <= len(example.atoms)
            ]
            st.success("Selected: " + ", ".join(selected_details))
        else:
            st.info("No atom is selected yet.")

        label_mode = st.radio(
            "Label entry",
            ["Build H label", "Free text"],
            horizontal=True,
            key=f"label_mode_{candidate_key}",
        )

        if label_mode == "Build H label":
            label_col1, label_col2 = st.columns([1.2, 1.0])
            position = label_col1.text_input(
                "Position number",
                value="",
                placeholder="e.g. 3",
                key=f"position_{candidate_key}",
            ).strip()
            prime = label_col2.selectbox(
                "Prime",
                ["", "′", "″", "‴"],
                key=f"prime_{candidate_key}",
            )
            proposed_label = f"H-{position}{prime}" if position else ""
            if proposed_label:
                st.markdown(f"Display label: **{proposed_label}**")
        else:
            proposed_label = st.text_input(
                "Label",
                value="",
                placeholder="e.g. H-2′ / H-6′",
                key=f"custom_label_{candidate_key}",
            ).strip()

        button_col1, button_col2 = st.columns(2)
        save_mapping = button_col1.button(
            "Save mapping",
            type="primary",
            use_container_width=True,
            key=f"save_mapping_{candidate_key}",
        )
        clear_selection = button_col2.button(
            "Clear selection",
            use_container_width=True,
            key=f"clear_selection_{candidate_key}",
        )

        if clear_selection:
            st.session_state.mapping_selections[candidate_key] = []
            st.rerun()

        if save_mapping:
            if not proposed_label:
                st.error("Enter a proton label.")
            elif not current_selection:
                st.error("Select at least one atom in the 3D structure.")
            else:
                non_h = [n for n in current_selection if example.atoms[n - 1] != "H"]
                if non_h:
                    st.error(
                        "The following selected atoms are not hydrogen atoms: "
                        + ", ".join(map(str, non_h))
                    )
                else:
                    atom_indices = [atom_index_from_user_number(n) for n in current_selection]
                    existing = next(
                        (item for item in mappings_for_candidate if item["label"] == proposed_label),
                        None,
                    )
                    if existing is None:
                        mappings_for_candidate.append(
                            {
                                "label": proposed_label,
                                "atom_numbers": list(current_selection),
                                "atom_indices": atom_indices,
                            }
                        )
                        st.success(f"Added {proposed_label}.")
                    else:
                        existing["atom_numbers"] = list(current_selection)
                        existing["atom_indices"] = atom_indices
                        st.success(f"Updated {proposed_label}.")
                    st.session_state.mapping_selections[candidate_key] = []
                    st.rerun()

        with st.expander("Manual selection fallback"):
            st.caption("Use this only when clicking the 3D structure is difficult.")
            manual_raw = st.text_input(
                "1-based atom numbers, separated by commas",
                value=",".join(map(str, current_selection)),
                key=f"manual_selection_{candidate_key}",
            )
            if st.button("Apply manual selection", key=f"apply_manual_{candidate_key}"):
                try:
                    manual_numbers = sorted({
                        int(x.strip())
                        for x in manual_raw.split(",")
                        if x.strip()
                    })
                    invalid = [n for n in manual_numbers if n < 1 or n > len(example.atoms)]
                    if invalid:
                        st.error("Out-of-range atom numbers: " + ", ".join(map(str, invalid)))
                    else:
                        st.session_state.mapping_selections[candidate_key] = manual_numbers
                        st.rerun()
                except ValueError:
                    st.error("Enter integers separated by commas.")

    st.markdown("#### Registered mappings for this candidate")
    if not mappings_for_candidate:
        st.info("No mappings have been registered for this candidate.")
    else:
        for idx, item in enumerate(list(mappings_for_candidate)):
            with st.container(border=True):
                info_col, select_col, delete_col = st.columns([4.5, 1.5, 1.0])
                atom_text = ", ".join(str(n) for n in item["atom_numbers"])
                info_col.markdown(f"**{item['label']}**  \nAtom(s): {atom_text}")
                if select_col.button(
                    "Show/edit atoms",
                    key=f"show_mapping_{candidate_key}_{idx}",
                    use_container_width=True,
                ):
                    st.session_state.mapping_selections[candidate_key] = list(item["atom_numbers"])
                    st.rerun()
                if delete_col.button(
                    "Delete",
                    key=f"delete_mapping_{candidate_key}_{idx}",
                    use_container_width=True,
                ):
                    mappings_for_candidate.pop(idx)
                    st.rerun()

    status_rows = []
    for group, candidate in unique_candidates:
        key = f"{group} :: {candidate}"
        candidate_mappings = st.session_state.atom_mappings.get(key, [])
        status_rows.append(
            {
                "group": group,
                "candidate": candidate,
                "registered_labels": len(candidate_mappings),
                "labels": ", ".join(str(x["label"]) for x in candidate_mappings),
            }
        )
    with st.expander("Mapping status for all candidates"):
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    # ==============================
    # Step 4: criteria
    # ==============================
    st.header("4. Distance criteria")
    st.write(
        "Enter the mapping labels to compare. The same criterion is applied to every candidate, "
        "using that candidate's own atom mapping."
    )

    criteria_default = pd.DataFrame(
        {
            "criterion": ["NOE 1"],
            "proton_or_group_A": [""],
            "proton_or_group_B": [""],
        }
    )
    criteria_df = st.data_editor(
        criteria_default,
        num_rows="dynamic",
        use_container_width=True,
        key="global_distance_criteria",
    )

    # ==============================
    # Step 5: analysis
    # ==============================
    st.header("5. Analysis")
    if st.button("Run distance analysis", type="primary"):
        analysis_rows = []
        detail_rows = []
        skipped_messages = []

        active_criteria_rows = []
        for _, criterion_row in criteria_df.iterrows():
            crit_label = str(criterion_row["criterion"]).strip()
            a_label = str(criterion_row["proton_or_group_A"]).strip()
            b_label = str(criterion_row["proton_or_group_B"]).strip()
            if crit_label and a_label and b_label:
                active_criteria_rows.append((crit_label, a_label, b_label))

        if not active_criteria_rows:
            st.error("Define at least one complete distance criterion.")
        else:
            candidate_groups = sorted({(r.group, r.candidate) for r in recs})
            for group, candidate in candidate_groups:
                subset = [r for r in recs if r.group == group and r.candidate == candidate]
                fallback_pop = 100.0 / len(subset) if subset else 0.0
                key = f"{group} :: {candidate}"
                candidate_mapping_items = st.session_state.atom_mappings.get(key, [])
                candidate_mapping: Dict[str, List[int]] = {
                    str(item["label"]): [int(i) for i in item["atom_indices"]]
                    for item in candidate_mapping_items
                }

                missing_labels = sorted({
                    label
                    for _, a_label, b_label in active_criteria_rows
                    for label in (a_label, b_label)
                    if label not in candidate_mapping
                })
                if missing_labels:
                    skipped_messages.append(
                        f"{key}: missing mapping(s): {', '.join(missing_labels)}"
                    )
                    continue

                for threshold in thresholds:
                    criterion_pop_sums: Dict[str, float] = {}
                    all_satisfied_pop = 0.0

                    for crit_label, a_label, b_label in active_criteria_rows:
                        a_indices = candidate_mapping[a_label]
                        b_indices = candidate_mapping[b_label]
                        sat_pop = 0.0

                        for rec in subset:
                            if (
                                not a_indices
                                or not b_indices
                                or max(a_indices) >= len(rec.atoms)
                                or max(b_indices) >= len(rec.atoms)
                            ):
                                continue

                            distance = min_between_groups(rec.coords, a_indices, b_indices)
                            pop = rec.population if rec.population is not None else fallback_pop
                            satisfies = distance <= threshold
                            detail_rows.append(
                                {
                                    "group": group,
                                    "candidate": candidate,
                                    "conformer_id": rec.conformer_id,
                                    "threshold_A": threshold,
                                    "criterion": crit_label,
                                    "mapping_A": a_label,
                                    "mapping_B": b_label,
                                    "distance_A": distance,
                                    "population_percent": pop,
                                    "satisfies": satisfies,
                                }
                            )
                            if satisfies:
                                sat_pop += pop

                        criterion_pop_sums[crit_label] = sat_pop

                    for rec in subset:
                        satisfied_all = True
                        for _, a_label, b_label in active_criteria_rows:
                            a_indices = candidate_mapping[a_label]
                            b_indices = candidate_mapping[b_label]
                            if (
                                not a_indices
                                or not b_indices
                                or max(a_indices) >= len(rec.atoms)
                                or max(b_indices) >= len(rec.atoms)
                            ):
                                satisfied_all = False
                                break
                            distance = min_between_groups(rec.coords, a_indices, b_indices)
                            if distance > threshold:
                                satisfied_all = False
                                break

                        if satisfied_all:
                            pop = rec.population if rec.population is not None else fallback_pop
                            all_satisfied_pop += pop

                    result_row = {
                        "group": group,
                        "candidate": candidate,
                        "threshold_A": threshold,
                        "all_criteria_same_conformer_population_sum": all_satisfied_pop,
                    }
                    result_row.update(
                        {f"population_sum::{name}": value for name, value in criterion_pop_sums.items()}
                    )
                    analysis_rows.append(result_row)

            if skipped_messages:
                st.warning("Some candidates were skipped:\n\n" + "\n\n".join(skipped_messages))

            if analysis_rows:
                result_df = pd.DataFrame(analysis_rows).sort_values(
                    by=["threshold_A", "all_criteria_same_conformer_population_sum"],
                    ascending=[True, False],
                )
                detail_df = pd.DataFrame(detail_rows)

                st.subheader("Candidate comparison")
                st.dataframe(result_df, use_container_width=True)

                st.subheader("Per-conformer detail")
                st.dataframe(detail_df, use_container_width=True, height=320)

                csv_result = result_df.to_csv(index=False).encode("utf-8-sig")
                csv_detail = detail_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "Download candidate comparison CSV",
                    csv_result,
                    "candidate_comparison.csv",
                    "text/csv",
                )
                st.download_button(
                    "Download per-conformer detail CSV",
                    csv_detail,
                    "per_conformer_detail.csv",
                    "text/csv",
                )
            else:
                st.warning("No analysis rows were generated. Check atom mappings and criteria.")


st.markdown("---")
st.caption(
    "Tips: Use Gaussian logs when populations should be computed from electronic/free energies. "
    "Use SDF when conformer populations or conformer-search energies are stored in the file."
)
