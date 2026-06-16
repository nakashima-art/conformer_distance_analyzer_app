import io
import json
import math
import re
import zipfile
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


APP_VERSION = "1.1.0"
PROJECT_FORMAT_VERSION = 1

TEXTS = {
    "ja": {
        "app_title": "Conformer Distance Analyzer",
        "app_caption": "SDFまたはGaussian log由来の配座集団を、指定したH···H距離条件と配座存在比に基づいて比較します。",
        "open_project": "保存済みプロジェクトを開く",
        "open_project_desc": "保存プロジェクトには、配座、座標、存在比、原子マッピング、距離条件、設定、および直近の解析結果が含まれます。",
        "saved_project": "保存済みCDAプロジェクト",
        "replace_session": "現在のセッションをアップロードしたプロジェクトで置き換える",
        "replace_warning": "プロジェクトを読み込むと、現在の配座、マッピング、距離条件、設定、および保存済み解析結果が置き換えられます。",
        "upload_project": "保存済みプロジェクトを読み込む",
        "load_project_error": "プロジェクトを読み込めませんでした：{error}",
        "project_loaded": "プロジェクト「{name}」を読み込みました（{count}配座）。",
        "analysis_settings": "解析設定",
        "minimum_distance": "最小距離（Å）",
        "maximum_distance": "最大距離（Å）",
        "distance_error": "最小距離は最大距離以下にしてください。",
        "distance_note": "NOE／ROE解析では通常、最小距離を0 Åとし、最大距離を設定します。",
        "temperature": "ボルツマン温度（K）",
        "input_source": "1. 入力ファイル",
        "choose_input": "入力形式を選択",
        "sdf_ensemble": "SDF配座集団",
        "gaussian_logs": "Gaussian logファイル",
        "upload_sdf": "SDF配座集団のアップロード",
        "upload_sdf_desc": "複数配座を含むSDFファイルを1つ以上アップロードしてください。各ファイルを1候補の配座集団として扱います。",
        "sdf_files": "SDFファイル",
        "default_group": "デフォルトのグループ名",
        "load_sdf": "SDF配座集団を読み込む",
        "loaded_sdf": "{files}個のSDFファイルから{count}配座を読み込みました。",
        "upload_gaussian": "Gaussian logファイルのアップロード",
        "upload_gaussian_desc": "Gaussian logファイルをグループごとにアップロードします。ファイルを追加すると、次のグループ用アップロード欄を追加できます。",
        "group_name": "グループ{number}の名前",
        "group_files": "グループ{number}のGaussian logファイル",
        "add_group": "Gaussian logグループを追加",
        "current_assignments": "#### 現在の割り当て",
        "load_gaussian": "Gaussian logファイルを読み込む",
        "need_gaussian": "Gaussian logグループを1つ以上アップロードしてください。",
        "loaded_gaussian": "{groups}グループ、{files}個のGaussian logファイルから{count}配座を読み込みました。",
        "loaded_conformers": "2. 読み込んだ配座",
        "atom_mapping": "3. 原子マッピング",
        "atom_mapping_desc": "候補を選択し、3D構造上の原子を1個以上クリックしてプロトンラベルを登録します。複数原子を選択した場合は、交換可能な原子グループとして扱います。",
        "candidate_mapping": "原子マッピングを行う候補",
        "copy_mappings": "候補間でマッピングをコピー",
        "copy_once": "マッピングは一度だけコピーされ、コピー後は各候補を独立して編集できます。",
        "copy_from": "コピー元の候補",
        "copy_mode": "コピー方法",
        "labels_atoms": "ラベル＋原子番号",
        "labels_only": "ラベルのみ",
        "compatible": "コピー元とコピー先の原子数および元素順序が一致しています。原子番号もコピーできます。",
        "incompatible": "コピー元とコピー先で原子順序が異なります。「ラベルのみ」を使用してください。",
        "labels_only_info": "ラベルのみをコピーし、原子番号は未割り当ての状態にします。",
        "copy_replace_note": "コピーすると、コピー先候補に現在登録されているマッピングはすべて置き換えられます。",
        "copy_to_candidate": "この候補にマッピングをコピー",
        "copied_mapping": "{detail}を「{source}」から「{destination}」へコピーしました。コピー後のマッピングは独立しています。",
        "detail_labels_atoms": "ラベルと原子番号",
        "detail_labels_only": "ラベルのみ",
        "apply_all": "##### 現在のマッピングを互換性のある全候補に適用",
        "apply_all_desc": "原子数と元素順序が一致する候補に、ラベルと原子番号をコピーします。コピー先の既存マッピングは置き換えられます。",
        "confirm_replace": "互換性のある候補の既存マッピングが置き換えられることを確認しました。",
        "apply_all_button": "現在のマッピングを互換性のある全候補に適用",
        "bulk_result": "{copied}個の互換性のある候補にコピーしました。",
        "bulk_skipped": " 原子順序が異なる{skipped}個の候補はスキップしました。",
        "click_atoms": "#### 3D構造上の原子をクリック",
        "viewer_caption": "初期状態では水素原子番号を表示します。ビューア上部のチェックボックスで全原子番号の表示や水素のみの表示を切り替えられます。",
        "new_mapping": "#### 新しいプロトンマッピング",
        "selected": "選択中：{items}",
        "none_selected": "原子がまだ選択されていません。",
        "label_entry": "ラベルの入力方法",
        "build_label": "Hラベルを作成",
        "free_text": "自由入力",
        "position_number": "位置番号",
        "position_placeholder": "例：3",
        "prime": "プライム",
        "display_label": "表示ラベル：**{label}**",
        "label": "ラベル",
        "label_placeholder": "例：H-2′ / H-6′",
        "save_mapping": "マッピングを保存",
        "clear_selection": "選択を解除",
        "enter_label": "プロトンラベルを入力してください。",
        "select_atom": "3D構造上で原子を1個以上選択してください。",
        "non_hydrogen": "次の選択原子は水素ではありません：{atoms}",
        "added": "{label}を追加しました。",
        "updated": "{label}を更新しました。",
        "manual_fallback": "手動選択（予備）",
        "manual_desc": "3D構造のクリック操作が難しい場合のみ使用してください。",
        "manual_numbers": "1始まりの原子番号（カンマ区切り）",
        "apply_manual": "手動選択を適用",
        "out_of_range": "範囲外の原子番号：{atoms}",
        "integer_error": "カンマ区切りの整数を入力してください。",
        "registered_mappings": "#### この候補に登録済みのマッピング",
        "no_mappings": "この候補にはマッピングが登録されていません。",
        "not_assigned": "未割り当て",
        "atoms_label": "原子",
        "show_edit": "原子を表示・編集",
        "delete": "削除",
        "mapping_status": "全候補のマッピング状況",
        "distance_criteria": "4. 距離条件",
        "criteria_desc": "比較するマッピングラベルを入力してください。同じ距離条件を、各候補固有の原子マッピングに基づいて適用します。",
        "analysis": "5. 解析",
        "run_analysis": "距離解析を実行",
        "need_criterion": "完全な距離条件を1つ以上設定してください。",
        "no_analysis": "解析結果が生成されませんでした。原子マッピングと距離条件を確認してください。",
        "skipped": "一部の候補をスキップしました：\n\n{messages}",
        "candidate_comparison": "候補間比較",
        "per_conformer": "配座ごとの詳細",
        "download_comparison": "候補間比較CSVをダウンロード",
        "download_detail": "配座詳細CSVをダウンロード",
        "save_project": "6. プロジェクトを保存",
        "save_project_desc": "現在の解析状態をPCに保存します。保存プロジェクトを読み込めば、原子ラベルの再入力や元のSDF／logファイルの再アップロードなしで解析を再開できます。",
        "project_name": "プロジェクト名",
        "download_project": "現在のプロジェクトをダウンロード",
        "project_contents": ".cda.zipファイルには、座標、存在比、マッピング、距離条件、設定、および直近の解析結果が含まれます。",
        "prepare_project_error": "プロジェクトファイルを作成できませんでした：{error}",
        "footer_tip": "ヒント：電子エネルギー／自由エネルギーから存在比を計算する場合はGaussian logを使用します。配座存在比または配座探索エネルギーが保存されている場合はSDFを使用します。",
        "version_line": "バージョン {app_version}｜プロジェクト形式 v{project_version}",
        "author_line": "作成者：Ken-ichi Nakashima（愛知学院大学薬学部 薬用資源学講座）",
        "col_file": "ファイル名", "col_group": "グループ", "col_candidate": "候補",
        "col_source": "元ファイル", "col_conformer": "配座ID", "col_atoms": "原子数",
        "col_energy": "エネルギー", "col_free_energy": "自由エネルギー", "col_population": "存在比（%）",
        "col_population_mode": "存在比の算出法", "col_registered": "登録ラベル数", "col_labels": "ラベル",
        "col_criterion": "距離条件", "col_a": "プロトン／グループA", "col_b": "プロトン／グループB",
        "col_min": "最小距離（Å）", "col_max": "最大距離（Å）", "col_distance": "距離（Å）",
        "col_satisfies": "条件適合", "col_all_sum": "全条件を同一配座で満たす存在比合計（%）",
        "col_pop_sum_prefix": "存在比合計",
    },
    "en": {
        "app_title": "Conformer Distance Analyzer",
        "app_caption": "Compare conformer ensembles from SDF or Gaussian log files using user-defined H···H distance criteria and conformer populations.",
        "open_project": "Open a saved project",
        "open_project_desc": "A saved project contains conformers, coordinates, populations, atom mappings, distance criteria, settings, and the most recent analysis results.",
        "saved_project": "Saved CDA project",
        "replace_session": "Replace the current session with the uploaded project.",
        "replace_warning": "Loading a project replaces the conformers, mappings, criteria, settings, and stored results currently open in this session.",
        "upload_project": "Upload saved project",
        "load_project_error": "Could not load the project: {error}",
        "project_loaded": "Loaded project '{name}' with {count} conformer(s).",
        "analysis_settings": "Analysis settings",
        "minimum_distance": "Minimum distance (Å)",
        "maximum_distance": "Maximum distance (Å)",
        "distance_error": "Minimum distance must be less than or equal to maximum distance.",
        "distance_note": "For NOE/ROE analysis, the minimum distance is normally set to 0 Å and an upper distance cutoff is applied.",
        "temperature": "Boltzmann temperature (K)",
        "input_source": "1. Input source",
        "choose_input": "Choose input type",
        "sdf_ensemble": "SDF ensemble",
        "gaussian_logs": "Gaussian log files",
        "upload_sdf": "Upload SDF ensembles",
        "upload_sdf_desc": "Upload one or more multi-conformer SDF files. Each file is treated as one candidate ensemble.",
        "sdf_files": "SDF files",
        "default_group": "Default group name",
        "load_sdf": "Load SDF ensembles",
        "loaded_sdf": "Loaded {count} conformers from {files} SDF file(s).",
        "upload_gaussian": "Upload Gaussian log files",
        "upload_gaussian_desc": "Upload Gaussian log files in groups. After files are added to one group, a new upload area can be added for the next group.",
        "group_name": "Group {number} name",
        "group_files": "Gaussian log files for Group {number}",
        "add_group": "Add another Gaussian log group",
        "current_assignments": "#### Current assignments",
        "load_gaussian": "Load Gaussian log files",
        "need_gaussian": "Please upload at least one Gaussian log group.",
        "loaded_gaussian": "Loaded {count} conformers from {files} Gaussian log file(s) across {groups} group(s).",
        "loaded_conformers": "2. Loaded conformers",
        "atom_mapping": "3. Atom mapping",
        "atom_mapping_desc": "Choose a candidate, click one or more atoms in the 3D structure, assign a proton label, and save the mapping. Multiple selected atoms are treated as an interchangeable group.",
        "candidate_mapping": "Candidate for atom mapping",
        "copy_mappings": "Copy mappings between candidates",
        "copy_once": "Mappings are copied once. After copying, each candidate can be edited independently.",
        "copy_from": "Copy mapping from",
        "copy_mode": "Copy mode",
        "labels_atoms": "Labels and atom numbers",
        "labels_only": "Labels only",
        "compatible": "The source and destination have the same atom count and element order. Atom numbers can be copied.",
        "incompatible": "The atom order differs between the source and destination. Use Labels only.",
        "labels_only_info": "Only the labels will be copied. Atom assignments will be left empty for this candidate.",
        "copy_replace_note": "Copying replaces all mappings currently registered for the selected destination candidate.",
        "copy_to_candidate": "Copy mapping to this candidate",
        "copied_mapping": "Copied {detail} from {source} to {destination}. The copied mapping is now independent.",
        "detail_labels_atoms": "labels and atom numbers",
        "detail_labels_only": "labels only",
        "apply_all": "##### Apply the current mapping to all compatible candidates",
        "apply_all_desc": "This copies labels and atom numbers to candidates with the same atom count and element order. Existing mappings in those candidates will be replaced.",
        "confirm_replace": "I understand that existing mappings in compatible candidates will be replaced.",
        "apply_all_button": "Apply current mapping to all compatible candidates",
        "bulk_result": "Copied the current mapping to {copied} compatible candidate(s).",
        "bulk_skipped": " Skipped {skipped} candidate(s) with different atom ordering.",
        "click_atoms": "#### Click atoms in the 3D structure",
        "viewer_caption": "Hydrogen atom numbers are shown by default. Use the checkboxes above the viewer to display all atom numbers or to show hydrogens only.",
        "new_mapping": "#### New proton mapping",
        "selected": "Selected: {items}",
        "none_selected": "No atom is selected yet.",
        "label_entry": "Label entry",
        "build_label": "Build H label",
        "free_text": "Free text",
        "position_number": "Position number",
        "position_placeholder": "e.g. 3",
        "prime": "Prime",
        "display_label": "Display label: **{label}**",
        "label": "Label",
        "label_placeholder": "e.g. H-2′ / H-6′",
        "save_mapping": "Save mapping",
        "clear_selection": "Clear selection",
        "enter_label": "Enter a proton label.",
        "select_atom": "Select at least one atom in the 3D structure.",
        "non_hydrogen": "The following selected atoms are not hydrogen atoms: {atoms}",
        "added": "Added {label}.",
        "updated": "Updated {label}.",
        "manual_fallback": "Manual selection fallback",
        "manual_desc": "Use this only when clicking the 3D structure is difficult.",
        "manual_numbers": "1-based atom numbers, separated by commas",
        "apply_manual": "Apply manual selection",
        "out_of_range": "Out-of-range atom numbers: {atoms}",
        "integer_error": "Enter integers separated by commas.",
        "registered_mappings": "#### Registered mappings for this candidate",
        "no_mappings": "No mappings have been registered for this candidate.",
        "not_assigned": "Not assigned",
        "atoms_label": "Atom(s)",
        "show_edit": "Show/edit atoms",
        "delete": "Delete",
        "mapping_status": "Mapping status for all candidates",
        "distance_criteria": "4. Distance criteria",
        "criteria_desc": "Enter the mapping labels to compare. The same criterion is applied to every candidate, using that candidate's own atom mapping.",
        "analysis": "5. Analysis",
        "run_analysis": "Run distance analysis",
        "need_criterion": "Define at least one complete distance criterion.",
        "no_analysis": "No analysis rows were generated. Check atom mappings and criteria.",
        "skipped": "Some candidates were skipped:\n\n{messages}",
        "candidate_comparison": "Candidate comparison",
        "per_conformer": "Per-conformer detail",
        "download_comparison": "Download candidate comparison CSV",
        "download_detail": "Download per-conformer detail CSV",
        "save_project": "6. Save project",
        "save_project_desc": "Download the current analysis state to your computer. The saved project can later be uploaded without re-entering atom labels or re-uploading the original SDF/log files.",
        "project_name": "Project name",
        "download_project": "Download current project",
        "project_contents": "The .cda.zip file contains coordinates, populations, mappings, criteria, settings, and the most recent analysis results.",
        "prepare_project_error": "Could not prepare the project file: {error}",
        "footer_tip": "Tips: Use Gaussian logs when populations should be computed from electronic/free energies. Use SDF when conformer populations or conformer-search energies are stored in the file.",
        "version_line": "Version {app_version} | Project format v{project_version}",
        "author_line": "Developed by Ken-ichi Nakashima (Laboratory of Pharmacognosy, School of Pharmacy, Aichi Gakuin University)",
        "col_file": "File name", "col_group": "Group", "col_candidate": "Candidate",
        "col_source": "Source", "col_conformer": "Conformer ID", "col_atoms": "Number of atoms",
        "col_energy": "Energy", "col_free_energy": "Free energy", "col_population": "Population (%)",
        "col_population_mode": "Population mode", "col_registered": "Registered labels", "col_labels": "Labels",
        "col_criterion": "Criterion", "col_a": "Proton/group A", "col_b": "Proton/group B",
        "col_min": "Minimum distance (Å)", "col_max": "Maximum distance (Å)", "col_distance": "Distance (Å)",
        "col_satisfies": "Satisfies", "col_all_sum": "Population sum satisfying all criteria in the same conformer (%)",
        "col_pop_sum_prefix": "Population sum",
    },
}

def current_language() -> str:
    return st.session_state.get("language", "ja")

def t(key: str, **kwargs) -> str:
    text = TEXTS.get(current_language(), TEXTS["ja"]).get(key, key)
    return text.format(**kwargs) if kwargs else text


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
    language: str = "ja",
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
        language=str(language),
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


def atom_numbering_is_compatible(source: ConformerRecord, destination: ConformerRecord) -> bool:
    """Return True when 1-based atom numbers refer to the same element sequence."""
    return source.atoms == destination.atoms


def clone_mapping_items(
    mapping_items: List[Dict[str, object]],
    include_atom_numbers: bool,
) -> List[Dict[str, object]]:
    """Create an independent copy of mapping items for one-time copying."""
    cloned: List[Dict[str, object]] = []
    for item in mapping_items:
        atom_numbers = [int(n) for n in item.get("atom_numbers", [])] if include_atom_numbers else []
        cloned.append(
            {
                "label": str(item.get("label", "")),
                "atom_numbers": atom_numbers,
                "atom_indices": [atom_index_from_user_number(n) for n in atom_numbers],
            }
        )
    return cloned


# ==============================
# Project save / load
# ==============================
PROJECT_FORMAT = "Conformer Distance Analyzer Project"
PROJECT_VERSION = PROJECT_FORMAT_VERSION


def _optional_float(value) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def conformer_record_to_dict(record: ConformerRecord) -> Dict[str, object]:
    return {
        "source_type": record.source_type,
        "source_name": record.source_name,
        "group": record.group,
        "candidate": record.candidate,
        "conformer_id": record.conformer_id,
        "atoms": list(record.atoms),
        "coords": np.asarray(record.coords, dtype=float).tolist(),
        "energy": _optional_float(record.energy),
        "free_energy": _optional_float(record.free_energy),
        "population": _optional_float(record.population),
        "metadata": {str(k): str(v) for k, v in record.metadata.items()},
    }


def conformer_record_from_dict(data: Dict[str, object]) -> ConformerRecord:
    required = [
        "source_type", "source_name", "group", "candidate",
        "conformer_id", "atoms", "coords",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError("A conformer record is missing: " + ", ".join(missing))

    atoms = [str(x) for x in data["atoms"]]
    coords = np.asarray(data["coords"], dtype=float)
    if coords.shape != (len(atoms), 3):
        raise ValueError(
            f"Invalid coordinate shape for {data.get('conformer_id', 'unknown conformer')}."
        )

    return ConformerRecord(
        source_type=str(data["source_type"]),
        source_name=str(data["source_name"]),
        group=str(data["group"]),
        candidate=str(data["candidate"]),
        conformer_id=str(data["conformer_id"]),
        atoms=atoms,
        coords=coords,
        energy=_optional_float(data.get("energy")),
        free_energy=_optional_float(data.get("free_energy")),
        population=_optional_float(data.get("population")),
        metadata={str(k): str(v) for k, v in dict(data.get("metadata", {})).items()},
    )


def serializable_atom_mappings(
    mappings: Dict[str, List[Dict[str, object]]],
) -> Dict[str, List[Dict[str, object]]]:
    result: Dict[str, List[Dict[str, object]]] = {}
    for candidate_key, items in mappings.items():
        result[str(candidate_key)] = [
            {
                "label": str(item.get("label", "")),
                "atom_numbers": [int(n) for n in item.get("atom_numbers", [])],
            }
            for item in items
        ]
    return result


def restore_atom_mappings(
    mappings: Dict[str, object],
) -> Dict[str, List[Dict[str, object]]]:
    restored: Dict[str, List[Dict[str, object]]] = {}
    for candidate_key, raw_items in mappings.items():
        items: List[Dict[str, object]] = []
        for raw_item in list(raw_items):
            item = dict(raw_item)
            atom_numbers = sorted({int(n) for n in item.get("atom_numbers", [])})
            items.append(
                {
                    "label": str(item.get("label", "")),
                    "atom_numbers": atom_numbers,
                    "atom_indices": [atom_index_from_user_number(n) for n in atom_numbers],
                }
            )
        restored[str(candidate_key)] = items
    return restored


def normalize_criteria_records(raw_records: object) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for raw in list(raw_records or []):
        row = dict(raw)
        normalized.append(
            {
                "criterion": str(row.get("criterion", "")),
                "proton_or_group_A": str(row.get("proton_or_group_A", "")),
                "proton_or_group_B": str(row.get("proton_or_group_B", "")),
            }
        )
    return normalized or [
        {"criterion": "NOE 1", "proton_or_group_A": "", "proton_or_group_B": ""}
    ]


def build_project_bytes() -> bytes:
    payload = {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "app_version": APP_VERSION,
        "language": current_language(),
        "project_name": str(st.session_state.project_name).strip() or "conformer_distance_project",
        "settings": {
            "minimum_distance_A": float(st.session_state.analysis_minimum_distance),
            "maximum_distance_A": float(st.session_state.analysis_maximum_distance),
            "boltzmann_temperature_K": float(st.session_state.analysis_temperature),
        },
        "records": [conformer_record_to_dict(r) for r in st.session_state.records],
        "atom_mappings": serializable_atom_mappings(st.session_state.atom_mappings),
        "distance_criteria": normalize_criteria_records(
            st.session_state.distance_criteria_records
        ),
        "analysis_results": list(st.session_state.analysis_result_rows),
        "analysis_details": list(st.session_state.analysis_detail_rows),
        "analysis_skipped_messages": list(st.session_state.analysis_skipped_messages),
    }

    json_bytes = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("project.json", json_bytes)
    return buffer.getvalue()


def read_project_bytes(raw_bytes: bytes) -> Dict[str, object]:
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes), "r") as archive:
            if "project.json" not in archive.namelist():
                raise ValueError("project.json was not found in the project archive.")
            data = json.loads(archive.read("project.json").decode("utf-8"))
    except zipfile.BadZipFile as exc:
        raise ValueError("The uploaded file is not a valid CDA project ZIP file.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("The project JSON is invalid.") from exc

    if data.get("format") != PROJECT_FORMAT:
        raise ValueError("This file is not a Conformer Distance Analyzer project.")
    version = int(data.get("version", 0))
    if version > PROJECT_VERSION:
        raise ValueError(
            f"This project was created by a newer format version ({version})."
        )
    if not isinstance(data.get("records"), list) or not data["records"]:
        raise ValueError("The project does not contain any conformer records.")
    return data


def load_project_into_state(data: Dict[str, object]) -> None:
    records = [conformer_record_from_dict(dict(item)) for item in data["records"]]
    settings = dict(data.get("settings", {}))

    st.session_state.records = records
    st.session_state.loaded = True
    st.session_state.atom_mappings = restore_atom_mappings(
        dict(data.get("atom_mappings", {}))
    )
    st.session_state.mapping_selections = {}
    st.session_state.distance_criteria_records = normalize_criteria_records(
        data.get("distance_criteria", [])
    )
    st.session_state.criteria_editor_revision += 1
    st.session_state.analysis_minimum_distance = float(
        settings.get("minimum_distance_A", 0.0)
    )
    st.session_state.analysis_maximum_distance = float(
        settings.get("maximum_distance_A", 3.5)
    )
    st.session_state.analysis_temperature = float(
        settings.get("boltzmann_temperature_K", 298.15)
    )
    st.session_state.analysis_result_rows = list(data.get("analysis_results", []))
    st.session_state.analysis_detail_rows = list(data.get("analysis_details", []))
    st.session_state.analysis_skipped_messages = list(
        data.get("analysis_skipped_messages", [])
    )
    st.session_state.project_name = str(
        data.get("project_name", "conformer_distance_project")
    )
    saved_language = str(data.get("language", current_language()))
    if saved_language not in {"ja", "en"}:
        saved_language = "ja"
    st.session_state.pending_language = saved_language
    notice_language = TEXTS.get(saved_language, TEXTS["ja"])
    st.session_state.project_notice = (
        "success",
        notice_language["project_loaded"].format(
            name=st.session_state.project_name,
            count=len(records),
        ),
    )


def safe_project_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._") or "conformer_distance_project"
    return f"{cleaned}.cda.zip"


# ==============================
# App state helpers
# ==============================
def init_state():
    defaults = {
        "records": [],
        "loaded": False,
        "atom_mappings": {},
        "mapping_selections": {},
        "mapping_notice": None,
        "project_name": "conformer_distance_project",
        "project_notice": None,
        "analysis_minimum_distance": 0.0,
        "analysis_maximum_distance": 3.5,
        "analysis_temperature": 298.15,
        "distance_criteria_records": [
            {"criterion": "NOE 1", "proton_or_group_A": "", "proton_or_group_B": ""}
        ],
        "criteria_editor_revision": 0,
        "analysis_result_rows": [],
        "analysis_detail_rows": [],
        "analysis_skipped_messages": [],
        "language": "ja",
        "language_display": "日本語",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

if "pending_language" in st.session_state:
    pending_language = st.session_state.pop("pending_language")
    st.session_state.language = pending_language
    st.session_state.language_display = "日本語" if pending_language == "ja" else "English"

language_display = st.sidebar.selectbox(
    "言語 / Language",
    ["日本語", "English"],
    key="language_display",
)
st.session_state.language = "ja" if language_display == "日本語" else "en"

st.title(t("app_title"))
st.caption(t("app_caption"))

project_notice = st.session_state.pop("project_notice", None)
if project_notice:
    notice_type, notice_text = project_notice
    if notice_type == "success":
        st.success(notice_text)
    elif notice_type == "warning":
        st.warning(notice_text)
    else:
        st.info(notice_text)

with st.expander(t("open_project"), expanded=False):
    st.write(t("open_project_desc"))
    uploaded_project = st.file_uploader(
        t("saved_project"),
        type=["zip"],
        key="saved_project_uploader",
    )
    confirm_project_replace = st.checkbox(
        t("replace_session"),
        key="confirm_project_replace",
    )
    if st.session_state.loaded:
        st.warning(t("replace_warning"))
    if st.button(
        t("upload_project"),
        disabled=uploaded_project is None or not confirm_project_replace,
        type="primary",
        key="load_saved_project_button",
    ):
        try:
            project_data = read_project_bytes(uploaded_project.getvalue())
            load_project_into_state(project_data)
            st.rerun()
        except Exception as exc:
            st.error(t("load_project_error", error=exc))

# ==============================
# Sidebar configuration
# ==============================
st.sidebar.header(t("analysis_settings"))
col_min_dist, col_max_dist = st.sidebar.columns(2)
minimum_distance = col_min_dist.number_input(
    t("minimum_distance"),
    min_value=0.0,
    step=0.1,
    format="%.2f",
    key="analysis_minimum_distance",
)
maximum_distance = col_max_dist.number_input(
    t("maximum_distance"),
    min_value=0.0,
    step=0.1,
    format="%.2f",
    key="analysis_maximum_distance",
)
distance_range_valid = float(minimum_distance) <= float(maximum_distance)
if not distance_range_valid:
    st.sidebar.error(t("distance_error"))
st.sidebar.caption(t("distance_note"))
temperature = st.sidebar.number_input(
    t("temperature"),
    step=1.0,
    key="analysis_temperature",
)


# ==============================
# Step 1: input mode
# ==============================
st.header(t("input_source"))
input_mode = st.radio(
    t("choose_input"),
    ["sdf", "gaussian"],
    format_func=lambda x: t("sdf_ensemble") if x == "sdf" else t("gaussian_logs"),
    horizontal=True,
)

if input_mode == "sdf":
    st.subheader(t("upload_sdf"))
    st.write(t("upload_sdf_desc"))
    sdf_files = st.file_uploader(t("sdf_files"), type=["sdf"], accept_multiple_files=True)
    default_group = st.text_input(t("default_group"), value="default_group")

    if sdf_files:
        file_col, group_col, candidate_col = t("col_file"), t("col_group"), t("col_candidate")
        mapping_df = pd.DataFrame(
            [
                {
                    file_col: f.name,
                    group_col: default_group,
                    candidate_col: re.sub(r"\.sdf$", "", f.name, flags=re.I),
                }
                for f in sdf_files
            ]
        )
        edited = st.data_editor(mapping_df, num_rows="fixed", use_container_width=True, key=f"sdf_map_{current_language()}")
        if st.button(t("load_sdf")):
            all_records: List[ConformerRecord] = []
            for f in sdf_files:
                matching = edited.loc[edited[file_col] == f.name]
                if matching.empty:
                    continue
                row = matching.iloc[0]
                recs = parse_sdf_ensemble(f, str(row[group_col]), str(row[candidate_col]))
                all_records.extend(recs)
            all_records = compute_boltzmann_populations(all_records, temperature=temperature)
            st.session_state.records = all_records
            st.session_state.loaded = True
            st.session_state.atom_mappings = {}
            st.session_state.mapping_selections = {}
            st.session_state.distance_criteria_records = [
                {"criterion": "NOE 1", "proton_or_group_A": "", "proton_or_group_B": ""}
            ]
            st.session_state.criteria_editor_revision += 1
            st.session_state.analysis_result_rows = []
            st.session_state.analysis_detail_rows = []
            st.session_state.analysis_skipped_messages = []
            st.success(t("loaded_sdf", count=len(all_records), files=len(sdf_files)))
else:
    st.subheader(t("upload_gaussian"))
    st.write(t("upload_gaussian_desc"))

    if "gaussian_group_count" not in st.session_state:
        st.session_state.gaussian_group_count = 1
    if "gaussian_group_specs" not in st.session_state:
        st.session_state.gaussian_group_specs = {}

    group_specs = []

    for i in range(st.session_state.gaussian_group_count):
        with st.container(border=True):
            group_name = st.text_input(
                t("group_name", number=i + 1),
                value=st.session_state.gaussian_group_specs.get(i, {}).get("group", f"Group {i + 1}"),
                key=f"gaussian_group_name_{i}",
            )
            files = st.file_uploader(
                t("group_files", number=i + 1),
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
    if last_group_has_files and st.button(t("add_group")):
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
        st.markdown(t("current_assignments"))
        st.dataframe(pd.DataFrame(summary_rows).rename(columns={"group": t("col_group"), "candidate": t("col_candidate"), "file_name": t("col_file")}), use_container_width=True)

    if st.button(t("load_gaussian")):
        if not populated_groups:
            st.error(t("need_gaussian"))
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
            st.session_state.distance_criteria_records = [
                {"criterion": "NOE 1", "proton_or_group_A": "", "proton_or_group_B": ""}
            ]
            st.session_state.criteria_editor_revision += 1
            st.session_state.analysis_result_rows = []
            st.session_state.analysis_detail_rows = []
            st.session_state.analysis_skipped_messages = []
            st.success(t(
                "loaded_gaussian",
                count=len(all_records),
                files=total_files,
                groups=len(populated_groups),
            ))


# ==============================
# Loaded summary
# ==============================
if st.session_state.loaded and st.session_state.records:
    st.header(t("loaded_conformers"))
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
    summary_display = summary_df.rename(columns={
        "group": t("col_group"),
        "candidate": t("col_candidate"),
        "source": t("col_source"),
        "conformer_id": t("col_conformer"),
        "n_atoms": t("col_atoms"),
        "energy": t("col_energy"),
        "free_energy": t("col_free_energy"),
        "population": t("col_population"),
        "population_mode": t("col_population_mode"),
    })
    st.dataframe(summary_display, use_container_width=True, height=280)

    # ==============================
    # Step 3: intuitive atom mapping
    # ==============================
    st.header(t("atom_mapping"))
    st.write(t("atom_mapping_desc"))

    unique_candidates = sorted({(r.group, r.candidate) for r in recs})
    candidate_options = [f"{g} :: {c}" for g, c in unique_candidates]
    selected_candidate = st.selectbox(t("candidate_mapping"), candidate_options)
    sel_group, sel_cand = selected_candidate.split(" :: ", 1)
    candidate_key = selected_candidate
    candidate_records = [r for r in recs if r.group == sel_group and r.candidate == sel_cand]
    example = candidate_records[0]

    mappings_for_candidate: List[Dict[str, object]] = st.session_state.atom_mappings.setdefault(
        candidate_key, []
    )
    current_selection = st.session_state.mapping_selections.get(candidate_key, [])

    mapping_notice = st.session_state.pop("mapping_notice", None)
    if mapping_notice:
        notice_type, notice_text = mapping_notice
        if notice_type == "success":
            st.success(notice_text)
        elif notice_type == "warning":
            st.warning(notice_text)
        else:
            st.info(notice_text)

    candidate_examples: Dict[str, ConformerRecord] = {}
    for group, candidate in unique_candidates:
        key = f"{group} :: {candidate}"
        candidate_examples[key] = next(
            r for r in recs if r.group == group and r.candidate == candidate
        )

    other_candidates = [option for option in candidate_options if option != candidate_key]
    if other_candidates:
        with st.expander(t("copy_mappings"), expanded=False):
            st.caption(t("copy_once"))

            copy_source = st.selectbox(
                t("copy_from"),
                other_candidates,
                key=f"copy_source_{candidate_key}",
            )
            copy_mode = st.radio(
                t("copy_mode"),
                ["labels_atoms", "labels_only"],
                format_func=lambda x: t(x),
                horizontal=True,
                key=f"copy_mode_{candidate_key}",
            )

            source_items = st.session_state.atom_mappings.get(copy_source, [])
            source_example = candidate_examples[copy_source]
            numbering_compatible = atom_numbering_is_compatible(source_example, example)

            if copy_mode == "labels_atoms":
                if numbering_compatible:
                    st.success(t("compatible"))
                else:
                    st.warning(t("incompatible"))
            else:
                st.info(t("labels_only_info"))

            if mappings_for_candidate:
                st.caption(t("copy_replace_note"))

            copy_disabled = (
                not source_items
                or (copy_mode == "labels_atoms" and not numbering_compatible)
            )
            if st.button(
                t("copy_to_candidate"),
                disabled=copy_disabled,
                use_container_width=True,
                key=f"copy_mapping_{candidate_key}",
            ):
                include_atoms = copy_mode == "labels_atoms"
                st.session_state.atom_mappings[candidate_key] = clone_mapping_items(
                    source_items,
                    include_atom_numbers=include_atoms,
                )
                st.session_state.mapping_selections[candidate_key] = []
                detail = t("detail_labels_atoms") if include_atoms else t("detail_labels_only")
                st.session_state.mapping_notice = (
                    "success",
                    t("copied_mapping", detail=detail, source=copy_source, destination=candidate_key),
                )
                st.rerun()

            st.markdown(t("apply_all"))
            st.caption(t("apply_all_desc"))
            confirm_bulk = st.checkbox(
                t("confirm_replace"),
                key=f"confirm_bulk_copy_{candidate_key}",
            )
            bulk_disabled = not mappings_for_candidate or not confirm_bulk
            if st.button(
                t("apply_all_button"),
                disabled=bulk_disabled,
                use_container_width=True,
                key=f"bulk_copy_mapping_{candidate_key}",
            ):
                copied_targets: List[str] = []
                skipped_targets: List[str] = []
                for target_key in other_candidates:
                    target_example = candidate_examples[target_key]
                    if atom_numbering_is_compatible(example, target_example):
                        st.session_state.atom_mappings[target_key] = clone_mapping_items(
                            mappings_for_candidate,
                            include_atom_numbers=True,
                        )
                        st.session_state.mapping_selections[target_key] = []
                        copied_targets.append(target_key)
                    else:
                        skipped_targets.append(target_key)

                message = t("bulk_result", copied=len(copied_targets))
                if skipped_targets:
                    message += t("bulk_skipped", skipped=len(skipped_targets))
                st.session_state.mapping_notice = ("success", message)
                st.rerun()

    left, right = st.columns([1.55, 1.0], gap="large")

    with left:
        st.markdown(t("click_atoms"))
        picked_atoms = atom_picker(
            example.atoms,
            example.coords,
            selected_atoms=current_selection,
            language=current_language(),
            key=f"atom_picker_{candidate_key}",
        )
        st.session_state.mapping_selections[candidate_key] = picked_atoms
        current_selection = picked_atoms

        st.caption(t("viewer_caption"))

    with right:
        st.markdown(t("new_mapping"))

        if current_selection:
            selected_details = [
                f"{example.atoms[number - 1]} {number}"
                for number in current_selection
                if 1 <= number <= len(example.atoms)
            ]
            st.success(t("selected", items=", ".join(selected_details)))
        else:
            st.info(t("none_selected"))

        label_mode = st.radio(
            t("label_entry"),
            ["build_label", "free_text"],
            format_func=lambda x: t(x),
            horizontal=True,
            key=f"label_mode_{candidate_key}",
        )

        if label_mode == "build_label":
            label_col1, label_col2 = st.columns([1.2, 1.0])
            position = label_col1.text_input(
                t("position_number"),
                value="",
                placeholder=t("position_placeholder"),
                key=f"position_{candidate_key}",
            ).strip()
            prime = label_col2.selectbox(
                t("prime"),
                ["", "′", "″", "‴"],
                key=f"prime_{candidate_key}",
            )
            proposed_label = f"H-{position}{prime}" if position else ""
            if proposed_label:
                st.markdown(t("display_label", label=proposed_label))
        else:
            proposed_label = st.text_input(
                t("label"),
                value="",
                placeholder=t("label_placeholder"),
                key=f"custom_label_{candidate_key}",
            ).strip()

        button_col1, button_col2 = st.columns(2)
        save_mapping = button_col1.button(
            t("save_mapping"),
            type="primary",
            use_container_width=True,
            key=f"save_mapping_{candidate_key}",
        )
        clear_selection = button_col2.button(
            t("clear_selection"),
            use_container_width=True,
            key=f"clear_selection_{candidate_key}",
        )

        if clear_selection:
            st.session_state.mapping_selections[candidate_key] = []
            st.rerun()

        if save_mapping:
            if not proposed_label:
                st.error(t("enter_label"))
            elif not current_selection:
                st.error(t("select_atom"))
            else:
                non_h = [n for n in current_selection if example.atoms[n - 1] != "H"]
                if non_h:
                    st.error(t("non_hydrogen", atoms=", ".join(map(str, non_h))))
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
                        st.success(t("added", label=proposed_label))
                    else:
                        existing["atom_numbers"] = list(current_selection)
                        existing["atom_indices"] = atom_indices
                        st.success(t("updated", label=proposed_label))
                    st.session_state.mapping_selections[candidate_key] = []
                    st.rerun()

        with st.expander(t("manual_fallback")):
            st.caption(t("manual_desc"))
            manual_raw = st.text_input(
                t("manual_numbers"),
                value=",".join(map(str, current_selection)),
                key=f"manual_selection_{candidate_key}",
            )
            if st.button(t("apply_manual"), key=f"apply_manual_{candidate_key}"):
                try:
                    manual_numbers = sorted({
                        int(x.strip())
                        for x in manual_raw.split(",")
                        if x.strip()
                    })
                    invalid = [n for n in manual_numbers if n < 1 or n > len(example.atoms)]
                    if invalid:
                        st.error(t("out_of_range", atoms=", ".join(map(str, invalid))))
                    else:
                        st.session_state.mapping_selections[candidate_key] = manual_numbers
                        st.rerun()
                except ValueError:
                    st.error(t("integer_error"))

    st.markdown(t("registered_mappings"))
    if not mappings_for_candidate:
        st.info(t("no_mappings"))
    else:
        for idx, item in enumerate(list(mappings_for_candidate)):
            with st.container(border=True):
                info_col, select_col, delete_col = st.columns([4.5, 1.5, 1.0])
                atom_numbers = [int(n) for n in item.get("atom_numbers", [])]
                atom_text = ", ".join(str(n) for n in atom_numbers) if atom_numbers else t("not_assigned")
                info_col.markdown(f"**{item['label']}**  \n{t('atoms_label')}: {atom_text}")
                if select_col.button(
                    t("show_edit"),
                    key=f"show_mapping_{candidate_key}_{idx}",
                    use_container_width=True,
                ):
                    st.session_state.mapping_selections[candidate_key] = list(item["atom_numbers"])
                    st.rerun()
                if delete_col.button(
                    t("delete"),
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
                t("col_group"): group,
                t("col_candidate"): candidate,
                t("col_registered"): len(candidate_mappings),
                t("col_labels"): ", ".join(str(x["label"]) for x in candidate_mappings),
            }
        )
    with st.expander(t("mapping_status")):
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    # ==============================
    # Step 4: criteria
    # ==============================
    st.header(t("distance_criteria"))
    st.write(t("criteria_desc"))

    criteria_internal = pd.DataFrame(
        normalize_criteria_records(st.session_state.distance_criteria_records)
    )
    criteria_column_map = {
        "criterion": t("col_criterion"),
        "proton_or_group_A": t("col_a"),
        "proton_or_group_B": t("col_b"),
    }
    criteria_default = criteria_internal.rename(columns=criteria_column_map)
    criteria_display_df = st.data_editor(
        criteria_default,
        num_rows="dynamic",
        use_container_width=True,
        key=f"global_distance_criteria_{st.session_state.criteria_editor_revision}_{current_language()}",
    )
    criteria_df = criteria_display_df.rename(
        columns={value: key for key, value in criteria_column_map.items()}
    )
    st.session_state.distance_criteria_records = normalize_criteria_records(
        criteria_df.fillna("").to_dict(orient="records")
    )

    # ==============================
    # Step 5: analysis
    # ==============================
    st.header(t("analysis"))
    if st.button(t("run_analysis"), type="primary"):
        if not distance_range_valid:
            st.error(t("distance_error"))
            st.stop()

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
            st.error(t("need_criterion"))
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
                    if label not in candidate_mapping or not candidate_mapping[label]
                })
                if missing_labels:
                    missing_text = "未設定のマッピング" if current_language() == "ja" else "missing mapping(s)"
                    skipped_messages.append(
                        f"{key}: {missing_text}: {', '.join(missing_labels)}"
                    )
                    continue

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
                        satisfies = float(minimum_distance) <= distance <= float(maximum_distance)
                        detail_rows.append(
                            {
                                "group": group,
                                "candidate": candidate,
                                "conformer_id": rec.conformer_id,
                                "minimum_distance_A": float(minimum_distance),
                                "maximum_distance_A": float(maximum_distance),
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
                        if not (float(minimum_distance) <= distance <= float(maximum_distance)):
                            satisfied_all = False
                            break

                    if satisfied_all:
                        pop = rec.population if rec.population is not None else fallback_pop
                        all_satisfied_pop += pop

                result_row = {
                    "group": group,
                    "candidate": candidate,
                    "minimum_distance_A": float(minimum_distance),
                    "maximum_distance_A": float(maximum_distance),
                    "all_criteria_same_conformer_population_sum": all_satisfied_pop,
                }
                result_row.update(
                    {f"population_sum::{name}": value for name, value in criterion_pop_sums.items()}
                )
                analysis_rows.append(result_row)

            st.session_state.analysis_skipped_messages = skipped_messages
            if analysis_rows:
                result_df = pd.DataFrame(analysis_rows).sort_values(
                    by=["all_criteria_same_conformer_population_sum"],
                    ascending=[False],
                )
                detail_df = pd.DataFrame(detail_rows)
                st.session_state.analysis_result_rows = result_df.to_dict(orient="records")
                st.session_state.analysis_detail_rows = detail_df.to_dict(orient="records")
            else:
                st.session_state.analysis_result_rows = []
                st.session_state.analysis_detail_rows = []
                st.warning(t("no_analysis"))

    if st.session_state.analysis_skipped_messages:
        st.warning(t(
            "skipped",
            messages="\n\n".join(st.session_state.analysis_skipped_messages),
        ))

    if st.session_state.analysis_result_rows:
        result_df = pd.DataFrame(st.session_state.analysis_result_rows)
        detail_df = pd.DataFrame(st.session_state.analysis_detail_rows)

        result_rename = {
            "group": t("col_group"),
            "candidate": t("col_candidate"),
            "minimum_distance_A": t("col_min"),
            "maximum_distance_A": t("col_max"),
            "all_criteria_same_conformer_population_sum": t("col_all_sum"),
        }
        for column in result_df.columns:
            if column.startswith("population_sum::"):
                result_rename[column] = f"{t('col_pop_sum_prefix')}::{column.split('::', 1)[1]}"
        result_display = result_df.rename(columns=result_rename)

        detail_display = detail_df.rename(columns={
            "group": t("col_group"),
            "candidate": t("col_candidate"),
            "conformer_id": t("col_conformer"),
            "minimum_distance_A": t("col_min"),
            "maximum_distance_A": t("col_max"),
            "criterion": t("col_criterion"),
            "mapping_A": t("col_a"),
            "mapping_B": t("col_b"),
            "distance_A": t("col_distance"),
            "population_percent": t("col_population"),
            "satisfies": t("col_satisfies"),
        })

        st.subheader(t("candidate_comparison"))
        st.dataframe(result_display, use_container_width=True)

        st.subheader(t("per_conformer"))
        st.dataframe(detail_display, use_container_width=True, height=320)

        csv_result = result_display.to_csv(index=False).encode("utf-8-sig")
        csv_detail = detail_display.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            t("download_comparison"),
            csv_result,
            "candidate_comparison.csv",
            "text/csv",
        )
        st.download_button(
            t("download_detail"),
            csv_detail,
            "per_conformer_detail.csv",
            "text/csv",
        )

    st.header(t("save_project"))
    st.write(t("save_project_desc"))
    st.text_input(t("project_name"), key="project_name")
    try:
        project_bytes = build_project_bytes()
        st.download_button(
            t("download_project"),
            data=project_bytes,
            file_name=safe_project_filename(st.session_state.project_name),
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )
        st.caption(t("project_contents"))
    except Exception as exc:
        st.error(t("prepare_project_error", error=exc))


st.markdown("---")
st.caption(t("footer_tip"))
st.caption(t(
    "version_line",
    app_version=APP_VERSION,
    project_version=PROJECT_VERSION,
))
st.caption(t("author_line"))

