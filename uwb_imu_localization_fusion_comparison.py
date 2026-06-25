"""
Compare UWB + IMU localization fusion methods.

Most fusion filters keep the original conservative policy: UWB position updates
use x/y, UWB range updates constrain horizontal position, and IMU z is applied
when available. The UWB-only Kalman baseline is different by design: it is a
constant-acceleration trajectory smoother that updates any valid UWB axes,
including z after the logical floor gate.

Example:
    python uwb_imu_localization_fusion_comparison.py ^
      --uwb data/uwb.csv ^
      --imu data/imu.csv ^
      --anchors config/anchors.yaml ^
      --ground-truth data/vicon.csv ^
      --output results ^
      --dt 0.02 ^
      --c-gamma 0.7 ^
      --a-gammas 0.2 0.4 0.6 0.75 0.9
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    HAVE_MATPLOTLIB = True
except Exception:
    plt = None
    HAVE_MATPLOTLIB = False


EPS = 1e-9

FILTER_COLOR_MAP = {
    "ground_truth": "#222222",
    "vicon": "#222222",
    "imu": "#1f77b4",
    "imu_z": "#1f77b4",
    "uwb": "#ff7f0e",
    "uwb_xy": "#ff7f0e",
    "uwb_z": "#ff7f0e",
    "uwb_z_valid": "#ff7f0e",
    "uwb_z_ignored": "#ff7f0e",
    "uwb_kalman": "#bcbd22",
    "ekf": "#2ca02c",
    "ukf": "#9467bd",
    "cekf": "#17becf",
    "adaptive_ekf": "#d62728",
    "adaptive_covariance_fusion": "#8c564b",
    "weighted_fusion": "#e377c2",
    "factor_graph": "#7f7f7f",
}


def plot_color(name: str) -> str:
    key = str(name).strip().lower()
    return FILTER_COLOR_MAP.get(key, "#4c78a8")


def display_filter_name(name: str) -> str:
    return {
        "ekf": "EKF",
        "ukf": "UKF",
        "cekf": "cEKF",
        "adaptive_ekf": "Adaptive EKF",
        "adaptive_covariance_fusion": "Adaptive cov.",
        "weighted_fusion": "Weighted",
        "factor_graph": "Factor graph",
        "imu": "IMU",
        "uwb": "UWB",
        "uwb_kalman": "UWB KF (CA)",
    }.get(str(name), str(name))


def copy_with_shift(df: pd.DataFrame | None, shift_s: float) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.copy()
    out.index = out.index.to_numpy(dtype=float) + float(shift_s)
    return out


def norm_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    by_norm = {norm_name(c): c for c in df.columns}
    for name in names:
        hit = by_norm.get(norm_name(name))
        if hit is not None:
            return hit
    return None


def parse_dt(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return pd.to_timedelta(str(value)).total_seconds()


def numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def rel_time_from_column(df: pd.DataFrame, dt: float) -> np.ndarray:
    col = first_col(
        df,
        [
            "time",
            "timestamp",
            "t",
            "time_s",
            "seconds",
            "elapsed_s",
            "Time",
            "datetime",
        ],
    )
    if col is None:
        return np.arange(len(df), dtype=float) * dt

    raw = df[col]
    num = pd.to_numeric(raw, errors="coerce")
    if num.notna().mean() >= 0.8:
        t = num.to_numpy(dtype=float)
        finite = t[np.isfinite(t)]
        if finite.size == 0:
            return np.arange(len(df), dtype=float) * dt
        med = float(np.nanmedian(np.abs(finite)))
        if med > 1e14:
            t = t / 1e9
        elif med > 1e11:
            t = t / 1000.0
        return t - np.nanmin(t[np.isfinite(t)])

    parsed = pd.to_datetime(raw, errors="coerce")
    if parsed.notna().sum() == 0:
        return np.arange(len(df), dtype=float) * dt
    t = parsed.astype("int64").to_numpy(dtype=float) / 1e9
    return t - np.nanmin(t[np.isfinite(t)])


def make_time_index(df: pd.DataFrame, t_s: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["time_s"] = t_s
    out = out[np.isfinite(out["time_s"])].sort_values("time_s")
    out = out.groupby("time_s", as_index=True).mean(numeric_only=True)
    return out


def axis_candidates(axis: str, prefixes: list[str]) -> list[str]:
    bases = {
        "x": ["x", "px", "pos_x", "position_x", "positionx", "tx", "east", "enu_x"],
        "y": ["y", "py", "pos_y", "position_y", "positiony", "ty", "north", "enu_y"],
        "z": ["z", "pz", "pos_z", "position_z", "positionz", "tz", "alt", "altitude", "height", "enu_z"],
    }[axis]
    out = []
    for prefix in prefixes:
        for base in bases:
            out += [f"{prefix}_{base}", f"{prefix}{base}", f"{prefix}.{base}", f"{prefix}-{base}"]
    out += bases
    return out


def extract_xyz(df: pd.DataFrame, prefixes: list[str]) -> pd.DataFrame | None:
    cols = {axis: first_col(df, axis_candidates(axis, prefixes)) for axis in "xyz"}
    if cols["x"] is None or cols["y"] is None:
        return None
    out = pd.DataFrame(index=df.index)
    for axis in "xyz":
        out[axis] = numeric_col(df, cols[axis]) if cols[axis] is not None else np.nan
    return out


def auto_scale_position(pos: pd.DataFrame) -> pd.DataFrame:
    out = pos.copy()
    vals = out[["x", "y", "z"]].to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size and np.nanmedian(np.abs(finite)) > 50.0:
        out[["x", "y", "z"]] = out[["x", "y", "z"]] / 1000.0
    return out


def apply_perm_sign_frame(df: pd.DataFrame | None, cols: list[str], perm: list[int] | None, sign: list[float] | None) -> pd.DataFrame | None:
    if df is None or perm is None:
        return df
    if len(perm) != 3:
        raise ValueError("--imu-perm must contain exactly three axes")
    sign = sign or [1.0, 1.0, 1.0]
    if len(sign) != 3:
        raise ValueError("--imu-sign must contain exactly three signs")
    perm0 = [int(p) - 1 if int(p) in (1, 2, 3) else int(p) for p in perm]
    if sorted(perm0) != [0, 1, 2]:
        raise ValueError(f"Invalid axis permutation: {perm}")
    out = df.copy()
    arr = out[cols].to_numpy(dtype=float)
    out[cols] = arr[:, perm0] * np.asarray(sign, dtype=float)
    return out


def extract_accel(df: pd.DataFrame) -> pd.DataFrame | None:
    cols = {
        "ax": first_col(df, ["ax", "Ax", "acc_x", "accel_x", "acceleration_x", "imu_ax"]),
        "ay": first_col(df, ["ay", "Ay", "acc_y", "accel_y", "acceleration_y", "imu_ay"]),
        "az": first_col(df, ["az", "Az", "acc_z", "accel_z", "acceleration_z", "imu_az"]),
    }
    if not all(cols.values()):
        return None
    return pd.DataFrame({k: numeric_col(df, c) for k, c in cols.items()}, index=df.index)


def parse_uwb_pos_log(path: Path, uwb_id: int | None) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ",POS," not in line:
            continue
        parts = line.strip().split(",")
        if len(parts) < 7:
            continue
        stamp = parts[0]
        if len(stamp) != 9 or not stamp.isdigit():
            continue
        try:
            drone_id = int(parts[2])
            x = float(parts[4])
            y = float(parts[5])
            z = float(parts[6])
            qf = float(parts[7]) if len(parts) > 7 else np.nan
        except ValueError:
            continue
        seconds = int(stamp[0:2]) * 3600 + int(stamp[2:4]) * 60 + int(stamp[4:6]) + int(stamp[6:9]) / 1000.0
        rows.append({"time_raw_s": seconds, "drone_id": drone_id, "x": x, "y": y, "z": z, "qf": qf})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No POS rows found in {path}")
    if uwb_id is None:
        uwb_id = int(sorted(df["drone_id"].dropna().unique())[0])
        print(f"UWB log contains multiple ids; using --uwb-id {uwb_id}")
    df = df[df["drone_id"] == uwb_id].copy()
    if df.empty:
        raise ValueError(f"No UWB POS rows for id {uwb_id}")
    df["time_s"] = df["time_raw_s"] - df["time_raw_s"].min()
    return make_time_index(df[["time_s", "x", "y", "z", "qf"]], df["time_s"].to_numpy(dtype=float))


def read_uwb(path: Path, dt: float, uwb_id: int | None) -> pd.DataFrame:
    if path.suffix.lower() == ".log":
        return parse_uwb_pos_log(path, uwb_id)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty")
    id_col = first_col(df, ["drone_id", "uwb_id", "tag_id", "tag", "id"])
    if id_col is not None:
        ids = pd.to_numeric(df[id_col], errors="coerce")
        if uwb_id is None and ids.notna().any():
            uwb_id = int(sorted(ids.dropna().unique())[0])
            print(f"UWB CSV contains ids; using --uwb-id {uwb_id}")
        if uwb_id is not None:
            df = df[ids == uwb_id].copy()
    return make_time_index(df, rel_time_from_column(df, dt))


def read_imu(
    path: Path,
    dt: float,
    pos_scale: float,
    imu_perm: list[int] | None = None,
    imu_sign: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    df = pd.read_csv(path)
    timed = make_time_index(df, rel_time_from_column(df, dt))
    pos = extract_xyz(timed, ["imu", "pos", "position", ""])
    if pos is not None:
        pos = auto_scale_position(pos * pos_scale)
        pos = apply_perm_sign_frame(pos, ["x", "y", "z"], imu_perm, imu_sign)
    acc = extract_accel(timed)
    if acc is not None and acc.notna().sum().sum() > 0:
        med_abs = np.nanmedian(np.abs(acc.to_numpy(dtype=float)))
        if med_abs > 50.0:
            acc = acc / 100.0
        acc = acc - acc.median(numeric_only=True)
        acc = apply_perm_sign_frame(acc, ["ax", "ay", "az"], imu_perm, imu_sign)
    return timed, pos, acc


def read_vicon_or_xyz(path: Path, dt: float, fps: float) -> pd.DataFrame:
    try:
        generic = pd.read_csv(path)
        t_s = rel_time_from_column(generic, dt)
        timed = make_time_index(generic, t_s)
        xyz = extract_xyz(timed, ["vicon", "gt", "truth", "position", ""])
        if xyz is not None:
            return auto_scale_position(xyz)
    except Exception:
        pass

    raw = pd.read_csv(path, skiprows=3, header=None).iloc[:, :5].copy()
    raw.columns = ["frame", "subframe", "tx_mm", "ty_mm", "tz_mm"]
    for col in raw.columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["frame", "tx_mm", "ty_mm", "tz_mm"])
    if raw.empty:
        raise ValueError(f"Could not parse ground-truth positions from {path}")
    t_s = (raw["frame"].to_numpy(dtype=float) - float(raw["frame"].iloc[0])) / fps
    out = pd.DataFrame(
        {
            "x": raw["tx_mm"].to_numpy(dtype=float) / 1000.0,
            "y": raw["ty_mm"].to_numpy(dtype=float) / 1000.0,
            "z": raw["tz_mm"].to_numpy(dtype=float) / 1000.0,
        },
        index=t_s,
    )
    out.index.name = "time_s"
    return out.groupby(level=0).mean().sort_index()


def load_anchors(path: Path | None) -> dict[str, np.ndarray]:
    if path is None or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        raw = yaml.safe_load(text)
    except Exception:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            raw = []
            current: dict[str, object] | None = None
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or stripped in {"anchors:", "units: m"}:
                    continue
                id_match = re.search(r"\bid\s*:\s*([A-Za-z0-9_.-]+)", stripped)
                if stripped.startswith("-") and id_match:
                    if current:
                        raw.append(current)
                    current = {"id": id_match.group(1)}
                    continue
                if current is None:
                    current = {}
                if id_match and "id" not in current:
                    current["id"] = id_match.group(1)
                pos_match = re.search(r"position\s*:\s*\[([^\]]+)\]", stripped)
                if pos_match:
                    current["position"] = [float(v.strip()) for v in pos_match.group(1).split(",")]
            if current:
                raw.append(current)
    if isinstance(raw, dict) and "anchors" in raw:
        raw = raw["anchors"]

    anchors: dict[str, np.ndarray] = {}

    def add(anchor_id: object, value: object) -> None:
        if isinstance(value, dict):
            pos = value.get("position", value.get("pos", value.get("xyz")))
            x = value.get("x", value.get("X"))
            y = value.get("y", value.get("Y"))
            z = value.get("z", value.get("Z", 0.0))
            if pos is not None and len(pos) >= 2:
                x, y = pos[0], pos[1]
                z = pos[2] if len(pos) > 2 else z
            if x is None or y is None:
                return
            anchors[str(anchor_id)] = np.array([float(x), float(y), float(z)], dtype=float)
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            z = value[2] if len(value) > 2 else 0.0
            anchors[str(anchor_id)] = np.array([float(value[0]), float(value[1]), float(z)], dtype=float)

    if isinstance(raw, dict):
        for key, value in raw.items():
            add(key, value)
    elif isinstance(raw, list):
        for i, value in enumerate(raw):
            aid = value.get("id", value.get("name", i)) if isinstance(value, dict) else i
            add(aid, value)
    return anchors


def manifest_sensor_overrides(manifest: dict, drone_name: str, sensor_name: str) -> dict:
    out = {}
    out.update(manifest.get("overrides", {}).get(drone_name, {}).get(sensor_name, {}) or {})
    out.update(manifest.get("drones", {}).get(drone_name, {}).get("overrides", {}).get(sensor_name, {}) or {})
    return out


def manifest_shift_seconds(manifest: dict, drone_name: str, sensor_name: str) -> float:
    over = manifest_sensor_overrides(manifest, drone_name, sensor_name)
    if "time_shift_seconds" in over:
        return float(over["time_shift_seconds"])
    drone = manifest.get("drones", {}).get(drone_name, {})
    for key in (f"{sensor_name}_time_shift_seconds", f"{sensor_name}_time_shift"):
        if key in drone:
            raw = drone[key]
            break
    else:
        raw = manifest.get(f"{sensor_name}_time_shift", 0.0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        try:
            td = pd.to_timedelta(raw).total_seconds()
            return 0.0 if abs(td) >= 300 else float(td)
        except Exception:
            return 0.0


def parse_position_offset(value: object | None) -> dict[str, float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: float(v) for k, v in value.items() if k in {"x", "y", "z"}}
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return {"x": float(value[0]), "y": float(value[1]), "z": float(value[2])}
    return None


def resolve_manifest_args(args: argparse.Namespace) -> dict | None:
    if not args.manifest:
        args.manifest_data = None
        return None
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exp_dir = Path(manifest.get("source_exp_dir", manifest_path.parent))
    args.exp_id = exp_dir.name
    drone_name = args.drone_name
    drones = manifest.get("drones", {})
    if drone_name not in drones:
        available = ", ".join(drones.keys())
        raise ValueError(f"Drone {drone_name!r} not found in {manifest_path}. Available: {available}")
    drone = drones[drone_name]

    if args.uwb is None and manifest.get("uwb", {}).get("path"):
        args.uwb = str(exp_dir / manifest["uwb"]["path"])
    if args.imu is None and drone.get("imu"):
        args.imu = str(exp_dir / drone["imu"])
    if args.imu is None and (exp_dir / drone_name / "imu.csv").exists():
        args.imu = str(exp_dir / drone_name / "imu.csv")
    if args.ground_truth is None and drone.get("vicon"):
        args.ground_truth = str(exp_dir / drone["vicon"])
    if args.ground_truth is None and (exp_dir / drone_name / "vicon.csv").exists():
        args.ground_truth = str(exp_dir / drone_name / "vicon.csv")
    if args.anchors is None:
        for candidate in (exp_dir / "anchors.yaml", exp_dir.parent / "anchors.yaml"):
            if candidate.exists():
                args.anchors = str(candidate)
                break
    if args.uwb_id is None and "uwb_id" in drone:
        args.uwb_id = int(drone["uwb_id"])

    if args.uwb_time_shift is None:
        args.uwb_time_shift = float(manifest.get("uwb_time_shift", manifest_shift_seconds(manifest, drone_name, "uwb")))
    if args.imu_time_shift is None:
        args.imu_time_shift = manifest_shift_seconds(manifest, drone_name, "imu")
    if args.gt_time_shift is None:
        args.gt_time_shift = manifest_shift_seconds(manifest, drone_name, "vicon")
    if getattr(args, "trim_t_end", None) is None and manifest.get("trim_t_end") is not None:
        args.trim_t_end = parse_dt(manifest["trim_t_end"])
    if args.imu_pos_scale is None:
        args.imu_pos_scale = float(manifest.get("imu_pos_scale", manifest.get("default_imu_pos_scale", 0.1)))

    imu_over = manifest_sensor_overrides(manifest, drone_name, "imu")
    if args.imu_perm is None and "perm" in imu_over:
        args.imu_perm = [int(v) for v in imu_over["perm"]]
    if args.imu_sign is None and "sign" in imu_over:
        args.imu_sign = [float(v) for v in imu_over["sign"]]
    if args.imu_position_offset is None:
        args.imu_position_offset = parse_position_offset(imu_over.get("position_offset_m"))
    if args.anchor_points is None:
        args.anchor_points = int(imu_over.get("anchor_points", 5))

    tuning = manifest.get("fusion_shift_tuning", {})
    if tuning.get("enable_adaptive_motion_q", False):
        args.enable_adaptive_motion_q = True
        if args.k_acc == 0.0:
            args.k_acc = float(tuning.get("k_acc", args.k_acc))
        if args.k_jerk == 0.0:
            args.k_jerk = float(tuning.get("k_jerk", args.k_jerk))
        args.adaptive_q_max_scale = float(tuning.get("adaptive_q_max_scale", args.adaptive_q_max_scale))
    args.drone_mass_scale = float(tuning.get("drone_mass_scale", args.drone_mass_scale))
    args.mass_q_exponent = float(tuning.get("mass_q_exponent", args.mass_q_exponent))
    args.uwb_kf_q_pos = float(tuning.get("uwb_kf_q_pos", args.uwb_kf_q_pos))
    args.uwb_kf_q_vel = float(tuning.get("uwb_kf_q_vel", args.uwb_kf_q_vel))
    args.uwb_kf_q_acc = float(tuning.get("uwb_kf_q_acc", args.uwb_kf_q_acc))
    args.uwb_kf_q_jerk = float(tuning.get("uwb_kf_q_jerk", args.uwb_kf_q_jerk))
    args.r_uwb_z = float(tuning.get("r_uwb_z", args.r_uwb_z))
    args.uwb_z_floor_threshold = float(tuning.get("uwb_z_floor_threshold", args.uwb_z_floor_threshold))
    if tuning.get("enable_temperature_proxy", False):
        args.enable_temperature_proxy = True
        args.temp_start_c = float(tuning.get("temp_start_c", args.temp_start_c))
        args.temp_end_c = float(tuning.get("temp_end_c", args.temp_end_c))
        args.temp_profile_power = float(tuning.get("temp_profile_power", args.temp_profile_power))
        if tuning.get("temp_ref_c", None) is not None:
            args.temp_ref_c = float(tuning["temp_ref_c"])
        if args.temp_z_bias_m_per_c == 0.0:
            args.temp_z_bias_m_per_c = float(tuning.get("temp_z_bias_m_per_c", args.temp_z_bias_m_per_c))
        args.temp_r_imu_z_gain = float(tuning.get("temp_r_imu_z_gain", args.temp_r_imu_z_gain))
    if tuning.get("enable_uwb_xy_nis_gate", False):
        args.enable_uwb_xy_nis_gate = True
        args.uwb_xy_nis_threshold = float(tuning.get("uwb_xy_nis_threshold", args.uwb_xy_nis_threshold))
        args.uwb_xy_gate_action = str(tuning.get("uwb_xy_gate_action", args.uwb_xy_gate_action))
        args.uwb_xy_gate_inflate_scale = float(tuning.get("uwb_xy_gate_inflate_scale", args.uwb_xy_gate_inflate_scale))

    args.manifest_data = manifest
    return manifest


def validate_required_paths(args: argparse.Namespace) -> None:
    missing = [name for name in ("uwb",) if getattr(args, name) is None]
    if missing:
        raise ValueError(f"Missing required input(s): {', '.join('--' + m for m in missing)}")
    for name in ("uwb", "imu", "ground_truth", "anchors"):
        value = getattr(args, name, None)
        if value and not Path(value).exists():
            if name == "anchors":
                print(f"Anchor file not found, continuing unless UWB uses ranges: {value}")
            else:
                raise FileNotFoundError(value)


def write_tuned_manifest(args: argparse.Namespace, out_dir: Path) -> None:
    manifest = getattr(args, "manifest_data", None)
    uwb_kf_customized = (
        args.uwb_kf_q_pos != 0.04
        or args.uwb_kf_q_vel != 0.45
        or args.uwb_kf_q_acc != 1.50
        or args.uwb_kf_q_jerk != 5.0
        or args.r_uwb_z != 0.35
        or args.uwb_z_floor_threshold != 0.0
    )
    if not manifest or not (
        args.auto_tune_shifts
        or args.tune_adaptive_motion_q
        or args.enable_temperature_proxy
        or args.enable_adaptive_motion_q
        or args.drone_mass_scale != 1.0
        or args.enable_uwb_xy_nis_gate
        or uwb_kf_customized
    ):
        return
    tuned = json.loads(json.dumps(manifest))
    drone_name = args.drone_name
    tuned["uwb_time_shift"] = float(args.uwb_time_shift)
    drone = tuned.setdefault("drones", {}).setdefault(drone_name, {})
    overrides = drone.setdefault("overrides", {}).setdefault("imu", {})
    overrides["time_shift_seconds"] = float(args.imu_time_shift)
    if getattr(args, "imu_offset_applied", None):
        overrides["position_offset_m"] = args.imu_offset_applied
    if args.imu_perm is not None:
        overrides["perm"] = [int(v) for v in args.imu_perm]
    if args.imu_sign is not None:
        overrides["sign"] = [float(v) for v in args.imu_sign]
    tuned.setdefault("fusion_shift_tuning", {})
    tuned["source_exp_dir"] = str(manifest.get("source_exp_dir", Path(args.manifest).parent))
    tuned["fusion_shift_tuning"].update(
        {
            "drone_name": drone_name,
            "dt": float(args.dt),
            "shift_search_radius": float(args.shift_search_radius),
            "shift_search_step": float(args.shift_search_step),
            "auto_tune_imu_offset": args.auto_tune_imu_offset,
            "enable_adaptive_motion_q": bool(args.enable_adaptive_motion_q),
            "k_acc": float(args.k_acc),
            "k_jerk": float(args.k_jerk),
            "adaptive_q_max_scale": float(args.adaptive_q_max_scale),
            "drone_mass_scale": float(args.drone_mass_scale),
            "mass_q_exponent": float(args.mass_q_exponent),
            "uwb_kf_motion_model": "constant_acceleration",
            "uwb_kf_q_pos": float(args.uwb_kf_q_pos),
            "uwb_kf_q_vel": float(args.uwb_kf_q_vel),
            "uwb_kf_q_acc": float(args.uwb_kf_q_acc),
            "uwb_kf_q_jerk": float(args.uwb_kf_q_jerk),
            "r_uwb_z": float(args.r_uwb_z),
            "uwb_z_floor_threshold": float(args.uwb_z_floor_threshold),
            "enable_temperature_proxy": bool(args.enable_temperature_proxy),
            "temp_start_c": float(args.temp_start_c),
            "temp_end_c": float(args.temp_end_c),
            "temp_profile_power": float(args.temp_profile_power),
            "temp_ref_c": None if args.temp_ref_c is None else float(args.temp_ref_c),
            "temp_z_bias_m_per_c": float(getattr(args, "temp_z_bias_m_per_c", 0.0)),
            "auto_tune_temp_z_bias": bool(args.auto_tune_temp_z_bias),
            "temp_r_imu_z_gain": float(args.temp_r_imu_z_gain),
            "enable_uwb_xy_nis_gate": bool(args.enable_uwb_xy_nis_gate),
            "uwb_xy_nis_threshold": float(args.uwb_xy_nis_threshold),
            "uwb_xy_gate_action": args.uwb_xy_gate_action,
            "uwb_xy_gate_inflate_scale": float(args.uwb_xy_gate_inflate_scale),
            "note": "Generated by uwb_imu_localization_fusion_comparison.py; original manifest was not modified.",
        }
    )
    path = Path(args.save_tuned_manifest) if args.save_tuned_manifest else out_dir / "tuned_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tuned, indent=2), encoding="utf-8")


@dataclass
class UWBSource:
    mode: str
    position: pd.DataFrame | None
    ranges: pd.DataFrame | None
    anchor_ids: list[str]
    anchor_pos: np.ndarray


def detect_uwb_source(uwb: pd.DataFrame, anchors: dict[str, np.ndarray]) -> UWBSource:
    pos = extract_xyz(uwb, ["uwb", "pos", "position", ""])
    if pos is not None and pos[["x", "y"]].notna().any().all():
        return UWBSource("position", auto_scale_position(pos), None, [], np.empty((0, 3)))

    range_col = first_col(uwb, ["range", "range_m", "distance", "dist", "uwb_range", "r"])
    anchor_col = first_col(uwb, ["anchor_id", "anchor", "anchorid", "beacon", "beacon_id", "base"])
    if range_col and anchor_col:
        long = uwb[[anchor_col, range_col]].copy()
        long["time_s"] = uwb.index.to_numpy(dtype=float)
        long[range_col] = pd.to_numeric(long[range_col], errors="coerce")
        ranges = long.pivot_table(index="time_s", columns=anchor_col, values=range_col, aggfunc="median")
    else:
        ranges = pd.DataFrame(index=uwb.index)
        for col in uwb.columns:
            if col == "time_s":
                continue
            n = norm_name(col)
            if n.startswith(("range", "dist")) or any(norm_name(aid) in n for aid in anchors):
                s = pd.to_numeric(uwb[col], errors="coerce")
                if s.notna().sum() > 1:
                    ranges[str(col)] = s

    if ranges.empty:
        raise ValueError("UWB must contain x/y[/z] columns or range measurements")
    if not anchors:
        raise ValueError("UWB contains ranges, so a valid --anchors YAML/JSON file is required")

    range_cols = [str(c) for c in ranges.columns]
    resolved_ids = []
    positions = []
    for col in range_cols:
        if col in anchors:
            aid = col
        else:
            hits = [aid for aid in anchors if norm_name(aid) in norm_name(col)]
            if not hits:
                raise ValueError(f"Range column {col!r} does not match any anchor id")
            aid = hits[0]
        resolved_ids.append(aid)
        positions.append(anchors[aid])
    ranges.columns = resolved_ids
    return UWBSource("ranges", None, ranges.sort_index(), resolved_ids, np.vstack(positions))


def interp_df(df: pd.DataFrame | None, timeline: np.ndarray) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    out = pd.DataFrame(index=timeline)
    x_old = df.index.to_numpy(dtype=float)
    for col in df.columns:
        y_old = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x_old) & np.isfinite(y_old)
        if ok.sum() == 0:
            out[col] = np.nan
        elif ok.sum() == 1:
            out[col] = y_old[ok][0]
        else:
            out[col] = np.interp(timeline, x_old[ok], y_old[ok], left=np.nan, right=np.nan)
    out.index.name = "time_s"
    return out


def derive_velocity(pos: np.ndarray | None, dt: float) -> np.ndarray | None:
    if pos is None:
        return None
    vel = np.full_like(pos, np.nan, dtype=float)
    for i in range(pos.shape[1]):
        s = pd.Series(pos[:, i]).interpolate(limit_direction="both")
        if s.notna().sum() > 1:
            vel[:, i] = np.gradient(s.to_numpy(dtype=float), dt)
    return vel


def interpolate_matrix(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    for axis in range(arr.shape[1]):
        s = pd.Series(arr[:, axis], dtype=float).interpolate(limit_direction="both")
        if s.notna().sum() > 0:
            out[:, axis] = s.to_numpy(dtype=float)
    return out


def compute_motion_q_profile(
    imu_acc: np.ndarray | None,
    imu_vel: np.ndarray | None,
    dt: float,
    q_vel_base: float,
    enabled: bool,
    k_acc: float,
    k_jerk: float,
    max_scale: float,
    smooth_window: int,
    drone_mass_scale: float = 1.0,
    mass_q_exponent: float = 0.5,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    n = 0
    if imu_acc is not None:
        n = len(imu_acc)
    elif imu_vel is not None:
        n = len(imu_vel)
    if n == 0:
        return None, None, None

    acc = interpolate_matrix(imu_acc)
    if acc is None or not np.isfinite(acc).any():
        vel = interpolate_matrix(imu_vel)
        if vel is not None and np.isfinite(vel).any():
            acc = np.gradient(vel, dt, axis=0)
    if acc is None or not np.isfinite(acc).any():
        zeros = np.zeros(n, dtype=float)
        mass_factor = max(float(drone_mass_scale), EPS) ** (-float(mass_q_exponent))
        q_eff = np.full(n, float(q_vel_base) * mass_factor, dtype=float)
        return zeros, zeros, q_eff

    acc = np.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)
    jerk = np.gradient(acc, dt, axis=0) if len(acc) > 1 else np.zeros_like(acc)
    acc_norm = np.linalg.norm(acc, axis=1)
    jerk_norm = np.linalg.norm(jerk, axis=1)

    if not enabled:
        mass_factor = max(float(drone_mass_scale), EPS) ** (-float(mass_q_exponent))
        return acc_norm, jerk_norm, np.full(n, float(q_vel_base) * mass_factor, dtype=float)

    scale = 1.0 + float(k_acc) * acc_norm + float(k_jerk) * jerk_norm
    scale = np.clip(scale, 1.0, max(float(max_scale), 1.0))
    if smooth_window and smooth_window > 1:
        scale = (
            pd.Series(scale)
            .rolling(int(smooth_window), center=True, min_periods=1)
            .mean()
            .to_numpy(dtype=float)
        )
    mass_factor = max(float(drone_mass_scale), EPS) ** (-float(mass_q_exponent))
    q_eff = float(q_vel_base) * scale * mass_factor
    return acc_norm, jerk_norm, q_eff


def build_temperature_profile(
    n: int,
    start_c: float,
    end_c: float,
    profile_power: float,
) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=float)
    u = np.linspace(0.0, 1.0, n)
    u = np.power(u, max(float(profile_power), EPS))
    return float(start_c) + (float(end_c) - float(start_c)) * u


def apply_temperature_proxy_to_imu_z(
    imu_pos: np.ndarray | None,
    gt_pos: np.ndarray | None,
    temp_c: np.ndarray | None,
    temp_ref_c: float,
    z_bias_m_per_c: float,
    auto_tune: bool,
) -> tuple[np.ndarray | None, float]:
    if imu_pos is None or temp_c is None:
        return imu_pos, float(z_bias_m_per_c)
    out = imu_pos.copy()
    slope = float(z_bias_m_per_c)
    if auto_tune and gt_pos is not None:
        residual = out[:, 2] - gt_pos[:, 2]
        t_center = temp_c - np.nanmedian(temp_c)
        r_center = residual - np.nanmedian(residual)
        ok = np.isfinite(t_center) & np.isfinite(r_center)
        denom = float(np.sum(t_center[ok] * t_center[ok])) if ok.any() else 0.0
        if denom > EPS:
            slope = float(np.sum(t_center[ok] * r_center[ok]) / denom)
    out[:, 2] = out[:, 2] - slope * (temp_c - float(temp_ref_c))
    return out, slope


def build_r_imu_z_profile(
    n: int,
    r_imu_z_base: float,
    temp_c: np.ndarray | None,
    temp_ref_c: float,
    temp_gain: float,
) -> np.ndarray:
    base = np.full(n, float(r_imu_z_base), dtype=float)
    if temp_c is None or temp_gain == 0.0:
        return base
    scale = 1.0 + float(temp_gain) * np.abs(temp_c - float(temp_ref_c))
    return base * np.maximum(scale, 1.0)


def score_sensor_shift(
    sensor: pd.DataFrame | None,
    gt: pd.DataFrame | None,
    cols: list[str],
    shift_s: float,
    dt: float,
    allow_offset: bool = False,
) -> tuple[float, np.ndarray]:
    if sensor is None or gt is None or sensor.empty or gt.empty:
        return float("nan"), np.zeros(len(cols), dtype=float)
    shifted = copy_with_shift(sensor[cols], shift_s)
    start = max(float(shifted.index.min()), float(gt.index.min()))
    end = min(float(shifted.index.max()), float(gt.index.max()))
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return float("nan"), np.zeros(len(cols), dtype=float)
    timeline = np.arange(start, end + dt / 2, dt, dtype=float)
    if len(timeline) < 5:
        return float("nan"), np.zeros(len(cols), dtype=float)
    s = interp_df(shifted, timeline)
    g = interp_df(gt[cols], timeline)
    if s is None or g is None:
        return float("nan"), np.zeros(len(cols), dtype=float)
    sv = s[cols].to_numpy(dtype=float)
    gv = g[cols].to_numpy(dtype=float)
    ok = np.isfinite(sv).all(axis=1) & np.isfinite(gv).all(axis=1)
    if ok.sum() < 5:
        return float("nan"), np.zeros(len(cols), dtype=float)
    offset = np.nanmedian(gv[ok] - sv[ok], axis=0) if allow_offset else np.zeros(len(cols), dtype=float)
    err = sv[ok] + offset - gv[ok]
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1)))), offset


def tune_one_shift(
    sensor: pd.DataFrame | None,
    gt: pd.DataFrame | None,
    cols: list[str],
    base_shift: float,
    dt: float,
    radius: float,
    step: float,
    allow_offset: bool = False,
) -> tuple[float, float, np.ndarray, pd.DataFrame]:
    deltas = np.arange(-radius, radius + step / 2, step, dtype=float)
    rows = []
    best = (float("inf"), base_shift, np.zeros(len(cols), dtype=float))
    for delta in deltas:
        candidate = float(base_shift + delta)
        score, offset = score_sensor_shift(sensor, gt, cols, candidate, dt, allow_offset=allow_offset)
        rows.append({"shift_s": candidate, "delta_s": float(delta), "score_rmse_m": score, **{f"offset_{c}": offset[i] for i, c in enumerate(cols)}})
        if np.isfinite(score) and score < best[0]:
            best = (score, candidate, offset)
    return best[1], best[0], best[2], pd.DataFrame(rows)


def apply_constant_offset(df: pd.DataFrame | None, cols: list[str], offset: np.ndarray) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.copy()
    for i, col in enumerate(cols):
        out[col] = out[col] + float(offset[i])
    return out


def apply_named_position_offset(df: pd.DataFrame | None, offset: dict[str, float] | None) -> pd.DataFrame | None:
    if df is None or not offset:
        return df
    out = df.copy()
    for col, value in offset.items():
        if col in out.columns:
            out[col] = out[col] + float(value)
    return out


def align_imu_to_vicon_start(
    imu_pos: pd.DataFrame | None,
    gt_pos: pd.DataFrame | None,
    anchor_points: int,
) -> tuple[pd.DataFrame | None, np.ndarray]:
    """Translate resampled IMU position into the Vicon coordinate frame.

    The offset is the mean Vicon-minus-IMU position over the first K rows where
    both sources have finite x/y/z. This is intentionally applied after
    perm/sign, time shifts, overlap trimming, and resampling.
    """
    shift = np.full(3, np.nan, dtype=float)
    if imu_pos is None or gt_pos is None or imu_pos.empty or gt_pos.empty:
        return imu_pos, shift
    cols = ["x", "y", "z"]
    if not set(cols).issubset(imu_pos.columns) or not set(cols).issubset(gt_pos.columns):
        return imu_pos, shift
    im = imu_pos[cols].to_numpy(dtype=float)
    gt = gt_pos[cols].to_numpy(dtype=float)
    valid = np.isfinite(im).all(axis=1) & np.isfinite(gt).all(axis=1)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return imu_pos, shift
    idx = idx[: max(int(anchor_points), 1)]
    delta = gt[idx] - im[idx]
    with np.errstate(invalid="ignore"):
        shift = np.nanmean(delta, axis=0)
    out = imu_pos.copy()
    for i, col in enumerate(cols):
        if np.isfinite(shift[i]):
            out[col] = pd.to_numeric(out[col], errors="coerce") + shift[i]
    return out, shift


def first_valid(values: np.ndarray | None, axis: int) -> float | None:
    if values is None:
        return None
    col = values[:, axis]
    valid = col[np.isfinite(col)]
    return float(valid[0]) if valid.size else None


def build_uwb_axis_validity(uwb_pos: np.ndarray | None, z_floor_threshold: float) -> np.ndarray | None:
    if uwb_pos is None:
        return None
    valid = np.isfinite(uwb_pos[:, :3])
    z = uwb_pos[:, 2]
    valid[:, 2] &= np.isfinite(z) & (z >= float(z_floor_threshold))
    return valid


def first_valid_with_axis_gate(values: np.ndarray | None, valid_axes: np.ndarray | None, axis: int) -> float | None:
    if values is None:
        return None
    col = values[:, axis]
    valid = np.isfinite(col)
    if valid_axes is not None:
        valid &= valid_axes[:, axis]
    hits = col[valid]
    return float(hits[0]) if hits.size else None


def gated_uwb_axis(data: "Prepared", axis: int) -> np.ndarray | None:
    if data.uwb_pos is None:
        return None
    out = data.uwb_pos[:, axis].copy()
    if data.uwb_valid_axes is not None:
        out[~data.uwb_valid_axes[:, axis]] = np.nan
    return out


def range_h_xy_only(state: np.ndarray, anchors: np.ndarray, z_ref: float | None = None) -> np.ndarray:
    z = state[2] if z_ref is None else z_ref
    d = state[:2] - anchors[:, :2]
    dz = z - anchors[:, 2]
    return np.sqrt(np.sum(d * d, axis=1) + dz * dz + EPS)


def range_H_xy_only(state: np.ndarray, anchors: np.ndarray, z_ref: float | None = None) -> np.ndarray:
    pred = range_h_xy_only(state, anchors, z_ref)
    H = np.zeros((len(anchors), 6), dtype=float)
    H[:, 0] = (state[0] - anchors[:, 0]) / pred
    H[:, 1] = (state[1] - anchors[:, 1]) / pred
    return H


def k_update(x: np.ndarray, P: np.ndarray, z: np.ndarray, h: np.ndarray, H: np.ndarray, R: np.ndarray):
    innov = z - h
    S = H @ P @ H.T + R
    S = 0.5 * (S + S.T) + np.eye(S.shape[0]) * EPS
    K = P @ H.T @ np.linalg.pinv(S)
    x_new = x + K @ innov
    I = np.eye(P.shape[0])
    P_new = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
    return x_new, 0.5 * (P_new + P_new.T), innov, S


def likelihood(innov: np.ndarray | None, S: np.ndarray | None) -> float:
    if innov is None or S is None or len(innov) == 0:
        return 1.0
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        return 1e-12
    val = -0.5 * (float(innov.T @ np.linalg.pinv(S) @ innov) + len(innov) * math.log(2 * math.pi) + logdet)
    return float(np.clip(math.exp(np.clip(val, -60, 30)), 1e-12, 1e12))


@dataclass
class Prepared:
    exp_id: str
    drone_name: str
    time_s: np.ndarray
    uwb_mode: str
    uwb_pos: np.ndarray | None
    ranges: np.ndarray | None
    anchor_pos: np.ndarray
    known_anchor_ids: list[str]
    known_anchor_pos: np.ndarray
    imu_pos: np.ndarray | None
    imu_vel: np.ndarray | None
    imu_acc: np.ndarray | None
    motion_acc_norm: np.ndarray | None
    motion_jerk_norm: np.ndarray | None
    q_vel_eff: np.ndarray | None
    temp_c: np.ndarray | None
    r_imu_z_eff: np.ndarray | None
    temp_z_bias_m_per_c: float
    gt_pos: np.ndarray | None
    uwb_xy_obs: np.ndarray | None
    uwb_valid_axes: np.ndarray | None
    uwb_z_floor_threshold_m: float
    shift_tuning: pd.DataFrame | None = None
    adaptive_q_tuning: pd.DataFrame | None = None


def multilaterate_xy(ranges: np.ndarray, anchors: np.ndarray, z_ref: float) -> np.ndarray:
    ok = np.isfinite(ranges)
    if ok.sum() < 3 or not np.isfinite(z_ref):
        return np.array([np.nan, np.nan])
    a = anchors[ok]
    r = ranges[ok]
    a0 = a[0]
    r0 = r[0]
    A, b = [], []
    for i in range(1, len(r)):
        ai = a[i]
        A.append([2 * (ai[0] - a0[0]), 2 * (ai[1] - a0[1])])
        b.append(
            r0 * r0
            - r[i] * r[i]
            + ai[0] ** 2
            - a0[0] ** 2
            + ai[1] ** 2
            - a0[1] ** 2
            + (z_ref - ai[2]) ** 2
            - (z_ref - a0[2]) ** 2
        )
    try:
        xy, *_ = np.linalg.lstsq(np.asarray(A), np.asarray(b), rcond=None)
        return xy
    except np.linalg.LinAlgError:
        return np.array([np.nan, np.nan])


def build_uwb_xy_obs(prep: Prepared) -> np.ndarray | None:
    if prep.uwb_mode == "position" and prep.uwb_pos is not None:
        return prep.uwb_pos[:, :2]
    if prep.ranges is None:
        return None
    z = np.zeros(len(prep.time_s))
    if prep.imu_pos is not None and np.isfinite(prep.imu_pos[:, 2]).any():
        z = pd.Series(prep.imu_pos[:, 2]).interpolate(limit_direction="both").to_numpy(dtype=float)
    xy = np.full((len(prep.time_s), 2), np.nan)
    for k in range(len(prep.time_s)):
        xy[k] = multilaterate_xy(prep.ranges[k], prep.anchor_pos, z[k])
    return xy


def prepare(args: argparse.Namespace) -> Prepared:
    dt = args.dt
    uwb_df = read_uwb(Path(args.uwb), dt, args.uwb_id)
    imu_pos_df = None
    imu_acc_df = None
    if args.imu:
        _, imu_pos_df, imu_acc_df = read_imu(Path(args.imu), dt, args.imu_pos_scale, args.imu_perm, args.imu_sign)
    gt_df = read_vicon_or_xyz(Path(args.ground_truth), dt, args.vicon_fps) if args.ground_truth else None
    anchors = load_anchors(Path(args.anchors)) if args.anchors else {}
    uwb = detect_uwb_source(uwb_df, anchors)
    gt_for_tuning = copy_with_shift(gt_df, args.gt_time_shift)
    if not args.auto_tune_shifts:
        imu_pos_df = apply_named_position_offset(imu_pos_df, args.imu_position_offset)

    tuning_tables = []
    if args.auto_tune_shifts:
        if gt_for_tuning is None:
            raise ValueError("--auto-tune-shifts requires --ground-truth or a manifest with vicon")
        if uwb.mode == "position" and uwb.position is not None:
            best_uwb, best_uwb_score, _, table = tune_one_shift(
                uwb.position,
                gt_for_tuning,
                ["x", "y"],
                args.uwb_time_shift,
                dt,
                args.shift_search_radius,
                args.shift_search_step,
                allow_offset=False,
            )
            table.insert(0, "sensor", "uwb_xy")
            tuning_tables.append(table)
            args.uwb_time_shift = best_uwb
            print(f"Tuned UWB time shift: {best_uwb:.3f}s (XY RMSE {best_uwb_score:.4f} m)")
        if imu_pos_df is not None:
            offset_mode = args.auto_tune_imu_offset
            cols = ["z"] if offset_mode == "z" else ["x", "y", "z"]
            best_imu, best_imu_score, offset, table = tune_one_shift(
                imu_pos_df,
                gt_for_tuning,
                cols,
                args.imu_time_shift,
                dt,
                args.shift_search_radius,
                args.shift_search_step,
                allow_offset=(offset_mode != "none"),
            )
            table.insert(0, "sensor", "imu_" + "".join(cols))
            tuning_tables.append(table)
            args.imu_time_shift = best_imu
            print(f"Tuned IMU time shift: {best_imu:.3f}s ({'/'.join(cols)} RMSE {best_imu_score:.4f} m)")
            if offset_mode != "none":
                imu_pos_df = apply_constant_offset(imu_pos_df, cols, offset)
                args.imu_offset_applied = {col: float(offset[i]) for i, col in enumerate(cols)}
                print(f"Applied IMU position offset: {args.imu_offset_applied}")

    uwb_df = copy_with_shift(uwb_df, args.uwb_time_shift)
    if uwb.position is not None:
        uwb.position = copy_with_shift(uwb.position, args.uwb_time_shift)
    if uwb.ranges is not None:
        uwb.ranges = copy_with_shift(uwb.ranges, args.uwb_time_shift)
    imu_pos_df = copy_with_shift(imu_pos_df, args.imu_time_shift)
    imu_acc_df = copy_with_shift(imu_acc_df, args.imu_time_shift)
    gt_df = copy_with_shift(gt_df, args.gt_time_shift)

    starts = [uwb_df.index.min()]
    ends = [uwb_df.index.max()]
    if imu_pos_df is not None:
        starts.append(imu_pos_df.index.min())
        ends.append(imu_pos_df.index.max())
    if imu_acc_df is not None:
        starts.append(imu_acc_df.index.min())
        ends.append(imu_acc_df.index.max())
    if gt_df is not None:
        starts.append(gt_df.index.min())
        ends.append(gt_df.index.max())
    start = max(starts)
    end = min(ends)
    if getattr(args, "trim_t_end", None) is not None:
        end = min(end, start + float(args.trim_t_end))
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise ValueError("Inputs do not have an overlapping relative time window")
    time_s = np.arange(start, end + dt / 2, dt, dtype=float)
    time_s = time_s - time_s[0]

    # Rebase all sources so interpolation aligns relative starts.
    def rebase(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None:
            return None
        out = df.copy()
        out.index = out.index.to_numpy(dtype=float) - start
        return out

    uwb_pos = interp_df(rebase(uwb.position), time_s)
    ranges = interp_df(rebase(uwb.ranges), time_s)
    imu_pos = interp_df(rebase(imu_pos_df), time_s)
    imu_acc = interp_df(rebase(imu_acc_df), time_s)
    gt_pos = interp_df(rebase(gt_df), time_s)
    imu_anchor_shift = np.full(3, np.nan, dtype=float)
    if imu_pos is not None and gt_pos is not None:
        imu_pos, imu_anchor_shift = align_imu_to_vicon_start(imu_pos, gt_pos, args.anchor_points or 5)
        args.imu_anchor_shift_applied = {
            "x": float(imu_anchor_shift[0]) if np.isfinite(imu_anchor_shift[0]) else float("nan"),
            "y": float(imu_anchor_shift[1]) if np.isfinite(imu_anchor_shift[1]) else float("nan"),
            "z": float(imu_anchor_shift[2]) if np.isfinite(imu_anchor_shift[2]) else float("nan"),
        }
        if np.isfinite(imu_anchor_shift).any():
            args.imu_offset_applied = args.imu_anchor_shift_applied
            print(f"Applied IMU-to-Vicon start offset: {np.round(imu_anchor_shift, 6)}")

    prep = Prepared(
        exp_id=getattr(args, "exp_id", ""),
        drone_name=getattr(args, "drone_name", ""),
        time_s=time_s,
        uwb_mode=uwb.mode,
        uwb_pos=uwb_pos[["x", "y", "z"]].to_numpy(dtype=float) if uwb_pos is not None else None,
        ranges=ranges.to_numpy(dtype=float) if ranges is not None else None,
        anchor_pos=uwb.anchor_pos,
        known_anchor_ids=list(anchors.keys()),
        known_anchor_pos=np.vstack(list(anchors.values())) if anchors else np.empty((0, 3), dtype=float),
        imu_pos=imu_pos[["x", "y", "z"]].to_numpy(dtype=float) if imu_pos is not None else None,
        imu_vel=None,
        imu_acc=imu_acc[["ax", "ay", "az"]].to_numpy(dtype=float) if imu_acc is not None else None,
        motion_acc_norm=None,
        motion_jerk_norm=None,
        q_vel_eff=None,
        temp_c=None,
        r_imu_z_eff=None,
        temp_z_bias_m_per_c=0.0,
        gt_pos=gt_pos[["x", "y", "z"]].to_numpy(dtype=float) if gt_pos is not None else None,
        uwb_xy_obs=None,
        uwb_valid_axes=None,
        uwb_z_floor_threshold_m=float(args.uwb_z_floor_threshold),
    )
    prep.imu_vel = derive_velocity(prep.imu_pos, dt)
    if args.enable_temperature_proxy:
        prep.temp_c = build_temperature_profile(
            len(prep.time_s),
            args.temp_start_c,
            args.temp_end_c,
            args.temp_profile_power,
        )
        if args.temp_ref_c is None:
            args.temp_ref_c = float(prep.temp_c[0])
        prep.imu_pos, prep.temp_z_bias_m_per_c = apply_temperature_proxy_to_imu_z(
            prep.imu_pos,
            prep.gt_pos,
            prep.temp_c,
            args.temp_ref_c,
            args.temp_z_bias_m_per_c,
            args.auto_tune_temp_z_bias,
        )
        prep.imu_vel = derive_velocity(prep.imu_pos, dt)
        if args.auto_tune_temp_z_bias:
            print(f"Tuned temperature proxy IMU z slope: {prep.temp_z_bias_m_per_c:.8f} m/C")
            args.temp_z_bias_m_per_c = float(prep.temp_z_bias_m_per_c)
    prep.r_imu_z_eff = build_r_imu_z_profile(
        len(prep.time_s),
        args.r_imu_z,
        prep.temp_c,
        args.temp_ref_c if args.temp_ref_c is not None else args.temp_start_c,
        args.temp_r_imu_z_gain,
    )
    prep.motion_acc_norm, prep.motion_jerk_norm, prep.q_vel_eff = compute_motion_q_profile(
        prep.imu_acc,
        prep.imu_vel,
        dt,
        args.q_vel,
        args.enable_adaptive_motion_q,
        args.k_acc,
        args.k_jerk,
        args.adaptive_q_max_scale,
        args.adaptive_q_smooth_window,
        args.drone_mass_scale,
        args.mass_q_exponent,
    )
    prep.uwb_xy_obs = build_uwb_xy_obs(prep)
    prep.uwb_valid_axes = build_uwb_axis_validity(prep.uwb_pos, args.uwb_z_floor_threshold)
    if tuning_tables:
        prep.shift_tuning = pd.concat(tuning_tables, ignore_index=True)
    else:
        prep.shift_tuning = pd.DataFrame()
    return prep


def initial_state(data: Prepared) -> np.ndarray:
    x = np.zeros(6)
    if data.uwb_xy_obs is not None:
        for axis in (0, 1):
            v = data.uwb_xy_obs[:, axis]
            v = v[np.isfinite(v)]
            if v.size:
                x[axis] = v[0]
    if data.imu_pos is not None:
        z0 = first_valid(data.imu_pos, 2)
        if z0 is not None:
            x[2] = z0
    elif data.uwb_pos is not None:
        z0 = first_valid_with_axis_gate(data.uwb_pos, data.uwb_valid_axes, 2)
        if z0 is not None:
            x[2] = z0
    if data.imu_vel is not None:
        for axis in range(3):
            v0 = first_valid(data.imu_vel, axis)
            if v0 is not None:
                x[3 + axis] = v0
    return x


@dataclass
class RunResult:
    states: np.ndarray
    extra: dict[str, np.ndarray] | None = None


def ca_transition(dt: float) -> np.ndarray:
    F = np.eye(9, dtype=float)
    I3 = np.eye(3, dtype=float)
    F[:3, 3:6] = dt * I3
    F[:3, 6:9] = 0.5 * dt * dt * I3
    F[3:6, 6:9] = dt * I3
    return F


def ca_process_noise(dt: float, q_pos: float, q_vel: float, q_acc: float, q_jerk: float) -> np.ndarray:
    Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel, q_acc, q_acc, q_acc]) ** 2
    if q_jerk > 0.0:
        g = np.array([dt**3 / 6.0, dt**2 / 2.0, dt], dtype=float)
        Qa = (float(q_jerk) ** 2) * np.outer(g, g)
        for axis in range(3):
            idx = np.array([axis, 3 + axis, 6 + axis])
            Q[np.ix_(idx, idx)] += Qa
    return Q


def initial_ca_state(data: Prepared) -> np.ndarray:
    x = np.zeros(9, dtype=float)
    if data.uwb_pos is not None:
        for axis in range(3):
            v0 = first_valid_with_axis_gate(data.uwb_pos, data.uwb_valid_axes, axis)
            if v0 is not None:
                x[axis] = v0
    elif data.uwb_xy_obs is not None:
        for axis in (0, 1):
            v = data.uwb_xy_obs[:, axis]
            v = v[np.isfinite(v)]
            if v.size:
                x[axis] = v[0]
    return x


def run_uwb_kalman(args: argparse.Namespace, data: Prepared) -> RunResult:
    n = len(data.time_s)
    states = np.full((n, 9), np.nan)
    x = initial_ca_state(data)
    P = np.diag([0.8, 0.8, 0.5, 1.5, 1.5, 1.0, 3.0, 3.0, 2.0]) ** 2
    F = ca_transition(args.dt)
    Q = ca_process_noise(args.dt, args.uwb_kf_q_pos, args.uwb_kf_q_vel, args.uwb_kf_q_acc, args.uwb_kf_q_jerk)
    axis_updated = np.zeros((n, 3), dtype=float)
    z_floor_rejected = np.zeros(n, dtype=float)

    for k in range(n):
        x = F @ x
        P = F @ P @ F.T + Q

        if data.uwb_mode == "position" and data.uwb_pos is not None:
            valid = data.uwb_valid_axes[k].copy() if data.uwb_valid_axes is not None else np.isfinite(data.uwb_pos[k, :3])
            raw_z = data.uwb_pos[k, 2]
            z_floor_rejected[k] = float(np.isfinite(raw_z) and raw_z < data.uwb_z_floor_threshold_m)
            idx = np.flatnonzero(valid)
            if idx.size:
                H = np.zeros((idx.size, 9), dtype=float)
                for row_i, axis in enumerate(idx):
                    H[row_i, axis] = 1.0
                z = data.uwb_pos[k, idx]
                h = x[idx]
                sigmas = np.array([args.r_uwb_z if axis == 2 else args.r_uwb_xy for axis in idx], dtype=float)
                R = np.diag(sigmas * sigmas)
                x, P, _, _ = k_update(x, P, z, h, H, R)
                axis_updated[k, idx] = 1.0
        elif data.uwb_mode == "ranges" and data.ranges is not None:
            ok = np.isfinite(data.ranges[k])
            if ok.sum() > 0:
                anchors = data.anchor_pos[ok]
                H6 = range_H_xy_only(x[:6], anchors, z_ref=x[2])
                H = np.zeros((ok.sum(), 9), dtype=float)
                H[:, :6] = H6
                z = data.ranges[k, ok]
                h = range_h_xy_only(x[:6], anchors, z_ref=x[2])
                R = np.eye(ok.sum()) * args.r_uwb_range**2
                x, P, _, _ = k_update(x, P, z, h, H, R)

        states[k] = x

    extra = {
        "axis_update_x": axis_updated[:, 0],
        "axis_update_y": axis_updated[:, 1],
        "axis_update_z": axis_updated[:, 2],
        "uwb_z_floor_rejected": z_floor_rejected,
    }
    return RunResult(states, extra)


def empty_uwb_gate_info() -> dict[str, float]:
    return {
        "attempted": 0.0,
        "updated": 0.0,
        "rejected": 0.0,
        "inflated": 0.0,
        "nis": np.nan,
        "pred_x": np.nan,
        "pred_y": np.nan,
    }


class EKF:
    def __init__(
        self,
        data: Prepared,
        dt: float,
        q_pos: float,
        q_vel: float,
        r_uwb_xy: float,
        r_uwb_range: float,
        r_imu_z: float,
        imu_vel_weight: float,
        gamma: float | None = None,
        adaptive_r: bool = False,
        uwb_xy_gate_enabled: bool = False,
        uwb_xy_gate_threshold: float = 9.21,
        uwb_xy_gate_action: str = "skip",
        uwb_xy_gate_inflate_scale: float = 100.0,
    ):
        self.data = data
        self.dt = dt
        self.r_uwb_xy = r_uwb_xy
        self.r_uwb_range = r_uwb_range
        self.r_imu_z = r_imu_z
        self.imu_vel_weight = imu_vel_weight
        self.gamma = gamma
        self.adaptive_r = adaptive_r
        self.uwb_xy_gate_enabled = bool(uwb_xy_gate_enabled)
        self.uwb_xy_gate_threshold = float(uwb_xy_gate_threshold)
        self.uwb_xy_gate_action = str(uwb_xy_gate_action)
        self.uwb_xy_gate_inflate_scale = float(uwb_xy_gate_inflate_scale)
        self.F = np.block([[np.eye(3), dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)
        self.P = np.eye(6)
        self.last_gate_info = empty_uwb_gate_info()

    def process_noise(self, k: int) -> np.ndarray:
        q_vel_eff = self.q_vel
        if self.data.q_vel_eff is not None and k < len(self.data.q_vel_eff) and np.isfinite(self.data.q_vel_eff[k]):
            q_vel_eff = float(self.data.q_vel_eff[k])
        return np.diag([self.q_pos, self.q_pos, self.q_pos, q_vel_eff, q_vel_eff, q_vel_eff]) ** 2

    def predict(self, x: np.ndarray, P: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        xp = self.F @ x
        if self.data.imu_acc is not None and np.isfinite(self.data.imu_acc[k]).all():
            a = self.data.imu_acc[k]
            xp[:3] += 0.5 * a * self.dt * self.dt
            xp[3:] += a * self.dt
        if self.data.imu_vel is not None and np.isfinite(self.data.imu_vel[k]).all():
            xp[3:] = self.imu_vel_weight * self.data.imu_vel[k] + (1 - self.imu_vel_weight) * xp[3:]
        return xp, self.F @ P @ self.F.T + self.process_noise(k)

    def uwb_update(self, x: np.ndarray, P: np.ndarray, prev_x: np.ndarray, prev_y: np.ndarray | None, k: int, r_scale: float):
        gamma = self.gamma
        self.last_gate_info = empty_uwb_gate_info()
        if self.data.uwb_mode == "position":
            if self.data.uwb_pos is None or not np.isfinite(self.data.uwb_pos[k, :2]).all():
                return x, P, None, None, prev_y
            y_now = self.data.uwb_pos[k, :2].copy()
            H = np.zeros((2, 6))
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            R = np.eye(2) * self.r_uwb_xy**2 * r_scale
            raw_innov = y_now - x[:2]
            S_gate = H @ P @ H.T + R
            nis = float(raw_innov.T @ np.linalg.pinv(S_gate) @ raw_innov)
            self.last_gate_info = {
                "attempted": 1.0,
                "updated": 1.0,
                "rejected": 0.0,
                "inflated": 0.0,
                "nis": nis,
                "pred_x": float(x[0]),
                "pred_y": float(x[1]),
            }
            if self.uwb_xy_gate_enabled and np.isfinite(nis) and nis > self.uwb_xy_gate_threshold:
                if self.uwb_xy_gate_action == "skip":
                    self.last_gate_info["updated"] = 0.0
                    self.last_gate_info["rejected"] = 1.0
                    return x, P, None, None, prev_y
                R = R * self.uwb_xy_gate_inflate_scale
                self.last_gate_info["inflated"] = 1.0
            z = y_now
            h = x[:2]
            if gamma is not None and prev_y is not None and np.isfinite(prev_y[:2]).all():
                z = y_now - gamma * prev_y[:2]
                h = x[:2] - gamma * prev_x[:2]
            x, P, innov, S = k_update(x, P, z, h, H, R)
            return x, P, innov, S, y_now

        if self.data.ranges is None:
            return x, P, None, None, prev_y
        ok = np.isfinite(self.data.ranges[k])
        if ok.sum() == 0:
            return x, P, None, None, prev_y
        anchors = self.data.anchor_pos[ok]
        y_now = self.data.ranges[k].copy()
        z = y_now[ok]
        h = range_h_xy_only(x, anchors, z_ref=x[2])
        H = range_H_xy_only(x, anchors, z_ref=x[2])
        if gamma is not None and prev_y is not None and prev_y.shape == y_now.shape and np.isfinite(prev_y[ok]).all():
            z = y_now[ok] - gamma * prev_y[ok]
            h = h - gamma * range_h_xy_only(prev_x, anchors, z_ref=prev_x[2])
        R = np.eye(ok.sum()) * self.r_uwb_range**2 * r_scale
        x, P, innov, S = k_update(x, P, z, h, H, R)
        return x, P, innov, S, y_now

    def imu_z_update(self, x: np.ndarray, P: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.data.imu_pos is None or not np.isfinite(self.data.imu_pos[k, 2]):
            return x, P
        H = np.zeros((1, 6))
        H[0, 2] = 1.0
        z = np.array([self.data.imu_pos[k, 2]])
        h = np.array([x[2]])
        r_imu_z = self.r_imu_z
        if self.data.r_imu_z_eff is not None and k < len(self.data.r_imu_z_eff) and np.isfinite(self.data.r_imu_z_eff[k]):
            r_imu_z = float(self.data.r_imu_z_eff[k])
        R = np.array([[r_imu_z**2]])
        x, P, _, _ = k_update(x, P, z, h, H, R)
        return x, P

    def step(self, x: np.ndarray, P: np.ndarray, prev_x: np.ndarray, prev_y: np.ndarray | None, k: int, r_scale: float = 1.0):
        xp, Pp = self.predict(x, P, k)
        xu, Pu, innov, S, new_prev_y = self.uwb_update(xp, Pp, prev_x, prev_y, k, r_scale)
        xz, Pz = self.imu_z_update(xu, Pu, k)
        return xz, Pz, innov, S, new_prev_y

    def run(self) -> RunResult:
        n = len(self.data.time_s)
        states = np.full((n, 6), np.nan)
        x = initial_state(self.data)
        P = np.diag([0.8, 0.8, 0.2, 1.0, 1.0, 0.6]) ** 2
        prev_x = x.copy()
        prev_y = None
        r_scale = 1.0
        r_hist = np.full(n, np.nan)
        gate_attempted = np.zeros(n, dtype=float)
        gate_updated = np.zeros(n, dtype=float)
        gate_rejected = np.zeros(n, dtype=float)
        gate_inflated = np.zeros(n, dtype=float)
        gate_nis = np.full(n, np.nan, dtype=float)
        gate_pred_x = np.full(n, np.nan, dtype=float)
        gate_pred_y = np.full(n, np.nan, dtype=float)
        for k in range(n):
            x, P, innov, S, new_prev_y = self.step(x, P, prev_x, prev_y, k, r_scale)
            info = self.last_gate_info
            gate_attempted[k] = info["attempted"]
            gate_updated[k] = info["updated"]
            gate_rejected[k] = info["rejected"]
            gate_inflated[k] = info["inflated"]
            gate_nis[k] = info["nis"]
            gate_pred_x[k] = info["pred_x"]
            gate_pred_y[k] = info["pred_y"]
            if self.adaptive_r and innov is not None and S is not None:
                nis = float(innov.T @ np.linalg.pinv(S) @ innov) / max(len(innov), 1)
                r_scale = float(np.clip(0.95 * r_scale + 0.05 * max(nis, 0.1), 0.25, 25.0))
            if new_prev_y is not None:
                prev_y = new_prev_y
            prev_x = x.copy()
            states[k] = x
            r_hist[k] = r_scale
        extra = {}
        if self.adaptive_r:
            extra["uwb_r_scale"] = r_hist
        if self.data.q_vel_eff is not None:
            extra["q_vel_eff"] = self.data.q_vel_eff
        if self.data.motion_acc_norm is not None:
            extra["motion_acc_norm"] = self.data.motion_acc_norm
        if self.data.motion_jerk_norm is not None:
            extra["motion_jerk_norm"] = self.data.motion_jerk_norm
        if self.data.temp_c is not None:
            extra["temp_c"] = self.data.temp_c
        if self.data.r_imu_z_eff is not None:
            extra["r_imu_z_eff"] = self.data.r_imu_z_eff
        extra["uwb_xy_gate_attempted"] = gate_attempted
        extra["uwb_xy_gate_updated"] = gate_updated
        extra["uwb_xy_gate_rejected"] = gate_rejected
        extra["uwb_xy_gate_inflated"] = gate_inflated
        extra["uwb_xy_gate_nis"] = gate_nis
        extra["uwb_xy_gate_pred_x"] = gate_pred_x
        extra["uwb_xy_gate_pred_y"] = gate_pred_y
        if not extra:
            extra = None
        return RunResult(states, extra)


def sigma_points(x: np.ndarray, P: np.ndarray, alpha: float, beta: float, kappa: float):
    n = len(x)
    lam = alpha * alpha * (n + kappa) - n
    c = n + lam
    P = 0.5 * (P + P.T) + np.eye(n) * EPS
    U = np.linalg.cholesky(c * P)
    pts = [x]
    for i in range(n):
        pts.append(x + U[:, i])
        pts.append(x - U[:, i])
    wm = np.full(2 * n + 1, 1 / (2 * c))
    wc = wm.copy()
    wm[0] = lam / c
    wc[0] = lam / c + 1 - alpha * alpha + beta
    return np.asarray(pts), wm, wc


def ukf_update(x: np.ndarray, P: np.ndarray, z: np.ndarray, fn: Callable[[np.ndarray], np.ndarray], R: np.ndarray, alpha: float, beta: float, kappa: float):
    pts, wm, wc = sigma_points(x, P, alpha, beta, kappa)
    ys = np.asarray([fn(p) for p in pts])
    ybar = np.sum(wm[:, None] * ys, axis=0)
    S = R.copy()
    C = np.zeros((len(x), len(z)))
    for i in range(len(pts)):
        dy = ys[i] - ybar
        dx = pts[i] - x
        S += wc[i] * np.outer(dy, dy)
        C += wc[i] * np.outer(dx, dy)
    S = 0.5 * (S + S.T) + np.eye(S.shape[0]) * EPS
    K = C @ np.linalg.pinv(S)
    x = x + K @ (z - ybar)
    P = P - K @ S @ K.T
    return x, 0.5 * (P + P.T)


def run_ukf(args: argparse.Namespace, data: Prepared) -> RunResult:
    n = len(data.time_s)
    states = np.full((n, 6), np.nan)
    x = initial_state(data)
    P = np.diag([0.8, 0.8, 0.2, 1.0, 1.0, 0.6]) ** 2
    F = np.block([[np.eye(3), args.dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
    gate_attempted = np.zeros(n, dtype=float)
    gate_updated = np.zeros(n, dtype=float)
    gate_rejected = np.zeros(n, dtype=float)
    gate_inflated = np.zeros(n, dtype=float)
    gate_nis = np.full(n, np.nan, dtype=float)
    gate_pred_x = np.full(n, np.nan, dtype=float)
    gate_pred_y = np.full(n, np.nan, dtype=float)
    for k in range(n):
        q_vel_eff = args.q_vel
        if data.q_vel_eff is not None and np.isfinite(data.q_vel_eff[k]):
            q_vel_eff = float(data.q_vel_eff[k])
        Q = np.diag([args.q_pos, args.q_pos, args.q_pos, q_vel_eff, q_vel_eff, q_vel_eff]) ** 2
        pts, wm, wc = sigma_points(x, P, args.ukf_alpha, args.ukf_beta, args.ukf_kappa)
        pred = []
        for p in pts:
            xp = F @ p
            if data.imu_acc is not None and np.isfinite(data.imu_acc[k]).all():
                a = data.imu_acc[k]
                xp[:3] += 0.5 * a * args.dt * args.dt
                xp[3:] += a * args.dt
            if data.imu_vel is not None and np.isfinite(data.imu_vel[k]).all():
                xp[3:] = args.imu_vel_weight * data.imu_vel[k] + (1 - args.imu_vel_weight) * xp[3:]
            pred.append(xp)
        pred = np.asarray(pred)
        x = np.sum(wm[:, None] * pred, axis=0)
        P = Q.copy()
        for i in range(len(pred)):
            d = pred[i] - x
            P += wc[i] * np.outer(d, d)
        if data.uwb_mode == "position" and data.uwb_pos is not None and np.isfinite(data.uwb_pos[k, :2]).all():
            R_xy = np.eye(2) * args.r_uwb_xy**2
            raw_innov = data.uwb_pos[k, :2] - x[:2]
            H_xy = np.zeros((2, 6), dtype=float)
            H_xy[0, 0] = 1.0
            H_xy[1, 1] = 1.0
            S_gate = H_xy @ P @ H_xy.T + R_xy
            nis = float(raw_innov.T @ np.linalg.pinv(S_gate) @ raw_innov)
            gate_attempted[k] = 1.0
            gate_updated[k] = 1.0
            gate_nis[k] = nis
            gate_pred_x[k] = float(x[0])
            gate_pred_y[k] = float(x[1])
            skip_update = False
            if args.enable_uwb_xy_nis_gate and np.isfinite(nis) and nis > args.uwb_xy_nis_threshold:
                if args.uwb_xy_gate_action == "skip":
                    gate_updated[k] = 0.0
                    gate_rejected[k] = 1.0
                    skip_update = True
                else:
                    gate_inflated[k] = 1.0
                    R_xy = R_xy * args.uwb_xy_gate_inflate_scale
            if not skip_update:
                x, P = ukf_update(
                    x, P, data.uwb_pos[k, :2], lambda s: s[:2], R_xy,
                    args.ukf_alpha, args.ukf_beta, args.ukf_kappa,
                )
        elif data.uwb_mode == "ranges" and data.ranges is not None:
            ok = np.isfinite(data.ranges[k])
            if ok.sum() > 0:
                anchors = data.anchor_pos[ok]
                z_ref = float(x[2])
                x, P = ukf_update(
                    x, P, data.ranges[k, ok],
                    lambda s, a=anchors, zr=z_ref: range_h_xy_only(s, a, zr),
                    np.eye(ok.sum()) * args.r_uwb_range**2,
                    args.ukf_alpha, args.ukf_beta, args.ukf_kappa,
                )
        if data.imu_pos is not None and np.isfinite(data.imu_pos[k, 2]):
            r_imu_z = args.r_imu_z
            if data.r_imu_z_eff is not None and np.isfinite(data.r_imu_z_eff[k]):
                r_imu_z = float(data.r_imu_z_eff[k])
            x, P = ukf_update(
                x, P, np.array([data.imu_pos[k, 2]]), lambda s: np.array([s[2]]),
                np.array([[r_imu_z**2]]), args.ukf_alpha, args.ukf_beta, args.ukf_kappa,
            )
        states[k] = x
    extra = {}
    if data.q_vel_eff is not None:
        extra["q_vel_eff"] = data.q_vel_eff
    if data.motion_acc_norm is not None:
        extra["motion_acc_norm"] = data.motion_acc_norm
    if data.motion_jerk_norm is not None:
        extra["motion_jerk_norm"] = data.motion_jerk_norm
    if data.temp_c is not None:
        extra["temp_c"] = data.temp_c
    if data.r_imu_z_eff is not None:
        extra["r_imu_z_eff"] = data.r_imu_z_eff
    extra["uwb_xy_gate_attempted"] = gate_attempted
    extra["uwb_xy_gate_updated"] = gate_updated
    extra["uwb_xy_gate_rejected"] = gate_rejected
    extra["uwb_xy_gate_inflated"] = gate_inflated
    extra["uwb_xy_gate_nis"] = gate_nis
    extra["uwb_xy_gate_pred_x"] = gate_pred_x
    extra["uwb_xy_gate_pred_y"] = gate_pred_y
    return RunResult(states, extra or None)


def run_adaptive_ekf(args: argparse.Namespace, data: Prepared) -> RunResult:
    gammas = list(args.a_gammas)
    filters = [
        EKF(
            data,
            args.dt,
            args.q_pos,
            args.q_vel,
            args.r_uwb_xy,
            args.r_uwb_range,
            args.r_imu_z,
            args.imu_vel_weight,
            gamma=g,
            uwb_xy_gate_enabled=args.enable_uwb_xy_nis_gate,
            uwb_xy_gate_threshold=args.uwb_xy_nis_threshold,
            uwb_xy_gate_action=args.uwb_xy_gate_action,
            uwb_xy_gate_inflate_scale=args.uwb_xy_gate_inflate_scale,
        )
        for g in gammas
    ]
    n = len(data.time_s)
    m = len(gammas)
    xs = [initial_state(data).copy() for _ in gammas]
    Ps = [np.diag([0.8, 0.8, 0.2, 1.0, 1.0, 0.6]) ** 2 for _ in gammas]
    prev_xs = [x.copy() for x in xs]
    prev_ys = [None for _ in xs]
    weights = np.ones(m) / m
    states = np.full((n, 6), np.nan)
    wh = np.full((n, m), np.nan)
    gate_attempted = np.zeros(n, dtype=float)
    gate_updated = np.zeros(n, dtype=float)
    gate_rejected = np.zeros(n, dtype=float)
    gate_inflated = np.zeros(n, dtype=float)
    gate_nis = np.full(n, np.nan, dtype=float)
    gate_pred_x = np.full(n, np.nan, dtype=float)
    gate_pred_y = np.full(n, np.nan, dtype=float)
    for k in range(n):
        new_xs, new_Ps, new_prev_ys, likes = [], [], [], []
        for i, f in enumerate(filters):
            x, P, innov, S, py = f.step(xs[i], Ps[i], prev_xs[i], prev_ys[i], k)
            if i == 0:
                info = f.last_gate_info
                gate_attempted[k] = info["attempted"]
                gate_updated[k] = info["updated"]
                gate_rejected[k] = info["rejected"]
                gate_inflated[k] = info["inflated"]
                gate_nis[k] = info["nis"]
                gate_pred_x[k] = info["pred_x"]
                gate_pred_y[k] = info["pred_y"]
            new_xs.append(x)
            new_Ps.append(P)
            new_prev_ys.append(py if py is not None else prev_ys[i])
            likes.append(likelihood(innov, S))
        weights = weights * np.asarray(likes)
        weights = np.ones(m) / m if not np.isfinite(weights).all() or weights.sum() <= 0 else weights / weights.sum()
        xf = sum(weights[i] * new_xs[i] for i in range(m))
        states[k] = xf
        wh[k] = weights
        xs, Ps, prev_xs, prev_ys = new_xs, new_Ps, [x.copy() for x in new_xs], new_prev_ys
    extra = {f"weight_gamma_{str(g).replace('.', 'p')}": wh[:, i] for i, g in enumerate(gammas)}
    if data.q_vel_eff is not None:
        extra["q_vel_eff"] = data.q_vel_eff
    if data.motion_acc_norm is not None:
        extra["motion_acc_norm"] = data.motion_acc_norm
    if data.motion_jerk_norm is not None:
        extra["motion_jerk_norm"] = data.motion_jerk_norm
    if data.temp_c is not None:
        extra["temp_c"] = data.temp_c
    if data.r_imu_z_eff is not None:
        extra["r_imu_z_eff"] = data.r_imu_z_eff
    extra["uwb_xy_gate_attempted"] = gate_attempted
    extra["uwb_xy_gate_updated"] = gate_updated
    extra["uwb_xy_gate_rejected"] = gate_rejected
    extra["uwb_xy_gate_inflated"] = gate_inflated
    extra["uwb_xy_gate_nis"] = gate_nis
    extra["uwb_xy_gate_pred_x"] = gate_pred_x
    extra["uwb_xy_gate_pred_y"] = gate_pred_y
    return RunResult(states, extra)


def run_weighted(args: argparse.Namespace, data: Prepared) -> RunResult:
    n = len(data.time_s)
    pos = np.full((n, 3), np.nan)
    if data.uwb_xy_obs is not None and data.imu_pos is not None:
        wu = 1 / max(args.r_uwb_xy**2, EPS)
        wi = 1 / max(args.r_imu_xy**2, EPS)
        for a in (0, 1):
            u = data.uwb_xy_obs[:, a]
            im = data.imu_pos[:, a]
            both = np.isfinite(u) & np.isfinite(im)
            only_u = np.isfinite(u) & ~np.isfinite(im)
            only_i = np.isfinite(im) & ~np.isfinite(u)
            pos[both, a] = (wu * u[both] + wi * im[both]) / (wu + wi)
            pos[only_u, a] = u[only_u]
            pos[only_i, a] = im[only_i]
    elif data.uwb_xy_obs is not None:
        pos[:, :2] = data.uwb_xy_obs
    elif data.imu_pos is not None:
        pos[:, :2] = data.imu_pos[:, :2]
    if data.imu_pos is not None:
        pos[:, 2] = data.imu_pos[:, 2]
    elif data.uwb_pos is not None:
        pos[:, 2] = gated_uwb_axis(data, 2)
    for a in range(3):
        pos[:, a] = pd.Series(pos[:, a]).interpolate(limit_direction="both").to_numpy(dtype=float)
    vel = derive_velocity(pos, args.dt)
    return RunResult(np.hstack([pos, vel if vel is not None else np.zeros((n, 3))]))


def fgo_initial_states(args: argparse.Namespace, data: Prepared) -> np.ndarray:
    init = run_weighted(args, data).states.copy()
    if not np.isfinite(init[:, :3]).all():
        fallback = initial_state(data)
        for i in range(6):
            s = pd.Series(init[:, i]).replace([np.inf, -np.inf], np.nan)
            init[:, i] = s.interpolate(limit_direction="both").fillna(fallback[i]).to_numpy(dtype=float)
    return init


def fgo_window_residual_jacobian(
    args: argparse.Namespace,
    data: Prepared,
    xw: np.ndarray,
    start: int,
    prior_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    w = len(xw)
    nvar = 6 * w
    rows = []
    residuals = []

    def add_res(local_idx: int, values: np.ndarray, jac_block: np.ndarray, sigma: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        jac_block = np.asarray(jac_block, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        for r_i in range(len(values)):
            if not np.isfinite(values[r_i]) or sigma[r_i] <= 0:
                continue
            row = np.zeros(nvar, dtype=float)
            row[local_idx * 6 : local_idx * 6 + 6] = jac_block[r_i] / sigma[r_i]
            rows.append(row)
            residuals.append(values[r_i] / sigma[r_i])

    prior_sigma = np.asarray(args.fgo_prior_sigma, dtype=float)
    prior = xw[0] - prior_states[start]
    add_res(0, prior, np.eye(6), prior_sigma)

    for i in range(w):
        k = start + i
        x = xw[i]
        if data.uwb_mode == "position" and data.uwb_pos is not None and np.isfinite(data.uwb_pos[k, :2]).all():
            H = np.zeros((2, 6), dtype=float)
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            add_res(i, x[:2] - data.uwb_pos[k, :2], H, np.full(2, args.r_uwb_xy))
        elif data.uwb_mode == "ranges" and data.ranges is not None:
            ok = np.isfinite(data.ranges[k])
            if ok.sum() > 0:
                anchors = data.anchor_pos[ok]
                pred = range_h_xy_only(x, anchors, z_ref=x[2])
                H = range_H_xy_only(x, anchors, z_ref=x[2])
                add_res(i, pred - data.ranges[k, ok], H, np.full(ok.sum(), args.r_uwb_range))

        if data.imu_pos is not None and np.isfinite(data.imu_pos[k, 2]):
            H = np.zeros((1, 6), dtype=float)
            H[0, 2] = 1.0
            r_imu_z = args.r_imu_z
            if data.r_imu_z_eff is not None and np.isfinite(data.r_imu_z_eff[k]):
                r_imu_z = float(data.r_imu_z_eff[k])
            add_res(i, np.array([x[2] - data.imu_pos[k, 2]]), H, np.array([r_imu_z]))

        if args.fgo_use_imu_velocity and data.imu_vel is not None and np.isfinite(data.imu_vel[k]).all():
            H = np.zeros((3, 6), dtype=float)
            H[:, 3:6] = np.eye(3)
            add_res(i, x[3:6] - data.imu_vel[k], H, np.full(3, args.fgo_r_imu_vel))

    for i in range(1, w):
        k = start + i
        prev = xw[i - 1]
        cur = xw[i]
        dt = float(data.time_s[k] - data.time_s[k - 1])
        if not np.isfinite(dt) or dt <= 0:
            dt = args.dt
        q_vel_eff = args.q_vel
        if data.q_vel_eff is not None and np.isfinite(data.q_vel_eff[k - 1]):
            q_vel_eff = float(data.q_vel_eff[k - 1])
        sigma_pos = max(args.fgo_motion_pos_scale * args.q_pos, EPS)
        sigma_vel = max(args.fgo_motion_vel_scale * q_vel_eff, EPS)

        e_pos = cur[:3] - prev[:3] - dt * prev[3:6]
        e_vel = cur[3:6] - prev[3:6]

        for axis in range(3):
            row = np.zeros(nvar, dtype=float)
            row[i * 6 + axis] = 1.0 / sigma_pos
            row[(i - 1) * 6 + axis] = -1.0 / sigma_pos
            row[(i - 1) * 6 + 3 + axis] = -dt / sigma_pos
            rows.append(row)
            residuals.append(e_pos[axis] / sigma_pos)

            row = np.zeros(nvar, dtype=float)
            row[i * 6 + 3 + axis] = 1.0 / sigma_vel
            row[(i - 1) * 6 + 3 + axis] = -1.0 / sigma_vel
            rows.append(row)
            residuals.append(e_vel[axis] / sigma_vel)

    if not rows:
        return np.zeros(0), np.zeros((0, nvar))
    return np.asarray(residuals, dtype=float), np.vstack(rows)


def optimize_fgo_window(args: argparse.Namespace, data: Prepared, states: np.ndarray, start: int, end: int) -> np.ndarray:
    xw = states[start:end].copy()
    lm = float(args.fgo_lm_damping)
    for _ in range(args.fgo_iterations):
        r, J = fgo_window_residual_jacobian(args, data, xw, start, states)
        if len(r) == 0:
            break
        cost = float(r @ r)
        H = J.T @ J
        g = J.T @ r
        diag = np.diag(H).copy()
        diag[diag <= EPS] = 1.0
        accepted = False
        for _try in range(6):
            try:
                dx = -np.linalg.solve(H + lm * np.diag(diag), g)
            except np.linalg.LinAlgError:
                dx = -np.linalg.pinv(H + lm * np.diag(diag)) @ g
            candidate = xw + dx.reshape((-1, 6))
            r_new, _ = fgo_window_residual_jacobian(args, data, candidate, start, states)
            new_cost = float(r_new @ r_new) if len(r_new) else cost
            if new_cost <= cost:
                xw = candidate
                lm = max(lm * 0.5, 1e-8)
                accepted = True
                break
            lm = min(lm * 5.0, 1e8)
        if not accepted or np.linalg.norm(dx) < args.fgo_step_tol:
            break
    return xw


def run_factor_graph(args: argparse.Namespace, data: Prepared) -> RunResult:
    n = len(data.time_s)
    init = fgo_initial_states(args, data)
    accum = np.zeros_like(init)
    weights = np.zeros(n, dtype=float)
    win = max(3, int(args.fgo_window_size))
    stride = max(1, int(args.fgo_stride))
    starts = list(range(0, n, stride))
    if starts[-1] != max(0, n - win):
        starts.append(max(0, n - win))
    for start in starts:
        end = min(n, start + win)
        if end - start < 3:
            continue
        opt = optimize_fgo_window(args, data, init, start, end)
        local_n = end - start
        if args.fgo_blend == "triangular" and local_n > 2:
            u = np.linspace(-1.0, 1.0, local_n)
            ww = 1.0 - np.abs(u)
            ww = np.maximum(ww, 0.2)
        else:
            ww = np.ones(local_n, dtype=float)
        accum[start:end] += opt * ww[:, None]
        weights[start:end] += ww
    out = init.copy()
    ok = weights > 0
    out[ok] = accum[ok] / weights[ok, None]
    extra = {"fgo_window_weight": weights}
    if data.q_vel_eff is not None:
        extra["q_vel_eff"] = data.q_vel_eff
    return RunResult(out, extra)


def run_all(args: argparse.Namespace, data: Prepared, include_factor_graph: bool = True) -> dict[str, RunResult]:
    base = dict(
        data=data,
        dt=args.dt,
        q_pos=args.q_pos,
        q_vel=args.q_vel,
        r_uwb_xy=args.r_uwb_xy,
        r_uwb_range=args.r_uwb_range,
        r_imu_z=args.r_imu_z,
        imu_vel_weight=args.imu_vel_weight,
        uwb_xy_gate_enabled=args.enable_uwb_xy_nis_gate,
        uwb_xy_gate_threshold=args.uwb_xy_nis_threshold,
        uwb_xy_gate_action=args.uwb_xy_gate_action,
        uwb_xy_gate_inflate_scale=args.uwb_xy_gate_inflate_scale,
    )
    runs = {
        "uwb_kalman": run_uwb_kalman(args, data),
        "ekf": EKF(**base).run(),
        "ukf": run_ukf(args, data),
        "cekf": EKF(**base, gamma=args.c_gamma).run(),
        "adaptive_ekf": run_adaptive_ekf(args, data),
        "adaptive_covariance_fusion": EKF(**base, adaptive_r=True).run(),
        "weighted_fusion": run_weighted(args, data),
    }
    if include_factor_graph and not args.disable_factor_graph:
        runs["factor_graph"] = run_factor_graph(args, data)
    return runs


def refresh_motion_q_profile(args: argparse.Namespace, data: Prepared) -> None:
    data.motion_acc_norm, data.motion_jerk_norm, data.q_vel_eff = compute_motion_q_profile(
        data.imu_acc,
        data.imu_vel,
        args.dt,
        args.q_vel,
        args.enable_adaptive_motion_q,
        args.k_acc,
        args.k_jerk,
        args.adaptive_q_max_scale,
        args.adaptive_q_smooth_window,
        args.drone_mass_scale,
        args.mass_q_exponent,
    )


def run_adaptive_motion_q_grid(args: argparse.Namespace, data: Prepared) -> pd.DataFrame:
    if data.gt_pos is None:
        raise ValueError("--tune-adaptive-motion-q requires ground truth")
    rows = []
    original = (args.enable_adaptive_motion_q, args.k_acc, args.k_jerk)
    args.enable_adaptive_motion_q = True
    for k_acc in args.k_acc_grid:
        for k_jerk in args.k_jerk_grid:
            args.k_acc = float(k_acc)
            args.k_jerk = float(k_jerk)
            refresh_motion_q_profile(args, data)
            runs = run_all(args, data, include_factor_graph=False)
            for name, run in runs.items():
                err = run.states[:, :3] - data.gt_pos
                e3d = np.linalg.norm(err, axis=1)
                exy = np.linalg.norm(err[:, :2], axis=1)
                ez = err[:, 2]
                rows.append(
                    {
                        "k_acc": float(k_acc),
                        "k_jerk": float(k_jerk),
                        "algorithm": name,
                        "rmse_3d_m": rmse(e3d),
                        "rmse_xy_m": rmse(exy),
                        "rmse_z_m": rmse(ez),
                        "q_vel_eff_mean": float(np.nanmean(data.q_vel_eff)) if data.q_vel_eff is not None else np.nan,
                        "q_vel_eff_max": float(np.nanmax(data.q_vel_eff)) if data.q_vel_eff is not None else np.nan,
                    }
                )
    summary = pd.DataFrame(rows)
    target = summary[summary["algorithm"] == args.adaptive_q_tune_target]
    if target.empty:
        target = summary
    best = target.sort_values("rmse_3d_m").iloc[0]
    args.enable_adaptive_motion_q = True
    args.k_acc = float(best["k_acc"])
    args.k_jerk = float(best["k_jerk"])
    refresh_motion_q_profile(args, data)
    print(
        "Best adaptive motion Q: "
        f"k_acc={args.k_acc:g}, k_jerk={args.k_jerk:g} "
        f"by {best['algorithm']} RMSE={best['rmse_3d_m']:.4f} m"
    )
    if not original[0]:
        # Keep adaptive mode enabled after tuning; the final run should use the selected pair.
        pass
    return summary


def rmse(v: np.ndarray) -> float:
    ok = np.isfinite(v)
    return float(np.sqrt(np.mean(v[ok] ** 2))) if ok.any() else float("nan")


def mae(v: np.ndarray) -> float:
    ok = np.isfinite(v)
    return float(np.mean(np.abs(v[ok]))) if ok.any() else float("nan")


def gate_stats_from_extra(extra: dict[str, np.ndarray] | None) -> dict[str, float]:
    if not extra or "uwb_xy_gate_attempted" not in extra:
        return {
            "uwb_xy_gate_attempted": 0,
            "uwb_xy_gate_updated": 0,
            "uwb_xy_gate_rejected": 0,
            "uwb_xy_gate_inflated": 0,
            "uwb_xy_gate_outliers": 0,
            "uwb_xy_gate_rejection_rate": 0.0,
            "uwb_xy_gate_inflation_rate": 0.0,
            "uwb_xy_gate_mean_nis": np.nan,
            "uwb_xy_gate_max_nis": np.nan,
        }
    attempted = np.asarray(extra.get("uwb_xy_gate_attempted", []), dtype=float)
    updated = np.asarray(extra.get("uwb_xy_gate_updated", []), dtype=float)
    rejected = np.asarray(extra.get("uwb_xy_gate_rejected", []), dtype=float)
    inflated = np.asarray(extra.get("uwb_xy_gate_inflated", []), dtype=float)
    nis = np.asarray(extra.get("uwb_xy_gate_nis", []), dtype=float)
    attempted_count = int(np.nansum(attempted))
    rejected_count = int(np.nansum(rejected))
    inflated_count = int(np.nansum(inflated))
    valid_nis = nis[np.isfinite(nis) & (attempted > 0)]
    return {
        "uwb_xy_gate_attempted": attempted_count,
        "uwb_xy_gate_updated": int(np.nansum(updated)),
        "uwb_xy_gate_rejected": rejected_count,
        "uwb_xy_gate_inflated": inflated_count,
        "uwb_xy_gate_outliers": rejected_count + inflated_count,
        "uwb_xy_gate_rejection_rate": rejected_count / attempted_count if attempted_count else 0.0,
        "uwb_xy_gate_inflation_rate": inflated_count / attempted_count if attempted_count else 0.0,
        "uwb_xy_gate_mean_nis": float(np.nanmean(valid_nis)) if valid_nis.size else np.nan,
        "uwb_xy_gate_max_nis": float(np.nanmax(valid_nis)) if valid_nis.size else np.nan,
    }


def gate_rejection_rows(
    algorithm: str,
    data: Prepared,
    extra: dict[str, np.ndarray] | None,
    threshold: float,
    action: str,
) -> list[dict[str, float | str]]:
    if not extra or "uwb_xy_gate_attempted" not in extra or data.uwb_pos is None:
        return []
    rejected = np.asarray(extra.get("uwb_xy_gate_rejected", []), dtype=float)
    inflated = np.asarray(extra.get("uwb_xy_gate_inflated", []), dtype=float)
    nis = np.asarray(extra.get("uwb_xy_gate_nis", []), dtype=float)
    pred_x = np.asarray(extra.get("uwb_xy_gate_pred_x", []), dtype=float)
    pred_y = np.asarray(extra.get("uwb_xy_gate_pred_y", []), dtype=float)
    rows = []
    for k in np.where((rejected > 0) | (inflated > 0))[0]:
        rows.append(
            {
                "exp_id": getattr(data, "exp_id", ""),
                "drone_name": getattr(data, "drone_name", ""),
                "time_s": float(data.time_s[k]),
                "algorithm": algorithm,
                "gate_action": "rejected" if rejected[k] > 0 else "inflated",
                "configured_action": action,
                "nis": float(nis[k]) if np.isfinite(nis[k]) else np.nan,
                "threshold": float(threshold),
                "uwb_x": float(data.uwb_pos[k, 0]),
                "uwb_y": float(data.uwb_pos[k, 1]),
                "pred_x": float(pred_x[k]) if np.isfinite(pred_x[k]) else np.nan,
                "pred_y": float(pred_y[k]) if np.isfinite(pred_y[k]) else np.nan,
                "innovation_x": float(data.uwb_pos[k, 0] - pred_x[k]) if np.isfinite(pred_x[k]) else np.nan,
                "innovation_y": float(data.uwb_pos[k, 1] - pred_y[k]) if np.isfinite(pred_y[k]) else np.nan,
            }
        )
    return rows


def z_policy_for_algorithm(algorithm: str, data: Prepared) -> str:
    if algorithm == "uwb_kalman":
        return f"UWB-only CA KF; z updates only when uwb_z >= {data.uwb_z_floor_threshold_m:g} m"
    if algorithm == "weighted_fusion":
        if data.imu_pos is not None:
            return "weighted x/y; IMU z used"
        if data.uwb_pos is not None:
            return f"UWB x/y/z interpolation; z uses floor gate {data.uwb_z_floor_threshold_m:g} m"
    if data.imu_pos is not None:
        return "UWB x/y updates; IMU z update"
    if data.uwb_pos is not None:
        return "UWB x/y updates; z propagated from gated initial UWB z"
    return "UWB range updates; no direct z measurement"


def motion_model_for_algorithm(algorithm: str) -> str:
    if algorithm == "uwb_kalman":
        return "constant_acceleration_uwb_only"
    if algorithm == "factor_graph":
        return "fixed_lag_smoothing"
    if algorithm == "weighted_fusion":
        return "measurement_weighted_interpolation"
    return "constant_velocity_flexible_process"


def write_outputs(args: argparse.Namespace, data: Prepared, runs: dict[str, RunResult]) -> pd.DataFrame:
    out = Path(args.output)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    traj = pd.DataFrame({"time_s": data.time_s})
    if data.gt_pos is not None:
        traj[["gt_x", "gt_y", "gt_z"]] = data.gt_pos
    if data.imu_pos is not None:
        traj[["imu_x", "imu_y", "imu_z"]] = data.imu_pos
    if data.uwb_pos is not None:
        traj[["uwb_x", "uwb_y", "uwb_z"]] = data.uwb_pos
    if data.uwb_valid_axes is not None:
        traj[["uwb_valid_x", "uwb_valid_y", "uwb_valid_z"]] = data.uwb_valid_axes
        if data.uwb_pos is not None:
            traj["uwb_z_valid"] = np.where(data.uwb_valid_axes[:, 2], data.uwb_pos[:, 2], np.nan)
    if data.uwb_xy_obs is not None:
        traj[["uwb_obs_x", "uwb_obs_y"]] = data.uwb_xy_obs
    if data.known_anchor_pos is not None and len(data.known_anchor_pos) > 0:
        anchor_df = pd.DataFrame(
            {
                "anchor_id": data.known_anchor_ids,
                "x": data.known_anchor_pos[:, 0],
                "y": data.known_anchor_pos[:, 1],
                "z": data.known_anchor_pos[:, 2],
            }
        )
        anchor_df.to_csv(out / "anchor_positions.csv", index=False)
    if data.motion_acc_norm is not None:
        traj["motion_acc_norm"] = data.motion_acc_norm
    if data.motion_jerk_norm is not None:
        traj["motion_jerk_norm"] = data.motion_jerk_norm
    if data.q_vel_eff is not None:
        traj["q_vel_eff"] = data.q_vel_eff
    if data.temp_c is not None:
        traj["temp_proxy_c"] = data.temp_c
    if data.r_imu_z_eff is not None:
        traj["r_imu_z_eff"] = data.r_imu_z_eff

    metrics_rows = []
    error_rows = []
    rejection_rows = []
    trajectory_columns: dict[str, np.ndarray] = {}
    for name, run in runs.items():
        gate_stats = gate_stats_from_extra(run.extra)
        rejection_rows.extend(
            gate_rejection_rows(
                name,
                data,
                run.extra,
                args.uwb_xy_nis_threshold,
                args.uwb_xy_gate_action,
            )
        )
        for i, suffix in enumerate(("x", "y", "z")):
            trajectory_columns[f"{name}_{suffix}"] = run.states[:, i]
        for i, suffix in enumerate(("vx", "vy", "vz")):
            trajectory_columns[f"{name}_{suffix}"] = run.states[:, 3 + i]
        if run.states.shape[1] >= 9:
            for i, suffix in enumerate(("ax", "ay", "az")):
                trajectory_columns[f"{name}_{suffix}"] = run.states[:, 6 + i]
        if run.extra:
            for key, value in run.extra.items():
                trajectory_columns[f"{name}_{key}"] = value
        z_policy = z_policy_for_algorithm(name, data)
        motion_model = motion_model_for_algorithm(name)
        if data.gt_pos is None:
            metrics_rows.append({
                "exp_id": data.exp_id,
                "drone_name": data.drone_name,
                "algorithm": name,
                "uwb_mode": data.uwb_mode,
                "n_samples": len(data.time_s),
                "rmse_3d_m": np.nan,
                "rmse_xy_m": np.nan,
                "rmse_z_m": np.nan,
                "mae_3d_m": np.nan,
                "mae_xy_m": np.nan,
                "mae_z_m": np.nan,
                "z_policy": z_policy,
                "motion_model": motion_model,
                "uwb_z_floor_threshold_m": float(data.uwb_z_floor_threshold_m),
                "adaptive_motion_q": bool(args.enable_adaptive_motion_q),
                "k_acc": float(args.k_acc),
                "k_jerk": float(args.k_jerk),
                "drone_mass_scale": float(args.drone_mass_scale),
                "temp_proxy_enabled": bool(args.enable_temperature_proxy),
                "temp_z_bias_m_per_c": float(data.temp_z_bias_m_per_c),
                "uwb_xy_gate_enabled": bool(args.enable_uwb_xy_nis_gate),
                "uwb_xy_gate_threshold": float(args.uwb_xy_nis_threshold),
                "uwb_xy_gate_action": args.uwb_xy_gate_action,
                **gate_stats,
            })
            continue
        err = run.states[:, :3] - data.gt_pos
        e_xy = np.linalg.norm(err[:, :2], axis=1)
        e_z_signed = err[:, 2]
        e_3d = np.linalg.norm(err, axis=1)
        ok = np.isfinite(err).all(axis=1)
        metrics_rows.append(
            {
                "exp_id": data.exp_id,
                "drone_name": data.drone_name,
                "algorithm": name,
                "uwb_mode": data.uwb_mode,
                "n_samples": int(ok.sum()),
                "rmse_3d_m": rmse(e_3d),
                "rmse_xy_m": rmse(e_xy),
                "rmse_z_m": rmse(e_z_signed),
                "mae_3d_m": mae(e_3d),
                "mae_xy_m": mae(e_xy),
                "mae_z_m": mae(e_z_signed),
                "median_3d_m": float(np.nanmedian(e_3d)) if np.isfinite(e_3d).any() else np.nan,
                "max_3d_m": float(np.nanmax(e_3d)) if np.isfinite(e_3d).any() else np.nan,
                "z_policy": z_policy,
                "motion_model": motion_model,
                "uwb_z_floor_threshold_m": float(data.uwb_z_floor_threshold_m),
                "adaptive_motion_q": bool(args.enable_adaptive_motion_q),
                "k_acc": float(args.k_acc),
                "k_jerk": float(args.k_jerk),
                "drone_mass_scale": float(args.drone_mass_scale),
                "temp_proxy_enabled": bool(args.enable_temperature_proxy),
                "temp_z_bias_m_per_c": float(data.temp_z_bias_m_per_c),
                "uwb_xy_gate_enabled": bool(args.enable_uwb_xy_nis_gate),
                "uwb_xy_gate_threshold": float(args.uwb_xy_nis_threshold),
                "uwb_xy_gate_action": args.uwb_xy_gate_action,
                **gate_stats,
            }
        )
        for k, t in enumerate(data.time_s):
            error_rows.append(
                {
                    "time_s": t,
                    "exp_id": data.exp_id,
                    "drone_name": data.drone_name,
                    "algorithm": name,
                    "error_3d_m": e_3d[k],
                    "error_xy_m": e_xy[k],
                    "error_z_m": abs(e_z_signed[k]),
                    "signed_error_z_m": e_z_signed[k],
                }
            )

    metrics = pd.DataFrame(metrics_rows)
    errors = pd.DataFrame(error_rows)
    if trajectory_columns:
        traj = pd.concat([traj, pd.DataFrame(trajectory_columns, index=traj.index)], axis=1)
    rejection_columns = [
        "exp_id",
        "drone_name",
        "time_s",
        "algorithm",
        "gate_action",
        "configured_action",
        "nis",
        "threshold",
        "uwb_x",
        "uwb_y",
        "pred_x",
        "pred_y",
        "innovation_x",
        "innovation_y",
    ]
    rejections = pd.DataFrame(rejection_rows, columns=rejection_columns)
    metrics.to_csv(out / "comparison_metrics.csv", index=False)
    errors.to_csv(out / "error_timeseries.csv", index=False)
    rejections.to_csv(out / "uwb_xy_rejections.csv", index=False)
    traj.to_csv(out / "fused_trajectories.csv", index=False)
    if data.shift_tuning is not None and not data.shift_tuning.empty:
        data.shift_tuning.to_csv(out / "shift_tuning_summary.csv", index=False)
    if data.adaptive_q_tuning is not None and not data.adaptive_q_tuning.empty:
        data.adaptive_q_tuning.to_csv(out / "adaptive_q_tuning_summary.csv", index=False)
    write_tuned_manifest(args, out)
    if not getattr(args, "suppress_plots", False):
        make_plots(data, runs, metrics, plots)
    return metrics


def draw_fallback_plot(path: Path, title: str, series: list[tuple[str, np.ndarray, np.ndarray]], xlabel: str, ylabel: str) -> None:
    from PIL import Image, ImageDraw

    w, h = 1100, 650
    left, right, top, bottom = 90, 30, 50, 80
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    colors = [
        (76, 120, 168),
        (245, 133, 24),
        (84, 162, 75),
        (178, 121, 162),
        (255, 157, 167),
        (118, 183, 178),
        (89, 89, 89),
    ]
    xs = np.concatenate([x[np.isfinite(x) & np.isfinite(y)] for _, x, y in series if len(x) and len(y)] or [np.array([0, 1])])
    ys = np.concatenate([y[np.isfinite(x) & np.isfinite(y)] for _, x, y in series if len(x) and len(y)] or [np.array([0, 1])])
    if xs.size == 0:
        xs = np.array([0.0, 1.0])
    if ys.size == 0:
        ys = np.array([0.0, 1.0])
    xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
    ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
    if abs(xmax - xmin) < EPS:
        xmax = xmin + 1.0
    if abs(ymax - ymin) < EPS:
        ymax = ymin + 1.0
    ypad = 0.05 * (ymax - ymin)
    ymin -= ypad
    ymax += ypad

    def map_xy(x: float, y: float) -> tuple[int, int]:
        px = left + int((x - xmin) / (xmax - xmin) * (w - left - right))
        py = h - bottom - int((y - ymin) / (ymax - ymin) * (h - top - bottom))
        return px, py

    draw.text((left, 15), title, fill=(0, 0, 0))
    draw.line((left, top, left, h - bottom, w - right, h - bottom), fill=(0, 0, 0), width=2)
    draw.text((w // 2 - 30, h - 35), xlabel, fill=(0, 0, 0))
    draw.text((10, h // 2), ylabel, fill=(0, 0, 0))
    draw.text((left, h - bottom + 10), f"{xmin:.2f}", fill=(80, 80, 80))
    draw.text((w - right - 70, h - bottom + 10), f"{xmax:.2f}", fill=(80, 80, 80))
    draw.text((20, top), f"{ymax:.2f}", fill=(80, 80, 80))
    draw.text((20, h - bottom - 10), f"{ymin:.2f}", fill=(80, 80, 80))

    legend_y = 35
    for i, (name, x, y) in enumerate(series):
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            continue
        pts = [map_xy(float(a), float(b)) for a, b in zip(x[mask], y[mask])]
        hex_color = plot_color(name)
        try:
            color = tuple(int(hex_color.lstrip("#")[j : j + 2], 16) for j in (0, 2, 4))
        except Exception:
            color = colors[i % len(colors)]
        draw.line(pts, fill=color, width=2)
        lx = w - 250
        draw.line((lx, legend_y + i * 18 + 7, lx + 25, legend_y + i * 18 + 7), fill=color, width=3)
        draw.text((lx + 32, legend_y + i * 18), name[:26], fill=(0, 0, 0))
    img.save(path)


def make_plots_pillow(data: Prepared, runs: dict[str, RunResult], metrics: pd.DataFrame, plots: Path) -> None:
    from PIL import Image, ImageDraw

    if data.gt_pos is not None:
        series = [
            (name, data.time_s, np.linalg.norm(run.states[:, :3] - data.gt_pos, axis=1))
            for name, run in runs.items()
        ]
    else:
        series = []
    draw_fallback_plot(plots / "error_over_time.png", "Error over time", series, "time [s]", "error [m]")

    xy_series = []
    if data.gt_pos is not None:
        xy_series.append(("ground_truth", data.gt_pos[:, 0], data.gt_pos[:, 1]))
    xy_series += [(name, run.states[:, 0], run.states[:, 1]) for name, run in runs.items()]
    if data.known_anchor_pos is not None and len(data.known_anchor_pos) > 0:
        xy_series.append(("anchors_floor", data.known_anchor_pos[:, 0], data.known_anchor_pos[:, 1]))
    draw_fallback_plot(plots / "xy_trajectory_comparison.png", "XY trajectory comparison", xy_series, "x [m]", "y [m]")

    z_series = []
    if data.gt_pos is not None:
        z_series.append(("ground_truth_z", data.time_s, data.gt_pos[:, 2]))
    if data.imu_pos is not None:
        z_series.append(("imu_z", data.time_s, data.imu_pos[:, 2]))
    if data.uwb_pos is not None:
        z_series.append(("uwb_z", data.time_s, data.uwb_pos[:, 2]))
        if data.uwb_valid_axes is not None:
            z_series.append(("uwb_z_valid", data.time_s, np.where(data.uwb_valid_axes[:, 2], data.uwb_pos[:, 2], np.nan)))
    z_series += [(name, data.time_s, run.states[:, 2]) for name, run in runs.items()]
    draw_fallback_plot(plots / "z_comparison.png", "Z comparison", z_series, "time [s]", "z [m]")

    img = Image.new("RGB", (1000, 600), "white")
    draw = ImageDraw.Draw(img)
    draw.text((60, 20), "RMSE barplot", fill=(0, 0, 0))
    if metrics["rmse_3d_m"].notna().any():
        ordered = metrics.sort_values("rmse_3d_m").reset_index(drop=True)
        vals = ordered["rmse_3d_m"].to_numpy(dtype=float)
        vmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0
        vmax = vmax if vmax > EPS else 1.0
        left, base, bar_w = 80, 500, 95
        for i, row in ordered.iterrows():
            val = float(row["rmse_3d_m"])
            bh = int((val / vmax) * 380)
            x0 = left + i * (bar_w + 45)
            hex_color = plot_color(row["algorithm"])
            try:
                color = tuple(int(hex_color.lstrip("#")[j : j + 2], 16) for j in (0, 2, 4))
            except Exception:
                color = (76, 120, 168)
            draw.rectangle((x0, base - bh, x0 + bar_w, base), fill=color)
            draw.text((x0, base + 10), str(row["algorithm"])[:14], fill=(0, 0, 0))
            draw.text((x0, base - bh - 20), f"{val:.3f}", fill=(0, 0, 0))
        draw.line((60, base, 940, base), fill=(0, 0, 0), width=2)
    else:
        draw.text((420, 280), "Ground truth not provided", fill=(0, 0, 0))
    img.save(plots / "rmse_barplot.png")


def make_plots(data: Prepared, runs: dict[str, RunResult], metrics: pd.DataFrame, plots: Path) -> None:
    if not HAVE_MATPLOTLIB:
        make_plots_pillow(data, runs, metrics, plots)
        return

    plt.figure(figsize=(11, 6))
    if data.gt_pos is not None:
        for name, run in runs.items():
            plt.plot(data.time_s, np.linalg.norm(run.states[:, :3] - data.gt_pos, axis=1), label=name, linewidth=1.2, color=plot_color(name))
        plt.ylabel("3D error [m]")
    else:
        plt.text(0.5, 0.5, "Ground truth not provided", ha="center", va="center")
        plt.ylabel("error [m]")
    plt.xlabel("time [s]")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(plots / "error_over_time.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 7))
    if data.gt_pos is not None:
        plt.plot(data.gt_pos[:, 0], data.gt_pos[:, 1], linestyle="--", color=plot_color("ground_truth"), linewidth=2, label="ground_truth")
    if data.uwb_xy_obs is not None:
        plt.scatter(data.uwb_xy_obs[:, 0], data.uwb_xy_obs[:, 1], s=4, alpha=0.25, color=plot_color("uwb_xy"), label="uwb_xy")
    if data.known_anchor_pos is not None and len(data.known_anchor_pos) > 0:
        plt.scatter(data.known_anchor_pos[:, 0], data.known_anchor_pos[:, 1], marker="^", s=70, color="black", label="anchors_floor")
        for anchor_id, pos in zip(data.known_anchor_ids, data.known_anchor_pos):
            plt.annotate(anchor_id, (pos[0], pos[1]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    for name, run in runs.items():
        plt.plot(run.states[:, 0], run.states[:, 1], linewidth=1.2, label=name, color=plot_color(name))
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plots / "xy_trajectory_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(11, 6))
    if data.gt_pos is not None:
        plt.plot(data.time_s, data.gt_pos[:, 2], linestyle="--", color=plot_color("ground_truth"), linewidth=2, label="ground_truth_z")
    if data.imu_pos is not None:
        plt.plot(data.time_s, data.imu_pos[:, 2], color=plot_color("imu_z"), linewidth=1.1, label="imu_z")
    if data.uwb_pos is not None:
        plt.plot(data.time_s, data.uwb_pos[:, 2], color=plot_color("uwb_z"), alpha=0.25, linewidth=1.0, label="uwb_z_raw")
        if data.uwb_valid_axes is not None:
            plt.plot(
                data.time_s,
                np.where(data.uwb_valid_axes[:, 2], data.uwb_pos[:, 2], np.nan),
                color=plot_color("uwb_z_valid"),
                linewidth=1.2,
                label="uwb_z_valid",
            )
    for name, run in runs.items():
        plt.plot(data.time_s, run.states[:, 2], linewidth=1.2, label=name, color=plot_color(name))
    plt.xlabel("time [s]")
    plt.ylabel("z [m]")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(plots / "z_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    if metrics["rmse_3d_m"].notna().any():
        ordered = metrics.sort_values("rmse_3d_m")
        plt.bar(ordered["algorithm"], ordered["rmse_3d_m"], color=[plot_color(v) for v in ordered["algorithm"]])
        plt.ylabel("3D RMSE [m]")
        plt.xticks(rotation=25, ha="right")
    else:
        plt.text(0.5, 0.5, "Ground truth not provided", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(plots / "rmse_barplot.png", dpi=300)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="UWB + IMU localization fusion comparison")
    p.add_argument("--manifest", default=None, help="Optional experiment manifest.json with sensor paths, ids, scales, and time shifts.")
    p.add_argument("--data-clean-root", default=None, help="Batch mode: root folder containing expXXX/manifest.json experiment folders.")
    p.add_argument("--drone-name", default="drone1", help="Drone key to use from manifest.json.")
    p.add_argument("--uwb", default=None)
    p.add_argument("--imu", default=None)
    p.add_argument("--anchors", default=None)
    p.add_argument("--ground-truth", default=None)
    p.add_argument("--output", default="results")
    p.add_argument("--dt", type=parse_dt, default=0.02)
    p.add_argument("--c-gamma", type=float, default=0.7)
    p.add_argument("--a-gammas", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.75, 0.9])
    p.add_argument("--uwb-id", type=int, default=None)
    p.add_argument("--uwb-time-shift", type=float, default=None, help="Seconds added to UWB timestamps. Manifest value is used when omitted.")
    p.add_argument("--imu-time-shift", type=float, default=None, help="Seconds added to IMU timestamps. Manifest value is used when omitted.")
    p.add_argument("--gt-time-shift", type=float, default=None, help="Seconds added to ground-truth timestamps.")
    p.add_argument("--trim-t-end", type=parse_dt, default=None, help="Optional duration in seconds after common overlap start, normally read from manifest trim_t_end.")
    p.add_argument("--imu-pos-scale", type=float, default=None)
    p.add_argument("--imu-perm", nargs=3, type=int, default=None, help="Axis permutation for IMU position/accel, one-based like manifest perm 2 1 3.")
    p.add_argument("--imu-sign", nargs=3, type=float, default=None, help="Axis signs for IMU position/accel.")
    p.add_argument("--imu-position-offset", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"), help="Constant IMU position offset in meters.")
    p.add_argument("--anchor-points", type=int, default=None)
    p.add_argument("--auto-tune-shifts", action="store_true", help="Tune UWB and IMU time shifts against ground truth before running filters.")
    p.add_argument("--shift-search-radius", type=float, default=2.0, help="Search +/- this many seconds around manifest/manual shifts.")
    p.add_argument("--shift-search-step", type=float, default=0.1, help="Time-shift grid step in seconds.")
    p.add_argument(
        "--auto-tune-imu-offset",
        choices=["none", "z", "xyz"],
        default="z",
        help="Constant IMU position offset estimated from ground truth during auto-tune. Use none for no coordinate offset.",
    )
    p.add_argument("--save-tuned-manifest", default=None, help="Path for tuned manifest copy. Defaults to <output>/tuned_manifest.json.")
    p.add_argument("--vicon-fps", type=float, default=120.0)
    p.add_argument("--q-pos", type=float, default=0.05, help="Position process std for CV-family filters; kept flexible for hover, turns, and vertical changes.")
    p.add_argument("--q-vel", type=float, default=0.35, help="Velocity process std for CV-family filters; higher default makes these filters smoothers rather than rigid CV trackers.")
    p.add_argument("--uwb-kf-q-pos", type=float, default=0.04, help="UWB-only CA Kalman process std for position states.")
    p.add_argument("--uwb-kf-q-vel", type=float, default=0.45, help="UWB-only CA Kalman process std for velocity states.")
    p.add_argument("--uwb-kf-q-acc", type=float, default=1.50, help="UWB-only CA Kalman process std for acceleration random walk.")
    p.add_argument("--uwb-kf-q-jerk", type=float, default=5.0, help="UWB-only CA Kalman white-jerk process std; keeps the smoother flexible through starts, stops, turns, and PID corrections.")
    p.add_argument("--enable-adaptive-motion-q", action="store_true", help="Adapt q_vel from IMU motion: q_vel_eff = q_vel * (1 + k_acc*|a| + k_jerk*|jerk|).")
    p.add_argument("--k-acc", type=float, default=0.0, help="Acceleration gain for adaptive q_vel_eff.")
    p.add_argument("--k-jerk", type=float, default=0.0, help="Jerk gain for adaptive q_vel_eff.")
    p.add_argument("--adaptive-q-max-scale", type=float, default=10.0, help="Upper clamp for q_vel_eff / q_vel.")
    p.add_argument("--adaptive-q-smooth-window", type=int, default=5, help="Centered rolling mean window for adaptive q scale.")
    p.add_argument("--drone-mass-scale", type=float, default=1.0, help="Relative mass/inertia scale. Values >1 simulate a heavier, less agile drone by reducing q_vel_eff.")
    p.add_argument("--mass-q-exponent", type=float, default=0.5, help="Exponent for mass scaling: q_vel_eff *= mass_scale^(-exponent).")
    p.add_argument("--tune-adaptive-motion-q", action="store_true", help="Grid-search small k_acc/k_jerk values using ground truth.")
    p.add_argument("--k-acc-grid", nargs="+", type=float, default=[0.01, 0.02, 0.03, 0.04, 0.05])
    p.add_argument("--k-jerk-grid", nargs="+", type=float, default=[0.0005, 0.001, 0.0015, 0.002, 0.003])
    p.add_argument("--adaptive-q-tune-target", default="adaptive_covariance_fusion", help="Algorithm used to choose the best k grid pair.")
    p.add_argument("--enable-temperature-proxy", action="store_true", help="Build a synthetic IMU temperature profile over flight time.")
    p.add_argument("--temp-start-c", type=float, default=25.0)
    p.add_argument("--temp-end-c", type=float, default=70.0)
    p.add_argument("--temp-profile-power", type=float, default=1.0)
    p.add_argument("--temp-ref-c", type=float, default=None)
    p.add_argument("--temp-z-bias-m-per-c", type=float, default=0.0, help="Subtract this IMU z drift per degree C from the trusted z measurement.")
    p.add_argument("--auto-tune-temp-z-bias", action="store_true", help="Estimate temperature-proxy IMU z slope from ground truth.")
    p.add_argument("--temp-r-imu-z-gain", type=float, default=0.0, help="Increase r_imu_z with temperature distance from temp-ref-c.")
    p.add_argument("--disable-factor-graph", action="store_true", help="Do not run the factor graph smoother.")
    p.add_argument("--include-factor-graph", action="store_true", help="Batch mode only: include the slower factor graph smoother. Default is off in batch mode.")
    p.add_argument("--fgo-window-size", type=int, default=40, help="Number of states in each fixed-lag factor graph window.")
    p.add_argument("--fgo-stride", type=int, default=20, help="Stride between factor graph windows.")
    p.add_argument("--fgo-iterations", type=int, default=3, help="Gauss-Newton/LM iterations per factor graph window.")
    p.add_argument("--fgo-lm-damping", type=float, default=1e-3)
    p.add_argument("--fgo-step-tol", type=float, default=1e-5)
    p.add_argument("--fgo-motion-pos-scale", type=float, default=1.0, help="Multiplier for q_pos in factor graph dynamics factors.")
    p.add_argument("--fgo-motion-vel-scale", type=float, default=1.0, help="Multiplier for q_vel_eff in factor graph dynamics factors.")
    p.add_argument("--fgo-prior-sigma", nargs=6, type=float, default=[0.25, 0.25, 0.10, 1.0, 1.0, 0.5])
    p.add_argument("--fgo-use-imu-velocity", action="store_true", help="Add velocity measurement factors derived from IMU position.")
    p.add_argument("--fgo-r-imu-vel", type=float, default=0.8)
    p.add_argument("--fgo-blend", choices=["triangular", "uniform"], default="triangular")
    p.add_argument("--r-uwb-xy", type=float, default=0.25)
    p.add_argument("--r-uwb-z", type=float, default=0.35, help="Measurement std for valid UWB z in the UWB-only Kalman baseline.")
    p.add_argument("--uwb-z-floor-threshold", type=float, default=0.0, help="UWB z values below this threshold are marked invalid and skipped for z updates.")
    p.add_argument("--enable-uwb-xy-nis-gate", action="store_true", help="Reject or downweight coordinate-only UWB x/y updates using a 2D NIS gate.")
    p.add_argument("--uwb-xy-nis-threshold", type=float, default=9.21, help="2D chi-square NIS threshold. 9.21 is about 99%% for 2 DOF.")
    p.add_argument("--uwb-xy-gate-action", choices=["skip", "inflate"], default="skip", help="Skip the UWB x/y update or inflate R when the gate fails.")
    p.add_argument("--uwb-xy-gate-inflate-scale", type=float, default=100.0, help="R multiplier used when --uwb-xy-gate-action inflate is selected.")
    p.add_argument("--r-uwb-range", type=float, default=0.35)
    p.add_argument("--r-imu-z", type=float, default=0.03)
    p.add_argument("--r-imu-xy", type=float, default=1.0)
    p.add_argument("--imu-vel-weight", type=float, default=0.60)
    p.add_argument("--ukf-alpha", type=float, default=0.1)
    p.add_argument("--ukf-beta", type=float, default=2.0)
    p.add_argument("--ukf-kappa", type=float, default=0.0)
    p.add_argument("--top-k-best-filters", type=int, default=None, help="Batch summary plot: keep only the top-k filters by winner count.")
    p.add_argument("--only-best-filters-plots", action="store_true", help="Batch summary plots show only winner filters. This is the default selection policy.")
    return p


def normalize_runtime_args(args: argparse.Namespace) -> None:
    if args.uwb_time_shift is None:
        args.uwb_time_shift = 0.0
    if args.imu_time_shift is None:
        args.imu_time_shift = 0.0
    if args.gt_time_shift is None:
        args.gt_time_shift = 0.0
    if args.imu_pos_scale is None:
        args.imu_pos_scale = 0.1
    if isinstance(args.imu_position_offset, list):
        args.imu_position_offset = {"x": args.imu_position_offset[0], "y": args.imu_position_offset[1], "z": args.imu_position_offset[2]}
    if args.anchor_points is None:
        args.anchor_points = 5


def manifest_paths_under(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("manifest.json") if path.is_file())


def batch_drone_names(manifest: dict) -> list[str]:
    drones = manifest.get("drones", {}) or {}
    ordered = [name for name in ("drone1", "drone2") if name in drones]
    return ordered or sorted(drones.keys())


def make_batch_run_args(base_args: argparse.Namespace, manifest_path: Path, drone_name: str, output_dir: Path, make_color_plots: bool) -> argparse.Namespace:
    run_args = argparse.Namespace(**vars(base_args))
    run_args.manifest = str(manifest_path)
    run_args.drone_name = drone_name
    run_args.output = str(output_dir)
    run_args.save_tuned_manifest = None
    run_args.suppress_plots = not make_color_plots
    run_args.manifest_data = None
    run_args.exp_id = manifest_path.parent.name
    run_args.uwb = None
    run_args.imu = None
    run_args.ground_truth = None
    run_args.uwb_id = None
    return run_args


def summarize_batch_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(columns=["exp_id", "drone", "filter", "rmse_3d", "rmse_xy", "rmse_z", "n_valid", "wins_flag"])
    summary = metrics.rename(
        columns={
            "drone_name": "drone",
            "algorithm": "filter",
            "rmse_3d_m": "rmse_3d",
            "rmse_xy_m": "rmse_xy",
            "rmse_z_m": "rmse_z",
            "n_samples": "n_valid",
        }
    )
    keep = ["exp_id", "drone", "filter", "rmse_3d", "rmse_xy", "rmse_z", "n_valid"]
    summary = summary[[c for c in keep if c in summary.columns]].copy()
    for col in ("rmse_3d", "rmse_xy", "rmse_z"):
        summary[col] = pd.to_numeric(summary[col], errors="coerce")
    summary["n_valid"] = pd.to_numeric(summary["n_valid"], errors="coerce").fillna(0).astype(int)
    summary["wins_flag"] = False
    for _, group in summary[np.isfinite(summary["rmse_3d"])].groupby(["exp_id", "drone"], dropna=False):
        if group.empty:
            continue
        winner_idx = group["rmse_3d"].idxmin()
        summary.loc[winner_idx, "wins_flag"] = True
    return summary


def selected_summary_filters(summary: pd.DataFrame, top_k: int | None) -> list[str]:
    if summary.empty:
        return []
    winners = summary[summary["wins_flag"]]["filter"].astype(str)
    if winners.empty:
        return sorted(summary["filter"].astype(str).unique())
    counts = winners.value_counts()
    if top_k is not None and top_k > 0:
        return counts.head(int(top_k)).index.tolist()
    return counts.index.tolist()


def make_summary_best_filters_plot(
    summary: pd.DataFrame,
    selected: list[str],
    out_dir: Path,
    stem: str = "summary_best_filters",
    title: str = "Best-filter RMSE distribution across experiments and drones",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_df = summary[summary["filter"].isin(selected) & np.isfinite(summary["rmse_3d"])].copy()
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    if plot_df.empty or not selected:
        pd.DataFrame({"message": ["No finite RMSE values for summary plot"]}).to_csv(out_dir / f"{stem}_note.csv", index=False)
        return

    if HAVE_MATPLOTLIB:
        labels = selected
        data = [plot_df.loc[plot_df["filter"] == name, "rmse_3d"].to_numpy(dtype=float) for name in labels]
        fig_w = max(7.0, 0.8 * len(labels))
        fig, ax = plt.subplots(figsize=(fig_w, 4.2))
        bp = ax.boxplot(data, labels=[display_filter_name(v) for v in labels], patch_artist=True, showmeans=True, meanline=False)
        for patch, name in zip(bp["boxes"], labels):
            patch.set_facecolor(plot_color(name))
            patch.set_alpha(0.55)
            patch.set_edgecolor("#222222")
        for key in ("whiskers", "caps", "medians", "means"):
            for artist in bp.get(key, []):
                artist.set_color("#222222")
                artist.set_linewidth(1.1)
        for i, name in enumerate(labels, start=1):
            vals = plot_df.loc[plot_df["filter"] == name, "rmse_3d"].to_numpy(dtype=float)
            if vals.size:
                jitter = np.linspace(-0.08, 0.08, vals.size) if vals.size > 1 else np.array([0.0])
                ax.scatter(np.full(vals.size, i) + jitter, vals, s=24, color=plot_color(name), edgecolor="#222222", linewidth=0.4, alpha=0.85, zorder=3)
        ax.set_ylabel("3D RMSE [m]")
        ax.set_xlabel("Selected winner filters")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=25)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        fig.tight_layout()
        fig.savefig(png_path, dpi=300)
        fig.savefig(pdf_path)
        plt.close(fig)
        return

    from PIL import Image, ImageDraw, ImageFont

    w, h = max(1200, 180 * len(selected)), 760
    left, right, top, bottom = 100, 40, 70, 155
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    font_dir = Path("C:/Windows/Fonts")

    def font(size: int, bold: bool = False):
        names = ["timesbd.ttf", "arialbd.ttf"] if bold else ["times.ttf", "arial.ttf"]
        for name in names:
            path = font_dir / name
            if path.exists():
                return ImageFont.truetype(str(path), size)
        return ImageFont.load_default()

    f_title = font(24, True)
    f_axis = font(18)
    f_tick = font(15)
    f_small = font(13)

    def draw_centered(text: str, xy: tuple[int, int], fill=(0, 0, 0), f=None):
        f = f or f_tick
        box = draw.textbbox((0, 0), text, font=f)
        draw.text((xy[0] - (box[2] - box[0]) // 2, xy[1] - (box[3] - box[1]) // 2), text, fill=fill, font=f)

    def draw_rotated(text: str, xy: tuple[int, int], angle: float, f=None):
        f = f or f_tick
        box = draw.textbbox((0, 0), text, font=f)
        tmp = Image.new("RGBA", (box[2] - box[0] + 12, box[3] - box[1] + 12), (255, 255, 255, 0))
        td = ImageDraw.Draw(tmp)
        td.text((6, 6), text, fill=(0, 0, 0, 255), font=f)
        rot = tmp.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        img.paste(rot, (xy[0] - rot.width // 2, xy[1] - rot.height // 2), rot)

    vals_all = plot_df["rmse_3d"].to_numpy(dtype=float)
    ymin, ymax = 0.0, float(np.nanmax(vals_all))
    ymax = ymax * 1.15 if ymax > EPS else 1.0

    def ymap(v: float) -> int:
        return h - bottom - int((v - ymin) / (ymax - ymin) * (h - top - bottom))

    draw_centered(title, (w // 2, 30), f=f_title)
    draw.line((left, top, left, h - bottom, w - right, h - bottom), fill=(0, 0, 0), width=2)
    for tick in np.linspace(ymin, ymax, 5):
        y = ymap(float(tick))
        draw.line((left, y, w - right, y), fill=(220, 220, 220), width=1)
        draw.text((34, y - 8), f"{tick:.2f}", fill=(0, 0, 0), font=f_tick)
    step = (w - left - right) / max(len(selected), 1)
    for i, name in enumerate(selected):
        vals = plot_df.loc[plot_df["filter"] == name, "rmse_3d"].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        q1, med, q3 = np.nanpercentile(vals, [25, 50, 75])
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        cx = int(left + step * (i + 0.5))
        box_w = int(min(70, step * 0.55))
        hex_color = plot_color(name).lstrip("#")
        color = tuple(int(hex_color[j : j + 2], 16) for j in (0, 2, 4))
        draw.line((cx, ymap(lo), cx, ymap(hi)), fill=(30, 30, 30), width=2)
        draw.rectangle((cx - box_w // 2, ymap(q3), cx + box_w // 2, ymap(q1)), fill=color, outline=(30, 30, 30))
        draw.line((cx - box_w // 2, ymap(med), cx + box_w // 2, ymap(med)), fill=(0, 0, 0), width=2)
        for v in vals:
            draw.ellipse((cx - 3, ymap(float(v)) - 3, cx + 3, ymap(float(v)) + 3), fill=color, outline=(30, 30, 30))
        draw_rotated(display_filter_name(name), (cx, h - bottom + 58), 25, f_small)
    draw_centered("Selected winner filters", (w // 2, h - 32), f=f_axis)
    draw.text((12, h // 2 - 12), "3D RMSE [m]", fill=(0, 0, 0), font=f_axis)
    img.save(png_path, dpi=(300, 300))
    try:
        img.save(pdf_path, "PDF", resolution=300.0)
    except Exception:
        pass


def run_batch(args: argparse.Namespace) -> None:
    root = Path(args.data_clean_root)
    if not root.exists():
        raise FileNotFoundError(root)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    manifests = manifest_paths_under(root)
    if not manifests:
        raise ValueError(f"No manifest.json files found under {root}")

    all_metrics = []
    failure_rows = []
    for exp_rank, manifest_path in enumerate(manifests, start=1):
        exp_dir = manifest_path.parent
        exp_id = exp_dir.name
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            drones = batch_drone_names(manifest)
        except Exception as exc:
            failure_rows.append({"exp_id": exp_id, "drone": "", "stage": "manifest", "error": f"{type(exc).__name__}: {exc}"})
            continue
        for drone_name in drones:
            run_out = out_root / exp_id / drone_name
            run_args = make_batch_run_args(args, manifest_path, drone_name, run_out, make_color_plots=(exp_rank <= 6))
            try:
                resolve_manifest_args(run_args)
                normalize_runtime_args(run_args)
                validate_required_paths(run_args)
                data = prepare(run_args)
                if run_args.tune_adaptive_motion_q:
                    data.adaptive_q_tuning = run_adaptive_motion_q_grid(run_args, data)
                runs = run_all(run_args, data, include_factor_graph=bool(run_args.include_factor_graph))
                metrics = write_outputs(run_args, data, runs)
                all_metrics.append(metrics)
                print(f"OK: {exp_id} {drone_name} -> {run_out}")
            except Exception as exc:
                failure_rows.append({"exp_id": exp_id, "drone": drone_name, "stage": "run", "error": f"{type(exc).__name__}: {exc}"})
                print(f"SKIP: {exp_id} {drone_name} -> {type(exc).__name__}: {exc}")

    if failure_rows:
        pd.DataFrame(failure_rows).to_csv(out_root / "batch_failures.csv", index=False)
    if not all_metrics:
        raise ValueError("Batch finished without successful experiment/drone runs")

    metrics_all = pd.concat(all_metrics, ignore_index=True, sort=False)
    summary = summarize_batch_metrics(metrics_all)
    selected = selected_summary_filters(summary, args.top_k_best_filters)
    summary.to_csv(out_root / "summary_rmse.csv", index=False)
    make_summary_best_filters_plot(summary, selected, out_root)
    created_summary_plots = ["summary_best_filters.png", "summary_best_filters.pdf"]
    for drone in sorted(summary["drone"].dropna().astype(str).unique()):
        drone_summary = summary[summary["drone"].astype(str) == drone].copy()
        drone_selected = selected_summary_filters(drone_summary, args.top_k_best_filters)
        stem = f"summary_best_filters_{norm_name(drone)}"
        make_summary_best_filters_plot(
            drone_summary,
            drone_selected,
            out_root,
            stem=stem,
            title=f"Best-filter RMSE distribution for {drone}",
        )
        created_summary_plots.extend([f"{stem}.png", f"{stem}.pdf"])
    print("Created:")
    for rel in ["summary_rmse.csv", *created_summary_plots, "batch_failures.csv"]:
        path = out_root / rel
        if path.exists():
            print(path)


def main() -> None:
    args = build_parser().parse_args()
    if args.data_clean_root:
        run_batch(args)
        return
    resolve_manifest_args(args)
    normalize_runtime_args(args)
    validate_required_paths(args)
    data = prepare(args)
    if args.tune_adaptive_motion_q:
        data.adaptive_q_tuning = run_adaptive_motion_q_grid(args, data)
    runs = run_all(args, data)
    write_outputs(args, data, runs)
    print("Created:")
    for rel in [
        "comparison_metrics.csv",
        "error_timeseries.csv",
        "uwb_xy_rejections.csv",
        "fused_trajectories.csv",
        "shift_tuning_summary.csv",
        "adaptive_q_tuning_summary.csv",
        "tuned_manifest.json",
        "plots/error_over_time.png",
        "plots/xy_trajectory_comparison.png",
        "plots/z_comparison.png",
        "plots/rmse_barplot.png",
    ]:
        path = Path(args.output) / rel
        if path.exists():
            print(path)


if __name__ == "__main__":
    main()
