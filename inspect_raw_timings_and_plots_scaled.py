
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CONFIG = {
    "manifest_name": "manifest.json",
    "drone_names": ["drone1", "drone2"],
    "default_uwb_date": "2025-11-07",
    "vicon_fps": 120.0,
    "synthetic_origin": "2025-01-01",
    "default_imu_pos_scale": 0.1,   # IMU positions are in dm -> convert to m
    "default_imu_speed_scale": 0.1, # if speeds are also in dm/s -> convert to m/s
    "out_summary_csv": "raw_timing_summary.csv",
}


def make_unique_time_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_index()
    if df.index.has_duplicates:
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        oth = [c for c in df.columns if c not in num]
        parts = []
        if num:
            parts.append(df[num].groupby(level=0).mean())
        for c in oth:
            parts.append(df[[c]].groupby(level=0).first())
        df = pd.concat(parts, axis=1).sort_index()
    return df


def seconds_to_synth_index(sec, origin: str) -> pd.DatetimeIndex:
    return pd.to_datetime(np.asarray(sec, dtype=float), unit="s", origin=pd.Timestamp(origin))


def index_to_seconds(idx: pd.DatetimeIndex) -> np.ndarray:
    return (idx.view("int64") - idx.view("int64")[0]) / 1e9


def estimate_freq_hz(idx: pd.DatetimeIndex) -> tuple[float, float]:
    if len(idx) < 2:
        return np.nan, np.nan
    dt = np.diff(idx.view("int64")) / 1e9
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return np.nan, np.nan
    return float(np.median(dt)), float(1.0 / np.median(dt))


def read_manifest(exp_dir: Path, cfg: dict) -> dict:
    path = exp_dir / cfg["manifest_name"]
    if not path.exists():
        raise FileNotFoundError(f"{exp_dir}: missing {cfg['manifest_name']}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_overrides(manifest: dict, drone_name: str, sensor_name: str) -> dict:
    out = {}
    out.update(manifest.get("overrides", {}).get(drone_name, {}).get(sensor_name, {}) or {})
    out.update(manifest.get("drones", {}).get(drone_name, {}).get("overrides", {}).get(sensor_name, {}) or {})
    return out


def apply_perm_sign(M: np.ndarray, perm: list[int], sign: list[float]) -> np.ndarray:
    perm0 = [p - 1 for p in perm]
    return M[:, perm0] * np.asarray(sign, dtype=float)


def resolve_relative_shift_seconds(manifest: dict, drone_name: str, sensor_name: str) -> float:
    over = get_overrides(manifest, drone_name, sensor_name)
    if "time_shift_seconds" in over:
        return float(over["time_shift_seconds"])

    raw = manifest.get("drones", {}).get(drone_name, {}).get(f"{sensor_name}_time_shift_seconds", None)
    if raw is not None:
        return float(raw)

    raw2 = manifest.get("drones", {}).get(drone_name, {}).get(f"{sensor_name}_time_shift", None)
    if raw2 is None:
        raw2 = manifest.get(f"{sensor_name}_time_shift", None)
    if raw2 is None:
        return 0.0

    try:
        td = pd.to_timedelta(raw2).total_seconds()
        return 0.0 if abs(td) >= 300 else float(td)
    except Exception:
        try:
            val = float(raw2)
            return 0.0 if abs(val) >= 300 else float(val)
        except Exception:
            return 0.0


def parse_uwb_pos_log(path: Path, date="2025-11-07") -> pd.DataFrame:
    rows = []
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ",POS," not in ln:
            continue
        p = ln.strip().split(",")
        if len(p) < 7:
            continue
        hhmmssms = p[0]
        if len(hhmmssms) != 9 or not hhmmssms.isdigit():
            continue
        hh = int(hhmmssms[0:2]); mm = int(hhmmssms[2:4]); ss = int(hhmmssms[4:6]); ms = int(hhmmssms[6:9])
        t = pd.Timestamp(date) + pd.Timedelta(hours=hh, minutes=mm, seconds=ss, milliseconds=ms)
        try:
            drone_id = int(p[2]) if p[2].isdigit() else np.nan
            x = float(p[4]); y = float(p[5]); z = float(p[6])
        except ValueError:
            continue
        qf = np.nan
        if len(p) > 7:
            try:
                qf = float(p[7])
            except Exception:
                pass
        rows.append({"t": t, "drone_id": drone_id, "x": x, "y": y, "z": z, "qf": qf})
    df = pd.DataFrame(rows)
    return make_unique_time_df(df.sort_values("t").set_index("t")) if not df.empty else df


def read_imu_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Time" not in df.columns:
        raise ValueError(f"IMU file {path.name} has no Time column")
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).copy()

    t = df["Time"].to_numpy(dtype=float)
    t_rel = (t - t[0]) / 1000.0 if np.nanmedian(t) > 1e11 else (t - t[0])
    df.index = seconds_to_synth_index(t_rel, CONFIG["synthetic_origin"])

    for c in ["Ax", "Ay", "Az", "Sx", "Sy", "Sz", "x", "y", "z"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return make_unique_time_df(df)


def read_vicon_csv(path: Path, fps: float) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=2, header=None).iloc[:, :5].copy()
    df.columns = ["frame", "subframe", "tx_mm", "ty_mm", "tz_mm"]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["frame", "tx_mm", "ty_mm", "tz_mm"]).sort_values(["frame", "subframe"])

    t = (df["frame"].to_numpy(dtype=float) - df["frame"].iloc[0]) / fps
    out = pd.DataFrame({
        "x": df["tx_mm"].to_numpy(dtype=float) / 1000.0,
        "y": df["ty_mm"].to_numpy(dtype=float) / 1000.0,
        "z": df["tz_mm"].to_numpy(dtype=float) / 1000.0,
    }, index=seconds_to_synth_index(t, CONFIG["synthetic_origin"]))
    return make_unique_time_df(out)


def maybe_apply_mapping(df_xyz: pd.DataFrame, manifest: dict, drone_name: str, sensor_name: str) -> pd.DataFrame:
    over = get_overrides(manifest, drone_name, sensor_name)
    if "perm" in over and "sign" in over and all(c in df_xyz.columns for c in ["x", "y", "z"]):
        M = apply_perm_sign(df_xyz[["x", "y", "z"]].to_numpy(dtype=float), over["perm"], over["sign"])
        out = df_xyz.copy()
        out[["x", "y", "z"]] = M
        return out
    return df_xyz


def apply_imu_scaling(imu_df: pd.DataFrame, manifest: dict) -> tuple[pd.DataFrame, float, float]:
    pos_scale = float(manifest.get("imu_pos_scale", CONFIG["default_imu_pos_scale"]))
    speed_scale = float(manifest.get("imu_speed_scale", CONFIG["default_imu_speed_scale"]))

    out = imu_df.copy()
    if all(c in out.columns for c in ["x", "y", "z"]):
        out[["x", "y", "z"]] = out[["x", "y", "z"]].to_numpy(dtype=float) * pos_scale
    if all(c in out.columns for c in ["Sx", "Sy", "Sz"]):
        out[["Sx", "Sy", "Sz"]] = out[["Sx", "Sy", "Sz"]].to_numpy(dtype=float) * speed_scale
    return out, pos_scale, speed_scale


def plot_coords_vs_time(exp_id: str, drone_name: str, out_dir: Path,
                        vicon: pd.DataFrame | None, imu: pd.DataFrame | None, uwb: pd.DataFrame | None):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    labels = [("Vicon", vicon), ("IMU", imu), ("UWB", uwb)]
    names = ["X", "Y", "Z"]

    for i, ax in enumerate(axes):
        for label, df in labels:
            if df is None or df.empty or not all(c in df.columns for c in ["x", "y", "z"]):
                continue
            t = index_to_seconds(df.index)
            y = df[["x", "y", "z"]].iloc[:, i].to_numpy(dtype=float)
            ax.plot(t, y, label=label, linewidth=1.1)
        ax.set_ylabel(f"{names[i]} [m]")
        ax.grid(True)
        if i == 0:
            ax.set_title(f"{exp_id} - {drone_name}: raw coordinates vs time")
        if i == 2:
            ax.set_xlabel("Time [s]")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{exp_id}_{drone_name}_raw_coords_vs_time.png", dpi=200)
    plt.close(fig)


def plot_xy(exp_id: str, drone_name: str, out_dir: Path,
            vicon: pd.DataFrame | None, imu: pd.DataFrame | None, uwb: pd.DataFrame | None):
    fig, ax = plt.subplots(figsize=(7, 7))
    for label, df in [("Vicon", vicon), ("IMU", imu), ("UWB", uwb)]:
        if df is None or df.empty or not all(c in df.columns for c in ["x", "y", "z"]):
            continue
        x = df["x"].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.any():
            ax.plot(x[mask], y[mask], label=label, linewidth=1.2)
    ax.set_title(f"{exp_id} - {drone_name}: raw XY trajectory")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{exp_id}_{drone_name}_raw_xy_trajectory.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Inspect raw Vicon/IMU/UWB timing and plots without filters/fusion.")
    parser.add_argument("--data-root", required=True, help="Path to data_clean")
    parser.add_argument("--out-dir", default="reports_raw_inspection_scaled", help="Output folder")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_dirs = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.lower().startswith("exp")])
    if not exp_dirs:
        raise FileNotFoundError("No exp* folders found")

    rows = []

    for exp_dir in exp_dirs:
        manifest = read_manifest(exp_dir, CONFIG)
        uwb_path = exp_dir / manifest["uwb"]["path"]
        uwb_date = manifest.get("date", CONFIG["default_uwb_date"])
        uwb_all = parse_uwb_pos_log(uwb_path, uwb_date)

        for drone_name in CONFIG["drone_names"]:
            info = manifest["drones"][drone_name]

            vicon = None
            vicon_dt = np.nan
            vicon_hz = np.nan
            vicon_path = exp_dir / info["vicon"]
            if vicon_path.exists():
                vicon = read_vicon_csv(vicon_path, CONFIG["vicon_fps"])
                vicon_dt, vicon_hz = estimate_freq_hz(vicon.index)

            imu = None
            imu_dt = np.nan
            imu_hz = np.nan
            imu_shift = 0.0
            imu_pos_scale = np.nan
            imu_speed_scale = np.nan
            imu_path_rel = info.get("imu")
            if imu_path_rel:
                imu_path = exp_dir / imu_path_rel
                if imu_path.exists():
                    imu = read_imu_csv(imu_path)
                    imu, imu_pos_scale, imu_speed_scale = apply_imu_scaling(imu, manifest)
                    imu = maybe_apply_mapping(imu[["x", "y", "z"]].copy(), manifest, drone_name, "imu")
                    imu_shift = resolve_relative_shift_seconds(manifest, drone_name, "imu")
                    imu.index = imu.index + pd.to_timedelta(imu_shift, unit="s")
                    imu_dt, imu_hz = estimate_freq_hz(imu.index)

            uwb = None
            uwb_dt = np.nan
            uwb_hz = np.nan
            uwb_shift = 0.0
            uwb_id = int(info["uwb_id"])
            if not uwb_all.empty:
                uwb = uwb_all[uwb_all["drone_id"] == uwb_id][["x", "y", "z", "qf"]].copy()
                if not uwb.empty:
                    t = index_to_seconds(uwb.index)
                    uwb.index = seconds_to_synth_index(t, CONFIG["synthetic_origin"])
                    uwb = maybe_apply_mapping(uwb, manifest, drone_name, "uwb")
                    uwb_shift = resolve_relative_shift_seconds(manifest, drone_name, "uwb")
                    if "uwb_time_shift" in manifest:
                        uwb_shift = float(manifest["uwb_time_shift"])
                    uwb.index = uwb.index + pd.to_timedelta(uwb_shift, unit="s")
                    uwb_dt, uwb_hz = estimate_freq_hz(uwb.index)

            rows.append({
                "exp_id": exp_dir.name,
                "drone_name": drone_name,
                "vicon_dt_s": vicon_dt,
                "vicon_hz": vicon_hz,
                "imu_dt_s": imu_dt,
                "imu_hz": imu_hz,
                "uwb_dt_s": uwb_dt,
                "uwb_hz": uwb_hz,
                "imu_shift_s": imu_shift,
                "uwb_shift_s": uwb_shift,
                "imu_pos_scale": imu_pos_scale,
                "imu_speed_scale": imu_speed_scale,
            })

            plot_coords_vs_time(exp_dir.name, drone_name, out_dir, vicon, imu, uwb)
            plot_xy(exp_dir.name, drone_name, out_dir, vicon, imu, uwb)

    summary = pd.DataFrame(rows)
    summary_path = out_dir / CONFIG["out_summary_csv"]
    summary.to_csv(summary_path, index=False)

    print("\nCreated:")
    print(summary_path)
    print(out_dir)
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
