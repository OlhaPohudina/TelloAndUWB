
from __future__ import annotations
import argparse, itertools
from dataclasses import dataclass
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import medfilt, butter, sosfiltfilt
from sklearn.preprocessing import StandardScaler

CONFIG = {
    "dt":"20ms","thr_pos":2.5,
    "use_qf_gate":False,"qf_min":55.0,"alpha_xy":0.35,"alpha_z":0.95,"force_z_from_imu":True,
    "med_win":5,"lp_order":2,"lp_fc_hz":2.0,"dtw_window":10,"dtw_psi":2,"dtw_quality_window":25,
    "enable_auto_tune":False,"auto_tune_experiments":[],"auto_tune_drones":[1,2],
    "enable_ekf":False,"q_pos":0.02,"q_vel":0.10,"r_base":0.15,"r_gain_q":0.15,"r_gain_disagreement":0.60,
    "out_samples_csv":"quality_gated_samples_from_table.csv","out_summary_csv":"quality_gated_summary_from_table.csv",
}

def fs_hz(dt:str)->float: return 1.0/pd.to_timedelta(dt).total_seconds()
def clip_xyz(P,thr): return np.clip(P,-thr,thr)
def normalize_xyz(X): return StandardScaler().fit_transform(np.asarray(X,dtype=np.double))
def make_unique_time_df(df):
    if df.empty: return df
    df=df.sort_index()
    if df.index.has_duplicates:
        num=df.select_dtypes(include=[np.number]).columns.tolist()
        oth=[c for c in df.columns if c not in num]
        parts=[]
        if num: parts.append(df[num].groupby(level=0).mean())
        for c in oth: parts.append(df[[c]].groupby(level=0).first())
        df=pd.concat(parts,axis=1).sort_index()
    return df
def rmse3d(p_ref,p_est):
    m=np.isfinite(p_ref).all(axis=1)&np.isfinite(p_est).all(axis=1)
    if m.sum()<5: return np.nan
    e=p_est[m]-p_ref[m]
    return float(np.sqrt(np.mean(np.sum(e*e,axis=1))))
def resample_to_ref_axis(sensor_df,ref_index):
    sensor_df=make_unique_time_df(sensor_df)
    combo=sensor_df.index.union(ref_index); tmp=sensor_df.reindex(combo).sort_index().interpolate("time")
    return make_unique_time_df(tmp.reindex(ref_index))
def prefilter_xyz(df_xyz,fs,med_win=5,lp_order=2,lp_fc_hz=2.0):
    out=make_unique_time_df(df_xyz.copy()); med_win=int(med_win); med_win=med_win if med_win%2==1 else med_win+1
    for c in ["x","y","z"]: out[c]=medfilt(out[c].to_numpy(),kernel_size=med_win)
    sos=butter(lp_order,lp_fc_hz,btype="low",fs=fs,output="sos")
    for c in ["x","y","z"]: out[c]=sosfiltfilt(sos,out[c].to_numpy())
    return out
def local_dtw_quality(sensor_df,ref_df,window_size,dtw_window,dtw_psi):
    sensor_df=make_unique_time_df(sensor_df).copy(); ref_df=make_unique_time_df(ref_df).copy()
    sensor_df.columns=["x","y","z"]; ref_df.columns=["x","y","z"]
    M=pd.concat([sensor_df.add_prefix("s_"),ref_df.add_prefix("r_")],axis=1,sort=False).dropna()
    if len(M)<window_size+2: return pd.Series(np.nan,index=ref_df.index)
    vals=np.full(len(M),np.nan,dtype=float)
    try:
        from dtaidistance import dtw_ndim
        S=M[["s_x","s_y","s_z"]].to_numpy(dtype=np.double); R=M[["r_x","r_y","r_z"]].to_numpy(dtype=np.double)
        for k in range(window_size,len(M)):
            s_win=normalize_xyz(S[k-window_size:k]); r_win=normalize_xyz(R[k-window_size:k])
            vals[k]=float(dtw_ndim.distance(s_win,r_win,window=min(dtw_window,window_size-1),psi=min(dtw_psi,2),use_c=False,use_pruning=False))
    except Exception:
        pass
    return pd.Series(vals,index=M.index).reindex(ref_df.index).interpolate("time")
def build_measurement_sigma(frame,cfg):
    q=pd.to_numeric(frame["q_uwb"],errors="coerce"); med=np.nanmedian(q.to_numpy(dtype=float))
    if not np.isfinite(med) or med<=0: med=1.0
    d=pd.to_numeric(frame["uwb_raw_filt_disagreement"],errors="coerce").fillna(0.0)
    sigma=cfg["r_base"]+cfg["r_gain_q"]*(q.fillna(med)/med)+cfg["r_gain_disagreement"]*d
    return sigma.clip(lower=cfg["r_base"])
def run_ekf_on_fusion_pre(frame,cfg):
    out=frame.copy(); dt=pd.to_timedelta(cfg["dt"]).total_seconds()
    X=np.zeros((len(out),6),dtype=float); x=np.zeros(6,dtype=float)
    p0=out[["fusion_pre_x","fusion_pre_y","fusion_pre_z"]].iloc[0].to_numpy(dtype=float)
    x[:3]=p0 if np.isfinite(p0).all() else out[["uwb_filt_x","uwb_filt_y","uwb_filt_z"]].iloc[0].to_numpy(dtype=float)
    P=np.diag([0.25,0.25,0.25,0.50,0.50,0.50])**2
    Q=np.diag([cfg["q_pos"],cfg["q_pos"],cfg["q_pos"],cfg["q_vel"],cfg["q_vel"],cfg["q_vel"]])**2
    H=np.hstack([np.eye(3),np.zeros((3,3))]); I=np.eye(6); imu_prev=None; sigma_meas=build_measurement_sigma(out,cfg)
    for k in range(len(out)):
        row=out.iloc[k]
        if np.isfinite(row[["imu_x","imu_y","imu_z"]].to_numpy(dtype=float)).all():
            p_imu=row[["imu_x","imu_y","imu_z"]].to_numpy(dtype=float)
            v_imu=np.zeros(3,dtype=float) if imu_prev is None else (p_imu-imu_prev)/dt
            imu_prev=p_imu.copy(); F=np.block([[np.eye(3),np.zeros((3,3))],[np.zeros((3,3)),np.zeros((3,3))]])
            B=np.vstack([dt*np.eye(3),np.eye(3)]); x=F@x+B@v_imu; P=F@P@F.T+Q
        else:
            F=np.block([[np.eye(3),dt*np.eye(3)],[np.zeros((3,3)),np.eye(3)]])
            x=F@x; P=F@P@F.T+Q
        z=None
        if np.isfinite(row[["fusion_pre_x","fusion_pre_y","fusion_pre_z"]].to_numpy(dtype=float)).all():
            z=row[["fusion_pre_x","fusion_pre_y","fusion_pre_z"]].to_numpy(dtype=float)
        elif np.isfinite(row[["uwb_filt_x","uwb_filt_y","uwb_filt_z"]].to_numpy(dtype=float)).all():
            z=row[["uwb_filt_x","uwb_filt_y","uwb_filt_z"]].to_numpy(dtype=float)
        if z is not None:
            s=float(sigma_meas.iloc[k]) if np.isfinite(sigma_meas.iloc[k]) else cfg["r_base"]
            R=np.diag([s,s,s])**2; y=z-H@x; S=H@P@H.T+R; K=P@H.T@np.linalg.inv(S); x=x+K@y; P=(I-K@H)@P
        X[k]=x
    out["fusion_ekf_x"]=X[:,0]; out["fusion_ekf_y"]=X[:,1]; out["fusion_ekf_z"]=X[:,2]
    return out

@dataclass
class PreparedDrone:
    exp_id:str
    drone_name:str
    frame:pd.DataFrame

def preprocess_one_group(df_group,cfg):
    exp_id=str(df_group["exp_id"].iloc[0]); drone_name=str(df_group["drone_name"].iloc[0])
    idx=pd.to_datetime(df_group.index); t0=idx.min()
    def extract(prefix, cols):
        wanted=[f"{prefix}_{c}" for c in cols if f"{prefix}_{c}" in df_group.columns]
        if not wanted: return None
        out=df_group[wanted].copy(); out.columns=cols[:len(wanted)]; out.index=idx
        out=out.dropna(how="all")
        return make_unique_time_df(out)
    vicon=extract("vicon",["x","y","z"])
    if vicon is None or vicon.empty: raise ValueError("Prepared group has no Vicon")
    imu=extract("imu",["x","y","z"])
    uwb=extract("uwb",["x","y","z"])
    qf=extract("uwb",["qf"])
    fs=fs_hz(cfg["dt"])
    vicon_rs=make_unique_time_df(vicon.resample(cfg["dt"]).mean().interpolate("time"))
    if uwb is not None and not uwb.empty:
        uwb_rs=make_unique_time_df(uwb.resample(cfg["dt"]).mean().interpolate("time"))
        uwb_raw=make_unique_time_df(resample_to_ref_axis(uwb_rs, vicon_rs.index))
        uwb_raw=pd.DataFrame(clip_xyz(uwb_raw.to_numpy(dtype=float),cfg["thr_pos"]),index=vicon_rs.index,columns=["x","y","z"])
        uwb_filt=prefilter_xyz(uwb_raw,fs,cfg["med_win"],cfg["lp_order"],cfg["lp_fc_hz"])
    else:
        uwb_raw=pd.DataFrame(np.nan,index=vicon_rs.index,columns=["x","y","z"]); uwb_filt=uwb_raw.copy()
    if imu is not None and not imu.empty:
        imu_rs=make_unique_time_df(imu.resample(cfg["dt"]).mean().interpolate("time"))
        imu_aligned=make_unique_time_df(resample_to_ref_axis(imu_rs,vicon_rs.index))
        imu_aligned=pd.DataFrame(clip_xyz(imu_aligned.to_numpy(dtype=float),cfg["thr_pos"]),index=vicon_rs.index,columns=["x","y","z"])
    else:
        imu_aligned=pd.DataFrame(np.nan,index=vicon_rs.index,columns=["x","y","z"])
    q_uwb=local_dtw_quality(uwb_filt,vicon_rs,cfg["dtw_quality_window"],cfg["dtw_window"],cfg["dtw_psi"])
    q_imu=local_dtw_quality(imu_aligned,vicon_rs,cfg["dtw_quality_window"],cfg["dtw_window"],cfg["dtw_psi"])
    disagreement=np.linalg.norm(uwb_raw[["x","y","z"]].to_numpy(dtype=float)-uwb_filt[["x","y","z"]].to_numpy(dtype=float),axis=1)
    disagreement=pd.Series(disagreement,index=vicon_rs.index,name="uwb_raw_filt_disagreement")
    qf1=qf.resample(cfg["dt"]).nearest().iloc[:,0].reindex(vicon_rs.index).interpolate("nearest") if qf is not None and not qf.empty else pd.Series(np.nan,index=vicon_rs.index)
    qf_gate=(~cfg["use_qf_gate"])|(qf1>=cfg["qf_min"])
    pI=imu_aligned[["x","y","z"]].to_numpy(dtype=float); pU=uwb_filt[["x","y","z"]].to_numpy(dtype=float); pF=np.full_like(pU,np.nan)
    axy=float(cfg["alpha_xy"]); az=float(cfg["alpha_z"]); m_all=np.isfinite(pI).all(axis=1)&np.isfinite(pU).all(axis=1); m_q=np.asarray(qf_gate.fillna(False),dtype=bool); m_xy=m_all&m_q
    pF[m_xy,0]=axy*pI[m_xy,0]+(1-axy)*pU[m_xy,0]; pF[m_xy,1]=axy*pI[m_xy,1]+(1-axy)*pU[m_xy,1]; pF[~m_q,0]=pI[~m_q,0]; pF[~m_q,1]=pI[~m_q,1]
    pF[:,2]=pI[:,2] if cfg["force_z_from_imu"] else np.where(m_all&m_q, az*pI[:,2]+(1-az)*pU[:,2], pI[:,2])
    frame=pd.concat([vicon_rs.add_prefix("vicon_"),uwb_raw.add_prefix("uwb_raw_"),uwb_filt.add_prefix("uwb_filt_"),imu_aligned.add_prefix("imu_"),pd.DataFrame(pF,index=vicon_rs.index,columns=["fusion_pre_x","fusion_pre_y","fusion_pre_z"]),qf1.rename("q_uwb"),q_uwb.rename("dtw_q_uwb"),q_imu.rename("dtw_q_imu"),disagreement],axis=1,sort=False)
    frame["qf_gate"]=qf_gate.astype(float); frame["exp_id"]=exp_id; frame["drone_name"]=drone_name
    if cfg.get("enable_ekf",False): frame=run_ekf_on_fusion_pre(frame,cfg)
    return PreparedDrone(exp_id,drone_name,frame)

def parse_exp_number(exp_id):
    digits="".join(ch for ch in exp_id if ch.isdigit()); return int(digits) if digits else -1

def auto_tune_alpha(frames,cfg):
    grid=np.arange(0.0,1.01,0.01); best=cfg["alpha_xy"]; best_rmse=np.inf
    for alpha in grid:
        errs=[]
        for frame in frames:
            exp_num=parse_exp_number(str(frame["exp_id"].iloc[0])); drone_num=int(str(frame["drone_name"].iloc[0]).replace("drone",""))
            if cfg["auto_tune_experiments"] and exp_num not in cfg["auto_tune_experiments"]: continue
            if drone_num not in cfg["auto_tune_drones"]: continue
            pV=frame[["vicon_x","vicon_y","vicon_z"]].to_numpy(dtype=float); pI=frame[["imu_x","imu_y","imu_z"]].to_numpy(dtype=float); pU=frame[["uwb_filt_x","uwb_filt_y","uwb_filt_z"]].to_numpy(dtype=float); qf_gate=frame["qf_gate"].to_numpy(dtype=float)>0.5
            pF=np.full_like(pV,np.nan); m_all=np.isfinite(pI).all(axis=1)&np.isfinite(pU).all(axis=1); m=m_all&qf_gate
            pF[m,0]=alpha*pI[m,0]+(1-alpha)*pU[m,0]; pF[m,1]=alpha*pI[m,1]+(1-alpha)*pU[m,1]; pF[~qf_gate,0]=pI[~qf_gate,0]; pF[~qf_gate,1]=pI[~qf_gate,1]; pF[:,2]=pI[:,2]
            m_eval=np.isfinite(pV).all(axis=1)&np.isfinite(pF).all(axis=1)
            if m_eval.sum()<20: continue
            e=pF[m_eval,:2]-pV[m_eval,:2]; errs.extend(np.sum(e*e,axis=1).tolist())
        if errs:
            rmse=float(np.sqrt(np.mean(errs)))
            if rmse<best_rmse: best_rmse=rmse; best=float(alpha)
    return best

def summarize(frames,cfg):
    rows=[]
    for frame in frames:
        pV=frame[["vicon_x","vicon_y","vicon_z"]].to_numpy(dtype=float); pI=frame[["imu_x","imu_y","imu_z"]].to_numpy(dtype=float); pUr=frame[["uwb_raw_x","uwb_raw_y","uwb_raw_z"]].to_numpy(dtype=float); pUf=frame[["uwb_filt_x","uwb_filt_y","uwb_filt_z"]].to_numpy(dtype=float); pFp=frame[["fusion_pre_x","fusion_pre_y","fusion_pre_z"]].to_numpy(dtype=float)
        row={"exp_id":str(frame["exp_id"].iloc[0]),"drone_name":str(frame["drone_name"].iloc[0]),"imu_rmse":rmse3d(pV,pI),"uwb_raw_rmse":rmse3d(pV,pUr),"uwb_filt_rmse":rmse3d(pV,pUf),"fusion_pre_rmse":rmse3d(pV,pFp),"n_samples":len(frame)}
        if cfg.get("enable_ekf",False) and all(c in frame.columns for c in ["fusion_ekf_x","fusion_ekf_y","fusion_ekf_z"]): row["fusion_ekf_rmse"]=rmse3d(pV,frame[["fusion_ekf_x","fusion_ekf_y","fusion_ekf_z"]].to_numpy(dtype=float))
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(description="Modified v5 that reads prepared table")
    p.add_argument("--prepared-csv",required=True); p.add_argument("--out-dir",default="reports_quality_gated_from_table")
    p.add_argument("--alpha-xy",type=float,default=None); p.add_argument("--use-qf-gate",action="store_true"); p.add_argument("--enable-auto-tune",action="store_true"); p.add_argument("--auto-tune-experiments",nargs="*",type=int,default=None); p.add_argument("--enable-ekf",action="store_true")
    args=p.parse_args()
    cfg=dict(CONFIG)
    if args.alpha_xy is not None: cfg["alpha_xy"]=float(args.alpha_xy)
    if args.use_qf_gate: cfg["use_qf_gate"]=True
    if args.enable_auto_tune: cfg["enable_auto_tune"]=True
    if args.auto_tune_experiments is not None: cfg["auto_tune_experiments"]=list(args.auto_tune_experiments)
    if args.enable_ekf: cfg["enable_ekf"]=True
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.prepared_csv,index_col=0); df.index=pd.to_datetime(df.index)
    frames=[]
    for (exp_id,drone_name), g in df.groupby(["exp_id","drone_name"]):
        try:
            item=preprocess_one_group(g,cfg); frames.append(item.frame); print(f"OK: {exp_id} {drone_name} -> {len(item.frame)} samples")
        except Exception as e:
            print(f"SKIP: {exp_id} {drone_name} -> {type(e).__name__}: {e}")
    if not frames: raise ValueError("No valid groups processed")
    if cfg["enable_auto_tune"]:
        best=auto_tune_alpha(frames,cfg); print(f"AUTO-TUNE: best alpha_xy = {best:.2f}"); cfg["alpha_xy"]=best
        frames2=[]
        for (exp_id,drone_name), g in df.groupby(["exp_id","drone_name"]):
            try: frames2.append(preprocess_one_group(g,cfg).frame)
            except Exception: pass
        frames=frames2
    samples=pd.concat(frames,axis=0,sort=False); summary=summarize(frames,cfg)
    samples.to_csv(out_dir/cfg["out_samples_csv"],index=True); summary.to_csv(out_dir/cfg["out_summary_csv"],index=False)
    print("Created:"); print(out_dir/cfg["out_samples_csv"]); print(out_dir/cfg["out_summary_csv"])
    print(summary.to_string(index=False))

if __name__=="__main__":
    main()
