"""Dual-axis XJTU-SY importer, reproducible simulation and causal alarm evaluation."""
import argparse
import csv
import re
import sqlite3
from pathlib import Path

import numpy as np
from scipy import signal
from signal_core import amplitude_spectrum

FS, N, STEPS, ONSET = 25600, 32768, 720, 361
ROTATION = (35, 37.5, 40)
CASES = ("dual", "horizontal", "transient", "healthy")
NAMES = ("rms_h", "rms_v", "peak_h", "peak_v", "kurt_h", "kurt_v")
LENGTHS = ((123,161,158,122,52),(491,161,533,42,339),(2538,2496,371,1515,114))
T = np.arange(N) / FS
BASE = [np.column_stack((0.15*np.sin(2*np.pi*f*T), 0.10*np.sin(2*np.pi*f*T+0.4))) for f in ROTATION]
# Fixed outer-race impact sequence exciting a damped 3.2 kHz structural mode.
RING = np.sin(2*np.pi*3200*T[:160]) * np.exp(-T[:160]*1200)
CARRIER = []
for f in ROTATION:
    impulses = np.zeros(N)
    bpfo = 4*f*(1-7.92/34.55)
    impulses[np.rint(np.arange(0, N-1, FS/bpfo)).astype(int)] = 1
    CARRIER.append(np.convolve(impulses, RING, mode="full")[:N])

def simulated(c, case, minute):
    idx = CASES.index(case)
    rng = np.random.default_rng(np.random.SeedSequence([20261124, c, idx, minute]))
    x = BASE[c] + rng.normal(0, 0.12, (N, 2))
    strength = max(0, minute-ONSET+1) * 0.015
    if case in ("dual", "horizontal"):
        x[:,0] += strength*CARRIER[c]
        if case == "dual":
            x[:,1] += 0.65*strength*CARRIER[c]
    elif case == "transient" and minute in (180, 360, 540):
        x += 3*CARRIER[c][:,None]
    return x

def waveform(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        first = stream.readline()
        header = [v.strip().lower().replace(" ", "_") for v in first.split(",")]
        try:
            [float(v) for v in header]
        except ValueError:
            if len(header) != 2 or not header[0].startswith("horizontal") or not header[1].startswith("vertical"):
                raise ValueError("表头应依次为 horizontal、vertical 振动通道")
        else:
            stream.seek(0)
        x = np.loadtxt(stream, delimiter=",")
    if x.shape != (N,2) or not np.isfinite(x).all():
        raise ValueError("每个 CSV 须为 32768 行、水平与垂直两列有限数值")
    return x

def features(x):
    d = x - x.mean(axis=0)
    power = (d*d).mean(axis=0)
    rms = np.sqrt(power)
    peaks = np.max(np.abs(d), axis=0)
    kurt = np.divide(np.mean((d*d)**2,axis=0), power**2,
                     out=np.full(2,np.nan), where=power>0)
    return tuple(map(float, np.r_[rms, peaks, kurt]))

def collect(output, folder=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if folder:
        root = Path(folder).resolve()
        directories = sorted(d for d in [root, *root.rglob("Bearing*")] if d.is_dir() and re.fullmatch(r"Bearing[1-3]_[1-5]", d.name,re.I))
        if not directories or len({d.name.lower() for d in directories}) != len(directories):
            raise ValueError("请选择含有唯一轴承编号的 XJTU-SY 数据目录")
        source = str(root)
    else:
        source = "sim-v1"
    db = sqlite3.connect(output/"monitor.sqlite")
    try:
        db.execute("CREATE TABLE IF NOT EXISTS metadata(source TEXT UNIQUE)")
        old = db.execute("SELECT source FROM metadata").fetchall()
        if old and old != [(source,)]:
            raise ValueError("此输出目录属于另一数据源，请选择独立目录")
        db.execute("INSERT OR IGNORE INTO metadata VALUES (?)",(source,))
        db.execute("CREATE TABLE IF NOT EXISTS features (bearing TEXT, minute INTEGER, signature TEXT, rms_h REAL, rms_v REAL, peak_h REAL, peak_v REAL, kurt_h REAL, kurt_v REAL, PRIMARY KEY(bearing,minute))")
        db.commit()
        if folder:
            groups = []
            for d in directories:
                paths = {int(p.stem):p for p in d.glob("*.csv") if p.stem.isdigit()}
                if not paths or any(p.name != str(i)+".csv" for i,p in paths.items()) or len(paths) != sum(p.stem.isdigit() for p in d.glob("*.csv")):
                    raise ValueError("文件名应为唯一的正整数序号，例如 1.csv："+str(d))
                if min(paths)<1:
                    raise ValueError("分钟序号须从正整数开始")
                groups.append((d.name,sorted(paths),paths))
        else:
            groups = [(f"S{c+1}_{case}",range(1,STEPS+1),{}) for c in range(3) for case in CASES]
        all_keys = set()
        for name, minutes, paths in groups:
            existing = dict(db.execute("SELECT minute,signature FROM features WHERE bearing=?",(name,)))
            for minute in minutes:
                all_keys.add((name,minute))
                if folder:
                    p = paths[minute]
                    st = p.stat()
                    sig = str(st.st_size)+"/"+str(st.st_mtime_ns)+"/"+str(p.relative_to(root))
                else:
                    sig = source
                if existing.get(minute) == sig:
                    continue
                x = waveform(p) if folder else simulated(int(name[1])-1, name.split("_",1)[1], minute)
                with_error = (name,minute,sig,*features(x))
                db.execute("INSERT OR REPLACE INTO features VALUES (?,?,?,?,?,?,?,?,?)",with_error)
                if minute%60 == 0:
                    db.commit()
            db.commit()
        stale = set(db.execute("SELECT bearing,minute FROM features")) - all_keys
        db.executemany("DELETE FROM features WHERE bearing=? AND minute=?",stale)
        db.commit()
        return export(db,output,folder is None)
    finally:
        db.close()

def alarms(minutes, rms, threshold_k=6, run=3):
    """Prefix-only calibration; a missing minute breaks persistence."""
    if len(minutes)<20 or list(minutes[:20]) != list(range(1,21)):
        return np.full(2,np.nan), np.zeros((len(minutes),2),bool), {}
    base = rms[:20]
    median = np.median(base,axis=0)
    scale = 1.4826*np.median(abs(base-median),axis=0)
    lim = median + np.maximum(threshold_k*scale, np.maximum(0.1*median, 1e-9))
    flags = rms>lim
    first = {}
    policies = (flags[:,0],flags[:,1],flags.any(axis=1),flags.all(axis=1))
    for key, trace in zip(("H","V","OR","AND"),policies):
        streak = 0
        for i in range(20,len(trace)):
            if minutes[i] != minutes[i-1]+1:
                streak = 0
            streak = streak+1 if trace[i] else 0
            if streak >= run:
                first[key] = int(minutes[i])
                break
    return lim, flags, first

def export(db, out, simulation):
    rows = db.execute("SELECT bearing,minute,rms_h,rms_v,peak_h,peak_v,kurt_h,kurt_v FROM features ORDER BY bearing,minute").fetchall()
    import itertools
    tables = {"features.csv":[("bearing","minute",*NAMES,"threshold_h","threshold_v","over_h","over_v")],
              "alarms.csv":[("bearing","n_records","expected","missing_minutes","baseline_ok","H","V","OR","AND","injected_onset")],
              "sensitivity.csv":[("bearing","k","persistence","OR","AND")]}
    for name, group in itertools.groupby(rows,key=lambda row:row[0]):
        chunk = list(group)
        mins = np.array([r[1] for r in chunk])
        rms = np.array([r[2:4] for r in chunk])
        lim, flags, alerts = alarms(mins,rms)
        for r,flag in zip(chunk,flags):
            tables["features.csv"].append((*r,*lim,*map(int,flag)))
        expected = STEPS if simulation else LENGTHS[int(name[-3])-1][int(name[-1])-1]
        missing = sorted(set(range(1,expected+1))-set(mins))
        tables["alarms.csv"].append((name,len(mins),expected," ".join(map(str,missing)),int(np.isfinite(lim).all()),
                                      *(alerts.get(k,"") for k in ("H","V","OR","AND")),
                                      ONSET if simulation and name.split("_")[1] in ("dual","horizontal") else ""))
        for k in (3,6,9):
            for run in (1,3,5):
                a = alarms(mins,rms,k,run)[2]
                tables["sensitivity.csv"].append((name,k,run,a.get("OR",""),a.get("AND","")))
    for name, table in tables.items():
        tmp=out/(name+".new")
        with tmp.open("w",newline="",encoding="utf-8-sig") as f:
            csv.writer(f).writerows(table)
        tmp.replace(out/name)
    return len(rows)

def diagnostic(x):
    """Return spectral products for an individual snapshot, channels along axis 1."""
    d=x-x.mean(axis=0)
    fft = [amplitude_spectrum(v,FS)[1] for v in d.T]
    f = np.fft.rfftfreq(len(d),1/FS)
    fw, psd=signal.welch(d, fs=FS, axis=0,nperseg=2048,noverlap=1024)
    sos=signal.butter(4,[2000,5000],fs=FS,btype="bandpass",output="sos")
    env=np.abs(signal.hilbert(signal.sosfiltfilt(sos,d,axis=0),axis=0))
    env-=env.mean(axis=0)
    es=np.array([amplitude_spectrum(v,FS)[1] for v in env.T]).T
    ft,ts,z=signal.stft(d[:,0],fs=FS,nperseg=512,noverlap=384,boundary=None,padded=False)
    return f,np.array(fft).T,fw,psd,es,ft,ts,np.abs(z)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="解压后的 XJTU-SY 根目录")
    source.add_argument("--simulation", action="store_true")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    print("Records:",collect(args.out,args.data))

