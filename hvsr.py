#!/usr/bin/env python3
"""
HVSR Mikrotremor — pengganti modul H/V Geopsy (Python, jalan di Mac M-series).

Alur (meniru Geopsy H/V):
  1. Baca file .mseed (3 komponen Z, N, E) per titik/stasiun.
  2. Detrend, potong menjadi window (default 40 s), buang window transien
     dengan anti-trigger STA/LTA, buang window yang mengandung gap.
  3. FFT tiap window (taper cosine 5%), spektrum dihaluskan Konno-Ohmachi (b=40).
  4. H = rata-rata kuadrat sqrt((E^2+N^2)/2), H/V per window.
  5. Kurva rata-rata lognormal (garis hitam) + standar deviasi (garis putus-putus),
     f0 per window -> pita abu-abu f0 ± std.
  6. f0 (frekuensi dominan) dan A0 (amplifikasi) + uji kriteria SESAME 2004.
  7. Ekspor JPEG, CSV kurva, dan laporan teks.

Contoh pemakaian:
  python hvsr.py DATA/CLG07                 # satu titik (folder berisi 3 file mseed)
  python hvsr.py DATA/TDOM                  # semua subfolder titik sekaligus
  python hvsr.py Z.mseed N.mseed E.mseed    # file satu per satu
  python hvsr.py DATA/TDOM --win 30 --fmin 0.5 --fmax 15 --reject 2
  python hvsr.py   (tanpa argumen) / --gui  # jendela GUI (pilih folder)
  python hvsr.py DATA/CLG07 --manual        # picking window manual seperti Geopsy
"""
import argparse
import glob
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib

GUI_MODE = len(sys.argv) == 1 or any(x in sys.argv for x in ("--gui", "--manual", "--show"))
matplotlib.use("TkAgg" if GUI_MODE else "Agg")
import matplotlib.pyplot as plt
from obspy import read, Stream

MSEED_EXT = (".mseed", ".miniseed", ".msd", ".seed", ".sac")


# ----------------------------------------------------------------------------
# Membaca data
# ----------------------------------------------------------------------------
def _comp(cha):
    c = cha[-1].upper()
    return {"1": "N", "2": "E"}.get(c, c)


def load_three_components(paths):
    """Baca file, gabungkan fragmen, kembalikan dict {'Z','N','E'} -> Trace (gap = masked)."""
    st = Stream()
    for p in paths:
        try:
            st += read(p)
        except Exception as e:  # file bukan mseed
            print(f"  [lewati] {os.path.basename(p)}: {e}")
    if len(st) == 0:
        raise ValueError("tidak ada data seismik yang terbaca")

    # gabung per id channel (gap dibiarkan masked agar window bergap dibuang)
    merged = {}
    for tid in sorted({tr.id for tr in st}):
        sub = st.select(id=tid).copy()
        sub.merge(method=1)  # overlap identik digabung, gap -> masked array
        tr = sub[0]
        comp = _comp(tr.stats.channel)
        if comp not in ("Z", "N", "E"):
            continue
        valid = np.count_nonzero(~np.ma.getmaskarray(tr.data))
        # jika ada >1 versi komponen yang sama (mis. SHZ & BHZ), pakai yang paling lengkap
        if comp not in merged or valid > merged[comp][1]:
            merged[comp] = (tr, valid)

    missing = {"Z", "N", "E"} - set(merged)
    if missing:
        raise ValueError(f"komponen tidak lengkap, kurang: {', '.join(sorted(missing))}")

    trs = {c: merged[c][0] for c in "ZNE"}
    sr = {tr.stats.sampling_rate for tr in trs.values()}
    if len(sr) > 1:
        target = min(sr)
        for c, tr in trs.items():
            if tr.stats.sampling_rate != target:
                tr.data = np.ma.filled(tr.data.astype(float), 0.0)
                tr.resample(target)
    t0 = max(tr.stats.starttime for tr in trs.values())
    t1 = min(tr.stats.endtime for tr in trs.values())
    if t1 <= t0:
        raise ValueError("ketiga komponen tidak tumpang tindih dalam waktu")
    out = {}
    for c, tr in trs.items():
        tr = tr.copy().trim(t0, t1, nearest_sample=True)
        out[c] = tr
    n = min(len(tr.data) for tr in out.values())
    for tr in out.values():
        tr.data = tr.data[:n]
    return out


# ----------------------------------------------------------------------------
# Pengolahan sinyal
# ----------------------------------------------------------------------------
def konno_ohmachi_matrix(f_raw, f_out, b=40.0):
    """Matriks bobot Konno-Ohmachi (len(f_out) x len(f_raw))."""
    with np.errstate(divide="ignore", invalid="ignore"):
        x = b * np.log10(f_raw[None, :] / f_out[:, None])
        w = (np.sin(x) / x) ** 4
    w[np.isnan(w)] = 1.0           # x = 0
    w[:, f_raw <= 0] = 0.0
    w[np.abs(np.nan_to_num(x, nan=99.0, posinf=99.0, neginf=99.0)) > np.pi] = 0.0  # lobus utama saja
    s = w.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return w / s


def prepare_signals(trs, sta, lta, smin, smax, use_antitrigger=True):
    """Kembalikan data terisi, mask bebas-gap, mask lolos anti-trigger, sampling rate."""
    sr = trs["Z"].stats.sampling_rate
    n = len(trs["Z"].data)
    gap_ok = np.ones(n, bool)
    filled = {}
    for c, tr in trs.items():
        mask = np.ma.getmaskarray(tr.data)
        gap_ok &= ~mask
        d = np.ma.filled(tr.data.astype(float), np.nan)
        # isi gap sementara (hanya untuk STA/LTA & tampilan), window bergap tetap dibuang
        if np.isnan(d).any():
            idx = np.arange(n)
            ok = ~np.isnan(d)
            d = np.interp(idx, idx[ok], d[ok])
        filled[c] = d - np.mean(d)

    trig_ok = np.ones(n, bool)
    if use_antitrigger:
        nsta, nlta = int(sta * sr), int(lta * sr)
        if n > nlta * 2:
            idx = np.arange(1, n + 1)
            for c in "ZNE":
                # STA/LTA pada amplitudo absolut (seperti Geopsy), rata-rata bergerak ke belakang
                cs = np.concatenate([[0.0], np.cumsum(np.abs(filled[c]))])
                s_sta = (cs[idx] - cs[np.maximum(idx - nsta, 0)]) / np.minimum(idx, nsta)
                s_lta = (cs[idx] - cs[np.maximum(idx - nlta, 0)]) / np.minimum(idx, nlta)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(s_lta > 0, s_sta / s_lta, 1.0)
                ratio[:nlta] = 1.0  # periode awal LTA belum stabil
                trig_ok &= (ratio >= smin) & (ratio <= smax)
    return filled, gap_ok, trig_ok, sr


def sliding_starts(good, nwin, step):
    """Window otomatis: geser, lompati bagian yang tidak lolos."""
    n, starts, i = len(good), [], 0
    while i + nwin <= n:
        if good[i:i + nwin].all():
            starts.append(i)
            i += step
        else:
            i += int(np.where(~good[i:i + nwin])[0][-1]) + 1
    return starts


def grid_windows(gap_ok, trig_ok, nwin, step):
    """Window pada grid tetap (untuk picking manual): start, bebas-gap, lolos-auto."""
    n = len(gap_ok)
    starts = np.arange(0, n - nwin + 1, step)
    cg = np.concatenate([[0], np.cumsum(~gap_ok)])
    ct = np.concatenate([[0], np.cumsum(~trig_ok)])
    valid = (cg[starts + nwin] - cg[starts]) == 0
    auto = valid & ((ct[starts + nwin] - ct[starts]) == 0)
    return starts, valid, auto


def window_hv(data, starts, nwin, sr, f_out, ko_b=40.0, horizontal="squared", taper=0.05):
    from scipy.signal.windows import tukey
    from scipy.signal import detrend
    f_raw = np.fft.rfftfreq(nwin, 1.0 / sr)
    W = konno_ohmachi_matrix(f_raw, f_out, ko_b)
    tp = tukey(nwin, taper * 2)
    hv = np.full((len(starts), len(f_out)), np.nan)
    for k, s in enumerate(starts):
        spec = {}
        for c in "ZNE":
            x = detrend(data[c][s:s + nwin]) * tp
            spec[c] = W @ np.abs(np.fft.rfft(x))
        if horizontal == "geometric":
            H = np.sqrt(spec["N"] * spec["E"])
        elif horizontal == "arithmetic":
            H = (spec["N"] + spec["E"]) / 2
        else:  # squared average (default Geopsy)
            H = np.sqrt((spec["N"] ** 2 + spec["E"] ** 2) / 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            hv[k] = H / spec["Z"]
    hv[~np.isfinite(hv)] = np.nan
    return hv


def freq_axis(fmin, fmax, sr, nfreq=300):
    fmax = min(fmax, 0.45 * sr)
    return np.logspace(np.log10(fmin), np.log10(fmax), nfreq)


def hv_stats(f_out, hv, keep, win_s, sr, peak_fmin=None, peak_fmax=None, reject_n=0.0, times=None):
    """Statistik lognormal, f0, A0 dari window yang dipilih (keep)."""
    keep = np.asarray(keep, bool).copy()
    if keep.sum() == 0:
        raise ValueError("tidak ada window yang dipilih")
    pmin = peak_fmin if peak_fmin else f_out[0]
    pmax = peak_fmax if peak_fmax else f_out[-1]
    band = (f_out >= pmin) & (f_out <= pmax)
    if not band.any():
        raise ValueError("rentang pencarian f0 kosong")

    def win_peaks(h):
        hb = np.where(band[None, :], np.nan_to_num(h, nan=-np.inf), -np.inf)
        return f_out[np.argmax(hb, axis=1)]

    if reject_n and reject_n > 0:  # penolakan window domain frekuensi (Cox dkk., 2020, disederhanakan)
        for _ in range(50):
            lf = np.log(win_peaks(hv[keep]))
            mu, sd = lf.mean(), (lf.std(ddof=1) if keep.sum() > 1 else 0)
            allf = np.log(win_peaks(hv))
            new = keep & (np.abs(allf - mu) <= reject_n * sd + 1e-12)
            if new.sum() < 3 or (new == keep).all():
                break
            keep = new

    hvk = hv[keep]
    lh = np.log(hvk)
    mean = np.exp(np.nanmean(lh, axis=0))
    sig = np.nanstd(lh, axis=0, ddof=1) if len(hvk) > 1 else np.zeros_like(mean)
    std_factor = np.exp(sig)

    i0 = np.where(band)[0][np.nanargmax(mean[band])]
    f0, A0 = f_out[i0], mean[i0]

    fw = win_peaks(hvk)
    f0_win = np.exp(np.mean(np.log(fw)))
    f0_win_sd = np.exp(np.std(np.log(fw), ddof=1)) if len(fw) > 1 else 1.0
    sigma_f = np.std(fw, ddof=1) if len(fw) > 1 else 0.0

    return dict(freq=f_out, hv=hv, keep=keep, times=times, mean=mean, std_factor=std_factor,
                f0=f0, A0=A0, i0=i0, f0_win=f0_win, f0_win_sd=f0_win_sd, sigma_f=sigma_f,
                fw=fw, win_s=win_s, n_total=len(hv), n_used=int(keep.sum()),
                band=(pmin, pmax), sr=sr)


def compute_hvsr(trs, win_s=40.0, overlap=0.0, ko_b=40.0, fmin=0.5, fmax=20.0,
                 nfreq=300, sta=1.0, lta=30.0, smin=0.2, smax=2.5,
                 antitrigger=True, horizontal="squared", peak_fmin=None,
                 peak_fmax=None, reject_n=0.0, taper=0.05):
    data, gap_ok, trig_ok, sr = prepare_signals(trs, sta, lta, smin, smax, antitrigger)
    nwin = int(round(win_s * sr))
    step = max(1, int(round(nwin * (1 - overlap))))
    starts = sliding_starts(gap_ok & trig_ok, nwin, step)
    if len(starts) == 0:
        raise ValueError("tidak ada window yang lolos (coba --win lebih kecil atau --no-antitrigger)")
    f_out = freq_axis(fmin, fmax, sr, nfreq)
    hv = window_hv(data, starts, nwin, sr, f_out, ko_b, horizontal, taper)
    return hv_stats(f_out, hv, np.ones(len(hv), bool), win_s, sr, peak_fmin, peak_fmax,
                    reject_n, times=np.array(starts) / sr)


# ----------------------------------------------------------------------------
# Kriteria SESAME (2004)
# ----------------------------------------------------------------------------
def sesame(r):
    f, m, sf = r["freq"], r["mean"], r["std_factor"]
    f0, A0, lw, nw = r["f0"], r["A0"], r["win_s"], r["n_used"]
    if f0 < 0.2:
        eps, theta = 0.25 * f0, 3.0
    elif f0 < 0.5:
        eps, theta = 0.20 * f0, 2.5
    elif f0 < 1.0:
        eps, theta = 0.15 * f0, 2.0
    elif f0 < 2.0:
        eps, theta = 0.10 * f0, 1.78
    else:
        eps, theta = 0.05 * f0, 1.58

    rel = []
    rel.append(("f0 > 10/Lw", f0 > 10 / lw, f"{f0:.3f} > {10/lw:.3f}"))
    nc = lw * nw * f0
    rel.append(("nc = Lw·nw·f0 > 200", nc > 200, f"{nc:.0f} > 200"))
    rng = (f >= 0.5 * f0) & (f <= 2 * f0)
    lim = 2.0 if f0 > 0.5 else 3.0
    smax = sf[rng].max() if rng.any() else np.nan
    rel.append((f"σA(f) < {lim:g} untuk 0.5f0–2f0", bool(smax < lim), f"maks {smax:.2f}"))

    clr = []
    lo = (f >= f0 / 4) & (f <= f0)
    hi = (f >= f0) & (f <= 4 * f0)
    fm = f[lo][m[lo] < A0 / 2]
    fp = f[hi][m[hi] < A0 / 2]
    clr.append(("∃ f⁻ ∈ [f0/4, f0]: A < A0/2", fm.size > 0, f"f⁻={fm[-1]:.3f} Hz" if fm.size else "tidak ada"))
    clr.append(("∃ f⁺ ∈ [f0, 4f0]: A < A0/2", fp.size > 0, f"f⁺={fp[0]:.3f} Hz" if fp.size else "tidak ada"))
    clr.append(("A0 > 2", A0 > 2, f"{A0:.2f} > 2"))
    b = (f >= r["band"][0]) & (f <= r["band"][1])
    fu = f[b][np.argmax((m * sf)[b])]
    fl = f[b][np.argmax((m / sf)[b])]
    ok4 = abs(fu - f0) <= 0.05 * f0 and abs(fl - f0) <= 0.05 * f0
    clr.append(("Puncak A(f)±σA dalam f0 ± 5%", ok4, f"{fl:.3f} / {fu:.3f} Hz"))
    clr.append((f"σf < ε(f0) = {eps:.3f}", r["sigma_f"] < eps, f"σf={r['sigma_f']:.3f}"))
    sA0 = sf[r["i0"]]
    clr.append((f"σA(f0) < θ(f0) = {theta:g}", sA0 < theta, f"{sA0:.3f}"))
    return rel, clr


# ----------------------------------------------------------------------------
# Plot & ekspor
# ----------------------------------------------------------------------------
def window_color(k, n):
    return plt.get_cmap("hsv")(k / max(1, n))


def draw_hvsr(ax, r, title, lw=0.8):
    """Gambar kurva H/V gaya Geopsy pada axes yang diberikan."""
    from matplotlib.ticker import ScalarFormatter
    f, m, sf = r["freq"], r["mean"], r["std_factor"]
    # pita f0 ± std (window) seperti Geopsy
    lo, hi = r["f0_win"] / r["f0_win_sd"], r["f0_win"] * r["f0_win_sd"]
    ax.axvspan(lo, hi, color="0.75", alpha=0.8, lw=0, zorder=0)
    ax.axvspan(r["f0_win"] / 1.02, r["f0_win"] * 1.02, color="0.55", alpha=0.8, lw=0, zorder=0)
    if r["band"][0] > f[0] * 1.001 or r["band"][1] < f[-1] * 0.999:  # rentang cari f0 manual
        ax.axvline(r["band"][0], color="tab:blue", ls=":", lw=1.2)
        ax.axvline(r["band"][1], color="tab:blue", ls=":", lw=1.2)
    n = r["n_total"]
    for k in np.where(r["keep"])[0]:
        ax.plot(f, r["hv"][k], color=window_color(k, n), lw=lw, alpha=0.85, zorder=1)
    ax.plot(f, m * sf, "k--", lw=2, zorder=3)
    ax.plot(f, m / sf, "k--", lw=2, zorder=3)
    ax.plot(f, m, "k-", lw=2.6, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(f[0], f[-1])
    ax.set_ylim(0, max(np.nanmax(m * sf) * 1.35, 2))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("H/V")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_title(title, fontsize=12, fontweight="bold")
    txt = (f"f0 = {r['f0']:.3f} Hz\nA0 = {r['A0']:.3f}\n"
           f"f0 (window) = {r['f0_win']:.3f} ×/÷ {r['f0_win_sd']:.3f}\n"
           f"Window: {r['n_used']}/{r['n_total']} × {r['win_s']:g} s")
    ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=9,
            family="monospace", bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9))


def plot_hvsr(r, title, out_jpg=None, show=False, dpi=200):
    from matplotlib.figure import Figure
    fig = plt.figure(figsize=(8, 6)) if show else Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    draw_hvsr(ax, r, title)
    fig.tight_layout()
    if out_jpg:
        fig.savefig(out_jpg, dpi=dpi, format="jpeg", pil_kwargs={"quality": 95})
    if show:
        plt.show()
        plt.close(fig)


def write_outputs(r, name, outdir, tstart, mode="otomatis (STA/LTA)"):
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, name)
    f, m, sf = r["freq"], r["mean"], r["std_factor"]
    np.savetxt(base + "_HVSR.csv", np.column_stack([f, m, m / sf, m * sf, sf]), delimiter=",",
               fmt="%.6f", header="frequency_Hz,HV_mean,HV_min(mean/std),HV_max(mean*std),std_factor",
               comments="")
    rel, clr = sesame(r)
    nrel, nclr = sum(c[1] for c in rel), sum(c[1] for c in clr)
    lines = [
        f"HASIL ANALISIS HVSR — {name}",
        "=" * 60,
        f"Awal rekaman        : {tstart}",
        f"Sampling rate       : {r['sr']:g} Hz",
        f"Seleksi window      : {mode}",
        f"Panjang window      : {r['win_s']:g} s",
        f"Window dipakai      : {r['n_used']} dari {r['n_total']}",
        f"Rentang cari puncak : {r['band'][0]:.2f} – {r['band'][1]:.2f} Hz",
        "",
        f"Frekuensi dominan f0 : {r['f0']:.4f} Hz   (periode T0 = {1/r['f0']:.4f} s)",
        f"Amplifikasi A0       : {r['A0']:.4f}",
        f"f0 dari window       : {r['f0_win']:.4f} Hz  (std faktor {r['f0_win_sd']:.4f}, σf = {r['sigma_f']:.4f} Hz)",
        f"Indeks kerentanan Kg : {r['A0']**2 / r['f0']:.4f}   (Kg = A0²/f0, Nakamura)",
        "",
        "KRITERIA SESAME (2004)",
        f"Reliabilitas kurva ({nrel}/3 terpenuhi; harus 3/3):",
    ]
    lines += [f"  [{'OK' if ok else '--'}] {n:<38} {v}" for n, ok, v in rel]
    lines.append(f"Puncak jelas ({nclr}/6 terpenuhi; minimal 5/6):")
    lines += [f"  [{'OK' if ok else '--'}] {n:<38} {v}" for n, ok, v in clr]
    lines += ["", f"Kesimpulan: kurva {'RELIABEL' if nrel == 3 else 'TIDAK reliabel'}, "
                  f"puncak {'JELAS' if nclr >= 5 else 'TIDAK jelas'}."]
    with open(base + "_hasil.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    if r.get("times") is not None:  # daftar window agar seleksi bisa ditelusuri
        with open(base + "_window.csv", "w") as fh:
            fh.write("no,mulai_s,selesai_s,dipakai,f0_window_Hz\n")
            fpk = r["freq"][np.argmax(np.nan_to_num(r["hv"], nan=-1), axis=1)]
            for k, t in enumerate(r["times"]):
                fh.write(f"{k+1},{t:.2f},{t + r['win_s']:.2f},{int(r['keep'][k])},{fpk[k]:.4f}\n")
    return nrel, nclr


# ----------------------------------------------------------------------------
# Pencarian titik
# ----------------------------------------------------------------------------
def find_stations(inputs):
    """Kembalikan list (nama, [file]) — satu entri per folder yang berisi file mseed."""
    files = [p for p in inputs if os.path.isfile(p)]
    dirs = [p for p in inputs if os.path.isdir(p)]
    out = []
    if files:
        name = os.path.basename(os.path.dirname(os.path.abspath(files[0]))) or "titik"
        out.append((name, files))
    for d in dirs:
        for root, _, fs in sorted(os.walk(d)):
            ms = sorted(os.path.join(root, f) for f in fs
                        if f.lower().endswith(MSEED_EXT) and not f.startswith("."))
            if ms:
                out.append((os.path.basename(os.path.abspath(root)), ms))
    return out


def process(name, files, args, outdir, show=False):
    print(f"\n▶ {name}: {len(files)} file")
    trs = load_three_components(files)
    r = compute_hvsr(trs, win_s=args.win, overlap=args.overlap / 100, ko_b=args.ko,
                     fmin=args.plot_fmin, fmax=args.plot_fmax, sta=args.sta, lta=args.lta,
                     smin=args.min_ratio, smax=args.max_ratio, antitrigger=not args.no_antitrigger,
                     horizontal=args.horizontal, peak_fmin=args.fmin, peak_fmax=args.fmax,
                     reject_n=args.reject)
    jpg = os.path.join(outdir, f"{name}_HVSR.jpg")
    os.makedirs(outdir, exist_ok=True)
    plot_hvsr(r, name, jpg, show=show, dpi=args.dpi)
    nrel, nclr = write_outputs(r, name, outdir, trs["Z"].stats.starttime)
    print(f"  f0 = {r['f0']:.3f} Hz | A0 = {r['A0']:.3f} | window {r['n_used']}/{r['n_total']} "
          f"| SESAME reliabel {nrel}/3, puncak {nclr}/6")
    print(f"  → {jpg}")
    return dict(name=name, f0=r["f0"], A0=r["A0"], f0_win=r["f0_win"], T0=1 / r["f0"],
                Kg=r["A0"] ** 2 / r["f0"], n=r["n_used"], rel=nrel, clr=nclr)


def build_parser():
    p = argparse.ArgumentParser(description="Analisis HVSR mikrotremor dari file MiniSEED (pengganti Geopsy H/V).")
    p.add_argument("input", nargs="*", help="folder titik, folder berisi banyak titik, atau 3 file mseed")
    p.add_argument("-o", "--out", default="hasil_hvsr", help="folder output (default: hasil_hvsr)")
    p.add_argument("--win", type=float, default=40, help="panjang window detik (default 40)")
    p.add_argument("--overlap", type=float, default=0, help="overlap window %% (default 0)")
    p.add_argument("--ko", type=float, default=40, help="konstanta Konno-Ohmachi b (default 40)")
    p.add_argument("--fmin", type=float, default=None, help="batas bawah pencarian puncak f0 (Hz)")
    p.add_argument("--fmax", type=float, default=None, help="batas atas pencarian puncak f0 (Hz)")
    p.add_argument("--plot-fmin", type=float, default=0.5, help="frekuensi minimum kurva (default 0.5)")
    p.add_argument("--plot-fmax", type=float, default=20, help="frekuensi maksimum kurva (default 20)")
    p.add_argument("--sta", type=float, default=1.0, help="STA detik (default 1)")
    p.add_argument("--lta", type=float, default=30.0, help="LTA detik (default 30)")
    p.add_argument("--min-ratio", type=float, default=0.2, help="STA/LTA minimum (default 0.2)")
    p.add_argument("--max-ratio", type=float, default=2.5, help="STA/LTA maksimum (default 2.5)")
    p.add_argument("--no-antitrigger", action="store_true", help="matikan seleksi STA/LTA")
    p.add_argument("--horizontal", choices=["squared", "geometric", "arithmetic"], default="squared",
                   help="cara menggabung N & E (default squared = Geopsy)")
    p.add_argument("--reject", type=float, default=0, metavar="N",
                   help="buang window yang f0-nya di luar N std (mis. 2). 0 = mati")
    p.add_argument("--dpi", type=int, default=200, help="resolusi JPEG (default 200)")
    p.add_argument("--show", action="store_true", help="tampilkan plot di layar juga")
    p.add_argument("--gui", action="store_true", help="buka jendela GUI")
    p.add_argument("--manual", action="store_true", help="picking window manual (jendela interaktif)")
    return p


def run(args):
    stations = find_stations(args.input)
    if not stations:
        print("Tidak ditemukan file mseed.")
        return []
    results = []
    for name, files in stations:
        try:
            results.append(process(name, files, args, args.out, show=args.show))
        except Exception as e:
            print(f"  [GAGAL] {name}: {e}")
    if len(results) > 1:
        path = os.path.join(args.out, "ringkasan_HVSR.csv")
        with open(path, "w") as fh:
            fh.write("titik,f0_Hz,A0,T0_s,Kg,f0_window_Hz,n_window,SESAME_reliabel(/3),SESAME_puncak(/6)\n")
            for r in results:
                fh.write(f"{r['name']},{r['f0']:.4f},{r['A0']:.4f},{r['T0']:.4f},{r['Kg']:.4f},"
                         f"{r['f0_win']:.4f},{r['n']},{r['rel']},{r['clr']}\n")
        print(f"\nRingkasan semua titik → {path}")
    return results


# ----------------------------------------------------------------------------
# Picking manual (seperti Geopsy: pilih/buang window di sinyal & kurva H/V)
# ----------------------------------------------------------------------------
PICK_HELP = (
    "SINYAL: klik = pilih/buang 1 window  •  seret = tambah/buang banyak window (sesuai mode)\n"
    "H/V: klik pada kurva = buang window itu  •  seret kiri-kanan = batasi rentang pencarian f0\n"
    "Kaca pembesar/geser di toolbar untuk zoom (matikan lagi sebelum picking)  •  Ctrl/Cmd+Z = undo"
)


class ManualPicker:
    def __init__(self, master, name, trs, args, outdir, on_saved=None):
        import tkinter as tk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        self.name, self.args, self.outdir, self.on_saved = name, args, outdir, on_saved
        self.tstart = trs["Z"].stats.starttime
        self.data, gap_ok, trig_ok, self.sr = prepare_signals(
            trs, args.sta, args.lta, args.min_ratio, args.max_ratio, True)
        self.nwin = int(round(args.win * self.sr))
        step = max(1, int(round(self.nwin * (1 - args.overlap / 100))))
        self.starts, self.valid, self.auto = grid_windows(gap_ok, trig_ok, self.nwin, step)
        if not self.valid.any():
            raise ValueError("tidak ada window bebas gap — perkecil panjang window")
        self.f = freq_axis(args.plot_fmin, args.plot_fmax, self.sr)
        self.hv = np.full((len(self.starts), len(self.f)), np.nan)
        iv = np.where(self.valid)[0]
        self.hv[iv] = window_hv(self.data, self.starts[iv], self.nwin, self.sr, self.f,
                                args.ko, args.horizontal)
        self.keep = self.auto.copy() if (self.auto.any() and not args.no_antitrigger) else self.valid.copy()
        self.band = [args.fmin, args.fmax]
        self.history = []
        self.r = None
        self.drag = None
        self.n = len(self.starts)
        self.t_mid = (self.starts + self.nwin / 2) / self.sr

        # --- jendela ---
        self.top = tk.Toplevel(master)
        self.top.title(f"Picking manual HVSR — {name}")
        self.top.geometry("1400x860")
        bar = tk.Frame(self.top)
        bar.pack(side="top", fill="x", padx=6, pady=4)
        self.mode = tk.StringVar(value="buang")
        tk.Label(bar, text="Seret di sinyal:").pack(side="left")
        tk.Radiobutton(bar, text="Buang", variable=self.mode, value="buang").pack(side="left")
        tk.Radiobutton(bar, text="Tambah", variable=self.mode, value="tambah").pack(side="left")
        for txt, cmd in [("Auto STA/LTA", self.sel_auto), ("Pilih semua", self.sel_all),
                         ("Kosongkan", self.sel_none), ("Reset rentang f0", self.reset_band),
                         ("Undo", self.undo), ("💾 Simpan hasil", self.save)]:
            tk.Button(bar, text=txt, command=cmd).pack(side="left", padx=3)
        self.status = tk.Label(self.top, anchor="w", justify="left", font=("Menlo", 11))
        self.status.pack(side="bottom", fill="x", padx=6)
        tk.Label(self.top, text=PICK_HELP, anchor="w", justify="left", fg="#555").pack(side="bottom", fill="x", padx=6)

        self.fig = Figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(3, 2, width_ratios=[1.35, 1], hspace=0.08, wspace=0.18,
                                   left=0.05, right=0.98, top=0.95, bottom=0.08)
        self.ax_ts = [self.fig.add_subplot(gs[0, 0])]
        self.ax_ts += [self.fig.add_subplot(gs[i, 0], sharex=self.ax_ts[0]) for i in (1, 2)]
        self.ax_hv = self.fig.add_subplot(gs[:, 1])
        self._draw_signals()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.top)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.top, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._press)
        self.canvas.mpl_connect("motion_notify_event", self._motion)
        self.canvas.mpl_connect("button_release_event", self._release)
        self.top.bind("<Control-z>", lambda e: self.undo())
        self.top.bind("<Command-z>", lambda e: self.undo())
        self.update()

    # ---------- gambar ----------
    def _draw_signals(self):
        n = len(self.data["Z"])
        dec = max(1, n // 20000)
        t = np.arange(0, n, dec) / self.sr
        self.spans = []
        for ax, c in zip(self.ax_ts, "ZNE"):
            d = self.data[c]
            # envelope min/maks per blok agar spike tetap terlihat
            m = (n // dec) * dec
            blk = d[:m].reshape(-1, dec)
            ax.fill_between(t[:len(blk)], blk.min(1), blk.max(1), color="0.25", lw=0)
            lim = np.percentile(np.abs(d), 98) * 2.5  # lonjakan besar boleh terpotong agar sinyal terbaca
            ax.set_ylim(-lim, lim)
            ax.set_ylabel(c, rotation=0, fontweight="bold", labelpad=12)
            ax.set_yticks([])
            row = []
            for k, s in enumerate(self.starts):
                p = ax.axvspan(s / self.sr, (s + self.nwin) / self.sr, lw=0, zorder=0)
                row.append(p)
            self.spans.append(row)
        for ax in self.ax_ts[:2]:
            ax.tick_params(labelbottom=False)
        self.ax_ts[0].set_xlim(0, n / self.sr)
        self.ax_ts[0].set_title(f"{self.name} — {self.tstart.strftime('%Y-%m-%d %H:%M:%S')} UTC", fontsize=11)
        self.ax_ts[2].set_xlabel("Waktu (detik)")

    def _color_spans(self):
        for k in range(self.n):
            if not self.valid[k]:
                fc, a = "black", 0.35          # ada gap: tidak bisa dipakai
            elif self.keep[k]:
                fc, a = window_color(k, self.n), 0.45
            else:
                fc, a = "0.85", 0.6            # dibuang
            for row in self.spans:
                row[k].set_facecolor(fc)
                row[k].set_alpha(a)

    def update(self):
        self._color_spans()
        self.ax_hv.cla()
        try:
            self.r = hv_stats(self.f, self.hv, self.keep & self.valid, self.args.win, self.sr,
                              self.band[0], self.band[1], times=self.starts / self.sr)
            draw_hvsr(self.ax_hv, self.r, self.name)
            rel, clr = sesame(self.r)
            nrel, nclr = sum(c[1] for c in rel), sum(c[1] for c in clr)
            self.status.config(text=(
                f"f0 = {self.r['f0']:.3f} Hz   A0 = {self.r['A0']:.3f}   T0 = {1/self.r['f0']:.3f} s   "
                f"window {self.r['n_used']}/{int(self.valid.sum())}   "
                f"SESAME: reliabel {nrel}/3, puncak jelas {nclr}/6"))
        except ValueError as e:
            self.r = None
            self.ax_hv.text(0.5, 0.5, str(e), ha="center", transform=self.ax_hv.transAxes)
            self.status.config(text=str(e))
        self.canvas.draw_idle() if hasattr(self, "canvas") else None

    # ---------- seleksi ----------
    def _push(self):
        self.history.append((self.keep.copy(), list(self.band)))
        self.history = self.history[-50:]

    def undo(self):
        if self.history:
            self.keep, self.band = self.history.pop()
            self.update()

    def sel_auto(self):
        self._push(); self.keep = self.auto.copy(); self.update()

    def sel_all(self):
        self._push(); self.keep = self.valid.copy(); self.update()

    def sel_none(self):
        self._push(); self.keep[:] = False; self.update()

    def reset_band(self):
        self._push(); self.band = [None, None]; self.update()

    # ---------- mouse ----------
    def _press(self, ev):
        if self.toolbar.mode or ev.inaxes is None or ev.button != 1:
            return
        self.drag = dict(ax=ev.inaxes, x0=ev.xdata, px0=ev.x, py0=ev.y, patches=[])

    def _motion(self, ev):
        d = self.drag
        if not d or ev.inaxes is not d["ax"] or ev.xdata is None:
            return
        for p in d["patches"]:
            p.remove()
        lo, hi = sorted([d["x0"], ev.xdata])
        if d["ax"] is self.ax_hv:
            d["patches"] = [self.ax_hv.axvspan(lo, hi, color="tab:blue", alpha=0.15)]
        else:
            col = "red" if self.mode.get() == "buang" else "green"
            d["patches"] = [ax.axvspan(lo, hi, color=col, alpha=0.25) for ax in self.ax_ts]
        self.canvas.draw_idle()

    def _release(self, ev):
        d, self.drag = self.drag, None
        if not d:
            return
        for p in d["patches"]:
            p.remove()
        x1 = ev.xdata if (ev.inaxes is d["ax"] and ev.xdata is not None) else d["x0"]
        is_click = abs(ev.x - d["px0"]) < 5
        self._push()
        if d["ax"] is self.ax_hv:
            if is_click:
                self._click_curve(ev)
            else:
                lo, hi = sorted([d["x0"], x1])
                self.band = [lo, hi]
        else:
            if is_click:
                inside = np.where((self.starts / self.sr <= d["x0"]) &
                                  ((self.starts + self.nwin) / self.sr >= d["x0"]) & self.valid)[0]
                if inside.size:
                    k = inside[np.argmin(np.abs(self.t_mid[inside] - d["x0"]))]
                    self.keep[k] = not self.keep[k]
                else:
                    self.history.pop()
                    return
            else:
                lo, hi = sorted([d["x0"], x1])
                sel = (self.t_mid >= lo) & (self.t_mid <= hi) & self.valid
                self.keep[sel] = self.mode.get() == "tambah"
        self.update()

    def _click_curve(self, ev):
        """Buang window yang kurvanya paling dekat dengan titik klik pada plot H/V."""
        idx = np.where(self.keep & self.valid)[0]
        if not idx.size or ev.xdata is None:
            self.history.pop()
            return
        j = int(np.argmin(np.abs(np.log(self.f) - np.log(ev.xdata))))
        tr = self.ax_hv.transData
        ypix = np.array([tr.transform((self.f[j], self.hv[k, j]))[1] for k in idx])
        dist = np.abs(ypix - ev.y)
        if dist.min() > 12:
            self.history.pop()
            return
        self.keep[idx[np.argmin(dist)]] = False

    # ---------- simpan ----------
    def save(self):
        from tkinter import messagebox
        if self.r is None:
            messagebox.showwarning("HVSR", "Belum ada window yang dipilih.", parent=self.top)
            return
        os.makedirs(self.outdir, exist_ok=True)
        jpg = os.path.join(self.outdir, f"{self.name}_HVSR.jpg")
        plot_hvsr(self.r, self.name, jpg, dpi=self.args.dpi)
        sig = os.path.join(self.outdir, f"{self.name}_sinyal_picking.jpg")
        self.fig.savefig(sig, dpi=150, format="jpeg", pil_kwargs={"quality": 92})
        nrel, nclr = write_outputs(self.r, self.name, self.outdir, self.tstart, mode="manual (picking)")
        msg = (f"Tersimpan di {os.path.abspath(self.outdir)}:\n"
               f"• {self.name}_HVSR.jpg\n• {self.name}_sinyal_picking.jpg\n"
               f"• {self.name}_HVSR.csv, _hasil.txt, _window.csv\n\n"
               f"f0 = {self.r['f0']:.3f} Hz, A0 = {self.r['A0']:.3f}")
        if self.on_saved:
            self.on_saved(msg)
        messagebox.showinfo("HVSR", msg, parent=self.top)


def open_file(path):
    import subprocess
    if sys.platform == "darwin":
        subprocess.Popen(["open", path])
    elif os.name == "nt":
        os.startfile(path)
    else:
        subprocess.Popen(["xdg-open", path])


# ----------------------------------------------------------------------------
# GUI utama (tkinter)
# ----------------------------------------------------------------------------
def gui(initial=None):
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext

    root = tk.Tk()
    root.title("HVSR Mikrotremor")
    defaults = build_parser().parse_args([])
    fields = [("Panjang window (s)", "win"), ("Overlap (%)", "overlap"), ("Konno-Ohmachi b", "ko"),
              ("Cari f0 dari (Hz, kosong=semua)", "fmin"), ("Cari f0 sampai (Hz)", "fmax"),
              ("Kurva fmin (Hz)", "plot_fmin"), ("Kurva fmax (Hz)", "plot_fmax"),
              ("Reject N std (0=mati, otomatis saja)", "reject")]
    path_var = tk.StringVar(value=initial or "")
    out_var = tk.StringVar(value=os.path.abspath("hasil_hvsr"))
    stations = []

    tk.Label(root, text="Folder data (titik atau kumpulan titik):").grid(row=0, column=0, sticky="w", padx=6)
    tk.Entry(root, textvariable=path_var, width=45).grid(row=0, column=1, padx=4)

    def choose():
        d = filedialog.askdirectory()
        if d:
            path_var.set(d)
            refresh()
    tk.Button(root, text="Pilih…", command=choose).grid(row=0, column=2, padx=4)
    tk.Label(root, text="Folder output:").grid(row=1, column=0, sticky="w", padx=6)
    tk.Entry(root, textvariable=out_var, width=45).grid(row=1, column=1, padx=4)
    tk.Button(root, text="Pilih…", command=lambda: out_var.set(filedialog.askdirectory() or out_var.get())
              ).grid(row=1, column=2, padx=4)
    vars_ = {}
    for i, (lab, key) in enumerate(fields, start=2):
        tk.Label(root, text=lab).grid(row=i, column=0, sticky="w", padx=6)
        v = tk.StringVar(value="" if getattr(defaults, key) is None else str(getattr(defaults, key)))
        tk.Entry(root, textvariable=v, width=12).grid(row=i, column=1, sticky="w", padx=4)
        vars_[key] = v
    at_var = tk.BooleanVar(value=True)
    show_var = tk.BooleanVar(value=False)
    r0 = len(fields) + 2
    tk.Checkbutton(root, text="Anti-trigger STA/LTA", variable=at_var).grid(row=r0, column=0, sticky="w", padx=6)
    tk.Checkbutton(root, text="Buka JPEG setelah proses otomatis", variable=show_var
                   ).grid(row=r0, column=1, sticky="w")

    tk.Label(root, text="Titik ditemukan:").grid(row=r0 + 1, column=0, sticky="nw", padx=6)
    lb = tk.Listbox(root, height=6, width=45, exportselection=False)
    lb.grid(row=r0 + 1, column=1, sticky="w", padx=4, pady=4)
    lb.bind("<Double-Button-1>", lambda e: manual())

    def refresh():
        stations.clear()
        lb.delete(0, "end")
        if path_var.get():
            stations.extend(find_stations([path_var.get()]))
            for nm, fs in stations:
                lb.insert("end", f"{nm}   ({len(fs)} file)")
            if stations:
                lb.selection_set(0)

    def get_args():
        a = build_parser().parse_args([path_var.get(), "-o", out_var.get()])
        for k, v in vars_.items():
            setattr(a, k, float(v.get()) if v.get().strip() else None)
        a.no_antitrigger = not at_var.get()
        return a

    def say(t):
        log.insert("end", t + "\n")
        log.see("end")

    def auto():
        if not path_var.get():
            messagebox.showwarning("HVSR", "Pilih folder data dulu.")
            return
        import io, contextlib
        a = get_args()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = run(a)
        say(buf.getvalue())
        if show_var.get():
            for r in res:
                open_file(os.path.join(a.out, f"{r['name']}_HVSR.jpg"))

    def manual():
        sel = lb.curselection()
        if not sel:
            messagebox.showwarning("HVSR", "Pilih satu titik di daftar dulu.")
            return
        nm, fs = stations[sel[0]]
        a = get_args()
        try:
            root.config(cursor="watch"); root.update()
            trs = load_three_components(fs)
            ManualPicker(root, nm, trs, a, a.out, on_saved=say)
            say(f"Picking manual dibuka: {nm}")
        except Exception as e:
            messagebox.showerror("HVSR", f"{nm}: {e}")
        finally:
            root.config(cursor="")

    btns = tk.Frame(root)
    btns.grid(row=r0 + 2, column=0, columnspan=3, pady=4)
    tk.Button(btns, text="▶ Proses otomatis (semua titik)", command=auto).pack(side="left", padx=6)
    tk.Button(btns, text="✋ Picking manual (titik terpilih)", command=manual).pack(side="left", padx=6)
    log = scrolledtext.ScrolledText(root, width=86, height=12, font=("Menlo", 10))
    log.grid(row=r0 + 3, column=0, columnspan=3, padx=6, pady=6)
    refresh()
    root.mainloop()


def manual_cli(args):
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    for name, files in find_stations(args.input):
        try:
            p = ManualPicker(root, name, load_three_components(files), args, args.out,
                             on_saved=print)
            root.wait_window(p.top)
        except Exception as e:
            print(f"  [GAGAL] {name}: {e}")
    root.destroy()


if __name__ == "__main__":
    a = build_parser().parse_args()
    if a.manual and a.input:
        manual_cli(a)
    elif a.gui or not a.input:
        gui(a.input[0] if a.input else None)
    else:
        run(a)
