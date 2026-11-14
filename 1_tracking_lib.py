"""Helpers for 1_tracking.ipynb: scenario rendering and truth, channel extraction, the tracking loops, lock tests
and scoring. The notebook holds the experiments; anything reusable or slow lives here.

Import it with  T = importlib.import_module("1_tracking_lib")  (the name starts with a digit).

Conventions
- Time is seconds from the scenario clock start (= first IQ sample). Per-second series are indexed by the whole
  second k and describe the bin [k - 0.5, k + 0.5), as in dopplerguesser.
- Inside a channel, frequencies are relative to the channel centre `fc`; the trackers add `fc` back on output.
- A *target* is a dict made by `make_target`: one emitter, the channel it is extracted into and what is assumed
  known about its modulation.
"""

import csv
import ctypes
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.fft as sfft
from scipy.signal import firwin, upfirdn

ROOT = Path(__file__).resolve().parent
ASSIST, SCEN, OUT = ROOT / "ASSIST", ROOT / "scenarios", ROOT / "out"
FS = 2_400_000  # stream sample rate, Hz
FC = 137.0e6  # stream centre frequency, Hz
FADE_S = 15.0  # ASSIST's horizon fade (in the IQ, not in the truth)
R_MIN, R_MAX = -95.0, 6.0  # Doppler-rate limits assumed for a satellite at 137 MHz, Hz/s (as in 0_detection)

# ---------------------------------------------------------------------------------------------------------------
# scenarios and truth
# ---------------------------------------------------------------------------------------------------------------


def render(name):
    """Render IQ (cu8) and truth of a scenario with ASSIST unless already there. Returns the cu8 path."""
    out = OUT / name
    iq = sorted(out.glob("*.cu8"))
    if not iq:
        subprocess.run(
            ["uv", "run", "assist", "run", str(SCEN / f"{name}.yaml"), "--sink", "cu8", "-o", str(out)],
            cwd=ASSIST,
            check=True,
            capture_output=True,
        )
        iq = sorted(out.glob("*.cu8"))
    if not (out / "truth.csv").exists():
        subprocess.run(
            ["uv", "run", "assist", "truth", str(SCEN / f"{name}.yaml"), "-o", str(out / "truth.csv")],
            cwd=ASSIST,
            check=True,
            capture_output=True,
        )
    return iq[0]


def _t(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _f(s):
    try:
        return float(s)
    except ValueError:
        return np.nan


_TRUTH = {}


def load_truth(name):
    """{"source/emitter": dict} for the satellite emitters of a scenario, on the whole-second grid k = 0..dur-1:
    f [Hz] (NaN below the horizon), cn0 [dB-Hz] as received (keying and tumble included, ASSIST's 15 s horizon fade
    applied here because the truth file leaves it out; -inf when off), on (fraction of the second the emitter is
    keyed on), rate [Hz/s], signal, f_rest."""
    if name in _TRUTH:
        return _TRUTH[name]
    out = OUT / name
    run = json.loads((out / "run.json").read_text())
    t0, dur = _t(run["render"]["start_utc"]), int(run["render"]["duration_s"])
    em = {}
    with open(out / "truth.csv") as fh:
        for r in csv.DictReader(fh):
            if not r["range_km"]:
                continue  # ground and local emitters
            key = f"{r['source_id']}/{r['emitter_id']}"
            e = em.setdefault(
                key,
                {
                    "signal": r["signal"],
                    "f_rest": _f(r["f_rest_hz"]),
                    "f": np.full(dur, np.nan),
                    "cn0": np.full(dur, -np.inf),
                    "on": np.zeros(dur),
                    "el": np.full(dur, np.nan),
                },
            )
            k = round((_t(r["t_utc"]) - t0).total_seconds())
            if 0 <= k < dur:
                e["f"][k], e["on"][k], e["el"][k] = _f(r["f_rx_hz"]), _f(r["active_fraction"]), _f(r["el_deg"])
                c = _f(r["cn0_dbhz"])
                e["cn0"][k] = c if np.isfinite(c) and e["on"][k] > 0 else -np.inf
    for e in em.values():
        up = np.flatnonzero(np.isfinite(e["f"]))
        e["aos"], e["los"] = int(up[0]), int(up[-1])
        k = np.arange(dur)
        x = np.clip(np.minimum(k - e["aos"], e["los"] - k) / FADE_S, 0, 1)
        with np.errstate(divide="ignore"):
            e["cn0"] = e["cn0"] + 20 * np.log10(0.5 - 0.5 * np.cos(np.pi * x))
        e["cn0"][~np.isfinite(e["cn0"])] = -np.inf
        e["rate"] = np.gradient(e["f"])
    _TRUTH[name] = em
    return em


def load_bursts(name, f_lo, f_hi):
    """Start/end times [s] of the truth burst events (VDL, AM voice, ...) whose frequency lies in [f_lo, f_hi]."""
    out = OUT / name
    t0 = _t(json.loads((out / "run.json").read_text())["render"]["start_utc"])
    ev = []
    with open(out / "truth.events.csv") as fh:
        for r in csv.DictReader(fh):
            f = _f(r["f_rx_hz"])
            if f_lo <= f <= f_hi:
                a, b = (_t(r["start"]) - t0).total_seconds(), (_t(r["end"]) - t0).total_seconds()
                ev.append((a, b, _f(r["cn0_dbhz"])))
    return np.array(ev).reshape(-1, 3)


# ---------------------------------------------------------------------------------------------------------------
# targets and channel extraction
# ---------------------------------------------------------------------------------------------------------------

# What is assumed known about each modulation (the notebook's section 6 shows where the lines come from):
#   decim  channel decimation from 2.4 MS/s; the channel must hold the signal over its whole Doppler excursion
#   m1     pre-filter boxcar length in channel samples, applied after the carrier estimate has been removed
#   lines  {power: offset of the usable spectral line of x**power from power * carrier, in units of the symbol rate}
MODEMS = {
    "cw": {"decim": 250, "m1": 1, "lines": {1: 0.0}},
    "sdpsk": {"decim_per_rs": {4800: 75, 9600: 40}, "m1": 5, "lines": {2: 0.5, 4: 0.0}},
    "oqpsk": {"decim_per_rs": {72000: 15, 40000: 25}, "m1": 1, "lines": {2: 1.0, 4: 0.0}},
}
DEFAULT_RS = {"sdpsk": 4800, "oqpsk": 72000}


def make_target(scenario, key, rs=None):
    e = load_truth(scenario)[key]
    kind = e["signal"]
    m = MODEMS[kind]
    rs = rs or DEFAULT_RS.get(kind)
    decim = m["decim"] if kind == "cw" else m["decim_per_rs"][rs]
    fc = round(e["f_rest"] / 1000) * 1000.0  # channel centre: the rest frequency, to 1 kHz
    return {
        "scenario": scenario,
        "key": key,
        "kind": kind,
        "rs": rs,
        "fc": fc,
        "decim": decim,
        "fs": FS / decim,
        "m1": m["m1"],
        "lines": {p: off * rs if rs else 0.0 for p, off in m["lines"].items()},
        "truth": e,
    }


def read_iq(mm, s0, n):
    """n complex samples from sample s0 of an interleaved unsigned 8-bit I/Q memmap (zeros outside the file)."""
    total = mm.size // 2
    a, b = max(s0, 0), min(s0 + n, total)
    out = np.zeros(n, np.complex64)
    if b > a:
        raw = np.asarray(mm[2 * a : 2 * b]).astype(np.float32)
        raw -= 127.5
        out[a - s0 : b - s0] = raw.view(np.complex64)
    return out


_LUT = None


def _mixer(s0, n, f_off):
    # exp(-2 pi j f_off (s0 + i) / FS), exact for integer f_off at any distance from the start
    global _LUT
    if _LUT is None:
        _LUT = np.exp(-2j * np.pi * np.arange(FS) / FS).astype(np.complex64)
    return _LUT[((s0 + np.arange(n, dtype=np.int64)) * int(f_off)) % FS]


def extract_channel(scenario, fc, decim, chunk_s=1.0):
    """The band fc +- 0.4 FS/decim of a scenario's IQ, mixed to 0 Hz and decimated to FS/decim. Cached as complex64
    .npy; returns a read-only memmap. Output sample m is centred on input sample m * decim. The filter is a
    Kaiser-windowed FIR of 20 * decim + 1 taps: flat to 0.4 of the output rate, about 60 dB down from 0.6."""
    cache = OUT / scenario / "trk_cache"
    path = cache / f"ch_{int(fc)}_{decim}.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    cache.mkdir(exist_ok=True)
    mm = np.memmap(render(scenario), dtype=np.uint8, mode="r")
    total = mm.size // 2
    pad = 10 * decim
    h = firwin(2 * pad + 1, 0.5 * FS / decim, window=("kaiser", 6.0), fs=FS).astype(np.float32)
    n_out = total // decim
    y = np.lib.format.open_memmap(path.with_suffix(".tmp.npy"), mode="w+", dtype=np.complex64, shape=(n_out,))
    step = int(chunk_s * FS) // decim * decim
    for s0 in range(0, n_out * decim, step):
        n = min(step, n_out * decim - s0)
        x = read_iq(mm, s0 - pad, n + 2 * pad) * _mixer(s0 - pad, n + 2 * pad, fc - FC)
        full = upfirdn(h, x, 1, decim)  # full[i] is centred on x[i * decim - pad]
        y[s0 // decim : (s0 + n) // decim] = full[2 * pad // decim : 2 * pad // decim + n // decim]
    y.flush()
    del y
    path.with_suffix(".tmp.npy").rename(path)
    return np.load(path, mmap_mode="r")


def extract_all(targets, workers=6):
    """Extract the channels of all targets (threads; cached). Adds target["x"]."""
    jobs = sorted({(t["scenario"], t["fc"], t["decim"]) for t in targets})
    with ThreadPoolExecutor(workers) as ex:
        res = dict(zip(jobs, ex.map(lambda j: extract_channel(*j), jobs), strict=True))
    for t in targets:
        t["x"] = res[(t["scenario"], t["fc"], t["decim"])]
    return targets


def noise_density(x, fs):
    """Noise power per Hz of a channel: median short-frame power (robust to bursts and to a narrow signal)."""
    n = min(len(x), int(60 * fs)) // 256 * 256
    P = np.abs(sfft.fft(np.asarray(x[:n]).reshape(-1, 256), axis=1)) ** 2 / 256
    return float(np.median(P) / np.log(2.0) / fs)


# ---------------------------------------------------------------------------------------------------------------
# burst blanking
# ---------------------------------------------------------------------------------------------------------------


def burst_mask(x, fs, frame_s=0.0025, thr=2.0, ref_s=2.0, hold=1):
    """Per-sample mask (uint8, 1 = blank) of the short frames in which the channel power jumps above `thr` times
    its recent level. Causal: the reference is the median frame power over the previous `ref_s` seconds, taken in
    blocks of ref_s / 4. Flags are widened by `hold` frames on both sides (the look-ahead costs a few ms)."""
    L = max(4, round(frame_s * fs))
    nfr = len(x) // L
    xv = np.asarray(x[: nfr * L])
    p = (xv.real**2 + xv.imag**2).reshape(nfr, L).mean(1)
    blk = max(8, round(ref_s / 4 / frame_s))
    nb = nfr // blk
    med = np.median(p[: nb * blk].reshape(nb, blk), axis=1)
    ref = np.empty(nb + 1)
    for b in range(nb + 1):  # median of the block medians of the previous ref_s
        ref[b] = np.median(med[max(0, b - 4) : b]) if b > 0 else med[0]
    flag = p > thr * np.repeat(ref, blk)[:nfr]
    wide = flag.copy()
    for s in range(1, hold + 1):
        wide[s:] |= flag[:-s]
        wide[:-s] |= flag[s:]
    out = np.zeros(len(x), np.uint8)
    out[: nfr * L] = np.repeat(wide, L)
    return out


# ---------------------------------------------------------------------------------------------------------------
# compiled loops
# ---------------------------------------------------------------------------------------------------------------

_LIB = None


def kernels():
    """The compiled loops of 1_tracking_kernels.cpp (built on first use, rebuilt when the source is newer)."""
    global _LIB
    if _LIB is None:
        src, so = ROOT / "1_tracking_kernels.cpp", ROOT / "1_tracking_kernels.so"
        if not so.exists() or so.stat().st_mtime < src.stat().st_mtime:
            subprocess.run(["c++", "-O3", "-std=c++17", "-shared", "-fPIC", "-o", str(so), str(src)], check=True)
        lib = ctypes.CDLL(str(so))
        P, D, N, L = ctypes.c_void_p, ctypes.c_double, ctypes.c_int, ctypes.c_int64
        lib.pll_classic.argtypes = [P, L, D, D, D, D, D, P, P, L, P, P]
        lib.pll_coh.argtypes = [P, L, D, N, N, N, D, N, D, D, D, P, P, P, P, P, P]
        lib.pll_classic.restype = lib.pll_coh.restype = None
        _LIB = lib
    return _LIB


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p) if a is not None else None


def bin_seconds(t, v, dur):
    """Mean of the samples v at times t in each bin [k - 0.5, k + 0.5), k = 0..dur-1 (NaN where empty)."""
    k = np.floor(np.asarray(t) + 0.5).astype(int)
    ok = (k >= 0) & (k < dur) & np.isfinite(v)
    s = np.bincount(k[ok], weights=np.asarray(v, float)[ok], minlength=dur)
    c = np.bincount(k[ok], minlength=dur)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def run_pll_classic(
    tgt,
    t_start,
    f0,
    r0=None,
    bw=100.0,
    power=1,
    bw_acq=None,
    t_acq=3.0,
    blank=None,
    t_end=None,
    x=None,
    fs=None,
    state=None,
):
    """dopplerguesser's PLL on a target's channel from t_start, the NCO centred on f0 [Hz absolute].
    bw is dopplerguesser's `bw_hz` (see the notebook for what it means in Hz). power > 1 raises the channel to that
    power first and tracks the line named in the target's `lines`. bw_acq/t_acq: a wider loop for the first t_acq
    seconds. r0 is ignored (the loop has no rate input). `state` (from a previous call's result) continues that
    run from t_start instead of starting a new one. Returns per-second absolute frequency (NaN outside the run),
    mean |phase error| and the state."""
    lib = kernels()
    x = tgt["x"] if x is None else x
    fs = tgt["fs"] if fs is None else fs
    dur = len(tgt["truth"]["f"])
    i0, i1 = int(t_start * fs), int((t_end or dur) * fs)
    blk = max(1, round(fs / 100))  # 10 ms averages out of the kernel
    seg = np.ascontiguousarray(x[i0:i1], np.complex64)
    off = tgt["lines"][power]
    if power > 1:
        seg = seg**power
    msk = None if blank is None else np.ascontiguousarray(blank[i0:i1])
    st = np.zeros(2) if state is None else state["st"].copy()
    f_center = power * (f0 - tgt["fc"]) + off if state is None else state["f_center"]
    if state is not None:
        bw_acq = None
    dphi_max = np.pi  # no limit beyond the channel itself
    f_out, e_out = [], []

    def gains(b, zeta=0.707):
        wn = b / (zeta + 1 / (4 * zeta))
        return 2 * zeta * wn / fs, (wn / fs) ** 2

    cuts = [0, len(seg) // blk * blk]
    bws = [bw]
    if bw_acq:
        cuts = [0, min(int(t_acq * fs) // blk * blk, cuts[1]), cuts[1]]
        bws = [bw_acq, bw]
    for a, b, bb in zip(cuts[:-1], cuts[1:], bws, strict=True):
        K1, K2 = gains(bb)
        fo, eo = np.zeros((b - a) // blk, np.float32), np.zeros((b - a) // blk, np.float32)
        s = seg[a:b]
        lib.pll_classic(
            _ptr(s),
            len(s),
            fs,
            f_center,
            dphi_max,
            K1,
            K2,
            _ptr(None if msk is None else msk[a:b]),
            _ptr(st),
            blk,
            _ptr(fo),
            _ptr(eo),
        )
        f_out.append(fo)
        e_out.append(eo)
    # float64 before adding fc: float32 has a 16 Hz step at 137 MHz
    f_out, e_out = np.concatenate(f_out).astype(float), np.concatenate(e_out).astype(float)
    t = t_start + (np.arange(len(f_out)) + 0.5) * blk / fs
    f = tgt["fc"] + (f_out - off) / power
    return {"f": bin_seconds(t, f, dur), "perr": bin_seconds(t, e_out, dur), "state": {"st": st, "f_center": f_center}}


def run_pll_coh(
    tgt,
    t_start,
    f0,
    r0=0.0,
    bn=5.0,
    t_coh=0.02,
    order=3,
    power=1,
    bn_acq=None,
    t_acq=3.0,
    blank=None,
    t_end=None,
    m1=None,
    state=None,
):
    """The integrate-then-detect loop (pll_coh in the kernels) on a target's channel from t_start, starting at
    frequency f0 [Hz absolute] and rate r0 [Hz/s]. bn: loop noise bandwidth [Hz]; t_coh: coherent integration per
    update [s]. Returns per-second absolute frequency, rate, and two lock statistics from the loop's own I/Q sums:
    pli = mean(I^2 - Q^2) / mean(I^2 + Q^2) (1 = phase locked, 0 = not) and snr = the same sums as a C/N0 estimate
    of the tracked line [dB-Hz]. `state` continues a previous call's run. t_start and t_end should be whole
    seconds when runs are chained."""
    lib = kernels()
    fs = tgt["fs"]
    dur = len(tgt["truth"]["f"])
    m1 = m1 or (tgt["m1"] if power > 1 else 1)
    m2 = max(1, round(t_coh * fs / m1))
    upd = m1 * m2
    i0 = int(t_start * fs)
    n_upd = (int((t_end or dur) * fs) - i0) // upd
    seg = np.ascontiguousarray(tgt["x"][i0 : i0 + n_upd * upd], np.complex64)
    msk = None if blank is None else np.ascontiguousarray(blank[i0 : i0 + n_upd * upd])
    st = np.array([0.0, f0 - tgt["fc"], r0, 0.0]) if state is None else state["st"].copy()
    cuts, bns = [0, n_upd], [bn]
    if bn_acq and state is None:
        cuts, bns = [0, min(int(t_acq * fs / upd), n_upd), n_upd], [bn_acq, bn]
    outs = [np.zeros(n_upd, np.float32) for _ in range(4)]
    for a, b, bb in zip(cuts[:-1], cuts[1:], bns, strict=True):
        s = seg[a * upd : b * upd]
        lib.pll_coh(
            _ptr(s),
            len(s),
            fs,
            power,
            m1,
            m2,
            tgt["lines"][power],
            order,
            bb,
            R_MIN - 5,
            R_MAX + 5,
            _ptr(None if msk is None else msk[a * upd : b * upd]),
            _ptr(st),
            *[_ptr(o[a:b]) for o in outs],
        )
    f_u, r_u, pi_u, pq_u = outs
    t = t_start + (np.arange(n_upd) + 0.5) * upd / fs
    f_u = f_u - r_u * upd / fs  # after an update the state already holds the prediction for the next one
    i2, q2 = bin_seconds(t, pi_u.astype(float) ** 2, dur), bin_seconds(t, pq_u.astype(float) ** 2, dur)
    with np.errstate(invalid="ignore", divide="ignore"):
        pli = (i2 - q2) / (i2 + q2)
        # signal power ~ I^2 - Q^2 (noise is in both), noise power per update ~ 2 Q^2; noise bandwidth 1 / T
        snr = 10 * np.log10(np.maximum(i2 - q2, 1e-30) / (2 * q2) / (upd / fs))
    return {
        "f": tgt["fc"] + bin_seconds(t, f_u, dur),
        "rate": bin_seconds(t, r_u, dur),
        "pli": pli,
        "snr": snr,
        "state": {"st": st},
    }


# ---------------------------------------------------------------------------------------------------------------
# FFT tracker
# ---------------------------------------------------------------------------------------------------------------


def line_search(seg, fs, f_c, r_c, f_half, rates, power=1, m1=1, f_line=0.0, blank=None, pad=2):
    """Find a spectral line in one block of channel samples by a search over frequency and Doppler rate.

    The block is first mixed with a chirp at the centre of the search region (frequency f_c at the block centre,
    rate r_c), which makes the wanted signal nearly stationary. Then, as in pll_coh: boxcar pre-filter m1, raise to
    `power`, shift the modulation's line (f_line from power * carrier) to 0; when f_line is not 0 the twin line at
    -f_line is shifted to 0 in a second copy and the two power spectra are added, so that a single line can never
    be mistaken for the wanted one (it would leave the wrong carrier off by f_line / power). The result is decimated
    to just cover
    +- power * f_half, and for every rate in `rates` it is multiplied by the residual chirp and Fourier transformed
    (zero-padded `pad` times). The largest bin within +- f_half wins and is refined by a parabola through its
    neighbours.

    Returns (f, r, score, cells): carrier frequency at the block centre and rate, the peak power over the noise
    level (median of all bins), and the number of independent cells searched, for setting a threshold.
    """
    n = len(seg)
    tau = (np.arange(n) + 0.5) / fs - 0.5 * n / fs
    u = seg * np.exp(-2j * np.pi * (f_c * tau + 0.5 * r_c * tau * tau)).astype(np.complex64)
    if blank is not None:
        u[blank > 0] = 0
    if power > 1:
        if m1 > 1:
            k = n // m1 * m1
            u, tau = u[:k].reshape(-1, m1).mean(1), tau[:k].reshape(-1, m1).mean(1)
        u = u**power
    # a line offset from power x carrier comes with a twin at the opposite offset: both are searched together
    us = [u * np.exp(-2j * np.pi * sg * f_line * tau).astype(np.complex64) for sg in ((1, -1) if f_line else (0,))]
    fs1 = fs * len(u) / n
    m2 = max(1, int(fs1 / (4 * power * f_half)))  # boxcar droop at the edge of the search band < 1 dB
    k = len(u) // m2 * m2
    us = [v[:k].reshape(-1, m2).mean(1) for v in us]
    tau = tau[:k].reshape(-1, m2).mean(1)
    fs_d = fs1 / m2
    nfft = sfft.next_fast_len(pad * len(tau))
    d = power * (np.asarray(rates, float) - r_c)
    chirp = np.exp(-1j * np.pi * d[:, None] * tau[None, :] ** 2).astype(np.complex64)
    S = sum(np.abs(sfft.fft(v[None, :] * chirp, nfft, axis=1)) ** 2 for v in us)
    fr = sfft.fftfreq(nfft, 1 / fs_d)
    sel = np.flatnonzero(np.abs(fr) <= power * f_half)
    # noise level of one line's bins (median of exponential: ln 2 x mean; of a sum of two: 1.678 x one's mean)
    noise = np.median(S) / (np.log(2.0) if len(us) == 1 else 1.678)
    Ss = S[:, sel]
    i, j = np.unravel_index(np.argmax(Ss), Ss.shape)
    b = sel[j]
    a0, a1, a2 = np.log(S[i, b - 1] + 1e-30), np.log(S[i, b] + 1e-30), np.log(S[i, (b + 1) % nfft] + 1e-30)
    den = a0 - 2 * a1 + a2
    delta = 0.5 * (a0 - a2) / den if den < 0 else 0.0
    f = f_c + (fr[b] + delta * fs_d / nfft) / power
    sc = float(Ss[i, j] / noise)
    if len(us) == 2:
        sc = sc - np.log1p(sc)  # same tail as one exponential: (1 + s) e^-s = e^-(s - ln(1 + s))
    return f, float(rates[i]), sc, len(sel) / pad * len(rates)


def fft_track(
    tgt,
    t_start,
    f0,
    r0=None,
    f_unc=15.0,
    r_unc=3.0,
    t_blk=1.0,
    power=1,
    blank=None,
    t_end=None,
    margin=7.0,
    hist_s=8.0,
    gate=6.0,
    m1=None,
    jerk=1.0,
    acq_timeout=None,
    rehandoff=None,
):
    """Causal FFT tracker: one line search per second on the block [k - t_blk/2, k + t_blk/2).

    Hand-off: at t_start the carrier is within f_unc of f0 [Hz absolute]; its rate is within r_unc of r0, or unknown
    (r0 = None: the whole range R_MIN..R_MAX is searched).

    Two modes per block.
    - Tracking (>= 2 accepted measurements in the last hist_s seconds, the latest at most 3 s old): a polynomial
      through them (degree 2 from 5 points, else 1) predicts frequency and rate; the search is +- gate around it.
    - Acquisition (otherwise): the search region follows from the last accepted point and the time since then,
      using only that a satellite's Doppler rate is between R_MIN and R_MAX and changes by at most `jerk` Hz/s^2.
      A hit here is held back until a second hit within 2 s continues it (same frequency after allowing for the
      measured rate), which removes isolated noise peaks from the large acquisition search.
    A measurement is accepted when its score exceeds ln(cells searched) + margin (noise alone does so with
    probability about exp(-margin) per block).

    acq_timeout [s]: after this long without an accepted measurement the tracker gives up and waits for a new
    hand-off from rehandoff(k) -> (f0, r0) or None (None: the signal is not detectable, nothing is searched).

    Returns per-second absolute f (NaN where nothing was accepted), rate, score [dB], thr [dB], mode (1 tracking,
    2 acquisition, 0 not run), handoffs (times of the hand-offs used).
    """
    fs, x, fc = tgt["fs"], tgt["x"], tgt["fc"]
    dur = len(tgt["truth"]["f"])
    m1 = m1 or (tgt["m1"] if power > 1 else 1)
    f_line = tgt["lines"][power]
    t_acq = min(t_blk, 1.0)  # acquisition searches are large: at most 1 s blocks
    out = {k: np.full(dur, np.nan) for k in ("f", "rate", "score", "thr")}
    out["mode"] = np.zeros(dur, int)
    hist = []  # accepted (t, f)
    last = (t_start, f0 - fc, r0)  # last known (t, f, r); r None = unknown
    lim = 0.42 * fs
    tent = None  # unconfirmed acquisition hit (t, f, r)
    out["handoffs"] = [t_start]
    for k in range(int(np.ceil(t_start + t_blk / 2)), int((t_end or dur) - t_blk / 2) + 1):
        gap = k - last[0]
        if acq_timeout and gap > acq_timeout:
            h = rehandoff(k) if rehandoff else None
            if h is None:
                continue  # given up: wait for a new detection
            last, hist, tent, gap = (k, h[0] - fc, h[1]), [], None, 0
            out["handoffs"].append(k)
        pts = [(t, f) for t, f in hist if k - t <= hist_s]
        tracking = len(pts) >= 2 and gap <= 3
        tb = t_blk if tracking else t_acq
        n = round(tb * fs)
        step = 1.0 / (power * tb * tb)  # rate grid: at most ~0.4 rad of chirp left at the edges
        i0 = round((k - tb / 2) * fs)
        if i0 < 0 or i0 + n > len(x):
            continue
        seg = np.asarray(x[i0 : i0 + n])
        msk = None if blank is None else blank[i0 : i0 + n]
        if tracking:
            tt, ff = np.array(pts).T
            c = (
                np.polyfit(tt - k, ff, 2 if len(pts) >= 5 else 1)
                if len(pts) >= 3
                else np.array([last[2], ff[-1] + last[2] * (k - tt[-1])])
            )
            f_p, r_p = c[-1], c[-2]
            f_half = gate + 3.0 * (gap - 1)
            rates = r_p + np.arange(-3, 4) * step
            mode = 1
        else:
            fu = f_unc if not hist else gate
            lo, hi = last[1] + R_MIN * gap - fu, last[1] + R_MAX * gap + fu
            lo, hi = max(lo, -lim), min(hi, lim)
            if hi <= lo:
                continue
            f_p, f_half = 0.5 * (lo + hi), 0.5 * (hi - lo)
            if last[2] is None:
                r_lo, r_hi = R_MIN, R_MAX
            else:
                ru = r_unc if not hist else 2.0
                r_lo, r_hi = max(R_MIN, last[2] - jerk * gap - ru), min(R_MAX, last[2] + jerk * gap + ru)
            rates = np.arange(r_lo, r_hi + step, step)
            mode = 2
        r_c = float(rates[len(rates) // 2])
        # after the pre-filter the power-law signal covers only +- 0.4 fs / m1 without aliasing: wider regions are
        # searched in tiles (an aliased search lands on one of the twin lines, a carrier error of f_line / power)
        # (aliased images of the twin line carry one line only, so the two-line sum still picks the right one)
        tile = 0.4 * fs / (m1 if power > 1 else 1) / power
        nt = max(1, int(np.ceil(f_half / tile)))
        tiles = [
            line_search(seg, fs, c, r_c, f_half / nt, rates, power, m1, f_line, msk)
            for c in f_p + (np.arange(nt) - (nt - 1) / 2) * (2 * f_half / nt)
        ]
        f, r, sc, _ = max(tiles, key=lambda res: res[2])
        cells = sum(res[3] for res in tiles)
        thr = np.log(max(cells, 1.0)) + margin
        out["score"][k], out["thr"][k], out["mode"][k] = 10 * np.log10(sc), 10 * np.log10(thr), mode
        if sc <= thr:
            continue
        if mode == 2:
            # an acquisition hit counts only once a second one, at most 2 s later, continues it
            pred = tent[1] + 0.5 * (tent[2] + r) * (k - tent[0]) if tent is not None else None
            if tent is not None and k - tent[0] <= 2 and abs(f - pred) < gate:
                out["f"][tent[0]], out["rate"][tent[0]] = fc + tent[1], tent[2]
                hist = [(tent[0], tent[1])]
                tent = None
            else:
                tent = (k, f, r)
                continue
        out["f"][k], out["rate"][k] = fc + f, r
        hist.append((k, f))
        hist = hist[-int(hist_s) - 2 :]
        if len(hist) >= 3 and hist[-1][0] - hist[-3][0] <= 4:
            r = (hist[-1][1] - hist[-3][1]) / (hist[-1][0] - hist[-3][0])
        last = (k, f, r)
    return out


# ---------------------------------------------------------------------------------------------------------------
# lock tests and scoring
# ---------------------------------------------------------------------------------------------------------------


def linefit_lock(f, window=10, deg=1, min_pts=None):
    """The line-fit lock test: for every second k, fit a polynomial of degree `deg` by least squares to the
    per-second frequencies in [k - window/2, k + window/2) and return (rms residual of the window, residual of
    point k). The decision for second k is available window/2 seconds later. NaN where the window has fewer than
    min_pts (default: all) points."""
    n = len(f)
    min_pts = min_pts or window
    rms, res = np.full(n, np.nan), np.full(n, np.nan)
    h = window // 2
    tt = np.arange(-h, window - h, dtype=float)
    for k in range(h, n - (window - h) + 1):
        w = f[k - h : k - h + window]
        ok = np.isfinite(w)
        if ok.sum() < max(min_pts, deg + 2) or not ok[h]:
            continue
        c = np.polyfit(tt[ok], w[ok], deg)
        r = w[ok] - np.polyval(c, tt[ok])
        rms[k] = np.sqrt(np.mean(r**2) * ok.sum() / (ok.sum() - deg - 1))
        res[k] = w[h] - np.polyval(c, 0.0)
    return rms, res


def handoff(tgt, cn0_start, f_err=5.0, r_err=1.0):
    """A hand-off as the detector would give it: (t_start, f0, r0) at the first second the truth C/N0 reaches
    cn0_start with the emitter on, the true frequency and rate off by f_err and r_err."""
    e = tgt["truth"]
    k = int(np.flatnonzero((e["cn0"] >= cn0_start) & (e["on"] >= 0.999))[0])
    return k, e["f"][k] + f_err, e["rate"][k] + r_err


def score(tgt, f_est, trusted=None, t_from=0, cn0_min=None, tol=5.0):
    """Compare a per-second estimate with the truth.
    trusted: which seconds the method stands behind (default: wherever it gave a number).
    Returns: n (trusted points), rmse and rmse_c (after removing the mean error, as dopplerguesser scores) over the
    trusted points [Hz], bias, outliers (fraction of trusted points more than tol off), coverage (trusted and
    within tol, as a fraction of the seconds from t_from in which the emitter is fully on with C/N0 >= cn0_min),
    and the error series."""
    e = tgt["truth"]
    k = np.arange(len(e["f"]))
    err = f_est - e["f"]
    tr = np.isfinite(err) & (k >= t_from)
    if trusted is not None:
        tr &= trusted
    exp = (k >= t_from) & (e["on"] >= 0.999) & np.isfinite(e["f"])
    if cn0_min is not None:
        exp &= e["cn0"] >= cn0_min
    good = tr & (np.abs(err) <= tol)
    ev = err[tr]
    return {
        "n": int(tr.sum()),
        "rmse": float(np.sqrt(np.mean(ev**2))) if len(ev) else np.nan,
        "rmse_c": float(np.std(ev)) if len(ev) else np.nan,
        "bias": float(np.mean(ev)) if len(ev) else np.nan,
        "outliers": float(np.mean(np.abs(ev) > tol)) if len(ev) else np.nan,
        "coverage": float((good & exp).sum() / max(exp.sum(), 1)),
        "expected": int(exp.sum()),
        "err": err,
        "trusted": tr,
    }


def run_segments(tgt, fn, seg_s=60, settle=5, cn0_min=8.0, f_err=5.0, r_err=1.0, **kw):
    """Steady-state test: cut the pass into seg_s-second segments and start the tracker `fn` afresh in each one
    from a detector-like hand-off (true frequency + f_err, true rate + r_err), so that every C/N0 level is tried
    with a good start. fn(tgt, t_start, f0, r0, t_end=..., **kw) -> dict with "f". Returns the merged per-second
    f and `use`, the seconds to score (the first `settle` of each segment are left out). Segments in which the
    emitter is not on throughout or is below cn0_min at the start are skipped."""
    e = tgt["truth"]
    dur = len(e["f"])
    f, use = np.full(dur, np.nan), np.zeros(dur, bool)
    for a in range(e["aos"] + int(FADE_S), e["los"] - seg_s, seg_s):
        if e["on"][a : a + seg_s].min() < 0.999 or e["cn0"][a] < cn0_min:
            continue
        o = fn(tgt, a, e["f"][a] + f_err, e["rate"][a] + r_err, t_end=a + seg_s, **kw)
        f[a : a + seg_s] = o["f"][a : a + seg_s]
        use[a + settle : a + seg_s] = True
    return {"f": f, "use": use}


def good_by_cn0(runs, edges, tol=5.0):
    """Fraction of scored seconds within tol of the truth, per true-C/N0 bin, pooled over (target, run) pairs
    from run_segments (or any dict with "f" and optional "use")."""
    c, g = [], []
    for tgt, o in runs:
        e = tgt["truth"]
        use = o.get("use", np.isfinite(e["f"]))
        c.append(e["cn0"][use])
        g.append(np.abs(o["f"] - e["f"])[use] <= tol)
    c, g = np.concatenate(c), np.concatenate(g)
    out = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        m = (c >= edges[i]) & (c < edges[i + 1])
        if m.sum() >= 20:
            out[i] = g[m].mean()
    return out


def lock_tests(o, f, tests, thr=3.0, window=10, deg=1, perr_thr=1.5, pli_thr=0.3):
    """Per-second lock decision of a loop output: every test named in `tests` must pass.
      "linefit"  rms residual of a degree-`deg` polynomial over `window` s below thr [Hz], and the second's own
                 residual below 3 thr (decision delayed by window/2)
      "perr"     the classic loop's mean |phase error| over the second below perr_thr [rad] (noise alone: pi/2)
      "pli"      the coherent loop's phase-lock indicator above pli_thr (noise alone: 0)
    Returns (ok, rms)."""
    ok = np.isfinite(f)
    rms = np.full(len(f), np.nan)
    if "linefit" in tests:
        rms, res = linefit_lock(f, window, deg)
        ok &= (rms < thr) & (np.abs(res) < 3 * thr)
    if "perr" in tests:
        ok &= o["perr"] < perr_thr
    if "pli" in tests:
        ok &= o["pli"] > pli_thr
    return ok, rms


def run_reseeded(
    tgt,
    fn,
    tests=("linefit",),
    det_cn0=12.0,
    patience=10,
    step=10,
    f_err=5.0,
    r_err=1.0,
    thr=3.0,
    window=10,
    deg=1,
    perr_thr=1.5,
    pli_thr=0.3,
    **kw,
):
    """A PLL with a supervisor, over a whole pass.

    The loop `fn` (run_pll_classic or run_pll_coh) is started from a hand-off and its per-second output is checked
    with `lock_tests`. When no second has passed for `patience` seconds in a row (counting only decisions already
    available) the loop is declared lost and is started again from a new hand-off. Hand-offs are detector-like:
    true frequency + f_err and rate + r_err, available only in seconds where the emitter is on and its C/N0 is at
    least det_cn0 (below that the supervisor waits). Everything is causal; with "linefit" the decision for second
    k is known at k + window/2, otherwise at k + 1.

    Returns per-second f, lock (the trusted seconds), rms (line-fit), seeds (hand-off times).
    """
    e = tgt["truth"]
    dur = len(e["f"])
    avail = (e["cn0"] >= det_cn0) & (e["on"] >= 0.999)
    f = np.full(dur, np.nan)
    aux = {"perr": np.full(dur, np.nan), "pli": np.full(dur, np.nan)}
    delay = (window - window // 2) if "linefit" in tests else 1
    targs = {"thr": thr, "window": window, "deg": deg, "perr_thr": perr_thr, "pli_thr": pli_thr}
    seeds = []
    t = 0
    while True:
        nxt = np.flatnonzero(avail[t:])
        if not len(nxt):
            break
        t = t0 = t + int(nxt[0])
        seeds.append(t0)
        state = None
        lost = False
        while t < e["los"] and not lost:
            t1 = min(t + step, e["los"])
            o = fn(tgt, t, e["f"][t0] + f_err, e["rate"][t0] + r_err, t_end=t1, state=state, **kw)
            f[t:t1] = o["f"][t:t1]
            for k in aux:
                if k in o:
                    aux[k][t:t1] = o[k][t:t1]
            state, t = o["state"], t1
            seg = np.full(dur, np.nan)
            seg[t0:t] = f[t0:t]  # the test must not see the previous run
            ok, _ = lock_tests(aux, seg, tests, **targs)
            known = ok[t0 : t - delay + 1]
            if len(known) >= patience and not known[-patience:].any():
                lost = True
        if t >= e["los"]:
            break
    # final trusted set: per run, so that a line-fit window never mixes two runs
    lock, rms_all = np.zeros(dur, bool), np.full(dur, np.nan)
    for a, b in zip(seeds, [*seeds[1:], dur], strict=True):
        seg = np.full(dur, np.nan)
        seg[a:b] = f[a:b]
        ok, rms = lock_tests(aux, seg, tests, **targs)
        lock[a:b], rms_all[a:b] = ok[a:b], rms[a:b]
    return {"f": f, "lock": lock, "rms": rms_all, "seeds": np.array(seeds), **aux}


# ---------------------------------------------------------------------------------------------------------------
# the methods compared in the notebook, with their settings, and a cached runner
# ---------------------------------------------------------------------------------------------------------------

# C/N0 at which 0_detection first reports each kind of signal: the hand-off level used in run_method
HANDOFF_CN0 = {"cw": 12.0, "sdpsk": 25.0, "oqpsk": 35.0}

# power used for each modulation: 1 for a carrier, 2 (square) for the PSK family (section 6 of the notebook)
POWER = {"cw": 1, "sdpsk": 2, "oqpsk": 2}

METHODS = {
    # dopplerguesser's loop, unchanged, + supervisor (line fit and mean |phase error|)
    # (line fit of degree 2: over 10 s near closest approach the true curve is up to 4 Hz rms from a straight line)
    "pll": {"fn": "run_pll_classic", "kw": {"bw": 100.0}, "tests": ("linefit", "perr"), "deg": 2},
    # integrate-then-detect loop, 2nd order, Bn 5 Hz, 20 ms updates, + supervisor (line fit and PLI)
    # (after a power law the detector wraps at +-1/(2p) cycle, which narrows the pull-in range: wider loop, shorter T)
    "cpll": {
        "fn": "run_pll_coh",
        "kw": {"bn": 5.0, "t_coh": 0.02, "order": 2},
        "kw_mod": {"bn": 20.0, "t_coh": 0.005},
        "tests": ("linefit", "pli"),
        "deg": 2,
    },
    # FFT tracker, 1 s blocks
    "fft": {"fn": "fft_track", "kw": {"t_blk": 1.0}},
}


def run_method(tgt, method, blank=False, cache=True, **over):
    """Run one of METHODS on a target over its whole pass, with detector-like hand-offs (from the first second the
    signal is at HANDOFF_CN0 for its kind, the level at which 0_detection first reports it):
    carriers: frequency within 5 Hz and rate within 1 Hz/s (what 0_detection measures);
    modulated: the PLLs get the same refined hand-off; the FFT tracker gets only what the modulated-signal detector
    gives (frequency within ~200 Hz, no rate) and acquires by itself. The FFT tracker re-acquires by itself after
    a loss, but gives up after 30 s without a measurement and then waits for a new hand-off, like the PLLs.
    blank=True applies `burst_mask` to the channel. Results are cached in out/<scenario>/trk_cache/.
    Returns f, trusted (bool per second), seeds, cpu_s."""
    import pickle
    import time as _time

    m = METHODS[method]
    p = POWER[tgt["kind"]]
    kw = {**m["kw"], **(m.get("kw_mod", {}) if p > 1 else {}), **over}
    deg = f"d{m['deg']}_" if "deg" in m else ""
    tag = f"{method}_{tgt['key'].replace('/', '-')}_{'b' if blank else 'n'}_{deg}" + "_".join(
        f"{k}{v}" for k, v in sorted(kw.items())
    )
    path = OUT / tgt["scenario"] / "trk_cache" / f"res_{tag}.pkl"
    if cache and path.exists():
        return pickle.loads(path.read_bytes())
    msk = burst_mask(tgt["x"], tgt["fs"]) if blank else None
    t0 = _time.process_time()
    if method == "fft":
        e = tgt["truth"]
        lvl = HANDOFF_CN0[tgt["kind"]]

        def rehandoff(k):  # a new detection, if the signal is detectable now
            if e["cn0"][k] < lvl or e["on"][k] < 0.999:
                return None
            return (e["f"][k] + 5.0, e["rate"][k] + 1.0) if p == 1 else (e["f"][k] + 200.0, None)

        ts = int(np.flatnonzero((e["cn0"] >= lvl) & (e["on"] >= 0.999))[0])
        f0, r0 = rehandoff(ts)
        o = fft_track(
            tgt,
            ts,
            f0,
            r0,
            f_unc=15.0 if p == 1 else 300.0,
            power=p,
            blank=msk,
            acq_timeout=30,
            rehandoff=rehandoff,
            **kw,
        )
        res = {"f": o["f"], "trusted": np.isfinite(o["f"]), "seeds": np.array(o["handoffs"]), "score": o["score"]}
    else:
        fn = globals()[m["fn"]]
        det = HANDOFF_CN0[tgt["kind"]]
        o = run_reseeded(tgt, fn, tests=m["tests"], det_cn0=det, deg=m["deg"], power=p, blank=msk, **kw)
        res = {"f": o["f"], "trusted": o["lock"], "seeds": o["seeds"]}
    res["cpu_s"] = _time.process_time() - t0
    if cache:
        path.write_bytes(pickle.dumps(res))
    return res


def cached(name, fn):
    """fn() once; afterwards its result from out/trk_results/<name>.pkl (delete the file to recompute)."""
    import pickle

    path = OUT / "trk_results" / f"{name}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    path.parent.mkdir(exist_ok=True)
    res = fn()
    path.write_bytes(pickle.dumps(res))
    return res


def steady_curve(name, tgts, fn, edges, tol=5.0, cn0_min=8.0, **kw):
    """Cached good_by_cn0 over run_segments of every target: the fraction of seconds within tol of the truth per
    true-C/N0 bin, with a good start in every 60 s segment (the loop's tracking ability, without acquisition)."""
    return cached(
        f"steady_{name}",
        lambda: good_by_cn0([(t, run_segments(t, fn, cn0_min=cn0_min, **kw)) for t in tgts], edges, tol),
    )
