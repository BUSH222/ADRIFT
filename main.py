"""
Extract narrow, continuous, drifting carriers ("ridges") from a huge waterfall PNG
without loading the whole image into RAM, and plot every ridge found.

Image layout assumed: x = FFT bin (frequency), y = time (1 row = 1 s).

Method (two sequential passes over the file, only ~chunk rows in RAM at a time):
  Pass 1 - detection
    1. read the PNG top->bottom in chunks (pyvips sequential access), keeping a
       small row overlap ("halo") so smoothing has no seams at chunk edges
    2. flatten: subtract per-row median (gain/AGC wander) and per-column median
       (stationary RFI, DC spike, fixed spurs)
    3. light Gaussian smoothing (more along time than frequency: a LEO track moves
       only ~0.03-0.35 px per row, so smoothing along time is nearly free SNR)
    4. normalise to robust sigma (MAD), pick local maxima along frequency in every
       row, refine to sub-bin with a parabola (= ridge centre line)
    5. link peaks row to row with a gated nearest-neighbour tracker with hysteresis
       (strong peaks start a track, weaker ones may continue it) and gap coasting
    6. keep tracks that are long, dense and actually drift (not stationary)
  Pass 2 - plotting
    re-read the file sequentially, cut out a small window around every accepted
    track and plot it. Title = coordinates of the start of the ridge.

"""

import argparse
import csv
import os

import numpy as np
import pyvips
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

FS = 2.4e6  # sample rate / bandwidth [Hz]
FC = 137.0e6  # centre frequency [Hz]
IGNORE_RANGE_PX = [7600, 8150]  # VDL interference, ignore this range

_VIPS_DTYPE = {
    "uchar": np.uint8,
    "char": np.int8,
    "ushort": np.uint16,
    "short": np.int16,
    "uint": np.uint32,
    "int": np.int32,
    "float": np.float32,
    "double": np.float64,
}


# --------------------------------------------------------------------------- I/O
def open_image(path):
    """Open for sequential (streaming) access, first band only."""
    img = None
    for kw in ({"unlimited": True}, {}):
        try:
            img = pyvips.Image.new_from_file(path, access="sequential", **kw)
            break
        except pyvips.Error:
            continue
    if img is None:
        raise RuntimeError("pyvips could not open " + path)
    if img.bands > 1:
        img = img.extract_band(0)
    return img


def iter_chunks(path, rows):
    """Yield (y0, 2D array) top->bottom, non-overlapping. Only one chunk in RAM."""
    img = open_image(path)
    W, H = img.width, img.height
    dtype = _VIPS_DTYPE[img.format]
    for y in range(0, H, rows):
        h = min(rows, H - y)
        region = img.crop(0, y, W, h)
        arr = np.frombuffer(region.write_to_memory(), dtype=dtype).reshape(h, W)
        yield y, arr


def image_size(path):
    img = pyvips.Image.new_from_file(path)
    return img.width, img.height


# ----------------------------------------------------------------------- helpers
class Geometry:
    def __init__(self, width, flip=False):
        self.W, self.flip = width, flip
        self.bin_hz = FS / width

    def freq(self, x):
        if self.flip:
            return FC + FS / 2 - (x + 0.5) * self.bin_hz
        return FC - FS / 2 + (x + 0.5) * self.bin_hz

    def drift_hz_s(self, px_per_row):
        return -px_per_row * self.bin_hz if self.flip else px_per_row * self.bin_hz


class Track:
    __slots__ = ("cols", "last_col", "misses", "rows", "snr")

    def __init__(self, row, col, snr):
        self.rows, self.cols, self.snr = [row], [col], [snr]
        self.last_col, self.misses = col, 0

    def add(self, row, col, snr):
        self.rows.append(row)
        self.cols.append(col)
        self.snr.append(snr)
        self.last_col, self.misses = col, 0


def find_peaks(buf, a):
    """Flatten + smooth + normalise a block, return per-row local maxima."""
    S = buf.astype(np.float32)
    S -= np.median(S, axis=1, keepdims=True)  # per-row (gain wander)
    S -= np.median(S, axis=0, keepdims=True)  # per-column (stationary stuff)
    S = gaussian_filter1d(S, a.sigma_t, axis=0, mode="nearest")
    S = gaussian_filter1d(S, a.sigma_f, axis=1, mode="nearest")
    sub = S[::4, ::4]
    med = np.median(sub)
    sig = 1.4826 * np.median(np.abs(sub - med)) + 1e-9
    S -= med
    S /= sig

    c = S[:, 1:-1]
    mask = (c > S[:, :-2]) & (c >= S[:, 2:]) & (c > a.snr_low)
    if a.wide_reject > 0:  # drop peaks sitting on wideband power
        bb = uniform_filter1d(S, a.wide_width, axis=1, mode="nearest")
        bsub = bb[::4, ::4]
        bmed = np.median(bsub)
        bsig = 1.4826 * np.median(np.abs(bsub - bmed)) + 1e-9
        mask &= (bb[:, 1:-1] - bmed) < a.wide_reject * bsig
    r, k = np.nonzero(mask)  # row-major -> sorted by row
    k += 1
    ctr = S[r, k]
    left, right = S[r, k - 1], S[r, k + 1]
    den = left - 2 * ctr + right
    off = np.where(
        den < -1e-6,
        0.5 * (left - right) / np.minimum(den, -1e-6),
        0.0,
    )
    cols = k + np.clip(off, -0.5, 0.5)

    lo, hi = IGNORE_RANGE_PX
    keep = (cols < lo) | (cols > hi)

    return r[keep], cols[keep], ctr[keep]


class Tracker:
    def __init__(self, a, geom):
        self.a, self.geom = a, geom
        self.active, self.accepted = [], []

    def step(self, row, cols, snrs):
        a = self.a
        if len(cols) > a.max_peaks:  # protect against wideband junk
            top = np.argsort(snrs)[-a.max_peaks :]
            cols, snrs = cols[top], snrs[top]
        used = np.zeros(len(cols), bool)
        matched = np.zeros(len(self.active), bool)

        if self.active and len(cols):
            tc = np.array([t.last_col for t in self.active])
            gate = a.gate + 0.5 * np.array([t.misses for t in self.active])
            D = np.abs(tc[:, None] - cols[None, :])
            ti, pj = np.nonzero(gate[:, None] >= D)
            for q in np.argsort(D[ti, pj]):  # greedy, closest pairs first
                i, j = ti[q], pj[q]
                if matched[i] or used[j]:
                    continue
                matched[i] = used[j] = True
                self.active[i].add(row, cols[j], snrs[j])

        keep = []
        for i, t in enumerate(self.active):
            if matched[i]:
                keep.append(t)
            else:
                t.misses += 1
                if t.misses > a.max_gap:
                    self.finish(t)
                else:
                    keep.append(t)
        for j in np.nonzero(~used & (snrs >= a.snr))[0]:  # hysteresis: only strong start
            keep.append(Track(row, cols[j], snrs[j]))
        self.active = keep

    def finish(self, t):
        if len(t.rows) >= self.a.frag_min:  # keep fragments, filter after merging
            self.accepted.append({"rows": np.asarray(t.rows), "cols": np.asarray(t.cols), "snr": np.asarray(t.snr)})

    def flush(self):
        for t in self.active:
            self.finish(t)
        self.active = []


def merge_fragments(frags, a):
    """Chain fragments separated by short dropouts if they line up when extrapolated."""
    frags.sort(key=lambda t: t["rows"][0])
    chains, open_ = [], []
    for f in frags:
        r0 = f["rows"][0]
        open_ = [c for c in open_ if c["rows"][-1] + a.merge_gap >= r0]
        best, bd = None, 1e9
        for c in open_:
            gap = r0 - c["rows"][-1]
            if gap <= 0:
                continue
            n = min(len(c["rows"]), 20)
            sl = np.polyfit(c["rows"][-n:], c["cols"][-n:], 1)[0] if n >= 5 else 0.0
            d = abs(c["cols"][-1] + sl * gap - f["cols"][0])
            if d <= a.gate + 0.15 * gap and d < bd:
                best, bd = c, d
        if best is None:
            c = {k: v.copy() for k, v in f.items()}
            chains.append(c)
            open_.append(c)
        else:
            for k in ("rows", "cols", "snr"):
                best[k] = np.concatenate([best[k], f[k]])
    return chains


def accept(t, a, geom):
    rows, cols = t["rows"], t["cols"]
    length = int(rows[-1] - rows[0] + 1)
    if length < a.min_len or len(rows) < a.fill * length:
        return False
    slope = np.polyfit(rows, cols, 1)[0]  # px / row
    if abs(slope * length) < a.min_excursion:
        return False  # stationary -> terrestrial
    d = geom.drift_hz_s(slope)
    if (a.drift == "neg" and d >= 0) or (a.drift == "pos" and d <= 0):
        return False
    t["slope"], t["length"] = slope, length
    return True


def detect(path, a, geom, H):
    halo = int(np.ceil(4 * a.sigma_t))
    tracker = Tracker(a, geom)
    tail, b0 = None, 0
    for y0, chunk in iter_chunks(path, a.chunk):
        buf = chunk if tail is None else np.vstack([tail, chunk])
        last = (y0 + len(chunk)) >= H
        lo = 0 if b0 == 0 else b0 + halo
        hi = H if last else b0 + len(buf) - halo

        r, c, s = find_peaks(buf, a)
        r = r + b0
        sel = (r >= lo) & (r < hi)
        r, c, s = r[sel], c[sel], s[sel]
        bounds = np.searchsorted(r, np.arange(lo, hi + 1))
        for i, row in enumerate(range(lo, hi)):
            tracker.step(row, c[bounds[i] : bounds[i + 1]], s[bounds[i] : bounds[i + 1]])

        tail = buf[-2 * halo :].copy()
        b0 += len(buf) - 2 * halo
        print(
            f"  pass 1: rows up to {min(hi, H)}/{H}, active={len(tracker.active)}, accepted={len(tracker.accepted)}",
            flush=True,
        )
    tracker.flush()
    chains = merge_fragments(tracker.accepted, a)
    return [t for t in chains if accept(t, a, geom)]


# ------------------------------------------------------------------------ plotting
def plot_track(tid, t, crop, box, geom, a):
    import matplotlib.pyplot as plt

    r0, r1, c0, c1 = box
    x0, y0 = t["cols"][0], int(t["rows"][0])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [3, 2]})
    lo, hi = np.percentile(crop, [1, 99.7])
    ax.imshow(crop, cmap="gray", aspect="auto", vmin=lo, vmax=hi, extent=[c0, c1, r1 + 1, r0], interpolation="nearest")
    ax.plot(t["cols"] + 0.5, t["rows"] + 0.5, "r-", lw=0.8, alpha=0.6)
    ax.plot(x0 + 0.5, y0 + 0.5, "co", ms=6)
    ax.set_xlabel("FFT bin (x)")
    ax.set_ylabel("time row (y) [s]")
    ax.set_title(f"Ridge start: x = {x0:.1f} px, y = {y0} px   ({geom.freq(x0) / 1e6:.4f} MHz, t = {y0} s)")

    ax2.plot(t["rows"], geom.freq(t["cols"]) / 1e6 - geom.freq(x0) / 1e6, "r.", ms=3)
    ax2.set_xlabel("time row (y) [s]")
    ax2.set_ylabel("f - f_start [MHz]")
    ax2.set_title(
        f"len {t['length']} s, drift {geom.drift_hz_s(t['slope']):+.2f} Hz/s, mean SNR {t['snr'].mean():.1f}",
        fontsize=9,
    )
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"ridge_{tid:04d}_x{x0:.0f}_y{y0}.png"), dpi=110)
    if a.show:
        plt.show()
    plt.close(fig)


def plot_all(path, tracks, a, geom, H):
    """Second sequential pass: cut a window around each track and plot it."""
    boxes = {}
    for tid, t in enumerate(tracks):
        r0 = max(0, int(t["rows"][0]) - a.pad_t)
        r1 = min(H - 1, int(t["rows"][-1]) + a.pad_t)
        c0 = max(0, int(np.floor(t["cols"].min())) - a.pad_f)
        c1 = min(geom.W, int(np.ceil(t["cols"].max())) + a.pad_f + 1)
        boxes[tid] = (r0, r1, c0, c1)
    pieces = {tid: [] for tid in boxes}
    pending = set(boxes)
    for y0, chunk in iter_chunks(path, 2048):
        y1 = y0 + len(chunk)
        for tid in sorted(pending):
            r0, r1, c0, c1 = boxes[tid]
            lo, hi = max(r0, y0), min(r1 + 1, y1)
            if lo < hi:
                pieces[tid].append(chunk[lo - y0 : hi - y0, c0:c1].astype(np.float32))
            if r1 < y1:  # this track is complete
                plot_track(tid, tracks[tid], np.vstack(pieces[tid]), boxes[tid], geom, a)
                del pieces[tid]
                pending.discard(tid)
        print(f"  pass 2: rows up to {y1}/{H}, plots left={len(pending)}", flush=True)


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("png")
    p.add_argument("--out", default="ridges", help="output directory")
    p.add_argument("--chunk", type=int, default=2048, help="rows per chunk (RAM ~ chunk*W*12 B)")
    p.add_argument("--sigma-t", type=float, default=1.5, help="smoothing along time [rows]")
    p.add_argument("--sigma-f", type=float, default=0.8, help="smoothing along freq [bins]")
    p.add_argument("--snr", type=float, default=12, help="peak SNR (sigma) needed to START a track")
    p.add_argument("--snr-low", type=float, default=10.0, help="peak SNR needed to CONTINUE a track")
    p.add_argument("--gate", type=float, default=2.0, help="max bin jump between rows")
    p.add_argument("--max-gap", type=int, default=3, help="rows a track may coast without a hit")
    p.add_argument("--min-len", type=int, default=30, help="min track duration [rows]")
    p.add_argument("--fill", type=float, default=0.6, help="min fraction of rows with a hit")
    p.add_argument(
        "--min-excursion", type=float, default=1.0, help="min total frequency change [bins]; rejects stationary RFI"
    )
    p.add_argument(
        "--drift",
        choices=["any", "neg", "pos"],
        default="any",
        help="keep only this drift sign (satellites: falling freq = 'neg' if low freq is on the left)",
    )
    p.add_argument("--frag-min", type=int, default=8, help="min fragment length kept before merging")
    p.add_argument("--merge-gap", type=int, default=25, help="max dropout [rows] bridged when merging")
    p.add_argument(
        "--wide-reject", type=float, default=6.0, help="reject peaks on wideband power above this many sigma (0 = off)"
    )
    p.add_argument("--wide-width", type=int, default=41, help="box width [bins] for wideband estimate")
    p.add_argument("--max-peaks", type=int, default=400, help="cap on peaks used per row")
    p.add_argument("--flip-freq", action="store_true", help="x=0 is the HIGH frequency edge")
    p.add_argument("--pad-t", type=int, default=20)
    p.add_argument("--pad-f", type=int, default=30)
    p.add_argument("--show", action="store_true", help="open each plot interactively")
    p.add_argument("--max-plots", type=int, default=500)
    a = p.parse_args()

    import matplotlib

    if not a.show:
        matplotlib.use("Agg")
    os.makedirs(a.out, exist_ok=True)

    W, H = image_size(a.png)
    geom = Geometry(W, a.flip_freq)
    print(f"image {W} x {H}, bin = {geom.bin_hz:.2f} Hz, chunk = {a.chunk} rows")

    tracks = detect(a.png, a, geom, H)
    print(f"{len(tracks)} ridges accepted")
    if len(tracks) > a.max_plots:
        tracks = sorted(tracks, key=lambda t: -t["snr"].mean() * t["length"])[: a.max_plots]
        tracks.sort(key=lambda t: t["rows"][0])
        print(f"plotting the {a.max_plots} strongest")

    with open(os.path.join(a.out, "ridges.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "start_x_px",
                "start_y_px",
                "start_freq_mhz",
                "end_x_px",
                "end_y_px",
                "length_s",
                "drift_hz_s",
                "mean_snr",
            ]
        )
        for i, t in enumerate(tracks):
            w.writerow(
                [
                    i,
                    f"{t['cols'][0]:.2f}",
                    int(t["rows"][0]),
                    f"{geom.freq(t['cols'][0]) / 1e6:.6f}",
                    f"{t['cols'][-1]:.2f}",
                    int(t["rows"][-1]),
                    t["length"],
                    f"{geom.drift_hz_s(t['slope']):.3f}",
                    f"{t['snr'].mean():.2f}",
                ]
            )
    if tracks:
        plot_all(a.png, tracks, a, geom, H)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
