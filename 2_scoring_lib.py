"""Helpers for 2_scoring.ipynb: tying tracked signals to catalog objects.

Sections
  1  elements and the candidate catalog (multi-stage filter, nearest-epoch element sets)
  2  geometry: SGP4 -> elevation and range rate at the station, pass table
  3  ASSIST truth for a campaign (scenario writer, per-day truth, compact cache)
  4  track emulator: truth -> what detection + FFT tracking would have stored
  5  evidence of one session: timing (visibility) and Doppler (robust fit with a transmitter model, against decoys
     and a freely fitted orbit)
  6  clustering by channel
  7  fusion over a campaign (mixture of transmitters) and the sequential decision rule
  8  context and Monte-Carlo workers for the notebook

Times are Unix seconds (UTC). Frequencies are Hz. Evidence is in nats (natural log) unless said otherwise.
"""

import csv
import datetime as dt
import math
import re
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from sgp4 import omm as sgp4_omm
from sgp4.api import Satrec, SatrecArray

ROOT = Path(__file__).resolve().parent
ASSIST, SCEN, OUT = ROOT / "ASSIST", ROOT / "scenarios", ROOT / "out" / "scoring"

C_LIGHT = 299_792_458.0
OMEGA_E = 7.292115e-5  # rad/s
MU_EARTH = 398600.4418  # km^3/s^2
A_WGS, F_WGS = 6378.137, 1.0 / 298.257223563  # km
F0 = 137.0e6  # nominal band centre used for Doppler scaling when nothing better is known


def unix(iso: str) -> float:
    """ISO 8601 (UTC assumed when no zone is given) -> Unix seconds."""
    d = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.UTC)
    return d.timestamp()


def iso(t: float, digits: int = 0) -> str:
    d = dt.datetime.fromtimestamp(float(t), tz=dt.UTC)
    return (
        d.strftime("%Y-%m-%dT%H:%M:%S")
        + (f".{int(d.microsecond / 1e6 * 10**digits):0{digits}d}" if digits else "")
        + "Z"
    )


# ---------------------------------------------------------------------------------------------------------------------
# 1 · Elements and the candidate catalog
# ---------------------------------------------------------------------------------------------------------------------

DEBRIS_RE = re.compile(r"\b(?:deb|r/?b)\b", re.IGNORECASE)


class ElementStore:
    """Every element set of every file, per NORAD id (GP history). `nearest` is the rule the pipeline uses:
    for a detection at time t, the set whose epoch is closest to t, not the latest one."""

    def __init__(self, paths):
        self.sets: dict[int, list[dict]] = {}
        seen = set()
        for p in paths:
            with Path(p).open(newline="") as f:
                for row in csv.DictReader(f):
                    key = (row["NORAD_CAT_ID"], row["EPOCH"])
                    if key in seen:
                        continue
                    seen.add(key)
                    row["_epoch"] = unix(row["EPOCH"])
                    self.sets.setdefault(int(row["NORAD_CAT_ID"]), []).append(row)
        for v in self.sets.values():
            v.sort(key=lambda r: r["_epoch"])

    def nearest(self, norad: int, t: float) -> dict:
        return min(self.sets[norad], key=lambda r: (abs(r["_epoch"] - t), -r["_epoch"]))

    def snapshot(self, t: float) -> list[dict]:
        """One element set per object: the one nearest t."""
        return [self.nearest(n, t) for n in sorted(self.sets)]


def filter_catalog(
    rows, constellations=("starlink", "oneweb"), min_rev_day=6.0, max_ecc=0.25, max_age_days=30.0, t_ref=None
):
    """dopplerguesser's multi-stage filter. Returns (kept rows, funnel); funnel lists the count after each stage."""
    funnel = [("all objects", len(rows))]
    rows = [r for r in rows if not any(c in r["OBJECT_NAME"].lower() for c in constellations)]
    funnel.append((f"minus {'/'.join(constellations)}", len(rows)))
    rows = [r for r in rows if not DEBRIS_RE.search(r["OBJECT_NAME"])]
    funnel.append(("minus debris and rocket bodies", len(rows)))
    rows = [r for r in rows if float(r["MEAN_MOTION"]) > min_rev_day]
    funnel.append((f"minus GEO/MEO (n <= {min_rev_day:g} rev/day)", len(rows)))
    rows = [r for r in rows if float(r["ECCENTRICITY"]) < max_ecc]
    funnel.append((f"minus HEO (e >= {max_ecc:g})", len(rows)))
    if t_ref is not None:
        rows = [r for r in rows if abs(r["_epoch"] - t_ref) <= max_age_days * 86400]
        funnel.append((f"minus element sets older than {max_age_days:g} d", len(rows)))
    return rows, funnel


def satrec_from_omm(row: dict) -> Satrec:
    s = Satrec()
    sgp4_omm.initialize(s, row)
    return s


@dataclass
class Catalog:
    """The candidate set: parallel arrays over objects, plus the SGP4 models."""

    norad: np.ndarray
    name: list[str]
    rows: list[dict]
    sats: list[Satrec] = field(repr=False, default_factory=list)
    arr: SatrecArray | None = field(repr=False, default=None)

    @classmethod
    def from_rows(cls, rows):
        sats = [satrec_from_omm(r) for r in rows]
        return cls(
            np.array([int(r["NORAD_CAT_ID"]) for r in rows]),
            [r["OBJECT_NAME"] for r in rows],
            rows,
            sats,
            SatrecArray(sats),
        )

    def __len__(self):
        return len(self.norad)

    def index(self, norad: int) -> int:
        i = np.flatnonzero(self.norad == norad)
        return int(i[0]) if i.size else -1

    def subset(self, idx) -> SatrecArray:
        return SatrecArray([self.sats[i] for i in idx])


# ---------------------------------------------------------------------------------------------------------------------
# 2 · Geometry
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Station:
    lat: float = 55.5
    lon: float = 37.1
    alt_m: float = 172.0

    @property
    def ecef_km(self) -> np.ndarray:
        la, lo, h = math.radians(self.lat), math.radians(self.lon), self.alt_m / 1000.0
        e2 = F_WGS * (2 - F_WGS)
        n = A_WGS / math.sqrt(1 - e2 * math.sin(la) ** 2)
        return np.array(
            [
                (n + h) * math.cos(la) * math.cos(lo),
                (n + h) * math.cos(la) * math.sin(lo),
                (n * (1 - e2) + h) * math.sin(la),
            ]
        )

    @property
    def up(self) -> np.ndarray:
        la, lo = math.radians(self.lat), math.radians(self.lon)
        return np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])


def gmst(t_unix):
    """Greenwich mean sidereal time (IAU 1982), radians, taking UT1 = UTC (error < 0.9 s of rotation)."""
    d = (np.asarray(t_unix, dtype=np.float64) - 946728000.0) / 86400.0  # days from J2000.0 (UT)
    tc = d / 36525.0
    g = 67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * tc + 0.093104 * tc**2 - 6.2e-6 * tc**3  # seconds
    return np.mod(g, 86400.0) * (2 * np.pi / 86400.0)


def jd_fr(t_unix):
    t = np.asarray(t_unix, dtype=np.float64)
    whole = np.floor(t / 86400.0)
    return whole + 2440587.5, t / 86400.0 - whole


def topocentric(arr: SatrecArray, station: Station, t_unix, want_el=True):
    """Elevation [deg] and range rate [m/s] of every satellite at every time: arrays (n_sat, n_t).

    SGP4 gives TEME; the station is rotated into the same frame by GMST (no polar motion, UT1 = UTC), which is good to
    a few hundred metres, i.e. well under 1 Hz of Doppler at 137 MHz (checked against ASSIST's truth in the
    notebook)."""
    t = np.atleast_1d(np.asarray(t_unix, dtype=np.float64))
    jd, fr = jd_fr(t)
    err, r, v = arr.sgp4(jd, fr)
    g = gmst(t)
    cg, sg = np.cos(g), np.sin(g)
    p = station.ecef_km
    rs = np.stack([cg * p[0] - sg * p[1], sg * p[0] + cg * p[1], np.full_like(g, p[2])], axis=-1)  # (n_t, 3)
    vs = OMEGA_E * np.stack([-rs[:, 1], rs[:, 0], np.zeros_like(g)], axis=-1)
    d = r - rs[None]
    dist = np.linalg.norm(d, axis=-1)
    rr = np.einsum("ijk,ijk->ij", d, v - vs[None]) / dist * 1000.0
    bad = err != 0
    rr[bad] = np.nan
    if not want_el:
        return None, rr
    u = station.up
    us = np.stack([cg * u[0] - sg * u[1], sg * u[0] + cg * u[1], np.full_like(g, u[2])], axis=-1)
    el = np.degrees(np.arcsin(np.clip(np.einsum("ijk,jk->ij", d, us) / dist, -1, 1)))
    el[bad] = -90.0
    return el, rr


def doppler_hz(arr: SatrecArray, station: Station, t_unix, f_hz=F0):
    """Predicted Doppler shift [Hz] of a carrier at f_hz: (n_sat, n_t)."""
    _, rr = topocentric(arr, station, t_unix, want_el=False)
    return -rr / C_LIGHT * f_hz


@dataclass
class PassTable:
    """Every pass of every catalog object over the campaign window, sorted by AOS."""

    cand: np.ndarray  # index into the catalog
    aos: np.ndarray
    los: np.ndarray
    max_el: np.ndarray
    t0: float
    t1: float
    n_cand: int

    @property
    def dur(self):
        return self.los - self.aos

    def containing(self, a: float, b: float, margin_s: float = 0.0):
        """Indices of passes that contain [a, b] (shrunk by margin_s at both ends, to forgive element error and
        refraction at the horizon)."""
        a, b = a + margin_s, b - margin_s
        if b < a:
            a = b = 0.5 * (a + b)
        hi = np.searchsorted(self.aos, a, side="right")
        lo = np.searchsorted(self.aos, a - 7200.0, side="left")  # no LEO pass at 137 MHz lasts two hours
        j = np.arange(lo, hi)
        return j[self.los[j] >= b]

    def visible_fraction(self, length_s):
        """q_H(L): the fraction of campaign time at which an interval of length L would lie wholly inside a pass of
        candidate H. Array (n_cand,) for scalar L. This is the chance that an unrelated signal of that length is
        'explained' by H's visibility; 1/q is the timing likelihood ratio."""
        w = np.maximum(self.dur - length_s, 0.0)
        return np.bincount(self.cand, weights=w, minlength=self.n_cand) / (self.t1 - self.t0)


def build_pass_table(
    cat: Catalog, station: Station, t0: float, t1: float, step_s=30.0, mask_deg=0.0, chunk=240, cache=None
):
    """Find all passes by sampling the elevation every step_s and interpolating the crossings (good to ~1 s)."""
    if cache is not None and Path(cache).exists():
        z = np.load(cache)
        if int(z["n_cand"]) == len(cat) and float(z["t0"]) == t0 and float(z["t1"]) == t1:
            return PassTable(z["cand"], z["aos"], z["los"], z["max_el"], t0, t1, len(cat))
    n = math.ceil((t1 - t0) / step_s) + 1
    t = t0 + np.arange(n) * step_s
    prev_el = None
    open_aos = np.full(len(cat), np.nan)
    open_max = np.full(len(cat), -90.0)
    out = []
    for k in range(0, n, chunk):
        tt = t[k : k + chunk]
        el, _ = topocentric(cat.arr, station, tt)
        el = el - mask_deg
        if prev_el is None:
            up0 = el[:, 0] > 0
            open_aos[up0] = tt[0]
            full = el
            tfull = tt
        else:
            full = np.concatenate([prev_el[:, None], el], axis=1)
            tfull = np.concatenate([[tt[0] - step_s], tt])
        above = full > 0
        rise = np.argwhere(~above[:, :-1] & above[:, 1:])
        fall = np.argwhere(above[:, :-1] & ~above[:, 1:])
        ev = [(i, j, 0) for i, j in rise] + [(i, j, 1) for i, j in fall]
        ev.sort(key=lambda e: (e[0], e[1], e[2]))
        # running max per object over this chunk, split at events (cheap: handle per event)
        last_j = {}
        for i, j, kind in ev:
            frac = full[i, j] / (full[i, j] - full[i, j + 1])
            tc = tfull[j] + frac * step_s
            if kind == 0:
                open_aos[i] = tc
                open_max[i] = -90.0
                last_j[i] = j + 1
            else:
                j0 = last_j.get(i, 0)
                m = max(open_max[i], float(full[i, j0 : j + 1].max()))
                if not np.isnan(open_aos[i]):
                    out.append((i, open_aos[i], tc, m + mask_deg))
                open_aos[i] = np.nan
                open_max[i] = -90.0
                last_j[i] = j + 1
        still = np.flatnonzero(~np.isnan(open_aos))
        for i in still:
            j0 = last_j.get(i, 0)
            open_max[i] = max(open_max[i], float(full[i, j0:].max()))
        prev_el = el[:, -1]
    for i in np.flatnonzero(~np.isnan(open_aos)):
        out.append((i, open_aos[i], t1, open_max[i] + mask_deg))
    a = np.array(out)
    order = np.argsort(a[:, 1], kind="stable")
    pt = PassTable(a[order, 0].astype(np.int32), a[order, 1], a[order, 2], a[order, 3], t0, t1, len(cat))
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, cand=pt.cand, aos=pt.aos, los=pt.los, max_el=pt.max_el, t0=t0, t1=t1, n_cand=len(cat))
    return pt


def sun_unit_vector(t_unix):
    """Low-precision direction to the Sun in the equatorial frame of date (good to ~0.01 deg): (n_t, 3)."""
    n = (np.asarray(t_unix, dtype=np.float64) - 946728000.0) / 86400.0
    lam_mean = np.radians((280.460 + 0.9856474 * n) % 360.0)
    g = np.radians((357.528 + 0.9856003 * n) % 360.0)
    lam = lam_mean + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 4e-7 * n)
    return np.stack([np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)], axis=-1)


def sunlit(sat: Satrec, t_unix):
    """True where the satellite is outside the Earth's shadow (cylindrical umbra)."""
    t = np.atleast_1d(np.asarray(t_unix, dtype=np.float64))
    jd, fr = jd_fr(t)
    _, r, _ = sat.sgp4_array(jd, fr)
    s = sun_unit_vector(t)
    along = np.einsum("ij,ij->i", r, s)
    perp = np.linalg.norm(r - along[:, None] * s, axis=1)
    return (along > 0) | (perp > 6378.137)


# ---------------------------------------------------------------------------------------------------------------------
# 3 · ASSIST truth for a campaign
# ---------------------------------------------------------------------------------------------------------------------

TRUTH_CN0_ZENITH = 40.0  # level every pool transmitter is rendered at; the emulator shifts it per transmitter
FADE_S = 15.0  # ASSIST's horizon fade: in the IQ but not in the truth's C/N0, so it is added here


def write_campaign_scenario(path, norads, t0_iso, days, perturb, station: Station, elements):
    """One continuous carrier per pool object, all at 137.000 MHz (Doppler is rescaled to the real channel later).
    `perturb[norad]` = (along_track_s, raan_deg): the TRUE orbit differs from the catalog entry by this much."""
    lines = [
        "# Scoring campaign: a pool of transmitters on catalog orbits, used through `assist truth` only (no IQ).",
        "# Written by 2_scoring_lib.write_campaign_scenario; keying, oscillator behaviour and measurement errors are",
        "# added at track level by the emulator in 2_scoring_lib.",
        "version: 1",
        "seed: 2",
        f'clock: {{start: "{t0_iso}", duration: {days}d}}',
        f"station: {{lat: {station.lat}, lon: {station.lon}, alt_m: {station.alt_m:g}}}",
        "profile: profiles/vhf137",
        f"elements: {{files: [{', '.join(elements)}]}}",
        "sources:",
    ]
    for n in norads:
        a, r = perturb[n]
        lines.append(f"  - id: n{n}")
        lines.append(f"    orbit: {{norad: {n}, perturb: {{along_track_s: {a:.4f}, raan_deg: {r:.5f}}}}}")
        lines.append(
            "    emitters: [{id: c, signal: {kind: cw}, freq_hz: 137.0e6, "
            f"level: {{cn0_zenith_dbhz: {TRUTH_CN0_ZENITH:g}}}}}]"
        )
    Path(path).write_text("\n".join(lines) + "\n")


@dataclass
class TruthPass:
    """One pass of one pool transmitter as ASSIST's truth gives it, at 1 s steps."""

    norad: int
    t: np.ndarray  # whole Unix seconds
    doppler: np.ndarray  # Hz at 137.000 MHz
    el: np.ndarray  # deg
    cn0: np.ndarray  # dB-Hz at the reference level, horizon fade included

    @property
    def max_el(self):
        return float(self.el.max())


def campaign_truth(scenario: Path, days: int, cache: Path, log=print) -> dict[int, list[TruthPass]]:
    """`assist truth` day by day (bounded memory), parsed into passes and cached as one npz."""
    cache = Path(cache)
    if not cache.exists():
        tmp = cache.parent / "truth_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        cols = {k: [] for k in ("norad", "t", "doppler", "el", "cn0")}
        for d in range(days):
            f = tmp / f"day{d:02d}.csv"
            subprocess.run(
                [
                    "uv",
                    "run",
                    "assist",
                    "truth",
                    str(scenario),
                    "-o",
                    str(f),
                    "--from",
                    f"+{d}d",
                    "--to",
                    f"+{d + 1}d",
                    "--platforms",
                    "satellite",
                ],
                cwd=ASSIST,
                check=True,
                capture_output=True,
            )
            with f.open(newline="") as fh:
                rd = csv.reader(fh)
                h = next(rd)
                ix = [h.index(c) for c in ("norad", "t_utc", "doppler_hz", "el_deg", "cn0_dbhz")]
                for row in rd:
                    cols["norad"].append(int(row[ix[0]]))
                    cols["t"].append(row[ix[1]])
                    cols["doppler"].append(float(row[ix[2]]))
                    cols["el"].append(float(row[ix[3]]))
                    cols["cn0"].append(float(row[ix[4]]))
            f.unlink()
            f.with_name(f.stem + ".events.csv").unlink(missing_ok=True)
            log(f"  truth day {d + 1}/{days}: {len(cols['t'])} rows so far")
        t = np.array([s.rstrip("Z") for s in cols["t"]], dtype="datetime64[ms]").astype(np.int64) / 1000.0
        np.savez_compressed(
            cache,
            norad=np.array(cols["norad"], dtype=np.int32),
            t=t,
            doppler=np.array(cols["doppler"]),
            el=np.array(cols["el"], dtype=np.float32),
            cn0=np.array(cols["cn0"], dtype=np.float32),
        )
    z = np.load(cache)
    order = np.lexsort((z["t"], z["norad"]))
    norad, t, dop, el, cn0 = (z[k][order] for k in ("norad", "t", "doppler", "el", "cn0"))
    cut = np.flatnonzero((np.diff(norad) != 0) | (np.diff(t) > 1.5)) + 1
    out: dict[int, list[TruthPass]] = {}
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(t)], strict=True):
        tt = t[a:b]
        # horizon fade (raised cosine in amplitude) that ASSIST applies to the IQ
        edge = np.minimum(tt - tt[0], tt[-1] - tt)
        amp = np.where(edge < FADE_S, 0.5 * (1 - np.cos(np.pi * np.clip(edge, 0, FADE_S) / FADE_S)), 1.0)
        fade = 20 * np.log10(np.maximum(amp, 1e-6))
        out.setdefault(int(norad[a]), []).append(
            TruthPass(int(norad[a]), tt, dop[a:b], el[a:b].astype(float), cn0[a:b].astype(float) + fade)
        )
    return out


# ---------------------------------------------------------------------------------------------------------------------
# 4 · Track emulator: ASSIST truth -> stored detections
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Osc:
    """Transmitter oscillator behaviour, injected at track level. All sigmas are of normal draws."""

    pass_sigma: float = 10.0  # rest-frequency change from pass to pass [Hz]
    lin_drift: float = 0.0  # linear in-pass drift [Hz/s], drawn per pass
    thermal_amp: float = 0.0  # A in A (1 - exp(-t/tau)) after every turn-on [Hz]
    thermal_tau: tuple[float, float] = (100.0, 400.0)  # tau, uniform [s]
    walk: float = 0.0  # random walk [Hz/sqrt(s)]
    jump_rate: float = 0.0  # frequency jumps per second (ALOUETTE-2 style)
    jump_size: tuple[float, float] = (100.0, 300.0)  # |jump|, uniform [Hz], random sign
    restart_sigma: float = 0.0  # offset drawn at every turn-on inside a pass [Hz]
    droop_amp: float = 0.0  # frequency excursion at the moment of turn-off [Hz]; sign fixed per transmitter
    droop_tau: tuple[float, float] = (4.0, 15.0)  # fast and slow time constants of the droop (droop.png) [s]


@dataclass(frozen=True)
class TxProfile:
    """What a transmitter does. `keying` = (on_s, off_s) periodic; `bursts` = (mean_gap_s, dur_lo_s, dur_hi_s)."""

    name: str
    cn0_zenith: float = 35.0
    keying: tuple[float, float] | None = None
    bursts: tuple[float, float, float] | None = None
    p_pass: float = 1.0  # probability of being on at all during a pass (silent passes)
    sunlit_only: bool = False
    osc: Osc = Osc()


PROFILES = {
    "beacon": TxProfile("beacon", cn0_zenith=40.0, osc=Osc(pass_sigma=5.0, lin_drift=0.01)),
    "zombie": TxProfile(
        "zombie", cn0_zenith=33.0, p_pass=0.8, sunlit_only=True,
        osc=Osc(pass_sigma=80.0, lin_drift=0.05, thermal_amp=80.0, droop_amp=350.0),
    ),
    "keyed": TxProfile(
        "keyed", cn0_zenith=35.0, keying=(40.0, 180.0), p_pass=0.8, sunlit_only=True,
        osc=Osc(pass_sigma=80.0, lin_drift=0.05, thermal_amp=60.0, restart_sigma=30.0, droop_amp=350.0),
    ),
    "wild": TxProfile(
        "wild", cn0_zenith=32.0, p_pass=0.8, sunlit_only=True,
        osc=Osc(pass_sigma=300.0, lin_drift=0.1, thermal_amp=100.0, walk=1.5, jump_rate=1 / 120.0, droop_amp=350.0),
    ),
    "burster": TxProfile(
        "burster", cn0_zenith=33.0, bursts=(240.0, 32.0, 45.0), osc=Osc(pass_sigma=30.0, lin_drift=0.03)
    ),
    "weak": TxProfile(
        "weak", cn0_zenith=20.0, p_pass=0.9, sunlit_only=True,
        osc=Osc(pass_sigma=80.0, thermal_amp=60.0, droop_amp=350.0),
    ),
}  # fmt: skip


@dataclass(frozen=True)
class TrackerModel:
    """What 0_detection + the FFT tracker of 1_tracking do to a signal, as measured in those notebooks."""

    det_cn0: float = 12.0  # dB-Hz from which the detector declares
    det_latency: tuple[float, float] = (12.0, 17.0)  # s of signal before a detection is declared
    trk_cn0: float = 11.5  # 50 % point of the per-second acceptance curve (>= 90 % at 13 dB-Hz)
    trk_slope: float = 0.7  # dB
    max_gap: float = 30.0  # s without a measurement before the tracker gives up
    window_hz: float = 6.0  # tracking search window: a larger jump loses the signal
    droop_rate: tuple[float, float] = (10.0, 40.0)  # the tracker follows a droop up to this rate [Hz/s]
    p_outlier: float = 0.02  # isolated bad bins
    gross_fraction: float = 0.3  # of those, the share that is gross (else inside the search window)
    gross_hz: tuple[float, float] = (20.0, 300.0)
    run_rate: float = 1 / 900.0  # per second: start of a run of bad bins (tracker follows something else)
    run_len: float = 6.0  # mean length of a run [s]
    run_hz: float = 40.0  # run offset, uniform +-
    min_len: int = 6  # stored tracks have at least this many seconds


TRACKER = TrackerModel()


@dataclass
class Detection:
    """What the system stores per detection: start/end time, centre frequency and the 1 s bins of the track."""

    t: np.ndarray
    f: np.ndarray
    cn0: np.ndarray
    truth: int = -1  # NORAD id of the transmitter (-1: not a satellite); for scoring the method only
    kind: str = ""
    bad: np.ndarray | None = None  # truth mask of injected bad bins

    @property
    def t0(self):
        return float(self.t[0])

    @property
    def t1(self):
        return float(self.t[-1])

    @property
    def fc(self):
        return float(np.median(self.f))


def _on_mask(n, prof: TxProfile, rng):
    on = np.ones(n, dtype=bool)
    if prof.keying is not None:
        on_s, off_s = prof.keying
        ph = rng.uniform(0, on_s + off_s)
        on &= ((np.arange(n) + ph) % (on_s + off_s)) < on_s
    if prof.bursts is not None:
        gap, lo, hi = prof.bursts
        on[:] = False
        k = rng.exponential(gap) - gap  # may start mid-burst
        while k < n:
            d = rng.uniform(lo, hi)
            on[max(0, int(k)) : max(0, int(k + d))] = True
            k += d + rng.exponential(gap)
    return on


def emulate_pass(
    tp: TruthPass, prof: TxProfile, f_rest: float, rng, trk: TrackerModel = TRACKER, lit=None, droop_sign=1.0
):
    """Stored detections (possibly several) of one pass. `lit`: sunlit mask of the pass (sunlit_only transmitters)."""
    n = len(tp.t)
    if rng.random() > prof.p_pass:
        return []
    on = _on_mask(n, prof, rng)
    if prof.sunlit_only and lit is not None:
        on &= lit
    if not on.any():
        return []
    o = prof.osc
    # transmitter frequency offset from f_rest
    df = np.full(n, rng.normal(0, o.pass_sigma)) + rng.normal(0, o.lin_drift) * np.arange(n)
    if o.walk > 0:
        df += np.cumsum(rng.normal(0, o.walk, n))
    jump = np.zeros(n, dtype=bool)
    if o.jump_rate > 0:
        jump = rng.random(n) < o.jump_rate
        df += np.cumsum(np.where(jump, rng.uniform(*o.jump_size, n) * rng.choice([-1.0, 1.0], n), 0.0))
    edges = np.flatnonzero(np.diff(np.r_[False, on, False].astype(np.int8)))
    follow = np.ones(n, dtype=bool)  # False where a droop is too fast for the tracker
    r_max = rng.uniform(*trk.droop_rate)
    for a, b in zip(edges[::2], edges[1::2], strict=True):  # on-periods [a, b)
        tt = np.arange(b - a, dtype=float)
        t_on = tt + (rng.uniform(0, 600.0) if a == 0 else 0.0)  # already on before AOS: warmed up for a while
        df[a:b] += rng.normal(0, o.restart_sigma) if a > 0 else 0.0
        if o.thermal_amp > 0:
            df[a:b] += rng.normal(0, o.thermal_amp) * (1 - np.exp(-t_on / rng.uniform(*o.thermal_tau)))
        if o.droop_amp > 0 and b < n:  # switched off by the transmitter, not by the horizon
            back = (b - a) - tt  # seconds until turn-off
            fast, slow = o.droop_tau
            amp = droop_sign * o.droop_amp * rng.uniform(0.6, 1.2)
            df[a:b] += amp * (0.85 * np.exp(-back / fast) + 0.15 * np.exp(-back / slow))
            rate = abs(amp) * (0.85 / fast * np.exp(-back / fast) + 0.15 / slow * np.exp(-back / slow))
            follow[a:b] = rate <= r_max
    f_tx = f_rest + df
    f_rx = f_tx + tp.doppler * (f_tx / F0)
    cn0 = tp.cn0 + (prof.cn0_zenith - TRUTH_CN0_ZENITH)
    lin = 10 ** (cn0 / 10)
    sigma = np.maximum(
        np.sqrt(3.0 / (2 * np.pi**2 * lin)), 0.02
    )  # Cramer-Rao bound of a 1 s block, as measured in 1_tracking
    p_acc = 1 / (1 + np.exp(-(cn0 - trk.trk_cn0) / trk.trk_slope))
    acc = on & (rng.random(n) < p_acc) & follow
    detectable = on & (cn0 >= trk.det_cn0) & follow
    out = []
    i = 0
    while i < n:
        # wait for a detection: `lat` consecutive detectable seconds
        lat = round(rng.uniform(*trk.det_latency))
        run = 0
        while i < n and run < lat:
            run = run + 1 if detectable[i] else 0
            i += 1
        if run < lat:
            break
        # track from here until the signal is lost
        idx, last = [], i
        while i < n and i - last <= trk.max_gap:
            if jump[i] and idx:  # a jump larger than the search window: lost, needs a new detection
                i += 1
                break
            if acc[i]:
                idx.append(i)
                last = i
            i += 1
        if len(idx) >= trk.min_len:
            idx = np.array(idx)
            f = f_rx[idx] + rng.normal(0, sigma[idx])
            bad = np.zeros(len(idx), dtype=bool)
            u = rng.random(len(idx))
            iso_bad = u < trk.p_outlier
            gross = iso_bad & (rng.random(len(idx)) < trk.gross_fraction)
            f[iso_bad & ~gross] += rng.uniform(-trk.window_hz, trk.window_hz, int((iso_bad & ~gross).sum()))
            f[gross] += rng.uniform(*trk.gross_hz, int(gross.sum())) * rng.choice([-1.0, 1.0], int(gross.sum()))
            bad |= iso_bad
            for s in np.flatnonzero(rng.random(len(idx)) < trk.run_rate):
                m = slice(s, s + 1 + rng.geometric(1 / trk.run_len))
                k = len(f[m])
                f[m] += rng.uniform(-trk.run_hz, trk.run_hz) + rng.uniform(-3, 3) * np.arange(k)
                bad[m] = True
            out.append(
                Detection(
                    tp.t[idx].astype(float), f, cn0[idx] + rng.normal(0, 1.0, len(idx)), tp.norad, prof.name, bad
                )
            )
    return out


def ground_carrier(t0, t1, f_rest, rng, n_per_day=3.0, dur=(60.0, 900.0), drift=0.02, kind="ground"):
    """Detections of a non-satellite carrier (a drifting spur or a ground transmitter): flat with a slow drift."""
    out = []
    for _ in range(rng.poisson(n_per_day * (t1 - t0) / 86400.0)):
        a = np.floor(rng.uniform(t0, t1 - dur[1]))
        n = int(rng.uniform(*dur))
        tt = a + np.arange(n)
        f = f_rest + rng.normal(0, 20.0) + rng.normal(0, drift) * np.arange(n) + rng.normal(0, 0.05, n)
        out.append(Detection(tt, f, np.full(n, 30.0), -1, kind, np.zeros(n, dtype=bool)))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# 5 · Fine pipeline: Doppler evidence per session
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Session:
    """The unit of evidence: all detections on one channel that belong to one pass (gaps shorter than `gap`).
    A keyed transmitter gives several detections per pass; they share the pass geometry, so they are one
    observation."""

    dets: list[Detection]

    def __post_init__(self):
        self.t = np.concatenate([d.t for d in self.dets])
        self.f = np.concatenate([d.f for d in self.dets])
        self.seg = np.concatenate([np.full(len(d.t), i) for i, d in enumerate(self.dets)])
        self.t0, self.t1 = float(self.t.min()), float(self.t.max())  # detections of two transmitters may overlap
        self.fc = float(np.median(self.f))

    @property
    def span(self):
        return self.t1 - self.t0

    @property
    def truth(self):
        return self.dets[0].truth


def make_sessions(dets: list[Detection], gap=400.0, max_span=2400.0) -> list[Session]:
    out, cur, end = [], [], -np.inf
    for d in sorted(dets, key=lambda d: d.t0):
        if cur and (d.t0 - end > gap or d.t1 - cur[0].t0 > max_span):
            out.append(Session(cur))
            cur, end = [], -np.inf
        cur.append(d)
        end = max(end, d.t1)
    if cur:
        out.append(Session(cur))
    return out


@dataclass(frozen=True)
class FitConfig:
    loss: str = "t"  # "t" (Student-t), "ls" (least squares)
    nu: float = 3.0
    sigma_floor: float = (
        2.0  # Hz. Model error that is tolerated (element sets, tracker): misfits below it are not evidence
    )
    drift_order: int = 1  # transmitter drift polynomial across the session (0: offset only)
    sigma_drift: float = 0.3  # Hz/s, prior sigma of the linear drift (and Hz/s^2 * 100 s for the quadratic one)
    sigma_seg: float = (
        150.0  # Hz, prior sigma of an offset step between segments (turn-on, jump). 0: one shared offset
    )
    droop_tau: tuple[float, ...] = (
        5.0,
        20.0,
    )  # s. Per detection, free tails exp(-(t_end - t)/tau): the turn-off droop
    sigma_droop: tuple[float, ...] = (300.0, 60.0)  # Hz, prior sigma of each tail's amplitude
    wander_s: float = 120.0  # s. Smooth wander of the rest frequency: one cosine term per this much session. 0: off
    sigma_wander: float = 20.0  # Hz, prior sigma of each wander term
    sigma_flat: float = 0.05  # Hz/s, drift allowed to a signal without Doppler (ground carrier, spur)
    nonsat_min_span: float = 120.0  # s: shorter sessions cannot count as evidence for 'not a satellite'
    sigma_along: float = 0.5  # s, prior sigma of the along-track (timing) error of an element set. 0: none allowed
    free_orbit: bool = True  # fit an orbit of its own to the track, as the reference 'a satellite that is not H'
    free_starts: int = 2  # ... started from this many of the best candidates and of the best decoys
    free_nfev: int = 25  # ... with at most this many Levenberg-Marquardt steps each
    orbit_bonus: float = 6.2  # nats: the most a catalog orbit is preferred to an equally good free one (ln 500)
    n_max: int = 300  # tracks are thinned to at most this many bins
    iters: int = 12
    n_eff: bool = True  # scale log-likelihoods by the effective sample size of the best model's residuals
    n_decoys: int = 500
    margin_s: float = 5.0  # visibility margin at both ends of a session
    w_cap: float = 8.0  # nats: the most one session may contribute (timing and Doppler together)
    kin_window: float = 600.0  # s: a candidate with a pass this close to a session it cannot contain was 'nearly up'


FIT = FitConfig()


def _design(t, seg, cfg: FitConfig, free_poly=0, wander=True):
    """Transmitter (nuisance) design matrix and ridge precisions: offset, a step per later detection, a droop tail
    per detection, smooth wander terms, drift. Also returns the index of the first drift column."""
    tc = (t - t.mean()) / 100.0  # units of 100 s keep the columns comparable
    cols, lam = [np.ones_like(tc)], [0.0]
    if cfg.sigma_seg > 0:
        for k in np.unique(seg)[1:]:
            cols.append((seg == k).astype(float))
            lam.append(1.0 / cfg.sigma_seg**2)
    for tau, amp in zip(cfg.droop_tau, cfg.sigma_droop, strict=False):
        for k in np.unique(seg):
            in_k = seg == k
            cols.append(np.where(in_k, np.exp(-(t[in_k].max() - t) / tau), 0.0))
            lam.append(1.0 / amp**2)
    if wander and cfg.wander_s > 0:
        span = float(t.max() - t.min())
        for k in range(1, min(8, int(span // cfg.wander_s)) + 1):
            cols.append(np.cos(k * np.pi * (t - t.min()) / span))
            lam.append(1.0 / cfg.sigma_wander**2)
    n_fixed = len(cols)
    if free_poly:
        for p in range(1, free_poly + 1):
            cols.append(tc**p)
            lam.append(0.0)
    else:
        for p in range(1, cfg.drift_order + 1):
            cols.append(tc**p)
            lam.append(1.0 / (cfg.sigma_drift * 100.0) ** 2)
    return np.stack(cols, axis=1), np.array(lam), n_fixed


_T_CONST = {}


def _t_const(nu):
    if nu not in _T_CONST:
        _T_CONST[nu] = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
    return _T_CONST[nu]


def robust_fit(y, X, lam, seg, cfg: FitConfig, slope_col=None, extra=None, lam_extra=0.0):
    """Fit y[m] = X beta + noise for every row m of y, under Student-t noise with the scale profiled (floored).

    y: (M, n); X: (n, p) shared nuisance design with ridge precisions lam; extra: optional (M, n) column that differs
    per row (the along-track term), appended last with ridge precision lam_extra. Returns (loglik (M,), beta (M, p'),
    sigma (M,), resid (M, n)). Two starts (segment medians; segment medians plus a robust slope), the better one kept:
    the t loss is not convex."""
    M, n = y.shape
    nu = cfg.nu
    if extra is not None:
        Xm = np.concatenate([np.broadcast_to(X, (M, *X.shape)), extra[:, :, None]], axis=2)
        lam = np.r_[lam, lam_extra]
    else:
        Xm = np.broadcast_to(X, (M, *X.shape))
    p = Xm.shape[2]
    XmT = Xm.transpose(0, 2, 1)
    segs = np.unique(seg)
    b0 = np.zeros((M, p))
    med0 = np.median(y[:, seg == segs[0]], axis=1)
    b0[:, 0] = med0
    if cfg.sigma_seg > 0:
        for col, k in enumerate(segs[1:], start=1):
            b0[:, col] = np.median(y[:, seg == k], axis=1) - med0
    else:
        b0[:, 0] = np.median(y, axis=1)
    starts = [b0]
    if slope_col is not None and X.shape[1] > slope_col:  # second start with a robust slope
        same = seg[1:] == seg[:-1]
        dx = (X[1:, slope_col] - X[:-1, slope_col])[same]
        if same.sum() >= 3:
            b1 = b0.copy()
            b1[:, slope_col] = np.median((y[:, 1:] - y[:, :-1])[:, same] / dx, axis=1)
            b1[:, 0] += np.median(y - np.einsum("mnp,mp->mn", Xm, b1), axis=1)  # re-centre after imposing the slope
            starts.append(b1)
    best = None
    ridge = np.diag(lam)[None]
    for b in starts:
        r = y - np.einsum("mnp,mp->mn", Xm, b)
        s2 = np.maximum(
            cfg.sigma_floor**2, (1.4826 * np.median(np.abs(r - np.median(r, axis=1, keepdims=True)), axis=1)) ** 2
        )
        for _ in range(cfg.iters):
            w = np.ones_like(r) if cfg.loss == "ls" else (nu + 1) / (nu + r * r / s2[:, None])
            xw = XmT * w[:, None, :]
            A = np.matmul(xw, Xm) + s2[:, None, None] * ridge
            rhs = np.matmul(xw, y[:, :, None])[..., 0]
            b = np.linalg.solve(A, rhs[..., None])[..., 0]
            r = y - np.einsum("mnp,mp->mn", Xm, b)
            s2 = np.maximum(cfg.sigma_floor**2, np.mean(w * r * r, axis=1))
        sg = np.sqrt(s2)
        if cfg.loss == "ls":
            ll = -0.5 * np.sum(r * r, axis=1) / s2 - n * np.log(sg) - 0.5 * n * math.log(2 * math.pi)
        else:
            ll = (
                n * _t_const(nu)
                - n * np.log(sg)
                - 0.5 * (nu + 1) * np.sum(np.log1p(r * r / (nu * s2[:, None])), axis=1)
            )
        ll = ll - 0.5 * np.sum(lam[None] * b * b, axis=1)
        if best is None:
            best = [ll, b, sg, r]
        else:
            better = ll > best[0]
            best[0] = np.where(better, ll, best[0])
            best[1] = np.where(better[:, None], b, best[1])
            best[2] = np.where(better, sg, best[2])
            best[3] = np.where(better[:, None], r, best[3])
    return tuple(best)


def effective_fraction(r, s, seg, t):
    """n_eff / n from the lag-1 autocorrelation of the (clipped) residuals of one model."""
    z = np.clip(r / s, -2.5, 2.5)
    z = z - z.mean()
    adj = (seg[1:] == seg[:-1]) & (np.diff(t) <= 1.5 * np.median(np.diff(t)))
    if adj.sum() < 4 or not np.any(z):
        return 1.0
    rho = float(np.sum(z[1:][adj] * z[:-1][adj]) / max(np.sum(z * z) * adj.sum() / len(z), 1e-12))
    rho = min(max(rho, 0.0), 0.999)
    return max((1 - rho) / (1 + rho), 4.0 / len(z))


def station_teme(station: Station, t_unix):
    """Station position [km] and velocity [km/s] in the TEME frame: (n, 3) each."""
    g = gmst(t_unix)
    cg, sg = np.cos(g), np.sin(g)
    p = station.ecef_km
    rs = np.stack([cg * p[0] - sg * p[1], sg * p[0] + cg * p[1], np.full_like(g, p[2])], axis=-1)
    return rs, OMEGA_E * np.stack([-rs[:, 1], rs[:, 0], np.zeros_like(g)], axis=-1)


def kepler_doppler(state, dt, rs, vs, f_hz):
    """Doppler [Hz] of two-body orbits given by state vectors (B, 6) [km, km/s] at dt = 0, at times dt (n,), seen from
    a station at rs, vs (n, 3). Closed-form f and g functions (the propagator dopplerguesser uses), no J2: good to
    about 1 Hz over a pass, which is below the scale the fit cares about."""
    r0, v0 = state[:, :3], state[:, 3:]
    r0n = np.linalg.norm(r0, axis=1)
    alpha = np.clip(2.0 / r0n - np.sum(v0 * v0, axis=1) / MU_EARTH, 1.0 / 5.0e4, 1.0 / 6400.0)  # 1/a, kept elliptic
    a = 1.0 / alpha
    nmm = np.sqrt(MU_EARTH / a**3)
    sig0 = np.sum(r0 * v0, axis=1) / np.sqrt(MU_EARTH * a)
    c0 = 1.0 - r0n / a
    m = nmm[:, None] * dt[None]
    e = m.copy()
    for _ in range(6):
        e -= (e - c0[:, None] * np.sin(e) + sig0[:, None] * (1 - np.cos(e)) - m) / (
            1 - c0[:, None] * np.cos(e) + sig0[:, None] * np.sin(e)
        )
    ce, se = np.cos(e), np.sin(e)
    f = 1 - (a / r0n)[:, None] * (1 - ce)
    g = dt[None] - (e - se) / nmm[:, None]
    r = f[..., None] * r0[:, None] + g[..., None] * v0[:, None]
    rn = a[:, None] * (1 - c0[:, None] * ce + sig0[:, None] * se)
    fd = -np.sqrt(MU_EARTH * a)[:, None] * se / (rn * r0n[:, None])
    gd = 1 - (a[:, None] / rn) * (1 - ce)
    v = fd[..., None] * r0[:, None] + gd[..., None] * v0[:, None]
    d = r - rs[None]
    return -np.sum(d * (v - vs[None]), axis=2) / np.linalg.norm(d, axis=2) * 1000.0 / C_LIGHT * f_hz


def free_orbit_curves(t, y, X, lam, starts, station: Station, f_hz, cfg: FitConfig):
    """Doppler curves (S, n) of orbits fitted freely to the track, one per start state (S, 6) given at mean(t).

    Six orbit parameters and the transmitter terms are fitted together (Levenberg-Marquardt, soft-L1 loss); the orbit
    is held within a few hundred km of its start so that a short track cannot send it somewhere unphysical."""
    tm = t.mean()
    dt = t - tm
    rs, vs = station_teme(station, t)
    n, p = X.shape
    sf = cfg.sigma_floor
    sx = np.array([300.0, 300.0, 300.0, 0.3, 0.3, 0.3])
    h = np.array([0.05, 0.05, 0.05, 5e-5, 5e-5, 5e-5])
    sq = np.sqrt(lam)
    out = []
    for x0 in starts:
        d0 = kepler_doppler(x0[None], dt, rs, vs, f_hz)[0]
        z0 = np.r_[x0, np.zeros(p)]
        z0[6] = np.median(y - d0)

        def fun(z, x0=x0):
            d = kepler_doppler(z[None, :6], dt, rs, vs, f_hz)[0]
            return np.r_[(y - d - X @ z[6:]) / sf, sq * z[6:], (z[:6] - x0) / sx]

        def jac(z):
            d = kepler_doppler(np.concatenate([z[None, :6], z[None, :6] + np.diag(h)]), dt, rs, vs, f_hz)
            j = np.zeros((n + p + 6, 6 + p))
            j[:n, :6] = -(d[1:] - d[0]).T / h / sf
            j[:n, 6:] = -X / sf
            j[n : n + p, 6:] = np.diag(sq)
            j[n + p :, :6] = np.diag(1.0 / sx)
            return j

        try:
            res = least_squares(fun, z0, jac=jac, loss="soft_l1", f_scale=3.0, max_nfev=cfg.free_nfev, x_scale="jac")
            out.append(kepler_doppler(res.x[None, :6], dt, rs, vs, f_hz)[0])
        except ValueError, np.linalg.LinAlgError:
            out.append(d0)
    return np.array(out)


def doppler_single(sat: Satrec, station: Station, t_unix):
    t = np.asarray(t_unix, dtype=np.float64)
    jd, fr = jd_fr(t)
    _, r, v = sat.sgp4_array(jd, fr)
    rs, vs = station_teme(station, t)
    d = r - rs
    return -np.einsum("ij,ij->i", d, v - vs) / np.linalg.norm(d, axis=1) * 1000.0 / C_LIGHT * F0


class DecoyLibrary:
    """Doppler curves of random catalog passes, used as stand-ins for 'some other satellite that was up'.

    For a session, a decoy is one of these curves laid over the session at a random offset such that the satellite is
    visible throughout, exactly the condition real candidates must meet. Decoys are therefore exchangeable with real
    candidates that have nothing to do with the signal, which is what calibrates the Doppler evidence."""

    def __init__(self, cat: Catalog, pt: PassTable, station: Station, n_passes=4000, seed=0, margin_s=5.0, cache=None):
        self.margin = margin_s
        if cache is not None and Path(cache).exists():
            z = np.load(cache)
            self.data, self.start, self.length, self.cand, self.t_start = (
                z[k] for k in ("data", "start", "length", "cand", "t_start")
            )
            return
        rng = np.random.default_rng(seed)
        ok = np.flatnonzero((pt.dur > 60.0) & (pt.aos > pt.t0 + 60) & (pt.los < pt.t1 - 60))
        pick = rng.choice(ok, size=n_passes, replace=False)
        chunks, t_start = [], []
        for j in pick:  # one extra second at each end, for the time derivative
            tt = np.arange(math.floor(pt.aos[j] - margin_s) - 1.0, math.ceil(pt.los[j] + margin_s) + 2.0)
            chunks.append(doppler_single(cat.sats[pt.cand[j]], station, tt).astype(np.float32))
            t_start.append(tt[1])
        self.length = np.array([len(c) - 2 for c in chunks])  # usable samples
        self.start = np.r_[0, np.cumsum(self.length + 2)[:-1]] + 1  # index of the first usable sample
        self.data = np.concatenate(chunks)
        self.cand = pt.cand[pick].astype(np.int32)
        self.t_start = np.array(t_start)
        if cache is not None:
            np.savez(cache, data=self.data, start=self.start, length=self.length, cand=self.cand, t_start=self.t_start)

    def draw(self, rel_t, k, rng):
        """k decoys on the session's relative time grid rel_t (whole seconds from its first bin). Returns the curves
        and their time derivatives [Hz and Hz/s at F0], and for each the library pass and the time of that pass that
        is laid on the session's first bin."""
        need = int(rel_t.max()) + 1
        room = self.length - need
        ok = np.flatnonzero(room >= 0)
        if ok.size == 0:
            z = np.zeros((0, len(rel_t)))
            return z, z, np.zeros(0, dtype=int), np.zeros(0)
        wts = (room[ok] + 1).astype(float)
        j = ok[rng.choice(ok.size, size=k, p=wts / wts.sum())]
        off = rng.integers(0, room[j] + 1)
        idx = (self.start[j] + off)[:, None] + rel_t.astype(np.int64)[None]
        d = self.data[idx].astype(np.float64)
        dd = 0.5 * (self.data[idx + 1].astype(np.float64) - self.data[idx - 1])
        return d, dd, j, self.t_start[j] + off


def _state_teme(sat: Satrec, t):
    jd, fr = jd_fr(np.array([t]))
    _, r, v = sat.sgp4_array(jd, fr)
    return np.r_[r[0], v[0]]


def _rotate_z(state, ang):
    c, s_ = math.cos(ang), math.sin(ang)
    rot = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    return np.r_[rot @ state[:3], rot @ state[3:]]


@dataclass
class SessionEvidence:
    """Everything the fusion needs from one session."""

    t0: float
    t1: float
    fc: float
    cand: np.ndarray  # catalog indices of the candidates visible throughout
    log_tau: np.ndarray  # timing evidence, ln(1 / q_H(L))
    log_ell: np.ndarray  # Doppler evidence (ln), in [-inf, ln(K + 1)]
    f_rest: np.ndarray  # implied rest frequency c_H per candidate [Hz]
    sigma: np.ndarray  # residual scale per candidate [Hz]
    log_nonsat: float  # ln likelihood ratio of 'not a satellite' against 'unknown satellite'
    n: int  # bins used
    n_eff: float
    dec_eff: float  # effective number of decoys that fit about as well as the best one (K: Doppler says nothing)
    log_x: np.ndarray  # unsaturated Doppler statistic: ln p(track | H) - ln mean over decoys of p(track | decoy)
    truth: int = -1
    kind: str = ""
    ref: dict = field(default_factory=dict)  # the reference terms (tempered ln likelihoods), for diagnostics
    near: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))  # candidates nearly up but not throughout
    near_p: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )  # ... and the chance of that for an unrelated signal

    def logw(self, mode="fused"):
        return {"fused": self.log_tau + self.log_ell, "timing": self.log_tau, "doppler": self.log_ell}[mode]


def session_evidence(
    sess: Session,
    cat: Catalog,
    pt: PassTable,
    station: Station,
    lib: DecoyLibrary,
    rng,
    cfg: FitConfig = FIT,
    exclude=(),
):
    """Coarse and fine evidence of one session for every candidate that was visible throughout it."""
    # ---- candidate generation: visible for the whole session
    j = pt.containing(sess.t0, sess.t1, cfg.margin_s)
    cand = np.unique(pt.cand[j])
    if len(exclude):
        cand = cand[~np.isin(cat.norad[cand], list(exclude))]
    length = max(sess.span - 2 * cfg.margin_s, 0.0)
    q = pt.visible_fraction(length)[cand]
    log_tau = -np.log(np.maximum(q, 1e-7))
    # ---- candidates that were 'nearly up': a pass within kin_window of the session, but not containing it
    lo, hi = np.searchsorted(pt.aos, [sess.t0 - cfg.kin_window - 7200.0, sess.t1 + cfg.kin_window])
    k = np.arange(lo, hi)
    near = np.setdiff1d(np.unique(pt.cand[k[pt.los[k] >= sess.t0 - cfg.kin_window]]), cand)
    pad = np.bincount(pt.cand, weights=pt.dur + 2 * cfg.kin_window + sess.span, minlength=pt.n_cand) / (pt.t1 - pt.t0)
    near_p = np.clip(pad[near] - pt.visible_fraction(length)[near], 0.0, 1.0)
    # ---- thin the track
    n_all = len(sess.t)
    stride = max(1, math.ceil(n_all / cfg.n_max))
    sel = slice(None, None, stride)
    t, y, seg = sess.t[sel], sess.f[sel] - sess.fc, sess.seg[sel]
    scale = sess.fc / F0
    n = len(t)
    # ---- model curves: candidates and decoys, all with the same transmitter model and the same along-track freedom
    nc = len(cand)
    if nc:
        both = doppler_hz(cat.subset(cand), station, np.r_[t, t + 1.0], F0) * scale
        d_cand, dd_cand = np.nan_to_num(both[:, :n], nan=1e6), np.nan_to_num(both[:, n:] - both[:, :n])
    else:
        d_cand = dd_cand = np.zeros((0, n))
    d_dec, dd_dec, dec_pass, dec_t = lib.draw(t - t.min(), cfg.n_decoys, rng)
    curves, slopes = np.concatenate([d_cand, d_dec * scale]), np.concatenate([dd_cand, dd_dec * scale])
    X, lam, nf = _design(t, seg, cfg)
    slope_col = nf if cfg.drift_order else None
    if len(curves):
        along = (slopes, 1.0 / cfg.sigma_along**2) if cfg.sigma_along > 0 else (None, 0.0)
        ll, beta, sig, res = robust_fit(y[None] - curves, X, lam, seg, cfg, slope_col, *along)
    else:
        ll, beta, sig, res = np.zeros(0), np.zeros((0, X.shape[1])), np.zeros(0), np.zeros((0, n))
    # ---- 'a satellite on some other orbit': orbits fitted freely to the track, started from the best candidates
    #      and decoys. Six free parameters, paid for below with a BIC penalty.
    ll_o, sig_o, res_o = np.full(1, -np.inf), np.ones(1), np.zeros((1, n))
    if cfg.free_orbit and len(curves) and n >= 12:
        tm = float(t.mean())
        starts = [_state_teme(cat.sats[cand[i]], tm) for i in np.argsort(-ll[:nc])[: cfg.free_starts]]
        for i in np.argsort(-ll[nc:])[: cfg.free_starts]:
            t_lib = dec_t[i] + (tm - t.min())  # the library pass's own time that corresponds to tm
            starts.append(
                _rotate_z(_state_teme(cat.sats[lib.cand[dec_pass[i]]], t_lib), float(gmst(tm) - gmst(t_lib)))
            )
        orbits = free_orbit_curves(t, y, X, lam, np.array(starts), station, sess.fc, cfg)
        ll_o, _, sig_o, res_o = robust_fit(y[None] - orbits, X, lam, seg, cfg, slope_col)
    # ---- 'a smooth curve that no orbit explains': a free cubic, with a BIC penalty for its three extra coefficients
    Xc, lamc, nfc = _design(t, seg, cfg, free_poly=3)
    ll_c, _, sig_c, res_c = robust_fit(y[None], Xc, lamc, seg, cfg, nfc)
    # ---- 'no Doppler at all' (a spur, a ground carrier): the transmitter model alone, with a much tighter drift
    flat_cfg = replace(cfg, sigma_drift=cfg.sigma_flat)
    Xf, lamf, nff = _design(t, seg, flat_cfg, wander=False)
    ll_f, _, _, _ = robust_fit(y[None], Xf, lamf, seg, flat_cfg, nff if cfg.drift_order else None)
    # ---- effective sample size, from the residuals of the best-fitting model of all
    gamma = 1.0
    if cfg.n_eff:
        pick = [(ll_c[0], res_c[0], sig_c[0])]
        if len(ll):
            b = int(np.argmax(ll))
            pick.append((ll[b], res[b], sig[b]))
        b = int(np.argmax(ll_o))
        pick.append((ll_o[b], res_o[b], sig_o[b]))
        _, r_best, s_best = max(pick, key=lambda x: x[0])
        gamma = effective_fraction(r_best, s_best, seg, t)
    n_eff = gamma * n
    ll = gamma * ll
    ll_cand = ll[:nc]
    pen = math.log(max(n_eff, 2.0))
    ll_cubic = gamma * ll_c[0] - 1.5 * pen
    ll_orbit = gamma * float(ll_o.max()) - min(3.0 * pen, cfg.orbit_bonus)
    ll_flat = gamma * ll_f[0] + math.log(
        cfg.sigma_drift / cfg.sigma_flat
    )  # the tighter prior is the more specific claim
    # ---- Doppler evidence. The reference is 'a satellite that is not H': the average over H itself, the decoys and
    #      the cubic, or the freely fitted orbit if that explains the track better. Against the average alone, a
    #      candidate exchangeable with the decoys has mean <= 1; the free orbit can only lower it further. Never
    #      more than K + 2.
    ll_ref = np.r_[ll[nc:], ll_cubic]
    k_ref = len(ll_ref)
    m = ll_ref.max()
    log_mref = m + math.log(np.mean(np.exp(ll_ref - m)))
    x = ll_cand - max(log_mref, ll_orbit)
    log_ell = math.log(k_ref + 1) + x - np.logaddexp(x, math.log(k_ref))
    log_nonsat = ll_flat - max(log_mref, ll_orbit)
    if (
        sess.span < cfg.nonsat_min_span
    ):  # a short burst of a slow, high satellite is flat too: it may refute, not support
        log_nonsat = min(log_nonsat, 0.0)
    pd = np.exp(ll_ref - m)
    dec_eff = float(pd.sum() ** 2 / np.sum(pd * pd))
    return SessionEvidence(
        sess.t0,
        sess.t1,
        sess.fc,
        cand,
        log_tau,
        log_ell,
        sess.fc + beta[:nc, 0],
        sig[:nc],
        float(log_nonsat),
        n,
        n_eff,
        dec_eff,
        x,
        sess.truth,
        sess.dets[0].kind,
        {
            "decoys": log_mref,
            "orbit": ll_orbit,
            "cubic": ll_cubic,
            "flat": ll_flat,
            "gamma": gamma,
            "sigma_orbit": float(sig_o[int(np.argmax(ll_o))]),
            "ll_cand": ll_cand,
        },
        near,
        near_p,
    )


# ---------------------------------------------------------------------------------------------------------------------
# 6 · Clustering by channel
# ---------------------------------------------------------------------------------------------------------------------

MAX_DOPPLER_HZ = F0 * 7.8e3 / C_LIGHT  # ~3.6 kHz: the fastest LEO at the horizon


def cluster_by_frequency(dets: list[Detection], drift_hz=500.0):
    """Group detections whose centre frequencies chain together within max Doppler + expected drift.
    Returns a list of lists of detections, sorted by frequency."""
    if not dets:
        return []
    tol = MAX_DOPPLER_HZ + drift_hz
    order = sorted(dets, key=lambda d: d.fc)
    out = [[order[0]]]
    for d in order[1:]:
        if d.fc - out[-1][-1].fc > tol:
            out.append([])
        out[-1].append(d)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# 7 · Fusion over a campaign and the decision rule
# ---------------------------------------------------------------------------------------------------------------------

EPS_GRID = np.array([0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 0.98])


@dataclass(frozen=True)
class DecisionConfig:
    alpha: float = 0.01  # tolerated probability of a false identification per channel
    n_candidates: int = 4321  # size of the catalog the threshold must pay for
    min_sessions: int = 3  # passes that must support an identification
    enter: float = 3.0  # nats a transmitter must add to be put into the channel's mixture
    max_components: int = 6
    max_group: int = 12
    top: int = 60  # candidates kept for the mixture (by their stand-alone score)
    mode: str = "fused"  # "fused" | "timing" | "doppler"
    w_cap: float = 8.0
    kin_alpha: float = (
        0.01  # odds below which a candidate is called a neighbour of the transmitter, not the transmitter
    )
    kin_share: float = 0.9  # share of a neighbour's uncontained sessions that fall close to its passes
    doppler_min: float = (
        8.0  # nats of Doppler-only evidence an identification needs (timing alone shows kinship, not identity)
    )
    doppler_sessions: int = 3  # ... and this many sessions whose Doppler evidence alone is at least doppler_strong
    doppler_strong: float = 3.0  # nats (20 : 1)
    nic_sessions: int = 5  # informative satellite-like sessions no candidate explains -> 'not in catalog'
    nic_share: float = 1 / 3  # ... and they must be at least this share of the channel's informative sessions
    nic_dec_eff: float = 25.0  # 'informative': fewer than this many of the decoys fit about as well as the best

    @property
    def threshold(self):
        return math.log(self.n_candidates / self.alpha)


DECISION = DecisionConfig()


def _escore(logr, cap):
    """ln of the average over eps of prod_s (1 - eps + eps r_s), for every column of logr (S, A).

    Each factor is what a session does to the odds that the column's transmitter is on this channel if it accounts
    for a share eps of the sessions: sessions it explains multiply the odds by about eps r, sessions it cannot
    explain cost only (1 - eps). Silence costs nothing: only sessions enter. r is capped at exp(cap), so that no
    single session decides. Averaging over eps keeps the product a non-negative martingale under 'this transmitter
    has nothing to do with the channel'."""
    r = np.exp(np.minimum(logr, cap))
    tot = np.stack([np.sum(np.log1p(e * (r - 1.0)), axis=0) for e in EPS_GRID])
    m = tot.max(axis=0)
    return m + np.log(np.mean(np.exp(tot - m), axis=0))


def _em(W, iters=200):
    """Mixture weights maximising sum_s log(W[s] @ pi). W: (S, C) likelihood ratios, column 0 = unknown satellite."""
    C = W.shape[1]
    pi = np.full(C, 1.0 / C)
    for _ in range(iters):
        r = W * pi
        r /= r.sum(axis=1, keepdims=True)
        new = r.mean(axis=0)
        if np.max(np.abs(new - pi)) < 1e-6:
            pi = new
            break
        pi = new
    return pi


def _build(W, banned, cfg: DecisionConfig):
    """Greedy sparse mixture for a channel: start from {unknown satellite, non-satellite}, add the transmitter that
    improves the explanation most, refit, repeat. W: (S, 2 + A). Returns (active columns, pi, m (S,))."""
    active = [h for h in (0, 1) if h not in banned]
    pi = _em(W[:, active])
    m = W[:, active] @ pi
    logW = np.log(np.maximum(W, 1e-300))
    while len(active) - 2 < cfg.max_components:
        g = _escore(logW - np.log(m)[:, None], cfg.w_cap)
        g[active] = -np.inf
        g[list(banned)] = -np.inf
        h = int(np.argmax(g))
        if not np.isfinite(g[h]) or g[h] < cfg.enter:
            break
        active.append(h)
        pi = _em(W[:, active])
        m = W[:, active] @ pi
    return active, pi, m


@dataclass
class Verdict:
    kind: str  # "identified" | "ambiguous" | "related" | "not_satellite" | "not_in_catalog"
    members: tuple[int, ...]  # catalog indices (empty for the last two)
    score: float  # nats
    support: int  # sessions


@dataclass
class ChannelState:
    """Result of scoring one channel on its sessions so far."""

    cols: np.ndarray  # catalog index of every scored candidate
    score: np.ndarray  # campaign score per scored candidate [nats]; -inf if never in the mixture
    support: np.ndarray
    dscore: np.ndarray  # Doppler-only score [nats]
    strong: np.ndarray  # sessions with strong Doppler evidence
    share: np.ndarray  # mixture weight
    verdicts: list[Verdict]
    resp_unknown: np.ndarray  # per session
    resp_nonsat: np.ndarray


def score_channel(evs: list[SessionEvidence], cfg: DecisionConfig = DECISION) -> ChannelState:
    """Campaign-wide score of every candidate for one channel, and the verdicts the decision rule allows."""
    S = len(evs)
    thr = cfg.threshold
    # ---- stand-alone scores of everyone who was ever visible, to pick the candidates worth modelling
    n_cat = max((int(e.cand.max()) for e in evs if len(e.cand)), default=-1) + 1
    lw = [e.logw(cfg.mode) for e in evs]
    acc = np.tile((S * np.log1p(-EPS_GRID))[:, None], (1, n_cat))
    for e, lw_s in zip(evs, lw, strict=True):
        w = np.exp(np.minimum(lw_s, cfg.w_cap))
        acc[:, e.cand] += np.log1p(EPS_GRID[:, None] * (w[None] - 1.0)) - np.log1p(-EPS_GRID)[:, None]
    mx = acc.max(axis=0) if n_cat else np.zeros(0)
    alone = mx + np.log(np.mean(np.exp(acc - mx), axis=0)) if n_cat else np.zeros(0)
    cols = np.argsort(-alone)[: cfg.top]
    cols = cols[alone[cols] > 0]
    A = len(cols)
    W = np.zeros((S, 2 + A))
    W[:, 0] = 1.0
    for s, (e, lw_s) in enumerate(zip(evs, lw, strict=True)):
        W[s, 1] = math.exp(min(max(e.log_nonsat, -300.0), 300.0)) if cfg.mode != "timing" else 0.0
        pos = np.searchsorted(e.cand, cols)
        ok = (
            (pos < len(e.cand)) & (e.cand[np.minimum(pos, len(e.cand) - 1)] == cols)
            if len(e.cand)
            else np.zeros(A, bool)
        )
        W[s, 2:][ok] = np.exp(np.clip(lw_s[pos[ok]], -300.0, 300.0))
    logW = np.log(np.maximum(W, 1e-300))
    active, pi, m = _build(W, (), cfg)
    resp = W[:, active] * pi / m[:, None]
    # Doppler-only evidence of every scored candidate, against 'unknown satellite' alone
    LD = np.full((S, A), -300.0)
    for s_i, e in enumerate(evs):
        pos = np.searchsorted(e.cand, cols)
        ok = (
            (pos < len(e.cand)) & (e.cand[np.minimum(pos, len(e.cand) - 1)] == cols)
            if len(e.cand)
            else np.zeros(A, bool)
        )
        LD[s_i, ok] = e.log_ell[pos[ok]]
    dscore = _escore(LD, cfg.w_cap) if A else np.zeros(0)
    strong = np.sum(cfg.doppler_strong <= LD, axis=0)
    fused = cfg.mode == "fused"
    need_dop, need_strong = (cfg.doppler_min, cfg.doppler_sessions) if fused else (-np.inf, 0)
    score = np.full(A, -np.inf)
    support = np.zeros(A, dtype=int)
    share = np.zeros(A)
    verdicts = []
    claimed = np.zeros(S, dtype=bool)
    for k, h in enumerate(active):
        if h == 0:
            continue
        _, _, m_wo = _build(W, {h}, cfg)
        sc = float(_escore((logW[:, h] - np.log(m_wo))[:, None], cfg.w_cap)[0])
        sup = int(np.sum(resp[:, k] >= 0.5))
        if h == 1:
            if sc >= thr and sup >= cfg.min_sessions:
                verdicts.append(Verdict("not_satellite", (), sc, sup))
                claimed |= resp[:, k] >= 0.5
            continue
        share[h - 2], support[h - 2] = pi[k], sup
        # Kin test. Take the sessions nobody in the catalog explains and that this candidate cannot contain. If the
        # candidate is the transmitter, they come from someone unrelated and only a share p of them falls close to
        # its passes. If the candidate is merely a neighbour of the real (uncatalogued) transmitter, nearly all do.
        # The log-likelihood ratio of the two readings is added to the score when it is negative.
        col = cols[h - 2]
        kin_llr = 0.0
        if cfg.mode == "fused":
            for e, m_ in zip(evs, (resp[:, 0] >= 0.5) & (W[:, h] == 0), strict=True):
                if m_:
                    at = np.flatnonzero(e.near == col)
                    p_near = float(np.clip(e.near_p[at[0]], 0.02, 0.8)) if at.size else 0.3
                    kin_llr += (
                        math.log(p_near / cfg.kin_share) if at.size else math.log((1 - p_near) / (1 - cfg.kin_share))
                    )
        kin = kin_llr <= math.log(cfg.kin_alpha)
        sc_adj = sc + min(0.0, kin_llr)
        if sc >= thr and sup >= cfg.min_sessions and kin:
            verdicts.append(Verdict("related", (int(col),), sc, sup))
        elif sc_adj >= thr and sup >= cfg.min_sessions and dscore[h - 2] >= need_dop and strong[h - 2] >= need_strong:
            verdicts.append(Verdict("identified", (int(col),), sc_adj, sup))
            claimed |= resp[:, k] >= 0.5
        score[h - 2] = sc_adj
        if sup >= cfg.min_sessions and sc < thr and not kin:
            # rivals may be hiding the evidence: grow a group until, as a whole, it is needed
            group, own = [h], resp[:, k] >= 0.5
            while len(group) < cfg.max_group:
                act2, pi2, m2 = _build(W, set(group), cfg)
                gsc = float(_escore((logW[:, group].max(axis=1) - np.log(m2))[:, None], cfg.w_cap)[0])
                gld = LD[:, [g - 2 for g in group]].max(axis=1)
                if (
                    gsc >= thr
                    and float(_escore(gld[:, None], cfg.w_cap)[0]) >= need_dop
                    and np.sum(gld >= cfg.doppler_strong) >= need_strong
                ):
                    verdicts.append(Verdict("ambiguous", tuple(sorted(int(cols[g - 2]) for g in group)), gsc, sup))
                    claimed |= own
                    break
                r2 = W[:, act2] * pi2 / m2[:, None]
                take = [(float(r2[own, i].sum()), a) for i, a in enumerate(act2) if a >= 2]
                if not take or max(take)[0] < 0.5 * own.sum():
                    break
                group.append(max(take)[1])
    # ---- a satellite that is not in the catalog: enough informative, satellite-like sessions nobody explains
    r_unknown = resp[:, 0]
    informative = (
        np.array([e.log_nonsat < -3.0 and e.dec_eff <= cfg.nic_dec_eff for e in evs]) if S else np.zeros(0, bool)
    )
    n_nic = int(np.sum(informative & (r_unknown >= 0.9) & ~claimed))
    if n_nic >= cfg.nic_sessions and n_nic >= cfg.nic_share * informative.sum():
        verdicts.append(Verdict("not_in_catalog", (), float(n_nic), n_nic))
    return ChannelState(
        cols, score, support, dscore, strong, share, verdicts, r_unknown, resp[:, 1] if 1 in active else np.zeros(S)
    )


# ---------------------------------------------------------------------------------------------------------------------
# 8 · Context and Monte-Carlo workers (used by the notebook; workers run in a process pool)
# ---------------------------------------------------------------------------------------------------------------------

ELEMENTS = ("elements/active_2026-11-14.csv", "elements/satnogs_2026-11-14.csv")
CAMPAIGN_START = "2026-11-14T00:00:00Z"
CAMPAIGN_DAYS = 14
ZOMBIES = (965, 1804, 4256, 4237, 5680)  # TRANSIT 5B-5, ALOUETTE 2, OPS 7613 P/L 1 and P/L 4, OPS 7898 P/L 2
TWINS = (
    44207,
    56154,
    25544,
    60211,
)  # TIANHUI 2-01A, PIESAT-1 B, ISS (ZARYA), TIANHUI 5C: each flies with catalogued companions


class Context:
    """Everything the experiments share: catalog, pass table, decoy library, ASSIST truth of the transmitter pool."""

    def __init__(self, log=print):
        OUT.mkdir(parents=True, exist_ok=True)
        self.station = Station()
        self.store = ElementStore([ASSIST / e for e in ELEMENTS])
        self.t0 = unix(CAMPAIGN_START)
        self.t1 = self.t0 + CAMPAIGN_DAYS * 86400.0
        rows, self.funnel = filter_catalog(self.store.snapshot(self.t0), t_ref=self.t0)
        self.cat = Catalog.from_rows(rows)
        self.pt = build_pass_table(self.cat, self.station, self.t0, self.t1, cache=OUT / "passes.npz")
        self.lib = DecoyLibrary(self.cat, self.pt, self.station, cache=OUT / "decoys_v2.npz")
        rng = np.random.default_rng(2026)
        rest = [int(n) for n in rng.permutation(self.cat.norad) if n not in ZOMBIES + TWINS][:51]
        self.pool = list(ZOMBIES + TWINS) + rest
        self.perturb = {n: (float(rng.normal(0, 0.25)), float(rng.normal(0, 0.01))) for n in self.pool}
        scen = SCEN / "sco_campaign.yaml"
        if not scen.exists():
            write_campaign_scenario(
                scen, self.pool, CAMPAIGN_START, CAMPAIGN_DAYS, self.perturb, self.station, ELEMENTS
            )
        self.truth = campaign_truth(scen, CAMPAIGN_DAYS, OUT / "truth.npz", log)
        self._lit: dict = {}
        self.companions = {n: self._companions(n) for n in self.pool}  # fly together: within a minute
        self.train = {
            n: self._companions(n, 300.0, 0.5) for n in self.pool
        }  # same train: within five minutes, half the passes

    def _companions(self, norad, window_s=60.0, share=0.8):
        """Catalog objects that fly with this one: in at least `share` of its passes they rise and set within
        window_s of it. Such objects cannot be told apart by timing and hardly by Doppler."""
        pt, i = self.pt, self.cat.index(norad)
        mine = np.flatnonzero(pt.cand == i)
        hits = np.zeros(len(self.cat), dtype=int)
        for j in mine:
            lo, hi = np.searchsorted(pt.aos, [pt.aos[j] - window_s, pt.aos[j] + window_s])
            k = np.arange(lo, hi)
            k = k[np.abs(pt.los[k] - pt.los[j]) < window_s]
            hits[np.unique(pt.cand[k])] += 1
        hits[i] = 0
        return [int(self.cat.norad[h]) for h in np.flatnonzero(hits >= share * len(mine))]

    def name(self, norad):
        i = self.cat.index(int(norad))
        return self.cat.name[i] if i >= 0 else f"#{norad}"

    def lit(self, tp: TruthPass):
        k = (tp.norad, float(tp.t[0]))
        if k not in self._lit:
            self._lit[k] = sunlit(self.cat.sats[self.cat.index(tp.norad)], tp.t)
        return self._lit[k]

    def emulate(self, norad, profile, f_rest, rng, trk: TrackerModel = TRACKER, t_max=None) -> list[Detection]:
        prof = PROFILES[profile] if isinstance(profile, str) else profile
        sign = 1.0 if (norad % 2) else -1.0
        out = []
        for tp in self.truth[int(norad)]:
            if t_max is not None and tp.t[0] > t_max:
                break
            out += emulate_pass(tp, prof, f_rest, rng, trk, self.lit(tp), sign)
        return out

    def evidence(self, sessions, rng, cfg: FitConfig = FIT, exclude=()):
        return [session_evidence(s, self.cat, self.pt, self.station, self.lib, rng, cfg, exclude) for s in sessions]


_CTX: Context | None = None


def ctx() -> Context:
    global _CTX
    if _CTX is None:
        _CTX = Context(log=lambda *_: None)
    return _CTX


def simple_scores(y, D, loss, sigma=2.0, nu=3.0):
    """Offset-only scoring of every candidate curve (the form the pasted spec and dopplerguesser use). Lower is better.

    ls: mean-centred sum of squares (dopplerguesser's RMSE). t: Student-t with the offset refined from the median.
    clip: chi-square clipped at 3 sigma, offset by coarse grid + local refinement (the loss is not convex)."""
    r = y[None] - D
    if loss == "ls":
        return np.sum((r - r.mean(axis=1, keepdims=True)) ** 2, axis=1)
    c = np.median(r, axis=1)
    if loss == "t":
        for _ in range(20):
            w = (nu + 1) / (nu + (r - c[:, None]) ** 2 / sigma**2)
            c = np.sum(w * r, axis=1) / np.sum(w, axis=1)
        return (nu + 1) * np.sum(np.log1p((r - c[:, None]) ** 2 / (nu * sigma**2)), axis=1)
    if loss == "clip":
        lim = (3 * sigma) ** 2
        grid = c[:, None] + np.arange(-12, 12.01, 0.5)[None] * sigma
        cost = np.stack(
            [np.sum(np.minimum((r - grid[:, [g]]) ** 2, lim), axis=1) for g in range(grid.shape[1])], axis=1
        )
        c = grid[np.arange(len(c)), np.argmin(cost, axis=1)]
        for _ in range(5):
            inl = (r - c[:, None]) ** 2 < lim
            c = np.where(inl.any(axis=1), np.sum(r * inl, axis=1) / np.maximum(inl.sum(axis=1), 1), c)
        return np.sum(np.minimum((r - c[:, None]) ** 2, lim), axis=1) / sigma**2
    raise ValueError(loss)


def mc_simple(task):
    """Experiment A: one clean track window per pass, bad bins injected, every visible candidate scored offset-only.
    task: seed, norad, window_s, bad (fraction), n (windows). Returns list of {loss: correct?} plus candidate count."""
    c = ctx()
    rng = np.random.default_rng(task["seed"])
    passes = [tp for tp in c.truth[task["norad"]] if len(tp.t) > task["window_s"] + 40]
    out = []
    ti = c.cat.index(task["norad"])
    for _ in range(task["n"]):
        tp = passes[rng.integers(len(passes))]
        a = rng.integers(20, len(tp.t) - task["window_s"] - 20)
        sl = slice(a, a + task["window_s"])
        t = tp.t[sl].astype(float)
        y = tp.doppler[sl] + rng.normal(0, 0.1, len(t)) + rng.normal(0, 100.0)
        bad = rng.random(len(t)) < task["bad"]
        y = y + bad * rng.uniform(20, 300, len(t)) * rng.choice([-1.0, 1.0], len(t))
        cand = np.unique(c.pt.cand[c.pt.containing(t[0], t[-1], 5.0)])
        if ti not in cand:
            continue
        D = np.nan_to_num(doppler_hz(c.cat.subset(cand), c.station, t), nan=1e6)
        k = int(np.flatnonzero(cand == ti)[0])
        rec = {"n_cand": len(cand)}
        for loss in ("ls", "t", "clip"):
            sc = simple_scores(y, D, loss)
            rec[loss] = bool(np.argmin(sc) == k)
        out.append(rec)
    return out


def _session_stats(c: Context, e: SessionEvidence, truth_norad):
    ti = c.cat.index(int(truth_norad)) if truth_norad >= 0 else -1
    k = np.flatnonzero(e.cand == ti)
    fused = e.log_tau + e.log_ell
    rec = {
        "span": e.t1 - e.t0, "n": e.n, "n_eff": e.n_eff, "dec_eff": e.dec_eff, "n_cand": len(e.cand),
        "log_nonsat": e.log_nonsat, "in": bool(k.size), "mean_ell_wrong": float("nan"), "max_ell_wrong": float("nan"),
    }  # fmt: skip
    wrong = np.ones(len(e.cand), dtype=bool)
    if k.size:
        k = int(k[0])
        wrong[k] = False
        rec |= {
            "rank_dop": int(1 + np.sum(e.log_x > e.log_x[k])), "rank_tim": int(1 + np.sum(e.log_tau > e.log_tau[k])),
            "rank_fus": int(1 + np.sum(fused > fused[k])), "log_ell": float(e.log_ell[k]),
            "log_tau": float(e.log_tau[k]), "sigma": float(e.sigma[k]), "f_rest": float(e.f_rest[k]),
        }  # fmt: skip
    if wrong.any():
        rec["mean_ell_wrong"] = float(np.mean(np.exp(e.log_ell[wrong])))
        rec["max_ell_wrong"] = float(e.log_ell[wrong].max())
        rec["sum_w_wrong"] = float(np.sum(np.exp(fused[wrong])))
    return rec


def mc_fine(task):
    """Experiments B and C: session-level Doppler evidence for one transmitter under several fit configurations.
    task: seed, norad, profile, cfgs {name: FitConfig kwargs}, optional window_s (cut one window per detection),
    optional trk (TrackerModel kwargs), max_sessions, exclude_truth, t_shift (extra along-track error in seconds)."""
    c = ctx()
    rng = np.random.default_rng(task["seed"])
    trk = TrackerModel(**task.get("trk", {}))
    dets = c.emulate(task["norad"], task["profile"], task.get("f_rest", 136.75e6), rng, trk)
    if task.get("window_s"):
        w, cut = task["window_s"], []
        for d in dets:
            if len(d.t) > w:
                a = rng.integers(0, len(d.t) - w)
                cut.append(
                    Detection(d.t[a : a + w], d.f[a : a + w], d.cn0[a : a + w], d.truth, d.kind, d.bad[a : a + w])
                )
        dets = cut
    if task.get("t_shift"):  # element-set error: the satellite runs this many seconds late against its catalog entry
        for d in dets:
            d.t = d.t + task["t_shift"]
    sessions = make_sessions(dets)
    if len(sessions) > task.get("max_sessions", 10**9):
        sessions = [sessions[i] for i in rng.choice(len(sessions), task["max_sessions"], replace=False)]
    out = {}
    for name, kw in task["cfgs"].items():
        sub = np.random.default_rng(task["seed"] + 1)
        ex = (task["norad"],) if task.get("exclude_truth") else ()
        evs = c.evidence(sessions, sub, FitConfig(**kw), ex)
        out[name] = [_session_stats(c, e, task["norad"]) for e in evs]
    return out


def trajectory(c: Context, evs, true_norads, dcfg: DecisionConfig, steps=None):
    """Score a channel after each new session. Returns per step: the true transmitters' scores and supports, the best
    score of any other candidate that has enough support, and the verdicts."""
    S = len(evs)
    steps = steps or [n for n in range(1, S + 1) if n <= 40 or n % 4 == 0 or n == S]
    true_idx = [c.cat.index(int(n)) for n in true_norads]
    rec = {
        "steps": steps,
        "t": [evs[n - 1].t1 for n in steps],
        "true_score": [],
        "true_support": [],
        "wrong_score": [],
        "wrong_id": [],
        "verdicts": [],
        "true_dop": [],
        "wrong_dop": [],
    }
    for n in steps:
        st = score_channel(evs[:n], dcfg)
        ts, tsup, tdop = [], [], []
        for ti in true_idx:
            k = np.flatnonzero(st.cols == ti)
            ts.append(float(st.score[k[0]]) if k.size else float("-inf"))
            tsup.append(int(st.support[k[0]]) if k.size else 0)
            tdop.append((float(st.dscore[k[0]]), int(st.strong[k[0]])) if k.size else (float("-inf"), 0))
        wrong = ~np.isin(st.cols, true_idx) & (st.support >= dcfg.min_sessions)
        if wrong.any():
            j = int(np.argmax(np.where(wrong, st.score, -np.inf)))
            rec["wrong_score"].append(float(st.score[j]))
            rec["wrong_id"].append(int(c.cat.norad[st.cols[j]]))
            rec["wrong_dop"].append((float(st.dscore[j]), int(st.strong[j])))
        else:
            rec["wrong_score"].append(float("-inf"))
            rec["wrong_id"].append(-1)
            rec["wrong_dop"].append((float("-inf"), 0))
        rec["true_dop"].append(tdop)
        rec["true_score"].append(ts)
        rec["true_support"].append(tsup)
        rec["verdicts"].append(
            [
                (v.kind, tuple(int(c.cat.norad[m]) for m in v.members), round(v.score, 2), v.support)
                for v in st.verdicts
            ]
        )
    return rec


def mc_channel(task):
    """One channel over the whole campaign. task: seed, txs [(norad, profile, f_rest)], ground (bool), exclude_truth,
    modes (decision modes to run), fit (FitConfig kwargs), dec (DecisionConfig kwargs), days."""
    c = ctx()
    rng = np.random.default_rng(task["seed"])
    t_max = c.t0 + task.get("days", CAMPAIGN_DAYS) * 86400.0
    dets = []
    for norad, profile, f_rest in task["txs"]:
        dets += c.emulate(norad, profile, f_rest, rng, TrackerModel(**task.get("trk", {})), t_max)
    if task.get("ground"):
        dets += ground_carrier(c.t0, t_max, task["ground"], rng)
    sessions = make_sessions(dets)
    true_norads = [n for n, _, _ in task["txs"]]
    ex = tuple(true_norads) if task.get("exclude_truth") else ()
    evs = c.evidence(sessions, rng, FitConfig(**task.get("fit", {})), ex)
    out = {
        "task": task,
        "n_sessions": len(sessions),
        "n_dets": len(dets),
        "sessions": [_session_stats(c, e, e.truth) | {"t1": e.t1, "truth": e.truth} for e in evs],
    }
    for mode in task.get("modes", ("fused",)):
        out[mode] = trajectory(
            c,
            evs,
            [] if ex else true_norads,
            DecisionConfig(n_candidates=len(c.cat), mode=mode, **task.get("dec", {})),
        )
    return out


def run_pool(fn, tasks, workers=6, log=None):
    """Map tasks over a process pool (spawned workers build their own Context from the caches)."""
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(workers) as pool:
        out = []
        for i, r in enumerate(pool.imap(fn, tasks)):
            out.append(r)
            if log and (i + 1) % max(1, len(tasks) // 10) == 0:
                log(f"  {i + 1}/{len(tasks)}")
    return out
