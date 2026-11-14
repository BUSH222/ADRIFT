// Per-sample tracking loops for 1_tracking.ipynb. Plain C ABI, loaded with ctypes by 1_tracking_lib.py, which
// compiles this file on first use:  c++ -O3 -shared -fPIC -o 1_tracking_kernels.so 1_tracking_kernels.cpp
//
// Complex samples are interleaved float32 (numpy complex64). Both loops keep their state in a caller-owned array,
// so a recording can be processed in segments (e.g. to change the loop bandwidth) without a discontinuity.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>

extern "C" {

// ---------------------------------------------------------------------------------------------------------------
// pll_classic: dopplerguesser's DPLL (pll/_dpll_ext.cpp) unchanged in behaviour.
//   phase detector  phi_e = arg(x) - theta          (per sample, amplitude ignored)
//   loop filter     integ += K2 phi_e;  v = K1 phi_e + integ        (2nd order, both clamped to dphi_max)
//   estimate        f = f_center + integ fs / 2 pi  (the integrator only)
// Additions that do not change the loop: the per-sample outputs are averaged over `blk` samples, and an optional
// `blank` mask (1 = ignore this sample: phi_e = 0, the loop coasts on its integrator).
// st = [theta, integ]. Outputs have n / blk elements.
void pll_classic(const float* x, int64_t n, double fs, double f_center, double dphi_max, double K1, double K2,
                 const uint8_t* blank, double* st, int64_t blk, float* f_out, float* e_out) {
    const double twopi = 2.0 * M_PI;
    const double dphi0 = twopi * f_center / fs;
    double theta = st[0], integ = st[1];
    double f_acc = 0.0, e_acc = 0.0;
    int64_t c = 0, o = 0;
    for (int64_t i = 0; i < n; ++i) {
        double phi_e = 0.0;
        if (!blank || !blank[i]) {
            phi_e = std::atan2((double)x[2 * i + 1], (double)x[2 * i]) - theta;
            while (phi_e > M_PI) phi_e -= twopi;
            while (phi_e < -M_PI) phi_e += twopi;
        }
        integ += K2 * phi_e;
        integ = std::max(-dphi_max, std::min(dphi_max, integ));
        double v = K1 * phi_e + integ;
        v = std::max(-dphi_max, std::min(dphi_max, v));
        theta += dphi0 + v;
        while (theta > M_PI) theta -= twopi;
        while (theta < -M_PI) theta += twopi;
        f_acc += f_center + integ * fs / twopi;
        e_acc += std::fabs(phi_e);
        if (++c == blk) {
            f_out[o] = (float)(f_acc / blk);
            e_out[o] = (float)(e_acc / blk);
            ++o; c = 0; f_acc = 0.0; e_acc = 0.0;
        }
    }
    st[0] = theta; st[1] = integ;
}

// ---------------------------------------------------------------------------------------------------------------
// pll_coh: a loop that integrates before it detects.
//
//   x --(derotate by NCO)--> boxcar M1, decimate --> ( )^npow --> shift by -f_off --> sum M2 --> atan2 --> filter
//                 ^                                                                                          |
//                 +------------------------------------------------------------------------------------------+
//
// * The NCO removes the estimated carrier first, so everything after it sees a signal that stands still.
// * boxcar M1 is the pre-filter (noise bandwidth fs / M1). With npow = 1 it is just part of the integration.
// * ( )^npow turns a suppressed-carrier signal into a line at npow times the carrier error (+ f_off, for
//   modulations whose line is offset by a fraction of the symbol rate). npow = 1, 2 or 4.
// * sum M2 is the coherent integration: the phase detector sees an SNR of C/N0 x T, T = M1 M2 / fs, instead of
//   C/N0 / fs as in pll_classic.
// * The phase error is atan2(Q, I) / npow, in carrier cycles.
// * Loop filter, order 2 or 3 (Kaplan & Hegarty table 5.6), noise bandwidth Bn [Hz]:
//     order 2: w0 = Bn / 0.53      f += w0^2 e T                 f_nco = f + 1.414 w0 e
//     order 3: w0 = Bn / 0.7845    r += w0^3 e T                 f += (1.1 w0^2 e + r) T      f_nco = f + 2.4 w0 e
//   The third-order loop carries the Doppler rate r [Hz/s] as a state, so a constant rate causes no phase error
//   and the loop keeps sweeping when updates are blanked. r is clamped to [r_lo, r_hi]: without that, a third-order
//   loop that has lost the signal integrates noise into r and runs away for good.
// * blank (optional, per input sample): blanked samples are left out of the sums; an update with fewer than half
//   of its samples left gets e = 0.
//
// n must be a multiple of M1 * M2. st = [theta (cycles), f (Hz), r (Hz/s), phase of the f_off oscillator (cycles)].
// Per update: f_out = f (the frequency integrator, as in pll_classic), r_out = r, p_i / p_q = the complex sum
// divided by the number of pre-filter outputs in it.
void pll_coh(const float* x, int64_t n, double fs, int npow, int M1, int M2, double f_off, int order, double Bn,
             double r_lo, double r_hi, const uint8_t* blank, double* st, float* f_out, float* r_out, float* p_i,
             float* p_q) {
    const double twopi = 2.0 * M_PI;
    const double T = (double)M1 * M2 / fs;
    const double w0 = order == 3 ? Bn / 0.7845 : Bn / 0.53;
    double theta = st[0], f = st[1], r = st[2], phi2 = st[3];
    double f_nco = f;
    const double dphi2 = f_off * M1 / fs;
    int64_t i = 0, o = 0;
    const int64_t n_upd = n / ((int64_t)M1 * M2);
    for (int64_t u = 0; u < n_upd; ++u) {
        std::complex<double> acc2(0.0, 0.0);
        int64_t good = 0;
        const double dtheta = f_nco / fs;
        for (int m2 = 0; m2 < M2; ++m2) {
            std::complex<double> acc1(0.0, 0.0);
            int c1 = 0;
            for (int m1 = 0; m1 < M1; ++m1, ++i) {
                if (!blank || !blank[i]) {
                    const double a = twopi * theta;
                    const double cs = std::cos(a), sn = std::sin(a);
                    const double xr = x[2 * i], xi = x[2 * i + 1];
                    acc1 += std::complex<double>(xr * cs + xi * sn, xi * cs - xr * sn);
                    ++c1;
                }
                theta += dtheta;
            }
            theta -= std::floor(theta);
            if (2 * c1 > M1) {
                std::complex<double> v = acc1 / (double)c1;
                if (npow >= 2) v = v * v;
                if (npow == 4) v = v * v;
                if (f_off != 0.0) {
                    const double a2 = twopi * phi2;
                    v *= std::complex<double>(std::cos(a2), -std::sin(a2));
                }
                acc2 += v;
                ++good;
            }
            phi2 += dphi2;
            phi2 -= std::floor(phi2);
        }
        double e = 0.0;
        if (2 * good > M2) {
            e = std::atan2(acc2.imag(), acc2.real()) / (twopi * npow);
            acc2 /= (double)good;
        } else {
            acc2 = 0.0;
        }
        if (order == 3) {
            r += w0 * w0 * w0 * e * T;
            r = std::max(r_lo, std::min(r_hi, r));          // a satellite's Doppler rate is bounded
            f += (1.1 * w0 * w0 * e + r) * T;
            f_nco = f + 2.4 * w0 * e;
        } else {
            f += (w0 * w0 * e + r) * T;        // r is not updated: external rate aiding (0 = none)
            f_nco = f + 1.414 * w0 * e;
        }
        f_out[o] = (float)f; r_out[o] = (float)r;
        p_i[o] = (float)acc2.real(); p_q[o] = (float)acc2.imag();
        ++o;
    }
    st[0] = theta; st[1] = f; st[2] = r; st[3] = phi2;
}

}  // extern "C"
