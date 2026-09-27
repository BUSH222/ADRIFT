import socket
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import fft
from scipy.signal.windows import nuttall

RTL_TCP_HOST = "localhost"
RTL_TCP_PORT = 1234

SAMPLE_RATE = 2_400_000

FFT_SIZE = 16384
IMG_WIDTH = FFT_SIZE
IMG_HEIGHT = 86400
ROW_SECONDS = 1.0

FFT_MAX_DIFF_DB = 30
CALIBRATION_SECONDS = 5.0

CHECKPOINT_EVERY = 120

OUTPUT_DIR = Path(".")


# Sockets


def recv_exact(sock, nbytes):
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def read_iq(sock, n_samples):
    raw = recv_exact(sock, n_samples * 2)
    if raw is None:
        raise ConnectionError("rtl_tcp connection closed")
    u8 = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    iq = (u8[0::2] - 127.5) / 127.5 + 1j * (u8[1::2] - 127.5) / 127.5
    return iq


# FFT


def power_spectrum(iq, window):
    spectrum = np.fft.fftshift(fft(iq * window, workers=-1))
    return spectrum.real**2 + spectrum.imag**2


def estimate_noise_floor(sock, window):
    n_blocks = max(1, int(CALIBRATION_SECONDS * SAMPLE_RATE / FFT_SIZE))
    power_sum = np.zeros(FFT_SIZE, dtype=np.float64)
    for _ in range(n_blocks):
        iq = read_iq(sock, FFT_SIZE)
        power_sum += power_spectrum(iq, window)
    return power_sum / n_blocks


def save_png(rows, row_idx, outfile):
    if row_idx == 0:
        return
    image_array = rows[:row_idx]
    Image.fromarray(image_array, mode="L").save(outfile)


def save_preview_png(raw_path, row_idx, img_width, outfile):
    with open(raw_path, "rb") as f:
        data = np.frombuffer(f.read(row_idx * img_width), dtype=np.uint8)
    Image.fromarray(data.reshape(row_idx, img_width), mode="L").save(outfile)


# Capture


def capture():
    start_timestamp = int(time.time())
    outfile = OUTPUT_DIR / f"waterfall_{start_timestamp}.png"
    rawfile = OUTPUT_DIR / f"waterfall_{start_timestamp}.raw"

    window = nuttall(FFT_SIZE)
    avg_count = max(1, round(ROW_SECONDS * SAMPLE_RATE / FFT_SIZE))

    rows = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
    row_idx = 0

    with socket.create_connection((RTL_TCP_HOST, RTL_TCP_PORT), timeout=5) as sock:
        sock.settimeout(None)

        print(f"Calibrating noise floor over {CALIBRATION_SECONDS:.0f}s ...")
        noise_power = estimate_noise_floor(sock, window)
        noise_db = 10 * np.log10(noise_power + 1e-12)
        min_db = noise_db.mean() - 5
        max_db = min_db + FFT_MAX_DIFF_DB
        print(f"min_db={min_db:.1f} max_db={max_db:.1f}")
        print(f"avg_count={avg_count} rows={IMG_HEIGHT} (~{IMG_HEIGHT * IMG_WIDTH / 1e6:.0f} MB raw)")

        with open(rawfile, "wb") as raw_f:
            try:
                while row_idx < IMG_HEIGHT:
                    power_sum = np.zeros(FFT_SIZE, dtype=np.float64)
                    for _ in range(avg_count):
                        iq = read_iq(sock, FFT_SIZE)
                        power_sum += power_spectrum(iq, window)

                    power_db = 10 * np.log10(power_sum / avg_count + 1e-12)
                    row = np.clip((power_db - min_db) / (max_db - min_db) * 255, 0, 255).astype(np.uint8)
                    rows[row_idx] = row
                    raw_f.write(row.tobytes())
                    row_idx += 1

                    if row_idx % 20 == 0 or row_idx == IMG_HEIGHT:
                        print(f"row {row_idx}/{IMG_HEIGHT}")

                    if row_idx % CHECKPOINT_EVERY == 0:
                        raw_f.flush()

            except KeyboardInterrupt:
                print("Interrupted, saving what we have ...")
            finally:
                save_png(rows, row_idx, outfile)
                print(f"Saved {row_idx} rows to {outfile} (raw dump: {rawfile})")


if __name__ == "__main__":
    capture()
