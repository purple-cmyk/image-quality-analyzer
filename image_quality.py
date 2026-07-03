
"""
image_quality.py
Classical image quality metrics (no deep learning).

Requires:
    pip install opencv-python numpy scipy pywavelets
"""

import sys
import time
import tracemalloc
from pathlib import Path

import cv2
import pywt
import numpy as np
from scipy import fftpack


class ImageQualityAnalyzer:
    def __init__(self, image):
        if isinstance(image, str):
            path = Path(image).expanduser()
            if not path.is_file():
                path = Path.cwd() / path
            if not path.is_file():
                matches = sorted(
                    p.name
                    for p in Path.cwd().iterdir()
                    if p.is_file() and p.stem.lower() == Path(image).stem.lower()
                )
                msg = f"Image not found: {image}"
                if matches:
                    msg += f"\nDid you mean: {', '.join(matches)}?"
                raise FileNotFoundError(msg)
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Could not decode image: {path}")
            self.bgr = img
        else:
            self.bgr = image.copy()

        self.rgb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB)
        self.gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2HSV)
        self.height, self.width = self.gray.shape[:2]

    def _timed(self, name, timings, fn, *args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        timings[name] = (time.perf_counter() - start) * 1000
        return result

    # ---------------- Blur ----------------

    def variance_of_laplacian(self):
        return float(cv2.Laplacian(self.gray, cv2.CV_64F).var())

    def tenengrad(self):
        gx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, 3)
        gy = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, 3)
        return float(np.mean(gx * gx + gy * gy))

    def brenner(self):
        g = self.gray.astype(np.float64)
        return float(np.mean((g[:, 2:] - g[:, :-2]) ** 2))

    # ---------------- Motion blur ----------------

    def directional_gradient_energy(self):
        energies = {}
        kernels = {
            "0": np.array([[-1,0,1]]),
            "90": np.array([[-1],[0],[1]]),
            "45": np.array([[0,1,1],[-1,0,1],[-1,-1,0]],np.float32),
            "135": np.array([[1,1,0],[1,0,-1],[0,-1,-1]],np.float32),
        }
        for k,v in kernels.items():
            f = cv2.filter2D(self.gray.astype(np.float32),-1,v)
            energies[k]=float(np.mean(np.abs(f)))
        return energies

    def fft_motion_blur_score(self):
        f = fftpack.fftshift(fftpack.fft2(self.gray))
        mag = np.log(np.abs(f)+1)
        h,w = mag.shape
        c = mag[h//2-10:h//2+10,w//2-10:w//2+10]
        return float(np.mean(c)/np.mean(mag))

    # ---------------- Noise ----------------

    def wavelet_noise_sigma(self):
        coeffs = pywt.wavedec2(self.gray.astype(np.float32),"db1",level=1)
        _,(cH,cV,cD)=coeffs
        return float(np.median(np.abs(cD))/0.6745)

    def residual_noise(self):
        blur=cv2.GaussianBlur(self.gray,(3,3),0)
        return float(np.std(self.gray.astype(np.float32)-blur.astype(np.float32)))

    # ---------------- Exposure ----------------

    def mean_brightness(self):
        return float(np.mean(self.gray))

    def clipped_pixels(self):
        dark=np.mean(self.gray<5)*100
        bright=np.mean(self.gray>250)*100
        return float(dark),float(bright)

    def histogram_entropy(self):
        hist=cv2.calcHist([self.gray],[0],None,[256],[0,256]).ravel()
        p=hist/np.sum(hist)
        p=p[p>0]
        return float(-np.sum(p*np.log2(p)))

    # ---------------- Contrast ----------------

    def rms_contrast(self):
        return float(np.std(self.gray))

    def michelson_contrast(self):
        mx=float(np.max(self.gray))
        mn=float(np.min(self.gray))
        if mx+mn==0:
            return 0.0
        return (mx-mn)/(mx+mn)

    def local_contrast(self,win=15):
        f=self.gray.astype(np.float32)
        mean=cv2.blur(f,(win,win))
        mean2=cv2.blur(f*f,(win,win))
        std=np.sqrt(np.maximum(mean2-mean*mean,0))
        return float(np.mean(std))

    # ---------------- Haze ----------------

    def dark_channel(self,size=15):
        m=np.min(self.rgb,axis=2)
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
        return cv2.erode(m,kernel)

    def haze_score(self):
        dc=self.dark_channel()
        return float(np.mean(dc)/255.0)

    # ---------------- Color ----------------

    def colorfulness(self):
        R=self.rgb[:,:,0].astype(np.float32)
        G=self.rgb[:,:,1].astype(np.float32)
        B=self.rgb[:,:,2].astype(np.float32)
        rg=R-G
        yb=0.5*(R+G)-B
        std_rg=np.std(rg)
        std_yb=np.std(yb)
        mean_rg=np.mean(rg)
        mean_yb=np.mean(yb)
        return float(np.sqrt(std_rg**2+std_yb**2)+0.3*np.sqrt(mean_rg**2+mean_yb**2))

    # ---------------- Saturation ----------------

    def saturation_mean(self):
        return float(np.mean(self.hsv[:,:,1]))

    def saturation_std(self):
        return float(np.std(self.hsv[:,:,1]))

    # ---------------- Aggregate ----------------

    def analyze(self):
        timings = {}
        tracemalloc.start()

        d, b = self._timed("clipped_pixels", timings, self.clipped_pixels)
        report = {
            "blur": {
                "laplacian_variance": self._timed(
                    "laplacian_variance", timings, self.variance_of_laplacian
                ),
                "tenengrad": self._timed("tenengrad", timings, self.tenengrad),
                "brenner": self._timed("brenner", timings, self.brenner),
            },
            "motion_blur": {
                "directional_energy": self._timed(
                    "directional_energy", timings, self.directional_gradient_energy
                ),
                "fft_score": self._timed("fft_score", timings, self.fft_motion_blur_score),
            },
            "noise": {
                "wavelet_sigma": self._timed(
                    "wavelet_sigma", timings, self.wavelet_noise_sigma
                ),
                "residual_noise": self._timed("residual_noise", timings, self.residual_noise),
            },
            "exposure": {
                "mean_brightness": self._timed(
                    "mean_brightness", timings, self.mean_brightness
                ),
                "dark_clipped_percent": d,
                "bright_clipped_percent": b,
                "entropy": self._timed("entropy", timings, self.histogram_entropy),
            },
            "contrast": {
                "rms": self._timed("rms_contrast", timings, self.rms_contrast),
                "michelson": self._timed("michelson_contrast", timings, self.michelson_contrast),
                "local": self._timed("local_contrast", timings, self.local_contrast),
            },
            "haze": {
                "dark_channel_mean": self._timed(
                    "dark_channel", timings, lambda: float(np.mean(self.dark_channel()))
                ),
                "score": self._timed("haze_score", timings, self.haze_score),
            },
            "color": {
                "colorfulness": self._timed("colorfulness", timings, self.colorfulness),
                "mean_saturation": self._timed(
                    "saturation_mean", timings, self.saturation_mean
                ),
                "std_saturation": self._timed(
                    "saturation_std", timings, self.saturation_std
                ),
            },
        }

        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total_ms = sum(timings.values())
        mp = (self.width * self.height) / 1_000_000
        report["performance"] = {
            "image_size": f"{self.width}x{self.height} ({mp:.2f} MP)",
            "total_analysis_ms": round(total_ms, 3),
            "peak_memory_mb": round(peak_bytes / (1024 * 1024), 2),
            "metric_timings_ms": dict(
                sorted(timings.items(), key=lambda item: item[1], reverse=True)
            ),
        }
        return report


def print_report(report):
    print("=" * 60)
    print("IMAGE QUALITY REPORT")
    print("=" * 60)
    for section, vals in report.items():
        print(f"\n[{section.upper()}]")
        if section == "performance":
            print(f"{'image_size':25s}: {vals['image_size']}")
            print(f"{'load_ms':25s}: {vals['load_ms']}")
            print(f"{'total_analysis_ms':25s}: {vals['total_analysis_ms']}")
            print(f"{'end_to_end_ms':25s}: {vals['end_to_end_ms']}")
            print(f"{'peak_memory_mb':25s}: {vals['peak_memory_mb']}")
            print("\n  Per-metric timings (slowest first):")
            for name, ms in vals["metric_timings_ms"].items():
                print(f"    {name:23s}: {ms:8.3f} ms")
            continue
        for k, v in vals.items():
            print(f"{k:25s}: {v}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python image_quality.py image.jpg")
        return

    load_start = time.perf_counter()
    analyzer = ImageQualityAnalyzer(sys.argv[1])
    load_ms = (time.perf_counter() - load_start) * 1000

    report = analyzer.analyze()
    report["performance"]["load_ms"] = round(load_ms, 3)
    report["performance"]["end_to_end_ms"] = round(
        load_ms + report["performance"]["total_analysis_ms"], 3
    )

    print_report(report)


if __name__=="__main__":
    main()
