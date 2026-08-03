"""PhishGuard runtime performance benchmark (reproduces Figure 4.7).

Run from the project root (the folder that contains the models/ directory):

    python benchmark.py

It loads the served multiclass model, then times 500 predictions and reports
model size, feature width, cold start, peak memory, latency, and throughput.
The email text is processed in memory only and nothing is stored.
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")  # keep the console output clean for the screenshot

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "webapp"))
MODELS = os.path.join(ROOT, "models")

from inference import PhishGuardMulticlass  # noqa: E402

N = 500

SAMPLES = [
    {"sender": "security@paypa1-verify.com",
     "subject": "Urgent: verify your account now",
     "body": "Your account will be suspended in 24 hours. Click http://paypa1-verify.com to confirm now."},
    {"sender": "hr@company.com",
     "subject": "Re: lunch on Thursday",
     "body": "Hi Sam, Thursday works for me. Let us meet at the usual place around noon."},
    {"sender": "billing@invoice-portal.net",
     "subject": "Invoice overdue - immediate payment required",
     "body": "Please review the attached invoice and pay immediately to avoid a late fee."},
]


def peak_rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        try:
            import resource
            kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return kb / 1024  # Linux reports KB
        except Exception:
            return None


def main():
    size_bytes = (os.path.getsize(os.path.join(MODELS, "20_multiclass_model.joblib"))
                  + os.path.getsize(os.path.join(MODELS, "20_multiclass_vectorizer.joblib")))

    t0 = time.perf_counter()
    model = PhishGuardMulticlass(MODELS)
    cold_start_ms = (time.perf_counter() - t0) * 1000

    # warm up (excluded from timing) and capture the feature vector width
    width = model.predict(**SAMPLES[0])["n_features"]

    lat = []
    for i in range(N):
        s = SAMPLES[i % len(SAMPLES)]
        t = time.perf_counter()
        model.predict(**s)
        lat.append((time.perf_counter() - t) * 1000)
    lat.sort()
    mean = sum(lat) / len(lat)
    p50 = lat[int(0.50 * len(lat))]
    p95 = lat[int(0.95 * len(lat))]
    rss = peak_rss_mb()

    print(f"PhishGuard performance benchmark  (n = {N} predictions)")
    print("-" * 54)
    print()
    W = 27
    print(f"{'model + vectorizer on disk':<{W}}: {size_bytes / 1024:.1f} KB")
    print(f"{'feature vector width':<{W}}: {width}")
    print(f"{'cold start (load model)':<{W}}: {cold_start_ms:.1f} ms")
    print(f"{'peak process memory (RSS)':<{W}}: " + (f"{rss:.1f} MB" if rss else "n/a (pip install psutil)"))
    print(f"{'mean inference latency':<{W}}: {mean:.2f} ms")
    print(f"{'median (p50) latency':<{W}}: {p50:.2f} ms")
    print(f"{'p95 latency':<{W}}: {p95:.2f} ms")
    print(f"{'throughput':<{W}}: {1000 / mean:.0f} predictions per second")


if __name__ == "__main__":
    main()
