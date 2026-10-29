import json
import math
import statistics


def wilson(k, n, z=1.96):
    """95% confidence range for a success rate (Wilson method)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def summarize(results):
    print("\n=========== BENCHMARK SUMMARY ===========")
    for cond, rs in results.items():
        valid = [r for r in rs if "error" not in r]
        n = len(valid)
        wins = [r for r in valid if r["success"]]
        first = [r for r in wins if r["attempts"] == 1]
        lo1, hi1 = wilson(len(wins), n)
        lo2, hi2 = wilson(len(first), n)
        avg_att = sum(r["attempts"] for r in wins) / len(wins) if wins else 0
        med_act = statistics.median(r["actions"] for r in wins) if wins else 0
        print(f"{cond:12s} (n={n}): success {len(wins)}/{n} [{lo1:.0%}-{hi1:.0%}]  "
              f"first-try {len(first)}/{n} [{lo2:.0%}-{hi2:.0%}]  "
              f"avg attempts {avg_att:.2f}  median actions {med_act}")


if __name__ == "__main__":
    summarize(json.load(open("benchmark_results.json")))