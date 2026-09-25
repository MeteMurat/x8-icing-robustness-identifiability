"""Sanity-check selected published values and deterministic restatements.

This script does not replace the full flight-data analysis. It verifies internal
arithmetic consistency among selected values reported in the manuscript.
"""
from math import isclose

n_total = 18
n_holm = 3
assert isclose(100*n_holm/n_total, 16.6666666667, rel_tol=0, abs_tol=1e-8)

d_cfg = 0.83568
ci_cfg = (0.69191, 1.25995)
d_unity = 2.66970
ci_unity = (2.36030, 2.98004)

assert ci_cfg[0] > 0
assert ci_unity[0] > 0
assert d_unity / d_cfg > 3.0

support_coverage = 95.39
unsupported = 100.0 - support_coverage
assert isclose(unsupported, 4.61, abs_tol=1e-12)

print("PASS: selected manuscript-reported arithmetic checks.")
print(f"Holm+sensitivity retention: {100*n_holm/n_total:.1f}%")
print(f"Outside primary shared support: {unsupported:.2f}%")
print(f"Unity/configuration-specific drag ratio: {d_unity/d_cfg:.3f}")