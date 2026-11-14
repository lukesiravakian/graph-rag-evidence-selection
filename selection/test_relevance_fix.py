"""Smoke test: the relevance term pulls in a candidate coverage alone would skip.

Pool of four: 0 and 1 are near-duplicates, 2 and 3 are near-duplicates, and 2 is the
only one that is actually relevant to the query. Coverage alone has no reason to prefer
2 over 3; with alpha = 0.7 the relevance term decides it.
"""
from pathlib import Path
from sys import path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in path:
    path.insert(0, str(PROJECT_ROOT))

from numpy import array

from selection.facility_location import (
    facility_location_select,
    relevance_weighted_facility_location_select,
)

fake_matrix = array([
    [1.0, 0.9, 0.1, 0.1],
    [0.9, 1.0, 0.1, 0.1],
    [0.1, 0.1, 1.0, 0.8],
    [0.1, 0.1, 0.8, 1.0],
])
fake_query_sims = array([0.1, 0.1, 0.9, 0.2])

weighted = relevance_weighted_facility_location_select(
    fake_matrix, fake_query_sims, k=2, alpha=0.7
)
print(f"alpha = 0.7 selected indices: {weighted}")
print("Expected: index 2 should be included (it's the genuinely relevant one)")
assert 2 in weighted, f"relevance term did not pull in index 2: got {weighted}"

# alpha = 0 must reproduce the coverage-only selector exactly.
coverage_only = relevance_weighted_facility_location_select(
    fake_matrix, fake_query_sims, k=2, alpha=0.0
)
assert coverage_only == facility_location_select(fake_matrix, k=2), (
    f"alpha = 0 diverged from facility_location_select: "
    f"{coverage_only} vs {facility_location_select(fake_matrix, k=2)}"
)
print(f"alpha = 0   selected indices: {coverage_only} (matches facility_location_select)")
print("OK")
