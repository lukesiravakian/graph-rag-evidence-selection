from numpy import array
from selection.facility_location import relevance_weighted_facility_location_select

fake_matrix = array([
    [1.0, 0.9, 0.1, 0.1],
    [0.9, 1.0, 0.1, 0.1],
    [0.1, 0.1, 1.0, 0.8],
    [0.1, 0.1, 0.8, 1.0],
])
fake_query_sims = array([0.1, 0.1, 0.9, 0.2])

result = relevance_weighted_facility_location_select(
    fake_matrix, fake_query_sims, k=2, alpha=0.7
)
print(f"Selected indices: {result}")
print("Expected: index 2 should be included (it's the genuinely relevant one)")