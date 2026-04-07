from __future__ import annotations

from globalcy.data.bundle_adapter import load_bundle_batch


def test_bundle_adapter_smoke(synthetic_bundle):
    batch = load_bundle_batch(synthetic_bundle)
    assert batch.local_features.shape == (6, 6)
    assert batch.invariant_features.shape[0] == 6
    assert batch.metadata["case_id"] == "fermat_quartic"
    assert batch.symmetry_features.shape[-1] == batch.invariant_features.shape[-1] + 1
    assert batch.symmetry_metadata["has_orbits"] is True
