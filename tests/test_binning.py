import numpy as np
import pandas as pd
import pytest

from interpretable_ts_vit import TimeSeriesBinner


def test_binner_global_bins_aggregation_and_mask():
    records = pd.DataFrame(
        [
            ["p1", "hr", 80.0, "2026-01-01 00:05:00"],
            ["p1", "hr", 100.0, "2026-01-01 00:20:00"],
            ["p1", "bp", 120.0, "2026-01-01 00:40:00"],
            ["p2", "hr", 70.0, "2026-01-01 00:10:00"],
        ],
        columns=["patient_id", "variable", "value", "timestamp"],
    )
    labels = pd.DataFrame({"patient_id": ["p1", "p2"], "label": ["case", "control"]})
    binner = TimeSeriesBinner(granularity="30min", time_start="2026-01-01 00:00:00", time_end="2026-01-01 01:00:00")
    binner.fit(records, labels)
    binned = binner.transform(records, labels)
    assert binner.variable_vocab_ == ["bp", "hr"]
    assert binned.x.shape == (2, 2, 2, 2)
    p1 = binned.patient_ids.index("p1")
    hr = binner.variable_vocab_.index("hr")
    bp = binner.variable_vocab_.index("bp")
    assert binned.x[p1, 1, hr, 0] == 1.0
    assert binned.x[p1, 1, bp, 1] == 1.0
    assert binned.x[p1, 1, bp, 0] == 0.0
    assert binned.x[p1, 0, bp, 0] == 0.0


def test_unknown_variables_do_not_change_shape():
    train = pd.DataFrame(
        [["p1", "hr", 80.0, "2026-01-01 00:00:00"]],
        columns=["patient_id", "variable", "value", "timestamp"],
    )
    labels = pd.DataFrame({"patient_id": ["p1"], "label": ["case"]})
    binner = TimeSeriesBinner(granularity="1h", time_start="2026-01-01", time_end="2026-01-02").fit(train, labels)
    test = pd.DataFrame(
        [["p2", "new_var", 1.0, "2026-01-01 00:00:00"]],
        columns=["patient_id", "variable", "value", "timestamp"],
    )
    out = binner.transform(test)
    assert out.x.shape == (1, 2, 1, 24)
    assert np.all(out.x == 0)


def test_unseen_label_fails_clearly():
    records = pd.DataFrame(
        [["p1", "hr", 80.0, "2026-01-01"]],
        columns=["patient_id", "variable", "value", "timestamp"],
    )
    binner = TimeSeriesBinner(time_start="2026-01-01", time_end="2026-01-02").fit(records, {"p1": "a"})
    with pytest.raises(ValueError, match="Labels not seen"):
        binner.transform(records, {"p1": "b"})
