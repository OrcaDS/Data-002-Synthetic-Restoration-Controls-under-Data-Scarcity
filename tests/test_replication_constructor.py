from __future__ import annotations

import pandas as pd
import pytest

from data002.replication_constructor import construct_replication_control


def retained() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": [10.0, 20.0, 30.0, 40.0, 50.0],
            "target": [0, 1, 0, 1, 0],
        },
        index=pd.Index([8, 3, 12, 5, 20], dtype="int64"),
    )


def construct(frame: pd.DataFrame | None = None, final_size: int = 17):
    return construct_replication_control(
        retained() if frame is None else frame,
        target_column="target",
        final_size=final_size,
        dataset="diabetes",
        retained_fraction=0.01,
        split_seed=29,
    )


def test_constructor_preserves_originals_and_orders_class_blocks() -> None:
    source = retained()
    result, record = construct(source)

    assert list(result.index[: len(source)]) == list(source.index)
    duplicate_labels = result["target"].tolist()[len(source) :]
    assert duplicate_labels == sorted(duplicate_labels)
    assert record["class_blocks"][0]["class_label"] == 0
    assert record["class_blocks"][1]["class_label"] == 1
    assert len(result) == 17


def test_constructor_has_exact_counts_membership_and_balance() -> None:
    source = retained()
    result, record = construct(source, 24)

    assert set(result.index) == set(source.index)
    assert result["target"].value_counts().sort_index().to_dict() == {
        int(key): value for key, value in record["final_target_counts"].items()
    }
    for block in record["class_blocks"]:
        assert (
            block["maximum_duplicates_per_source"]
            - block["minimum_duplicates_per_source"]
            <= 1
        )


def test_constructor_is_bitwise_record_and_table_deterministic() -> None:
    first_table, first_record = construct()
    second_table, second_record = construct()

    pd.testing.assert_frame_equal(first_table, second_table)
    assert first_record == second_record


@pytest.mark.parametrize("labels", [[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [False, True, False, True, False]])
def test_constructor_rejects_invalid_labels(labels) -> None:
    frame = retained()
    frame["target"] = labels

    with pytest.raises(ValueError, match="exactly integer"):
        construct(frame)


def test_constructor_rejects_duplicate_source_indices() -> None:
    frame = retained()
    frame.index = [1, 1, 2, 3, 4]

    with pytest.raises(ValueError, match="unique"):
        construct(frame)


def test_constructor_rejects_negative_source_index() -> None:
    frame = retained()
    frame.index = [-1, 1, 2, 3, 4]

    with pytest.raises(ValueError, match="nonnegative"):
        construct(frame)


def test_constructor_rejects_noninteger_source_index() -> None:
    frame = retained()
    frame.index = ["a", "b", "c", "d", "e"]

    with pytest.raises(ValueError, match="integers"):
        construct(frame)
