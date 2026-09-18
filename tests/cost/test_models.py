"""P3-C0 contract tests for immutable cost-gate models."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError


def _import_cost_models() -> Any:
    try:
        from auditor.cost import (
            Budget,
            BudgetLimit,
            WorkEstimate,
            WorkItem,
            WorkUnit,
        )
    except ImportError as exc:
        pytest.fail(
            "auditor.cost is not implemented yet; "
            f"P3-C0 models are unavailable ({exc})"
        )
    return Budget, BudgetLimit, WorkEstimate, WorkItem, WorkUnit


def test_work_unit_contains_only_current_phase2_units() -> None:
    _, _, _, _, WorkUnit = _import_cost_models()

    assert {member.name for member in WorkUnit} == {
        "RECORDS_TO_SCAN",
        "FILES_TO_OPEN",
    }
    assert {member.value for member in WorkUnit} == {
        "RECORDS_TO_SCAN",
        "FILES_TO_OPEN",
    }
    assert WorkUnit.RECORDS_TO_SCAN.value == "RECORDS_TO_SCAN"
    assert WorkUnit.FILES_TO_OPEN.value == "FILES_TO_OPEN"
    assert len(WorkUnit) == 2
    assert isinstance(WorkUnit.RECORDS_TO_SCAN, str)
    assert isinstance(WorkUnit.FILES_TO_OPEN, str)


def test_work_item_accepts_zero_positive_and_unknown_magnitudes() -> None:
    _, _, _, WorkItem, WorkUnit = _import_cost_models()

    zero = WorkItem(unit=WorkUnit.RECORDS_TO_SCAN, magnitude=0)
    positive = WorkItem(unit=WorkUnit.FILES_TO_OPEN, magnitude=12)
    unknown = WorkItem(unit=WorkUnit.RECORDS_TO_SCAN, magnitude=None)

    assert zero == WorkItem(unit=WorkUnit.RECORDS_TO_SCAN, magnitude=0)
    assert positive == WorkItem(unit=WorkUnit.FILES_TO_OPEN, magnitude=12)
    assert unknown == WorkItem(unit=WorkUnit.RECORDS_TO_SCAN, magnitude=None)
    assert zero.magnitude == 0
    assert positive.magnitude == 12
    assert unknown.magnitude is None

    with pytest.raises(ValidationError):
        WorkItem(unit=WorkUnit.RECORDS_TO_SCAN)


def test_work_item_rejects_negative_magnitude() -> None:
    _, _, _, WorkItem, WorkUnit = _import_cost_models()

    with pytest.raises(ValidationError):
        WorkItem(unit=WorkUnit.RECORDS_TO_SCAN, magnitude=-1)


def test_work_item_rejects_bool_float_and_string_magnitudes() -> None:
    _, _, _, WorkItem, WorkUnit = _import_cost_models()

    for invalid in (True, False, 1.0, 2.5, "3"):
        with pytest.raises(ValidationError):
            WorkItem(unit=WorkUnit.RECORDS_TO_SCAN, magnitude=invalid)


def test_work_item_forbids_extra_fields_and_is_frozen() -> None:
    _, _, _, WorkItem, WorkUnit = _import_cost_models()

    with pytest.raises(ValidationError):
        WorkItem(unit=WorkUnit.RECORDS_TO_SCAN, magnitude=1, extra="nope")

    item = WorkItem(unit=WorkUnit.FILES_TO_OPEN, magnitude=1)
    with pytest.raises(ValidationError):
        item.magnitude = 2  # type: ignore[misc]


def test_work_estimate_normalizes_items_by_unit() -> None:
    _, _, WorkEstimate, WorkItem, WorkUnit = _import_cost_models()

    expected = (
        WorkItem(unit=WorkUnit.FILES_TO_OPEN, magnitude=2),
        WorkItem(unit=WorkUnit.RECORDS_TO_SCAN, magnitude=10),
    )
    estimate_from_list = WorkEstimate(
        items=[
            WorkItem(unit=WorkUnit.RECORDS_TO_SCAN, magnitude=10),
            WorkItem(unit=WorkUnit.FILES_TO_OPEN, magnitude=2),
        ]
    )
    estimate_from_tuple = WorkEstimate(
        items=(
            WorkItem(unit=WorkUnit.RECORDS_TO_SCAN, magnitude=10),
            WorkItem(unit=WorkUnit.FILES_TO_OPEN, magnitude=2),
        )
    )

    assert isinstance(estimate_from_list.items, tuple)
    assert estimate_from_list.items == expected
    assert [item.unit.value for item in estimate_from_list.items] == [
        "FILES_TO_OPEN",
        "RECORDS_TO_SCAN",
    ]
    assert isinstance(estimate_from_tuple.items, tuple)
    assert estimate_from_tuple.items == expected
    assert estimate_from_tuple == estimate_from_list

    with pytest.raises(ValidationError):
        WorkEstimate(
            items=[WorkItem(unit=WorkUnit.RECORDS_TO_SCAN, magnitude=1)],
            extra="nope",
        )


def test_work_estimate_rejects_duplicate_units() -> None:
    _, _, WorkEstimate, WorkItem, WorkUnit = _import_cost_models()

    with pytest.raises(ValidationError):
        WorkEstimate(
            items=[
                WorkItem(unit=WorkUnit.RECORDS_TO_SCAN, magnitude=1),
                WorkItem(unit=WorkUnit.RECORDS_TO_SCAN, magnitude=2),
            ]
        )


def test_work_estimate_allows_empty_items() -> None:
    _, _, WorkEstimate, _, _ = _import_cost_models()

    estimate = WorkEstimate(items=[])
    assert estimate.items == ()
    assert isinstance(estimate.items, tuple)

    with pytest.raises(ValidationError):
        WorkEstimate()


def test_work_estimate_does_not_share_input_container() -> None:
    _, _, WorkEstimate, WorkItem, WorkUnit = _import_cost_models()

    items = [WorkItem(unit=WorkUnit.RECORDS_TO_SCAN, magnitude=1)]
    estimate = WorkEstimate(items=items)
    items.append(WorkItem(unit=WorkUnit.FILES_TO_OPEN, magnitude=2))

    assert estimate.items == (WorkItem(unit=WorkUnit.RECORDS_TO_SCAN, magnitude=1),)
    assert isinstance(estimate.items, tuple)

    with pytest.raises(ValidationError):
        estimate.items = ()  # type: ignore[misc]


def test_budget_limit_accepts_non_negative_strict_integer() -> None:
    _, BudgetLimit, _, _, WorkUnit = _import_cost_models()

    zero = BudgetLimit(unit=WorkUnit.RECORDS_TO_SCAN, maximum=0)
    positive = BudgetLimit(unit=WorkUnit.FILES_TO_OPEN, maximum=100)

    assert zero == BudgetLimit(unit=WorkUnit.RECORDS_TO_SCAN, maximum=0)
    assert positive == BudgetLimit(unit=WorkUnit.FILES_TO_OPEN, maximum=100)
    assert zero.maximum == 0
    assert positive.maximum == 100

    with pytest.raises(ValidationError):
        BudgetLimit(unit=WorkUnit.RECORDS_TO_SCAN, maximum=1, extra="nope")

    with pytest.raises(ValidationError):
        positive.maximum = 99  # type: ignore[misc]

    with pytest.raises(ValidationError):
        BudgetLimit(unit=WorkUnit.RECORDS_TO_SCAN)


def test_budget_limit_rejects_negative_bool_float_and_string_values() -> None:
    _, BudgetLimit, _, _, WorkUnit = _import_cost_models()

    for invalid in (-1, True, False, 1.0, "10"):
        with pytest.raises(ValidationError):
            BudgetLimit(unit=WorkUnit.RECORDS_TO_SCAN, maximum=invalid)


def test_budget_normalizes_limits_by_unit() -> None:
    Budget, BudgetLimit, _, _, WorkUnit = _import_cost_models()

    expected = (
        BudgetLimit(unit=WorkUnit.FILES_TO_OPEN, maximum=5),
        BudgetLimit(unit=WorkUnit.RECORDS_TO_SCAN, maximum=50),
    )
    budget_from_list = Budget(
        limits=[
            BudgetLimit(unit=WorkUnit.RECORDS_TO_SCAN, maximum=50),
            BudgetLimit(unit=WorkUnit.FILES_TO_OPEN, maximum=5),
        ]
    )
    budget_from_tuple = Budget(
        limits=(
            BudgetLimit(unit=WorkUnit.RECORDS_TO_SCAN, maximum=50),
            BudgetLimit(unit=WorkUnit.FILES_TO_OPEN, maximum=5),
        )
    )

    assert isinstance(budget_from_list.limits, tuple)
    assert budget_from_list.limits == expected
    assert [limit.unit.value for limit in budget_from_list.limits] == [
        "FILES_TO_OPEN",
        "RECORDS_TO_SCAN",
    ]
    assert isinstance(budget_from_tuple.limits, tuple)
    assert budget_from_tuple.limits == expected
    assert budget_from_tuple == budget_from_list

    with pytest.raises(ValidationError):
        Budget(
            limits=[BudgetLimit(unit=WorkUnit.RECORDS_TO_SCAN, maximum=1)],
            extra="nope",
        )


def test_budget_rejects_duplicate_units() -> None:
    Budget, BudgetLimit, _, _, WorkUnit = _import_cost_models()

    with pytest.raises(ValidationError):
        Budget(
            limits=[
                BudgetLimit(unit=WorkUnit.FILES_TO_OPEN, maximum=1),
                BudgetLimit(unit=WorkUnit.FILES_TO_OPEN, maximum=2),
            ]
        )


def test_budget_allows_empty_limits() -> None:
    Budget, _, _, _, _ = _import_cost_models()

    budget = Budget(limits=[])
    assert budget.limits == ()
    assert isinstance(budget.limits, tuple)

    with pytest.raises(ValidationError):
        Budget()


def test_budget_does_not_share_input_container_and_is_frozen() -> None:
    Budget, BudgetLimit, _, _, WorkUnit = _import_cost_models()

    limits = [BudgetLimit(unit=WorkUnit.RECORDS_TO_SCAN, maximum=3)]
    budget = Budget(limits=limits)
    limits.append(BudgetLimit(unit=WorkUnit.FILES_TO_OPEN, maximum=4))

    assert budget.limits == (BudgetLimit(unit=WorkUnit.RECORDS_TO_SCAN, maximum=3),)
    assert isinstance(budget.limits, tuple)

    with pytest.raises(ValidationError):
        budget.limits = ()  # type: ignore[misc]


def test_models_round_trip_through_json_without_losing_tuple_or_enum_semantics() -> (
    None
):
    Budget, BudgetLimit, WorkEstimate, WorkItem, WorkUnit = _import_cost_models()

    estimate = WorkEstimate(
        items=[
            WorkItem(unit=WorkUnit.FILES_TO_OPEN, magnitude=None),
            WorkItem(unit=WorkUnit.RECORDS_TO_SCAN, magnitude=7),
        ]
    )
    budget = Budget(
        limits=[
            BudgetLimit(unit=WorkUnit.FILES_TO_OPEN, maximum=0),
            BudgetLimit(unit=WorkUnit.RECORDS_TO_SCAN, maximum=7),
        ]
    )

    restored_estimate = WorkEstimate.model_validate_json(estimate.model_dump_json())
    restored_budget = Budget.model_validate_json(budget.model_dump_json())

    assert restored_estimate == estimate
    assert restored_budget == budget
    assert isinstance(restored_estimate.items, tuple)
    assert isinstance(restored_budget.limits, tuple)
    assert restored_estimate.items[0].unit is WorkUnit.FILES_TO_OPEN
    assert restored_estimate.items[1].unit is WorkUnit.RECORDS_TO_SCAN
    assert restored_budget.limits[0].unit is WorkUnit.FILES_TO_OPEN
    assert restored_budget.limits[1].unit is WorkUnit.RECORDS_TO_SCAN
    assert restored_estimate.items[0].unit == WorkUnit.FILES_TO_OPEN
    assert restored_budget.limits[1].unit == WorkUnit.RECORDS_TO_SCAN
    assert [item.unit.value for item in restored_estimate.items] == [
        "FILES_TO_OPEN",
        "RECORDS_TO_SCAN",
    ]
    assert [limit.unit.value for limit in restored_budget.limits] == [
        "FILES_TO_OPEN",
        "RECORDS_TO_SCAN",
    ]
