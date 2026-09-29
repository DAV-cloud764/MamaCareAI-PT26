"""
API schema tests.

`schemas.py` is the boundary the review UI is written against, so its job is to
reject a malformed request before a service ever sees it. These tests cover the
validation that can happen without a database: shape and internal consistency
of the submitted units.

The other half of the contract — that the submitted `base_version_number` still
matches stored state — cannot be tested here, because a Pydantic model has no
database access. That half belongs to
`VersionRepository.save_version_if_current` and is covered by
`test_review_persistence.py`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from modules.pipeline.api.schemas import EditRequest, UnitSchema

UNIT_ONE = UnitSchema(order=0, source_text="Dalili", translated_text="Dalili")
UNIT_TWO = UnitSchema(order=1, source_text="Afya", translated_text="Afya")


def make_request(*units: UnitSchema, base_version_number: int = 2) -> EditRequest:
    return EditRequest(
        units=list(units), base_version_number=base_version_number, note=None
    )


def test_accepts_a_well_formed_edit() -> None:
    request = make_request(UNIT_ONE, UNIT_TWO)

    assert [unit.order for unit in request.units] == [0, 1]
    assert request.base_version_number == 2


def test_rejects_an_edit_with_no_units() -> None:
    with pytest.raises(ValidationError, match="at least one unit"):
        make_request()


def test_rejects_duplicate_orders() -> None:
    with pytest.raises(ValidationError, match="duplicate unit order"):
        make_request(UNIT_ONE, UNIT_ONE.model_copy())


def test_rejects_out_of_order_units() -> None:
    with pytest.raises(ValidationError, match="ascending order"):
        make_request(UNIT_TWO, UNIT_ONE)


def test_rejects_a_base_version_number_below_one() -> None:
    """Version numbering starts at 1 (the machine translation).

    A 0 here means the client never loaded a version at all, which is a
    different bug from a stale one and should not read as a valid edit.
    """
    with pytest.raises(ValidationError):
        make_request(UNIT_ONE, base_version_number=0)


def test_rejects_a_non_integer_base_version_number() -> None:
    with pytest.raises(ValidationError):
        EditRequest(units=[UNIT_ONE], base_version_number="two")  # type: ignore[arg-type]


def test_unit_kind_defaults_to_paragraph_so_existing_clients_still_validate() -> None:
    """`review.html` posts units without `kind`; that must keep working.

    The field was added so a client can distinguish a heading from a list item.
    Defaulting rather than requiring it means adding the field did not break a
    reviewer mid-edit.
    """
    assert UnitSchema(order=0, source_text="a", translated_text="a").kind == "paragraph"


def test_unit_kind_is_carried_when_supplied() -> None:
    unit = UnitSchema(order=0, source_text="a", translated_text="a", kind="heading")

    assert unit.kind == "heading"
