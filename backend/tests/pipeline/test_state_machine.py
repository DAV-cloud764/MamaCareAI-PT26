from __future__ import annotations

from itertools import pairwise

import pytest
from backend.modules.pipeline.domain.enums import ResourceStatus
from backend.modules.pipeline.domain.errors import InvalidStateTransition
from backend.modules.pipeline.domain.state_machine import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    assert_can_transition,
    can_transition,
)

S = ResourceStatus


def test_happy_path_is_fully_connected():
    path = [
        S.SUBMITTED, S.FETCHED, S.EXTRACTED, S.LANGUAGE_DETECTED,
        S.TRANSLATED, S.STORED, S.IN_REVIEW, S.APPROVED, S.PUBLISHED,
    ]
    assert all(can_transition(current, target) for current, target in pairwise(path))


def test_already_swahili_skips_translation():
    assert can_transition(S.LANGUAGE_DETECTED, S.STORED)


def test_review_loop():
    assert can_transition(S.IN_REVIEW, S.NEEDS_EDIT)
    assert can_transition(S.NEEDS_EDIT, S.EDITED)
    assert can_transition(S.EDITED, S.APPROVED)
    assert can_transition(S.EDITED, S.NEEDS_EDIT)


def test_cannot_unapprove():
    assert not can_transition(S.APPROVED, S.IN_REVIEW)


def test_cannot_skip_review():
    assert not can_transition(S.STORED, S.PUBLISHED)


def test_terminal_states_have_no_exits():
    assert all(not ALLOWED_TRANSITIONS[state] for state in TERMINAL_STATES)


def test_every_status_appears_in_the_map():
    assert set(ResourceStatus) == set(ALLOWED_TRANSITIONS)


def test_every_target_is_a_real_status():
    assert all(
        target in ResourceStatus
        for targets in ALLOWED_TRANSITIONS.values()
        for target in targets
    )


def test_assert_can_transition_raises_with_a_useful_message():
    with pytest.raises(InvalidStateTransition, match=r"approved.*in_review"):
        assert_can_transition(S.APPROVED, S.IN_REVIEW)


def test_all_states_reachable_from_submitted():
    reachable = {S.SUBMITTED}
    pending = [S.SUBMITTED]
    while pending:
        current = pending.pop()
        for target in ALLOWED_TRANSITIONS[current] - reachable:
            reachable.add(target)
            pending.append(target)

    assert set(ResourceStatus) - TERMINAL_STATES <= reachable