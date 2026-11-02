"""One real call to the scripted pharmacy, against the real API. Skipped unless POLYAI_LIVE=1.

    POLYAI_LIVE=1 uv run pytest -m live
"""
import os

import pytest

import agent
from line import PharmacyLine

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("POLYAI_LIVE") != "1", reason="set POLYAI_LIVE=1 to call the real API"),
]


async def test_a_real_call_to_the_pharmacy():
    line = PharmacyLine(hold_seconds=10, record_to=None)
    caller = await agent.call(line)
    summary = line.summary()
    assert summary["pressed"].startswith("2")  # it picked prescriptions from the menu
    assert summary["spoke_on_hold"] == []  # it said nothing to the hold queue
    assert summary["answered_sam_after"] is not None  # and it answered the person
    assert caller.result  # it hung up with an answer
