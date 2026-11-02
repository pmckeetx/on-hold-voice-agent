import asyncio
import time

import numpy as np
import pytest

import line
from line import PharmacyLine


def ulaw_reference(sample: int) -> int:
    """The textbook G.711 mu-law encoder, one sample at a time."""
    sign = 0x80 if sample < 0 else 0
    magnitude = min(abs(sample), 32635) + 0x84
    exponent = max(magnitude.bit_length() - 8, 0)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


def test_mu_law_matches_the_reference():
    samples = np.arange(-32768, 32768, 7)
    encoded = line.to_ulaw(samples.astype(np.float32) / 32767)
    expected = bytes(ulaw_reference(int(s)) for s in np.clip(np.round(samples / 32767 * 32767), -32768, 32767))
    assert encoded == expected
    assert line.to_ulaw(np.zeros(4, dtype=np.float32)) == b"\xff" * 4


def test_agent_audio_comes_down_to_8_khz():
    pcm = (np.ones(2400, dtype="<i2") * 1000).tobytes()  # 100 ms at 24 kHz
    x = line.from_pcm24(pcm)
    assert len(x) == 800
    assert x[0] == pytest.approx(1000 / 32768)


def test_hold_music_is_quiet_and_the_right_length():
    music = line.hold_music(5.0)
    assert len(music) == 5 * line.RATE
    assert 0.05 < np.abs(music).max() <= 0.1


def test_every_clip_is_there():
    for name in ["menu", "hours", "connecting", "still-holding", "sam-hello", "sam-spell", "sam-check",
                 "sam-ready", "sam-bye"]:
        assert len(line.load(name)) > line.RATE  # each is over a second of 8 kHz audio


async def test_play_queues_and_clear_stops():
    pharmacy = PharmacyLine(record_to=None)
    pharmacy.t0 = 1.0  # the call has started
    one_second = (np.ones(24000, dtype="<i2") * 1000).tobytes()
    pharmacy.play(one_second)
    pharmacy.play(one_second)
    assert pharmacy.agent_until - pharmacy.now() == pytest.approx(2.0, abs=0.05)
    assert len(pharmacy.stretches) == 1  # back to back is one stretch of speech
    pharmacy.clear()
    assert pharmacy.agent_until <= pharmacy.now()
    assert sum(len(x) for _, x in pharmacy.agent) < line.RATE // 10


async def test_keys_are_heard_and_recorded():
    pharmacy = PharmacyLine(record_to=None)
    pharmacy.t0 = 1.0
    pharmacy.press("2x")  # x isn't a key, so it's dropped
    assert await pharmacy.key(timeout=0.1) == "2"
    assert await pharmacy.key(timeout=0.1) is None
    assert [d for _, d in pharmacy.pressed] == ["2"]
    assert len(pharmacy.agent) == 1  # the key's tone, for the recording


async def test_sam_waits_for_the_agent_to_finish():
    pharmacy = PharmacyLine(record_to=None)
    pharmacy.t0 = time.monotonic()
    assert await pharmacy.reply(timeout=0.2) is False  # nobody answered
    waiting = asyncio.create_task(pharmacy.reply(timeout=2, quiet=0.2))
    await asyncio.sleep(0.1)
    pharmacy.play((np.ones(2400, dtype="<i2") * 1000).tobytes())  # 100 ms of agent
    assert await waiting is True


async def test_the_whole_call_is_recorded(tmp_path):
    out = tmp_path / "call.wav"
    pharmacy = PharmacyLine(hold_seconds=0.2, record_to=str(out))
    frames = 0
    async for _ in pharmacy.frames():
        frames += 1
        if frames == 1:
            pharmacy.play((np.ones(2400, dtype="<i2") * 1000).tobytes())
        if frames == 25:
            pharmacy.hang_up()
    assert out.stat().st_size > 44 + 25 * 160 * 2 - 1


def test_summary_from_a_finished_call():
    pharmacy = PharmacyLine(record_to=None)
    pharmacy.marks = {"connecting": (8.0, 11.0), "sam-hello": (48.0, 52.0)}
    pharmacy.pressed = [(7.4, "2")]
    pharmacy.stretches = [52.8, 61.0]
    assert pharmacy.summary() == {"pressed": "2", "seconds_on_hold": 40.0, "spoke_on_hold": [],
                                  "answered_sam_after": 0.8}


async def test_marks_end_where_the_speech_ends():
    pharmacy = PharmacyLine(record_to=None)
    pharmacy.t0 = time.monotonic()
    clip = np.concatenate([np.full(800, 0.5, dtype=np.float32), np.zeros(400, dtype=np.float32)])
    await pharmacy.send(clip, "hello")  # 100 ms of sound, then 50 ms of silence
    start, end = pharmacy.marks["hello"]
    assert end - start == pytest.approx(0.1, abs=0.011)
