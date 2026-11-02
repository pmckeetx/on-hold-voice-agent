import asyncio
import base64
import json

import pytest

import agent
from fake_realtime import FakeRealtime


class StubLine:
    """A phone line that sends silence and remembers what the agent did to it."""

    def __init__(self, seconds: float = 5.0):
        self.seconds = seconds
        self.played = 0
        self.cleared = 0
        self.pressed = ""
        self.closed = asyncio.Event()

    async def frames(self):
        for _ in range(int(self.seconds / 0.02)):
            if self.closed.is_set():
                return
            await asyncio.sleep(0.01)  # twice real speed
            yield b"\xff" * 160  # 20 ms of mu-law silence

    def play(self, pcm: bytes):
        self.played += len(pcm)

    def clear(self):
        self.cleared += 1

    def press(self, digits: str):
        self.pressed += digits

    async def finish_playing(self):
        pass

    def hang_up(self):
        self.closed.set()


PHARMACY_CALL = [
    (0.2, "For prescriptions, press 2.", ("tool", "press_keys", {"digits": "2"}, None)),
    (0.4, "Please hold.", ("tool", "wait", {}, None)),
    (0.6, "Your call is important to us.", ("tool", "wait", {}, None)),
    (0.8, "This is Sam. How can I help?", ("say", "Hi Sam, I'm an automated assistant calling for Jane Porter.")),
    (1.0, "Yes, it's ready.", ("tool", "end_call", {"result": "Ready after two.", "goodbye": "Thanks, bye."},
                               "Thanks, bye.")),
]


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("POLY_API_KEY", "test-key")


async def test_menu_hold_and_pickup(key, monkeypatch, capsys):
    async with FakeRealtime(PHARMACY_CALL) as fake:
        monkeypatch.setattr(agent, "URL", fake.url)
        line = StubLine()
        caller = await agent.call(line)

    session = fake.sent("session.update")[0]["session"]
    assert session["audio"]["input"]["format"] == {"type": "audio/pcmu"}
    assert [t["name"] for t in session["tools"]] == ["wait", "press_keys", "end_call"]
    assert line.pressed == "2"
    assert line.played > 0
    assert line.cleared == 5  # every turn the server heard
    assert caller.result == "Ready after two."
    assert line.closed.is_set()  # the agent hung up before the line ran out

    outputs = fake.sent("conversation.item.create")
    assert [o["item"]["call_id"] for o in outputs] == ["call_1", "call_2", "call_3", "call_5"]
    assert json.loads(outputs[0]["item"]["output"]) == {"pressed": "2"}
    assert not fake.sent("response.create")  # tool results never ask for a reply

    out = capsys.readouterr().out
    assert "(presses 2)" in out and out.count("(waits)") == 2 and "(hangs up)" in out


async def test_frames_go_out_as_base64_mu_law(key, monkeypatch):
    async with FakeRealtime([]) as fake:
        monkeypatch.setattr(agent, "URL", fake.url)
        await agent.call(StubLine(seconds=0.2))
    frames = fake.sent("input_audio_buffer.append")
    assert len(frames) == 10
    assert base64.b64decode(frames[0]["audio"]) == b"\xff" * 160  # 20 ms of mu-law silence


async def test_unknown_tool_gets_an_error_result(key, monkeypatch):
    turns = [(0.1, "Hello?", ("tool", "transfer_call", {}, None))]
    async with FakeRealtime(turns) as fake:
        monkeypatch.setattr(agent, "URL", fake.url)
        await agent.call(StubLine(seconds=0.4))
    output = json.loads(fake.sent("conversation.item.create")[0]["item"]["output"])
    assert output == {"error": "no tool called transfer_call"}


async def test_missing_key_is_one_line(monkeypatch):
    monkeypatch.delenv("POLY_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="POLY_API_KEY isn't set"):
        await agent.call(StubLine())
