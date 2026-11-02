"""The other end of the call: Riverside Pharmacy's phone menu, its hold queue and Sam, the pharmacist.

PharmacyLine stands in for a real phone line, so you can run the agent without dialing anyone. It sends 8 kHz
G.711 mu-law in 20 ms frames, in real time, the same audio a phone trunk or a Twilio media stream carries. It
hears the agent too: Sam waits for the agent to finish talking before he answers, and the menu waits for a key.
At the end it writes the whole call, both sides, to a WAV file you can listen to.
"""
import asyncio
import time
import wave
from pathlib import Path

import numpy as np

RATE = 8000
FRAME = RATE // 50  # 20 ms
SAMPLES = Path(__file__).parent / "samples"
KEYPAD = {"1": (697, 1209), "2": (697, 1336), "3": (697, 1477), "4": (770, 1209), "5": (770, 1336),
          "6": (770, 1477), "7": (852, 1209), "8": (852, 1336), "9": (852, 1477), "*": (941, 1209),
          "0": (941, 1336), "#": (941, 1477)}


def load(name: str) -> np.ndarray:
    with wave.open(str(SAMPLES / f"{name}.wav")) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768


def hold_music(seconds: float) -> np.ndarray:
    """A looping four-chord pad with a plucked arpeggio, the generic sound of being on hold."""
    bar = 2.4
    chords = [[261.63, 329.63, 392.0], [220.0, 261.63, 329.63], [174.61, 220.0, 261.63], [196.0, 246.94, 293.66]]
    t = np.arange(int(bar * RATE)) / RATE
    fade = np.minimum(1, t / 0.3) * np.minimum(1, (bar - t) / 0.3)
    loop = []
    for chord in chords:
        pad = sum(np.sin(2 * np.pi * f * t) for f in chord) * fade / 3
        step = len(t) // 8
        pluck = np.concatenate([np.sin(2 * np.pi * chord[k % 3] * 2 * t[:step]) * np.exp(-t[:step] * 6)
                                for k in range(8)])
        loop.append(0.5 * pad + 0.8 * np.pad(pluck, (0, len(t) - len(pluck))))
    music = np.tile(np.concatenate(loop), int(seconds / (bar * 4)) + 1)[:int(seconds * RATE)]
    return (0.1 * music / np.abs(music).max()).astype(np.float32)


def tone(digit: str) -> np.ndarray:
    lo, hi = KEYPAD[digit]
    t = np.arange(int(0.15 * RATE)) / RATE
    return (0.2 * (np.sin(2 * np.pi * lo * t) + np.sin(2 * np.pi * hi * t))).astype(np.float32)


def to_ulaw(x: np.ndarray) -> bytes:
    """Encode float samples as G.711 mu-law, one byte per sample."""
    s = np.clip(np.round(x * 32767), -32768, 32767).astype(np.int32)
    sign = np.where(s < 0, 0x80, 0)
    magnitude = np.minimum(np.abs(s), 32635) + 0x84
    exponent = np.maximum(np.floor(np.log2(magnitude)).astype(np.int32) - 7, 0)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8).tobytes()


def from_pcm24(pcm: bytes) -> np.ndarray:
    """The agent's 24 kHz PCM16 as 8 kHz floats, the way a phone line would carry it."""
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    x = x[:len(x) // 3 * 3].reshape(-1, 3).mean(axis=1)
    return x


class PharmacyLine:
    def __init__(self, hold_seconds: float = 30.0, record_to: str | None = "call.wav"):
        self.hold_seconds = hold_seconds
        self.record_to = record_to
        self.queue = np.zeros(0, dtype=np.float32)  # what the pharmacy will send next
        self.sent: list[np.ndarray] = []  # every frame the pharmacy sent, for the recording
        self.agent: list[tuple[float, np.ndarray]] = []  # (start, samples) of everything the agent sent
        self.agent_until = 0.0  # call time when the agent's audio so far finishes playing
        self.agent_started = -1.0  # call time the agent's latest stretch of speech started
        self.stretches: list[float] = []  # call times the agent started talking
        self.keys: asyncio.Queue[str] = asyncio.Queue()
        self.marks: dict[str, tuple[float, float]] = {}  # clip -> (start, end of speech) in call time
        self.pressed: list[tuple[float, str]] = []
        self.closed = asyncio.Event()
        self.t0 = 0.0

    def now(self) -> float:
        return time.monotonic() - self.t0 if self.t0 else 0.0

    # The phone line, as the agent sees it

    async def frames(self):
        """Yield the pharmacy's audio as 20 ms mu-law frames, in real time, until either side hangs up."""
        self.t0 = time.monotonic()
        script = asyncio.create_task(self.script())
        n = 0
        try:
            while not self.closed.is_set():
                await asyncio.sleep(max(0.0, self.t0 + n * 0.02 - time.monotonic()))
                frame, self.queue = self.queue[:FRAME], self.queue[FRAME:]
                frame = np.pad(frame, (0, FRAME - len(frame)))
                self.sent.append(frame)
                n += 1
                yield to_ulaw(frame)
        finally:
            script.cancel()
            self.save()

    def play(self, pcm24: bytes):
        """The agent's voice, arriving down the line."""
        x = from_pcm24(pcm24)
        now = self.now()
        if self.agent_until < now:
            self.agent_started = now
            self.stretches.append(round(now, 2))
        start = max(now, self.agent_until)
        self.agent.append((start, x))
        self.agent_until = start + len(x) / RATE

    def clear(self):
        """Stop the agent's audio that hasn't played yet: the other side started talking over it."""
        now = self.now()
        self.agent = [(s, x[:max(0, int((now - s) * RATE))]) for s, x in self.agent]
        self.agent_until = min(self.agent_until, now)

    def press(self, digits: str):
        for d in digits:
            if d in KEYPAD:
                self.agent.append((max(self.now(), self.agent_until), tone(d)))
                self.pressed.append((round(self.now(), 2), d))
                self.keys.put_nowait(d)

    async def finish_playing(self):
        while self.now() < self.agent_until:
            await asyncio.sleep(0.05)

    def hang_up(self):
        self.closed.set()

    # The pharmacy's side

    async def say(self, name: str):
        await self.send(load(name), name)

    async def music(self, seconds: float):
        await self.send(hold_music(seconds), None)

    async def send(self, x: np.ndarray, name: str | None):
        start = self.now() + len(self.queue) / RATE
        self.queue = np.concatenate([self.queue, x])
        end = start + len(x) / RATE
        if name:
            speech = np.nonzero(np.abs(x) > 0.01)[0]  # the clip's last loud sample, not its trailing silence
            self.marks[name] = (round(start, 2), round(start + (speech[-1] + 1) / RATE, 2))
        while self.now() < end:
            await asyncio.sleep(0.02)

    async def key(self, timeout: float) -> str | None:
        try:
            return await asyncio.wait_for(self.keys.get(), timeout)
        except asyncio.TimeoutError:
            return None

    async def reply(self, timeout: float, quiet: float = 1.0) -> bool:
        """Wait like a person on the phone: for the agent to say something and then stop for a second."""
        asked = self.now()
        while True:
            await asyncio.sleep(0.05)
            spoke = self.agent_started >= asked - 0.5
            if spoke and self.now() > self.agent_until + quiet:
                return True
            if not spoke and self.now() - asked > timeout:
                return False

    async def script(self):
        await asyncio.sleep(0.5)
        for _ in range(3):
            await self.say("menu")
            key = await self.key(timeout=6)
            if key == "2":
                break
            if key == "1":
                await self.say("hours")
        await self.say("connecting")
        await self.music(self.hold_seconds * 0.5)
        await self.say("still-holding")
        await self.music(self.hold_seconds * 0.5)
        await self.say("sam-hello")
        await self.reply(timeout=10)
        await self.say("sam-spell")
        await self.reply(timeout=10)
        await self.say("sam-check")
        await self.reply(timeout=3)
        await self.music(8)
        await self.say("sam-ready")
        await self.reply(timeout=10)
        await self.say("sam-bye")
        await asyncio.sleep(3)
        self.hang_up()

    # After the call

    def summary(self) -> dict:
        """How the agent behaved: what it pressed, whether it spoke on hold, how fast it answered Sam."""
        hello = self.marks.get("sam-hello")
        on_hold = (self.marks["connecting"][0], hello[0]) if hello and "connecting" in self.marks else None
        spoke = self.stretches
        return {
            "pressed": "".join(d for _, d in self.pressed),
            "seconds_on_hold": round(on_hold[1] - on_hold[0], 1) if on_hold else None,
            "spoke_on_hold": [round(s, 1) for s in spoke if on_hold and on_hold[0] <= s < on_hold[1]],
            "answered_sam_after": round(min((s for s in spoke if s >= hello[1]), default=0) - hello[1], 2)
            if hello and any(s >= hello[1] for s in spoke) else None,
        }

    def save(self):
        if not self.record_to:
            return
        mix = np.concatenate(self.sent) if self.sent else np.zeros(0, dtype=np.float32)
        for start, x in self.agent:
            i = int(start * RATE)
            if i + len(x) > len(mix):
                mix = np.pad(mix, (0, i + len(x) - len(mix)))
            mix[i:i + len(x)] += x
        with wave.open(self.record_to, "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(RATE)
            w.writeframes(np.clip(np.round(mix * 32767), -32768, 32767).astype("<i2").tobytes())
