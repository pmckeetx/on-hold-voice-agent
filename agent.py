"""An outbound voice agent on Dialog-RSN-1 that calls a pharmacy, gets through the phone menu, waits on hold in
silence and talks to the person who picks up.

    uv run agent.py              # call the scripted Riverside Pharmacy line, and save the call to call.wav
    uv run agent.py --hold 120   # two minutes of hold music instead of 30 seconds
"""
import argparse
import asyncio
import base64
import json
import os
import time

import websockets
from dotenv import load_dotenv

from line import PharmacyLine

load_dotenv()

URL = os.environ.get("DIALOG_REALTIME_URL", "wss://api.studio.poly.ai/v1/realtime")

INSTRUCTIONS = """You are an assistant making a phone call on behalf of a patient, Jane Porter, date of birth
4 May 1982. You're calling Riverside Pharmacy to ask whether her prescription is ready to collect.

You'll often reach a phone menu or a hold queue before a person. Pick the menu option for prescriptions with
press_keys. While you're on hold, or hear hold music or a recorded message, call wait and say nothing. Never
talk to a recording.

When a person answers, tell them you're an automated assistant calling for Jane Porter, and why, in one or
two short sentences. Answer their questions about the patient. If they ask you to hold, say "Sure" and wait.
Once you have the answer, thank them and call end_call."""

TOOLS = [
    {
        "type": "function",
        "name": "wait",
        "description": "Stay silent and keep listening. Call this instead of speaking whenever you're on hold, "
                       "or hear hold music, a recorded message, or silence on the line.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "press_keys",
        "description": "Press keys on the phone keypad, to choose an option in an automated phone menu.",
        "parameters": {
            "type": "object",
            "properties": {"digits": {"type": "string", "description": "The keys to press, like '2' or '1#'."}},
            "required": ["digits"],
        },
    },
    {
        "type": "function",
        "name": "end_call",
        "description": "Say goodbye and hang up, once you have the answer you called for.",
        "parameters": {
            "type": "object",
            "properties": {
                "result": {"type": "string", "description": "What you found out, in one sentence, for the patient."},
                "goodbye": {"type": "string", "poly_spoken": True,
                            "description": "A short goodbye to say before hanging up, like 'Thanks, bye.'"},
            },
            "required": ["result", "goodbye"],
        },
    },
]


class Caller:
    def __init__(self, ws, line: PharmacyLine):
        self.ws = ws
        self.line = line
        self.result = None
        self.hanging_up = None
        self.t0 = time.monotonic()
        self.started: dict[str, float] = {}  # item or response id -> when it started, for the log

    def log(self, who: str, text: str, at: float | None = None):
        t = (at or time.monotonic()) - self.t0
        print(f"{int(t // 60):02d}:{t % 60:04.1f}  {who:<9}{text}", flush=True)

    async def send(self, event: dict):
        await self.ws.send(json.dumps(event))

    async def start(self):
        await self.ws.recv()  # session.created
        await self.send({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": INSTRUCTIONS,
                "tools": TOOLS,
                "audio": {"input": {"format": {"type": "audio/pcmu"}, "turn_detection": {"type": "server_vad"}}},
            },
        })

    async def stream_line(self):
        """Forward the other end of the call to Dialog-RSN-1, 20 ms at a time, hold music and all."""
        async for frame in self.line.frames():
            await self.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(frame).decode()})

    async def listen(self):
        async for message in self.ws:
            event = json.loads(message)
            kind = event["type"]
            if kind == "response.output_audio.delta":
                self.started.setdefault(event["response_id"], time.monotonic())
                self.line.play(base64.b64decode(event["delta"]))
            elif kind == "input_audio_buffer.speech_started":
                self.started[event["item_id"]] = time.monotonic()
                self.line.clear()  # they're talking: stop any of our reply that hasn't played yet
            elif kind == "conversation.item.input_audio_transcription.completed":
                self.log("line", event["transcript"].strip(), self.started.get(event["item_id"]))
            elif kind == "response.output_audio_transcript.done":
                self.log("agent", event["transcript"], self.started.get(event["response_id"]))
            elif kind == "response.function_call_arguments.done":
                output = self.on_tool(event["name"], json.loads(event["arguments"] or "{}"))
                # Answer the call, but don't send response.create: the next thing on the line is the answer
                await self.send({"type": "conversation.item.create", "item": {
                    "type": "function_call_output", "call_id": event["call_id"], "output": json.dumps(output)}})
            elif kind == "error":
                self.log("error", event["error"].get("message", ""))

    def on_tool(self, name: str, args: dict) -> dict:
        if name == "wait":
            self.log("agent", "(waits)")
            return {"status": "waiting"}
        if name == "press_keys":
            self.log("agent", f"(presses {args['digits']})")
            self.line.press(args["digits"])
            return {"pressed": args["digits"]}
        if name == "end_call":
            self.result = args["result"]
            self.hanging_up = asyncio.create_task(self.hang_up_after_goodbye())
            return {"status": "hanging up"}
        return {"error": f"no tool called {name}"}

    async def hang_up_after_goodbye(self):
        await asyncio.sleep(1.0)  # the goodbye is spoken alongside the call, so let the last of it arrive
        await self.line.finish_playing()
        self.log("agent", "(hangs up)")
        self.line.hang_up()


async def call(line) -> Caller:
    """Place the call: stream the line to Dialog-RSN-1 until one side hangs up."""
    key = os.environ.get("POLY_API_KEY")
    if not key:
        raise SystemExit("POLY_API_KEY isn't set. Copy .env.example to .env and add your key from studio.poly.ai.")
    async with websockets.connect(URL, additional_headers={"Authorization": f"Bearer {key}"}) as ws:
        caller = Caller(ws, line)
        await caller.start()
        listener = asyncio.create_task(caller.listen())
        await caller.stream_line()
        listener.cancel()
    return caller


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hold", type=float, default=30, help="seconds of hold music before Sam picks up")
    args = parser.parse_args()

    line = PharmacyLine(hold_seconds=args.hold)
    try:
        caller = asyncio.run(call(line))
    except websockets.InvalidStatus as error:
        raise SystemExit(f"The server refused the connection (HTTP {error.response.status_code}). Check "
                         f"POLY_API_KEY, and set DIALOG_REALTIME_URL if your key is for a host other than {URL}.")
    print(f"\nResult: {caller.result}")
    print(f"Call summary: {json.dumps(line.summary())}")
    print("Saved the call to call.wav")


if __name__ == "__main__":
    main()
