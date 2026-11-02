"""A small stand-in for the Dialog-RSN-1 realtime endpoint, so the tests run with no key and no network.

It plays a script: once the client has streamed a given amount of audio, the server "hears" a turn and answers
it the way the real one did on a call to the scripted pharmacy (api.studio.poly.ai, November 2026). A tool call's
response completes as soon as the call is sent, a spoken reply streams audio and its transcript, and a
poly_spoken argument is spoken in the same response as the call.
"""
import base64
import json

import websockets

FRAME_SECONDS = 0.02


def audio(seconds: float) -> str:
    return base64.b64encode(b"\x10\x00" * int(24000 * seconds)).decode()


class FakeRealtime:
    def __init__(self, turns):
        """turns: (seconds of audio heard, what the line said, what the agent does), in order. The agent does
        ("say", text) or ("tool", name, arguments, spoken_text_or_None)."""
        self.turns = list(turns)
        self.received: list[dict] = []
        self.url = None

    def sent(self, event_type: str) -> list[dict]:
        return [e for e in self.received if e["type"] == event_type]

    async def __aenter__(self):
        self.server = await websockets.serve(self.handler, "127.0.0.1", 0, max_size=None)
        self.url = f"ws://127.0.0.1:{self.server.sockets[0].getsockname()[1]}/v1/realtime"
        return self

    async def __aexit__(self, *exc):
        self.server.close()
        await self.server.wait_closed()

    async def handler(self, ws):
        send = lambda event: ws.send(json.dumps(event))  # noqa: E731
        await send({"type": "session.created", "session": {"poly_session_id": "conversation_test"}})
        heard, n = 0.0, 0
        async for raw in ws:
            event = json.loads(raw)
            self.received.append(event)
            if event["type"] == "session.update":
                await send({"type": "session.updated", "session": event["session"]})
            if event["type"] != "input_audio_buffer.append":
                continue
            heard += FRAME_SECONDS
            if not self.turns or heard < self.turns[0][0]:
                continue
            _, said, action = self.turns.pop(0)
            n += 1
            item, rid = f"item_{n}", f"resp_{n}"
            await send({"type": "input_audio_buffer.speech_started", "item_id": item})
            await send({"type": "input_audio_buffer.speech_stopped", "item_id": item})
            await send({"type": "response.created", "response": {"id": rid}})
            kind, *rest = action
            spoken = rest[0] if kind == "say" else rest[2]
            if kind == "tool":
                name, args, _ = rest
                await send({"type": "response.function_call_arguments.done", "response_id": rid,
                            "call_id": f"call_{n}", "name": name, "arguments": json.dumps(args)})
            if spoken:
                await send({"type": "response.output_audio.delta", "response_id": rid, "delta": audio(0.3)})
                await send({"type": "response.output_audio_transcript.done", "response_id": rid,
                            "transcript": spoken})
            await send({"type": "conversation.item.input_audio_transcription.completed", "item_id": item,
                        "transcript": said})
            await send({"type": "response.done", "response": {"id": rid, "status": "completed"}})
