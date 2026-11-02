"""Make the pharmacy's recordings in samples/ with Cartesia text-to-speech, as 8 kHz phone audio.

You don't need this to run the agent: the clips are in the repo. It's here so you can see how they were made,
or record your own business. Needs CARTESIA_API_KEY.

    uv run make_samples.py
"""
import json
import os
import urllib.request
import wave
from pathlib import Path

from dotenv import load_dotenv

RECORDING = "71a7ad14-091c-4e8e-a314-022ece01c121"  # Cartesia "Charlotte", for the recorded messages
SAM = "1463a4e1-56a1-4b41-b257-728d56e93605"  # Cartesia "Hugo", the pharmacist who picks up

CLIPS = {
    "menu": (RECORDING, "Thank you for calling Riverside Pharmacy. For our opening hours, press 1. "
                        "For prescriptions, press 2."),
    "hours": (RECORDING, "We're open from 9 till 6, Monday to Saturday."),
    "connecting": (RECORDING, "Please hold while we connect you to the pharmacy team."),
    "still-holding": (RECORDING, "Thank you for holding. All of our team are busy helping other customers. "
                                 "Your call is important to us, so please stay on the line."),
    "sam-hello": (SAM, "Hi, thanks for holding, this is Sam at Riverside Pharmacy. How can I help?"),
    "sam-spell": (SAM, "Sure, can you spell her surname for me?"),
    "sam-check": (SAM, "Thanks. Let me have a look. Can you hold on a moment?"),
    "sam-ready": (SAM, "Thanks for waiting. Yes, that's ready. She can collect it any time after two today."),
    "sam-bye": (SAM, "No problem. Bye now."),
}


def tts(key: str, voice: str, text: str) -> bytes:
    body = {
        "model_id": "sonic-3.6",
        "transcript": text,
        "voice": {"mode": "id", "id": voice},
        "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 8000},
        "language": "en",
    }
    req = urllib.request.Request("https://api.cartesia.ai/tts/bytes", data=json.dumps(body).encode(), headers={
        "X-API-Key": key, "Cartesia-Version": "2025-04-16", "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def main():
    load_dotenv()
    key = os.environ["CARTESIA_API_KEY"]
    out = Path(__file__).parent / "samples"
    out.mkdir(exist_ok=True)
    for name, (voice, text) in CLIPS.items():
        with wave.open(str(out / f"{name}.wav"), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(8000)
            w.writeframes(tts(key, voice, text))
        print(f"samples/{name}.wav  {text}")


if __name__ == "__main__":
    main()
