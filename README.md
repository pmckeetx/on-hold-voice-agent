# On-hold voice agent

An outbound voice agent on PolyAI's Dialog-RSN-1 API that phones a pharmacy for a patient. It presses the right key
in the phone menu, stays silent through the recorded messages and the hold music, answers the person who picks up,
waits again when they ask it to hold, and hangs up with the answer.

The walkthrough is the post: [Build an outbound voice agent that waits on hold](https://www.petermckee.dev/blog/build-an-outbound-voice-agent-that-waits-on-hold).

## What's here

`agent.py` is the agent, and the file the post walks through. `line.py` is the other end of the call: a scripted
Riverside Pharmacy line with a phone menu, a hold queue and Sam, the pharmacist, sent as 8 kHz G.711 mu-law in real
time, the way a phone line carries it. Nobody gets dialed. Every call is saved, both sides, to `call.wav`.
`make_samples.py` shows how the recordings in `samples/` were made.

## Prerequisites

Python 3.13 and [uv](https://docs.astral.sh/uv/). A Dialog-RSN-1 API key: sign in at
[studio.poly.ai](https://studio.poly.ai) and create one in the **API Keys** tab.

## Set up

```bash
git clone https://github.com/pmckeetx/on-hold-voice-agent.git
cd on-hold-voice-agent
cp .env.example .env    # then add your POLY_API_KEY
uv sync
```

## Run

```bash
uv run agent.py              # call the pharmacy: 30 seconds of hold music before Sam picks up
uv run agent.py --hold 120   # a longer hold
afplay call.wav              # listen to the call (macOS; any audio player works)
```

A variable set in your shell beats `.env`. If the server refuses the connection (HTTP 401), check
`echo $POLY_API_KEY`, and set `DIALOG_REALTIME_URL` if your key is for another PolyAI host.

## Test

```bash
uv run pytest                           # offline: a fake realtime server, no key needed
POLYAI_LIVE=1 uv run pytest -m live     # one call to the pharmacy against the real API
```

## License

Apache 2.0. See [LICENSE](LICENSE).
