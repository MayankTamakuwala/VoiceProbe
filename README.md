# VoiceProbe

Load and evaluation harness for voice agents.

VoiceProbe phones a voice agent with simulated callers, talks to it out loud and scores every call. Each caller is a local LLM following a YAML scenario ("book Tuesday at 3pm, then change your mind"), voiced with Piper TTS and streamed to the agent as real-time audio over a WebSocket. It interrupts, corrects itself and hangs up like a person would.

It measures:

- **Time to first audio** (p50/p95): from the moment the caller stops talking to the first sound back from the agent, measured on the caller's side.
- **Barge-in**: when the caller talks over the agent, how long until the agent goes quiet (pass is under 500 ms).
- **Task success**: whether the bookings ended up right. Right slot, right name, no double booking, bookings that existed before the call left alone.
- **Tool use**: whether the expected tools were called and the forbidden ones weren't.
- **Transcription error rate**: WER between what the caller said and what the agent heard.
- **Load**: how many concurrent calls the agent takes before p95 latency breaks.

It comes with two agents to point it at: a reference agent that runs fully local, and an ElevenLabs agent set up with the same prompt and tools.

## How it works

```
 scenario.yaml ─► caller LLM ─► Piper TTS ─► 16 kHz PCM, 20 ms frames ─► agent under test
                    ▲                          (WebSocket, real time)        │
                    └──────────────── what the agent said ◄──────────────────┘
                                                                             │ tools
                                                          booking API (SQLite) ◄┘
```

The call runner keeps a full-duplex line open the whole call, sending silence when the caller isn't talking, the way a phone does. It models playback on the caller's side, so time to first audio means when the caller would actually hear the agent, not when bytes arrived. It fires scripted interruptions at set points in the agent's speech and writes a timeline of every turn, tool call and interruption.

**Reference agent**: FastAPI WebSocket server. Silero VAD (ONNX), faster-whisper `base.en`, Llama 3.1 8B via Ollama, Piper for speech. Its tools hit a small booking API backed by SQLite. Every write takes an idempotency key, so a retried request or a duplicate tool call can't book twice. Barge-in cancels the reply mid-sentence and trims the agent's memory down to what the caller actually heard.

**ElevenLabs agent**: `voiceprobe elevenlabs-setup` creates one with the same system prompt, greeting and four booking tools. The tools are client tools, so ElevenLabs sends each call to the harness and the harness runs it against the same local booking API. Bookings get scored the same way for both agents and nothing local has to be exposed to the internet. The reference agent's own safety checks (see below) live in the reference agent, so ElevenLabs runs without them.

## Setup

You need Python 3.11+, [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com).

```bash
uv sync
ollama pull llama3.1:8b
uv run python -m piper.download_voices en_US-lessac-medium --data-dir models
cp .env.example .env
```

Whisper downloads itself into `models/whisper` the first time it runs.

## Usage

Start the pieces, each in its own terminal:

```bash
ollama serve
uv run voiceprobe booking     # booking api on :8001
uv run voiceprobe agent       # reference agent on :8000
```

Then:

```bash
uv run voiceprobe run                          # every scenario, scored at the end
uv run voiceprobe run --only change-mind cancel --repeat 3
uv run voiceprobe score                        # score the latest run again
uv run voiceprobe load                         # ramp up concurrent calls
```

Each run goes to `out/runs/<time>-<label>/`, one JSON per call with the full timeline and the final bookings, plus `report.json`. Only one run or load test can go at a time, since they share the booking database and Ollama.

The caller reacts to the agent's text by default. `--hear audio` makes it transcribe the agent's audio instead, which is closer to a real caller but slower.

**ElevenLabs**: put `ELEVENLABS_API_KEY` in `.env` (the key needs write access to ElevenLabs Agents), then:

```bash
uv run voiceprobe elevenlabs-setup             # prints ELEVENLABS_AGENT_ID for .env
uv run voiceprobe run --agent elevenlabs
```

Each call is capped at 150 seconds so a stuck call can't eat through your minutes.

There are some smaller commands for poking at single pieces: `say` (TTS to a wav), `transcribe`, `chat` (text chat with the agent) and `dial` (speak a few lines at an agent).

### Docker

```bash
docker compose up --build                       # booking api + reference agent
docker compose --profile probe run --rm probe   # run the scenarios from a container
```

Ollama stays on the host by default, because Docker on a Mac can't use the GPU. To run it in a container as well:

```bash
COMPOSE_OLLAMA_HOST=http://ollama:11434 docker compose --profile ollama up --build
docker compose exec ollama ollama pull llama3.1:8b
```

Whisper and Piper run on the Docker VM's CPU, so expect higher latency than running natively (about 6 s to first audio against 4 s on the same Mac).

## Scenarios

```yaml
id: change-mind
description: Books one slot, then moves it. Should reschedule, not double book.
name: Marcus Bell
style: A bit scattered, changes plans easily.
goal:
  - Book a checkup for Tuesday at 3pm.
  - Wait until the receptionist says the Tuesday booking is done. Only then change your mind
    and ask to move it to Wednesday at 10am instead.
  - When the move is confirmed, say goodbye.
expect:
  booking: {name: Marcus Bell, day: wednesday, time: "10:00"}
  tools: [book_appointment, reschedule_appointment]
  forbidden_tools: [cancel_appointment]
```

Other fields: `interrupts` (what to say, which agent turn to cut into and how far into it), `existing_bookings` (seeded before the call), `expect.no_booking` and `max_turns`. The seven scenarios in `scenarios/` cover a simple booking, a cancellation, a change of mind, a slot that's already taken, an off-topic question, and interrupting both the greeting and a reply.

## Results

All numbers are from one Apple M1 Pro, one call per scenario. Treat them as rough: the same agent moves a few hundred ms and a scenario or so between runs. The runs behind them, with every call's full timeline, are in [`results/`](results).

### Reference agent vs ElevenLabs

|                         | Reference (Llama 3.1 8B, local) | ElevenLabs (Gemini 2.5 Flash) |
|-------------------------|------------------|-----------------|
| Time to first audio p50 / p95 | 4.3 s / 4.7 s | 1.6 s / 2.9 s |
| Task success            | 5/7              | 7/7             |
| Tool use                | 7/7              | 7/7             |
| Barge-in                | 2/2, ~130 ms     | 1/2             |
| Transcription error     | 3.5%             | 0.8%            |

ElevenLabs is about 2.5x faster and hears names better. Its time includes the round trip to its servers. The reference agent's two failures are both misheard names: it booked "Tom Raze" for Tom Reyes and "Ruth Akuffer" for Ruth Okafor.

ElevenLabs is slower to yield, though. Talked over in the middle of a reply it stopped in 409 ms against ~130 ms for the reference agent, and it didn't let the caller cut into the greeting at all. It played the whole thing and never sent an interruption event.

### What the harness caught in the reference agent

| Run | Task | Tools | TTFA p50 / p95 | Barge-in |
|-----|------|-------|----------------|----------|
| First run | 4/7 | 5/7 | 4.1 s / 6.2 s | 2/2 |
| + booking guards | 5/7 | 7/7 | 4.7 s / 8.9 s | 2/2 |
| + only offer cancel when asked | 5/7 | 7/7 | 4.3 s / 4.7 s | 2/2 |

One of the first run's failures was a timeout bug in the harness itself, fixed since.

- **Double booking on a change of mind.** Asked to move Tuesday to Wednesday, the 8B model booked Wednesday as well. The agent now turns a second booking under the same name into a reschedule.
- **Booking under "caller".** When nobody had given a name, the model filled in "caller". Bookings are now refused until the caller has actually said a name.
- **Cancelling on goodbye.** Llama 3.1 8B reaches for `cancel_appointment` on almost every "thanks, bye". A guard blocked it, but each blocked attempt cost a whole extra LLM round. Only offering the cancel tool once the caller says something like "cancel" got rid of those rounds, and that's where the p95 drop in the last row comes from.
- **Misheard names.** Forcing the agent to read the name back before booking didn't fix them. The caller caught the wrong spelling, but its correction went through the same speech recognizer and came out wrong again ("Reyes" became "Raze", "Ray's", "Ray"). Whisper `small.en` gets the same names wrong and takes three times as long, so this stays a known gap of the local stack.

### Load (reference agent)

```
calls  peak     p50     p95     max  turns  failed
    1     1    3.7s    4.8s    5.0s      3  0
    2     2    3.2s    4.2s    4.4s      6  0
    3     3    3.6s    5.5s    6.0s      9  0
    4     4    3.3s    5.5s    5.6s     12  0
    6     6    5.6s    9.7s   10.0s     18  0
    8     8    9.5s   11.9s   12.5s     24  0

breaks at 8 concurrent calls (p95 limit 9.7s), holds up to 6
```

The load test uses scripted callers so they don't compete with the agent for Ollama. Each one asks something that needs a tool, asks something that doesn't, then hangs up. A level "breaks" when its p95 goes over twice the single-call p95, or when any call fails. Ollama serves one request at a time here (`OLLAMA_NUM_PARALLEL=1`), so past a handful of calls they're queueing for the LLM. Before the cancel change the same agent broke at 3 calls (p95 11.0 s).

## Layout

```
src/voiceprobe/
  audio/     pcm helpers, Piper TTS, faster-whisper STT, Silero VAD
  booking/   SQLite store with idempotency keys, FastAPI booking api
  agent/     reference agent: ollama client, tools, dialog, websocket server
  probe/     scenarios, caller, call runner, adapters, scoring, load, elevenlabs setup
  cli.py
scenarios/   yaml scenarios
tests/
```

## Tests

```bash
uv run pytest
```

Tests that need speech models skip when the Piper voice isn't downloaded, and the LLM tests skip when Ollama isn't running. The agent and runner tests start a real agent server with a scripted LLM and talk to it over real audio, so they take a minute or two.
