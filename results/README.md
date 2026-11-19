# Results

The runs behind the numbers in the main README. Everything ran on one Apple M1 Pro with one call per scenario.

Each scenario run has one JSON per call, holding the full timeline (when each side spoke, what was said and heard, tool calls with results and timings, barge-ins) and the bookings at the end of the call. `report.json` is the score. To score a run again, or with a different barge-in limit:

```bash
uv run voiceprobe score results/reference-final
uv run voiceprobe score results/elevenlabs --barge-ms 300
```

| Folder | What it is |
|--------|------------|
| [`reference-first-run`](reference-first-run) | Reference agent, first full run |
| [`reference-final`](reference-final) | Reference agent with the booking guards and cancel gating |
| [`reference-read-back`](reference-read-back) | An experiment that forced a name read-back before booking, since removed |
| [`elevenlabs`](elevenlabs) | ElevenLabs agent, same scenarios |
| [`load-before`](load-before), [`load-after`](load-after) | Load ramps on the reference agent, before and after the cancel gating |

## reference-first-run

```
scenario                ended            time  task  tools    ttfa  barge      wer
book-simple             caller            69s  ok    ok       6.1s  -         0.0%
cancel                  caller            69s  ok    ok       5.0s  -         0.0%
change-mind             caller            81s  ok    FAIL     4.0s  -         0.0%
    - never called reschedule_appointment
interrupt-correction    no_reply          37s  FAIL  FAIL     3.6s  1/1       4.3%
    - nothing booked
    - never called book_appointment
interrupt-greeting      caller           131s  FAIL  ok       4.3s  1/1       3.3%
    - wrong name: caller
off-topic               caller           100s  FAIL  ok       3.8s  -         6.6%
    - wrong name: Ruth Akafer
slot-taken              caller            87s  ok    ok       3.8s  -         0.0%

calls              7 (caller 6, no_reply 1)
time to 1st audio  p50 4.1s  p95 6.2s  over 37 turns
barge-in           2/2 stopped within 500ms  (p50 151ms)
task success       4/7
tool use           5/7  (2 failed tool calls)
transcription wer  2.4% over 376 words
```

Two of these failures weren't the agent's fault. `interrupt-correction` hit a timeout bug in the harness: the reply timer also counted the time the agent spent talking. That's fixed. In `change-mind` the simulated caller moved to Wednesday before Tuesday was booked, so there was nothing to reschedule, and the scenario was reworded after this run.

## reference-final

```
scenario                ended            time  task  tools    ttfa  barge      wer
book-simple             caller            57s  ok    ok       3.3s  -         0.0%
cancel                  caller            54s  ok    ok       4.3s  -         0.0%
change-mind             caller            75s  ok    ok       4.4s  -         0.0%
interrupt-correction    caller            74s  ok    ok       3.6s  1/1       0.0%
interrupt-greeting      caller           122s  FAIL  ok       4.3s  1/1      12.7%
    - wrong name: Tom Raze
    . blocked: book_appointment
off-topic               caller            77s  FAIL  ok       3.8s  -         2.0%
    - wrong name: Ruth Akuffer
slot-taken              caller            50s  ok    ok       3.6s  -         0.0%

calls              7 (caller 7)
time to 1st audio  p50 4.3s  p95 4.7s  over 36 turns
barge-in           2/2 stopped within 500ms  (p50 126ms)
task success       5/7
tool use           7/7  (1 failed tool call, 1 blocked by the agent)
transcription wer  3.5% over 290 words
```

Lines starting with `.` are tool calls the agent's own guards stopped before they reached the booking API. Here it was a booking under the name "caller" before the caller had given one.

## reference-read-back

```
scenario                ended            time  task  tools    ttfa  barge      wer
book-simple             caller            74s  ok    ok       4.0s  -         0.0%
    . blocked: book_appointment
cancel                  caller            70s  ok    ok       4.6s  -         0.0%
change-mind             caller           108s  ok    ok       3.7s  -         6.0%
    . blocked: book_appointment
interrupt-correction    no_reply          99s  FAIL  ok       3.8s  1/1       4.1%
    - wrong name: Leena Park
interrupt-greeting      caller           126s  FAIL  ok       5.0s  1/1      11.4%
    - wrong name: Tom Ray
    . blocked: book_appointment, book_appointment, book_appointment, book_appointment
off-topic               caller           129s  FAIL  ok       3.9s  -         3.6%
    - wrong name: Ruth Akafur
    . blocked: book_appointment
slot-taken              caller           100s  ok    ok       3.9s  -        14.3%
    . blocked: book_appointment, book_appointment

calls              7 (caller 6, no_reply 1)
time to 1st audio  p50 4.1s  p95 5.6s  over 45 turns
barge-in           2/2 stopped within 500ms  (p50 162ms)
task success       4/7
tool use           7/7  (1 failed tool call, 9 blocked by the agent)
transcription wer  5.8% over 381 words
```

This run used a version of the agent that held every booking until it had read the name back to the caller. It made calls longer and didn't fix a single name. `interrupt-greeting` shows why: the caller corrects "Tom Raze" three times, and every correction is misheard too ("Ray's", "Ray"). The code was taken out again, so this agent isn't in the repo anymore. The `no_reply` is the 8B model returning an empty reply after the caller's last line.

## elevenlabs

```
scenario                ended            time  task  tools    ttfa  barge      wer
book-simple             caller            63s  ok    ok       1.1s  -         0.0%
cancel                  caller            53s  ok    ok       1.9s  -         0.0%
change-mind             caller            72s  ok    ok       1.3s  -         0.0%
interrupt-correction    caller            53s  ok    ok       2.3s  1/1       0.0%
interrupt-greeting      caller            40s  ok    ok       1.8s  0/1      11.1%
off-topic               caller            80s  ok    ok       1.2s  -         0.0%
slot-taken              caller            72s  ok    ok       1.1s  -         0.0%

calls              7 (caller 7)
time to 1st audio  p50 1.6s  p95 2.9s  over 36 turns
barge-in           1/2 stopped within 500ms  (p50 1013ms)
task success       7/7
tool use           7/7  (0 failed tool calls)
transcription wer  0.8% over 250 words
```

This folder is put together from two runs. Two runs were accidentally started at the same time, and one reset the booking database in the middle of the other's `slot-taken` call. The first six calls here come from the first run, which was clean up to that point. `slot-taken` comes from a later clean call. The two broken calls aren't included. Running two at once is blocked now.

The greeting barge-in failed because ElevenLabs played the whole greeting and never sent an interruption event. In the middle of a reply it stopped in 409 ms.

## load-before

```
calls  peak     p50     p95     max  turns  failed
    1     1    5.0s    5.2s    5.2s      3  0
    2     2    4.0s    5.2s    5.4s      6  0
    3     3    4.7s   11.0s   11.8s      9  0

breaks at 3 concurrent calls (p95 limit 10.4s), holds up to 2
```

## load-after

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

Each level folder holds the call records for that level, and `load.json` has the summary. The slow turns in `load-before` were all goodbyes. On every "thanks, bye" the 8B model tried to cancel the booking, the guard blocked it, and that cost an extra LLM round, which queued up behind the other calls. Only offering the cancel tool when the caller asks to cancel removed those rounds.
