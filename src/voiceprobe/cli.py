import argparse
import logging
import os
import time

from dotenv import load_dotenv


def cmd_booking(args):
    import uvicorn

    from voiceprobe.booking.api import create_app

    uvicorn.run(create_app(args.db), host=args.host, port=args.port)


def cmd_say(args):
    from voiceprobe.audio import pcm
    from voiceprobe.audio.tts import Tts
    from voiceprobe.audio.wav import write_wav

    tts = Tts(args.voice)
    start = time.perf_counter()
    audio = tts.synthesize(args.text)
    took = time.perf_counter() - start
    write_wav(args.out, audio)
    print(f"wrote {args.out} ({pcm.duration(audio):.2f}s audio in {took:.2f}s)")


def cmd_transcribe(args):
    from voiceprobe.audio.stt import Stt
    from voiceprobe.audio.wav import read_wav

    audio = read_wav(args.path)
    stt = Stt(args.model)
    start = time.perf_counter()
    text = stt.transcribe(audio)
    print(text)
    print(f"({time.perf_counter() - start:.2f}s)")


def cmd_chat(args):
    import asyncio
    import uuid

    from voiceprobe.agent.dialog import GREETING, Dialog
    from voiceprobe.agent.llm import Ollama
    from voiceprobe.agent.tools import BookingTools

    async def run():
        llm = Ollama()
        tools = BookingTools(f"chat-{uuid.uuid4().hex[:6]}")
        dialog = Dialog(llm, tools)
        await llm.warmup()
        print(f"agent> {GREETING}")
        try:
            while True:
                try:
                    line = await asyncio.to_thread(input, "you> ")
                except EOFError:
                    break
                if not line.strip():
                    continue
                start = time.perf_counter()
                first = None
                async for sentence in dialog.reply(line):
                    first = first or time.perf_counter() - start
                    print(f"agent> {sentence}")
                if args.timing and first:
                    print(f"  (first sentence {first * 1000:.0f}ms)")
            for t in tools.history:
                print(f"tool {t.name} {t.arguments} -> {t.result}")
        finally:
            await llm.close()
            await tools.close()

    asyncio.run(run())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voiceprobe")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("booking", help="run the booking api")
    p.add_argument("--host", default=os.getenv("BOOKING_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.getenv("BOOKING_PORT", "8001")))
    p.add_argument("--db", default=os.getenv("BOOKING_DB", "data/bookings.db"))
    p.set_defaults(func=cmd_booking)

    p = sub.add_parser("say", help="synthesize text to a wav file")
    p.add_argument("text")
    p.add_argument("-o", "--out", default="out/say.wav")
    p.add_argument("--voice")
    p.set_defaults(func=cmd_say)

    p = sub.add_parser("transcribe", help="transcribe a wav file")
    p.add_argument("path")
    p.add_argument("--model")
    p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("chat", help="text chat with the agent (needs the booking api running)")
    p.add_argument("--timing", action="store_true")
    p.set_defaults(func=cmd_chat)

    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)
