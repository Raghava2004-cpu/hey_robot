import json
import os
import sys
import time
import uuid
import wave

WS_URL = "wss://api.meta.ai/v1/asr/realtime"
MODEL = "muse-voice-transcribe-1.0"
RATE = 24000          # Meta's native rate (16000 also works)


def record(seconds=6, path="runs/order.wav"):
    import sounddevice as sd
    print(f"🎤 Speak now ({seconds} s)...")
    audio = sd.rec(int(seconds * RATE), samplerate=RATE, channels=1, dtype="int16")
    sd.wait()
    print("🎤 Got it.")
    pcm = audio.tobytes()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:                 # keep the recording for the video
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return pcm, RATE


def load_wav(path):
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() not in (16000, 24000):
            raise ValueError("the WAV must be mono, 16-bit, 16000 or 24000 Hz")
        return w.readframes(w.getnframes()), w.getframerate()


def transcribe(pcm, rate=RATE, language="en", verbose=False):
    import websocket
    ws = websocket.create_connection(f"{WS_URL}?sessionId=stream-{uuid.uuid4()}",
                                     header=["User-Agent: Meta/1.0 (integration=HeyRobot)"],
                                     timeout=20)
    ws.send(json.dumps({
        "authorization": {"accessToken": f"Bearer {os.environ['MODEL_API_KEY']}"},
        "audioEncoding": f"PCM_{rate // 1000}KHZ",
        "model": MODEL,
        "mode": "ENDPOINTING",
        "partialMode": "CUMULATIVE",
        "emitAudioProgress": False,
        "languageBias": [language],
    }))
    ack = json.loads(ws.recv())
    if verbose:
        print("  <-", ack)
    if "sessionId" not in ack:
        ws.close()
        raise RuntimeError(f"handshake refused: {ack}")

    step = rate // 10 * 2                            # 0.1 s of 16-bit audio per frame
    for i in range(0, len(pcm), step):
        ws.send_binary(pcm[i:i + step])
        time.sleep(0.02)                             # about 5x faster than real time
    ws.send(json.dumps({"type": "endStream"}))       # "I'm done talking, send what you owe"

    complete, finals, partial = [], [], ""
    ws.settimeout(15)
    try:
        while True:
            raw = ws.recv()
            if not raw:
                break
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if verbose:
                print("  <-", msg)
            kind = msg.get("type")
            text = (msg.get("text") or msg.get("transcript") or "").strip()
            if kind == "error":
                raise RuntimeError(f"transcription error: {msg}")
            if kind == "speechComplete" and text:
                complete.append(text)
            elif kind == "transcript" and msg.get("final") and text:
                finals.append(text)
            elif kind == "transcript" and text:
                partial = text
    except (websocket.WebSocketTimeoutException, websocket.WebSocketConnectionClosedException):
        pass
    finally:
        ws.close()
    return " ".join(complete or finals) or partial


def listen(seconds=6, language="en", verbose=False):
    pcm, rate = record(seconds)
    return transcribe(pcm, rate, language, verbose)


if __name__ == "__main__":
    args = sys.argv[1:]
    lang = args[args.index("--lang") + 1] if "--lang" in args else "en"
    verbose = "--verbose" in args
    if "--file" in args:
        pcm, rate = load_wav(args[args.index("--file") + 1])
    else:
        secs = int(args[args.index("--seconds") + 1]) if "--seconds" in args else 6
        pcm, rate = record(secs)
    print(f'TRANSCRIPT: "{transcribe(pcm, rate, lang, verbose)}"')
