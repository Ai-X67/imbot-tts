from fastapi import FastAPI, Query
from fastapi.responses import Response
import edge_tts

app = FastAPI()

VOICES = {
    "pradeep": "bn-BD-PradeepNeural",
    "nabanita": "bn-BD-NabanitaNeural",
    "bashkar": "bn-IN-BashkarNeural",
    "tanishaa": "bn-IN-TanishaaNeural",
}

@app.get("/{full_path:path}")
async def handle_all(
    full_path: str,
    text: str = Query(None, max_length=300),
    voice: str = Query("tanishaa"),
    rate: str = Query("+5%"),
    pitch: str = Query("+0Hz")
):
    if not text:
        return {"status": "IMBOT Edge TTS Server Running!"}

    v = VOICES.get(voice.lower(), "bn-IN-TanishaaNeural")
    communicate = edge_tts.Communicate(text=text, voice=v, rate=rate, pitch=pitch)
    audio_data = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data.extend(chunk["data"])
    return Response(
        content=bytes(audio_data),
        media_type="audio/mpeg",
        headers={"Content-Length": str(len(audio_data))}
    )
