from fastapi import FastAPI, Query
from fastapi.responses import Response
import edge_tts

app = FastAPI()

VOICES = {
    "pradeep": "bn-BD-PradeepNeural",
    "nabanita": "bn-BD-NabanitaNeural",
}

@app.get("/{full_path:path}")
async def handle_all(
    full_path: str,
    text: str = Query(None, max_length=300),
    voice: str = Query("pradeep"),
    rate: str = Query("+5%")
):
    if not text:
        return {"status": "IMBOT Edge TTS Server Running!"}

    v = VOICES.get(voice.lower(), "bn-BD-PradeepNeural")
    communicate = edge_tts.Communicate(text=text, voice=v, rate=rate)
    audio_data = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data.extend(chunk["data"])
    return Response(
        content=bytes(audio_data),
        media_type="audio/mpeg",
        headers={"Content-Length": str(len(audio_data))}
    )
