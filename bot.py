# """
# Audio-Only Gemini Live Client
# """

# import os
# import asyncio
# import base64
# import io
# import traceback
# import argparse

# import pyaudio
# from google import genai
# from google.genai import types
# from dotenv import load_dotenv

# load_dotenv()

# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# SEND_SAMPLE_RATE = 16000
# RECEIVE_SAMPLE_RATE = 24000
# CHUNK_SIZE = 1024

# MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"

# client = genai.Client(
#     http_options={"api_version": "v1beta"},
#     api_key=os.environ.get("GEMINI_API_KEY"),
# )

# CONFIG = types.LiveConnectConfig(
#     response_modalities=["AUDIO"],
#     media_resolution="MEDIA_RESOLUTION_MEDIUM",
#     speech_config=types.SpeechConfig(
#         voice_config=types.VoiceConfig(
#             prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
#         )
#     ),
#     context_window_compression=types.ContextWindowCompressionConfig(
#         trigger_tokens=25600,
#         sliding_window=types.SlidingWindow(target_tokens=12800),
#     ),
#     system_instruction=types.Content(
#         parts=[
#             types.Part.from_text(
#                 text=(
#                     """"Role & Identity:
# You are Aditi, a warm and friendly call-center representative from Tata Motors. You speak in natural conversational Hinglish.
# Keep everything short, simple, conversational.and also keep your voice pace a bit aster not very slow"""""
        
#                 )
#             )
#         ],
#         role="user",
#     ),
# )

# pya = pyaudio.PyAudio()


# class AudioLoop:
#     def __init__(self):
#         self.audio_in_queue = None
#         self.out_queue = None
#         self.session = None

#     async def send_text(self):
#         while True:
#             text = await asyncio.to_thread(input, "message > ")
#             if text.lower() == "q":
#                 break
#             await self.session.send(input=text or ".", end_of_turn=True)

#     async def listen_audio(self):
#         mic_info = pya.get_default_input_device_info()

#         self.audio_stream = await asyncio.to_thread(
#             pya.open,
#             format=FORMAT,
#             channels=CHANNELS,
#             rate=SEND_SAMPLE_RATE,
#             input=True,
#             input_device_index=mic_info["index"],
#             frames_per_buffer=CHUNK_SIZE,
#         )

#         kwargs = {"exception_on_overflow": False} if __debug__ else {}

#         while True:
#             data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
#             await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

#     async def send_realtime(self):
#         while True:
#             msg = await self.out_queue.get()
#             await self.session.send(input=msg)

#     async def receive_audio(self):
#         """Reads model output and enqueues PCM audio."""
#         while True:
#             turn = self.session.receive()
#             async for response in turn:
#                 if data := response.data:
#                     self.audio_in_queue.put_nowait(data)
#                     continue
#                 if text := response.text:
#                     print(text, end="")

#             # Clear unread audio if turn completes
#             while not self.audio_in_queue.empty():
#                 self.audio_in_queue.get_nowait()

#     async def play_audio(self):
#         stream = await asyncio.to_thread(
#             pya.open,
#             format=FORMAT,
#             channels=CHANNELS,
#             rate=RECEIVE_SAMPLE_RATE,
#             output=True,
#         )
#         while True:
#             pcm = await self.audio_in_queue.get()
#             await asyncio.to_thread(stream.write, pcm)

#     async def run(self):
#         try:
#             async with (
#                 client.aio.live.connect(model=MODEL, config=CONFIG) as session,
#                 asyncio.TaskGroup() as tg,
#             ):
#                 self.session = session
#                 self.audio_in_queue = asyncio.Queue()
#                 self.out_queue = asyncio.Queue(maxsize=5)

#                 send_text_task = tg.create_task(self.send_text())
#                 tg.create_task(self.send_realtime())
#                 tg.create_task(self.listen_audio())
#                 tg.create_task(self.receive_audio())
#                 tg.create_task(self.play_audio())

#                 await send_text_task
#                 raise asyncio.CancelledError("User requested exit")

#         except asyncio.CancelledError:
#             pass
#         except ExceptionGroup as EG:
#             self.audio_stream.close()
#             traceback.print_exception(EG)


# if __name__ == "__main__":
#     main = AudioLoop()
#     asyncio.run(main.run())


"""
Backend Server for Gemini Live (Aditi - Tata Motors)
Run this with: python bot.py
"""
import os
import asyncio
import json
import base64
import traceback
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=API_KEY,
)

# Your Custom Persona
SYS_INSTRUCT = """Role & Identity:
You are Aditi, a warm and friendly call-center representative from Tata Motors. You speak in natural conversational Hinglish.
Keep everything short, simple, conversational and also keep your voice pace a bit faster, not very slow."""

CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    media_resolution="MEDIA_RESOLUTION_MEDIUM",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    system_instruction=types.Content(
        parts=[types.Part.from_text(text=SYS_INSTRUCT)],
        role="user",
    ),
)

app = FastAPI()

# Allow the frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    try:
        async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
            
            # Task 1: Receive Audio from Gemini -> Send to Browser
            async def receive_from_gemini():
                while True:
                    turn = session.receive()
                    async for response in turn:
                        if data := response.data:
                            # Send raw PCM audio to browser as base64 JSON
                            b64_audio = base64.b64encode(data).decode("utf-8")
                            await websocket.send_json({"type": "audio", "data": b64_audio})
                        if text := response.text:
                            print(f"Aditi: {text}")

            # Task 2: Receive Audio from Browser -> Send to Gemini
            async def receive_from_browser():
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    if message["type"] == "audio":
                        # Browser sends base64, we decode to bytes for Gemini
                        audio_bytes = base64.b64decode(message["data"])
                        await session.send(input={"data": audio_bytes, "mime_type": "audio/pcm"}, end_of_turn=False)
            
            # Run both tasks simultaneously
            await asyncio.gather(receive_from_gemini(), receive_from_browser())

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        traceback.print_exc()
    finally:
        await websocket.close()

if __name__ == "__main__":
    # This starts the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)