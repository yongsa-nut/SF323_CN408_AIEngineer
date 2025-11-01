# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script lets you to talk to a Gemini native audio model using the Live API.

## Setup

To install the dependencies for this script, run:

```
pip install portaudio  
pip install -U google-genai pyaudio
```

If Python < 3.11, also install `pip install taskgroup`.

## API key

Ensure the `GEMINI_API_KEY` environment variable is set to the api-key
you obtained from Google AI Studio.

## Run

To run the script:

```
python Get_started_LiveAPI_NativeAudio.py
```

Start talking to Gemini
"""

import asyncio
import sys
import traceback
from datetime import datetime

import pyaudio

from google import genai
from google.genai import types

# Add these lines
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

pya = pyaudio.PyAudio()


client = genai.Client(http_options={"api_version": "v1alpha"})  # GEMINI_API_KEY must be set as env variable

system_instruction = f"""
You are a helpful and friendly AI assistant.
Your default tone is helpful, engaging, and clear, with a touch of optimistic wit.
Anticipate user needs by clarifying ambiguous questions and always conclude your responses
with an engaging follow-up question to keep the conversation flowing. 

You have access to a google search tool to retrieve up-to-date information.
Today is {datetime.now()}.
"""

MODEL = "gemini-2.5-flash-native-audio-preview-09-2025"
tools = [{'google_search': {}}]
CONFIG = {
    "system_instruction": system_instruction,
    "response_modalities": ["AUDIO"],
    "tools": tools,
    "enable_affective_dialog":True,
    "proactivity": {'proactive_audio': True},
    "speech_config": {
        "voice_config": {"prebuilt_voice_config": {"voice_name": "Leda"}},
        "language_code": "th-TH"  #	en-US
    },
    "realtime_input_config": {
        "automatic_activity_detection": {
            "disabled": False,  # Keep VAD enabled
            "start_of_speech_sensitivity": types.StartSensitivity.START_SENSITIVITY_LOW,
            "end_of_speech_sensitivity": types.EndSensitivity.END_SENSITIVITY_LOW,
            "prefix_padding_ms": 20,
            "silence_duration_ms": 100,
        }
    }
}


class AudioLoop:
    def __init__(self):
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.audio_stream = None
        self.receive_audio_task = None
        self.play_audio_task = None
        self.is_playing = False  # Track if audio is being played

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        while True:
            # Only send audio when NOT playing Gemini's response
            if not self.is_playing:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE)
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
            else:
                # Skip reading to avoid echo, but keep loop running
                await asyncio.sleep(0.01)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send_realtime_input(audio=msg)

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")
                # The model might generate and execute Python code to use Search
                model_turn = response.server_content.model_turn
                if model_turn:
                    for part in model_turn.parts:
                      if part.executable_code is not None:
                        print(part.executable_code.code)

                      if part.code_execution_result is not None:
                        print(part.code_execution_result.output)
            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            
            self.is_playing = True  # Signal playback started
            await asyncio.to_thread(stream.write, bytestream)
            
            # Check if queue is empty (finished speaking)
            if self.audio_in_queue.empty():
                self.is_playing = False

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
        except asyncio.CancelledError:
            pass
        except asyncio.ExceptionGroup as eg:
            if self.audio_stream:
                self.audio_stream.close()
            traceback.print_exception(eg)


if __name__ == "__main__":
    print("Start talking to Gemini (use Ctrl-C to stop)...")
    loop = AudioLoop()
    asyncio.run(loop.run())