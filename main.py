from openai import OpenAI
import sounddevice as sd
import base64
import numpy as np
from pydub import AudioSegment
from io import BytesIO
import os

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

class ChatSession:
    def __init__(self):
        self.messages = []
        self.temperature = 0.7

    def get_text_response(self):
        data_text = {
            "model": "gpt-4o-mini",
            "messages": self.messages,
            "temperature": self.temperature
        }
        return client.chat.completions.create(**data_text)
    
    def get_audio_response(self):
        data_audio = {
            "model": "gpt-4o-audio-preview",
            "modalities": ["text", "audio"],
            "audio": {"voice": "alloy", "format": "wav"},
            "messages": self.messages
        }
        return client.chat.completions.create(**data_audio)
        

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
    
    def add_audio_reference(self, role, audio_id):
        self.messages.append({"role": role, "audio": {
            "id": audio_id
        }})


    def run_text_chat(self):
        while True:
            input_text = input("You(enter 'exit' to quit    ):  ")
            if input_text == "exit":
                break
            self.add_message("user", input_text)
            completion = self.get_text_response()
            self.add_message("assistant", completion.choices[-1].message.content)
            print(completion.choices[-1].message.content)

    def run_audio_chat(self):
        while True:
            input_text = input("You(enter 'exit' to quit    ):  ")
            if input_text == "exit":
                break
            self.add_message("user", input_text)
            completion = self.get_audio_response()
            wav_bytes = base64.b64decode(completion.choices[-1].message.audio.data)
            audio_id = completion.choices[-1].message.audio.id
            audio_segment = AudioSegment.from_wav(BytesIO(wav_bytes))
            # to numpy array
            np_array = np.array(audio_segment.get_array_of_samples())
            sd.play(np_array, samplerate=audio_segment.frame_rate)
            print(completion.choices[-1].message.audio.transcript)
            self.add_audio_reference("assistant", audio_id)            
            sd.wait()
            

if __name__ == "__main__":
    chat_session = ChatSession()
    chat_session.run_audio_chat()

