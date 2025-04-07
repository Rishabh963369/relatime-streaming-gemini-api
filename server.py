import asyncio
import json
import os
import uuid
from google import genai
import base64
import websockets
import io
from pydub import AudioSegment
import google.generativeai as generative
import wave
import datetime

# Load and configure API key globally for google.generativeai
gemini_api_key = "AIzaSyC0jAjJsgxGUBOvEw8h_L5HlzRVkaS9T-0"
generative.configure(api_key=gemini_api_key)  # Configure API key for generative module

MODEL = "gemini-2.0-flash-exp"  # Updated to a known model (verify availability)

# Initialize the client with the API key
client = genai.Client(
    api_key=gemini_api_key,
    http_options={
        'api_version': 'v1alpha',
    }
)

async def gemini_session_handler(websocket: websockets.WebSocketServerProtocol):
    try:
        session_id = str(uuid.uuid4())
        current_conversation = []
        
        config_message = await websocket.recv()
        config_data = json.loads(config_message)
        config = config_data.get("setup", {})

        config["system_instruction"] = """You are a helpful math tutor. Start with foundational explanations, break down complex concepts into simpler parts, and use clear examples and step-by-step solutions. Always be patient, encouraging, and focus on helping the student develop strong mathematical intuition and problem-solving skills."""
        
        # Initialize audio buffers and control flags
        has_user_audio = False
        user_audio_buffer = b''
        has_assistant_audio = False
        assistant_audio_buffer = b''
        should_accumulate_user_audio = True

        async with client.aio.live.connect(model=MODEL, config=config) as session:
            print(f"Connected to Gemini API for session {session_id}")

            async def send_to_gemini():
                nonlocal has_user_audio, user_audio_buffer, should_accumulate_user_audio
                try:
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            
                            if "realtime_input" in data:
                                for chunk in data["realtime_input"]["media_chunks"]:
                                    if chunk["mime_type"] == "audio/pcm":
                                        if should_accumulate_user_audio:
                                            try:
                                                audio_chunk = base64.b64decode(chunk["data"])
                                                has_user_audio = True
                                                user_audio_buffer += audio_chunk
                                            except Exception as e:
                                                print(f"Error processing audio chunk: {e}")
                                        await session.send(input={
                                            "mime_type": "audio/pcm",
                                            "data": chunk["data"]
                                        })
                                    elif chunk["mime_type"].startswith("image/"):
                                        current_conversation.append({
                                            "role": "user", 
                                            "content": "[Image shared by user]"
                                        })
                                        await session.send(input={
                                            "mime_type": chunk["mime_type"],
                                            "data": chunk["data"]
                                        })
                            
                            elif "text" in data:
                                text_content = data["text"]
                                current_conversation.append({
                                    "role": "user", 
                                    "content": text_content
                                })
                                await session.send(input={
                                    "mime_type": "text/plain",
                                    "data": text_content
                                })
                                
                        except Exception as e:
                            print(f"Error sending to Gemini: {e}")
                    print("Client connection closed (send)")
                except Exception as e:
                    print(f"Error sending to Gemini: {e}")
                finally:
                    print("send_to_gemini closed")

            async def receive_from_gemini():
                nonlocal has_assistant_audio, assistant_audio_buffer, has_user_audio, user_audio_buffer, should_accumulate_user_audio
                try:
                    while True:
                        try:
                            async for response in session.receive():
                                if response.server_content is None:
                                    continue

                                model_turn = response.server_content.model_turn
                                if model_turn:
                                    for part in model_turn.parts:
                                        if hasattr(part, 'text') and part.text is not None:
                                            await websocket.send(json.dumps({"text": part.text}))
                                        elif hasattr(part, 'inline_data') and part.inline_data is not None:
                                            try:
                                                should_accumulate_user_audio = False
                                                audio_data = part.inline_data.data
                                                base64_audio = base64.b64encode(audio_data).decode('utf-8')
                                                await websocket.send(json.dumps({
                                                    "audio": base64_audio,
                                                }))
                                                has_assistant_audio = True
                                                assistant_audio_buffer += audio_data
                                            except Exception as e:
                                                print(f"Error processing assistant audio: {e}")

                                if response.server_content and response.server_content.turn_complete:
                                    print('\n<Turn complete>')
                                    user_text = None
                                    assistant_text = None
                                    
                                    if has_user_audio and user_audio_buffer:
                                        try:
                                            user_wav_base64 = convert_pcm_to_wav(user_audio_buffer, is_user_input=True)
                                            if user_wav_base64:
                                                user_text = transcribe_audio(user_wav_base64)
                                                print(f"Transcribed user audio: {user_text}")
                                            else:
                                                print("User audio conversion failed")
                                        except Exception as e:
                                            print(f"Error processing user audio: {e}")
                                    
                                    if has_assistant_audio and assistant_audio_buffer:
                                        try:
                                            assistant_wav_base64 = convert_pcm_to_wav(assistant_audio_buffer, is_user_input=False)
                                            if assistant_wav_base64:
                                                assistant_text = transcribe_audio(assistant_wav_base64)
                                                if assistant_text:    
                                                    await websocket.send(json.dumps({
                                                        "text": assistant_text
                                                    }))
                                            else:
                                                print("Assistant audio conversion failed")
                                        except Exception as e:
                                            print(f"Error processing assistant audio: {e}")
                                    
                                    has_user_audio = False
                                    user_audio_buffer = b''
                                    has_assistant_audio = False
                                    assistant_audio_buffer = b''
                                    should_accumulate_user_audio = True
                                    print("Re-enabling user audio accumulation for next turn")
                        except websockets.exceptions.ConnectionClosedOK:
                            print("Client connection closed normally (receive)")
                            break
                        except Exception as e:
                            print(f"Error receiving from Gemini: {e}")
                            break

                except Exception as e:
                    print(f"Error receiving from Gemini: {e}")
                finally:
                    print("Gemini connection closed (receive)")

            send_task = asyncio.create_task(send_to_gemini())
            receive_task = asyncio.create_task(receive_from_gemini())
            await asyncio.gather(send_task, receive_task)

    except Exception as e:
        print(f"Error in Gemini session: {e}")
    finally:
        print("Gemini session closed.")

def transcribe_audio(audio_data):
    """Transcribes audio using Gemini 1.5 Flash."""
    try:
        if not audio_data:
            return "No audio data received."
        
        if isinstance(audio_data, str):
            wav_audio_base64 = audio_data
        else:
            return "Invalid audio data format."
            
        transcription_client = generative.GenerativeModel(model_name="gemini-1.5-flash")  # Updated model
        
        prompt = """Generate a transcript of the speech. 
        Please do not include any other text in the response. 
        If you cannot hear the speech, please only say '<Not recognizable>'."""
        
        response = transcription_client.generate_content(
            [
                prompt,
                {
                    "mime_type": "audio/wav", 
                    "data": base64.b64decode(wav_audio_base64),
                }
            ]
        )
        return response.text

    except Exception as e:
        print(f"Transcription error: {e}")
        return "Transcription failed."

def convert_pcm_to_wav(pcm_data, is_user_input=False):
    """Converts PCM audio to base64 encoded WAV."""
    try:
        if not isinstance(pcm_data, bytes):
            print(f"PCM data is not bytes, it's {type(pcm_data)}")
            return None

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000 if is_user_input else 24000)
            wav_file.writeframes(pcm_data)
        
        wav_buffer.seek(0)
        wav_base64 = base64.b64encode(wav_buffer.getvalue()).decode('utf-8')
        return wav_base64
        
    except Exception as e:
        print(f"Error converting PCM to WAV: {e}")
        return None
    
async def main() -> None:
    async with websockets.serve(
        gemini_session_handler,
        host="0.0.0.0",
        port=9084,
        compression=None
    ):
        print("Running websocket server on 0.0.0.0:9084...")
        print("Math tutoring assistant ready to help")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
