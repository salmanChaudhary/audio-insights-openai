import openai
import os
import json
from dotenv import load_dotenv

load_dotenv()

def set_openai_credentials():
    api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key


def transcribe_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)

    return transcript

def extract_data_from_transcript(transcript):
    prompt = f"""
        Identify the following items from the audio transcript:
        - Unit Number
        - Page Number
        
        The audio transcript is delimited with triple backticks.
        Format your response as a JSON object with "Unit Number" and "Page Number" as the keys.
        If the information isn't present, use "unknown" as the value.
        Make your response as short as possible.
        
        Review text: '''{transcript}'''
        """
    
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0
    )
    chat_response = response.choices[0].message["content"]

    response_object = json.loads(chat_response)
    
    return response_object

# Example usage
def main():
    # Set up OpenAI API credentials
    set_openai_credentials()
    
    audio_path = "trash/sample.mp3"
    transcript = transcribe_audio(audio_path)
    print("Transcript:", transcript)
    
    extracted_data = extract_data_from_transcript(transcript)
    print("Extracted data:", extracted_data)

if __name__ == "__main__":
    main()
