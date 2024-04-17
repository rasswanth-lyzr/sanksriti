import os

from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.tasks.task_literals import InputType, OutputType

client = OpenAI()
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

AUDIO_FILE_NAME = "ted talk mini.mp3"
TRANSLATED_AUDIO_FILE_NAME = "translated_audio.mp3"

open_ai_model_text = OpenAIModel(
api_key=OPENAI_API_KEY,
parameters={
    "model": "gpt-4-turbo-preview",
    "temperature": 0.2,
    "max_tokens": 4096,
    },
)

def convert_language(translation, language):
    translator_agent = Agent(
        prompt_persona="You are an intelligent agent that can translate from English to {output_language} without missing any detail or altering the timestamp.".format(output_language = language),
        role="Subtitle translator",
    )

    translate_content_task = Task(
        name="Subtitle translator",
        agent=translator_agent,
        output_type=OutputType.TEXT,
        input_type=InputType.TEXT,
        model=open_ai_model_text,
        instructions="Translate the given text from English to {output_language}. Return only the translated text and timestamps.".format(output_language = language),
        log_output=True,
        enhance_prompt=False,
        default_input=translation,
    ).execute()

    return translate_content_task

def extract_text_from_srt(content):
    text = ""
    for line in content.splitlines():
        if line.strip() and not line.isdigit() and "-->" not in line.strip():
            text += line.strip() + "\n"

    return text

def text_to_speech(content):
    text_content = extract_text_from_srt(content)
    speech_file_path = TRANSLATED_AUDIO_FILE_NAME
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        speed=1,
        input=text_content
    )

    response.stream_to_file(speech_file_path)

st.write("# Audio File")

st.audio(AUDIO_FILE_NAME, format="audio/mpeg", loop=False)

input_language = st.selectbox(
    "Select Language",
    ("english", "hindi", "spanish", "tamil")
)

submit_button = st.button("Submit")

if submit_button:
    placeholder = st.empty()
    placeholder.text("Transcribing...")
    if input_language == "english":
        audio_file= open(AUDIO_FILE_NAME, "rb")
        translated_output = client.audio.translations.create(
            model="whisper-1",
            response_format="srt",
            file=audio_file
        )
    else:
        audio_file= open(AUDIO_FILE_NAME, "rb")
        translation = client.audio.translations.create(
            model="whisper-1",
            response_format="srt",
            file=audio_file
        )
        placeholder.empty()
        placeholder.text("Translating to " + input_language + "...")
        translated_output = convert_language(translation, input_language)

    placeholder.empty()
    st.write("# Translation")
    for line in translated_output.splitlines():
        st.write(line)

    st.download_button('Download srt', translated_output, file_name="generated_subtitles.srt", mime="text/vtt") # Download output

    
    placeholder.text("Generating Audio...")
    text_to_speech(translated_output)
    placeholder.empty()

    st.write("# Translated Audio")
    st.audio(TRANSLATED_AUDIO_FILE_NAME, format="audio/mpeg", loop=False)