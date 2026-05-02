import torch
import os
import gradio as gr

# from langchain.llms import OpenAI
from langchain.llms import HuggingFaceHub

from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams


# ---------------- Credentials ----------------
my_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    # "apikey": "YOUR_API_KEY"  # Add this for local execution
}

params = {
    GenParams.MAX_NEW_TOKENS: 800,   # Max tokens generated
    GenParams.TEMPERATURE: 0.1,      # Lower = more deterministic
}


# ---------------- Model ----------------
LLAMA2_model = Model(
    model_id='meta-llama/llama-3-2-11b-vision-instruct',
    credentials=my_credentials,
    params=params,
    project_id="skills-network",
)

llm = WatsonxLLM(LLAMA2_model)


# ---------------- Prompt Template ----------------
temp = """
<>
List the key points with details from the context:
[INST]
The context : {context}
[/INST]
<>
"""

pt = PromptTemplate(
    input_variables=["context"],
    template=temp
)

prompt_to_LLAMA2 = LLMChain(llm=llm, prompt=pt)


# ---------------- Speech to Text ----------------
def transcript_audio(audio_file):
    # Initialize speech recognition pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )

    # Transcribe audio
    transcript_txt = pipe(audio_file, batch_size=8)["text"]

    # Send to LLM
    result = prompt_to_LLAMA2.run(transcript_txt)

    return result


# ---------------- Gradio UI ----------------
audio_input = gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox()

iface = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=output_text,
    title="Audio Transcription App",
    description="Upload the audio file"
)

iface.launch(server_name="0.0.0.0", server_port=7860)