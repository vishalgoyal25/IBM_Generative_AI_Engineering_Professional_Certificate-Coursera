# To call watsonx's LLM, we need to import the library of IBM Watson Machine Learning
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model

# placeholder for Watsonx_API and Project_id incase you need to use the code outside this environment
# API_KEY = "Your WatsonX API"
PROJECT_ID = "skills-network"

# Define the credentials
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
    # "apikey": API_KEY
}

# Specify model_id that will be used for inferencing
model_id = "mistralai/mistral-medium-2505"

# Define the model parameters
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 1024
}

# Define the LLM
model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=PROJECT_ID
)

import requests


def speech_to_text(audio_binary):
    # Set up Watson Speech-to-Text HTTP Api url
    base_url = 'https://sn-watson-stt.labs.skills.network'
    api_url = base_url + '/speech-to-text/api/v1/recognize'

    # Set up parameters for our HTTP request
    params = {
        'model': 'en-US_Multimedia',
    }

    # Send a HTTP POST request
    response = requests.post(api_url, params=params, data=audio_binary).json()

    # Parse the response
    text = 'null'
    while bool(response.get('results')):
        print('Speech-to-Text response:', response)
        text = response.get('results').pop().get('alternatives').pop().get('transcript')
        print('recognised text:', text)
        return text


def text_to_speech(text, voice=""):
    # Set up Watson Text-to-Speech HTTP Api url
    base_url = 'https://sn-watson-tts.labs.skills.network'
    api_url = base_url + '/text-to-speech/api/v1/synthesize?output=output_text.wav'

    # Add voice parameter if provided
    if voice != "" and voice != "default":
        api_url += "&voice=" + voice

    # Headers
    headers = {
        'Accept': 'audio/wav',
        'Content-Type': 'application/json',
    }

    # Body
    json_data = {
        'text': text,
    }

    # Send request
    response = requests.post(api_url, headers=headers, json=json_data)
    print('Text-to-Speech response:', response)

    return response.content


def watsonx_process_message(user_message):
    # Set the prompt for Watsonx API
    prompt = f"""
Translate the following English sentence into Spanish.
Reply ONLY with the translation, no explanations, no formatting, no extra text.

English: {user_message}
Spanish:
"""

    response_text = model.generate_text(prompt=prompt)
    print("watsonx response:", response_text)

    return response_text.strip()
