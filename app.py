import streamlit as st
import boto3
import json
import base64
import requests
import io
from PIL import Image
import os
import sounddevice as sd
import soundfile as sf
import tempfile
from PyPDF2 import PdfReader
from dotenv import load_dotenv
load_dotenv()
# Configure page
st.set_page_config(
    page_title="AWS Services Integration",
    page_icon="🚀",
    layout="wide"
)

# Initialize AWS clients
@st.cache_resource
def get_aws_clients():
    try:
        bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
        lambda_client = boto3.client('lambda', region_name='us-east-1')
        lex_client = boto3.client('lexv2-runtime', region_name='us-east-1')
        return bedrock, lambda_client, lex_client
    except Exception as e:
        st.error(f"Error initializing AWS clients: {str(e)}")
        return None, None, None

# Audio recording utility
def record_audio(duration=5, samplerate=16000):
    st.info(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(temp_file.name, recording, samplerate)
    return temp_file.name

def generate_text(prompt, bedrock_client):
    try:
        response = bedrock_client.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.7
            })
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]
    except Exception as e:
        return f"Error generating text: {str(e)}"

def generate_image(prompt, bedrock_client):
    try:
        response = bedrock_client.invoke_model(
            modelId="stability.stable-diffusion-xl-v1",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "text_prompts": [{"text": prompt}],
                "cfg_scale": 10,
                "seed": 42,
                "steps": 50
            })
        )
        result = json.loads(response['body'].read())
        image_data = base64.b64decode(result['artifacts'][0]['base64'])
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        return f"Error generating image: {str(e)}"

def call_lambda_summarize(text, lambda_client):
    try:
        payload = {"body": json.dumps({"text": text})}
        response = lambda_client.invoke(
            FunctionName='summarize_lambda',
            InvocationType='RequestResponse',
            Payload=json.dumps(payload),
        )
        response_payload = json.load(response['Payload'])
        return json.loads(response_payload['body']).get("summary", "No summary returned")
    except Exception as e:
        return f"Error calling Lambda: {str(e)}"

def call_api_gateway_translate(text, direction):
    try:
        url = "https://4tud9deny0.execute-api.us-east-1.amazonaws.com/translationstage"
        payload = {"body": json.dumps({"text": text, "direction": direction})}
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            inner_body = json.loads(json.loads(response.text)["body"])
            return inner_body.get("translation", "Translation not found")
        else:
            return f"API Gateway error: {response.status_code}"
    except Exception as e:
        return f"Error calling API Gateway: {str(e)}"

def process_audio_with_lex(audio_path, lex_client):
    try:
        bot_id = 'ZTEA8D6PJD'
        bot_alias_id = 'TSTALIASID'
        locale_id = 'en_US'
        session_id = 'streamlit-session'
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        response = lex_client.recognize_utterance(
            botId=bot_id,
            botAliasId=bot_alias_id,
            localeId=locale_id,
            sessionId=session_id,
            requestContentType='audio/l16; rate=16000; channels=1',
            responseContentType='audio/mpeg',
            inputStream=audio_bytes
        )
        return response.get("inputTranscript", "No text recognized"), response.get("audioStream")
    except Exception as e:
        return f"Error processing audio with Lex: {str(e)}", None
    
def ask_question_about_pdf(pdf_file, question, bedrock_client):
    try:
        reader = PdfReader(pdf_file)
        text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        prompt = f"Here is a document: {text[:4000]}. Now answer the question: {question}"
        return generate_text(prompt, bedrock_client)
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def main():
    st.title("\U0001F680 AWS Services Integration App")

    bedrock_client, lambda_client, lex_client = get_aws_clients()
    if not all([bedrock_client, lambda_client, lex_client]):
        st.error("AWS client init failed.")
        return

    tab1, tab2, tab3 = st.tabs(["\U0001F4DD Text Input", "\U0001F3A4 Audio Input", "\U0001F3A4 PdfReader"])

    with tab1:
        st.header("Text Input Processing")
        user_input = st.text_area("Enter your text:", height=150)
        direction = st.selectbox("Translation direction:", ["auto-en", "en-hi", "hi-en", "en-es", "es-en"])

        if st.button("Process Text", type="primary"):
            if user_input:
                user_input_lower = user_input.lower()
                if 'generate image' in user_input_lower:
                    prompt = user_input.replace('generate image', '').strip()
                    result = generate_image(prompt or "A beautiful landscape", bedrock_client)
                    st.image(result) if isinstance(result, Image.Image) else st.error(result)
                elif 'summarize' in user_input_lower:
                    text = user_input.replace('summarize', '').strip(':').strip()
                    result = call_lambda_summarize(text or "Please provide text", lambda_client)
                    st.success(result)
                elif 'translate' in user_input_lower:
                    text = user_input.replace('translate', '').strip(':').strip()
                    result = call_api_gateway_translate(text or "Hello world", direction)
                    st.success(result)
                else:
                    result = generate_text(user_input, bedrock_client)
                    st.write(result)
            else:
                st.warning("Enter some text first.")

    with tab2:
        st.header("Audio Input Processing")
        duration = st.slider("Recording Duration (seconds)", 1, 10, 5)
        if st.button("Record and Process Audio"):
            with st.spinner("Recording and sending to Lex..."):
                audio_path = record_audio(duration)
                recognized_text, audio_response = process_audio_with_lex(audio_path, lex_client)
                if not recognized_text.startswith("Error"):
                    # st.success(f"Recognized: {recognized_text}")
                    if audio_response:
                        audio_data = audio_response.read()
                        st.audio(audio_data, format='audio/mp3')
                else:
                    st.error(recognized_text)

    with tab3:
        st.header("Ask a Question About a PDF")
        pdf_file = st.file_uploader("Upload a PDF", type=['pdf'])
        question = st.text_input("Ask a question about the PDF")
        if st.button("Ask About PDF") and pdf_file and question:
            result = ask_question_about_pdf(pdf_file, question, bedrock_client)
            st.write(result)


if __name__ == "__main__":
    main()
