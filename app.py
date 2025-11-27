import streamlit as st
import tempfile
import os
import re
import io
from pydub import AudioSegment
from pydub.generators import Silent
from PyPDF2 import PdfReader
from openai import OpenAI

# Inicializar cliente OpenAI (usa variável OPENAI_API_KEY do Streamlit Secrets)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="Gerador de Áudio — 2 Vozes (OpenAI TTS)", layout="wide")

st.title("Gerador de Áudio — Entrevista (2 Vozes)")
st.caption("Gera áudio Pergunta/Resposta usando duas vozes diferentes com OpenAI TTS.")

# Sidebar
st.sidebar.header("Configurações")
voice1 = st.sidebar.selectbox("Voz 1 (Perguntas)", ["alloy", "verse", "shimmer"], index=0)
voice2 = st.sidebar.selectbox("Voz 2 (Respostas)", ["alloy", "verse", "shimmer"], index=1)
pause_ms = st.sidebar.slider("Pausa entre falas (ms)", 200, 1500, 500)
export_format = st.sidebar.selectbox("Formato de exportação", ["mp3", "wav"])

uploaded = st.file_uploader("Carregue o PDF com Perguntas/Respostas", type=["pdf"])

# PDF extraction
def extract_text(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n\n".join([page.extract_text() or "" for page in reader.pages])

# Parse QA blocks
def parse_qa(text):
    pattern = r"Pergunta[:\s]*([\s\S]*?)Resposta[:\s]*([\s\S]*?)(?=Pergunta[:\s]|$)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return [(q.strip(), a.strip()) for q, a in matches]

# OpenAI TTS wrapper
def tts(text, voice, fmt):
    output = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        format=fmt
    )
    return output.read()

# Generate audio
def synthesize(qa_list, v1, v2, pause, fmt):
    segments = []
    silence = Silent(duration=pause).to_audio_segment()

    for q, a in qa_list:
        q_audio = AudioSegment.from_file(io.BytesIO(tts(q, v1, fmt)), format=fmt)
        segments.append(q_audio)
        segments.append(silence)

        a_audio = AudioSegment.from_file(io.BytesIO(tts(a, v2, fmt)), format=fmt)
        segments.append(a_audio)
        segments.append(silence)

    combined = segments[0]
    for seg in segments[1:]:
        combined += seg
    return combined

# Main logic
if uploaded:
    st.info("A extrair texto...")
    raw = extract_text(uploaded.read())
    st.text_area("Texto extraído (preview)", raw[:2000])

    qa = parse_qa(raw)
    st.success(f"Foram encontrados {len(qa)} blocos Pergunta/Resposta.")

    if st.button("Gerar Áudio"):
        with st.spinner("A gerar áudio com OpenAI TTS..."):
            audio = synthesize(qa, voice1, voice2, pause_ms, export_format)

        buffer = io.BytesIO()
        audio.export(buffer, format=export_format)
        buffer.seek(0)

        st.audio(buffer.read(), format=f"audio/{export_format}")

        st.download_button(
            "Descarregar áudio final",
            data=buffer,
            file_name=f"entrevista_2vozes.{export_format}",
            mime=f"audio/{export_format}"
        )
else:
    st.info("Carregue um PDF para começar.")
