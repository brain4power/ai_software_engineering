import os
import io
import base64
import pydub
import resampy

import librosa
import librosa.display

import streamlit as st
import numpy as np

from matplotlib import pyplot as plt
from scipy.io import wavfile

from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests


SAMPLE_RATE = 16000

plt.rcParams["figure.figsize"] = (12, 10)

endpoint_enhancement = os.getenv('API_ENHANCEMENT_URI')
endpoint_recognition = os.getenv('API_RECOGNITION_URI')

def process(sound_data, server_url: str):
    buf = io.BytesIO()
    np.save(buf, sound_data)
    data = buf.getvalue()
    m = MultipartEncoder(fields={"file": ("filename", data, "image/png")})
    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000)
    return r

def create_audio_player(audio_data, sample_rate):
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, rate=sample_rate, data=audio_data)
    return virtualfile

@st.cache
def handle_uploaded_audio_file(uploaded_file):
    a = pydub.AudioSegment.from_file(
        file=uploaded_file, format=uploaded_file.name.split(".")[-1]
    )

    channel_sounds = a.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]
    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    return fp_arr[:, 0], a.frame_rate

def add_h_space():
    st.markdown("<br></br>", unsafe_allow_html=True)

def plot_wave(source, sample_rate):
    fig, ax = plt.subplots()
    img = librosa.display.waveshow(source, sr=sample_rate, x_axis="time", ax=ax)
    return plt.gcf()

def do_audio_processing(source, sample_rate, option):
    if option == 'Speech Recognition':
        st.markdown(
        f"<h4 style='text-align: center; color: black;'>Audio</h5>",
        unsafe_allow_html=True,)

        st.audio(create_audio_player(source, sample_rate))
        result = process(source, endpoint_recognition)
        st.markdown("---")
        if result.status_code == 200:
            st.write(result.json()['text'])
        else:
            st.error(f'error {result.status_code}')

    elif option == 'Speech Enhancement':
        cols = [1, 1]
        col1, col2 = st.columns(cols)

        with col1:
            st.markdown(
                f"<h4 style='text-align: center; color: black;'>Original</h5>",
                unsafe_allow_html=True,
            )
            st.audio(create_audio_player(source, sample_rate))
        with col2:
            st.markdown(
                f"<h4 style='text-align: center; color: black;'>Wave plot </h5>",
                unsafe_allow_html=True,
            )
            st.pyplot(plot_wave(source, sample_rate))
            add_h_space()

        cols = [1, 1]
        col1, col2 = st.columns(cols)
        result = process(source, endpoint_enhancement)

        if result.status_code != 200:
          st.error(f'error {result.status_code}')
          return
        result = result.json()['playload']
        result = result.encode(encoding='UTF-8')
        buff = base64.decodebytes(result)
        sound = np.frombuffer(buff, dtype=np.float32)
        with col1:
            st.markdown(
                f"<h4 style='text-align: center; color: black;'>Original</h5>",
                unsafe_allow_html=True,
            )
            st.audio(create_audio_player(sound, sample_rate))
        with col2:
            st.markdown(
                f"<h4 style='text-align: center; color: black;'>Wave plot </h5>",
                unsafe_allow_html=True,
            )
            st.pyplot(plot_wave(sound, sample_rate))
            add_h_space()

def action(file_uploader, option):
    if file_uploader is not None:
      source, sample_rate = handle_uploaded_audio_file(file_uploader)
      source = resampy.resample(source, sample_rate, SAMPLE_RATE, axis=0, filter='kaiser_best')
      do_audio_processing(source, SAMPLE_RATE, option)

def main():
    placeholder = st.empty()
    placeholder2 = st.empty()
    placeholder.markdown(
        "# Processing of sound and speech by SpeechBrain framework\n"
        "### Select the processing procedure in the sidebar.\n"
        "Once you have chosen processing procedure, select or upload an audio file\n. "
        'Then click "Apply" to start! \n\n'
    )
    placeholder2.markdown(
        "After clicking start,the result of the selected procedure are visualized."
    )

    option = st.sidebar.selectbox('Audio Processing Task', options=('Speech Recognition', 'Speech Enhancement'))
    st.sidebar.markdown("---")
    st.sidebar.markdown("(Optional) Upload an audio file here:")
    file_uploader = st.sidebar.file_uploader(
        label="", type=[".wav", ".wave", ".flac", ".mp3", ".ogg"]
    )
    st.sidebar.markdown("---")
    if st.sidebar.button("Apply"):
        placeholder.empty()
        placeholder2.empty()
        action(file_uploader=file_uploader,
               option=option)

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Speech brain audio file processing")
    main()