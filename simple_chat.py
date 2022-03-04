import tensorflow as tf
tf.autograph.set_verbosity(0)

from scipy.io.wavfile import write
import numpy as np

from transformers import *
from transformers import AutoModelForCausalLM, AutoTokenizer

from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

import torch
import torchaudio

import sounddevice as sd
import soundfile as sf

from pydub import AudioSegment
from pydub.playback import play

import os
import yaml
import re

# Models for Simple Chat

def init_speech_recognition_model():
    # model_name = "facebook/wav2vec2-base-960h" # 360MB
    model_name = "facebook/wav2vec2-large-960h-lv60-self" # 1.18GB
    processor  = Wav2Vec2Processor.from_pretrained(model_name)
    model      = Wav2Vec2ForCTC.from_pretrained(model_name)
    return model, processor

def init_dialog_model():
    # model_name = "microsoft/DialoGPT-large"
    model_name = "microsoft/DialoGPT-medium"
    # model_name = "microsoft/DialoGPT-small"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def init_speech_synthesizer_model():
    # initialize fastspeech2 model, mb_melgan model and inference
    fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")
    mb_melgan  = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-ljspeech-en")
    processor  = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")
    return fastspeech2, mb_melgan, processor

# Get input audio from Mic

def get_input_audio(input_audio_path):
    fs = 44100  # Sample rate
    seconds = 3  # Duration of recording
    #print(">>")      
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write(input_audio_path, fs, myrecording)  # Save as WAV file 
    return

# Convert input audio to input text

def get_transcription(model, processor, input_audio_path):
    
    # load our wav file
    speech, sr = torchaudio.load(input_audio_path)
    speech = speech.squeeze()
    
    # resample from whatever the audio sampling rate to 16000
    resampler = torchaudio.transforms.Resample(sr, 16000)
    speech = resampler(speech)
    
    # tokenize our wav
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000)["input_values"]
    
    # perform inference
    logits = model(input_values)["logits"]
    
    # use argmax to get the predicted IDs
    predicted_ids = torch.argmax(logits, dim=-1)
    
    # decode the IDs to text
    transcription = processor.decode(predicted_ids[0])
    return transcription.lower()

# Convert response/output text to output audio

def get_output_audio(fastspeech2, mb_melgan, processor, output_text, audio_before_path, audio_after_path):

    input_ids = processor.text_to_sequence(output_text)

    # fastspeech inference
    mel_before, mel_after, duration_outputs, _, _ = fastspeech2.inference(
        input_ids     = tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        speaker_ids   = tf.convert_to_tensor([0],   dtype=tf.int32),
        speed_ratios  = tf.convert_to_tensor([1.0], dtype=tf.float32),
        f0_ratios     = tf.convert_to_tensor([1.0], dtype=tf.float32),
        energy_ratios = tf.convert_to_tensor([1.0], dtype=tf.float32),
    )

    # melgan inference
    audio_before = mb_melgan.inference(mel_before)[0, :, 0]
    audio_after = mb_melgan.inference(mel_after)[0, :, 0]

    # save to file
    sf.write(audio_before_path, audio_before, 22050, "PCM_16")
    sf.write(audio_after_path, audio_after, 22050, "PCM_16")
    return

# Play output audio

def play_output_audio(output_audio_path):
    response = AudioSegment.from_wav(output_audio_path)
    play(response)
    return

# Get chat response text for input text

def get_chatbot_response(model, tokenizer, chat_history_ids, input_text): 
    
    # encode the input and add end of string token
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
    
    # concatenate new user input with chat history (if there is)
    bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if chat_history_ids != None else input_ids
    
    # generate a bot response - with Top K sampling, nucleus sampling & tweaking temperature
    # new: top-K, top-p and temperature as 10, 0.9 and 0.9.
    # old: top_p=0.95,top_k=50, temperature=0.6,
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        do_sample=True,
        top_p=0.9,
        top_k=10,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response_text = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return chat_history_ids, response_text

# Recording from mic

from sys import byteorder
from array import array
from struct import pack

import pyaudio
import wave

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    silence = [0] * int(seconds * RATE)
    r = array('h', silence)
    r.extend(snd_data)
    r.extend(silence)
    return r

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 90:  # It was 30. Chnaged to 90 for check
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

# Simple chat starts here

tf.autograph.set_verbosity(0)

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

# Temp files to keep audio data

input_audio_file  = './input_audio.wav'
temp_audio_file   = './audio_before.wav'
output_audio_file = './audio_after.wav'

print("Initalizing models")

recognition_model, recognition_processor      = init_speech_recognition_model()
dialog_model, tokenizer                       = init_dialog_model()
fastspeech2, mb_melgan, synthesizer_processor = init_speech_synthesizer_model()

print("Models are ready now --------------")

# Set empty ChatBot Context
chat_history_ids = None

# chatting 100 times

for step in range(100):

    # Get user input
    
    #input_text = input(">> You:")
    
    print(">> You: ", end="")
    get_input_audio(input_audio_file)
    record_to_file(input_audio_file)
    
    input_text = get_transcription(recognition_model, recognition_processor, input_audio_file)
    
    if input_text == "good bye":
        print("Bye")
        break

    print(input_text)
    
    chat_history_ids, response_text = get_chatbot_response(dialog_model, tokenizer, chat_history_ids, input_text)
   
    if response_text == "":
        print("Sorry...")
        continue
        
    #print the output
    print(f"DialoGPT: {response_text}")
    
    response_text = re.sub(r'\W+', '', response_text)
    
    get_output_audio(fastspeech2, mb_melgan, synthesizer_processor, response_text, temp_audio_file, output_audio_file)
    play_output_audio(output_audio_file)
