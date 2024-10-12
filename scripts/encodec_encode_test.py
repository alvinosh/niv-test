import torchaudio
from transformers import EncodecModel, AutoProcessor
import torch
import json

# Load the .wav file using torchaudio (you can replace 'path_to_your_audio.wav' with your file's path)
waveform, sample_rate = torchaudio.load("../test.wav")

# Load the Encodec model and processor
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

# Resample the audio to match the processor's sampling rate (if necessary)
if sample_rate != processor.sampling_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=processor.sampling_rate)
    waveform = resampler(waveform)

# Convert the waveform to the required format and create input tensors

frames = model.encode(waveform[None])


# Forward pass through the model to get the audio values
audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values

# # Optionally, extract the discrete codebook representation for LM tasks
# audio_codes = model(inputs["input_values"], inputs["padding_mask"]).audio_codes

# print(f"Audio Values: {audio_values}")
# print(f"Audio Codes: {audio_codes}")
