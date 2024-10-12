from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor
import torch
import torchaudio


waveform, sample_rate = torchaudio.load("../test.wav")

# Load the Encodec model and processor
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

# Resample the audio to match the processor's sampling rate (if necessary)
if sample_rate != processor.sampling_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=processor.sampling_rate)
    waveform = resampler(waveform)

class EncodecEncodeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz")

    def forward(self, input):
        return self.model.encode(input)

class EncodecDecodeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz")

    def forward(self, audio_codes, audio_scales):
        return self.model.decode(audio_codes, audio_scales)

encodec_encode_model = EncodecEncodeModel()
# encodec_decode_model = EncodecDecodeModel(model)

# encoder_outputs = encodec_encode_model(waveform[None])

# audio_values = encodec_decode_model(encoder_outputs["audio_codes"], encoder_outputs["audio_scales"])

# print(f"Audio Values: {audio_values}")

# export the models to unnx format
onnx_model_encode_program = torch.onnx.export(encodec_encode_model, waveform, "../models/encodec.onnx")
# onnx_model_decode_program = torch.onnx.export(encodec_decode_model, (encoder_outputs["audio_codes"], encoder_outputs["audio_scales"]), "../models/decodec.onnx")
