import torch
import torchaudio
import typing as tp
from encodec import EncodecModel
from encodec.utils import convert_audio
from torch import nn


EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]

# class EncodecEncodeModel(EncodecModel):
#     def forward(self, x: torch.Tensor) -> tp.List[EncodedFrame]:
#         return self.encode(x)

# class EncodecDecodeModel(EncodecModel):
#     def forward(self, encoded_frames: tp.List[EncodedFrame]) -> torch.Tensor:
#         return self.decode(encoded_frames)


class EncodecEncodeModel(nn.Module):
    def __init__(self, model: EncodecModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tp.List[EncodedFrame]:
        return self.model.encode(x)

class EncodecDecodeModel(nn.Module):
    def __init__(self, model: EncodecModel):
        super().__init__()
        self.model = model

    def forward(self, encoded_frames: tp.List[EncodedFrame]) -> torch.Tensor:
        return self.model.decode(encoded_frames)


wav, sr = torchaudio.load("../StarWars60.wav")

model = EncodecModel.encodec_model_24khz()
encodec_model = EncodecEncodeModel(model)   

wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.unsqueeze(0)

frames = encodec_model(wav)

model = EncodecModel.encodec_model_24khz()
decodec_model = EncodecDecodeModel(model)

audio_values = decodec_model(frames)

print(f"Audio: {wav}")
print(f"Encodec Model: {frames}")
print(f"Decodec Model: {audio_values}")

# export the models to unnx format
onnx_model_encode_program = torch.onnx.export(
    encodec_model, 
    wav, 
    "../client/public/encodec.onnx", 
    opset_version=11,
    input_names=['audio'], 
    output_names=['encoded_frames'],
    dynamic_axes={
        'audio': {2: 'audio_length'},
        'encoded_frames': {0: 'batch', 2: 'length'}
    }

)

onnx_model_decode_program = torch.onnx.export(decodec_model, frames, "../client/public/decodec.onnx")
