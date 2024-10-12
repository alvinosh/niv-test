import torch
import torchaudio
import typing as tp
from encodec import EncodecModel
from encodec.utils import convert_audio


EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]

class EncodecEncodeModel(EncodecModel):
    def forward(self, x: torch.Tensor) -> tp.List[EncodedFrame]:
        return self.encode(x)

class EncodecDecodeModel(EncodecModel):
    def forward(self, encoded_frames: tp.List[EncodedFrame]) -> torch.Tensor:
        return self.decode(encoded_frames)

encodec_model = EncodecEncodeModel.encodec_model_24khz()
wav, sr = torchaudio.load("../test.wav")
wav = convert_audio(wav, sr, encodec_model.sample_rate, encodec_model.channels)
wav = wav.unsqueeze(0)

frames = encodec_model(wav)

decodec_model = EncodecDecodeModel.encodec_model_24khz()
audio_values = decodec_model(frames)

print(f"Encodec Model: {frames}")
print(f"Decodec Model: {audio_values}")

# encodec_decode_model = EncodecDecodeModel(model)

# encoder_outputs = encodec_model(waveform[None])

# audio_values = encodec_decode_model(encoder_outputs["audio_codes"], encoder_outputs["audio_scales"])

# print(f"Audio Values: {audio_values}")

# export the models to unnx format
onnx_model_encode_program = torch.onnx.export(encodec_model, wav, "../models/encodec.onnx", opset_version=11)
onnx_model_decode_program = torch.onnx.export(decodec_model, frames, "../models/decodec.onnx")
