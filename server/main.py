from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from encodec import EncodecModel
import torch
import torchaudio

app = FastAPI()

# CORS Middleware settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow specific origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/decode")
async def decode_json(request: Request):
    json_payload = await request.json()
    model = EncodecModel.encodec_model_24khz()


    data = json_payload.get("data")
    dims = json_payload.get("dims")
    size = json_payload.get("size")
    audio_length = json_payload.get("audio_length")

    # the data is an array of numbers, it should be converted to a tensor
    # with the specified dimensions 

    tensor = torch.tensor(data).view(size)
    tensor_reshaped = tensor.reshape(*dims)

    with torch.no_grad():
        wav = model.decode([(tensor_reshaped,None)])

    wav2 = wav[0, :, :audio_length]

    torchaudio.save("./output.wav", wav2, sample_rate=model.sample_rate, encoding='PCM_S', bits_per_sample=16)

    return {"message": "JSON received", "data": json_payload}