import { useRef, useState } from 'react'
import { FFmpeg } from '@ffmpeg/ffmpeg';
import * as ort from 'onnxruntime-web';
import './App.css'
import { toBlobURL } from '@ffmpeg/util';
import decode, {decoders} from 'audio-decode';


ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";


async function EncodeAudio(ffmpeg: FFmpeg, arrayBuffer: ArrayBuffer) {
  // use ffmpeg to convert the audio to 24khz mono
  const byteArray = new Uint8Array(arrayBuffer);
  await ffmpeg.writeFile('input.wav', byteArray);
  ffmpeg.exec([
    '-i', 'input.wav',
    '-ac', '1',
    '-ar', '24000',

    'output.wav'
  ]);
  const data = await ffmpeg.readFile('output.wav');
  const blob = new Blob([data]);  
  const buffer = await blob.arrayBuffer();
  const bufferUint8 = new Uint8Array(buffer);
  const wavFile = await decoders.wav(bufferUint8)
  const wavBuffer = wavFile.getChannelData(0);
  let tensorData = new ort.Tensor('float32', wavBuffer, [wavBuffer.length]);
  const bytes = await fetch('http://localhost:5173/encodec.onnx').then(response => response.arrayBuffer());
  const session = await ort.InferenceSession.create(bytes);
  tensorData = tensorData.reshape([1,1,tensorData.dims[0]]);
  // console.log("[INPUT]  ", tensorData);
  const feeds = { 'audio': tensorData };
  const output = await session.run(feeds);

  const array = output['encoded_frames'].data as BigInt64Array;
  const dims = output['encoded_frames'].dims;
  const size = output['encoded_frames'].size;



  const jsonPayload = {
    'data': Array.from(array).map((x) => Number(x)),
    'dims': dims,
    'size': size,
    "audio_length": wavBuffer.length
  }

  const json = JSON.stringify(jsonPayload);

  await fetch('http://localhost:8000/decode', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: json,
  });
  

}

function App() {
  const ffmpegRef = useRef(new FFmpeg());
  const messageRef = useRef<HTMLParagraphElement>(null);
  const [loaded, setLoaded] = useState(false);
  const [loadingFfmpeg, setLoadingFfmpeg] = useState(false);

  const [message, setMessage] = useState('');

  const load = async () => {
    setLoadingFfmpeg(true);
    const baseURL = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/esm'
    const ffmpeg = ffmpegRef.current;
    ffmpeg.on('log', ({ message }) => {
      messageRef.current!.innerHTML = message;
      // console.log(message);
    });
    // toBlobURL is used to bypass CORS issue, urls with the same
    // domain can be used directly.
    await ffmpeg.load({
      coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
      wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
    });
    console.log('FFmpeg loaded successfully.');
    setLoaded(true);
    setLoadingFfmpeg(false);
  }

  const handleFileChange: React.ChangeEventHandler<HTMLInputElement> = (event) => {
    const files = event.target.files;
    if (!files || !files[0]) {
      setMessage('Please upload a file.');
      return;
    }
    const file = files[0];
    if (file && file.type === 'audio/wav') {
      const reader = new FileReader();

      reader.onload = (e) => {
        setMessage('Encoding audio file...');

        EncodeAudio(ffmpegRef.current, e.target?.result as ArrayBuffer).then(() => {
          setMessage('Audio file encoded successfully.');
        })
      };

      reader.readAsArrayBuffer(file)
    } else {
      setMessage('Please upload a valid .wav file.');
    }
  };

  return (loaded ? (
    <div>
      <h1>Upload WAV File</h1>
      <input type="file" accept=".wav" onChange={handleFileChange} />
      <p>{message}</p>

      <h2>Logs</h2>

      <p ref={messageRef}></p>

    </div>
  )
    : (
      <button onClick={load}>{loadingFfmpeg ? "Loading..." : 'Load ffmpeg-core (~31 MB)'}</button>
    )
  );
}

export default App
