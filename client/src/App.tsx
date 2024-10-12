import { useRef, useState } from 'react'
import { FFmpeg } from '@ffmpeg/ffmpeg';

import './App.css'
import { toBlobURL } from '@ffmpeg/util';


async function EncodeAudio(ffmpeg: FFmpeg,arrayBuffer: ArrayBuffer) {
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
 
  

}

function App() {
  const ffmpegRef = useRef(new FFmpeg());
  const messageRef = useRef<HTMLParagraphElement>(null);
  const [loaded, setLoaded] = useState(false);

  const [message, setMessage] = useState('');

  const load = async () => {
    const baseURL = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/esm'
    const ffmpeg = ffmpegRef.current;
    ffmpeg.on('log', ({ message }) => {
      messageRef.current!.innerHTML = message;
      console.log(message);
    });
    // toBlobURL is used to bypass CORS issue, urls with the same
    // domain can be used directly.
    await ffmpeg.load({
      coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
      wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
    });
    console.log('FFmpeg loaded successfully.');  
    setLoaded(true);
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

        EncodeAudio(ffmpegRef.current, e.target?.result).then(() => {
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
      <button onClick={load}>Load ffmpeg-core (~31 MB)</button>
    )
  );
}

export default App
