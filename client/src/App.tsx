import { useState } from 'react'
import { AutoProcessor, env } from '@xenova/transformers';

import './App.css'

function App() {
  const [message, setMessage] = useState('');

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
        env.allowLocalModels = false;
        env.useBrowserCache = false;

        const arrayBuffer = e.target?.result as ArrayBuffer;
        const byteArray = new Uint8Array(arrayBuffer);
        // convert the audio to waveform 24khz mono 
        AutoProcessor.from_pretrained("facebook/encodec_24khz").then((processor) => {
          console.log(123,processor);
          // const waveform = processor.encode(byteArray);
          // console.log(waveform);
        });
      };

      reader.readAsArrayBuffer(file)
    } else {
      setMessage('Please upload a valid .wav file.');
    }
  };

  return (
    <div>
      <h1>Upload WAV File</h1>
      <input type="file" accept=".wav" onChange={handleFileChange} />
      <p>{message}</p>
    </div>
  )
}

export default App
