from flask import Flask, request, jsonify, render_template, render_template_string
import io
from pydub import AudioSegment, silence
import speech_recognition as sr
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model
import librosa
import numpy as np
import glob

global audioLen
audiolen = 0

def process_file(filePath):
    audio = AudioSegment.from_file(filePath)
    min_silence_len = 300  # Silence shorter than this will be considered as pause in speech
    silence_thresh = -60   # Silence threshold
    chunks = silence.split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    # Initialize recognizer
    recognizer = sr.Recognizer()
    processed_audio = AudioSegment.empty()
    cnt = 0
    for chunk in chunks:

        with chunk.export('tmp/' + str(cnt)+'.wav', format="wav") as exported_chunk:
            with sr.AudioFile(exported_chunk.name) as source:
                audio_data = recognizer.record(source)
                try:
                    # Check if speech is recognized in the chunk
                    text = recognizer.recognize_google(audio_data)
                    print(text)
                    processed_audio += chunk  # Add chunk if speech is detected
                except sr.UnknownValueError:
                    print("Speech is not recognized")
                    pass  # Silence or noise detected, do not add chunk
        cnt += 1
    processed_audio.export("temp/to_analyse.mp3", format="mp3")

# emotion recognittion
def emo_recognise():
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec2 = Wav2Vec2Model.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
    if torch.cuda.is_available():
        wav2vec2 = wav2vec2.to('cuda')
        print("Model moved to CUDA.")

    #load the model

    class EmotionDetectionModel(torch.nn.Module):
        def __init__(self):
            super(EmotionDetectionModel, self).__init__()
            self.lstm = torch.nn.LSTM(1024, 256, batch_first=True)  # 768 is the feature size from Wav2Vec2
            #self.linear = torch.nn.Linear(768, 128)
            #self.td = torch.nn.Dropout(0.2)
            self.fc = torch.nn.Linear(256, 256)
            self.fc1 = torch.nn.Linear(256, 3) # 3 outputs: valence, arousal and dominance
        def forward(self, x):
            l, _ = self.lstm(x)
            #m = torch.mean(x, dim=1)
            #m = self.linear(m)
            k = l[:, -1, :]
            #x = torch.cat((m, k), dim=1)
            #x = self.fc(x[:, -1, :])  # Use the last LSTM output
            #x = self.td(k)
            x = self.fc(k)
            x = self.fc1(x)
            return x

    # Instantiate the model
    model = EmotionDetectionModel()

    model.load_state_dict(torch.load('model_trained_ravdess.pth'))
    # If GPU is available, move the model to GPU
    if torch.cuda.is_available():
        model.cuda()
        print("Model moved to CUDA.")

    model.eval()

    # labels
    label_conversion = {'0': 'Neutral',
                            '1': 'Calm',
                            '2': 'Happy',
                            '3': 'Sad',
                            '4': 'Angry',
                            '5': 'Anxiety',
                            '6': 'Disgust',
                            '7': 'Surprised'}

    emotion_to_vad_generated = {
            "0" : (0.883, 0.17, 0.304),
"1" : (0.744, 0.085, 0.191),
"2" : (0.987, 0.529, 0.596),
"3" : (0.736, 0.305, 0.401),
"4" : (0.235, 0.863, 0.918),
"5" : (0.431, 0.607, 0.415),
"6" : (0.397, 0.386, 0.493),
"7" : (0.709, 0.623, 0.683),
    }
    def closest(x):
        c = '0'
        cd = 10000
        for labelInt in emotion_to_vad_generated:
            label = emotion_to_vad_generated[labelInt]
            d = np.sqrt(np.sum((x - label) ** 2))
            if d < cd:
                c = labelInt
                cd = d
            # try with douoble intensity setting
            d = np.sqrt(np.sum((x - label) ** 2))
            if d < cd:
                c = labelInt
                cd = d
        return label_conversion[c]
    cSize = 3
    def load_audio_file(file_path, input_length=int(cSize*16000)):
        data = librosa.load(file_path, sr=16000)[0] #, sr=16000
        global audioLen
        audioLen = len(data) // 16000
        nChunks = data.shape[0]//input_length
        nSeconds = data.shape[0]//16000
        chunks = []
        for i in range(nSeconds-input_length//16000 - 1):
            chunk = data[i*16000:i*16000 + input_length]
            chunks.append(chunk)

        return chunks

    def process_chunks(chunks):
        res = []
        for chunk in chunks:
            input_values = tokenizer(chunk, return_tensors="pt").input_values
            input_values = input_values.to('cuda')
            with torch.no_grad():
                features = wav2vec2(input_values).last_hidden_state
            out = model(features)
            out = out.cpu().detach().numpy()
            res.append(out)
        return res
    chunks = load_audio_file('temp/to_analyse.mp3')
    pres = process_chunks(chunks)
    res= [pres[0]]


    for i in range(1, len(pres)):
        #res.append(np.mean(pres[i-1:i+cSize-1], axis=0))
        inertia = pres[i] - res[i-1]
        res.append(res[i-1] + inertia/cSize)
    return res

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']

    if file.filename is None or file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if file:
        file_like_object = io.BytesIO(file.read())
        fileName = 'temp/' + file.filename
        with open(fileName, 'wb') as f:
            f.write(file_like_object.getbuffer())
        process_file(fileName)
        res = emo_recognise()
        valence = []
        arousal = []
        dominance = []
        for r in res:
            valence.append(r[0][0])
            arousal.append(r[0][1])
            dominance.append(r[0][2])
        valenceString = ','.join([str(x) for x in valence])
        arousalString = ','.join([str(x) for x in arousal])
        dominanceString = ','.join([str(x) for x in dominance])
        return render_template('upload.html', data_valence = valenceString, data_arousal = arousalString, data_dominance = dominanceString, data_time = str(audioLen))
        return jsonify({'message': 'File successfully uploaded and processed'})


if __name__ == '__main__':
    app.run(debug=True)

