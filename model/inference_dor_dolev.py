import os
#import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torchaudio
import os
import sounddevice
from scipy.io.wavfile import write
from numpy import mat
from model.Dataset import Data


from dadaNet import ConvNet


SAMPLE_RATE = 16000

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# fine-tuning from wav2vec pytorch pipline
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)

classes = {0: "Positive", 1: "Neutral", 2: "Negative"}


def Norm(X):
    new_embedding = X.detach().cpu().numpy()
    for i in range(len(new_embedding)):
        mlist = new_embedding[0][i]
        new_embedding[0][i] = 2 * (mlist - np.max(mlist)) / (np.max(mlist) - np.min(mlist)) + 1
    return torch.from_numpy(new_embedding).to(device)


def recording(name):
    filename = name
    duration = 3
    print("Recording ..")
    recording = sounddevice.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sounddevice.wait()
    print("Done.")
    write(filename + ".wav", SAMPLE_RATE, recording)
    return filename + ".wav"


def inference(file_name):
    waveform, sr = torchaudio.load(recording(file_name), num_frames = SAMPLE_RATE*3)
    
    if sr != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

    waveform = waveform.to(device)

    return waveform


def results(y):
    y = y.cpu().detach().numpy()
    predict = [np.exp(c) for c in y]
    max = np.argmax(predict)
    result = f'Predicted: {classes[max].capitalize()} \n'
    result += f'Positive: {round(predict[0][0] * 100, 4)}% \n'
    result += f'Neutral:  {round(predict[0][1] * 100, 4)}% \n'
    result += f'Negative: {round(predict[0][2] * 100, 4)}% \n'

    return result

def print_results(y):
    y = y.cpu().detach().numpy()
    predict = [np.exp(c) for c in y]
    max = np.argmax(predict)
    print(f'Predicted: {classes[max].capitalize()}')
    print(f'Positive: {round(predict[0][0] * 100, 4)}%')
    print(f'Neutral:  {round(predict[0][1] * 100, 4)}%')
    print(f'Negative: {round(predict[0][2] * 100, 4)}%')

def upload(file_name):
    waveform, sr = torchaudio.load(file_name, num_frames = SAMPLE_RATE*3)
    
    if sr != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

    waveform = waveform.to(device)

    return waveform

def inf_mood(cnn,embedding):
    #with torch.inference_mode():
        #embedding = embedding.unsqueeze(0)
        #embedding = Norm(embedding)
        y = cnn(embedding)
        #y = cnn(embedding.unsqueeze(0))
        return results(y)


classes = {0: "Positive", 1: "Neutral", 2: "Negative"}
def test(file_path,cnn):

    with torch.inference_mode():
        waveform, sr = torchaudio.load(file_path, num_frames = SAMPLE_RATE*3)
        
        if sr != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

        waveform = waveform.to(device)
        embedding, _ = model(waveform)
        embedding = embedding.unsqueeze(0)
        embedding = Norm(embedding)
        y = cnn(embedding)
        print_results(y)

if __name__ == '__main__':
    cnn = torch.load("model_dor_dolev.pth", map_location=torch.device("cpu"))
    cnn.eval()

    with torch.inference_mode():
        tor = inference("negative_01.wav")
        embedding, _ = model(tor)
        embedding = embedding.unsqueeze(0)
        embedding = Norm(embedding)
        y = cnn(embedding)
        print_results(y)
