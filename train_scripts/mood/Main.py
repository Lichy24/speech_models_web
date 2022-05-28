import os
import IPython
import numpy as np
import torch
import torchaudio
import random
import sounddevice
from scipy.io.wavfile import write
from numpy import mat
from Model import ConvNet


SAMPLE_RATE = 16000

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# fine-tuning from wav2vec pytorch pipline
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)

classes = {0: "Positive", 1: "Neutral", 2: "Negative"}


def Norm(X):
    embedding = X.detach().cpu().numpy()
    for i in range(len(embedding)):
        mlist = embedding[0][i]
        embedding[0][i] = 2 * (mlist - np.max(mlist)) / (np.max(mlist) - np.min(mlist)) + 1
    return torch.from_numpy(embedding).to(device)


def recording(name):
    filename = name
    duration = 3
    print("Recording ..")
    recording = sounddevice.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sounddevice.wait()
    print("Done.")
    write(filename + ".wav", SAMPLE_RATE, recording)
    return filename + ".wav"


def inference(path):
    signal = np.zeros((int(SAMPLE_RATE*3 ,)))

    waveform, sampling_rate = torchaudio.load(filepath=recording(path), num_frames=SAMPLE_RATE * 3)
    # waveform, sampling_rate = torchaudio.load(filepath=path, num_frames=SAMPLE_RATE * 3)


    waveform = waveform.to(device)
    waveform = waveform.detach().cpu().numpy()[0]

    if len(waveform) <= 48000 and len(waveform) >= 32000:
        signal[:len(waveform)] = waveform

        if sampling_rate < 48000: # if there is more to fill
          rest = len(signal) - len(waveform) # get the "rest length"
          filled_list = signal[:len(waveform)] # we don't want to choose zero values, so this list contains non-zero values only.
          signal[len(waveform):] = random.choices(filled_list, k=rest) # choose k values from the filled_list
          
        signal_final = np.array([np.array(signal)])
        signal_final = torch.from_numpy(signal_final).to(device)
        signal_final = signal_final.type(torch.FloatTensor).to(device)

        return signal_final

    return -1


def print_results(y):
    y = y.cpu().detach().numpy()
    predict = [np.exp(c) for c in y]
    max = np.argmax(predict)
    print(f'Predicted: {classes[max].capitalize()}')
    print(f'Positive: {round(predict[0][0] * 100, 4)}%')
    print(f'Neutral:  {round(predict[0][1] * 100, 4)}%')
    print(f'Negative: {round(predict[0][2] * 100, 4)}%')


if __name__ == '__main__':
    cnn = torch.load("dadaNet.pth", map_location=torch.device("cpu"))
    cnn.eval()

    with torch.inference_mode():
        tor = inference("recoring_test")
        embedding, _ = model(tor)
        embedding = embedding.unsqueeze(0)
        embedding = Norm(embedding)
        y = cnn(embedding)
        print_results(y)