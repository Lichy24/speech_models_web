import numpy as np
import torch
import torchaudio

SAMPLE_RATE = 16000

device = torch.device("cpu")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
wav_model = bundle.get_model().to(device)

classes = {0: "teen", 1: "adult"}

def Norm(X):
    embedding = X.detach().cpu().numpy()
    for i in range(len(embedding)):
        mlist = embedding[0][i]
        embedding[0][i] = 2 * (mlist - np.max(mlist)) / (np.max(mlist) - np.min(mlist)) + 1
    return torch.from_numpy(embedding).to(device)


def inference(file_path):
    waveform, sr = torchaudio.load(file_path, num_frames=SAMPLE_RATE * 3)

    if sr != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

    waveform = waveform.to(device)

    return waveform

def print_results(y):
    y = y.cpu().detach().numpy()
    predict = [np.exp(c) for c in y]
    max = np.argmax(predict)
    return classes[max].capitalize()

def inf_age(cnn,audio_file_path):
        tor = inference(file_path=audio_file_path)
        embedding, _ = wav_model(tor)
        embedding = embedding.unsqueeze(0)
        embedding = Norm(embedding)
        y = cnn(embedding)
        ans = print_results(y)
        return ans