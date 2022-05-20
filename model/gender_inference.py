import numpy as np
import torch
import torchaudio
import sounddevice
from scipy.io.wavfile import write
from cnn_model_definition_gender import ConvNet_roi_orya

SAMPLE_RATE = 16000

device = torch.device("cpu")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
wav_model = bundle.get_model().to(device)

classes = {0: "male", 1: "female"}


def Norm(X):
    new_embedding = X.detach().cpu().numpy()
    for i in range(len(new_embedding)):
        mlist = new_embedding[0][i]
        new_embedding[0][i] = 2 * (mlist - np.max(mlist)) / (np.max(mlist) - np.min(mlist)) + 1
    return torch.from_numpy(new_embedding).to(device)


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
    result = f'Predicted: {classes[max].capitalize()} \n'
    result += f'Male: {round(predict[0][0] * 100, 4)}% \n'
    result += f'Female:  {round(predict[0][1] * 100, 4)}% \n'
    return result

def inf_gender(cnn,embedding):
        #embedding = embedding.unsequeeze(0)
        #embedding = Norm(embedding)
        y = cnn(embedding)
        ans = print_results(y)
        return ans

if __name__ == '__main__':
    gender_model = torch.load("36gender_Model-epoch_36_Weights.pth", map_location=torch.device("cpu"))
    gender_model.eval()

    with torch.inference_mode():
        tor = inference(file_path="dvir.wav")
        embedding, _ = wav_model(tor)
        embedding = embedding.unsqueeze(0)
        embedding = Norm(embedding)
        y = gender_model(embedding)
        ans = print_results(y)
        print(ans)
