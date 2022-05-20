import numpy as np
import torch
import torchaudio
import sounddevice
from scipy.io.wavfile import write

predict_gender = {0: "male", 1: "female"}


def record(filename):
    sr = 16000
    sec = 4
    print("Recording")

    rec = sounddevice.rec(int(sec * sr), samplerate=sr, channels=1)
    sounddevice.wait()
    write(filename + ".wav", sr, rec)


def inferrnce_rec_or_upload(model, name):
    even = 0

    path = f"{name}"

    # loading the audio
    speech_array, sampling_rate = torchaudio.load(path, normalize=True)
    transform = torchaudio.transforms.Resample(sampling_rate, 16_000)
    speech_array = transform(speech_array)
    array = speech_array[0]

    # removing empty frames from the beginning of the recording
    array = array[44:]

    # model definition - num of labels
    sum = [0, 0]

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    WAV2VEC2_model = bundle.get_model()

    # defining 3 seconds per recording
    window = 48000
    step = 32000
    pointer = 0
    n_samples = 0

    while pointer + window <= len(array):
        curr = array[pointer:pointer + window]
        #windows OS check
        #tensor_data = torch.tensor(np.array(curr))
        #linux
        tensor_data = torch.tensor(np.array([curr]))

        with torch.inference_mode():
            features, _ = WAV2VEC2_model(tensor_data)
            features = features.unsqueeze(0)

            with torch.no_grad():
                outputs = model(features)
                p = torch.exp(outputs)
                _, predicted = torch.max(p, 1)
                pred = predicted[0].item()

                n_samples += 1
                sum[pred] += 1

        pointer += step

    predict = predict_gender[np.argmax(sum)]
    # if sum[0] == sum[1]:
    #     predict = "female"
    #     even += 1
    print(sum, f'-> {predict}')
    return f"{predict}"


def inf_rec(model1):
    # recording and saving the audio
    record("recording")
    return inferrnce_rec_or_upload(model1, name="recording.wav") + "\n" 


def inf_upload(model1, name):
    if name == '':
        return "No file found"
    return inferrnce_rec_or_upload(model1, name=name) + "\n" 
