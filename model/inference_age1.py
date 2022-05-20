import numpy as np
import torch
import torchaudio


# 10-20 , 20-80  binary
# youth , adult

def inf_age(age_model, recording_path):

    # loading the audio
    speech_array, sampling_rate = torchaudio.load(recording_path, normalize=True)
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
        tensor_data = torch.tensor(np.array([curr]))

        with torch.inference_mode():
            features, _ = WAV2VEC2_model(tensor_data)
            features = features.unsqueeze(0)

            with torch.no_grad():
                outputs = age_model(features)

                # euler fix
                p = torch.exp(outputs)
                _, predicted = torch.max(p, 1)
                pred = predicted[0].item()

                n_samples += 1
                sum[pred] += 1

        pointer += step

    ans = np.argmax(sum)
    if ans == 1:
        return 'Adult'
    else:
        return 'Teen'


# an example of using inf_age

#model = torch.load("binary_models_age\\27.4\\CoVoVox_lr0.0001_Weights_NM\\age_Binary_Model-e_14_Weights.pth")
#rec_path = '..\\Age_Model\\data_test\\id10256_male_sixties_EWhT8PtQ7rs_0007.wav'  # random rec to test
#print(inf_age(model, rec_path))
