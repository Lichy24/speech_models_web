import torch
import torchaudio
import sounddevice as sd
from net import Convolutional_Speaker_Identification
import os
from pydub import AudioSegment


def most_prob_voting(List):
    return list(sorted(List.items(), key=lambda x: x[1]))[-1][0]

def file_to_wav(filename):
    name, extension = os.path.splitext(filename)
    AudioSegment.from_file(filename,extension[1:]).export(name+'.wav',format='wav')
    return name+'.wav'

def split_wev(speech):
    file_list = []
    if len(speech[0]) >= 48000:
        speech = speech[0][44:]
        splits = int(len(speech) / 47956)
        speech = speech[None, :]
        for j in range(splits):
            file_list.append(speech[0][j * 47956:(j + 1) * 47956])
    else:
        print("file to small")
    return file_list


def record(freq=48000, duration=3):
    recording = sd.rec(int(duration * freq),
                       samplerate=freq, channels=1)
    print("Recording...")
    sd.wait()
    transform = torchaudio.transforms.Resample(freq, 16_000)

    return transform(torch.Tensor(recording).T)


def get_model_and_dict(lan):
    if lan == "en":
        lan_dict = {0: 'us_en', 1: 'england_en', 2: 'canada_en'}
        model = Convolutional_Speaker_Identification()
        model.load_state_dict(torch.load("models1/models/" + lan + "stat0.pth", map_location=torch.device('cpu')))
        model.eval()
        return model, lan_dict
    elif lan == "ca":
        lan_dict = {0: 'balearic_ca', 1: 'central_ca', 2: 'valencian_ca'}
        model = Convolutional_Speaker_Identification()
        model.load_state_dict(torch.load("models1/models/" + lan + "stat0.pth", map_location=torch.device('cpu')))
        model.eval()
        return model, lan_dict
    elif lan == "fr":
        model = torch.load("models1/models/" + lan + "model0.pth", map_location=torch.device('cpu'))
        lan_dict = {0: 'canada_fr', 1: 'france_fr', 2: 'belgium_fr', 3: 'france_fr'}
        model.eval()
        return model, lan_dict
    elif lan == "eu":
        model = torch.load("models1/models/" + lan + "model0.pth", map_location=torch.device('cpu'))
        lan_dict = {0: 'mendebalekoa_eu', 1: 'erdialdekoa_nafarra_eu'}
        model.eval()
        return model, lan_dict
    elif lan == "zh-CN":
        model = torch.load("models1/models/" + lan + "model0.pth", map_location=torch.device('cpu'))
        lan_dict = {0: '440000_zh-CN', 1: '450000_zh-CN', 2: '110000_zh-CN', 3: '330000_zh-CN'}
        model.eval()
        return model, lan_dict
    elif lan == "es":
        model = torch.load("models1/models/" + lan + "model0.pth", map_location=torch.device('cpu'))
        lan_dict = {0: 'andino_es', 1: 'nortepeninsular_es', 2: 'chileno_es'}
        model.eval()
        return model, lan_dict
    elif lan == "de":
        model = torch.load("models1/models/" + lan + "model0.pth", map_location=torch.device('cpu'))
        lan_dict = {0: 'switzerland_de', 1: 'austria_de', 2: 'germany_de'}
        model.eval()
        return model, lan_dict
    elif lan == "lan":
        model = torch.load("models1/models/" + lan + "model0.pth", map_location=torch.device('cpu'))
        lan_dict = {0: 'eu', 1: 'de', 2: 'es', 3: 'ca', 4: 'fr', 5: 'zh-CN', 6: 'en'}
        model.eval()
        return model, lan_dict
    else:
        print(lan == "zh-CN")


def pedict_from_list(file_list, res_dict, model1):
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    w2v = bundle.get_model()
    frequents = {}
    for f in file_list:
        f = f[None, :]
        with torch.inference_mode():
            emission, _ = w2v(f)
            print(emission.size())

        x = emission[None, :].clone()
        softmax = torch.exp(model1(x))
        out = torch.argmax(softmax)

        print(x.size())

        for i, x in enumerate(softmax[0]):
            print(f'{res_dict[i]} = {x * 100} %', )

        proba = int(torch.max(softmax) * 100)
        print("model 1 is " + str(proba) + " % sure")
        res = res_dict[int(out)]
        print(res)
        print("--------")
        if res not in frequents.keys():
            frequents[res] = 0
        if res == "zh-CN":
            frequents[res] += proba * 0.3
        elif res == "ca":
            if "es" not in frequents.keys():
                frequents["es"] = 0
            frequents["es"] += proba
        else:
            frequents[res] += proba

    return most_prob_voting(frequents)


def pred(file_name='de_german.mp3'):
    file_name = file_to_wav(file_name)
    speech_array, sampling_rate = torchaudio.load(file_name, normalize=True)
    transform = torchaudio.transforms.Resample(sampling_rate, 16_000)
    speech_array = transform(speech_array)


    # if you want to record by yourself...
    # speech_array = record()

    files = split_wev(speech_array)

    lan_model, result_dict = get_model_and_dict("lan")

    selected_lan = str(pedict_from_list(files, result_dict, lan_model))
    # print("\n The language model select: " + selected_lan)
    # print("\n--------")
    accent_model, result_dict = get_model_and_dict(selected_lan)
    # print("\n |The answer is : " + pedict_from_list(files, result_dict, accent_model) + "|")
    return "\n The language model select: " + selected_lan + "\n--------" + "\n |The answer is : " + pedict_from_list(
        files, result_dict, accent_model) + "|"


print(pred("fr_belgium.mp3"))
