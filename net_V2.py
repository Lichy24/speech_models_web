from multiprocessing.sharedctypes import Value
from unittest import result
import torch
import torchaudio
import sounddevice as sd
import numpy as np
from model.net import Convolutional_Speaker_Identification


#def most_prob_voting(List):
    #return list(sorted(List.items()))[0][0]
#new voting
def most_prob_voting(List):
    return list(sorted(List.items(), key=lambda x: x[1]))[-1][0]


def split_wev(speech):
    file_list = []
    #print(len(speech[0]))
    if len(speech[0]) >= 48000:#48044
        speech = speech[0][44:]
        splits = int(len(speech) / 47956)
        speech = speech[None, :]
        for j in range(splits):
            file_list.append(speech[0][j * 47956:(j + 1) * 47956])
    else:
        print("file to small")
    return file_list


def record(freq=48000, duration=4):
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
        model = torch.load("models1/models/" + lan.lower() + "model0.pth", map_location=torch.device('cpu'))
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

        x = emission[None, :].clone()
        softmax = torch.exp(model1(x))
        out = torch.argmax(softmax)

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

def predict_acc(wav2vec,res_dict,model):
    frequents = {}
    x = wav2vec[None, :].clone()
    softmax = torch.exp(model(x))
    out = torch.argmax(softmax)
    res_string = "("
    for i, z in enumerate(softmax[0]):
        res_string += f'{res_dict[i]} = {torch.round(z, decimals=2) * 100} %,'
    res_string += ")"
    proba = int(torch.max(softmax) * 100)
    #print("model 1 is " + str(proba) + " % sure")
    res = res_dict[int(out)]
    #print(res)
    #print("--------")
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

    return most_prob_voting(frequents),res_string

def predict_from_list(file_list, res_dict, model1):
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    w2v = bundle.get_model()
    frequents = {}
    results = []
    for f in file_list:
        f = f[None, :]
        with torch.inference_mode():
            emission, _ = w2v(f)

        x = emission[None, :].clone()
        softmax = torch.exp(model1(x))
        out = torch.argmax(softmax)
        
        result_list = []
        for i, z in enumerate(softmax[0]):
            result_list.append(z)
        #np_result = np.array([result_list])
        np_result = softmax[0].detach().numpy()
        proba = int(torch.max(softmax) * 100)
        #print("model 1 is " + str(proba) + " % sure")
        res = res_dict[int(out)]
        #print(res)
        #print("--------")
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

        if results:
            results[0] = np.add(results[0],np_result)
            results[1] += 1    
        else:
            results = [np_result,1]

    final_result = np.divide(results[0],results[1])
    
    res_string = "("
    for i, z in enumerate(final_result):
        res_string += f'{res_dict[i]} = {round(z * 100,2)} %,'
    res_string += ")"

    return str(most_prob_voting(frequents)),res_string

def accent_inferece(wav2vec,accent_models):
    #files = split_wev(wav2vec)

    lan_model, result_dict = accent_models['lan']

    #selected_lan = str(pedict_from_list(files, result_dict, lan_model))
    ans,li_lan = predict_acc(wav2vec, result_dict, lan_model)
    
    selected_lan = str(ans)
    result = "\n The language model select: " + selected_lan + "\n--------"+li_lan
    # print("\n The language model select: " + selected_lan)
    # print("\n--------")
    accent_model, result_dict = accent_models[selected_lan]
    ans,li_lan = predict_acc(wav2vec, result_dict, accent_model)
    result += "\n |The answer is : " + ans +  "\n--------"+li_lan
    # print("\n |The answer is : " + pedict_from_list(files, result_dict, accent_model) + "|")
    #return "\n The language model select: " + selected_lan + "\n--------" + "\n |The answer is : " + predict_acc(wav2vec, result_dict, accent_model) + "|"
    return result

def inf_pred(model,file_name):
    speech_array, sampling_rate = torchaudio.load(file_name, normalize=True)
    transform = torchaudio.transforms.Resample(sampling_rate, 16_000)
    speech_array = transform(speech_array)
    #print(len(speech_array[0]))
    # if you want to record by yourself...
    # speech_array = record()

    files = split_wev(speech_array)
    #print(files)
    lan_model, result_dict = model['lan']

    selected_lan,prob_result = predict_from_list(files, result_dict, lan_model)
    # print("\n The language model select: " + selected_lan)
    # print("\n--------")
    result = "\n The language model select: " + selected_lan + "\n"+prob_result
    accent_model, result_dict = model[selected_lan]
    selected_accent,accent_prob_result = predict_from_list(files, result_dict, accent_model)

    result += "\n |The accent of the language is : " + selected_accent + "\n"+accent_prob_result
    # print("\n |The answer is : " + pedict_from_list(files, result_dict, accent_model) + "|")
    return result

def pred(file_name='common_voice_en_19654103.mp3'):
    speech_array, sampling_rate = torchaudio.load(file_name, normalize=True)
    transform = torchaudio.transforms.Resample(sampling_rate, 16_000)
    speech_array = transform(speech_array)
    print(len(speech_array[0]))
    # if you want to record by yourself...
    # speech_array = record()

    files = split_wev(speech_array)
    print(files)
    lan_model, result_dict = get_model_and_dict("lan")

    selected_lan = str(pedict_from_list(files, result_dict, lan_model))
    # print("\n The language model select: " + selected_lan)
    # print("\n--------")
    accent_model, result_dict = get_model_and_dict(selected_lan)
    # print("\n |The answer is : " + pedict_from_list(files, result_dict, accent_model) + "|")
    return "\n The language model select: " + selected_lan + "\n--------" + "\n |The answer is : " + pedict_from_list(
        files, result_dict, accent_model) + "|"


#print(pred(r"C:\\Users\\ASUS\\OneDrive\\Desktop\\audio files\\Langauge\\french_male.wav"))
