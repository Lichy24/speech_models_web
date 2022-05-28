import os
from flask import Flask, render_template, request,jsonify,make_response,send_from_directory
from model.inference_dor_dolev import inf_mood
from model.gender_inference import inf_gender
from dadaNet import ConvNet
from model.Dataset import Data
from cnn_model_definition_gender import ConvNet_roi_orya
from model.language_inference import Recording_language_classification as rec
import torch
from net_V2 import pred,accent_inferece,inf_pred
from net import Convolutional_Speaker_Identification
from pydub import AudioSegment
from model.age_model_inf import inf_age
import torchaudio
import numpy as np


app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def selected_models(request):
    print(request.form.getlist('models_selection'))
    return request.form.getlist('models_selection')

def get_list_models(request):
    list_models_selected = {"gender":"","age":"","mood":"","accent":"","language_identification":""}
    current_selected = selected_models(request)
    for model in list_models_selected.keys():
        if model in current_selected:
            list_models_selected[model] = "checked"
        else:
            list_models_selected[model] = ""
    return list_models_selected


def Norm(X):
    new_embedding = X.detach().cpu().numpy()
    for i in range(len(new_embedding)):
        mlist = new_embedding[0][i]
        new_embedding[0][i] = 2 * (mlist - np.max(mlist)) / (np.max(mlist) - np.min(mlist)) + 1
    return torch.from_numpy(new_embedding)


def load_model(name):
    model = torch.load(f"model/{name}", map_location=device)
    model.eval()

    return model

def load_model_new(name):
    model = torch.load(f"{name}", map_location=device)
    model.eval()

    return model

saved_model = {"gender":"model_gender_orya_roi.pth",\
                "age":"age_Binary_Model-e_11_Weights.pth",\
                "mood":"model_dor_dolev.pth",\
                "language_identification":"model_language_identification.pth"}


def load_state(name):
    model = Convolutional_Speaker_Identification()
    model.load_state_dict(torch.load(name, map_location=device))
    model.eval()

    return model

accent_models = {"en":[load_state("models1/models/enstat0.pth"),{0: 'us_en', 1: 'england_en', 2: 'canada_en'}],\
                "ca":[load_state("models1/models/castat0.pth"),{0: 'balearic_ca', 1: 'central_ca', 2: 'valencian_ca'}],\
                "fr":[load_model_new("models1/models/frmodel0.pth"),{0: 'canada_fr', 1: 'france_fr', 2: 'belgium_fr', 3: 'france_fr'}],\
                "eu":[load_model_new("models1/models/eumodel0.pth"),{0: 'mendebalekoa_eu', 1: 'erdialdekoa_nafarra_eu'}],\
                "zh-CN":[load_model_new("models1/models/zh-cnmodel0.pth"),{0: '440000_zh-CN', 1: '450000_zh-CN', 2: '110000_zh-CN', 3: '330000_zh-CN'}],\
                "es":[load_model_new("models1/models/esmodel0.pth"),{0: 'andino_es', 1: 'nortepeninsular_es', 2: 'chileno_es'}],\
                "de":[load_model_new("models1/models/demodel0.pth"), {0: 'switzerland_de', 1: 'austria_de', 2: 'germany_de'}],\
                "lan":[load_model_new("models1/models/lanmodel0.pth"), {0: 'eu', 1: 'de', 2: 'es', 3: 'ca', 4: 'fr', 5: 'zh-CN', 6: 'en'}]}

def file_to_wav(filename):
    name, extension = os.path.splitext(filename)
    AudioSegment.from_file(filename,extension[1:]).export(name+'.wav',format='wav')
    return name+'.wav'








def load_wav(filename):
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    wav_model = bundle.get_model().to(device)

    waveform, sr = torchaudio.load(filename)#normalized here is good for most models
    if sr != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)


    waveform = waveform[:1,:48000]
    waveform.to(device)
    embedding, _ = wav_model(waveform)

    #print(str(embedding.size()))

    return embedding

def send_to_models(filename,selected_models):
    if not selected_models:
        return "No Model has been selected."
    dict_res = {"gender":["display:none;",""],"age":["display:none;",""],"mood":["display:none;",""],"accent":["display:none;",""],"language_identification":["display:none;",""]}
    wav2vec_2 = load_wav(filename)
    wav2vec_2 = wav2vec_2.unsqueeze(0)
    wav2vec_2 = Norm(wav2vec_2)

    if "accent" in selected_models:
        dict_res['accent'] = ["",inf_pred(accent_models,filename)]
    if "language_identification" in selected_models:
        dict_res['language_identification'] = ["",rec.get_string_of_ans('model/'+saved_model["language_identification"],filename) + "\n\n"]
    if "mood" in selected_models:
        dict_res['mood'] = ["",inf_mood(model_mood,wav2vec_2) + "\n\n"]
    if "gender" in selected_models:
        dict_res['gender'] = ["",inf_gender(model_gender,wav2vec_2) + "\n\n"]
    if "age" in selected_models:
        dict_res['age'] = ["",inf_age(wav2vec_2,model_age) + "\n\n"]


    return dict_res



model_gender = load_model(saved_model["gender"])
model_age = load_model(saved_model["age"])
model_mood = load_model(saved_model["mood"])


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),'favicon.ico')


@app.route('/')
def upload():
    return render_template("index.html",check_gender="checked",audio_record="display:none;",
    gender_model_hide="display:none;",age_model_hide="display:none;",mood_model_hide="display:none;",
    accent_model_hide="display:none;",language_model_hide="display:none;",results_hide="display:none;",audio_player="display:none;")

@app.route('/index', methods=['POST'])
def send_text():
    if request.method == 'POST':
        if "audio_file" in request.files:
            print("Upload file")
            uploaded_file = request.files['audio_file']
            filename_to_play = uploaded_file.filename
            uploaded_file.save(filename_to_play)
            name, extension = os.path.splitext(filename_to_play)
            if extension != '.wav':
                filename_to_play = file_to_wav(filename_to_play)
            dict_res = send_to_models(filename_to_play,selected_models(request))
            os.remove(f'{uploaded_file.filename}')
            if uploaded_file.filename != filename_to_play:
                os.remove(f'{filename_to_play}')
            return make_response(jsonify(dict_res),200)

        return make_response(jsonify({'error':'missing file.'}),500)
        

def upload_file():
    print("file")



if __name__ == '__main__':
    """
    NOTE: using WSL which does not support USB ports hence cannot notice any microphones connected,
    there for need to run the recording code on WINDOWS OS and use saved wav file to upload to models using LINUX OS.

    already working:
    4 models which can select which model to use

    need help with models:
    #language identification does not recevie a wav file

    need to run:
    #new age model
    """

    app.run()