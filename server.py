from locale import normalize
import os
#from pyexpat import model
#from unicodedata import name
from flask import Flask, render_template, request,Response,send_file,send_from_directory
#from model.inference_orya_roi_gender import inf_rec, inf_upload
#from model.inference_gender import inf_rec, inf_upload
#from model.inference_age1 import inf_age
from model.inference_dor_dolev import recording,inference,inf_mood,test
#from model.age_inference import inf_age
from model.gender_inference import inf_gender
from dadaNet import ConvNet
from model.Dataset import Data
from cnn_model_definition_gender import ConvNet_roi_orya
#from Binary_Age_Model.model_definition_age import Convolutional_Neural_Network_Age
#from model_definition_age import Convolutional_Neural_Network_Age
from model.language_inference import Recording_language_classification as rec
import torch
from net_V2 import pred,accent_inferece,inf_pred
from net import Convolutional_Speaker_Identification
from pydub import AudioSegment
from model.age_model_inf import inf_age
import sounddevice
import torchaudio
#import tensorflow as tf
from scipy.io.wavfile import write
import random

#import simpleaudio as sa

#import wave
import numpy as np


app = Flask(__name__)

""""
            print('play')
            uploaded_file = request.files['file']
            if 'play_obj' in locals():
                if play_obj.is_playing():
                    print('stop playing')
                    play_obj.stop()
            else:
                wave_obj = sa.WaveObject.from_wave_file(uploaded_file)
                play_obj = wave_obj.play()
                print('playing')
            return render_template("index.html")

@app.route("/wav")
def streamwav():
    def generate():
        with open("recording.wav", "rb") as fwav:
            data = fwav.read(1024)
            while data:
                yield data
                data = fwav.read(1024)
    return Response(generate(), mimetype="audio/x-wav")

            wav = wave.open(uploaded_file.filename,'r')
            filename = uploaded_file.filename
            flag = False
            if wav.getnchannels() > 1:
                name, extension = os.path.splitext(filename)
                filename = name+'_ch1'+extension
                save_wav_channel(filename,wav,0)
                flag = True
            wav.close()
"""


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
    model = torch.load(f"model/{name}", map_location=torch.device('cpu'))
    model.eval()

    return model

def load_model_new(name):
    model = torch.load(f"{name}", map_location=torch.device('cpu'))
    model.eval()

    return model

saved_model = {"gender":"model_gender_orya_roi.pth",\
                "age":"age_Binary_Model-e_11_Weights.pth",\
                "mood":"model_dor_dolev.pth",\
                "language_identification":"model_language_identification.pth"}


def load_state(name):
    model = Convolutional_Speaker_Identification()
    model.load_state_dict(torch.load(name, map_location=torch.device('cpu')))
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






def record(filename):
    sr = 16000
    sec = 3
    print("Recording")

    rec = sounddevice.rec(int(sec * sr), samplerate=sr, channels=1)
    sounddevice.wait()
    write(filename + ".wav", sr, rec)
    return "recorded"
    #return send_to_models(filename + ".wav",selected_models)

def split_wev(speech):
    file_list = []
    if len(speech[0]) > 48000:
        speech = speech[0][44:]
        splits = int(len(speech) / 47956)
        speech = speech[None, :]
        for j in range(splits):
            file_list.append(speech[0][j * 47956:(j + 1) * 47956])
    else:
        print("file to small")
    return file_list

def load_wav(filename):
    device = torch.device("cpu")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    wav_model = bundle.get_model().to(device)

    waveform, sr = torchaudio.load(filename,normalize=True)#normalized here is good for most models
    if sr != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)


    clear = waveform[0][44:]
    clear = clear[None, :]
    clear = clear[0][:47956]
    embedding, _ = wav_model(clear[None,:])

    print(str(embedding.size()))
    return embedding

def load_wav_new(filename):
    device = torch.device("cpu")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    wav_model = bundle.get_model().to(device)

    waveform, sr = torchaudio.load(filename)#normalized here is good for most models
    if sr != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)


    waveform = waveform[:1,:48000]
    embedding, _ = wav_model(waveform)

    print(str(embedding.size()))

    return embedding

def send_to_models(filename,selected_models):
    if not selected_models:
        return "No Model has been selected."
    dict_res = {"gender":["display:none;",""],"age":["display:none;",""],"mood":["display:none;",""],"accent":["display:none;",""],"language_identification":["display:none;",""]}
    #wav2vec_3 = load_wav(filename)
    wav2vec_1 = load_wav_new(filename)
    wav2vec_2 = load_wav_new(filename)
    wav2vec_2 = wav2vec_2.unsqueeze(0)
    wav2vec_2 = Norm(wav2vec_2)

    if "accent" in selected_models:
        dict_res['accent'] = ["",inf_pred(accent_models,filename) + "\n"]
    if "language_identification" in selected_models:
        dict_res['language_identification'] = ["",rec.get_string_of_ans('model/'+saved_model["language_identification"],wav2vec_1) + "\n\n"]
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


@app.route('/')
def upload():
    return render_template("index.html",check_gender="checked",audio_record="display:none;",
    gender_model_hide="display:none;",age_model_hide="display:none;",mood_model_hide="display:none;",
    accent_model_hide="display:none;",language_model_hide="display:none;",results_hide="display:none;",audio_player="display:none;")

@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/index', methods=['POST', 'GET'])
def send_text():
    if request.method == 'POST':
        list_models_selected = get_list_models(request)
        if "submit_button" in request.form:
            
            if request.form['submit_button'] == 'rec':

                filename_recording = request.remote_addr+"_recording_1"
                if os.path.exists(request.remote_addr+"_recording_1.wav"):
                    os.remove(request.remote_addr+"_recording_1.wav")
                    filename_recording =request.remote_addr+"_recording_2"
                elif os.path.exists(request.remote_addr+"_recording_2.wav"):
                    os.remove(request.remote_addr+"_recording_2.wav")
                result = render_template("index.html", pred=record(filename_recording),check_gender=list_models_selected['gender'],
                                        check_age=list_models_selected['age'],check_mood=list_models_selected['mood'],
                                        check_accent=list_models_selected['accent'],check_language_identification=list_models_selected['language_identification'],audio_record="",audio_player="display:none;",audio_file=filename_recording+".wav",
                                        gender_model_hide="display:none;",age_model_hide="display:none;",mood_model_hide="display:none;",accent_model_hide="display:none;",language_model_hide="display:none;",results_hide="display:none;")
                returnAudioFile(filename_recording+".wav")
                return result
            elif request.form['submit_button'] == 'file':
                print("Upload file")
                uploaded_file = request.files['file']
                if uploaded_file.filename != '':
                    filename_to_play = uploaded_file.filename
                    uploaded_file.save(filename_to_play)
                    name, extension = os.path.splitext(filename_to_play)
                    if extension != '.wav':
                        filename_to_play = file_to_wav(filename_to_play)
                        os.remove(f'{uploaded_file.filename}')

                else:
                    if os.path.exists(request.remote_addr+"_recording_1.wav"):
                        filename_to_play = request.remote_addr+"_recording_1.wav"
                    else:
                        filename_to_play = request.remote_addr+"_recording_2.wav"
                dict_res = send_to_models(filename_to_play,selected_models(request))
                result = render_template("index.html",
                                        check_gender=list_models_selected['gender'],
                                        check_age=list_models_selected['age'],check_mood=list_models_selected['mood'],
                                        check_accent=list_models_selected['accent'],check_language_identification=list_models_selected['language_identification'],audio_record="display:none;",
                                        gender_model_hide=dict_res['gender'][0],gender_model=dict_res['gender'][1],age_model_hide=dict_res['age'][0],age_model=dict_res['age'][1],mood_model_hide=dict_res['mood'][0],
                                        mood_model=dict_res['mood'][1],accent_model_hide=dict_res['accent'][0],accent_model=dict_res['accent'][1],
                                        language_model_hide=dict_res['language_identification'][0],language_model=dict_res['language_identification'][1],results_hide="",results=filename_to_play)
                if uploaded_file.filename != '':
                    os.remove(f'{filename_to_play}')

                return result

            elif request.form['submit_button'] == 'clean':
                #get_list_models(request)
                return render_template("index.html", pred="")
                
            elif request.form['submit_button'] == 'rec_new':
                #get_list_models(request)
                return render_template("index.html")
        else:
            print("error")
            #return render_template("index.html", pred="error")
            return "got post"
        

        


@app.route('/wav/<filename>')
def returnAudioFile(filename):
    return send_file(
       filename, 
       mimetype="audio/wav", 
       as_attachment=True, 
       attachment_filename=filename)


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

    app.run(debug=True)

    #file_to_wav("M:\\audio files\\Accent\\de_german.mp3")
    #load_wav("de_german.wav")
    """
    for i in range(1,10):
        print(f'------{i}-----')
        test(f'/home/adi/python/audio files/Mood/negative_0{i}.wav',model_mood)
    """