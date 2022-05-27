import torch
import numpy as np
from model.cnn_model_definition_language_identification import Convolutional_Language_Identification
import torchaudio

NUM_LANGUAGE = 30
SAMPLE_RATE = 16000
rep = {0: 'Estonian', 1: 'Portuguese', 2: 'Tatar', 3: 'Welsh', 4: 'Arabic', 5: 'Catalan', 6: 'German', 7: 'Spanish', 8: 'Basque', 9: 'English', 10: 'French', 11: 'Esperanto', 12: 'Italian', 13: 'Kabyle', 14: 'Rwanda', 15: 'Russian', 16: 'Chinese', 17: 'Latvian', 18: 'Indonesian', 19: 'Sorbian', 20: 'Slovenian', 21: 'Tamil', 22: 'Romansh', 23: 'Greek', 24: 'Hungarian', 25: 'Mongolian', 26: 'Thai', 27: 'Sakha', 28: 'Frisian', 29: 'Persian'}



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
wav_model = bundle.get_model().to(device)

class Recording_language_classification:

    @classmethod
    def _normalization(cls, proba_pred_y):
        res =[]
        t = proba_pred_y[0]
        for i in t:
            x = np.exp(i.item())
            res.append(x)
        return res

    @classmethod
    def _get_model(cls,model_path):
        model = Convolutional_Language_Identification(NUM_LANGUAGE).to(device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        return model

    @classmethod
    def _get_list_of_results(cls, sample,model_path):
        model = cls._get_model(model_path)
        with torch.no_grad():
            model.eval()
            proba_pred_y = model(sample.to(device))
            normalized_res = cls._normalization(proba_pred_y)
            return normalized_res

    @classmethod
    def _get_language_by_vector(cls, sample,model_path):
        list_of_x = [sample]
        x_stack = torch.stack(list_of_x)
        list_of_results = cls._get_list_of_results(x_stack,model_path)
        top_k = cls._get_top_k_from_res(list_of_results, 3)
        res_relative_to_top_k = cls._get_res_relative_to_top_k(top_k)
        return top_k, res_relative_to_top_k

    @classmethod
    def _get_top_k_from_res(cls, res, k):
        arr = np.array(res)
        top_k = arr.argsort(axis=0)[-k:]
        arr2 = []
        for j in reversed(range(len(top_k))):
            m = top_k[j]
            arr2.append((rep[m], round(arr[m], 4)))
        return arr2

    @classmethod
    def _get_res_relative_to_top_k(cls, top_k):
        sum_of_top_k = sum([i[1] for i in top_k])
        res_in_k = []
        for dup in top_k:
            res_in_k.append((dup[0],dup[1]/sum_of_top_k))
        return res_in_k

    @classmethod
    def inference(cls,file_name):
        waveform, sr = torchaudio.load(file_name, num_frames = SAMPLE_RATE*3)
    
        if sr != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)

        waveform = waveform.to(device)

        return waveform
    @classmethod   
    def Norm(cls,X):
        embedding = X.detach().cpu().numpy()
        for i in range(len(embedding)):
            mlist = embedding[0][i]
            embedding[0][i] = 2 * (mlist - np.max(mlist)) / (np.max(mlist) - np.min(mlist)) + 1
        return torch.from_numpy(embedding).to(device)
    @classmethod
    def split_wev(cls,speech):
        file_list = []
        if len(speech[0]) > 48044:
            speech = speech[0][44:]
            splits = int(len(speech) / 48000)
            speech = speech[None, :]
            for j in range(splits):
                file_list.append(speech[0][j * 48000:(j + 1) * 48000])
        else:
            print("file to small")
        return file_list

    @classmethod
    def get_string_of_ans(cls, model_path,sample):
        #tor = cls.inference(sample)
        #embedding, _ = wav_model(tor)
        #sample = cls.Norm(sample)
        top_k, result_relative_to_top_k = cls._get_language_by_vector(sample,model_path)
        ans = {}
        for i in range(len(top_k)):
            ans[top_k[i][0]] = (top_k[i][1], result_relative_to_top_k[i][1])
        str_of_res = []
        str_of_res.append("Language:".ljust(10) + " | Total: |  For k:    ")
        str_of_res.append("___________|________|_________")
        for k, v in ans.items():
            str_of_res.append(f'{k.ljust(10)} : {round(v[0]*100,3)}  |  {round(v[1]*100,3)}')

        final_answer = "\n".join(str_of_res)
        return final_answer


#print(Recording_language_classification.get_string_of_ans("/home/adi/python/DeepLearningSoundUI-master/recording.wav",'/home/adi/python/DeepLearningSoundUI-master/model/model_language_identification.pth'))