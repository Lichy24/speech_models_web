import torch
import numpy as np
from model.cnn_model_definition_language_identification import Convolutional_Language_Identification


NUM_LANGUAGE = 30
rep = {0: 'Estonian', 1: 'Portuguese', 2: 'Tatar', 3: 'Welsh', 4: 'Arabic', 5: 'Catalan', 6: 'German', 7: 'Spanish', 8: 'Basque', 9: 'English', 10: 'French', 11: 'Esperanto', 12: 'Italian', 13: 'Kabyle', 14: 'Rwanda', 15: 'Russian', 16: 'Chinese', 17: 'Latvian', 18: 'Indonesian', 19: 'Sorbian', 20: 'Slovenian', 21: 'Tamil', 22: 'Romansh', 23: 'Greek', 24: 'Hungarian', 25: 'Mongolian', 26: 'Thai', 27: 'Sakha', 28: 'Frisian', 29: 'Persian'}

# rep = {0: 'et', 1: 'pt', 2: 'tt', 3: 'cy', 4: 'ar', 5: 'ca', 6: 'de', 7: 'es', 8: 'eu', 9: 'en', 10: 'fr', 11: 'eo', 12: 'it', 13: 'kab', 14: 'rw', 15: 'ru', 16: 'zh-CN', 17: 'lv', 18: 'id', 19: 'hsb', 20: 'sl', 21: 'ta', 22: 'rm-sursilv', 23: 'el', 24: 'hu', 25: 'mn', 26: 'th', 27: 'sah', 28: 'fy-NL', 29: 'fa'}
# recode = {'et': 'Estonian', 'pt': 'Portuguese', 'tt': 'Tatar', 'cy': 'Welsh', 'ar': 'Arabic', 'ca': 'Catalan', 'de': 'German', 'es': 'Spanish', 'eu': 'Basque', 'en': 'English', 'fr': 'French', 'eo': 'Esperanto', 'it': 'Italian', 'kab': 'Kabyle', 'rw': 'Rwanda', 'ru': 'Russian', 'zh-CN': 'Chinese', 'lv': 'Latvian', 'id': 'Indonesian', 'hsb': 'Sorbian', 'sl': 'Slovenian', 'ta': 'Tamil', 'rm-sursilv': 'Romansh', 'el': 'Greek', 'hu': 'Hungarian', 'mn': 'Mongolian', 'th': 'Thai', 'sah': 'Sakha', 'fy-NL': 'Frisian', 'fa': 'Persian'}

MODEL_PATH = '/home/adi/python/DeepLearningSoundUI-master/model/model_language_identification.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def _get_model(cls):
        model = Convolutional_Language_Identification(NUM_LANGUAGE).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        return model

    @classmethod
    def _get_list_of_results(cls, sample):
        model = cls._get_model()
        with torch.no_grad():
            model.eval()
            proba_pred_y = model(sample.to(device))
            normalized_res = cls._normalization(proba_pred_y)
            return normalized_res

    @classmethod
    def _get_language_by_vector(cls, sample):
        list_of_x = [sample]
        x_stack = torch.stack(list_of_x)
        list_of_results = cls._get_list_of_results(x_stack)
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
    def get_string_of_ans(cls, sample):
        top_k, result_relative_to_top_k = cls._get_language_by_vector(sample)
        ans = {}
        for i in range(len(top_k)):
            ans[top_k[i][0]] = (top_k[i][1], result_relative_to_top_k[i][1])
        str_of_res = []
        dict_place = {1:'First',2:'Second',3: 'Third'}
        j = 1
        for k, v in ans.items():
            str_of_res.append(f'{dict_place[j]} place: the {k} language the result is {round(v[0]*100,3)} out of all and {round(v[1]*100,3)} out of the top 3')
            j+=1
        final_answer = "\n".join(str_of_res)
        return final_answer

