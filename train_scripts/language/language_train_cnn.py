
from cnn_model_definition import Convolutional_Language_Identification
import torch
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from preprocessing.Data import Data





# init data
data = Data()
data.init_data_train()
x_train, y_train, x_val, y_val, w = data.get_data()
size_of_train = len(x_train)
size_of_val = len(x_val)
num_language= data.get_num_language()

ROOT_PATH = '/content/drive/MyDrive/Spoken-language-identification/'
# ROOT_PATH = ''
# DATA_3_LANG_PATH = ROOT_PATH+'data/pickles/db_3_langs.pkl'
ALL_DATA_PATH = ROOT_PATH + 'data/pickles/total_big.pkl'
TRAINING_RESULTS_PATH = ROOT_PATH + 'results/'

print('Start training:')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ConvNet(num_languag).to(device)
model = Convolutional_Language_Identification(num_languag).to(device)
# Convolutional_Language_Identification

# ******************************LOAD MODEL**************************************
# TRAINED_MODEL_PATH = '/content/drive/MyDrive/Spoken-language-identification/results/14-04-2022_23-24-06/ConvNet_dialect_w-epoch_10.pth'
# model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=torch.device('cpu')))
# ַ *******************************************************************************


# setting model's parameters
learning_rate = model.get_learning_rate()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss(weight=w.to(device))

epoch = model.get_epochs()
batch_size = model.get_batch_size()

# preparing txt report file
timezone = pytz.timezone("Israel")
now_time = datetime.now(timezone).strftime("%d-%m-%Y_%H-%M-%S")
dir_of_res_path = TRAINING_RESULTS_PATH + now_time  # same to loger
if not os.path.isdir(dir_of_res_path):
    os.makedirs(dir_of_res_path)

res_file = open(dir_of_res_path + '/reuslts.txt', 'w')  # same to loger
file_txt = ['Date and time :  ' + now_time, 'Learning Rate : ' + str(learning_rate), 'Batch Size : ' + str(batch_size),
            'Epoch Number : ' + str(epoch)]
for s in file_txt:
    res_file.write('\r----------------------------\r\r')
    res_file.write(s)

results_df = pd.DataFrame([], columns=['train_loss', 'val_loss'])

# all epoch run on all data of train
model.train()
for e in range(1, epoch + 1):
    train_loss, val_loss = 0, 0
    count_train = 0

    train_epoch_idx = np.random.permutation(size_of_train)
    batch_num = int(np.ceil(size_of_train / batch_size))
    for b in tqdm(range(batch_num)):
        optimizer.zero_grad()
        batch_loc = train_epoch_idx[(b * batch_size):((b + 1) * batch_size)]
        x_batch = x_train[batch_loc]
        y_batch = y_train[batch_loc]
        y_pred = model(x_batch.to(device))
        loss = criterion(y_pred, y_batch.long().to(device))
        train_loss += loss.item()
        # ? all sample
        loss.backward()
        optimizer.step()
        count_train += 1

    train_loss = np.round(train_loss / count_train, 4)

    # checking the model's performances per epoch
    with torch.no_grad():
        model.eval()
        count_val = 0
        val_epoch_idx = np.random.permutation(size_of_val)
        for b in range(int(size_of_val / batch_size) - 1):
            val_batch_loc = val_epoch_idx[(b * batch_size): ((b + 1) * batch_size)]
            mini_x_val, mini_y_val = x_val[val_batch_loc], y_val[val_batch_loc]
            y_pred_val = model(mini_x_val.to(device))
            val_loss += criterion(y_pred_val, mini_y_val.long().to(device)).item()
            count_val += 1
        val_loss = np.round(val_loss / count_val, 4)

    results_df.loc[len(results_df)] = [train_loss, val_loss]
    if 0 == (e % 5):
        torch.save(model.state_dict(), dir_of_res_path + "/" + model.to_string() + str(e) + ".pth")
        results_df.to_excel(dir_of_res_path + '/final_report.xlsx')

    plt.figure(figsize=(10, 10))
    plt.plot(results_df['train_loss'], color='gold', label='train')
    plt.plot(results_df['val_loss'], color='purple', label='val')
    plt.ylabel('Loss', fontsize=25)
    plt.xlabel('Epoch', fontsize=25)
    plt.legend()
    plt.savefig(dir_of_res_path + f'/final_loss_plot_.jpeg')

    print('Epoch: ', e, ' -', dict(results_df.iloc[-1]))

    epoch_report = "Epoch : " + str(e) + " | Train_Loss: " + str(train_loss) + ' , Val_Loss: ' + str(val_loss)
    res_file.write('\r----------------------------\r\r')
    res_file.write(epoch_report)

res_file.close()


plt.figure(figsize=(10, 10))
plt.plot(results_df['train_loss'], color='gold', label='train')
plt.plot(results_df['val_loss'], color='purple', label='val')
plt.ylabel('Loss', fontsize=25)
plt.xlabel('Epoch', fontsize=25)
plt.legend()
plt.savefig(dir_of_res_path + '/final_loss_plot.jpeg')




