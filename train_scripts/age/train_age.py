from datetime import datetime
import time
import torch
from sklearn.utils import class_weight
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from Binary_Age_Model.model_definition_age import Convolutional_Neural_Network_Age
from Binary_Age_Model.preprocessing_age import prepareData


start = time.time()
print('Start')
print('Start training:')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Convolutional_Neural_Network_Age().to(device)
data = prepareData()

dir_path = 'binary_models_age/1.5/CoVoVox_lr0.0001_NM\\    '

sum_op = 2
# setting model's parameters
learning_rate = model.get_learning_rate()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epoch, batch_size = model.get_epochs(), model.get_batch_size()

# preparing txt report file
results = pd.DataFrame([], columns=['train_loss', 'val_loss', 'test_loss', 'Accuracy of the network',
                                    'f1_score', 'f1_score_avg'])
file = open(dir_path + '/reuslts.txt', 'w')
file_txt = ['Date and time :  ' + datetime.now().strftime("%d-%m-%Y_%H-%M-%S"),
            'Learning Rate : ' + str(learning_rate),
            'Epoch Number : ' + str(epoch)]
for s in file_txt:
    file.write(s)
    file.write('\r----------------------------\r\r')


class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(data.y), y=data.y)
class_weights = torch.tensor(class_weights, dtype=torch.float)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# to plot each the loss of each epoch
results_df = pd.DataFrame([], columns=['train_loss', 'val_loss', 'test_loss'])

for e in range(epoch):
    print("\nepoch - ", e)

    model.train()
    train_loss, val_loss, test_loss = 0, 0, 0
    count_train = 0
    train_size = 0
    test_size = 0

    for tensor, label in data.train_loader:

        epoch_size = len(tensor)
        train_epoch_idx = np.random.choice(len(label), epoch_size, replace=False)
        np.random.shuffle(train_epoch_idx)
        batch_num = int(np.ceil(epoch_size / batch_size))

        for b in tqdm(range(batch_num)):
            optimizer.zero_grad()
            batch_loc = train_epoch_idx[(b * batch_size):((b + 1) * batch_size)]
            x_batch, y_batch = tensor[batch_loc], label[batch_loc]

            y_pred = model(x_batch.to(device))
            loss = criterion(y_pred, y_batch.long().to(device))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            count_train += 1
            train_size += len(label)

    train_loss = np.round(train_loss / count_train, 4)
    print("\ntrain_size - ", train_size)
    print("\ntrain loss - ", train_loss)

    # checking the model's performances per epoch
    with torch.no_grad():

        # Validation
        val = data.val_loader
        count_val = 0
        for tensor, label in data.val_loader:
            val_epoch_idx = np.random.choice(len(label), len(label), replace=False)
            for h in range(int(np.ceil(len(label) / batch_size))):
                val_batch_loc = val_epoch_idx[(h * batch_size): ((h + 1) * batch_size)]
                mini_x_val, mini_y_val = tensor[val_batch_loc], label[val_batch_loc]
                y_pred_val = model(mini_x_val.to(device))
                val_loss += criterion(y_pred_val, mini_y_val.long().to(device)).item()
                count_val += 1
        val_loss = np.round(val_loss / count_val, 4)
        print("\nval loss - ", val_loss)

        # Testing

        count_test = 0
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(sum_op)]
        n_class_samples = [0 for i in range(sum_op)]

        y_pred = []
        y_true = []

        for embedding, labels in data.test_loader:
            embedding = embedding.to(device)

            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            outputs = model(embedding)

            # test loss
            test_loss += criterion(outputs, labels.long()).item()
            count_test += 1

            # euler FIX
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                _label = labels[i]
                pred = predicted[i]
                if _label == pred:
                    n_class_correct[_label] += 1
                n_class_samples[_label] += 1

            y_pred.extend(predicted.data.cpu().numpy())
            y_true.extend(labels.data.cpu().numpy())

        # f1 score
        f1 = f1_score(y_true, y_pred, average=None)
        f1_avg = f1_score(y_true, y_pred, average='weighted')
        print("f-score: ", f1, "  f-score avg: ", f1_avg, "\n")

        test_loss = np.round(test_loss / count_test, 4)
        print("\ntest loss - ", test_loss)

        results_df.loc[len(results_df)] = [train_loss, val_loss, test_loss]

        acc = 100.0 * n_correct / n_samples
        print(f'\nAccuracy of the network: {acc} %')

        # calculating accuracy
        for i in range(sum_op):
            if n_class_correct[i] == 0:
                acc_ = 0
            else:
                acc_ = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {i + 1}: {acc_} %')
        print("\n")

        # saving model and creating confusion matrix
        torch.save(model, dir_path + "/" + "age_" + model.to_string() + str(e) + "_Weights.pth")

        # Build confusion matrix
        classes = ['Teen', 'Adult']  # data.ages.keys()
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes], columns=[i for i in classes])
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig(dir_path + "/" + 'confusion_matrix' + "_age_" + model.to_string() + str(e) + "_Weights.png")

    results.loc[len(results)] = [train_loss, val_loss, test_loss] + [acc] + [f1, f1_avg]

file.write(results.to_string())


print("End")
end = time.time()
print("Total time = ", end - start)


# Loss plot
plt.figure(figsize=(10, 10))
plt.plot(results_df['train_loss'], color='gold', label='train')
plt.plot(results_df['val_loss'], color='purple', label='val')
plt.plot(results_df['test_loss'], color='blue', label='test')

plt.ylabel('Loss', fontsize=25)
plt.xlabel('Epoch', fontsize=25)
plt.legend()
plt.savefig(dir_path + '/final_loss_plot.jpeg')
print(results_df)
