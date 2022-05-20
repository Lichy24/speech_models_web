"""******************************************************************
The code is based on : https://github.com/a-nagrani/VGGVox/issues/1
******************************************************************"""

from torch import nn

DROP_OUT = 0.5


class Convolutional_Neural_Network_Age(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_2d_1 = nn.Conv2d(1, 96, kernel_size=(3, 3), padding=1)
        self.bn_1 = nn.BatchNorm2d(96)
        self.max_pool_2d_1 = nn.MaxPool2d(kernel_size=(3, 3))

        self.conv_2d_2 = nn.Conv2d(96, 256, kernel_size=(3, 3), padding=1)
        self.bn_2 = nn.BatchNorm2d(256)
        self.max_pool_2d_2 = nn.MaxPool2d(kernel_size=(3, 3))

        self.conv_2d_3 = nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1)
        self.bn_3 = nn.BatchNorm2d(384)

        self.conv_2d_4 = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1)
        self.bn_4 = nn.BatchNorm2d(256)

        self.conv_2d_5 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.bn_5 = nn.BatchNorm2d(256)
        self.max_pool_2d_3 = nn.MaxPool2d(kernel_size=(5, 3), stride=(3, 2))

        self.conv_2d_6 = nn.Conv2d(256, 4096, kernel_size=(4, 1), padding=0)
        self.drop_1 = nn.Dropout(p=DROP_OUT)

        self.global_avg_pooling_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.dense_1 = nn.Linear(4096, 1024)
        self.drop_2 = nn.Dropout(p=DROP_OUT)

        self.dense_2 = nn.Linear(1024, 2)  # TEEN / ADULT

    def forward(self, X):
        x = nn.ReLU()(self.conv_2d_1(X))
        x = self.bn_1(x)
        x = self.max_pool_2d_1(x)

        x = nn.ReLU()(self.conv_2d_2(x))
        x = self.bn_2(x)
        x = self.max_pool_2d_2(x)

        x = nn.ReLU()(self.conv_2d_3(x))
        x = self.bn_3(x)

        x = nn.ReLU()(self.conv_2d_4(x))
        x = self.bn_4(x)

        x = nn.ReLU()(self.conv_2d_5(x))
        x = self.bn_5(x)
        x = self.max_pool_2d_3(x)

        x = nn.ReLU()(self.conv_2d_6(x))
        x = self.drop_1(x)
        x = self.global_avg_pooling_2d(x)

        x = x.view(-1, x.shape[1])  # output channel for flatten before entering the dense layer
        x = nn.ReLU()(self.dense_1(x))
        x = self.drop_2(x)

        x = self.dense_2(x)
        y = nn.LogSoftmax(dim=1)(x)  # consider using Log-Softmax

        return y

    def get_epochs(self):
        return 30  # 15

    def get_learning_rate(self):
        return 0.0001

    def get_batch_size(self):
        return 50

    def to_string(self):
        return "Binary_Model-e_"
