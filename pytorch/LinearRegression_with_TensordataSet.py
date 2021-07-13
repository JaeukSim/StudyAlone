# DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# TensorDataset and DataLoader
from torch.utils.data import TensorDataset # 텐서 데이터 셋
from torch.utils.data import DataLoader # 데이터 로더

# TensorDataset은 기본적으로 텐서를 입력받습니다. 텐서 형태로 데이터를 정의

x_train = torch.FloatTensor([
    [73, 80, 75],
    [93, 88, 93],
    [89, 91, 90],
    [96, 98, 100],
    [73, 66, 70]
    ])

y_train = torch.FloatTensor([[152],[185],[180],[196], [142]])

# TensorDataset의 입력으로 사용하고 dataset을 지정합니다.
dataset = TensorDataset(x_train, y_train)

# dataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 모델과 옵티마이저 설계
model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

epoch_nb=500
for epoch in range(epoch_nb+1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples

        # H(X) 계산
        prediction = model(x_train)
        # loss
        loss = F.mse_loss(prediction, y_train)

        # loss H(x) 3가지 짝궁
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 ==0 :
            print("Epoch {:4d}/{} Batch {}/{} Loss : {:.6f}".format(epoch, epoch_nb, batch_idx+1, len(dataloader), loss.item()))

# 모델의 입력으로 임의의 값을 주고 예측값을 확인
# 임의의 값
test_val = torch.FloatTensor([[96, 98, 100]])
pred_y = model(test_val)
print("훈련 후 입력이 73, 80, 75 일때 예측 값 : ", pred_y)
print(1e-5)