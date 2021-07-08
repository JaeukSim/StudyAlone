import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# 문제 iris데이터를 뽑아서 Train set과 Test set으로 나누고 데이터를 학습하고 평가

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64,128)
        self.fc3 = nn.Linear(128,3)
        self.relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(3, 3)
        
        self.sig = nn.Sigmoid()
        

    def forward(self, x):
        x = self.fc1(x)   
        x = self.relu(x)     
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sig(x)
        
        # x = self.fc2(x)
        # x = self.sig(x)
        
        return x


class CustomDataset(Dataset):
    def __init__(self, data1, data2, transforms=None):
        self.x = data1
        self.y = data2
        # print(self.x) 
        # print(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        x = np.array(x)
        y = np.array(y)
        return x, y
# 전공이 수학이다 보니 그냥 loss값이 줄어드는 걸 보고 "아 좋다ㅎㅎ" 이렇게 되는게 아니라 작동원리가 어떻길래 왜 loss가 줄어드는지 알지 못하면 매우 답답해서 ㅋㅋㅋㅋ
# iris데이터를 뽑아서 Train set과 Test set으로 나누기 
# # Train 데이터를 학습하고 
iris = load_iris()
iris_data=iris.data
iris_target= iris.target
x_train, x_test, x_traintarget, x_testtarget = train_test_split(iris_data, iris_target, test_size=0.25, shuffle = True, stratify = iris_target, random_state=100)
#print(x_train) #Data 의 개수 112
#print("\n")
#print(x_traintarget) 

# x_train : feature 값의 75% 가 할당됨, x_test : feature 값의 25% 가 할당됨,x_traintarget : x_train에 할당된 75%에 해당하는 target값이 할당됨, x_testtarget : x_test에 할당된 25%에 해당하는 target 값이 할당됨
train_dataset = CustomDataset(x_train, x_traintarget, transforms=None) 
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  
model = Net().to(device) 
# criterion = nn.MSELoss() 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1) #주어진 learning  rate로 1000스텝을 지날때 마다 learning rate에 gamma값을 곱해줌(보폭을 줄임)
epoch = 10  # 학습을 몇번 돌릴건지 결정
total_loss = 0  
model.train() 
        
for i in range(epoch):
    for x, y in train_loader:
        x = x.float().to(device)
        y = y.long().to(device)
                     
        outputs = model(x)  # model x를 호출함과 동시에 Net의 forward가 실행되면서 변화가 일어남
        # print(outputs)

        print("outputs : ", outputs)
        print("y:  ", y)
        

        loss = criterion(outputs, y)
        print(loss)
        exit()

        # outputs = outputs.detach().numpy()
        optimizer.zero_grad()  # 기울기 초기화
        loss.backward()  # 가중치와 편향에 대해 기울기 계산
        total_loss += loss.item() #loss값이 텐서값이 나오므로 그 안의 숫자만 받겠다는 뜻
        outputs = outputs.detach().numpy()
        y = y.numpy()
        # print(outputs)
        # print(y)
    if i % 2 == 0:
        torch.save(model.state_dict(), f"model_last.pth")
        print(f"epoch -> {i}      loss -- > ", total_loss / len(train_loader))
    
    optimizer.step()
    total_loss = 0

model.eval()  
model.load_state_dict(torch.load('model_last.pth'))


test_dataset = CustomDataset(x_test,x_testtarget, transforms=None)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
for x, y in test_loader:
    x = x.float().to(device)
    y = y.long().to(device)
    # print(x, y)
    outputs = model(x)
    print(outputs, y)