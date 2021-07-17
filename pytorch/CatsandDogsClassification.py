# 라이브러리 추가 
import torch 
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import cv2
# file loader 
import glob
import os 

from torchvision import datasets, models, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# 1. 맨처음으로 device 설정을 하여 GPU 사용 가능 여부에 따라 device 정보를 저장해야한다.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

#2. data location 학습을 데이터가 어디에 있는지 데이터 경로를 저장한다.
data_dir = ".\cat_dog_data"
#3. hyper parameter 
batch_size =32
num_epochs=1
learning_rate =0.01


class CatsandDogsDataset(Dataset):
    def __init__(self, data_dir, mode, transform=None):
        self.all_data=sorted(glob.glob(os.path.join(data_dir, mode,"*","*")))
        self.transform = transform
    def __getitem__(self, index):
        data_path = self.all_data[index]
        img = Image.open(data_path)
        img = img.convert("RGB") 
        label_name = os.path.basename(data_path) 
        label_str = str(label_name)

        if label_str.startswith("cat") == True:
            label = 0 
        else :
            label =1
        if self.transform is not None :
            img =self.transform(img)
        return img, label

    def __len__(self):
        length = len(self.all_data)
        return length

        # data transforms
# image size 224 224
# image net ->

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

data_transforms ={
    'train' : transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)

    ]),
    'val' : transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std) 

    ]),
    'test' : transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)                         
    ])
}

# data정의 data loader
train_data_set = CatsandDogsDataset(data_dir, mode="train", transform=data_transforms['train'])
val_data_set = CatsandDogsDataset(data_dir, mode='val', transform=data_transforms['val'])
test_data_set = CatsandDogsDataset(data_dir, mode='test', transform=data_transforms['test'])

#data loader
train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, drop_last=True)
val_data_loader = DataLoader(val_data_set, batch_size=batch_size, shuffle=False, drop_last=True)
test_data_loader = DataLoader(test_data_set, batch_size=1, shuffle=False, drop_last=True)

# print(train_data_set.__getitem__(0))

# Vgg
net = models.vgg11(pretrained=True).to(device)
#어떤 모델을 쓸것인지 고르고
net.classifier[6] = nn.Linear(4096,2)
# 그모델의 classifier속성을 우리가 원하는 크기로 맞추고 난후
model = net.to(device)
# 편의대로 조작한 모델을 model에 저장한다.

# loss function, opitmizer scheduler
criterion = nn.CrossEntropyLoss().to(device)
optimizer =optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# train함수를 정의하자
def train(num_epochs, model, data_loader, criterion, optimizer, save_dir, val_every, device):
    print("Training has been started")
    best_loss = 9999

    for epoch in range(num_epochs):
        for i, (imgs, labels) in enumerate(data_loader):
            images, labels = imgs.to(device), labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, argmax = torch.max(outputs, 1)

            accuracy = (labels == argmax).float().mean()

            if (i + 1) % 3 == 0 :
                print("Epoch [{}/{}], step{}/{}, loss:{:4f}, Accuracy {:.2f}%"
                .format(epoch+1, num_epochs, i+1, len(train_data_loader), loss.item(), accuracy.item()))
        if (epoch+1)% val_every ==0:
            average_loss = validation(epoch+1, model, val_data_loader, criterion, device)

            if average_loss < best_loss :
                print("Best performance at epoch :{}".format(epoch+1))
                print("Save model in", save_dir)
                best_loss =average_loss
                save_model(model, save_dir)


def validation(epoch, model, data_loader, criterion, device):
    print("Validation has been started")
    model.eval()

    with torch.no_grad():
        # 초기화 값
        total =0
        correct =0
        total_loss =0
        count =0

        for i, (imgs, labels) in enumerate(data_loader):
            images, labels = imgs.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total += imgs.size(0) # 이미지의 사이즈에 첫번째에는 loss가 반환되어있음
            _, argmax = torch.max(outputs, 1)
            accuracy = (labels == argmax).float().mean()
            correct += (labels ==argmax).sum().item()
            total_loss += loss
            count+=1
        
        average_loss = total_loss /count
        print("Validation #{} Accuracy {:.2f * 100} % Average Loss : {:.4f}"
        .format(epoch, accuracy, average_loss))

    model.train() # train코드 중간에 valiation 코드가 주기적으로 한번씩 들어가 있는 것이므로 
    # train모드 상에서  valiation 함수가 호출되고 난 후에 다시 train모드로 변경해주어야 한다.

    return average_loss

def save_model(model, save_dir, file_name="best_model_vgg_2.pt"):
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), output_path)

    # train start 
val_every = 1 
save_dir = "./ai_torch2"

train(num_epochs, model, train_data_loader, criterion, optimizer, save_dir, val_every, device)