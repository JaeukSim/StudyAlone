import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import torchvision

from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt

import time
import os
import copy

plt.ion() # 대화형 모드 

# 데이터를 불러오기
# 학습을 위해 데이터 증가(augmentation) 및 일반화(normalize)
data_transforms = {
    'train': transforms.Compose([
        # aum 정의 
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        
        # 마지막 텐서 정규화 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.299,0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        
        # 마지막 텐서 정규화
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.299, 0.224, 0.225])
    ])
} 



data_path= "./env_test/ai_torch2/hymenoptera_data"

image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x), data_transforms[x]) for x in ['train','val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 4, 
                    shuffle=True, num_workers = 0) for x in ['train' ,'val']}
dataset_size = {x : len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes


# device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    inp = std*inp+mean
    inp = np.clip(inp,0,1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

inputs, classes = next(iter(dataloaders['train']))
# print(inputs)
print(classes)
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])
# train looop
def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()

    best_model_status=copy.deepcopy(model.state_dict())
    best_accuracy =.0

    # train loop
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-------------------------------------------")

        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else :
                model.eval()
            running_loss=.0
            running_corrects =.0

            for inputs, classes in dataloaders[phase]:
                inputs = inputs.to(device)
                classes = classes.to(device)
                
                # optimizer 0으로 설정
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs,1 )
                    loss = criterion(outputs, classes)
                    # 지급 학습모드인 경우에만 진행되고 있는 code
                    if phase =='train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == classes.data)

                epoch_loss = running_loss / dataset_size[phase]
                epoch_accuracy = running_corrects.float() / dataset_size[phase]

                print("{} loss : {:.4f} acc : {:.4f}". format(phase, epoch_loss, epoch_accuracy))

                if phase == 'val' and epoch_accuracy > best_accuracy:
                    best_accuracy = epoch_accuracy
                    best_model_status = copy.deepcopy(model.state_dict())
            
            time_elapsed = time.time() - since
            print("Training complete in {:.0f}m {:.0f}s"
                  .format(time_elapsed // 60, time_elapsed % 60))
            
            print("Best val ACC : {:4f}".format(best_accuracy))

            # 가장 좋은 모델 가중치를 불러옴
            model.load_state_dict(best_model_status)

            return model


                    
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far=0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            classes = classes.to(device)
            

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far +=1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title("predicted: {}".format(class_names[preds[j]]))

                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode = was_training)
                    return
        model.train(mode=was_training)


model_ft = models.resnet18(pretrained=True)
# print(model_ft)

num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, 2)
# print(model_ft.fc)

model_ft = model_ft.to(device)

#loss
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer_ft = optim.SGD(model_ft.parameters(), lr = 0.001, momentum = 0.9)

#scheduler 10 epoch -> 0.1 0.01-> 0.001 -> 0.0001
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)   

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=1)

visualize_model(model_ft)