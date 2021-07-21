import PIL
import cv2
import numpy as np
from numpy.lib.function_base import interp
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
from PIL import ImageFont, ImageDraw, Image
import torch
import torchvision
import torch.nn as nn
import numpy as np
import cv2
from matplotlib import pyplot as plt
from torchvision import datasets, models, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# This is the Label
Labels = {0: 'rock',
          1: 'scissors',
          2: 'paper'
          }

r_img = cv2.imread('rock.jpg')
p_img = cv2.imread('paper.jpg')
s_img = cv2.imread('scissors.jpg')

# Let's preprocess the inputted frame
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transforms = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
)



#########################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
model_loader_path ="./env_test/ai_torch3//RockScissorPaper_vgg.pt"
model_ft = (torch.load(model_loader_path))
model = model_ft.to(device)  # set where to run the model and matrix calculation
model.eval()  # set the device to eval() mode for testing
print("일단 여기까지 하면 모델 불러오기 성공!!!???????")
#########################################################################



# image preprocessing
def preprocess(image):
    image = PIL.Image.fromarray(image)
    image = data_transforms(image)
    image = image.float()
    image = image.unsqueeze(0)
    return image

def argmax(prediction):
    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()
    top_predict_class = np.argmax(prediction, axis=1)
    score = np.amax(prediction)
    score = '{:6f}'.format(score)
    prediction = top_predict_class[0]
    result = Labels[prediction]

    return result, score

webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not webcam.isOpened():
    print("Could not open webcam")

while webcam.isOpened():

    status, frame = webcam.read()
    if not status :
        break

    image = cv2.resize(frame, (224,224), interpolation=cv2.INTER_AREA)
    image_data = preprocess(image)
    # print("image data :", image_data)
    # image_data = torch.FloatTensor(image_data)
    # print("image data 텐서화 후:", image_data)
    # print(image_data)
    
    # image_data = image_data.unsqueeze()
    prediction = model(image_data)
    

    result, score = argmax(prediction)

     #display
    fontpath = "font/gulim.ttc"
    font = ImageFont.truetype(fontpath, 100)
    frame_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame_pil)
    draw.text((50, 50), result, font=font, fill=(0, 0, 255, 3))
    frame = np.array(frame_pil)
    cv2.imshow('RPS', frame)

    if cv2.waitKey(1) == ord('q'): break

webcam.release()
cv2.destroyAllWindows()

