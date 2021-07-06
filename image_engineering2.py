import cv2
import numpy as np
import json
import os
###################### 1. �� ã�Ƽ� �� ������ �ڽ��׸��� 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('face.jpg')
print("img", img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # �ϴ� �޾ƿ� ������ ������� ��ȯ
faces = face_cascade.detectMultiScale(gray, 1.3,1)
print(faces)
# �� ���� ��ġ
# [  5  20  69  69] ���ʴ�� �� �׵θ� �簢���� ���� �� �������� x��ǥ y��ǥ ���α���, ���α���
#  [163  25  62  62]
#  [ 87  30  60  60]
#  [163 133  64  64]
#  [  9 135  62  62]
#  [ 86 136  61  61
for face in faces:
    img= cv2.rectangle(img, (face[0], face[1]), (face[0]+face[2], face[1]+face[3]), (0,255,0),3 )
    
cv2.imshow("", img)
cv2.waitKey()


######################## 2. ���� ���� 500,500 ���� �е��� �ϰ� ���� ��ġ�� ���̽����� �����Ѵ�
import cv2
import numpy as np
import json
import os
img =cv2.imread("test_img3.jpg")
img_pad = cv2.cv2.copyMakeBorder(img, 100,100,80,80,cv2.BORDER_CONSTANT, value=[10,10,0])
# ������ img_pad = cv2.cv2.copyMakeBorder(img, 50,50,100,100, ������� top bottom left right cv2.BORDER_CONSTANT �� �̹����� �߽ɿ� ���ڴٴ� ��, value=[0,0,0])
# cv2.imshow("padding", img_pad)
# cv2.waitKey()
gray = cv2.cvtColor(img_pad, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces_location = face_cascade.detectMultiScale(gray, 1.3,1)
# print(faces_location)
#  ������ ���� ��ġ
# [[342 208  26  26]
#  [193 276  29  29]
#  [284 278  28  28]
#  [146 220  27  27]]
faces_location_list=[]
for i in range(len(faces_location)):
    dictionary={}
    dictionary[f"person{i}"]=i
    dictionary["box"]=faces_location[i].tolist()
    faces_location_list.append(dictionary)
    facelist=faces_location[i].tolist()
    # print("facelist\n", facelist)
    cv2.rectangle(img_pad, (facelist[0],facelist[1]), (facelist[0]+facelist[2], facelist[1]+facelist[3]), (0,255,0), 3)

cv2.imshow(" ",img_pad)
cv2.waitKey()

with open('location.json','w', encoding='utf-8') as f:
    json.dump(faces_location_list, f, indent='\t')

######## �ǽ�3 Json���Ͽ� �ִ� �������� ������ �ͼ� �� ������ �������� ������ �󱼵��� �� ó���ϱ�
import json
import cv2
import numpy as np
import json
import os
json_dir = 'location.json'
print(os.path.isfile(json_dir))
with open(json_dir) as f:
    json_data = json.load(f)

img = cv2.imread("test_img3.jpg")
img = cv2.cv2.copyMakeBorder(img, 100, 100, 80, 80, cv2.BORDER_CONSTANT, value=[0, 0, 0])


for j_data in json_data:
    target= img[(j_data["box"][1]):(j_data["box"][1]+j_data["box"][3]), (j_data["box"][0]):(j_data["box"][0]+j_data["box"][2])]
    target = cv2.blur(target, (10, 10))
    img[(j_data["box"][1]):(j_data["box"][1]+j_data["box"][3]), (j_data["box"][0]):(j_data["box"][0]+j_data["box"][2])]=target

cv2.imshow('blur', img)
cv2.waitKey()


