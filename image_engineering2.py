import cv2
import numpy as np
import json
import os
###################### 1. 얼굴 찾아서 그 주위에 박스그리기 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('face.jpg')
print("img", img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 일단 받아온 사진을 흑백으로 변환
faces = face_cascade.detectMultiScale(gray, 1.3,1)
print(faces)
# 얼굴 각각 위치
# [  5  20  69  69] 차례대로 얼굴 테두리 사각형의 왼쪽 위 꼭지점의 x좌표 y좌표 가로길이, 세로길이
#  [163  25  62  62]
#  [ 87  30  60  60]
#  [163 133  64  64]
#  [  9 135  62  62]
#  [ 86 136  61  61
for face in faces:
    img= cv2.rectangle(img, (face[0], face[1]), (face[0]+face[2], face[1]+face[3]), (0,255,0),3 )
    
cv2.imshow("", img)
cv2.waitKey()


######################## 2. 가로 세로 500,500 으로 패딩을 하고 얼굴의 위치를 제이슨으로 저장한다
import cv2
import numpy as np
import json
import os
img =cv2.imread("test_img3.jpg")
img_pad = cv2.cv2.copyMakeBorder(img, 100,100,80,80,cv2.BORDER_CONSTANT, value=[10,10,0])
# 위에서 img_pad = cv2.cv2.copyMakeBorder(img, 50,50,100,100, 순서대로 top bottom left right cv2.BORDER_CONSTANT 이 이미지를 중심에 놓겠다는 뜻, value=[0,0,0])
# cv2.imshow("padding", img_pad)
# cv2.waitKey()
gray = cv2.cvtColor(img_pad, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces_location = face_cascade.detectMultiScale(gray, 1.3,1)
# print(faces_location)
#  각각의 얼굴의 위치
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

######## 실습3 Json파일에 있는 정보들을 가지고 와서 그 정보를 바탕으로 사진속 얼굴들을 블러 처리하기
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


