import cv2
import numpy as np
import json
import os

img =cv2.imread("dog.jpg")
c_img = cv2.imread("cat.jpg")

print(img.shape)

# 이미지저장

# v2.imwrite("copy, img.jpg", img)

# 이미지보기
# cv2.imshow("dog", img)
# cv2.waitKey()  #없으면 바로 켰다 꺼짐
#
# #그레이로 변환 컨버터
# rgb_img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# gray_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# cv2.imshow(" ", rgb_img)
# cv2.imshow("", gray_img)
# cv2.waitKey()

# 색으로 보기
# (B,G,R)=cv2.split(img)
# color =R
# cv2.imshow("", color)
# cv2.waitKey()
#
#
# zeros = np.zeros(img.shape[:2], dtype='uint8')
# print(zeros.shape)
# cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
# cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
# cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))
# cv2.waitKey(0)

 #픽셀 값 접근
# print(img[100,200])
# cv2.imshow("", img)
# cv2.waitKey()
# cv2.destroyAllWindows()
#
# cv2.imshow("", img)
# img=cv2.resize(img, (400,300))
# cv2.imshow("big", img)
# img=cv2.resize(img, (100,50))
# cv2.imshow("small", img)
# cv2.waitKey()

# cv2.imshow("", img[0:150, 0:100])
#
#
# cv2.imshow("change", img[100:150, 50:100])
#
# h,w,c =img.shape
# cv2.imshow("crop", img[int(h/2 -50):int(h/2 +50), int(w/2 -50):int(w/2 +50)])
# print(int(h/2 -50),int(h/2 +50), int(w/2 -50),int(w/2 +50))
# cv2.waitKey()

#도형 그리기
#Line 그리기
#
# img = cv2.line(img, (100,100), (180,150), (0,0,200),1)  #img = cv2.line(img, (100,100)이것이 첫번째 점, (180,150) 이것이 두번째 점 , (0,255,0) 이것은 선색깔 ,4 이것은 선 굵기)
# cv2.imshow("", img)
# cv2.waitKey()

#rectangle
img = cv2.rectangle(img, (80,50), (100,170), (0,255,0),3)
cv2.imshow("",img)

cv2.waitKey()

#circle
# img = cv2.circle(img, (200,100), 30, (0,255,0),3) #img = cv2.circle(img, (200,100)원의 중심, 30 반지름, (0,255,0), 3)
# cv2.imshow("", img)
# cv2.waitKey()


#polygon
# pts = np.array([[35,26], [35,170],[160,170], [190,26]])
# img = cv2.polylines(img, [pts], True, (0,255,0),3) #img = cv2.polylines(img, [pts](좌표 반시계방향순), True(닫힌 도형으로 선을 긋겠다), (0,255,0),3)
# cv2.imshow("", img)
# cv2.waitKey()

#text
# img = cv2.putText(img, "dog", (200,100), 0,1, (0,255,0),2 )
# cv2.imshow("", img)
# cv2.waitKey()

# 이미지 붙여넣기
# img = cv2.rectangle(img, (200,100), (275,183), (0,255,0),2)
#c_img=cv2.resize(c_img, (75,83))
# img[100:183, 200:275]=c_img
# cv2.imshow("change", img)
# cv2.waitKey()

# 이미지 더하기
# img = cv2.resize(img, (217,232))
# add1= img +c_img
# add2= cv2.addWeighted(img, float(0.5), c_img, float(0.5),5)
# cv2.imshow("1", add1)
# cv2.imshow("2", add2)
# cv2.waitKey()

# 이미지 회전
# height, width, c = img.shape
# img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
# img270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
# img180 = cv2.rotate(img, cv2.ROTATE_180)
# img_r = cv2.getRotationMatrix2D((width/2, height/2), 45,10) #img_r = cv2.getRotationMatrix2D((width/2, height/2) 이것은 회전할때의 중심점 , 45 이것은 몇도 돌릴것이냐,1이미지 사이즈를 얼마나 확대할 것이냐)
#
# cv2.imshow('90', img90)
# cv2.imshow('270',img270)
# cv2.imshow('180', img180)
# cv2.imshow('45', img_r)
# cv2.waitKey()

#이미지 반전
# img = cv2.flip(img, 0)
# cv2.imshow('270', img)
# cv2.waitKey()

#이미지 아핀
# height, width, channel = img.shape
# matrix = cv2.getRotationMatrix2D((width/2, height/2), 45 , 0.5) #matrix = cv2.getRotationMatrix2D((width/2, height/2) 회전 중심점 , 45 회전각도, 0.5)
# img = cv2.warpAffine(img, matrix, (width, height))
# cv2.imshow('270', img)
# cv2.waitKey()

#이미지 밝기, 감마
# nimg = cv2.imread("night.jpg")
# table = np.array([((i/255.0)**0.8)*255 for i in np.arange(0,256)]).astype("uint8")  #table = np.array([((i/255.0)**0.5 이것이 감마값 가중치)*255 for i in np.arange(0,256)]).astype("uint8")
# # 위의 테이블 값에서 가중치를 주고 있는 것이다.
# print('table', table)
# gamma_img = cv2.LUT(nimg, table)
# val = 50 #randint(10,50
#
# array = np.full(nimg.shape, (val, val, val), dtype=np.uint8)
# print(array)
# print(array.shape)
# all_array = np.full(nimg.shape, (30,30,30), dtype=np.uint8)
# print(all_array)
# print(all_array.shape)
# bright_img=cv2.add(nimg, array).astype("uint8")
# all_img = cv2.add(gamma_img, all_array).astype("uint8")
#
# cv2.imshow('all', all_img)
# cv2.imshow('bright', bright_img)
# cv2.imshow('gamma', gamma_img)
# cv2.waitKey()

# 이미지 블러링
blu_img= cv2.blur(img, (10,10)) #이렇게 하면 img의 전체가 다 블러 처리 됨
roi = img[28:74, 95:165] # 이미지의 일부분을 roi로 저장함
roi =cv2.blur(roi, (15,15)) #저장한 이미지 roi(부분)를 블러처리함
img[28:74, 95:165]=roi #원래 이미지의 그 부분만 roi로 대체시킴
cv2.imshow('blu', blu_img)
cv2.imshow('s_blu', img)
cv2.waitKey()

# 이미지 패딩
img_pad = cv2.cv2.copyMakeBorder(img, 50,50,100,100,cv2.BORDER_CONSTANT, value=[10,10,0]) #img_pad = cv2.cv2.copyMakeBorder(img, 50,50,100,100, 순서대로 top bottom left right cv2.BORDER_CONSTANT 이 이미지를 중심에 놓겠다는 뜻, value=[0,0,0])
cv2.imshow("img_padd", img_pad)
cv2.waitKey()



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('face1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3,1)
print("img", img.shape)
print("face location", faces)
cv2.imshow("face", img)
cv2.waitKey()


