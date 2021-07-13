
import itertools
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.preprocessing import StandardScaler, RobustScaler
import os 

file_path = "./env_test/ai_torch2/dataset"
train_fn = "FordA_TRAIN.arff"
test_fn = "FordA_TEST.arff"

def read_ariff(path):
    raw_data, meta = arff.loadarff(path)
    cols = [x for x in meta]

    data2d = np.zeros([raw_data.shape[0], len(cols)])

    for index, col in zip(range(len(cols)), cols):
        data2d[:,index]=raw_data[col]

    return data2d


train_path = os.path.join(file_path, train_fn)
test_path = os.path.join(file_path, test_fn)
train = read_ariff(train_path)
test = read_ariff(test_path)
print("train >>", len(train))
print("train >>", train)
print("test>>", len(test))
print("test>>", test)

x_train_temp = train[:,:-1] # 마지막 컬럼이 레이블 값이므로 삭제해주는 것임(즉 feature값만 모아둠)
y_train_temp = train[:,-1] # 마지막 컬럼이 레이블 값(즉 정답지를 만든다)
print(y_train_temp)

x_test = test[:,:-1]
y_test = test[:,-1]

print(x_test, y_test)

# 학습용 검증용 테스트용 데이터셋 나누기
normal_x = x_train_temp[y_train_temp==1] # train_x 데이터 중 정상데이터
abnormal_x = x_train_temp[y_train_temp ==-1] # train_x 데이터 중 비정상 데이터

normal_y = y_train_temp[y_train_temp==1] # train_y 데이터 중 정상데이터
abnormal_y= y_train_temp[y_train_temp==-1]# train_y 데이터 중 정상데이터

# 정상데이터 8:2
ind_x_normal = int(normal_x.shape[0]*0.8) # 정상데이터를 8:2로 나누기 위한 인덱스 설정
ind_y_normal = int(normal_y.shape[0]*0.8) # 정상데이터를 8:2로 나누기 위한 인덱스 설정

# 비정상데이터 8:2
ind_x_abnormal = int(abnormal_x.shape[0]*0.8) # 비정상데이터를 8:2로 나누기 위한 인덱스 설정
ind_y_abnormal = int(abnormal_y.shape[0]*0.8) # 비정상데이터를 8:2로 나누기 위한 인덱스 설정
#  print(ind_x_abnormal)

x_train = np.concatenate((normal_x[:ind_x_normal], abnormal_x[:ind_x_abnormal]), axis=0) #80
x_test = np.concatenate((normal_x[ind_x_normal:], abnormal_x[ind_x_abnormal:]), axis=0) #20

x_traintarget = np.concatenate((normal_y[:ind_y_normal], abnormal_y[:ind_y_abnormal]), axis=0) #80
x_testtarget = np.concatenate((normal_y[ind_y_normal:], abnormal_y[ind_y_abnormal:]), axis=0) #20

# 시각화

# class 종류 정상 1 비정상 -1 두개 밖에 없다는 것을 확인
classes = np.unique(np.concatenate((x_traintarget, y_test), axis=0))
print("################################",len(classes))


x = np.arange(len(classes))

lables = ["Abnormal", "Normal"] #plot x축 이름

# train, valid, test 세 세트가 필요함
valuse_train = [(x_traintarget == i).sum() for i in classes]
valuse_valid = [(x_testtarget == i).sum() for i in classes]
valuse_test = [(y_test == i).sum() for i in classes]
# 이 부분의 작동 방식이 잘 이해가 안됩니다. ㅠㅠ classes의 원소는 -1 1 두개밖에 없고, classes의 길이도 2인데
# for 문을 두번 밖에 안 도는데 어떻게 결과값이 1400이 넘게 나오죠?
# 해답 : classes 에는 -1과 1 두개밖에 없는데 i=-1일때 만약 x_traintarget[i] == i(-1) 이면 그 횟수만큼 1씩 더해 나간다.
# 밑에 있는 코드를 실행시키면 이해가 될것이다.
#################################################
# sum=0
# for k in range(0, 1476+1404):
#     if x_traintarget[k] == -1:
#         sum +=1
# print(sum)
# exit()
###############################################


print(valuse_train, valuse_valid, valuse_test)


plt.figure(figsize=(8,4))
plt.subplot(1,3,1)
plt.title("Train_data")
plt.bar(x, valuse_train, width=0.6, color=['red', 'blue'])
plt.ylim([0, 1500])
plt.xticks(x, lables)


plt.subplot(1,3,2)
plt.title("val_data")
plt.bar(x, valuse_valid, width=0.6, color=['red', 'blue'])
plt.ylim([0, 1500])
plt.xticks(x, lables)

plt.subplot(1,3,3)
plt.title("test_data")
plt.bar(x, valuse_test, width=0.6, color=['red', 'blue'])
plt.ylim([0, 1500])
plt.xticks(x, lables)
plt.show()

# 시각화 특정 시간에서의 시계열 샘플을 플롯
import random
# 정상 : 1 비정상 :-1

labels = np.unique(np.concatenate((x_traintarget, y_test), axis=0))
print(labels)
plt.figure(figsize=(10,4))
for c in labels:
    C_X_train = x_train[x_traintarget ==c]

    if c ==-1 : c =c+1
    time_t = random.randint(0, C_X_train.shape[0]) # 0~1404 사이의 랜덤한 정수 특정 time t 가 됨
    plt.scatter(range(0,500), C_X_train[time_t], label = "class = "+str(int(c)), marker='o', s=5)

plt.legend(loc='lower right')
plt.xlabel('Sensor', fontsize =15)
plt.xlabel("Sensor", fontsize=15)
plt.show()


