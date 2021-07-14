
import itertools
from re import T
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
x_valid = np.concatenate((normal_x[ind_x_normal:], abnormal_x[ind_x_abnormal:]), axis=0) #20

y_train = np.concatenate((normal_y[:ind_y_normal], abnormal_y[:ind_y_abnormal]), axis=0) #80
y_valid = np.concatenate((normal_y[ind_y_normal:], abnormal_y[ind_y_abnormal:]), axis=0) #20

# 시각화

# class 종류 정상 1 비정상 -1 두개 밖에 없다는 것을 확인
classes = np.unique(np.concatenate((y_train, y_test), axis=0))
print("################################",len(classes))


x = np.arange(len(classes))

lables = ["Abnormal", "Normal"] #plot x축 이름

# train, valid, test 세 세트가 필요함
valuse_train = [(y_train == i).sum() for i in classes]
valuse_valid = [(y_valid == i).sum() for i in classes]
valuse_test = [(y_test == i).sum() for i in classes]
# 이 부분의 작동 방식이 잘 이해가 안됩니다. ㅠㅠ classes의 원소는 -1 1 두개밖에 없고, classes의 길이도 2인데
# for 문을 두번 밖에 안 도는데 어떻게 결과값이 1400이 넘게 나오죠?
# 해답 : classes 에는 -1과 1 두개밖에 없는데 i=-1일때 만약 y_train[i] == i(-1) 이면 그 횟수만큼 1씩 더해 나간다.
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
#정상 :1 비정상 :-1

labels = np.unique(np.concatenate((y_train, y_test), axis=0))
# y_train 코드에는 트레인 셋의 정답지 중 정상과 비정상데이터의 80퍼센트가 저장되어있다
# y_test에는 테스트 문서의 정답지가 들어가 있다.
print(labels)
plt.figure(figsize=(20,4))
count=0
for c in labels:
    count +=1
    C_X_train = x_train[y_train == c] # y_train(정답지 중 80%) 값중 c값과 일치하면 그 해당 번째의 x_train의 행을 가져와서 C_X_train에 놓는다.
    print("CXTRAIN########################\n",C_X_train)
    print("CXTRAIN########################\n",C_X_train.shape)
    
    
    if c==-1:
        c = c+1
    time_t = random.randint(0, C_X_train.shape[0]) # 0~1404 사이의 랜덤한 정수 특정 time t 가 됨
    plt.scatter(range(0,500), C_X_train[time_t], label = "class = "+str(int(c)), marker='o', s=5)


plt.legend(loc='lower right')
plt.xlabel('Sensor', fontsize =15)
plt.ylabel("Sensor value", fontsize=15)
plt.show()

# 특정 시간에서의 시계열 샘플을 (정상 비정상 샘플로 각각 출력)
def get_scatter_plot():
    time_t = random.randint(0, C_X_train.shape[0])
    print("Random time number : ", time_t)

    plt.scatter(range(0,  C_X_train.shape[1]),  C_X_train[time_t], marker='o', s=5, c="r" if c ==-1 else "b")
    plt.title(f"at time t_{time_t}", fontsize=20)
    plt.xlabel("Sensor", fontsize=15)
    plt.ylabel("Sensor value", fontsize=15)
    plt.show()

labels = np.unique(np.concatenate((y_train, y_test)), axis=0)
print(labels.dtype)
for c in labels :
    C_X_train = x_train[y_train ==c]

    if c == -1:
        print("Abnormal Label number data : ", len(C_X_train))
        get_scatter_plot()
    else:
        print("Normal Label number data : ", len(C_X_train))
        get_scatter_plot()

# 시각화 임의의 센서 값의 시계열 show
sensor_number = random.randint(0, 500)
print(f"random sensor number {sensor_number}")
plt.figure(figsize=(13,4))
plt.title(f"sensor number {sensor_number}", fontsize=20)
plt.plot(x_train[:, sensor_number])
plt.xlabel("time", fontsize=15)
plt.ylabel("sensor value", fontsize =15)
plt.show()



# 데이터 특성 파악
import matplotlib.cm as cm
from matplotlib.collections import EllipseCollection
df = pd.DataFrame(data = x_train, columns=["sensor_{}".format(label+1) for label in range(x_train.shape[-1])])

data = df.corr()
print(data)

M = np.array(data)

print(M.ndim)
print(M)
print(M.shape[1])
print(M.shape[0])

def plot_corr_ellipse(dat, ax=None, **kwargs):
    M = np.array(data)
    if not M.ndim == 2:
        return ValueError("data must be a 2D array")

    if ax is None:
        flg, ax = plt.subplot(1,1, subplot_kw={'aspect':'equal'})
        ax.set_xlim(-0.5, M.shape[1]-0.5)
        ax.set_ylim(-0.5, M.shape[0] -0.5)
    xy = np.indices(M.shape)[::-1].reshape(2,-1).T
    # np.indices((a,b))를 하면 a행 b열짜리 행렬이 두 덩어리가 생기는데
    # 첫번째 덩어리는  0 0 0 ... 0 0 0 0
    #                 1 1 1 ... 1 1 1 1
    #                 .................
    #                 b b b ... b b b b  이렇게 생기고 
    # 두번째 덩어리는  0 1 2 ..........b
    #                 0 1 2 ..........b
    #                 ................. 
    #                 0 1 2 ..........b 이렇게 생긴다.
    # [::-1] 은 행렬의 덩어리 순서를 거꾸로 뒤집는다.
    # reshape(a,-1) 하면 위의 두덩어리의 행렬을 다 펴서 a행 으로 만들고 끝까지 불러온다.
    # T는 트랜스포스
    # 헷갈리면 아래 코드 실행
    ###############################################
    # xy = np.indices((10,10))[::-1].reshape(2,-1)
    # print(xy)
    ###############################################

    
    w = np.ones_like(M).ravel()
    h = 1 - np.abs(M).ravel()
    a = 45 * np.sign(M).ravel() #sign함수는 부호를 판별하여 양이면 1 음이면 -1 0이면 0을 반환
    ec = EllipseCollection(widths=w, heights=h, angles = a, units = 'x', offsets=xy, transOffset = ax.transData, array=M.ravel(), **kwargs)
    ax.add_collection(ec)

    if isinstance(data, pd.DataFrame):
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_xticklabels(data.columns, rotation = 90)
        ax.set_yticks(np.arange(M.shape[0]))
        ax.set_yticklabels(data.index)
    return ec
fig, ax = plt.subplots(1,1, figsize=(20,20))
cmap = cm.get_cmap('jet', 31)
m = plot_corr_ellipse(data, ax=ax, cmap=cmap)
cb = fig.colorbar(m)
cb.set_label("Correlation coefficient")
plt.title("Correlation between Feature")
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)

plt.tight_layout()
plt.show()


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
"""
동일 시간 길이(3,600) 내 센서 값들이 상당히 넓은 범위로 퍼져 있을 뿐만 아니라, 변
수 간의 Scale이 서로 다르기 때문에, 데이터를 그대로 학습하는 것은 일반적으로 적
절하지 않다. 따라서 인풋 값들을 정규화(Normalization) 과정을 거치는데,
StandardScaler 또는 RobustScaler를 통해 진행한다.
흔히 공정 데이터에 이상치(Outlier)가 발생할 수 있는데 이에 강건한 정규화가 필요
할 때가 있다. 이때 RobustScaler를 사용한다. StandardScaler는 보다 더 일반적으
로 많이 사용하는 정규화 방법으로, 데이터를 단위 분산으로 조정함으로써 Outlier에
취약할 수 있는 반면, RobustScaler는 Feature 간 은 스케일을 갖게 되지만 평균과
분산 대신 중간 값(median)과 사분위값(quartile)을 사용함으로써, 극단값(Outlier)
에 영향을 받지 않는 특징이 있다.
"""
# Stander

stder = StandardScaler()
stder.fit(x_train)
x_train = stder.transform(x_train)
x_valid = stder.transform(x_valid)
print(x_train, x_valid)

# RobustScaler
# rscaler = RobustScaler()
# rscaler.fit(x_train)
# x_train = rscaler.transform(x_train)
# x_valid = rscaler.transform(x_valid)
# print(x_train, x_valid)

from sklearn.linear_model import LogisticRegression
clf_lr_1 = LogisticRegression(
penalty = 'l2',
C=1,
fit_intercept = True,
intercept_scaling = 1,
random_state = 2,
solver='lbfgs',
max_iter = 1000,
multi_class = 'auto',
verbose=0
)




# numpy 로 직접 : LogisticRegression 구현
# numpy 로 직접 : LogisticRegression 구현
class LogisticRegression2:
    def __init__(self, lr=0.01, num_iter =1000, fit_intercept=True, verbose =False ):
        self.lr =lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose= verbose
        self.eps = 1e-10
        self.threshold = 0.5
        self.loss_history = list()


    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0],1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def __loss(self, h, y):
        return (-y *np.log(h+self.eps)- (1-y) * np.log(1-h + self.eps)).mean()

    
        
    #fit() 학습된 데이터로 model을 학습하는  메서드
    def fit(self, X, y):
        if self.fit_intercept:
            x = self.__add_intercept(X)

        # weights
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            logit = np.dot(X, self.theta)
            hypothesis = self.__sigmoid(logit)
            gradint = np.dot(X.T, (hypothesis - y)) / y.size
            self.theta -= self.lr*gradint

            if self.verbose == True and i % 10 ==0:
                loss = self.__loss(hypothesis, y)
                print(f"epoch : {i} \t loss : {loss} \t")
                self.loss_history.append(loss)

        return self.loss_history
    
    
    #학습 데이터로 학습된 모델을 바탕으로 테스트 데이터의 각 인스턴스의 정상일 확률을 도출하는 메서드
    def predict_prob(self, X):
        if self.fit_intercept:
            X= self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.theta))

    #학습 데이터로 학습된 모델을 바탕으로 테스트 데이터의 라벨을 확인하는 메서드
    def predict(self, X):
        predict_label = np.where(self.predict_prob(X)>self.threshold, 1,0)
        return predict_label

    # 모델 테스트 정확도 도출
    def eval(self, x, y):
        res_y = np.round(self.predict_prob(x), 0)
        accuracy = np.sum(res_y == y) / len(y)
        return accuracy
      


      

x_train_lr = np.concatenate((x_train, x_valid), axis=0)
y_train_lr = np.concatenate((y_train, y_valid), axis=0)

# Scikit learn 에서 제공하는 로지스틱 회귀 학습
clf_lr_1.fit(x_train_lr, y_train_lr)

# test
y_pred = clf_lr_1.predict(x_test)
score = clf_lr_1.score(x_test, y_test)
print("Logistic Regression Prediction Rate : ", round(score*100, 2), "%")

clf_lr_2 = LogisticRegression2(lr=0.01, num_iter=1000, verbose = True)
history_lr = clf_lr_2.fit(x_train_lr, y_train_lr)



