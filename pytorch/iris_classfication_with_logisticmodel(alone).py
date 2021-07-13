# 로지스틱 모델 (붗꽃) 데이터 활용
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
torch.manual_seed(1)
import matplotlib.pyplot as plt
#데이터 로드
from sklearn import datasets

# 로지스틱 회귀모델 훈련
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
list_iris=[]


x = iris['data'][:, 3:] #꽃잎의 너비 변수만 사용
y = (iris['target']==2).astype('int') # iris-Versibica 1 or 0////////
# iris['target'] ==2 하면 2인 애들은 true, 아니면 false인 리스트가 나오는데 int로 바꿔서 true를 1로, false를 0으로 바꾼것같습니다

#print("iris 꽃잎의 너비 \n", x)
# print(y)
# print(x.shape)
# print(y.shape)

log_reg = LogisticRegression(solver = 'liblinear') #liblinear는 작은데이터에 적합 L1 L2제약 조건 두가지 모두 지원
log_reg.fit(x,y)
LogisticRegression(C=1.0 , class_weight=None, dual=False, fit_intercept=True, 
                    intercept_scaling=1, l1_ratio=None, max_iter=100, 
                    random_state=None, solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

# 이제 꽃잎이 너비가 0~3cm 인 꽃에 대해 모델의 추정확률을 계산
x_new = np.linspace(0,3, 100).reshape(-1,1) 
# print(x_new)
y_proba = log_reg.predict_proba(x_new)
# print(y_proba)
# plt.plot(x_new, y_proba[:,1], '-g', label = "Iris-virginica") #음성 클래스
# plt.plot(x_new, y_proba[:,0], 'b--', label = "not Iris-virginica")
# plt.legend()
# plt.show()

# 좀더 보기 좋게 변경
x_new = np.linspace(0,3, 1000).reshape(-1,1)
y_proba = log_reg.predict_proba(x_new)
# y_proba 에는 [a,b] 형식으로 나타나며 a, b 모두 확률을 나타낸다. 
# 이때 확률의 특성상 a+b=1 이고 a가 Iris_virginica가 아닐 확률 b는 iris_virginica가 맞을 확률이다.
decision_boundary = x_new[y_proba[:,1]>=0.5][0] 
# y_proba의 리스트 전체 중 1번째의 항목 0.5(확률) 이상이 되는 순간 그 해당번째에 해당하는 x_new의 0번째 원소를 가져온다

plt.figure(figsize=(8,3)) #그래프 사이즈
# 현재 x에는 꽃잎의 4번째 너비 변수만 저장되어 있음 총 150개의 데이터
# 현재 y에는 꽃의 종류가 iris_Versibica이면 1 아니면 0 으로 0과1만 저장되어 있음 총 150개의 데이터

plt.plot(x[y==0], y[y==0], 'bs') 
# x와 y에 저장되어 있는 값에 따라 (x,y)점을 그래프에 찍어나간다
# 그런데 y에 저장되어 있는 값이 0 이면 그때 해당 번째 (x data 값 , y data 값(즉=0)) 을 좌표에 찍는다
plt.plot(x[y==1], y[y==1], 'g^')
# x와 y에 저장되어 있는 값에 따라 (x,y)점을 그래프에 찍어나간다
# 그런데 y에 저장되어 있는 값이 1 이면 그때 해당 번째 (x data 값 , y data 값(즉=1)) 을 좌표에 찍는다

# 결정경계 표시하기
plt.plot([decision_boundary, decision_boundary], [-1,2], "k:", linewidth=2)
# plt.plot([x,y],[z,w]) 이렇게 하면 (x,z)와 (y,w)를 잇는 직선을 그린다.

# 추정확률
plt.plot(x_new, y_proba[:,1], 'g-', linewidth=2, label="Iris_virginica")
plt.plot(x_new, y_proba[:,0], 'b-', linewidth=2, label="Not Iris_virginica")
plt.text(decision_boundary+0.02, 0.15, "Decision Boundary", fontsize=14, color = 'k', ha='center')
plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width = 0.05, head_length = 0.1, fc='b', ec='b')
plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width = 0.05, head_length = 0.1, fc='g', ec='g')
# plt.arrow(화살표 시작지점의 x좌표, 화살표 시작지점의 y좌표, +-로 가로방향결정, +-로 세로방향 결정, head_width = 화살넓이, head_length = 화살머리 길이, fc='g', ec='g'                                )
plt.xlabel("Petal width (cm)", fontsize= 14)
plt.ylabel("Probability", fontsize= 14)
plt.legend(loc = 'center left', fontsize =14)
plt.axis([0,3, 0, 1]) #plt.axis([xmin, xmax ,ymin, ymax])

plt.show()

# 해석 그래프를 보면 iris virginica 1.4 -2.5
# 일반적인 꽃잎 -> 1.8 보다 작게 분포
# 중첩 구간이 존재합니다.

# 결정 경계가 어떤 값을 가지고 있는지 확인
print("decision_boundary : ", decision_boundary)

# 양쪽의 확률이 50퍼센트가 되는 1.6 근방에서 결정경계가 만들어지고 분류기는 1.6보다 크면 iris virginica 분류
# 작으면 일반 꽃 잎으로 분류

test_code = log_reg.predict([[1.8], [1.48]])
print(f"진짜 우리가 원하는 분류가 되는가 확인 하는 test code 입니다., 분리기준 : {decision_boundary} 결과 {test_code}")

# 꽃잎 너비와 꽃잎 길이 2개의 변수를 이용해서 훈련 실습
x = iris['data'][:, (2,3)] #petal length(2번째 feature), petal width(3번째 feature) 
print(x)
y = (iris["target"]==2).astype(np.int)

log_reg = LogisticRegression(solver='lbfgs', C=10**10, random_state=42)
log_reg.fit(x,y)

# 2차원 그리드 포인트 생성 (아까 위에서는 x축에는 꽃잎의 길이 y는 그에 해당하는 0혹은1로 나타내는 결과였다면 지금은 (petal length, petal width)를 좌표평면 상에 찍는다)
# 변수가 2개인 2차원 함수의 그래프를 그리거나 표로 작성하려면 2차원 영역에 대한 (x,y) 좌표값 쌍 즉 그리드 포인트를 생성하
# 각 좌표에 대한 함수값을 계산한다.
# -> meshgrid 명령어 : 그리드 포인트

x1, x2 = np.meshgrid(
    np.linspace(2.9, 7, 500).reshape(-1,1),
    np.linspace(0.8, 2.7, 200).reshape(-1,1)
)
# 위와 같은 명령을 하면 행렬처럼 생각하면 쉽다
# x1,x2에는 각각 같은 사이즈의 행렬이 생성된다.
# 위와 같은 경우는 첫번째로 np.linspace(2.9, 7, 500) 가 들어가고 두번째로 np.linspace(0.8, 2.7, 200).reshape(-1,1)이 들어간다
# 따라서 200X500(행 X 열) 사이즈의 행렬이 생성 되는데 그렇게 맞춰 주기 위해서는
# x1 에는 열에는 [2.9~~~~~7 까지 500개의 원소가 균등하게 들어가 있고] 이 줄이 밑으로 200줄 즉 200행이 반복된다.
# x2 에는 똑같이 200x500을 맞춰주기 위해서 0.8 부터 2.7 까지의 원소가 균등하게 밑으로 200줄(행)을 따라 분포되어 있고
# 열에는 첫번째 열에 위치한 원소가 500번 반복되어서 가로 방향으로 쭉 배열 된다.
# 그래도 이해가 안돼면 밑에 있는 코드를 실행해보면 된다. 
########################################
# x=np.linspace(2,5,4)
# y=np.linspace(2,4,3)
# xx,yy=np.meshgrid(x,y)
# print(xx)
# print(yy)
########################################


x_new = np.c_[x1.ravel(), x2.ravel()] # ravel()함수는 다차원 배열을 1차원 배열로 평평하게 만들어준다.
##############################################################################
# np.c_ : 두개의 1차원 배열을 컬럼으로 세로로 붙여서 2차원 배열을 만들어 준다.
# a = np.array([1,2])
# b = np.array([3,4])
# c = np.c_[a,b]
# print(c)
# c = [1 3]
#     [2,4]
##############################################################################      

# 훈련시킨 log_reg함수에 x_new데이터를 넣어 예상값을 뽑아낸다.
y_proba = log_reg.predict_proba(x_new)

plt.figure(figsize=(10,4))
plt.plot(x[y==0,0], x[y==0,1], "bs")
# [a,b]로 표현되는 변수인 x에 저장되어 있는 값에 따라 (x[0], x[1])점을 그래프에 찍어나간다
# 그런데 y에 저장되어 있는 값이 0 이면 그때 해당 번째 (x data[0] 값 , x data[1] 값을 좌표에 찍는다
plt.plot(x[y==1,0], x[y==1,1], "g^")
# [a,b]로 표현되는 변수인 x에 저장되어 있는 값에 따라 (x[0], x[1])점을 그래프에 찍어나간다
# 그런데 y에 저장되어 있는 값이 1 이면 그때 해당 번째 (x data[0] 값 , x data[1] 값을 좌표에 찍는다
print(y_proba)

zz = y_proba[:,1].reshape(x1.shape) #zz에는 좌표평면에 찍힌 각각의 점들에 대한 확률 값이다.
# 즉 각 점들은 좌표평면 상에 나타나고 그 각 점들은 Iris virginica에 대한 확률값을 모두 가지고 있다.(물론 좌표 평면상에 점 하나하나마다 확률 값이 표시되어있지는 않다.)
# 따라서 각 점에 따른 확률 값이 같은 것들이 존재할텐데 그 점들끼리 아래와 같이 이어주는 것이다.
contour = plt.contour(x1, x2, zz, cmap=plt.cm.brg) 


left_right = np.array([2.9,7])
# 위에서 log_reg로 훈련한 결과로서
# ax+by+c=0일때의 가중치와 intercept 값인 a,b가 log_reg[0][0]=a, log_reg[0][1]=b, c= log_reg.intercept_[0]에 저장된다.
boundary = -(log_reg.coef_[0][0]*left_right + log_reg.intercept_[0])/log_reg.coef_[0][1]
# ax+by+c =0 에서 y = -(ax+c)/b

plt.clabel(contour, inline=1, fontsize = 12) #이렇게 하면 등고선에 자동으로 같은 값의 확률이 그래프상에 표시 된다.


plt.plot(left_right, boundary, "k--", linewidth=3)
plt.text(3.5, 1.5, "Not Iris virginica", fontsize=14, color="b", ha='center')
plt.text(6.5, 2.3, "Iris virginica", fontsize=14, color="r", ha='center')
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.axis([2.9, 7, 0.8, 2.7])
plt.show()