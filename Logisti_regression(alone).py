import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
# data
x_data = [[1,2], [2,3], [3,1], [4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

# data -> tensor
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# class
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,1) # input dim 2 output dim 1
        self.sigmoid = nn.Sigmoid() # output -> sigmoid

    def forward(self,x):
        return self.sigmoid(self.linear(x))
        
# Model 선언
model = BinaryClassifier()

# optimizer 
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# epoch 설정
epoch_number = 300



# Train Loop

for epoch in range(epoch_number+1):
    # H(x) 계산
    hypothesis = model(x_train)

    # loss
    loss = F.binary_cross_entropy(hypothesis, y_train)

    # loss H(x) 개선
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #print 문 만들기
    if epoch % 10 ==0:
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        acc = correct_prediction.sum().item() / len(correct_prediction)
        print("Epoch : {:4d}/{} loss : {:.6f} Acc {:2.2f}%".format(epoch, epoch_number, loss, acc))
