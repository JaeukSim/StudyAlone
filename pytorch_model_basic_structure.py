from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CustomDataset(Dataset):
    def __init__(self, iris, transforms=None):
        self.x = [i[0] for i in data]
        self.y = [i[1] for i in data] 

    
    def __len__(self):
        return len(self.x)

    
    def __getitem__(self, idx): 
        x = self.x[idx]
        y = self.y[idx]

        return x,y


data=[[2,0],[4,0], [6,0],[8,1],[10,1],[12,1]]

train_dataset = CustomDataset(data, transforms=None)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

for x, y in train_loader:
    print(x, y)



    
