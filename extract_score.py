import csv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import pandas as pd
import warnings
from utils.parameters import *
warnings.filterwarnings("ignore")

args = extract_score_parser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(nn.Linear(512, 1))

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model) 

model.load_state_dict(torch.load(args.model), strict = False)
model.to(device)
print("Load Finished")

class TestDataset(Dataset):
    def __init__(self, transform=None):
        self.file_list = glob.glob('./data/cluster_kr_unified/*.png')
        self.transform = transform        

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        name = path.split("/")[-1].split(".png")[0]
        image = io.imread(path) / 255.0
        if self.transform:
            image = self.transform(np.stack([image])).squeeze()
        return image, name

# To enforce the batch normalization during the evaluation
model.eval()
    
    
# Testing part
_mean = [0.485, 0.456, 0.406]
_std = [0.229, 0.224, 0.225]
test_dataset = TestDataset(transform = transforms.Compose([ToTensor(), Normalize(mean=_mean, std=_std)]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=False, num_workers=4)
print(len(test_dataset))

df2 = pd.read_csv('./data/{}'.format(args.test))
df2['predict'] = -1

with torch.no_grad():
    for batch_idx, (data, name) in enumerate(test_loader):
        print(batch_idx)
        data = data.to(device)
        scores = model(data).squeeze()
        count = 0
        for each_name in name:
            df2.loc[df2['y_x'] == each_name, 'predict'] = scores[count].cpu().data.numpy()
            count += 1
            
df_log_predicted2 = df_predicted2.copy()
df_log_predicted2['area'] = np.log(df_log_predicted2['area'])
print("Pearson Correlation (Log)")
print(df_log_predicted2.corr(method = 'pearson'))

print("Spearman Correlation (Original)")
print(df_predicted2.corr(method = 'spearman'))