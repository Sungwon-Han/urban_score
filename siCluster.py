import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from utils.siCluster_utils import *
from utils.parameters import *


def main(args):
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    path = args.pretrained_path
    model = models.resnet18(pretrained=False)
    model = nn.DataParallel(model)
    model.module.fc = nn.Linear(512, 10)
    cudnn.benchmark = True
    model.load_state_dict(torch.load(path)['state_dict'], strict = False)
    model.module.fc = nn.Sequential()
    model.cuda()
    cudnn.benchmark = True
    
    cluster_transform =transforms.Compose([
                      transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    train_transform1 =transforms.Compose([
                      transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    train_transform2 =transforms.Compose([
                      transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    
    criterion = nn.CrossEntropyLoss().cuda()
    criterion2 = AUGLoss().cuda()
    if args.mode == 'rural':
        clusterset = GPSDataset('./meta_data/meta_rural.csv', './data/kr_data/', cluster_transform)
        trainset = GPSDataset('./meta_data/meta_rural.csv', './data/kr_data/', train_transform1, train_transform2)
    elif args.mode == 'city':
        clusterset = GPSDataset('./meta_data/meta_city.csv', './data/kr_data/', cluster_transform)
        trainset = GPSDataset('./meta_data/meta_city.csv', './data/kr_data/', train_transform1, train_transform2)
    else:
        raise ValueError
        
    clusterloader = torch.utils.data.DataLoader(clusterset, batch_size=args.batch, shuffle=False, num_workers=1)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=1, drop_last = True)
    deepcluster = Kmeans(args.nmb_cluster)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    
    
    for epoch in range(0, args.epochs):
        print("Epoch : %d"% (epoch))
        features = compute_features(clusterloader, model, len(clusterset), args.batch) 
        clustering_loss, p_label = deepcluster.cluster(features)
        p_label = p_label.tolist()
        p_label = torch.tensor(p_label).cuda()

        model.train()
        fc = nn.Linear(512, args.nmb_cluster)
        fc.weight.data.normal_(0, 0.01)
        fc.bias.data.zero_()
        fc.cuda()

        for batch_idx, (inputs1, inputs2, indexes) in enumerate(trainloader):
            inputs1, inputs2, indexes = inputs1.cuda(), inputs2.cuda(), indexes.cuda()           
            batch_size = inputs1.shape[0]
            labels = p_label[indexes].cuda()
            inputs = torch.cat([inputs1, inputs2])
            outputs = model(inputs)
            outputs1 = outputs[:batch_size]
            outputs2 = outputs[batch_size:]
            outputs3 = fc(outputs1)
            ce_loss = criterion(outputs3, labels)
            aug_loss = criterion2(outputs1, outputs2) / batch_size
            loss = ce_loss + aug_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 20 == 0:
                print("[BATCH_IDX : ", batch_idx, "LOSS : ",loss.item(), "CE_LOSS : ",ce_loss.item(),"AUG_LOSS : ",aug_loss.item(),"]" )
    torch.save(model.state_dict(), './checkpoint/ckpt_cluster_{}.t7'.format(args.mode))
                                                       
    
if __name__ == "__main__":
    args = siCluster_parser()
    main(args)    
    