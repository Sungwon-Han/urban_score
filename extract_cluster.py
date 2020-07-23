import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from utils.siCluster_utils import *
from utils.parameters import *
import glob
import shutil
import copy
import csv


def extract_city_cluster(args):
    convnet = models.resnet18(pretrained=True)
    convnet = torch.nn.DataParallel(convnet)    
    ckpt = torch.load('./checkpoint/{}'.format(args.city_model))
    convnet.load_state_dict(ckpt, strict = False)
    convnet.module.fc = nn.Sequential()
    convnet.cuda()
    cluster_transform =transforms.Compose([
                      transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])    
    
    clusterset = GPSDataset('./meta_data/meta_city.csv', './data/kr_data/', cluster_transform)
    clusterloader = torch.utils.data.DataLoader(clusterset, batch_size=256, shuffle=False, num_workers=1)
    
    deepcluster = Kmeans(args.city_cnum)
    features = compute_features(clusterloader, convnet, len(clusterset), 256) 
    clustering_loss, p_label = deepcluster.cluster(features)
    labels = p_label.tolist()
    f = open('./meta_data/meta_city.csv', 'r', encoding='utf-8')
    images = []
    rdr = csv.reader(f)
    for line in rdr:
        images.append(line[0])
    f.close()
    images.pop(0)    
    city_cluster = []
    for i in range(0, len(images)):
        city_cluster.append([images[i], labels[i]]) 
        
    return city_cluster

def extract_rural_cluster(args):
    convnet = models.resnet18(pretrained=True)
    convnet = torch.nn.DataParallel(convnet)    
    ckpt = torch.load('./checkpoint/{}'.format(args.rural_model))
    convnet.load_state_dict(ckpt, strict = False)
    convnet.module.fc = nn.Sequential()
    convnet.cuda()
    cluster_transform =transforms.Compose([
                      transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])    
    
    clusterset = GPSDataset('./meta_data/meta_rural.csv', './data/kr_data/', cluster_transform)
    clusterloader = torch.utils.data.DataLoader(clusterset, batch_size=256, shuffle=False, num_workers=1)
    
    deepcluster = Kmeans(args.rural_cnum)
    features = compute_features(clusterloader, convnet, len(clusterset), 256) 
    clustering_loss, p_label = deepcluster.cluster(features)
    labels = p_label.tolist()
    f = open('./meta_data/meta_rural.csv', 'r', encoding='utf-8')
    images = []
    rdr = csv.reader(f)
    for line in rdr:
        images.append(line[0])
    f.close()
    images.pop(0)    
    rural_cluster = []
    for i in range(0, len(images)):
        rural_cluster.append([images[i], labels[i] + args.city_cnum])
        
    return rural_cluster

def extract_nature_cluster(args):
    f = open('./meta_data/meta_nature.csv', 'r', encoding='utf-8')
    images = []
    rdr = csv.reader(f)
    for line in rdr:
        images.append(line[0])
    f.close()
    images.pop(0)    
    nature_cluster = []
    cnum = args.city_cnum + args.city_cnum
    for i in range(0, len(images)):
        nature_cluster.append([images[i], cnum])
            
    return nature_cluster



def main(args):
    # make cluster directory
    city_cluster = extract_city_cluster(args)
    rural_cluster = extract_rural_cluster(args)
    nature_cluster = extract_nature_cluster(args)
    total_cluster = city_cluster + rural_cluster + nature_cluster
    cnum = args.city_cnum + args.rural_cnum
    cluster_dir = './data/{}/'.format(args.cluster_dir)
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)
        for i in range(0, cnum + 1):
            os.makedirs(cluster_dir + str(i))
    else:
        raise ValueError
    
    for img_info in total_cluster:
        cur_dir = './data/kr_data/' + img_info[0]
        new_dir = cluster_dir + str(img_info[1])
        shutil.copy(cur_dir, new_dir)
        
    # make cluster census histogram for census mode
    data_dir = './data/kr_data/'
    cluster_list = []
    for i in range(0, cnum):
        cluster_list.append(os.listdir(cluster_dir + str(i)))
    cluster_score = []    
    for i in range(0, cnum):
        cluster_score.append(0)
    
    histogram_dir = cluster_dir + args.histogram
    f = open(histogram_dir, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)

    for i in range(1, 231):
        r_list = os.listdir(data_dir + str(i))
        r_score = copy.deepcopy(cluster_score)
        for region in r_list:
            for i in range(0, len(cluster_list)):
                if region in cluster_list[i]:
                    r_score[i] += 1
                    break
        wr.writerow(r_score)
    f.close()
    
    # make metadata for cluster && total dataset for eval
    file_list = glob.glob("./{}/*/*.png".format(args.cluster_dir))
    grid_dir = cluster_dir + args.grid
    f = open(grid_dir, 'w', encoding='utf-8')
    wr = csv.writer(f)
    wr.writerow(['y_x', 'cluster_id'])
    
    for file in file_list:
        file_split = file.split("/")
        folder_name = file_split[2]
        file_name = file_split[-1].split(".")[0]
        wr.writerow([file_name, folder_name])
    f.close()
    
        
    if not os.path.exists('./data/cluster_kr_unified'):
        os.makedirs('./data/cluster_kr_unified')
    for i in range(cnum + 1):
        file_dir = cluster_dir + '{}/*.png'
        file_list = glob.glob(file_dir.format(i))    
        for cur_dir in file_list:
            shutil.copy(cur_dir, './data/cluster_kr_unified')
    
if __name__ == "__main__":
    args = extract_cluster_parser()
    main(args)    
    
