import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import pandas as pd
from torch.utils.data import DataLoader
from utils.graph import *
from utils.siScore_utils import *
from utils.parameters import *

def make_data_loader(cluster_list, batch_sz):
    cluster_dataset = ClusterDataset(cluster_list, dir_name = args.dir_name, transform = transforms.Compose([
                                       RandomRotate(),
                                       ToTensor(),
                                       Grayscale(prob = 0.1),
                                       Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ]))
    cluster_loader = torch.utils.data.DataLoader(cluster_dataset, batch_size=batch_sz, shuffle=True, num_workers=4, drop_last=True)
    return cluster_loader
    
    
def generate_loader_dict(total_list, unified_cluster_list, batch_sz):
    loader_dict = {}
    for cluster_id in total_list:
        cluster_loader = make_data_loader([cluster_id], batch_sz)
        loader_dict[cluster_id] = cluster_loader        
    
    for cluster_tuple in unified_cluster_list:
        cluster_loader = make_data_loader(cluster_tuple, batch_sz)
        for cluster_num in cluster_tuple:
            loader_dict[cluster_num] = cluster_loader
    return loader_dict


def deactivate_batchnorm(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_parameters()
            m.eval()
            with torch.no_grad():
                m.weight.fill_(1.0)
                m.bias.zero_()
                

def train(args, epoch, model, optimizer, loader_list, cluster_path_list, device):
    model.train()
    # Deactivate the batch normalization before training
    deactivate_batchnorm(model.module)
    
    train_loss = AverageMeter()
    reg_loss = AverageMeter()
    
    # For each cluster route
    path_idx = 0
    avg_loss = 0
    count = 0
    for cluster_path in cluster_path_list:
        path_idx += 1
        dataloaders = []
        for cluster_id in cluster_path:
            dataloaders.append(loader_list[cluster_id])
    
 
        for batch_idx, data in enumerate(zip(*dataloaders)):
            cluster_num = len(data)
            data_zip = torch.cat(data, 0).to(device)

            # Generating Score
            scores = model(data_zip).squeeze()
            scores = torch.clamp(scores, min=0, max=1)
            score_list = torch.split(scores, args.batch_sz, dim = 0)
            
            # Standard deviation as a loss
            loss_var = torch.zeros(1).to(device)
            for score in score_list:
                loss_var += score.var()
            loss_var /= len(score_list)
            
            # Differentiable Ranking with sigmoid function
            rank_matrix = torch.zeros((args.batch_sz, cluster_num, cluster_num)).to(device)
            for itertuple in list(permutations(range(cluster_num), 2)):
                score1 = score_list[itertuple[0]]
                score2 = score_list[itertuple[1]]
                diff = args.lamb * (score2 - score1)
                results = torch.sigmoid(diff)
                rank_matrix[:, itertuple[0], itertuple[1]] = results
                rank_matrix[:, itertuple[1], itertuple[0]] = 1 - results

            rank_predicts = rank_matrix.sum(1)
            temp = torch.Tensor(range(cluster_num))
            target_rank = temp.unsqueeze(0).repeat(args.batch_sz, 1).to(device)

            # Equivalent to spearman rank correlation loss
            loss_train = ((rank_predicts - target_rank)**2).mean()
            loss = loss_train + loss_var * args.alpha
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.update(loss_train.item(), args.batch_sz)
            reg_loss.update(loss_var.item(), args.batch_sz)
            avg_loss += loss.item()
            count += 1

            # Print status
            if batch_idx % 10 == 0:
                print('Epoch: [{epoch}][{path_idx}][{elps_iters}] '
                      'Train loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                      'Reg loss: {reg_loss.val:.4f} ({reg_loss.avg:.4f})'.format(
                          epoch=epoch, path_idx=path_idx, elps_iters=batch_idx, train_loss=train_loss, reg_loss=reg_loss))
                
    return avg_loss / count
   

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Input example
    cluster_number = args.cluster_num
    
    # Graph generation mode
    if args.graph_config:
        graph_config = args.graph_config  
    elif args.mode == "census":
        df = pd.read_csv(args.census_path)
        hist = pd.read_csv(os.path.join('./data', args.dir_name, args.histogram_path), header = None)
        graph_config = graph_inference_census(df, hist, cluster_number, args.graph_name)
    elif args.mode == "nightlight":
        grid_df = pd.read_csv(os.path.join('./data', args.dir_name, args.grid_path))
        nightlight_df = pd.read_csv(args.nightlight_path)
        graph_config = graph_inference_nightlight(grid_df, nightlight_df, cluster_number, args.graph_name)        
    else:
        raise ValueError
    
    # Dataloader definition   
    start, end, partial_order, cluster_unify = graph_process(graph_config)   
    loader_list = generate_loader_dict(range(cluster_number), cluster_unify, args.batch_sz)
    cluster_graph = generate_graph(partial_order, cluster_number)
    cluster_path_list = cluster_graph.printPaths(start, end)
    print("Cluster_path: ", cluster_path_list)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model) 
        cudnn.benchmark = True

    model.load_state_dict(torch.load(args.pretrained_path)['state_dict'], strict = False)
    model.module.fc = nn.Sequential(nn.Linear(512, 1))
    model.to(device)
    

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    print("Pretrained net load finished")
    
    best_loss = float('inf')
    if args.load == False:    
        for epoch in range(args.epochs):          
            loss = train(args, epoch, model, optimizer, loader_list, cluster_path_list, device)

            if epoch % 10 == 0 and epoch != 0:                
                if best_loss > loss:
                    print("state saving...")
                    state = {
                        'model': model.state_dict(),
                        'loss': loss
                    }
                    if not os.path.isdir('checkpoint'):
                        os.mkdir('checkpoint')
                    torch.save(state, './checkpoint/{}'.format(args.name))
                    best_loss = loss
                    print("best loss: %.4f\n" % (best_loss))
        

if __name__ == "__main__":
    args = siScore_parser()
    main(args)

