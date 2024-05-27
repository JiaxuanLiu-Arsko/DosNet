import torch
from data._dataset import Dataset
from model._dos_net import DosNet
from training import train_val_func
import joblib
#from torchsummary import summary
import argparse
import torch.distributed as dist
import os

parser = argparse.ArgumentParser()

parser.add_argument('--world-size', default=4, type=int, help='number of distributed processes')
parser.add_argument('--local-rank', type=int, help='rank of distributed processes')
args = parser.parse_args()

if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(os.environ['LOCAL_RANK'])
else:
    rank = int(args.rank)
    world_size = int(args.world_size)
    gpu = int(args.gpu)
    print('Not using distributed mode')
    
# 设置当前程序使用的GPU。根据python的机制，在单卡训练时本质上程序只使用一个CPU核心，而DataParallel
# 不管用了几张GPU，依然只用一个CPU核心，在分布式训练中，每一张卡分别对应一个CPU核心，即几个 GPU几个CPU核心
torch.cuda.set_device(gpu)
 
# 分布式初始化
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
dist.init_process_group('nccl', rank=rank, world_size=world_size)
print('| distributed init (rank {})'.format(rank), flush=True)

target_file = "/work/jxliu/e3nn/DOS_net/DosNet_pl/data/targets.csv"
structure_path = "/work/jxliu/e3nn/DOS_net/DosNet_pl/data/structure/"
post_path = "/work/jxliu/e3nn/DOS_net/DosNet_pl/data/preprocessed/"
dataset = Dataset()
dataset.load_from_dir(target_file , structure_path, post_path)
joblib.dump(dataset, "./dataset.pth")
#dataset = joblib.load("./dataset_small.pth")
#dataset = joblib.load("./dataset.pth")
train_dataloader, valid_dataloader = dataset.get_data_Loader(15, train_ratio=0.8, multicard=True,
                                                            rank=gpu, world_size=world_size, random_seed=2024)

if gpu==0:
    print("dataset loaded", flush=True)

n = 118
n_basis = 400
layer_n_list = [n, int((n+n_basis)/2), n_basis]
layer_l_list = [3, 3, 0]
qk_irreps = f"{n}x0e+{n}x1o+{n}x2e"
layer_input = f"{n}x0e"
irreps_list = []
for i, layer_n in enumerate(layer_n_list):
    edge_irreps = f""
    layer_irreps = f""
    for l in range(0, layer_l_list[i]+1):
        s = "o"
        if l%2==0:
            s = "e"
        edge_irreps += f"1x{l}{s}+"
        layer_irreps += f"{layer_n}x{l}{s}+"
    layer_irreps = layer_irreps[:-1]
    edge_irreps = edge_irreps[:-1]
    irreps = {"input":layer_input, "output":layer_irreps, "edge":edge_irreps, "query":qk_irreps, "key":qk_irreps}
    irreps_list.append(irreps)
    layer_input = layer_irreps
config = {"num_types":n, 
        "irreps_list":irreps_list,
        "r_max":6, 
        "fc_neurons":8,
        "num_basis":n_basis,
        }

model = DosNet(config).cuda()
#model.load_state_dict(torch.load("./checkpoints/model_373.pth"))
model = torch.nn.parallel.DistributedDataParallel(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10,    
            verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
epochs = 500
for epoch in range(0, epochs):
    torch.cuda.empty_cache()
    train_loss = train_val_func.train(model, optimizer, train_dataloader)
    scheduler.step(train_loss)
    dist.barrier()
    if epoch%1 == 0 and gpu==0:
        val_loss = train_val_func.evaluate(model.module, valid_dataloader)
        print("epoch: ", epoch, "train loss ", train_loss, "val loss ", val_loss, flush=True)
        torch.save(model.module.state_dict(), f'./checkpoints/model_{epoch}.pth')
        torch.save(optimizer.state_dict(), f'./checkpoints/optimizer_{epoch}.pth')
        with open(f"./checkpoints/gradients_{epoch}.txt", "w") as logfile:
            for name, parms in model.module.named_parameters():	
                logfile.write(f'-->name:{name}\n')
                logfile.write(f'-->grad_requirs:{parms.requires_grad}\n')
                logfile.write(f'-->std_grad_value:{torch.std(parms.grad)}\n')
                logfile.write(f'-->mean_grad_value:{torch.mean(parms.grad)}\n')
                logfile.write(f'-->std_param_value:{torch.std(parms)}\n')
                logfile.write(f'-->mean_param_value:{torch.mean(parms)}\n')
                logfile.write("=====================================================\n")# Total params
                total_params = sum(p.numel() for p in model.parameters())
            logfile.write(f"{optimizer}")

dist.destroy_process_group()