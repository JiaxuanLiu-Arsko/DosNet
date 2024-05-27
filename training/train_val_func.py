import torch.nn.functional as F
import torch
import numpy as np
from torch_geometric.nn.pool import global_mean_pool, global_add_pool
from matplotlib import pyplot as plt

def train(model, optimizer, loader, loss = "mse_loss"):
    if loss != "mse_loss":
        print("Not implemented")
        return None
    model.train()
    model.cuda()
    loss_all = 0
    count = 0
    for data in loader:
        data.cuda()
        optimizer.zero_grad()
        dos_out = model(data)
        loss = F.mse_loss(global_mean_pool(dos_out, data.batch), global_add_pool(data.dos_vec*data.scale, data.batch))
        loss_all += loss * dos_out.size(0)
        loss.backward()
        optimizer.step()
        count = count + dos_out.size(0)
    loss_all = loss_all / count
    return loss_all


##Evaluation step, runs model in eval mode
def evaluate(model, loader, loss = "mse_loss"):
    if loss != "mse_loss":
        print("Not implemented")
        return None
    model.eval()
    model.cuda()
    model.requires_grad_(False)
    loss_all = 0
    count = 0
    for data in loader:
        data.cuda()
        dos_out = model(data)
        loss = torch.nn.functional.mse_loss(global_mean_pool(dos_out, data.batch), global_add_pool(data.dos_vec*data.scale, data.batch))
        loss_all += loss*dos_out.size(0)
        count = count + dos_out.size(0)
    model.requires_grad_(True)
    return loss_all/count

def eval_plot(model, loader, dir):
    model.eval()
    model.cuda()
    model.requires_grad_(False)
    loss_all = torch.tensor([0], dtype=torch.float32).cuda()
    for batch_idx, batch in enumerate(loader):
        batch.cuda()
        dos_out = model(batch)
        dos_list = global_mean_pool(dos_out, batch.batch).detach().cpu().numpy()
        dos_data = global_add_pool(batch.dos_vec*batch.scale, batch.batch).detach().cpu().numpy()
        loss = F.mse_loss(global_mean_pool(dos_out, batch.batch), global_add_pool(batch.dos_vec*batch.scale, batch.batch), reduction="none")
        x = np.linspace(-10, 10, 400)
        loss_all = torch.concat([loss_all, torch.mean(loss, dim=1)])
        if batch_idx <= 10 :
            for i, data in enumerate(dos_list):
                plt.clf()
                plt.plot(x, data, c="b", label="predicted DOS")
                plt.scatter(x, dos_data[i], c="r", label="DOS data")
                plt.tight_layout()
                plt.savefig(f"./{dir}/{batch_idx}_{i}.svg")
            print(batch_idx, batch.mat, flush=True)
            print(torch.mean(loss, dim=1).detach().cpu().numpy())
        plt.clf()
        plt.hist(loss_all[1:].detach().cpu().numpy(), bins=30)
        plt.xlim(0, 0.15)
        plt.xlabel("MSE loss")
        #plt.ylabel("# of data")
        plt.savefig(f"./{dir}/loss.svg")