import torch
from data._dataset import Dataset
from model._dos_net import DosNet
from training import train_val_func
import joblib
#from torchsummary import summary


def main():
    target_file = "/work/jxliu/e3nn/DOS_net/DosNet/data/targets.csv"
    structure_path = "/work/jxliu/e3nn/DOS_net/DosNet/data/structure/"
    post_path = "/work/jxliu/e3nn/DOS_net/DosNet/data/preprocessed/"
    dataset = Dataset()
    dataset.load_from_dir(target_file , structure_path, post_path)
    train_dataloader, valid_dataloader = dataset.get_data_Loader(20)
    n = 118
    n_basis = 400
    layer_n_list = [int((n+n_basis)/2), int((n+n_basis)), n_basis]
    layer_l_list = [3, 4, 0]
    qk_irreps = f"{n}x0e+{n}x1o+{n}x2o"
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

    model = DosNet(config)
    model.cuda()
    #summary(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=60,    
                verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    #loss = ""
    epochs = 500
    for epoch in range(epochs):
        train_loss = train_val_func.train(model, optimizer, train_dataloader)
        val_loss = train_val_func.evaluate(model, valid_dataloader)
        scheduler.step(train_loss)
        if epoch%10 == 0:
            print("epoch: ", epoch, "train loss ", train_loss, "val loss ", val_loss, flush=True)
            torch.save(model.state_dict(), f'./checkpoints/model_{epoch}.pth')
            torch.save(optimizer.state_dict(), f'./checkpoints/optimizer_{epoch}.pth')

if __name__ == "__main__":
    main()

