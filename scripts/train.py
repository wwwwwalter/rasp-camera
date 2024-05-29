import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import argparse
from networks import LightNet
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_dataset(data_path):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean = [0.5],std = [0.5])])
    train_data = DataLoader()
    test_data = DataLoader()

    return train_data, test_data
    

def calculate_loss(real_res, pre_res):
    loss = criterion(real_res,pre_res)
    return loss

def one_epoch(inp,lb):
    output = model(inp)
    loss = calculate_loss(lb,output)
    loss.backward()
    return output,loss


def main(args):

    model.train()

    for epoch in range(args.epoch):
        with tqdm(train_data) as pbar:
            for _, (images, labels) in enumerate(pbar):
                
                optimizer.zero_grad()
                output, loss = one_epoch(images, labels)
                optimizer.step()
                
                accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

        # Validation
        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for images, labels in test_data:
                output = model(images)
                val_loss += criterion(output, labels.to(device)).item()
                val_accuracy += (
                    (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                )
        val_loss /= len(test_data)
        val_accuracy /= len(test_data)

        # Update learning rate
        scheduler.step()


        print(
            f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("backbone", type=str, default="LightNet")
    parser.add_argument("savepath", type=str, default="./models")
    parser.add_argument("data",type=str,default="./data")
    parser.add_argument("optimizer", type=str, default="Adam")
    parser.add_argument("lossfun",type=str, default="CrossEntropy")
    parser.add_argument("lr", type=float, default=0.001)
    parser.add_argument("epoch", type=int, default=100)
    parser.add_argument("batchsize", type=int, default=16)
    parser.add_argument("randmoseed", type=int, default=2000)

    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, test_data = load_dataset()
    model = LightNet()
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    main()


