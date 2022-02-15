import numpy as np
import h5py
import torch
import os
import torch.nn as nn
import random
import scipy.io as scio

from tqdm import tqdm
from Model_define_pytorch import AutoEncoder, DatasetFolder, NMSE_cuda, NMSELoss



# Parameters for training
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


SEED = 42
seed_everything(SEED)

batch_size = 256
epochs = 1000
learning_rate = 1e-3  # bigger to train faster
num_workers = 4
print_freq = 100
train_test_ratio = 0.8
# parameters for data
feedback_bits = 512
img_height = 16
img_width = 32
img_channels = 2

# Model construction
model = AutoEncoder(feedback_bits)
if len(gpu_list.split(',')) > 1:
    model = torch.nn.DataParallel(model).cuda()  # model.module
else:
    model = model.cuda()
model.encoder.quantization = False
model.decoder.dequantization = False

criterion = NMSELoss(reduction='mean')  # nn.MSELoss()
criterion_test = NMSELoss(reduction='sum')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=1, eta_min=5e-6)

# Data loading
data_load_address = './data'
mat = scio.loadmat(data_load_address+'/Htrain.mat')
x_train = mat['H_train']  # shape=8000*126*128*2

x_train = np.transpose(x_train.astype('float32'),[0,3,1,2])
print(np.shape(x_train))
print(x_train)
mat = scio.loadmat(data_load_address+'/Htest.mat')
x_test = mat['H_test']  # shape=2000*126*128*2

x_test = np.transpose(x_test.astype('float32'),[0,3,1,2])
print(np.shape(x_test))

# # 选500个测试集样本放到训练集
# x_train = np.concatenate((x_train, x_test[:500]))
# x_test = x_test[500:]

# dataLoader for training
train_dataset = DatasetFolder(x_train,'train')
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

# dataLoader for training
test_dataset = DatasetFolder(x_test)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

best_loss = 1
for epoch in tqdm(range(epochs)):
    # if epoch == 300:
    #     optimizer.param_groups[0]['lr'] = learning_rate * 0.3
    print('\n========================')
    print('lr:%.4e' % optimizer.param_groups[0]['lr'])
    # model training
    model.train()


    train_loss = 0
    for i, input in enumerate(train_loader):
        input = input.cuda()
        output = model(input)

        loss = criterion(output, input)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss}\t'.format(
                epoch, i, len(train_loader), loss=loss.item()))
    lr_scheduler.step()

    model.eval()

    train_loss = train_loss / len(train_loader)


    total_loss = 0
    with torch.no_grad():
        for i, input in enumerate(test_loader):
            # convert numpy to Tensor
            input = input.cuda()
            output = model(input)
            total_loss += criterion_test(output, input).item()
        average_loss = total_loss / len(test_dataset)

        print('NMSE: ' + str(average_loss))
        if average_loss < best_loss:
            # model save
            # save encoder
            modelSave1 = './Modelsave/encoder.pth.tar'
            try:
                torch.save({'state_dict': model.encoder.state_dict(), }, modelSave1)
            except:
                torch.save({'state_dict': model.module.encoder.state_dict(), }, modelSave1)
            # save decoder
            modelSave2 = './Modelsave/decoder.pth.tar'
            try:
                torch.save({'state_dict': model.decoder.state_dict(), }, modelSave2)
            except:
                torch.save({'state_dict': model.module.decoder.state_dict(), }, modelSave2)
            print('Model saved!')
            best_loss = average_loss

print('Best NMSE: ' + str(best_loss))