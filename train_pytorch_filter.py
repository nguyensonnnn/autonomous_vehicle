import os
import torch
import torch.nn as nn
import torch.optim as optim
import driving_data
import model_pytorch
import numpy as np
from torch.utils import tensorboard

LOGDIR = './save'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
L2NormConst = 0.001

model = model_pytorch.MyModel().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00008)

# create a summary writer to monitor loss
logs_path = './logs'
summary_writer = torch.utils.tensorboard.SummaryWriter(logs_path)

epochs = 50
batch_size = 100
train_loss=[]
validation_loss=[]

# train over the dataset about 30 times
for epoch in range(epochs):
    loss_value_sum=0
    loss_train_sum=0
    for i in range(int(driving_data.num_images / batch_size*0.8)):
        
        xs, ys = driving_data.LoadTrainBatch(batch_size)
        xs=np.array(xs)
        xs = torch.Tensor(xs).to(device)
        ys = torch.Tensor(ys).to(device)
        xs = xs.permute(0, 3, 1, 2)
        #print(xs.shape)

        optimizer.zero_grad()
        outputs = model(xs)
        loss = criterion(outputs, ys) + L2NormConst * sum(p.norm(2) for p in model.parameters())
        loss.backward()
        optimizer.step()
        loss_train_sum=loss_train_sum+loss
        if i%100==0:
            print("Epoch: %d, Train Loss: %g" % (epoch, loss))
    for i in range(int(driving_data.num_images / batch_size*0.2)):
        xs_val, ys_val = driving_data.LoadValBatch(batch_size)
        xs_val=np.array(xs_val)
        xs_val = torch.Tensor(xs_val).to(device)
        ys_val = torch.Tensor(ys_val).to(device)
        xs_val = xs_val.permute(0, 3, 1, 2)
        loss_value = criterion(model(xs_val), ys_val).item()
        loss_value_sum=loss_value_sum+loss_value
        if i%25==0:
          
            print("Epoch: %d, Validation Loss: %g" % (epoch, loss_value))
 
        # write loss to Tensorboard
        summary_writer.add_scalar("loss", loss.item(), epoch * driving_data.num_images / batch_size + i)

        if i % batch_size == 0:
            if not os.path.exists(LOGDIR):
                os.makedirs(LOGDIR)
            checkpoint_path = os.path.join(LOGDIR, "model_seg_full_ver13.ckpt")
            torch.save(model.state_dict(), checkpoint_path)
    print("Model saved in file: %s" % checkpoint_path)
    
    loss_train_average=loss_train_sum/(driving_data.num_images / batch_size)
    print("Epoch: %d, Train Loss: %g" % (epoch, loss_train_average))
    #train_loss.append(loss_train_average)
    #print("the train loss: ")
    #print(train_loss)
    
    loss_value_average= loss_value_sum/((driving_data.num_images / batch_size)*0.2)
    print("Epoch: %d, Loss: %g" % (epoch, loss_value_average))
    #validation_loss.append(loss_value_average)
    #print("the validation loss: ")
    #print(validation_loss)
    print()
