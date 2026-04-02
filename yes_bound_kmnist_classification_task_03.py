# Adding gaussian noise option, as well as "green patience" halting

import numpy as np
import torch as tc
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from IPython.display import clear_output
import torch.optim.lr_scheduler as lr_scheduler

import openml, numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def add_gaussian_noise(x, sigma=0.2):
    """
    x: tensor (H,W) in [0,1]
    sigma: std dev of Gaussian noise (in normalized intensity units)
    """
    n = tc.randn_like(x) * sigma
    return tc.clamp(x + n, 0.0, 1.0)

ds = openml.datasets.get_dataset(41982)  # Kuzushiji-MNIST :contentReference[oaicite:1]{index=1}
X, y, _, _ = ds.get_data(dataset_format="array", target=ds.default_target_attribute)
y = np.array(y)
if y.dtype.kind in ("U", "S", "O"):
    y = y.astype(int)
else:
    y = y.astype(np.int64)

X = X.astype(np.float32)
y = y.astype(np.int64)

imgs = X.reshape(-1, 28, 28)        # (70000, 28, 28)
labels = y                          # (70000,)

# Make a reproducible 60k/10k split
sss = StratifiedShuffleSplit(n_splits=1, test_size=10000, random_state=0)
train_idx, test_idx = next(sss.split(imgs, labels))
train_imgs, train_labels = imgs[train_idx], labels[train_idx]
test_imgs,  test_labels  = imgs[test_idx],  labels[test_idx]



class struct():
    '''
    an empty class to use structure type variable
    '''
    pass



class KMNIST_20x20_Noisy_Z(Dataset):
    def __init__(self, imgs_np, labels_np, n=5000, resolution=20,
                 noise_type=None, noise_level=0.2, noise_p=0.02,
                 noise_on_train_only=False, is_train=True):
        if n is not None:
            imgs_np = imgs_np[:n]
            labels_np = labels_np[:n]

        imgs = tc.tensor(imgs_np, dtype=tc.float32) / 255.0   # (n,28,28)
        labels = tc.tensor(labels_np, dtype=tc.long)          # (n,)

        self.x = F.interpolate(imgs.unsqueeze(1), size=(resolution, resolution)).squeeze(1)  # (n,res,res)
        self.y = F.one_hot(labels, num_classes=10).to(tc.float32)                             # (n,10)

        z_vecs = tc.zeros((labels.shape[0], resolution, 1), dtype=tc.float32)
        z_vecs[tc.arange(labels.shape[0]), labels, 0] = 1.0
        self.z = z_vecs @ z_vecs.transpose(1, 2)  # (n,res,res)

        # Noise config
        self.noise_type = noise_type              # None | "gaussian" | "saltpepper" | "speckle"
        self.noise_level = noise_level            # used by gaussian/speckle
        self.noise_p = noise_p                    # used by saltpepper
        self.noise_on_train_only = noise_on_train_only
        self.is_train = is_train

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, ix):
        x = self.x[ix]
        y = self.y[ix]
        z = self.z[ix]

        do_noise = (self.noise_type is not None) and (not self.noise_on_train_only or self.is_train)
        if do_noise:
            if self.noise_type == "gaussian":
                x = add_gaussian_noise(x, sigma=self.noise_level)
            else:
                raise ValueError(f"Unknown noise_type: {self.noise_type}")

        return x, y, z


class KMNIST_20x20_Z(Dataset):
    def __init__(self, imgs_np, labels_np, n=5000, resolution=20):
        if n is not None:
            imgs_np = imgs_np[:n]
            labels_np = labels_np[:n]

        imgs = tc.tensor(imgs_np, dtype=tc.float32) / 255.0   # (n,28,28)
        labels = tc.tensor(labels_np, dtype=tc.long)          # (n,)

        self.x = F.interpolate(imgs.unsqueeze(1), size=(resolution, resolution)).squeeze(1)  # (n,20,20)
        self.y = F.one_hot(labels, num_classes=10).to(tc.float32)                             # (n,10)

        z_vecs = tc.zeros((labels.shape[0], resolution, 1), dtype=tc.float32)
        z_vecs[tc.arange(labels.shape[0]), labels, 0] = 1.0
        self.z = z_vecs @ z_vecs.transpose(1, 2)  # (n,20,20)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, ix):
        return self.x[ix], self.y[ix], self.z[ix]



train_ds = KMNIST_20x20_Noisy_Z(train_imgs, train_labels, n=5000, resolution=20,
                          noise_type="gaussian", noise_level=0.15,
                          noise_on_train_only=True, is_train=True)

test_ds  = KMNIST_20x20_Noisy_Z(test_imgs,  test_labels,  n=5000, resolution=20,
                          noise_type="gaussian", noise_level=0.15,
                          noise_on_train_only=True, is_train=False)

x, y, z = train_ds[0]
print(x.shape, y.shape, z.shape)
print("label:", int(y.argmax()))
print("z diag sum:", float(tc.diag(z).sum()))


# define the model
class classification_model(nn.Module):
    def __init__(self,params):
        super(classification_model, self).__init__()
        self.K = params.Layers
        Layers = params.Layers
        m = params.m
        self.W = nn.ModuleList([nn.Linear(m,m,bias=params.bias) for _ in range(Layers)])
        self.R = nn.ReLU()
        #self.S = nn.Sigmoid()
    def forward(self,x):
        per_out = []
        z = x
        for k in range(self.K):
          if k != self.K-1:
            z = self.R(self.W[k](z))
          else:
            #z = self.S(self.W[k](z))
            z = self.W[k](z)
          per_out.append(z)
        return z, per_out
# define training model
def train(f, params, train_dl, test_dl):
    # [train_dl: dataloader for training, test_dl: dataloader for test, f: model]
    # optimization
    opt = tc.optim.SGD(f.parameters(), lr=params.lr)
    if params.schedule:
      scheduler = lr_scheduler.StepLR(opt, step_size=50, gamma=0.7)
    #L = nn.CrossEntropyLoss()
    L = nn.MSELoss(reduction='sum')
    n_epochs = params.NUM_EPOCHS
    Layers = params.Layers
    BATCH_SIZE = params.BATCH_SIZE
    num_train = len(train_dl)*BATCH_SIZE
    num_test = len(test_dl)*BATCH_SIZE
    bias_cond = params.bias
    m = params.m
    X = params.Xdata
    Y = params.Ydata
    # create a class with tensor size (m) for decoder part
    num_digits = 10
    #one_vec = tc.ones((1,m), dtype = tc.float32)
    digit_tensor_class = tc.zeros((num_digits,m), dtype = tc.float32)
    for i in range(num_digits):
      zero_vec = tc.zeros((20,1), dtype = tc.float32)
      zero_vec[i] = 1
      digit_tensor_class[i,:] = tc.matmul(zero_vec,zero_vec.transpose(0,1)).view(-1)
      #digit_tensor_class[i,:] = ((i+1)/10)*one_vec
    # create a desired class based on one number from the (train/test) dataset
    # train
    digit_number_class_train = tc.zeros((num_train), dtype = tc.float32)
    Y_digits_train = tc.zeros((num_train,num_digits), dtype = tc.float32)
    j = 0
    for (x, y, z) in train_dl:
      y = y.type(tc.float32)
      Y_digits_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE,:] = y
      j+=1
    for i in range(num_train):
      digit_number_class_train[i] = tc.nonzero(Y_digits_train[i] == 1.).item()
    # test
    digit_number_class_test = tc.zeros((num_test), dtype = tc.float32)
    Y_digits_test = tc.zeros((num_test,num_digits), dtype = tc.float32)
    j = 0
    for (x, y, z) in test_dl:
      y = y.type(tc.float32)
      Y_digits_test[j*BATCH_SIZE:(j+1)*BATCH_SIZE,:] = y
      j+=1
    for i in range(num_test):
      digit_number_class_test[i] = tc.nonzero(Y_digits_test[i] == 1.).item()
    # computing YES-0 bound for training
    if not bias_cond:
      # with no bias
      W_k = tc.matmul(Y,tc.linalg.pinv(X))
      Y_k = tc.nn.functional.relu(tc.matmul(W_k,X))
      for k in range(Layers-1):
        W_k = tc.matmul(Y,tc.linalg.pinv(Y_k))
        if k != Layers-2:
          Y_k = tc.nn.functional.relu(tc.matmul(W_k,Y_k))
        else:
          #Y_k = tc.nn.functional.sigmoid(tc.matmul(W_k,Y_k))
          Y_k = tc.matmul(W_k,Y_k)
    else:
      # with bias
      Y_k = X
      for k in range(Layers):
       Y_t = tc.vstack((Y_k, tc.ones(1,num_train)))
       W_k = tc.matmul(Y,tc.linalg.pinv(Y_t))
       if k != Layers-1:
        Y_k = tc.nn.functional.relu(tc.matmul(W_k,Y_t))
       else:
        #Y_k = tc.nn.functional.sigmoid(tc.matmul(W_k,Y_t))
        Y_k = tc.matmul(W_k,Y_t)
    s = 0
    for i in range(len(train_dl)):
      s+=L(Y_k.transpose(0,1)[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:],Y.transpose(0,1)[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]).item()
    YES_0_bound = s/(num_train)
    YES_0_bound = tc.tensor([YES_0_bound])
    # Define colors
    spring_green = "#00FF7F"  # Bright and vibrant green
    golden_color = "#FFD700"   # Golden color for the yellow area
    pleasant_red = "#FFB6C1"   # Light pinkish red for the red area
    # Train model
    YES_k_bounds = tc.zeros((Layers-1,n_epochs), dtype = tc.float32)
    train_loss = []
    test_loss = []
    success_rate_train = []
    success_rate_test = []
    green_streak = 0 #ER, 03012026 0016
    
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        for (x, y, z) in train_dl:
          # Update the weights of the network
          x = x.type(tc.float32)
          z = z.type(tc.float32)
          x = x.view(-1,m)
          z = z.view(-1,m)
          opt.zero_grad()
          z_hat,_ = f(x)
          loss_value = L(z_hat, z)
          loss_value.backward()
          opt.step()
        if params.schedule:
          scheduler.step()
        current_lr = opt.param_groups[0]['lr']
        print(f"Epoch {epoch}, Learning Rate: {current_lr}")
        with tc.no_grad():
          All_out = []
          for i in range(Layers):
            wholeData = tc.zeros((num_train,m), dtype = tc.float32)
            j = 0
            for (x, y, z) in train_dl:
              x = x.type(tc.float32)
              x = x.view(-1,m)
              _,per_out = f(x)
              wholeData[BATCH_SIZE*j:BATCH_SIZE*(j+1),:] = per_out[i]
              j+=1
            All_out.append(wholeData)
          # general code for generating YES-k bounds
          if Layers != 1:
            layer_indces = tc.arange(0,Layers-1)
            for k in range(1,Layers):
              layer_combinations = tc.combinations(layer_indces, r=k)
              temp_k_bound = tc.zeros(layer_combinations.shape[0], dtype = tc.float32)
              for i in range(layer_combinations.shape[0]):
                l = 0
                Y_sigma = []
                for j in range(k):
                  Y_sigma.append(All_out[layer_combinations[i][j].item()].transpose(0,1))
                Y_k = X
                for j in range(k):
                  while l <= layer_combinations[i][j].item():
                    if not bias_cond:
                      # with no bias
                      W_k = tc.matmul(Y_sigma[j],tc.linalg.pinv(Y_k))
                      Y_k = tc.nn.functional.relu(tc.matmul(W_k,Y_k))
                    else:
                      # with bias
                      Y_t = tc.vstack((Y_k, tc.ones(1,num_train)))
                      W_k = tc.matmul(Y_sigma[j],tc.linalg.pinv(Y_t))
                      Y_k = tc.nn.functional.relu(tc.matmul(W_k,Y_t))
                    l+=1
                for cnt in range(Layers-l):
                  if not bias_cond:
                    # with no bias
                    W_k = tc.matmul(Y,tc.linalg.pinv(Y_k))
                    if cnt != Layers-l-1:
                      Y_k = tc.nn.functional.relu(tc.matmul(W_k,Y_k))
                    else:
                      #Y_k = tc.nn.functional.sigmoid(tc.matmul(W_k,Y_k))
                      Y_k = tc.matmul(W_k,Y_k)
                  else:
                    # with bias
                    Y_t = tc.vstack((Y_k, tc.ones(1,num_train)))
                    W_k = tc.matmul(Y,tc.linalg.pinv(Y_t))
                    if cnt != Layers-l-1:
                      Y_k = tc.nn.functional.relu(tc.matmul(W_k,Y_t))
                    else:
                      #Y_k = tc.nn.functional.sigmoid(tc.matmul(W_k,Y_t))
                      Y_k = tc.matmul(W_k,Y_t)
                s = 0
                for h in range(len(train_dl)):
                  s+=L(Y_k.transpose(0,1)[h*BATCH_SIZE:(h+1)*BATCH_SIZE,:],Y.transpose(0,1)[h*BATCH_SIZE:(h+1)*BATCH_SIZE,:]).item()
                temp_k_bound[i] = s/(num_train)
              #YES_k_bounds[k-1,epoch] = tc.min(temp_k_bound)
              if k == 1:
                YES_k_bounds[k-1,epoch] = tc.min(tc.cat((YES_0_bound,temp_k_bound)))
                if (epoch >= 1) and (YES_k_bounds[k-1,epoch]-YES_k_bounds[k-1,epoch-1] > 0):
                  YES_k_bounds[k-1,epoch] = YES_k_bounds[k-1,epoch-1]
              else:
                YES_k_bounds[k-1,epoch] = tc.min(tc.cat((temp_k_bound,tc.tensor([YES_k_bounds[k-2,epoch].item()]))))
                if (epoch >= 1) and (YES_k_bounds[k-1,epoch]-YES_k_bounds[k-1,epoch-1] > 0):
                  YES_k_bounds[k-1,epoch] = YES_k_bounds[k-1,epoch-1]
        if epoch%1==0:
          with tc.no_grad():
            # train error
            loss = 0
            for (x, y, z) in train_dl:
              x = x.type(tc.float32)
              z = z.type(tc.float32)
              x = x.view(-1,m)
              z = z.view(-1,m)
              z_hat,_ = f(x)
              loss += L(z_hat, z).item()
            loss /= (num_train)
            print(f'Training loss: {loss:.4f}')
            train_loss.append(loss)
            # test error
            loss = 0
            for (x, y, z) in test_dl:
              x = x.type(tc.float32)
              z = z.type(tc.float32)
              x = x.view(-1,m)
              z = z.view(-1,m)
              z_hat,_ = f(x)
              loss += L(z_hat, z).item()
            loss /= (num_test)
            print(f'Test loss: {loss:.4f}')
            test_loss.append(loss)
        # quantify digit identification rate for training
        if epoch%1==0:
          with tc.no_grad():
            # train
            j = 0
            Z = tc.zeros((num_train,m), dtype = tc.float32)
            for (x, y, z) in train_dl:
              x = x.type(tc.float32)
              x = x.view(-1,m)
              z_hat,_ = f(x)
              Z[BATCH_SIZE*j:BATCH_SIZE*(j+1),:] = z_hat
              j+=1
            # run the decoder for train
            num_correct = 0
            for i in range(num_train):
              z_i = Z[i,:]
              #mu_z_i = tc.mean(z_i)
              class_error = tc.zeros(num_digits, dtype = tc.float32)
              for j in range(num_digits):
                class_error[j] = tc.sum((z_i - digit_tensor_class[j,:]) ** 2)
                #mu_digit = tc.mean(digit_tensor_class[j,:])
                #class_error[j] = tc.abs(mu_z_i-mu_digit)
              digit_estimated = tc.argmin(class_error)
              digit_true = digit_number_class_train[i]
              if digit_estimated.item() == digit_true.item():
                num_correct += 1
            success_rate_train.append(num_correct/num_train)
            # test
            j = 0
            Z = tc.zeros((num_test,m), dtype = tc.float32)
            for (x, y, z) in test_dl:
              x = x.type(tc.float32)
              x = x.view(-1,m)
              z_hat,_ = f(x)
              Z[BATCH_SIZE*j:BATCH_SIZE*(j+1),:] = z_hat
              j+=1
            # run the decoder for test
            num_correct = 0
            for i in range(num_test):
              z_i = Z[i,:]
              #mu_z_i = tc.mean(z_i)
              class_error = tc.zeros(num_digits, dtype = tc.float32)
              for j in range(num_digits):
                class_error[j] = tc.sum((z_i - digit_tensor_class[j,:]) ** 2)
                #mu_digit = tc.mean(digit_tensor_class[j,:])
                #class_error[j] = tc.abs(mu_z_i-mu_digit)
              digit_estimated = tc.argmin(class_error)
              digit_true = digit_number_class_test[i]
              if digit_estimated.item() == digit_true.item():
                num_correct += 1
            success_rate_test.append(num_correct/num_test)
        #if success_rate_test[-1] >= 0.9 or epoch > 300:
            #print(f"Stopping early at epoch {epoch} — test accuracy reached {success_rate_test[-1]:.3f}")
            #break

        
        # real-time plot
        YES_plot = YES_k_bounds[Layers-2,0:epoch+1]

        # --- stop when we enter (and stay in) the green region ---
        if getattr(params, "stop_on_green", False) and len(train_loss) > 0:
            # current epoch's values
            current_train_loss = train_loss[-1]
            current_yes_k = float(YES_plot[-1].item())  # last point of YES_k curve

            in_green = (current_train_loss <= current_yes_k)

            if epoch >= getattr(params, "green_min_epoch", 0) and in_green:
                green_streak += 1
            else:
                green_streak = 0

            if green_streak >= getattr(params, "green_patience", 1):
                print(
                    f"Stopping early at epoch {epoch} — stayed in GREEN for "
                    f"{green_streak} epochs (train_loss={current_train_loss:.6f} <= YES_k={current_yes_k:.6f})"
                )
                break
        # ---------------------------------------------------------
        
        clear_output(wait=True)
        if params.in_log:
          fig1, ax1 = plt.subplots()
          # Plot the data with different colors and line styles
          ax1.plot([10*np.log10(YES_0_bound.item())] * len(YES_plot), color='red', linestyle='-', linewidth=2)
          ax1.plot(10*np.log10(YES_plot), color='orange', linestyle='-.', linewidth=2)
          ax1.plot(10*np.log10(train_loss), color='navy', linestyle='-', linewidth=2, label='Training Loss')
          #
          ymin, ymax = ax1.get_ylim()
          # Set the background color regions using fill_between for non-straight regions
          ax1.fill_between(np.arange(0, len(YES_plot), 1), 10*np.log10(YES_0_bound.item()), ymax, facecolor=pleasant_red, alpha=0.4, hatch='xx', edgecolor='red', label='Ineffective Training')
          ax1.fill_between(np.arange(0, len(YES_plot), 1), 10*np.log10(YES_plot), 10*np.log10(YES_0_bound.item()), facecolor=golden_color, alpha=0.4, hatch='//', edgecolor='gold', label='Caution')
          ax1.fill_between(np.arange(0, len(YES_plot), 1), ymin, 10*np.log10(YES_plot), facecolor=spring_green, alpha=0.4, label='Effective Training')
          ax1.set_ylabel('MSE (dB)')
          ax1.set_xlabel('Epoch number')
          #plt.legend(loc='center', bbox_to_anchor=(0.75, 0.85))
          ax1.legend(loc='center', bbox_to_anchor=(0.85, 0.8), fontsize='small', borderpad=0.2, labelspacing=0.3)
          ax1.grid(alpha=0.7)
          plt.show()
        else:
          fig1, ax1 = plt.subplots()
          # Plot the data with different colors and line styles
          ax1.plot([YES_0_bound.item()] * len(YES_plot), color='red', linestyle='-', linewidth=2)
          ax1.plot(YES_plot, color='orange', linestyle='-.', linewidth=2)
          ax1.plot(train_loss, color='navy', linestyle='-', linewidth=2, label='Training Loss')
          #
          ymin, ymax = ax1.get_ylim()
          # Set the background color regions using fill_between for non-straight regions
          ax1.fill_between(np.arange(0, len(YES_plot), 1), YES_0_bound.item(), ymax, facecolor=pleasant_red, alpha=0.4, hatch='xx', edgecolor='red', label='Ineffective Training')
          ax1.fill_between(np.arange(0, len(YES_plot), 1), YES_plot, YES_0_bound.item(), facecolor=golden_color, alpha=0.4, hatch='//', edgecolor='gold', label='Caution')
          ax1.fill_between(np.arange(0, len(YES_plot), 1), ymin, YES_plot, facecolor=spring_green, alpha=0.4, label='Effective Training')
          ax1.set_ylabel('MSE')
          ax1.set_xlabel('Epoch number')
          #plt.legend(loc='center', bbox_to_anchor=(0.75, 0.85))
          ax1.legend(loc='center', bbox_to_anchor=(0.85, 0.8), fontsize='small', borderpad=0.2, labelspacing=0.3)
          ax1.grid(alpha=0.7)
          plt.show()
        fig2, ax2 = plt.subplots()
        ax2.plot(success_rate_train, color='navy', linestyle='-', linewidth=2, label='Digit Identification Rate (Train)')
        ax2.set_ylabel('Digit Identification Rate (Train)')
        ax2.set_xlabel('Epoch number')
        ax2.grid(alpha=0.7)
        plt.show()
        fig3, ax3 = plt.subplots()
        ax3.plot(success_rate_test, color='orange', linestyle='-', linewidth=2, label='Digit Identification Rate (Test)')
        ax3.set_ylabel('Digit Identification Rate (Test)')
        ax3.set_xlabel('Epoch number')
        ax3.grid(alpha=0.7)
        plt.show()
        
    return train_loss, test_loss, YES_0_bound, YES_k_bounds, success_rate_train, success_rate_test

# change the batch size here
params = struct()
params.BATCH_SIZE = 100
#train_dl = DataLoader(train_ds, batch_size=params.BATCH_SIZE)
#test_dl = DataLoader(test_ds, batch_size=params.BATCH_SIZE)

train_dl = DataLoader(train_ds, batch_size=params.BATCH_SIZE, shuffle=True, drop_last=True)
test_dl  = DataLoader(test_ds,  batch_size=params.BATCH_SIZE, shuffle=False, drop_last=True)


BATCH_SIZE = params.BATCH_SIZE
num_train = len(train_dl)*BATCH_SIZE
params.m = 20**2
m = params.m
Y = tc.zeros((num_train,m), dtype = tc.float32)
X = tc.zeros((num_train,m), dtype = tc.float32)
j = 0
for (x, y, z) in train_dl:
  x = x.type(tc.float32)
  z = z.type(tc.float32)
  X[BATCH_SIZE*j:BATCH_SIZE*(j+1),:] = x.view(-1,m)
  Y[BATCH_SIZE*j:BATCH_SIZE*(j+1),:] = z.view(-1,m)
  j+=1
X = X.transpose(0,1)
Y = Y.transpose(0,1)
params.Ydata = Y
params.Xdata = X

params.Layers = 5
params.NUM_EPOCHS = 500
params.lr = 1e-4
params.bias = True
params.schedule = True
params.in_log = False

model = classification_model(params)
params.train_loss, params.test_loss, params.YES_0_bound, params.YES_k_bounds, params.success_rate_train, params.success_rate_test = train(model, params, train_dl, test_dl)

params.stop_on_green = True
params.green_patience = 5     # require green for 5 consecutive epochs
params.green_min_epoch = 10   # don't allow stopping too early

in_log = params.in_log

# ER 1 BEGIN
YES_0_bound = params.YES_0_bound

T = len(params.train_loss)  # epochs actually completed due to early stopping

YES_plot = params.YES_k_bounds[params.Layers-2, :T]
train_loss = params.train_loss[:T]
success_rate_train = params.success_rate_train[:T]
success_rate_test  = params.success_rate_test[:T]

zoom_up = T
zoom_indices = np.arange(0, zoom_up)

YES_plot_box = YES_plot.numpy()
train_loss_box = np.array(train_loss)
# ER 1 END


# Define colors
spring_green = "#00FF7F"  # Bright and vibrant green
golden_color = "#FFD700"  # Golden color for the yellow area
pleasant_red = "#FFB6C1"  # Light pinkish red for the red area
if in_log:
  fig, ax = plt.subplots()
  # Plot the data with different colors and line styles
  plt.plot(np.arange(0, zoom_up, 1), [10*np.log10(YES_0_bound.item())] * zoom_up, color='red', linestyle='-', linewidth=2, label='YES-0 Training Bound')
  plt.plot(np.arange(0, zoom_up, 1), 10*np.log10(YES_plot_box), color='green', linestyle='--', linewidth=2, label='YES-4 Training Bound')
  plt.plot(np.arange(0, zoom_up, 1), 10*np.log10(train_loss_box), color='navy', linestyle='-', linewidth=2, label='Training Loss')
  # get the y-axis limits
  ymin, ymax = ax.get_ylim()
  # Set the background color regions using fill_between for non-straight regions
  plt.fill_between(np.arange(0, zoom_up, 1), 10*np.log10(YES_0_bound.item()), ymax, facecolor=pleasant_red, alpha=0.4, hatch='xx', edgecolor='red', label='Ineffective Training')
  plt.fill_between(np.arange(0, zoom_up, 1), 10*np.log10(YES_plot_box), 10*np.log10(YES_0_bound.item()), facecolor=golden_color, alpha=0.4, hatch='//', edgecolor='gold', label='YES Cloud (Caution)')
  plt.fill_between(np.arange(0, zoom_up, 1), ymin, 10*np.log10(YES_plot_box), facecolor=spring_green, alpha=0.4, label='Effective Training')
  ###
  plt.tight_layout()
  plt.ylabel('MSE(dB)',fontsize=13)
  plt.xlabel('Epoch Number',fontsize=13)
  plt.title('YES Cloud and Training Loss Progression',fontsize=15)
  plt.legend(loc='center', bbox_to_anchor=(0.825, 0.78), fontsize='small', borderpad=0.2, labelspacing=0.3)
  plt.grid(alpha=0.7)
else:
  fig, ax = plt.subplots()
  # Plot the data with different colors and line styles
  plt.plot(np.arange(0, zoom_up, 1), [YES_0_bound.item()] * zoom_up, color='red', linestyle='-', linewidth=2, label='YES-0 Training Bound')
  plt.plot(np.arange(0, zoom_up, 1), YES_plot_box, color='green', linestyle='--', linewidth=2, label='YES-4 Training Bound')
  plt.plot(np.arange(0, zoom_up, 1), train_loss_box, color='navy', linestyle='-', linewidth=2, label='Training Loss')
  # get the y-axis limits
  ymin, ymax = ax.get_ylim()
  # Set the background color regions using fill_between for non-straight regions
  plt.fill_between(np.arange(0, zoom_up, 1), YES_0_bound.item(), ymax, facecolor=pleasant_red, alpha=0.4, hatch='xx', edgecolor='red', label='Ineffective Training')
  plt.fill_between(np.arange(0, zoom_up, 1), YES_plot_box, YES_0_bound.item(), facecolor=golden_color, alpha=0.4, hatch='//', edgecolor='gold', label='YES Cloud (Caution)')
  plt.fill_between(np.arange(0, zoom_up, 1), ymin, YES_plot_box, facecolor=spring_green, alpha=0.4, label='Effective Training')
  ###
  plt.tight_layout()
  plt.ylabel('MSE',fontsize=13)
  plt.xlabel('Epoch Number',fontsize=13)
  plt.title('YES Cloud and Training Loss Progression',fontsize=15)
  plt.legend(loc='center', bbox_to_anchor=(0.825, 0.78), fontsize='small', borderpad=0.2, labelspacing=0.3)
  plt.grid(alpha=0.7)
#plt.savefig("/content/gdrive/My Drive/ICLR_2025_YES_bound_paper/MNIST_cloud_newEnDecoder_1.png", format="png", bbox_inches="tight")

above_YES_0 = train_loss_box > YES_0_bound.item()
between_YES_k_and_YES_0 = (train_loss_box <= YES_0_bound.item()) & (train_loss_box > YES_plot_box)
below_YES_k = train_loss_box <= YES_plot_box
# Function to get contiguous regions
def get_contiguous_regions(mask):
    # Find the indices where the mask changes value
    d = np.diff(mask)
    idx, = np.nonzero(d)
    idx += 1

    if mask[0]:
        idx = np.r_[0, idx]
    if mask[-1]:
        idx = np.r_[idx, mask.size]
    idx.shape = (-1,2)
    return idx

success_rate_train = params.success_rate_train
fig, ax = plt.subplots()
ax.plot(success_rate_train[0:zoom_up], color='navy', linestyle='-', linewidth=2, label='Digit Identification Rate (Train)')
ax.set_ylabel('Digit Identification Rate (Train)')
ax.set_xlabel('Epoch Number')
ax.grid(alpha=0.7)
ymin, ymax = ax.get_ylim()
# Fill regions where train_loss_box > YES_0_bound
regions_above_YES_0 = get_contiguous_regions(above_YES_0)
for start, end in regions_above_YES_0:
    ax.fill_between(zoom_indices[start:end], ymin, ymax,
                     facecolor=pleasant_red, alpha=0.4, hatch='xx', edgecolor='red')
# Fill regions where YES_plot_box < train_loss_box <= YES_0_bound
regions_between = get_contiguous_regions(between_YES_k_and_YES_0)
for start, end in regions_between:
    ax.fill_between(zoom_indices[start:end], ymin, ymax,
                     facecolor=golden_color, alpha=0.4, hatch='//', edgecolor='gold')
# Fill regions where train_loss_box <= YES_plot_box
regions_below_YES_k = get_contiguous_regions(below_YES_k)
for start, end in regions_below_YES_k:
    ax.fill_between(zoom_indices[start:end], ymin, ymax,
                     facecolor=spring_green, alpha=0.4)
# (ER) plt.savefig("/content/gdrive/My Drive/ICLR_2025_YES_bound_paper/MNIST_success_rate_newEnDecoder_1.png", format="png", bbox_inches="tight")

success_rate_test = params.success_rate_test
fig, ax = plt.subplots()
ax.plot(success_rate_test[0:zoom_up], color='orange', linestyle='-', linewidth=2, label='Digit Identification Rate (Test)')
ax.set_ylabel('Digit Identification Rate (Test)')
ax.set_xlabel('Epoch Number')
ax.grid(alpha=0.7)
ymin, ymax = ax.get_ylim()
# Fill regions where train_loss_box > YES_0_bound
regions_above_YES_0 = get_contiguous_regions(above_YES_0)
for start, end in regions_above_YES_0:
    ax.fill_between(zoom_indices[start:end], ymin, ymax,
                     facecolor=pleasant_red, alpha=0.4, hatch='xx', edgecolor='red')
# Fill regions where YES_plot_box < train_loss_box <= YES_0_bound
regions_between = get_contiguous_regions(between_YES_k_and_YES_0)
for start, end in regions_between:
    ax.fill_between(zoom_indices[start:end], ymin, ymax,
                     facecolor=golden_color, alpha=0.4, hatch='//', edgecolor='gold')
# Fill regions where train_loss_box <= YES_plot_box
regions_below_YES_k = get_contiguous_regions(below_YES_k)
for start, end in regions_below_YES_k:
    ax.fill_between(zoom_indices[start:end], ymin, ymax,
                     facecolor=spring_green, alpha=0.4)
# (ER) plt.savefig("/content/gdrive/My Drive/ICLR_2025_YES_bound_paper/MNIST_success_rate_test_newEnDecoder_1.png", format="png", bbox_inches="tight")


KMNIST_ROMAJI = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]

def kmnist_name(k):
    return KMNIST_ROMAJI[int(k)]



print('Show test digits + predictions')
def build_digit_tensor_class(m=400, num_digits=10):
    digit_tensor_class = tc.zeros((num_digits, m), dtype=tc.float32)
    for i in range(num_digits):
        zero_vec = tc.zeros((20, 1), dtype=tc.float32)
        zero_vec[i] = 1.0
        digit_tensor_class[i, :] = (zero_vec @ zero_vec.T).view(-1)
    return digit_tensor_class

@tc.no_grad()
def decode_digit_from_z(z_flat, digit_tensor_class):
    # z_flat: (m,)  digit_tensor_class: (10,m)
    # returns int 0..9
    diffs = digit_tensor_class - z_flat.unsqueeze(0)      # (10,m)
    class_error = tc.sum(diffs * diffs, dim=1)            # (10,)
    return int(tc.argmin(class_error).item())

@tc.no_grad()
def plot_test_predictions(model, test_dl, device="cpu", n_show=16):
    model.eval()
    digit_tensor_class = build_digit_tensor_class(m=20*20, num_digits=10).to(device)

    # Grab one batch
    x, y_onehot, z = next(iter(test_dl))
    x = x.to(device).float()                 # (B,20,20)
    y_onehot = y_onehot.to(device).float()   # (B,10)

    B = x.shape[0]
    n = min(n_show, B)

    # Forward pass
    x_flat = x.view(B, -1)                   # (B,400)
    z_hat, _ = model(x_flat)                 # (B,400)

    # Convert one-hot to digit labels
    y_true = tc.argmax(y_onehot, dim=1)      # (B,)

    # Decode predictions using the same decoder logic as training :contentReference[oaicite:2]{index=2}
    y_pred = []
    for i in range(n):
        y_pred.append(decode_digit_from_z(z_hat[i], digit_tensor_class))
    y_pred = tc.tensor(y_pred, device=device)

    # Plot in a grid
    grid = int(tc.ceil(tc.sqrt(tc.tensor(n))).item())
    fig, axes = plt.subplots(grid, grid, figsize=(2.5*grid, 2.5*grid))
    axes = axes.reshape(grid, grid)

    idx = 0
    for r in range(grid):
        for c in range(grid):
            ax = axes[r, c]
            ax.axis("off")
            if idx < n:
                ax.imshow(x[idx].detach().cpu().numpy(), cmap="gray")
                t = int(y_true[idx].item())
                p = int(y_pred[idx].item())
                #ax.set_title(f"true: {t}  pred: {p}", fontsize=10)
                ax.set_title(f"true: {kmnist_name(t)}  pred: {kmnist_name(p)}", fontsize=10)

            idx += 1

    plt.tight_layout()
    plt.show()

# If trained on GPU, pass device="cuda"
plot_test_predictions(model, test_dl, device="cpu", n_show=25)


