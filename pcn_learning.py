# code for pcn learning experiments
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.autograd.functional as taf
from copy import deepcopy
import torch.nn.functional as F

BATCH_SIZE = 64



transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)

dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)

def get_mnist_dataset(batch_size):

  transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

  trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
  valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
  valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
  return trainloader, valloader

print(images.shape)
print(images.view(images.shape[0],-1).shape)

def accuracy(out, labels):
  labels = labels.cpu()
  with torch.no_grad():
    maxes = torch.argmax(out.detach().cpu(), dim=1)
    corrects = maxes == labels
    return torch.sum(corrects) / len(corrects)

input_size = 784
hidden_sizes = [128, 64]
output_size = 10
batch_size  = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE IS : ", DEVICE)

model2 = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

class NN_model(nn.Module):
  def __init__(self,input_size, hidden_sizes, output_size, batch_size, device="cpu"):
    super(NN_model, self).__init__()
    self.input_size = input_size
    self.hidden_sizes = hidden_sizes
    self.output_size = output_size
    self.batch_size = batch_size
    self.device = device
    self.fc1 = nn.Linear(self.input_size, self.hidden_sizes[0]).to(self.device)
    self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1]).to(self.device)
    self.fc3 = nn.Linear(hidden_sizes[1], output_size).to(self.device)
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inp):
    x = torch.relu(self.fc1(inp))
    x = torch.relu(self.fc2(x))
    out = self.logsoftmax(self.fc3(x))
    return out

def linear_base_function(input, weights, biases, **kwargs):
  f = kwargs["f"]
  return f(F.linear(input, weights, biases))


class PCLayer2_GPU(object):
  def __init__(self, device, base_fn,params, **kwargs):
    self.device = device
    self.base_fn = base_fn
    self.params = params
    # set to correct device
    self.set_device(self.device)
    self.kwargs = kwargs # how to handle this?

  def set_device(self, device):
    self.device = device
    for i in range(len(self.params)):
      self.params[i] = self.params[i].to(self.device)

  def mlp_forward(self, x):
    weights, biases = self.params
    self.x = x.clone()
    return self.f(self.base_fn(x, weights,biases))

  def forward(self, x):
    x = x.to(self.device)
    self.x = x.clone()
    return self.base_fn(x, *self.params, **self.kwargs)

  def backward(self, e):
    back_fn = lambda x,*p: self.base_fn(x, *p, **self.kwargs) 
    out, grads =  taf.vjp(back_fn, tuple([self.x] + self.params), e)
    self.dx = grads[0]#.to(self.device)
    self.dparams = grads[1:]#.to(self.device)
    return self.dx, self.dparams

  def update_params(self, lr):
    for i in range(len(self.params)):
      self.params[i] = self.params[i] - (lr * self.dparams[i]) 
    return self.dparams

  def set_params(self):
    self.params = [nn.Parameter(p) for p in self.params]

  def unset_params(self):
    self.params = [p.detach() for p in self.params]
    
    
# base PCN class

class PC_Net2_GPU(object):
  def __init__(self, pc_layers, batch_size, mu_dt=0.1, lr=0.005, N_reconf_steps=20,use_reconfiguration = False, use_backprop = False,  clamp_val = 1000,use_PPC = False,store_gradient_angle=False, device="cpu"):
    self.pc_layers = pc_layers
    self.batch_size = batch_size
    self.mu_dt = mu_dt
    self.lr = lr
    self.N_reconf_steps = N_reconf_steps
    self.clamp_val = clamp_val
    self.use_reconfiguration = use_reconfiguration
    self.use_backprop = use_backprop
    self.device = device
    self.use_PPC = use_PPC
    self.store_gradient_angle = store_gradient_angle
    self.eps = 1e-6
    for l in self.pc_layers:
      l.set_device(self.device)
    if self.store_gradient_angle:
      self.n_params = self.get_len_params()
      self.gradient_angles = [[] for i in range(self.n_params)]


  def forward(self, inp):
    with torch.no_grad():
      x = inp.clone().to(self.device)
      for l in self.pc_layers:
        l.x = deepcopy(x)
        x = l.forward(x)
      return x

  def set_params(self):
    for l in self.pc_layers:
      l.set_params()

  def unset_params(self):
    for l in self.pc_layers:
      l.unset_params()

  def batch_accuracy(self, preds, labels):
    pred_idxs = torch.argmax(preds,dim=0)
    corrects = pred_idxs == labels
    return torch.sum(corrects).item() / self.batch_size

  def onehot_batch(self, ls):
    return F.one_hot(ls, 10).permute(1,0)

  def backprop_infer(self, inp, label, loss_fn,loss_fn_str = "mse"):
    with torch.no_grad():
      inp = inp.to(self.device)
      label = label.to(self.device)
      out = self.forward(inp)
      es = [[] for i in range(len(self.pc_layers)+1)]
      loss, dl = taf.vjp(lambda out: loss_fn(out, label),out,torch.tensor(1).to(self.device))
      es[-1] = deepcopy(dl)
      dws = [[] for i in range(len(self.pc_layers))]
      for i in reversed(range(len(self.pc_layers))):
        dx, dparams = self.pc_layers[i].backward(es[i+1])
        es[i] = deepcopy(dx)
        dws[i] = deepcopy(dparams)
      self.dws = dws
      return es, dws

  def compare_bp_autograd(self, inp, label, loss_fn = "mse"):

    inp = torch.tensor(inp.reshape(784,self.batch_size),dtype=torch.float).to(self.device)
    label = label.to(self.device)
    if loss_fn == "mse":
      label = torch.tensor(self.onehot_batch(label).reshape(10,self.batch_size), dtype=torch.float)
    es, dws = self.backprop_infer(inp, label,loss_fn_str = loss_fn)
    self.set_params()
    out = self.forward(inp)
    if loss_fn == "mse":
      loss_fn = nn.MSELoss(reduction = "sum")
    if loss_fn == "crossentropy":
      loss_fn = nn.CrossEntropyLoss(reduction="sum")
      out = out.T
    
    loss = loss_fn(out, label)
    print("BP LOSS: ", loss.item())
    loss.backward()
    dwss = []
    paramss = []
    for i,l in enumerate(self.pc_layers):
      for j in range(len(dws[i])):
        print("DW")
        print(dws[i][j])
        print("GRAD")
        print(l.params[j].grad)
        dwss.append(deepcopy(dws[i][j]))
        paramss.append(deepcopy(l.params[j].grad.detach()))
    self.unset_params()
    return dwss, paramss

  def compare_bp_reconf(self, inp, label, loss_fn = "mse"):
    inp = torch.tensor(inp.reshape(784,self.batch_size),dtype=torch.float).to(self.device)
    label = label.to(self.device)
    if loss_fn == "mse":
      label = torch.tensor(self.onehot_batch(label).reshape(10,self.batch_size), dtype=torch.float)
    es_bp, dws_bp = self.backprop_infer(inp, label,loss_fn_str = loss_fn)
    es_fp, dws_fp = self.gradient_infer(inp, label, loss_fn_str = loss_fn)
    for i in range(len(es_bp)):
      print("BP:")
      print(es_bp[i].shape)
      print(es_bp[i])
      print("Reconf:")
      print(es_fp[i].shape)
      print(es_fp[i])
    for i in range(len(self.pc_layers)):
      param_example = self.pc_layers[0].dparams
      N = len(param_example)
      for n in range(N):
        print("BP: ", dws_bp[i][n].shape)
        print(dws_bp[i][n])
        print("Reconfiguration :", self.pc_layers[i].dparams[n].shape)
        print(self.pc_layers[i].dparams[n])
  

  def gradient_infer(self, inp, label,loss_fn, loss_fn_str = "mse", store_errors = False, track_reconf_loss = False):
    inp = inp.to(self.device)
    label = label.to(self.device)
    out = self.forward(inp)
    es = [[] for l in range(len(self.pc_layers)+1)]
    dparamss = [torch.zeros(1) for i in range(len(self.pc_layers))]
    loss, dl = taf.vjp(lambda out: loss_fn(out, label),out,torch.tensor(1).to(self.device))
    #print("loss: ", loss)
    begin_loss = deepcopy(loss)
    begin_F = deepcopy(begin_loss).item()
    loss_diff = 0
    diff_F = 0
    es[-1] = deepcopy(dl)
    #print("INIT LOSS: ", dl)
    es[0] = torch.zeros(1)
    if store_errors:
      errorlist = []
    for i in range(self.N_reconf_steps):
      for l in reversed(range(len(self.pc_layers)-1)):
        # compute prediction errors
        es[l+1] = -self.pc_layers[l+1].x + self.pc_layers[l].forward(self.pc_layers[l].x)
        #print(es[l+1])
        # compute gradient and update
        dx, dparams = self.pc_layers[l+1].backward(es[l+2])
        dparamss[l+1] = deepcopy(dparams)
        self.pc_layers[l+1].x += self.mu_dt * (es[l+1] - dx)
      dx, dparams = self.pc_layers[0].backward(es[1])
      #dparamss[l] = deepcopy(dparams)
      dparamss[0] = deepcopy(dparams)
      out = self.pc_layers[-1].forward(self.pc_layers[-1].x)
      loss, dl = taf.vjp(lambda out: loss_fn(out, label),out,torch.tensor(1).to(self.device))
      es[-1] = deepcopy(dl)
      #print("output error: ", es[-1])
      if self.use_PPC:
        self.dws = dparamss
        self.step()
      if store_errors:
        es_tilde = deepcopy(es)
        predicted_out = self.pc_layers[-1].forward(self.pc_layers[-1].x)
        loss, dl_2 = taf.vjp(lambda out: loss_fn(predicted_out, label),out,torch.tensor(1).to(self.device))
        #print(len(es_tilde))
        #es_tilde[-1] = deepcopy(dl_2)
        #print("estilde: ", es_tilde)
        #output_pred = self.pc_layers[-1].forward(self.pc_layers[-1].x) 
        #print("output pred:", output_pred.shape)
        #print("label:shape ", label.shape)
        #onehot_label = F.one_hot(label, num_classes=10)
        #out_error = output_pred - onehot_label 
        end_F = 0
        for e in es:
          end_F += torch.sum(torch.square(e)).item()
        diff_F = end_F - begin_F
        errorlist.append(deepcopy(es_tilde))
    self.dws = dparamss
    if track_reconf_loss:
      predicted_out = self.pc_layers[-1].forward(self.pc_layers[-1].x)
      loss, dl = taf.vjp(lambda out: loss_fn(predicted_out, label),out,torch.tensor(1).to(self.device))
      end_loss = deepcopy(loss)
      loss_diff = end_loss - begin_loss


    if store_errors:
      return es, dparamss, errorlist, loss_diff, diff_F
    return es,dparamss

  def get_len_params(self):
    total = 0
    for l in self.pc_layers:
      for p in l.params:
        total +=1
    return total

  def get_parameters(self):
    param_list = []
    for l in self.pc_layers:
      for p in l.params:
        param_list.append(p)
    return param_list

  def init_optimizer(self, opt):
    self.opt = opt
    param_list = self.get_parameters()
    self.opt.params = param_list

  def step(self):
    if self.opt is None:
      raise ValueError("Must provide an optimizer to step using the init_optimizer function")
    idx = 0
    for i,l in enumerate(self.pc_layers):
      for j,p in enumerate(l.params):
        self.opt.params[idx].grad = self.dws[i][j]
        idx +=1
    self.opt.step()

  def update_params(self,lr=None, store_updates = False):
    if lr is None:
      lr = self.lr
    if store_updates:
      dws = []
      for i,l in enumerate(self.pc_layers):
        dw = l.update_params(lr,fake_update = True)
        dws.append(dw)
      return dws
    else:
        for i,l in enumerate(self.pc_layers):
          dw = l.update_params(lr)


  def batched_cosine_similarity(self,x,y,cosine=True):
    similarities = []
    assert x.shape == y.shape, "must have same shape"
    if len(x.shape) == 1:
      # handle vectors
      x = x.reshape(len(x),1)
      y = y.reshape(len(y),1)
    for i in range(len(x)):
      sim = torch.dot(x[i,:],y[i,:]) / ((torch.norm(x[i,:]) + self.eps) * torch.norm(y[i,:])+self.eps)
      if cosine:
        sim = torch.acos(sim)
      similarities.append(sim.detach().cpu().numpy())
    similarities = np.array(similarities)
    mean_similarities = np.mean(similarities)
    std_similarities = np.std(similarities)
    return mean_similarities, std_similarities

  def train(self, trainset, N_epochs = 10,direct_bp = True, loss_fn_str = "mse",print_outputs = True):
    losses = []
    accs = []
    for n in range(N_epochs):
      print("EPOCH ", n)
      for i, (img, label) in enumerate(trainset):
        img = torch.tensor(img.reshape(784,self.batch_size),dtype=torch.float).to(self.device)
        if loss_fn_str == "mse":
          onehotted_label = torch.tensor(self.onehot_batch(label).reshape(10,self.batch_size), dtype=torch.float).to(self.device)
        else:
          onehotted_label = label.to(self.device)
        # inference
        if self.use_reconfiguration:
          es,dparams = self.gradient_infer(img, onehotted_label,loss_fn_str = loss_fn_str)
        if self.use_backprop:
          es,dparams = self.backprop_infer(img, onehotted_label,loss_fn_str = loss_fn_str)
        #weight update
        if not direct_bp:
          self.update_params()
        if direct_bp:
          self.set_params()
          out = self.forward(img)
          if loss_fn_str == "mse":
            loss_fn = nn.MSELoss(reduction="sum")
          if loss_fn_str == "crossentropy":
            loss_fn = nn.CrossEntropyLoss(reduction="sum")
            out =  out.T
      
          loss = loss_fn(out, onehotted_label)
          loss.backward()
          # iterate over all param list
          for i,l in enumerate(self.pc_layers):
            for j in range(len(l.params)):
              dparam = l.params[j].grad
              l.params[j] = l.params[j].detach()
              l.params[j] = l.params[j] - (self.lr *dparam.detach())
              l.params[j] = nn.Parameter(l.params[j])
        out = self.forward(img)
        acc = self.batch_accuracy(out, label)
        if print_outputs:
          print("acc: ", acc)
        accs.append(acc)
        if loss_fn_str == "mse":
          loss_fn = nn.MSELoss(reduction="sum")
        if loss_fn_str == "crossentropy":
          loss_fn = nn.CrossEntropyLoss(reduction="sum")
          out =  out.T
        #loss = torch.sum(torch.square(out - onehotted_label)).item()
        loss = loss_fn(out, onehotted_label).item()
        losses.append(loss)
        if print_outputs:
          print("loss: ", loss)
    return np.array(losses), np.array(accs)


def compare_pc_bp():
    
    model = NN_model(input_size, hidden_sizes, output_size, batch_size,device=DEVICE)
    print(model)

    w1, b1 = list(model.fc1.parameters())

    w2, b2 = list(model.fc2.parameters())
    w3,b3 = list(model.fc3.parameters())
    pc_fc1 = PCLayer2_GPU("cpu", linear_base_function,[w1.detach(),b1.detach()],f=torch.relu)
    pc_fc2 = PCLayer2_GPU("cpu", linear_base_function,  [w2.detach(),b2.detach()],f=torch.relu)
    pc_fc3 = PCLayer2_GPU("cpu", linear_base_function, [w3.detach(),b3.detach()],f=nn.LogSoftmax(dim=1))
    pcnet = PC_Net2_GPU([pc_fc1, pc_fc2, pc_fc3],batch_size, 0.1, 0.005,20,20,use_backprop=True, clamp_val=10000,device=DEVICE)

    images, labels = next(iter(trainloader))
    images = images.view(images.shape[0], -1).to(DEVICE)
    labels = labels.to(DEVICE)
    images2 = deepcopy(images)
    logps = model(images) 
    logps_pc = pcnet.forward(images2)
    print("BP NET: ", logps[0,:])
    print("PC NET: ", logps_pc[0,:])

    criterion = nn.NLLLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.0)
    opt_pc = optim.SGD(pcnet.get_parameters(), lr=0.005, momentum=0.0)
    pcnet.init_optimizer(opt_pc)
    optimizer.zero_grad()
    loss = criterion(logps, labels)
    print("loss 1",loss.item())
    loss.backward()

    bp_params = list(model.parameters())
    print("     PC GRADS    ")
    es, dws = pcnet.gradient_infer(images, labels, loss_fn = criterion)

    es, dws_bp = pcnet.backprop_infer(images, labels,loss_fn = criterion)
    i = 0
    for k,dw in enumerate(list(dws)):
        for j,d in enumerate(dw):
            print("PC:", d.shape)
            print(d)
            print("BP: ")
            print(bp_params[i].grad)
            i+=1
            print(dws_bp[k][j])
            
def compute_test_accuracy(pcnet, valloader):
    test_accs = []
    N_accs = 0
    total_acc = 0
    for images, labels in valloader:
        images = images.view(images.shape[0], -1)
        output = pcnet.forward(images)
        acc = accuracy(output,labels)
        test_accs.append(acc.item())
        total_acc += acc.item()
        N_accs +=1

    print("TOTAL TEST ACCURACY: ", total_acc/N_accs)
    return total_acc / N_accs

def compute_train_accuracy(pcnet,trainloader):
    train_accs = []
    N_accs = 0
    total_acc = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)
        output = pcnet.forward(images)
        acc = accuracy(output,labels)
        train_accs.append(acc.item())
        total_acc += acc.item()
        N_accs +=1

    print("TOTAL TRAIN ACCURACY: ", total_acc/N_accs)
    return total_acc / N_accs


def run_experiment(input_size, hidden_sizes, output_size, epochs = 15,batch_size=64, mu_dt=0.2, lr=0.003, N_reconf_steps=20, use_backprop=False,  use_reconf=False, clamp_val=10000, use_PPC=False, momentum=0.9, return_all_accs = False, store_gradient_angle=True):
    trainloader, valloader = get_mnist_dataset(batch_size = batch_size)
    
    model = NN_model(input_size, hidden_sizes, output_size, batch_size,device=DEVICE)

    w1, b1 = list(model.fc1.parameters())
    w2, b2 = list(model.fc2.parameters())
    w3,b3 = list(model.fc3.parameters())

    pc_fc1 = PCLayer2_GPU("cpu", linear_base_function,[deepcopy(w1.detach()),deepcopy(b1.detach())],f=torch.relu)
    pc_fc2 = PCLayer2_GPU("cpu", linear_base_function,  [deepcopy(w2.detach()),deepcopy(b2.detach())],f=torch.relu)
    pc_fc3 = PCLayer2_GPU("cpu", linear_base_function, [deepcopy(w3.detach()),deepcopy(b3.detach())],f=nn.LogSoftmax(dim=1))
    pcnet = PC_Net2_GPU([pc_fc1, pc_fc2, pc_fc3],batch_size, mu_dt = mu_dt, lr=lr,N_reconf_steps =N_reconf_steps,use_backprop=use_backprop, clamp_val=clamp_val,use_PPC = use_PPC,store_gradient_angle = store_gradient_angle, device=DEVICE)

    criterion = nn.NLLLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer_pc = optim.SGD(pcnet.get_parameters(), lr=lr, momentum=momentum)
    pcnet.init_optimizer(optimizer_pc)
    optimizer.zero_grad()
    time0 = time()
    
    #accs = []
    train_accs = []
    train_losses = []
    test_accs = []
    if return_all_accs:
        accs =[]
    for e in range(epochs):
        running_loss = 0
        running_loss_bp = 0
        for i,(images, labels) in enumerate(trainloader):
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1).to(DEVICE)
            labels = labels.to(DEVICE)
        
            # Training pass
            optimizer.zero_grad()
            output = pcnet.forward(images)
            loss = criterion(output, labels)
            #print("loss: ", loss.item())
            acc = accuracy(output, labels)
            if return_all_accs:
                accs.append(acc.item())
            #print("PC acc: ", acc.item())
            train_losses.append(loss.item())
            train_accs.append(acc.item())
            if i% 100 == 0:
                print("LOSS: ", loss.item())
                print("ACC: ", acc.item())


            if use_backprop:
                es, dws =pcnet.backprop_infer(images, labels,loss_fn = criterion)
            if use_reconf:
                es, dws = pcnet.gradient_infer(images, labels, loss_fn = criterion)
            pcnet.step()
            if store_gradient_angle:
                es, dws = pcnet.gradient_infer(images, labels, loss_fn = criterion)
                _, dws_bp = pcnet.backprop_infer(images, labels, loss_fn= criterion)
                i = 0
                for dw_bp, dw_pc in zip(dws_bp, dws):
                    for p_bp, p_pc in zip(dw_bp, dw_pc):
                        mean_angle, std_angle = pcnet.batched_cosine_similarity(p_bp, p_pc, cosine=False)
                        pcnet.gradient_angles[i].append(deepcopy(mean_angle))
                        i+=1
            
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
            print("\nTraining Time (in minutes) =",(time()-time0)/60)
            print("acc: ", acc)
            test_acc = compute_test_accuracy(pcnet, valloader)
            test_accs.append(test_acc)
    if return_all_accs:
        return pcnet, train_losses, train_accs, test_accs, accs 
    return pcnet, train_losses, train_accs, test_accs


def pc_learning_experiment():
    store_gradient_angle = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE IS : ", DEVICE)
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10
    batch_size  = 64
    lr = 0.003
    momentum = 0.9
    EPOCHS = 10
    N_runs = 3
    pcnets = []
    train_loss_list = []
    train_accs_list = []
    test_accs_list = []
    gradient_angle_list = []
    for N in range(N_runs):
        pcnet, train_losses, train_accs, test_accs  =run_experiment(input_size, hidden_sizes, output_size, batch_size=batch_size, use_reconf=True, return_all_accs=False,epochs = EPOCHS,lr=lr, use_PPC=False,store_gradient_angle = store_gradient_angle,momentum=momentum)
        pcnets.append(pcnets)
        train_loss_list.append(train_losses)
        train_accs_list.append(train_accs)
        test_accs_list.append(test_accs)
        if store_gradient_angle:
            gradient_angles = deepcopy(pcnet.gradient_angles)
            gradient_angle_list.append(gradient_angles)




    train_loss_list = np.array(train_loss_list)
    train_accs_list = np.array(train_accs_list)
    test_accs_list = np.array(test_accs_list)
    gradient_angle_list = np.array(gradient_angle_list)
    np.save("reconf_train_loss_list_avg_2.npy",train_loss_list)
    np.save("reconf_train_accs_list_avg_2.npy",train_accs_list)
    np.save("reconf_test_accs_list_avg_2.npy",test_accs_list)
    if store_gradient_angle:
        np.save("Gradient_angles_list.npy", gradient_angle_list)
    print("DONE!")
    return train_loss_list, train_accs_list, test_accs_list, gradient_angle_list
    
def backprop_experiment():
    store_gradient_angle = False
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE IS : ", DEVICE)
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10
    batch_size  = 64
    lr = 0.003
    momentum = 0.9
    EPOCHS = 10
    N_runs = 3
    pcnets = []
    train_loss_list = []
    train_accs_list = []
    test_accs_list = []
    gradient_angle_list = []
    for N in range(N_runs):
        pcnet, train_losses, train_accs, test_accs  =run_experiment(input_size, hidden_sizes, output_size, batch_size=batch_size, use_reconf=False,use_backprop=True, return_all_accs=False,epochs = EPOCHS,lr=lr, use_PPC=False,store_gradient_angle = store_gradient_angle,momentum=momentum)
        pcnets.append(pcnets)
        train_loss_list.append(train_losses)
        train_accs_list.append(train_accs)
        test_accs_list.append(test_accs)
        if store_gradient_angle:
            gradient_angles = deepcopy(pcnet.gradient_angles)
            gradient_angle_list.append(gradient_angles)




    train_loss_list = np.array(train_loss_list)
    train_accs_list = np.array(train_accs_list)
    test_accs_list = np.array(test_accs_list)
    gradient_angle_list = np.array(gradient_angle_list)
    np.save("backprop_train_loss_list_avg.npy",train_loss_list)
    np.save("backprop_train_accs_list_avg.npy",train_accs_list)
    np.save("backprop_test_accs_list_avg.npy",test_accs_list)
    if store_gradient_angle:
        np.save("Gradient_angles_list.npy", gradient_angle_list)
    print("DONE!")
    return train_loss_list, train_accs_list, test_accs_list, gradient_angle_list

if __name__ == '__main__':
    pc_learning_experiment()
    backprop_experiment()