# this is a copy out of the jupyter notebook and used to produce the decompositions of the FE script

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import seaborn as sns
from copy import deepcopy

import brewer2mpl


BATCH_SIZE = 64

bmap = brewer2mpl.get_map("Set2", 'qualitative',7)
colors = bmap.mpl_colors




transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])



def get_mnist_dataset(batch_size):

  transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

  trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
  valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
  valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
  return trainloader, valloader


def accuracy(out, labels):
  labels = labels.cpu()
  with torch.no_grad():
    maxes = torch.argmax(out.detach().cpu(), dim=1)
    corrects = maxes == labels
    return torch.sum(corrects) / len(corrects)



def cosine_similarity(x1,x2):
    angle = (torch.dot(x1,x2) / (torch.norm(x1,2) * torch.norm(x2,2)))
    return angle.item()
  #return torch.acos(angle).item()

def construct_precision_matrix(N, identity_scale, variance):
    Pi = (torch.eye(N) * identity_scale) + torch.abs(torch.tensor(np.random.normal(0,variance,(N,N))))
    return Pi

def set_tensor(x):
  return torch.tensor(x, dtype=torch.float)

### Activation functions ###
def tanh(xs):
    return torch.tanh(xs)

def linear(x):
    return x

def tanh_deriv(xs):
    return 1.0 - torch.tanh(xs) ** 2.0

def linear_deriv(x):
    return set_tensor(torch.ones((1,)))

def relu(xs):
  return torch.clamp(xs,min=0)

def relu_deriv(xs):
  rel = relu(xs)
  rel[rel>0] = 1
  return rel

def softmax(xs):
  return F.softmax(xs)

def sigmoid(xs):
  return F.sigmoid(xs)

def sigmoid_deriv(xs):
  return F.sigmoid(xs) * (torch.ones_like(xs) - F.sigmoid(xs))



def four_layer_test_net(learning_rate =0.1, weight_var = 0.05, activity_var = 1, dim =20, random_init = True,plot_energies = False):
  W1 = torch.tensor(np.random.normal(0,weight_var,(dim,dim)))
  W2 = torch.tensor(np.random.normal(0,weight_var,(dim,dim)))
  W3 = torch.tensor(np.random.normal(0,weight_var,(dim,dim)))
  W4 = torch.tensor(np.random.normal(0,weight_var,(dim,dim)))
  x1 = torch.tensor(np.random.normal(1,activity_var,(dim,1)))
  #x2 = torch.tensor(np.random.normal(0.0,0.05,(5,1)))

  f = torch.tanh_
  f_inv = torch.tanh
  fderiv = tanh_deriv

  if random_init:
    x2 = torch.tensor(np.random.normal(0,weight_var, (dim,1)))
    x3 = torch.tensor(np.random.normal(0,weight_var, (dim,1)))
    x4 = torch.tensor(np.random.normal(0,weight_var, (dim,1)))
  else:
    x2 = f(W1 @ x1)
    x3 = f(W2 @ x2)
    x4 = f(W3 @ x3)
  x5 = torch.tensor(np.random.normal(-1,activity_var,(dim,1)))

  x2s = []
  x3s = []
  x4s = []
  Fs = []
  FP_angles = []
  TP_angles = []
  FP_diffs = []
  TP_diffs = []
  x2_fps = []
  x3_fps = []
  x4_fps = []
  output_losses = []
  E_tildes = []
  with torch.no_grad():
      for i in range(100):
          x2s.append(deepcopy(x2.numpy()))
          x3s.append(deepcopy(x3.numpy()))
          x4s.append(deepcopy(x4.numpy()))         
          #FP_angles.append(cosine_similarity(x2.reshape(dim,), FP_x2.reshape(dim,)))
          #TP_angles.append(cosine_similarity(x2.reshape(dim,), TP_x2.reshape(dim,)))
          #FP_diffs.append(torch.sum(torch.square(x2 - FP_x2)).item())
          #TP_diffs.append(torch.sum(torch.square(x2 - TP_x2)).item())
          e2 = x2 - f(W1 @ x1)
          e3 = x3 - f(W2 @ x2)
          e4 = x4 - f(W3 @ x3)
          e5 = x5 - f(W4 @ x4)
          x2 -= learning_rate * (e2 - W2.T @ (e3 * fderiv(W2 @ x2)))
          x3 -= learning_rate * (e3 - W3.T @ (e4 * fderiv(W3 @ x3)))
          x4 -= learning_rate * (e4 - W4.T @ (e5 * fderiv(W4 @ x4)))
          #Fs.append(torch.sum(torch.square(e2)) + torch.sum(torch.square(e3)))
          F = torch.sum(torch.square(e2)) + torch.sum(torch.square(e3)) + torch.sum(torch.square(e4)) + torch.sum(torch.square(e5))
          Fs.append(F.item())
          output_losses.append(torch.sum(torch.square(e5)).item())
          E_tilde = torch.sum(torch.square(e2)) + torch.sum(torch.square(e3)) + torch.sum(torch.square(e4))
          E_tildes.append(E_tilde.item())
          

  if random_init:
    x2 = torch.tensor(np.random.normal(0,weight_var, (dim,1)))
    x3 = torch.tensor(np.random.normal(0,weight_var, (dim,1)))
    x4 = torch.tensor(np.random.normal(0,weight_var, (dim,1)))
  else:
    x2 = f(W1 @ x1)
    x3 = f(W2 @ x2)
    x4 = f(W3 @ x3)
    print("SHAPES:")
    print(x2.shape)
    print(x3.shape)
    print(x4.shape)
  x2_fps.append(deepcopy(x2.numpy()))
  x3_fps.append(deepcopy(x3.numpy()))
  x4_fps.append(deepcopy(x4.numpy()))
  x2_fp = deepcopy(x2)
  x3_fp = deepcopy(x3)
  x4_fp = deepcopy(x4)
  F_fps = []
  out_l_fps = []
  E_tilde_fps = []
  with torch.no_grad():
    for i in range(30):
      e2_fp = x2_fp - f(W1 @ x1)
      e3_fp = x3_fp - f(W2 @ x2_fp)
      e4_fp = x4_fp - f(W3 @ x3_fp)
      e5_fp = x5 - f(W4 @ x4_fp)
      x2_fp = f(W1 @ x1) + W2.T @ (e3_fp * fderiv(W2 @ x2_fp))
      #print("FP : ", fp)
      x2_fps.append(deepcopy(x2_fp.numpy()))
      #x2 = deepcopy(x2_fp)
      x3_fp = f(W2 @ x2_fp) + W3.T @ (e4_fp * fderiv(W3 @ x3_fp))
      x3_fps.append(deepcopy(x3_fp.numpy()))
      #x3 = deepcopy(x3_fp)
      x4_fp = f(W3 @ x3_fp) + W4.T @ (e5_fp * fderiv(W4 @ x4_fp))
      x4_fps.append(deepcopy(x4_fp.numpy()))
      #x4 = deepcopy(x4_fp)
      out_l_fps.append(torch.sum(torch.square(e5_fp)).item())
      E_tilde_fp = torch.sum(torch.square(e2_fp)) + torch.sum(torch.square(e3_fp)) + torch.sum(torch.square(e4_fp))
      E_tilde_fps.append(E_tilde_fp.item())
      F_fp = torch.sum(torch.square(e2_fp)) + torch.sum(torch.square(e3_fp)) + torch.sum(torch.square(e4_fp)) + torch.sum(torch.square(e5_fp))
      F_fps.append(F_fp.item())

  x2s = np.array(x2s)
  x3s = np.array(x3s)
  x4s = np.array(x4s)
  #plt.plot(x2s.reshape(100,dim))
  #plt.show()
  x2_fps = np.array(x2_fps)
  x3_fps = np.array(x3_fps)
  x4_fps = np.array(x4_fps)
  if plot_energies:
    fig = plt.figure(figsize=(14,12))
    xx2s = x2s[0:30,0:4].squeeze()
    print("xx2s: ", xx2s.shape)
    xx2fps = x2_fps[0:30, 0:4].squeeze()
    print("xx2fps: ", xx2fps.shape)
    for i in range(4):
      if i == 0:
        plt.plot(xx2s[:,i], label="Iterative", color = colors[i], linestyle="dashed")
        plt.plot(xx2fps[:,i], label="Fixed point",color=colors[i],linestyle="solid")
      else: # need to do this to not generate a crapton of legends. This is a hacky workaroudn
        plt.plot(xx2s[:,i],color = colors[i], linestyle="dashed")
        plt.plot(xx2fps[:,i],color=colors[i],linestyle="solid")
    sns.despine(left=False,top=True, right=True, bottom=False)
    plt.title("Activities for gradient descent and fixed point",fontsize=30)
    sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    plt.xlabel("Timestep",fontsize=28)
    plt.ylabel("Value",fontsize=28)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    fig.tight_layout()
    plt.savefig("fp_grad_activities.jpg", format="jpg")
    plt.show()

    fig = plt.figure(figsize=(14,12))
    plt.plot(Fs, label="Free Energy",linestyle="dashed",linewidth=2)
    plt.plot(output_losses, label="Output Loss", linestyle="solid",linewidth=2)
    plt.plot(E_tildes, label="Residual Energy", linestyle="dashed",linewidth=2)
    sns.despine(left=False,top=True, right=True, bottom=False)
    sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    plt.title("Evolution of Free Energy Components During Inference Phase",fontsize=30)
    plt.xlabel("Timestep",fontsize=28)
    plt.ylabel("Value",fontsize=28)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    fig.tight_layout()
    plt.savefig("energies_plot.jpg", format="jpg")
    plt.show()

    print(type(F_fps))
    print(type(out_l_fps))
    print(type(E_tilde_fps))
    fig = plt.figure(figsize=(14,12))
    plt.plot(F_fps, label="Free Energy",linestyle="dashed",linewidth=2)
    plt.plot(out_l_fps, label="Output Loss", linestyle="solid",linewidth=2)
    plt.plot(E_tilde_fps, label="Residual Energy", linestyle="dashed",linewidth=2)
    sns.despine(left=False,top=True, right=True, bottom=False)
    sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    plt.title("Evolution of Free Energy Components During Fixed Point Iteration",fontsize=30)
    plt.xlabel("Timestep",fontsize=28)
    plt.ylabel("Value",fontsize=28)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    fig.tight_layout()
    plt.savefig("fixed_point_energies_plot.jpg", format="jpg")
    plt.show()
    
  diffs = np.mean(np.abs(x2s[88:99] - x2_fps),axis=1) + np.mean(np.abs(x3s[88:99] - x3_fps),axis=1) + np.mean(np.abs(x4s[88:99] - x4_fps),axis=1)
  return diffs


def N_runs_diff_plot(N_runs):
  diffss = []
  for i in range(N_runs):
     diffs = four_layer_test_net(random_init = False, plot_energies = False) 
     diffss.append(deepcopy(diffs))
  diffss = np.array(diffss)
  print("shape: ", diffss.shape)
  mean_diffs = np.mean(diffss, axis=0)
  mean_diffs = mean_diffs.reshape(len(mean_diffs))
  std_diffs = np.std(diffss, axis=0).reshape(len(mean_diffs))# / np.sqrt(N_runs)
  print(mean_diffs.shape)
  print(std_diffs.shape)
  xs = np.arange(0, len(mean_diffs))
  fig = plt.figure(figsize=(14,12))
  plt.plot(xs, mean_diffs)
  plt.fill_between(xs, mean_diffs - std_diffs, mean_diffs + std_diffs, alpha=0.5)
  sns.despine(left=False,top=True, right=True, bottom=False)
  sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
  plt.title("Distance from Equilibrium",fontsize=30)
  plt.xlabel("Timestep",fontsize=28)
  plt.ylabel("Absolute Activity Difference",fontsize=28)
  plt.xticks(fontsize=25)
  plt.yticks(fontsize=25)
  plt.savefig("mean_diffs_fig.jpg", format="jpg")
  plt.show()
  
  
if __name__ == '__main__':
  four_layer_test_net(random_init=True,plot_energies = True)
  #N_runs_diff_plot(10)