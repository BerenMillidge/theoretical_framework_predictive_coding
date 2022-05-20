## Utility function
import torch
import numpy as np

def cosine_similarity(x1,x2):
    angle = (torch.dot(x1,x2) / (torch.norm(x1,2) * torch.norm(x2,2)))
    return angle.item()
  #return torch.acos(angle).item()

def construct_precision_matrix(N, identity_scale, variance):
    Pi = (torch.eye(N) * identity_scale) + torch.abs(torch.tensor(np.random.normal(0,variance,(N,N))))
    return Pi

### Activation functions ###
def tanh(xs):
    return torch.tanh(xs)

def linear(x):
    return x

def tanh_deriv(xs):
    return 1.0 - torch.tanh(xs) ** 2.0

def linear_deriv(x):
    return torch.ones(1,).float()

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
