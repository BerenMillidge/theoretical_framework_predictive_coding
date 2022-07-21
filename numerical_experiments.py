### Numerical experiments for the "Theoretical Foundation of Learning and Inference in Predictive Coding Networks" paper ###

import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from utils import *
from plotting import *

# Verify that the linear equilibrium derives is correct in practice
def verify_linear_equilibrium(dimension=5, var=1, weight_var = 0.05, learning_rate=0.1,dim=5):
    # initialize weights and activities randomly
    x1 = torch.tensor(np.random.normal(1,var,(dimension,1)))
    x2 = torch.tensor(np.random.normal(0.0,var,(dimension,1)))
    x3 = torch.tensor(np.random.normal(-1,var,(dimension,1)))
    W1 = torch.tensor(np.random.normal(0,weight_var,(dimension,dimension)))
    W2 = torch.tensor(np.random.normal(0,weight_var,(dimension,dimension)))

    # compute the analytical equilibrium
    prefactor = torch.inverse(torch.eye(dim) + W2.T @ W2)
    pred_eq =  prefactor @ (W1 @ x1) + prefactor @ (W2.T @ x3)

    # run predictive coding
    x2s = []
    Fs = []
    diffs_from_eq = []
    with torch.no_grad():
        for i in range(100):
            e2 = x2 - W1 @ x1
            e3 = x3 - W2 @ x2
            x2 -= learning_rate * (e2 - W2.T @ e3)
            Fs.append(torch.sum(torch.square(e2)) + torch.sum(torch.square(e3)))
            x2s.append(deepcopy(x2.numpy()))
            diffs_from_eq.append(deepcopy(x2.numpy()) - pred_eq.numpy())

    plot_line_graph(Fs, title="Free Energy of the Network", xlabel="Timestep", ylabel="Free Energy",label="Free-Energy",sname="figures/verify_linear_equilibrium_Fs")
    x2s = np.array(x2s)[:,:,0]
    plot_equilibrium_graph(x2s,pred_eq.numpy()[:,0], title="Activities Converging to Linear Equilibrium",xlabel="Timestep", ylabel="Activity Value",sname="figures/verify_linear_equilibrium_activities",label="Activity Value")
    diffs_from_eq = np.array(diffs_from_eq)[:,:,0]
    plot_line_graph(diffs_from_eq, title="Difference of activities from Linear Equilibrium", xlabel="Timestep",ylabel="Euclidean Distance",sname="figures/verify_linear_equilibrium_diffs")

    total_diffs_from_eq = np.sum(np.square(diffs_from_eq), axis=1)
    plot_line_graph(total_diffs_from_eq, title="Total Euclidean Distance from Equilibrium", xlabel="Timestep",ylabel="Euclidean Distance",label="Distance",sname="figures/verify_linear_equilibrium_total_diffs",divergence_graph=True)

# Function to show that when input layer is unconstrained converges to TP
def input_unconstrained_linear(dimension=5, var=1, weight_var = 1, learning_rate = 0.05,output_dim=5):
    # initialize weights and activiies randomly
    x1 = torch.tensor(np.random.normal(1,var,(dimension,1)))
    x2 = torch.tensor(np.random.normal(0.0,var,(dimension,1)))
    x3 = torch.tensor(np.random.normal(-1,var,(output_dim,1)))
    W1 = torch.tensor(np.random.normal(0,weight_var,(dimension,dimension)))
    W2 = torch.tensor(np.random.normal(0,weight_var,(output_dim,dimension)))
    # compute targetprop target for comparison
    if output_dim != dimension:
        # use pseudoinverse
        TP_x2 = torch.pinverse(W2) @ x3
    else:
        TP_x2 = torch.inverse(W2) @ x3
    # initialize lists
    x2s = []
    Fs = []
    diffs = []
    #TP_x2 = torch.inverse(W2) @ x3
    print(TP_x2)
    with torch.no_grad():
        for i in range(1000):
            e2 = x2 - W1 @ x1
            e3 = x3 - W2 @ x2
            x2 -= learning_rate * (- W2.T @ e3)
            x1 -= learning_rate * (- W1.T @ e2)
            Fs.append(torch.sum(torch.square(e2)) + torch.sum(torch.square(e3)))
            x2s.append(deepcopy(x2.numpy()))
            diffs.append(deepcopy(x2.numpy() - TP_x2.numpy()))

    plot_line_graph(Fs, title="Free Energy of the Network", xlabel="Timestep", ylabel="Free Energy",label="Free-Energy",sname="figures/input_unconstrained_linear_Fs")
    x2s = np.array(x2s)[:,:,0]
    plot_equilibrium_graph(x2s,TP_x2[:,0].numpy(), title="Activities Converging to Input-Unconstrained Equilibrium",xlabel="Timestep", ylabel="Activity Value",sname="figures/input_unconstrained_linear_activities")
    diffs = np.array(diffs)[:,:,0]
    plot_line_graph(diffs, title="Difference Between Activities and TP Targets", xlabel="Timestep", ylabel="Activity Difference",sname="figures/input_unconstrained_linear_diffs")
    euclid_dists = np.mean(np.square(diffs),axis=1)
    plot_line_graph(euclid_dists, title="Mean Euclidean Distance from TP Targets", xlabel="Timestep", ylabel="Mean Distance from targets",label="Mean Distance",sname="figures/input_unconstrained_linear_total_diffs",divergence_graph=True)

# multiple layer network for graph here
def multi_layer_input_unconstrained_linear(n_dimension = 2,N_layers = 5, var=1, weight_var = 1,learning_rate = 0.05,N_steps = 2000, plot_results = True):
    if type(n_dimension) != list:
        n_dimension = [n_dimension for i in range(N_layers)]

    sensible_init = False
    SENSIBLE_THRESHOLD = 10
    while sensible_init != True:
        xs = [torch.tensor(np.random.normal(1,var,(n_dimension[i],1))) for i in range(N_layers)]
        Ws = [torch.tensor(np.random.normal(0,weight_var,(n_dimension[i],n_dimension[i+1])))  for i in range(N_layers-1)]
        # normalize weights
        #xs = [x / torch.sum(x) for x in xs]
        #Ws = [W / torch.sum(W) for W in Ws]
        es = [torch.zeros_like(xs[i]) for i in range(N_layers)]
        pinvs = [[] for i in range(N_layers)]
        pinvs[-1] = xs[-1]
        print("FINAL PINVS", pinvs[-1])
        sensible_init = True
        for i in reversed(range(N_layers -1)):
            pinvs[i] = torch.pinverse(Ws[i]) @ pinvs[i+1]
            if torch.max(torch.abs(pinvs[i])) > SENSIBLE_THRESHOLD:
                sensible_init = False

    xss =[]
    Fs = []
    diffs = []
    print(pinvs)
    with torch.no_grad():
        for n in range(N_steps):
            Fs.append(0)
            xss.append([])
            diffs.append([])
            for l in range(N_layers):
                #print(l)
                if l != 0:
                    es[l] =  xs[l] - (Ws[l-1] @ xs[l-1])
                    Fs[n] += torch.sum(torch.square(es[l]))
                if l != N_layers -1:
                    xs[l] += learning_rate * (Ws[l].T @ es[l+1])
                    xss[n].append(xs[l])
                    diffs[n].append(np.abs(deepcopy(xs[l].numpy() - pinvs[l].numpy())))


    diffs = np.array(diffs)
    diffs = np.mean(diffs, axis=2)[:,:,0]
    print(diffs.shape)
    labels = ["Layer " + str(i+1) for i in range(N_layers)]
    if plot_results:
        # free energy graph
        plot_line_graph(Fs, title="Free Energy of the Network", xlabel="Timestep", ylabel="Free Energy",label="Free-Energy",sname="figures/multilayer_input_unconstrained_linear_Fs_3")
        # average diffs graph
        plot_line_graphs(diffs, title="Convergence of each layer to Target-Prop Targets", xlabel="Timestep", ylabel="Average distance to local target", labels=labels, sname="figures/multilayer_input_unconstrained_linear_diffs_3")
    return xss, Fs, diffs

def multi_trial_input_unconstrained_linear(N_trials = 100):
    diffss = []
    for n in range(N_trials):
        xss, Fs, diffs = multi_layer_input_unconstrained_linear(learning_rate = 0.05,plot_results = False)
        diffss.append(diffs)
    diffss = np.array(diffss)
    print(diffss.shape)
    means = np.mean(diffss, axis=0)
    stds = np.std(diffss, axis=0) #/ N_trials
    print(means.shape)
    print(stds.shape)
    N_layers = 5
    labels = ["Layer " + str(i+1) for i in range(N_layers)]
    plot_line_graphs(means, title="Convergence of each layer to Target-Prop Targets", xlabel="Timestep", ylabel="Average distance to local target", labels=labels, stds = stds, sname="figures/avg_seeds_multilayer_input_unconstrained_linear_diffs_100.pdf")

def input_unconstrained_nonlinear(learning_rate = 0.05, weight_var = 0.9, activity_var = 1, dim = 5, output_dim = 5):
    x1 = torch.tensor(np.random.normal(1,activity_var,(dim,1)))
    x2 = torch.tensor(np.random.normal(0.0,0.05,(dim,1)))
    x3 = torch.tensor(np.random.normal(0,0.5,(output_dim,1)))
    W1 = torch.tensor(np.random.normal(0,weight_var,(dim,dim)))
    W2 = torch.tensor(np.random.normal(0,weight_var,(output_dim,dim)))

    f = torch.tanh
    #f = torch.relu
    f_inv = torch.atanh
    fderiv = tanh_deriv


    learning_rate = 0.1
    x2s = []
    Fs = []
    diffs = []
    #TP_x2 = torch.inverse(W2) @ f_inv(x3)
    #TP_x2 = torch.linalg.pinv(W2) @ f_inv(x3)
    TP_x2 = torch.pinverse(W2) @ f_inv(x3)
    print(TP_x2)
    with torch.no_grad():
        for i in range(1000):
            e2 = x2 - f(W1 @ x1)
            e3 = x3 - f(W2 @ x2)
            x2 -= learning_rate * (- W2.T @ (e3 * fderiv(W2 @ x2)))
            x1 -= learning_rate * (- W1.T @ (e2 * fderiv(W1 @ x1)))
            Fs.append(torch.sum(torch.square(e2)) + torch.sum(torch.square(e3)))
            x2s.append(deepcopy(x2.numpy()))
            diffs.append(deepcopy(x2.numpy() - TP_x2.numpy()))

    plot_line_graph(Fs, title="Free Energy of the Network", xlabel="Timestep", ylabel="Free Energy",label="Free-Energy",sname="figures/input_unconstrained_nonlinear_Fs_2")
    x2s = np.array(x2s)[:,:,0]
    plot_equilibrium_graph(x2s, TP_x2[:,0].numpy(), title="Activities Converging to Input-Unconstrained Equilibrium",xlabel="Timestep", ylabel="Activity Value",sname="figures/input_unconstrained_nonlinear_activities_2")
    diffs = np.array(diffs)[:,:,0]
    plot_line_graph(diffs, title="Difference Between Activities and Target-Prop Targets", xlabel="Timestep", ylabel="Activity Difference",sname="figures/input_unconstrained_nonlinear_diffs_2")
    euclid_dists = np.mean(np.square(diffs),axis=1)
    plot_line_graph(euclid_dists, title="Mean Euclidean Distance From Target-Prop Targets", xlabel="Timestep", ylabel="Mean Distance from targets",label="Mean Distance",sname="figures/input_unconstrained_nonlinear_mean_diffs_2",divergence_graph=True)


def precision_equilibrium_check(pi2_scale=1, pi2_var=1, pi3_scale=1, pi3_var = 1,learning_rate = 0.1, activity_var = 1, weight_var = 0.05, dim=5):
    x1 = torch.tensor(np.random.normal(1,activity_var,(dim,1)))
    x2 = torch.tensor(np.random.normal(0.0,0.05,(dim,1)))
    x3 = torch.tensor(np.random.normal(-1,activity_var,(dim,1)))
    W1 = torch.tensor(np.random.normal(0,weight_var,(dim,dim)))
    W2 = torch.tensor(np.random.normal(0,weight_var,(dim,dim)))
    Pi2 = construct_precision_matrix(dim,pi2_scale,pi2_var)
    Pi3 = construct_precision_matrix(dim,pi3_scale,pi3_var)

    # compute precision equilibrium
    Pi2inv = torch.inverse(Pi2)
    prefactor = torch.inverse(torch.eye(dim) + Pi2inv @ (W2.T @ (Pi3 @ W2)))
    pred_eq =  prefactor @ (W1 @ x1) + prefactor @ (Pi2inv @ (W2.T @ (Pi3 @ x3)))

    # setup inference steps
    x2s = []
    Fs = []
    diffs = []
    with torch.no_grad():
        for i in range(100):
            e2 = Pi2 @ (x2 - W1 @ x1)
            e3 = Pi3 @ (x3 - W2 @ x2)
            x2 -= learning_rate * (e2 - W2.T @ e3)
            Fs.append(torch.sum(torch.square(e2)) + torch.sum(torch.square(e3)))
            x2s.append(deepcopy(x2.numpy()))
            diffs.append(deepcopy(x2.numpy())- pred_eq.numpy())

    plot_line_graph(Fs, title="Free Energy of the Network", xlabel="Timestep", ylabel="Free Energy",label="Free-Energy",sname="figures/check_precision_equilibrium_Fs")
    x2s = np.array(x2s)[:,:,0]
    plot_equilibrium_graph(x2s,pred_eq, title="Activities Converging to Precision Equilibrium",xlabel="Timestep", ylabel="Activity Value",sname="figures/check_precision_equilibrium_activities")
    diffs = np.array(diffs)[:,:,0]
    plot_line_graph(diffs, title="Difference of activities from Precision Equilibrium", xlabel="Timestep",ylabel="Activity Value",sname="figures/check_precision_equilibrium_diffs")

    total_diffs_from_eq = np.sum(np.square(diffs), axis=1)
    plot_line_graph(total_diffs_from_eq, title="Total Euclidean Distance from  Precision Equilibrium", xlabel="Timestep",ylabel="Euclidean Distance",label="Distance",sname="figures/check_precision_equilibrium_total_diffs",divergence_graph=True)

def low_precision_ratio_BP(pi2_scale=10, pi2_var=1, pi3_scale=1, pi3_var = 1,learning_rate = 0.1, activity_var = 1, weight_var = 0.05, dim=5):
    x1 = torch.tensor(np.random.normal(1,activity_var,(dim,1)))
    x2 = torch.tensor(np.random.normal(0.0,0.05,(dim,1)))
    x3 = torch.tensor(np.random.normal(-1,activity_var,(dim,1)))
    W1 = torch.tensor(np.random.normal(0,weight_var,(dim,dim)))
    W2 = torch.tensor(np.random.normal(0,weight_var,(dim,dim)))
    Pi2 = construct_precision_matrix(dim,pi2_scale,pi2_var)
    Pi3 = construct_precision_matrix(dim,pi3_scale,pi3_var)

    Pi2inv = torch.inverse(Pi2)
    prefactor = torch.inverse(torch.eye(dim) + Pi2inv @ (W2.T @ (Pi3 @ W2)))
    pred_eq =  prefactor @ (W1 @ x1) + prefactor @ (Pi2inv @ (W2.T @ (Pi3 @ x3)))

    # setup inference steps
    x2s = []
    Fs = []
    e2s = []
    e3 = Pi3 @ (x3 - W2 @ W1 @ x1)
    BP_e2s = deepcopy(W2.T @ e3)
    BP_diffs = []
    with torch.no_grad():
        for i in range(100):
            e2 = Pi2 @ (x2 - W1 @ x1)
            e3 = Pi3 @ (x3 - W2 @ x2)
            x2 -= learning_rate * (e2 - W2.T @ e3)
            Fs.append(torch.sum(torch.square(e2)) + torch.sum(torch.square(e3)))
            x2s.append(deepcopy(x2.numpy()))
            e2s.append(deepcopy(e2.numpy()))
            BP_diffs.append(e2.numpy() - BP_e2s.numpy())

    plot_line_graph(Fs, title="Free Energy of the Network", xlabel="Timestep", ylabel="Free Energy",label="Free-Energy",sname="figures/low_precision_ratio_Fs")
    x2s = np.array(x2s)[:,:,0]
    plot_equilibrium_graph(x2s,pred_eq, title="Activities Converging to Precision Equilibrium",xlabel="Timestep", ylabel="Activity Value",sname="figures/low_precision_ratio_activities")
    diffs = np.array(BP_diffs)[:,:,0]
    plot_line_graph(diffs, title="Difference of Prediction Errors from Backprop", xlabel="Timestep",ylabel="Prediction Error",sname="figures/low_precision_ratio_diffs")

    total_diffs = np.sum(np.square(diffs), axis=1)
    plot_line_graph(total_diffs, title="Total Euclidean Distance from Backprop Gradients", xlabel="Timestep",ylabel="Euclidean Distance",label="Distance",sname="figures/low_precision_ratio_total_diffs",divergence_graph=True)


def high_precision_ratio_TP(learning_rate = 0.1, activity_var = 1, weight_var = 0.05, dim=5):
    W1 = torch.tensor(np.random.normal(0,weight_var,(dim,dim)))
    W2 = torch.tensor(np.random.normal(0,weight_var,(dim,dim)))
    x1 = torch.tensor(np.random.normal(1,activity_var,(dim,1)))
    #x2 = torch.tensor(np.random.normal(0.0,0.05,(5,1)))
    x2 = W1 @ x1
    x3 = torch.tensor(np.random.normal(-1,activity_var,(dim,1)))

    prefactor = torch.inverse(torch.eye(dim) + W2.T @ W2)
    pred_eq =  prefactor @ (W1 @ x1) + prefactor @ (W2.T @ x3)
    #print(pred_eq.shape)
    #print(pred_eq)

    # setup inference steps
    FP_x2 = W1 @ x1
    TP_x2 = torch.inverse(W2) @ x3
    x2s = []
    Fs = []
    FP_angles = []
    TP_angles = []
    with torch.no_grad():
        for i in range(100):
            x2s.append(deepcopy(x2.numpy()))
            FP_angles.append(cosine_similarity(x2.reshape(dim,), FP_x2.reshape(dim,)))
            TP_angles.append(cosine_similarity(x2.reshape(dim,), TP_x2.reshape(dim,)))
            e2 = x2 - W1 @ x1
            e3 = x3 - W2 @ x2
            x2 -= learning_rate * (e2 - W2.T @ e3)
            Fs.append(torch.sum(torch.square(e2)) + torch.sum(torch.square(e3)))

    plot_line_graph(Fs, title="Free Energy of the Network", xlabel="Timestep", ylabel="Free Energy",label="Free-Energy",sname="figures/high_precision_ratio_Fs")
    x2s = np.array(x2s)[:,:,0]
    plot_equilibrium_graph(x2s,pred_eq, title="Activities Converging to Precision Equilibrium",xlabel="Timestep", ylabel="Activity Value",sname="figures/high_precision_ratio_activities")
    plot_line_graph(FP_angles, title="Angle to Initial Forward Pass During Convergence to Equilibrium", xlabel="Timestep", ylabel="Cosine Similarity",label="Similarity to Forward Pass",sname="figures/high_precision_ratio_FP_angles")
    plot_line_graph(TP_angles, title="Angle to Target-Prop Targets During Convergence to Equilibrium", xlabel="Timestep", ylabel="Cosine Similarity",label="Similarity to Target",sname="figures/high_precision_ratio_TP_angles")



def nonlinear_equilibrium_angles_diffs(learning_rate =0.1, weight_var = 0.05, activity_var = 1, dim =5):
    W1 = torch.tensor(np.random.normal(0,weight_var,(dim,dim)))
    W2 = torch.tensor(np.random.normal(0,weight_var,(dim,dim)))
    x1 = torch.tensor(np.random.normal(1,activity_var,(dim,1)))
    #x2 = torch.tensor(np.random.normal(0.0,0.05,(5,1)))
    x2 = W1 @ x1
    x3 = torch.tensor(np.random.normal(-1,activity_var,(dim,1)))

    f = torch.tanh
    f_inv = torch.tanh
    fderiv = tanh_deriv
    # setup inference steps
    FP_x2 = W1 @ x1
    TP_x2 = torch.inverse(W2) @ x3
    x2s = []
    Fs = []
    FP_angles = []
    TP_angles = []
    FP_diffs = []
    TP_diffs = []
    with torch.no_grad():
        for i in range(100):
            x2s.append(deepcopy(x2.numpy()))
            FP_angles.append(cosine_similarity(x2.reshape(dim,), FP_x2.reshape(dim,)))
            TP_angles.append(cosine_similarity(x2.reshape(dim,), TP_x2.reshape(dim,)))
            FP_diffs.append(torch.sum(torch.square(x2 - FP_x2)).item())
            TP_diffs.append(torch.sum(torch.square(x2 - TP_x2)).item())
            e2 = x2 - f(W1 @ x1)
            e3 = x3 - f(W2 @ x2)
            x2 -= learning_rate * (e2 - W2.T @ (e3 * fderiv(W2 @ x2)))
            Fs.append(torch.sum(torch.square(e2)) + torch.sum(torch.square(e3)))

    plot_line_graph(Fs, title="Free Energy of the Network", xlabel="Timestep", ylabel="Free Energy",label="Free-Energy",sname="figures/nonlinear_equilibrium_angle_diffs_Fs_2")
    x2s = np.array(x2s)[:,:,0]
    plot_line_graph(x2s, title="Activities Converging to Precision Equilibrium",xlabel="Timestep", ylabel="Activity Value",sname="figures/nonlinear_equilibrium_angle_diffs_activities_2")
    plot_line_graph(FP_angles, title="Angle to Initial Forward Pass During Convergence to Equilibrium", xlabel="Timestep", ylabel="Similarity",label="Similarity Forward Pass",sname="figures/nonlinear_equilibrium_angle_diffs_FP_angles_2")
    plot_line_graph(TP_angles, title="Angle to Target-Prop Targets During Convergence to Equilibrium", xlabel="Timestep", ylabel="Similarity",label="Similarity to Target",sname="figures/nonlinear_equilibrium_angle_diffs_TP_angles_2")
    plot_line_graph(FP_diffs, title="Total Euclidean Distance to Feedforward Pass Activities During Convergence to Equilibrium", xlabel="Timestep",ylabel="Total Distance",label="Distance",sname="figures/nonlinear_equilibrium_angle_diffs_FP_diffs_2")
    plot_line_graph(TP_diffs, title="Total Euclidean Distance to Target-Prop Targets During Convergence to Equilibrium", xlabel="Timestep", ylabel="Total Distance", label="Distance",sname="figures/nonlinear_equilibrium_angle_diffs_TP_diffs_2")


def high_precision_ratio_nonlinear(pi2_scale=1, pi2_var=1, pi3_scale=5,pi3_var=1,learning_rate = 0.05, weight_var = 0.5, activity_var = 0.5, dim=3):
    x1 = torch.tensor(np.random.normal(1,activity_var,(dim,1)))
    x3 = torch.tensor(np.random.normal(0,0.05,(dim,1))) # needs to be small as if goes out of [-1,1] nans
    W1 = torch.tensor(np.random.normal(0,weight_var,(dim,dim)))
    W2 = torch.tensor(np.random.normal(0,weight_var,(dim,dim)))
    Pi2 = construct_precision_matrix(dim,pi2_scale,pi2_var)
    Pi3 = construct_precision_matrix(dim,pi3_scale,pi3_var)

    f = torch.tanh
    f_inv = torch.atanh
    fderiv = tanh_deriv


    Pi2inv = torch.inverse(Pi2)
    prefactor = torch.inverse(torch.eye(dim) + Pi2inv @ (W2.T @ (Pi3 @ W2)))
    pred_eq =  prefactor @ (W1 @ x1) + prefactor @ (Pi2inv @ (W2.T @ (Pi3 @ x3)))

    x2 = f(W1 @ x1)

    FP_x2 = f(W1 @ x1)
    FP_x3 = f(W2 @ x2)
    TP_x2 = torch.inverse(W2) @ f_inv(x3)


    # setup inference steps
    learning_rate = 0.05
    x2s = []
    Fs = []
    FP_angles = []
    TP_angles = []
    diffs_FP = []
    diffs_TP = []
    with torch.no_grad():
        for i in range(500):
            diffs_FP.append(x2.numpy() - FP_x2.numpy())
            diffs_TP.append(x2.numpy() - TP_x2.numpy())
            FP_angles.append(cosine_similarity(x2.reshape(dim,), FP_x2.reshape(dim,)))
            TP_angles.append(cosine_similarity(x2.reshape(dim,), TP_x2.reshape(dim,)))
            e2 = Pi2 @ (x2 - f(W1 @ x1))
            e3 = Pi3 @ (x3 - f(W2 @ x2))
            x2 -= learning_rate * (e2 - W2.T @ (e3 * fderiv(W2 @ x2)))
            Fs.append(torch.sum(torch.square(e2)) + torch.sum(torch.square(e3)))
            x2s.append(deepcopy(x2.numpy()))

    plot_line_graph(Fs, title="Free Energy of the Network", xlabel="Timestep", ylabel="Free Energy",label="Free-Energy",sname="figures/high_precision_ratio_nonlinear_Fs")
    x2s = np.array(x2s)[:,:,0]
    diffs_FP = np.array(diffs_FP)[:,:,0]
    diffs_TP = np.array(diffs_TP)[:,:,0]
    plot_equilibrium_graph(x2s,pred_eq, title="Activities Converging to Precision Equilibrium",xlabel="Timestep", ylabel="Activity Value",sname="figures/high_precision_ratio_nonlinear_activities")
    plot_line_graph(FP_angles, title="Angle to Initial Forward Pass During Convergence to Equilibrium", xlabel="Timestep", ylabel="Angle",label="Angle to Forward Pass",sname="figures/high_precision_ratio_nonlinear_FP_angles")
    plot_line_graph(TP_angles, title="Angle to Target-Prop Targets During Convergence to Equilibrium", xlabel="Timestep", ylabel="Angle",label="Angle to Target",sname="figures/high_precision_ratio_nonlinear_TP_angles")
    total_diffs_FP = np.sum(np.square(diffs_FP), axis=1)
    total_diffs_TP = np.sum(np.square(diffs_TP), axis=1)   
    plot_line_graph(total_diffs_FP, title="Total Euclidean Distance to Feedforward Pass Activities During Convergence to Equilibrium", xlabel="Timestep",ylabel="Total Distance",label="Distance",sname="figures/high_precision_ratio_nonlinear_FP_total_diffs")
    plot_line_graph(total_diffs_TP, title="Total Euclidean Distance to Target-Prop Targets During Convergence to Equilibrium", xlabel="Timestep", ylabel="Total Distance", label="Distance",sname="figures/high_precision_ratio_nonlinear_TP_total_diffs")


def low_precision_ratio_nonlinear(pi2_scale=10, pi2_var=1, pi3_scale=1,pi3_var=1,learning_rate = 0.05, weight_var = 1, activity_var = 1, dim=5):
    x1 = torch.tensor(np.random.normal(1,activity_var,(dim,1)))
    x3 = torch.tensor(np.random.normal(0,0.05,(dim,1))) # needs to be small as if goes out of [-1,1] nans
    W1 = torch.tensor(np.random.normal(0,weight_var,(dim,dim)))
    W2 = torch.tensor(np.random.normal(0,weight_var,(dim,dim)))
    Pi2 = construct_precision_matrix(dim,pi2_scale,pi2_var)
    Pi3 = construct_precision_matrix(dim,pi3_scale,pi3_var)
    
    f = torch.tanh
    f_inv = torch.atanh
    fderiv = tanh_deriv
    Pi2inv = torch.inverse(Pi2)
    prefactor = torch.inverse(torch.eye(dim) + Pi2inv @ (W2.T @ (Pi3 @ W2)))
    pred_eq =  prefactor @ (W1 @ x1) + prefactor @ (Pi2inv @ (W2.T @ (Pi3 @ x3)))
    x2 = f(W1 @ x1)
    FP_x2 = f(W1 @ x1)
    FP_x3 = f(W2 @ x2)
    TP_x2 = torch.inverse(W2) @ f_inv(x3)

    # setup inference steps
    learning_rate = 0.1
    x2s = []
    Fs = []
    diffs_FP = []
    diffs_TP = []
    FP_angles = []
    TP_angles = []
    e3 = Pi3 @ (x3 - f(W2 @ x2))
    BP_e2 = W2.T @ (e3 * fderiv(W2 @ x2))
    BP_diffs = []
    with torch.no_grad():
        for i in range(100):
            diffs_FP.append(x2.numpy() - FP_x2.numpy())
            diffs_TP.append(x2.numpy() - TP_x2.numpy())
            FP_angles.append(cosine_similarity(x2.reshape(dim,), FP_x2.reshape(dim,)))
            TP_angles.append(cosine_similarity(x2.reshape(dim,), TP_x2.reshape(dim,)))
            e2 = Pi2 @ (x2 - f(W1 @ x1))
            e3 = Pi3 @ (x3 - f(W2 @ x2))
            x2 -= learning_rate * (e2 - W2.T @ (e3 * fderiv(W2 @ x2)))
            Fs.append(torch.sum(e2) + torch.sum(e3))
            x2s.append(deepcopy(x2.numpy()))
            BP_diffs.append(deepcopy(e2.numpy()) - BP_e2.numpy())

    plot_line_graph(Fs, title="Free Energy of the Network", xlabel="Timestep", ylabel="Free Energy",label="Free-Energy",sname="figures/low_precision_ratio_nonlinear_Fs")
    x2s = np.array(x2s)[:,:,0]
    diffs_FP = np.array(diffs_FP)[:,:,0]
    diffs_TP = np.array(diffs_TP)[:,:,0]
    plot_equilibrium_graph(x2s,pred_eq, title="Activities Converging to Precision Equilibrium",xlabel="Timestep", ylabel="Activity Value",sname="figures/low_precision_ratio_nonlinear_activities")
    plot_line_graph(FP_angles, title="Angle to Initial Forward Pass During Convergence to Equilibrium", xlabel="Timestep", ylabel="Angle",label="Angle to Forward Pass",sname="figures/low_precision_ratio_nonlinear_FP_angles")
    plot_line_graph(TP_angles, title="Angle to TP Targets During Convergence to Equilibrium", xlabel="Timestep", ylabel="Angle",label="Angle to Target",sname="figures/low_precision_ratio_nonlinear_TP_angles")
    total_diffs_FP = np.sum(np.square(diffs_FP), axis=1)
    total_diffs_TP = np.sum(np.square(diffs_TP), axis=1)   
    plot_line_graph(total_diffs_FP, title="Total Euclidean Distance to FF Activities During Convergence to Equilibrium", xlabel="Timestep",ylabel="Total Distance",label="Distance",sname="figures/low_precision_ratio_nonlinear_FP_total_diffs")
    plot_line_graph(total_diffs_TP, title="Total Euclidean Distance to TP Targets During Convergence to Equilibrium", xlabel="Timestep", ylabel="Total Distance", label="Distance",sname="figures/low_precision_ratio_nonlinear_TP_total_diffs")
    BP_diffs = np.array(BP_diffs)[:,:,0]
    total_BP_diffs = np.sum(np.square(BP_diffs),axis=1)
    plot_line_graph(total_BP_diffs, title="Total Euclidean Distance of Prediction Errors to Backprop Gradients",xlabel="Timestep", ylabel="Total Distance", label="Distance",sname="figures/low_precision_ratio_nonlinear_BP_total_diffs")


if __name__ == '__main__':
    verify_linear_equilibrium()
    input_unconstrained_linear()
    input_unconstrained_nonlinear()
    xss, Fs, diffs = multi_layer_input_unconstrained_linear(learning_rate = 0.05)
    multi_trial_input_unconstrained_linear()

    precision_equilibrium_check()
    low_precision_ratio_BP()
    high_precision_ratio_TP()
    nonlinear_equilibrium_angles_diffs()
    high_precision_ratio_nonlinear()
    low_precision_ratio_nonlinear()


