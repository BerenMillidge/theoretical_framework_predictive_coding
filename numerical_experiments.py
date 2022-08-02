### Numerical experiments for the "Theoretical Foundation of Learning and Inference in Predictive Coding Networks" paper ###

import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
from plotting import *

USE_SNS_THEME = False

# Verify that the linear equilibrium derives is correct in practice
def verify_linear_equilibrium(dimension=5, var=1, weight_var = 0.05, learning_rate=0.1,dim=5, plot_graphs = True):
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

    x2s = np.array(x2s)[:,:,0]
    diffs_from_eq = np.array(diffs_from_eq)[:,:,0]
    total_diffs_from_eq = np.sum(np.square(diffs_from_eq), axis=1)
    if plot_graphs:
        plot_line_graph(Fs, title="Free Energy of the Network", xlabel="Timestep", ylabel="Free Energy",label="Free-Energy",sname="figures/verify_linear_equilibrium_Fs")
        plot_equilibrium_graph(x2s,pred_eq.numpy()[:,0], title="Activities Converging to Linear Equilibrium",xlabel="Timestep", ylabel="Activity Value",sname="figures/verify_linear_equilibrium_activities",label="Activity Value")
        plot_line_graph(diffs_from_eq, title="Difference of activities from Linear Equilibrium", xlabel="Timestep",ylabel="Euclidean Distance",sname="figures/verify_linear_equilibrium_diffs")
        plot_line_graph(total_diffs_from_eq, title="Total Euclidean Distance from Equilibrium", xlabel="Timestep",ylabel="Euclidean Distance",label="Distance",sname="figures/verify_linear_equilibrium_total_diffs",divergence_graph=True)
    return total_diffs_from_eq

def multiple_networks_linear_eq_convergence(N_networks = 50,dimension = 5, var=1, weight_var = 0.05, learning_rate = 0.1, dim=5):
    total_diffs = []
    for i in range(N_networks):
        total_diffs_from_eq = verify_linear_equilibrium(dimension=dimension, var=var,weight_var = weight_var, learning_rate = learning_rate,dim=dim, plot_graphs = False)
        total_diffs.append(np.array(total_diffs_from_eq))
    mean_total_diffs = np.mean(total_diffs, axis=0)
    std_total_diffs = np.std(total_diffs, axis=0) #/ np.sqrt(N_networks)
    plot_line_graph(mean_total_diffs, stds=std_total_diffs, title="Average Distance from Linear Equilibrium", xlabel="Timestep", ylabel="Mean Euclidean Distance", sname="figures/average_linear_equilibrium_diffs")
    



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

def multi_trial_input_unconstrained(N_trials = 200, use_nonlinear=False):
    diffss = []
    for n in range(N_trials):
        if use_nonlinear:
            xss, Fs, diffs = input_unconstrained_nonlinear(learning_rate = 0.05, plot_results = False)
        else:
            xss, Fs, diffs = multi_layer_input_unconstrained_linear(learning_rate = 0.05,plot_results = False)
        diffss.append(diffs)
    diffss = np.array(diffss)
    print(diffss.shape)
    means = np.mean(diffss, axis=0)
    stds = np.std(diffss, axis=0) / np.sqrt(N_trials)
    print(means.shape)
    print(stds.shape)
    N_layers = 5
    labels = ["Layer " + str(i+1) for i in range(N_layers)]
    linear_str = "nonlinear" if use_nonlinear else "linear"
    plot_line_graphs(means, title="Convergence of each layer to Target-Prop Targets", xlabel="Timestep", ylabel="Average distance to local target", labels=labels, stds = stds, sname="figures/avg_seeds_multilayer_input_unconstrained_" + str(linear_str) + "_diffs_100_2.png")

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
    return xss, Fs, diffs

def precision_equilibrium_check(pi2_scale=1, pi2_var=0.1, pi3_scale=1, pi3_var = 0.1,learning_rate = 0.1, activity_var = 1, weight_var = 0.01, dim=5, plot_graphs = True):
    x1 = torch.tensor(np.random.normal(1,activity_var,(dim,1)))
    x2 = torch.tensor(np.random.normal(0.0,0.05,(dim,1)))
    x3 = torch.tensor(np.random.normal(-1,activity_var,(dim,1)))
    W1 = torch.tensor(np.random.normal(0.5,weight_var,(dim,dim)))
    W2 = torch.tensor(np.random.normal(0.5,weight_var,(dim,dim)))
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

    x2s = np.array(x2s)[:,:,0]
    diffs = np.array(diffs)[:,:,0]
    total_diffs_from_eq = np.sum(np.square(diffs), axis=1)
    if plot_graphs:
        plot_line_graph(Fs, title="Free Energy of the Network", xlabel="Timestep", ylabel="Free Energy",label="Free-Energy",sname="figures/check_precision_equilibrium_Fs")
        plot_equilibrium_graph(x2s,pred_eq, title="Activities Converging to Precision Equilibrium",xlabel="Timestep", ylabel="Activity Value",sname="figures/check_precision_equilibrium_activities")
        plot_line_graph(diffs, title="Difference of activities from Precision Equilibrium", xlabel="Timestep",ylabel="Activity Value",sname="figures/check_precision_equilibrium_diffs")
        plot_line_graph(total_diffs_from_eq, title="Total Euclidean Distance from  Precision Equilibrium", xlabel="Timestep",ylabel="Euclidean Distance",label="Distance",sname="figures/check_precision_equilibrium_total_diffs",divergence_graph=True)
    return total_diffs_from_eq

def multiple_networks_precision_eq_convergence(N_networks = 50, var=1, weight_var = 0.05, learning_rate = 0.1, dim=5):
    total_diffs = []
    for i in range(N_networks):
        total_diffs_from_eq = precision_equilibrium_check(plot_graphs = False)
        total_diffs.append(np.array(total_diffs_from_eq))
    mean_total_diffs = np.mean(total_diffs, axis=0)
    std_total_diffs = np.std(total_diffs, axis=0) #/ np.sqrt(N_networks)
    plot_line_graph(mean_total_diffs, stds=std_total_diffs, title="Average Distance from Precision Equilibrium", xlabel="Timestep", ylabel="Mean Euclidean Distance", sname="figures/average_precision_equilibrium_diffs")
    


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



def precision_ratio_correlation(N_trials, precision_ratios, pi2_var=0.1, pi3_scale=1, pi3_var = 0.1,learning_rate = 0.1, activity_var = 0.1, weight_var = 0.1, dim=3, individual_plot_graphs = False, use_cosine_similarity = True,sname="precision_ratio_graph_1.", save_format="png"):
    precision_ratio_list_TP = []
    precision_ratio_list_BP = []
    for n in range(N_trials):
        print("Trial: ", n)
        BP_angle_list = []
        TP_angle_list = []
        for precision_ratio in precision_ratios:
            print("precision ratio: ", precision_ratio)
            pi2_scale = precision_ratio
            pi3_scale = 1/precision_ratio
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
            FP_x2 = W1 @ x1
            TP_x2 = torch.inverse(W2) @ x3
            BP_angles = []
            TP_angles = []
            with torch.no_grad():
                for i in range(100):
                    e2 = Pi2 @ (x2 - W1 @ x1)
                    e3 = Pi3 @ (x3 - W2 @ x2)
                    x2 -= learning_rate * (e2 - W2.T @ e3)
                    Fs.append(torch.sum(torch.square(e2)) + torch.sum(torch.square(e3)))
                    x2s.append(deepcopy(x2.numpy()))
                    e2s.append(deepcopy(e2.numpy()))
                    if use_cosine_similarity:
                        BP_angles.append(np.abs(cosine_similarity(e2.reshape(dim,), BP_e2s.reshape(dim,))))
                        TP_angles.append(np.abs(cosine_similarity(x2.reshape(dim,),TP_x2.reshape(dim,))))
                    else:
                        BP_angles.append(np.sum(np.square(e2.numpy() - BP_e2s.numpy())))
                        TP_angles.append(np.sum(np.square(x2.numpy() - TP_x2.numpy())))
            x2s = np.array(x2s)[:,:,0]
            #diffs = np.array(BP_diffs)[:,:,0]
            #total_diffs = np.sum(np.square(diffs), axis=1)
            if individual_plot_graphs:
                plot_line_graph(Fs, title="Free Energy of the Network", xlabel="Timestep", ylabel="Free Energy",label="Free-Energy",sname="figures/low_precision_ratio_Fs")
                plot_equilibrium_graph(x2s,pred_eq, title="Activities Converging to Precision Equilibrium",xlabel="Timestep", ylabel="Activity Value",sname="figures/low_precision_ratio_activities")
                #plot_line_graph(diffs, title="Difference of Prediction Errors from Backprop", xlabel="Timestep",ylabel="Prediction Error",sname="figures/low_precision_ratio_diffs")
                #plot_line_graph(total_diffs, title="Total Euclidean Distance from Backprop Gradients", xlabel="Timestep",ylabel="Euclidean Distance",label="Distance",sname="figures/low_precision_ratio_total_diffs",divergence_graph=True)
            BP_angle_list.append(np.array(BP_angles))
            TP_angle_list.append(np.array(TP_angles))
        precision_ratio_list_TP.append(np.array(TP_angle_list))
        precision_ratio_list_BP.append(np.array(BP_angle_list))
    precision_ratio_list_TP = np.array(precision_ratio_list_TP)
    precision_ratio_list_BP = np.array(precision_ratio_list_BP)
    np.save("precision_ratio_list_TP.npy", precision_ratio_list_TP)
    np.save("precision_ratio_list_BP.npy", precision_ratio_list_BP)
    print(precision_ratio_list_TP.shape)
    print(precision_ratio_list_BP.shape)
    final_TP = precision_ratio_list_TP[:,:,-1]
    #print(final_TP)
    for i in range(N_trials):
        print(final_TP[i,:])
    final_BP = precision_ratio_list_BP[:,:,-1]
    # special plot here
    mean_final_TP = np.mean(final_TP, axis=0)
    mean_final_TP = np.nan_to_num(mean_final_TP, copy=True, nan=1.0) # remove occasional nan values
    mean_final_TP[mean_final_TP == 0.0] = 1.0
    std_final_TP = np.std(final_TP, axis=0) / np.sqrt(N_trials)
    mean_final_BP = np.mean(final_BP, axis=0)
    std_final_BP = np.std(final_BP, axis=0) / np.sqrt(N_trials)
    xs = np.arange(0,len(mean_final_TP))
    fig = plt.figure(figsize=(12,10))
    plt.grid(False)
    start_idx = 3
    plt.plot(precision_ratios[start_idx:], mean_final_TP[start_idx:], label="Similarity to targetprop targets")
    plt.fill_between(precision_ratios[start_idx:], mean_final_TP[start_idx:]- std_final_TP[start_idx:], mean_final_TP[start_idx:] + std_final_TP[start_idx:], alpha=0.5)
    plt.plot(precision_ratios[start_idx:], mean_final_BP[start_idx:], label="Similarity to backprop gradients")
    plt.fill_between(precision_ratios[start_idx:], mean_final_BP[start_idx:]- std_final_BP[start_idx:], mean_final_BP[start_idx:]+ std_final_BP[start_idx:], alpha=0.5)
    sns.despine(left=False,top=True, right=True, bottom=False)
    plt.xlabel("Precision Ratio",fontsize=24)
    plt.ylabel("Similarity",fontsize=24)
    plt.title("Similarity of BP and TP by Precision Ratio",fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=25)
    if USE_SNS_THEME:
        sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    plt.savefig(sname + "." + save_format,format=save_format,bbox_inches = "tight", pad_inches = 0)
    plt.show()

def identity(x):
    return x

def ones(x):
    return torch.ones(x.shape)

def nonlinear_equilibrium_angles_diffs(learning_rate =0.1, weight_var = 0.5, activity_var = 1, dim =5, plot_graphs = True):
    #W1 = torch.tensor(np.random.normal(1,weight_var,(dim,dim)))
    #W2 = torch.tensor(np.random.normal(,weight_var,(dim,dim)))
    W1 = torch.eye(dim) + torch.tensor(np.random.normal(0,weight_var,(dim,dim)))
    W2 = torch.eye(dim) + torch.tensor(np.random.normal(0,weight_var,(dim,dim)))
    x1 = torch.tensor(np.random.normal(0,activity_var,(dim,1)))
    #x2 = torch.tensor(np.random.normal(0.0,0.05,(5,1)))
    x2 = W1 @ x1
    x3 = torch.tensor(np.random.normal(0,activity_var,(dim,1)))

    #f = torch.tanh
    #f_inv = torch.tanh
    #fderiv = tanh_deriv
    f = identity
    fderiv = ones
    # setup inference steps
    e3 = x3 - W2 @ W1 @ x1
    FP_x2 = W1 @ x1
    #TP_x2 = x2 + (0.01 * torch.inverse(W2) @ x3) # nudging
    TP_x2 = torch.inverse(W2) @ x3
    print(torch.inverse(W2))
    print(TP_x2)
    BP_e2 = deepcopy(W2.T @ e3)
    x2s = []
    Fs = []
    FP_angles = []
    TP_angles = []
    FP_diffs = []
    TP_diffs = []
    BP_angles = []
    BP_diffs = []
    with torch.no_grad():
        for i in range(100):
            x2s.append(deepcopy(x2.numpy()))
            #e2 = x2 - f(W1 @ x1)
            #e3 = x3 - f(W2 @ x2)
            #x2 -= learning_rate * (e2 - W2.T @ (e3 * fderiv(W2 @ x2)))
            #e2 = x2 - W1 @ x1
            #e3 = x3 - W2 @ x2
            #x2 -= learning_rate * (e2 - (W2.T @ e3))
            e2 = x2 - W1 @ x1
            e3 = x3 - W2 @ x2
            x2 -= learning_rate * (- W2.T @ e3)
            x1 -= learning_rate * (- W1.T @ e2)
            print(x2)
            print(TP_x2)
            Fs.append(torch.sum(torch.square(e2)) + torch.sum(torch.square(e3)))
            FP_angles.append(np.abs(cosine_similarity(x2.reshape(dim,), FP_x2.reshape(dim,))))
            TP_angles.append(np.abs(cosine_similarity(x2.reshape(dim,), TP_x2.reshape(dim,))))
            BP_angles.append(np.abs(cosine_similarity(e2.reshape(dim,), BP_e2.reshape(dim,))))
            
            FP_diffs.append(torch.sum(torch.square(x2 - FP_x2)).item())
            TP_diffs.append(torch.sum(torch.square(x2 - TP_x2)).item())
            BP_diffs.append(torch.sum(torch.square(e2 - BP_e2)).item())

    x2s = np.array(x2s)[:,:,0]
    if plot_graphs:
        plot_line_graph(Fs, title="Free Energy of the Network", xlabel="Timestep", ylabel="Free Energy",label="Free-Energy",sname="figures/nonlinear_equilibrium_angle_diffs_Fs_2")
        plot_line_graph(x2s, title="Activities Converging to Precision Equilibrium",xlabel="Timestep", ylabel="Activity Value",sname="figures/nonlinear_equilibrium_angle_diffs_activities_2")
        plot_line_graph(FP_angles, title="Angle to Initial Forward Pass During Convergence to Equilibrium", xlabel="Timestep", ylabel="Similarity",label="Similarity Forward Pass",sname="figures/nonlinear_equilibrium_angle_diffs_FP_angles_2")
        plot_line_graph(TP_angles, title="Angle to Target-Prop Targets During Convergence to Equilibrium", xlabel="Timestep", ylabel="Similarity",label="Similarity to Target",sname="figures/nonlinear_equilibrium_angle_diffs_TP_angles_2")
        plot_line_graph(FP_diffs, title="Total Euclidean Distance to Feedforward Pass Activities During Convergence to Equilibrium", xlabel="Timestep",ylabel="Total Distance",label="Distance",sname="figures/nonlinear_equilibrium_angle_diffs_FP_diffs_2")
        plot_line_graph(TP_diffs, title="Total Euclidean Distance to Target-Prop Targets During Convergence to Equilibrium", xlabel="Timestep", ylabel="Total Distance", label="Distance",sname="figures/nonlinear_equilibrium_angle_diffs_TP_diffs_2")
    return FP_angles, BP_angles, TP_angles, FP_diffs, BP_diffs, TP_diffs

def multiple_networks_nonlinear_angles(N_trials,sname="nonlinear_multinet_bp_tp_inference_evolution", save_format="png",learning_rate =0.1, weight_var = 0.05, activity_var = 1, dim =5):
    FP_angless = []
    BP_angless = []
    TP_angless = []
    for n in range(N_trials):
        FP_angles, BP_angles, TP_angles, FP_diffs, BP_diffs, TP_diffs = nonlinear_equilibrium_angles_diffs(learning_rate = learning_rate, weight_var = weight_var, activity_var = activity_var, dim =dim, plot_graphs = False)
        FP_angless.append(np.array(FP_angles))
        BP_angless.append(np.array(BP_angles))
        TP_angless.append(np.array(TP_angles))
    FP_angless = np.array(FP_angless)
    BP_angless = np.array(BP_angless)
    TP_angless = np.array(TP_angless)

    mean_BP_angles = np.mean(BP_angless, axis=0)
    mean_TP_angles = np.mean(TP_angless, axis=0)
    std_BP_angles = np.std(BP_angless, axis=0) / np.sqrt(N_trials)
    std_TP_angles = np.std(TP_angless, axis=0) / np.sqrt(N_trials)
    fig = plt.figure(figsize=(12,10))
    plt.grid(False)
    xs = np.arange(0, len(mean_BP_angles))
    plt.plot(xs, mean_TP_angles, label="Similarity to targetprop targets")
    plt.fill_between(xs, mean_TP_angles - std_TP_angles, mean_TP_angles + std_TP_angles, alpha=0.5)
    plt.plot(xs, mean_BP_angles, label="Similarity to backprop gradients")
    plt.fill_between(xs, mean_BP_angles - std_BP_angles, mean_BP_angles + std_BP_angles, alpha=0.5)
    sns.despine(left=False,top=True, right=True, bottom=False)
    plt.xlabel("Inference step",fontsize=24)
    plt.ylabel("Similarity",fontsize=24)
    plt.title("Similarity to BP and TP throughout inference",fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=25)
    if USE_SNS_THEME:
        sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    plt.tight_layout()
    plt.savefig(sname + "." + save_format,format=save_format,bbox_inches = "tight", pad_inches = 0)
    plt.show()

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
    precision_ratios = [0.00001,0.0001,0.0003,0.0006,0.001,0.003,0.006,0.01,0.02,0.05,0.07,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]#,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.2,2.5,2.7,3.0]
    precision_ratio_correlation(N_trials=100,precision_ratios=precision_ratios)
    
    multiple_networks_linear_eq_convergence()
    multi_trial_input_unconstrained(use_nonlinear = False)
    multi_trial_input_unconstrained(use_nonlinear = True)
    verify_linear_equilibrium()
    input_unconstrained_linear()
    input_unconstrained_nonlinear()
    xss, Fs, diffs = multi_layer_input_unconstrained_linear(learning_rate = 0.05)

    precision_equilibrium_check()
    multiple_networks_precision_eq_convergence()
    multiple_networks_nonlinear_angles(N_trials = 50)
    low_precision_ratio_BP()
    high_precision_ratio_TP()
    nonlinear_equilibrium_angles_diffs()
    high_precision_ratio_nonlinear()
    low_precision_ratio_nonlinear()
