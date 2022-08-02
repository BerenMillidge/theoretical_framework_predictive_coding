# General plotting functions
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import brewer2mpl
from pcn_learning import *
from utils import *

bmap = brewer2mpl.get_map("Set2", 'qualitative',7)
colors = bmap.mpl_colors

USE_SNS_THEME = False


def plot_line_graph(vals,stds=None, title="", xlabel="", ylabel="", label=None,sname=None,use_legend = False,divergence_graph = False,save_format = "png"):
    fig = plt.figure(figsize=(12,10))
    if USE_SNS_THEME:
        sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    plt.grid(False)
    xs = np.arange(0,len(vals))
    if label is not None:
        plt.plot(xs,vals,label=label,linewidth=3)
    else:
        plt.plot(xs,vals,linewidth=3)
    if stds is not None:
        plt.fill_between(xs, vals - stds, vals+ stds, alpha=0.5)
    sns.despine(left=False,top=True, right=True, bottom=False)
    plt.xlabel(xlabel,fontsize=24)
    plt.ylabel(ylabel,fontsize=24)
    if divergence_graph:
        # set 0 to bottom ylim
        plt.ylim([0, None])
    plt.title(title,fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    if use_legend:
        plt.legend(fontsize=25)
    #plt.tight_layout()
    if sname is not None:
        plt.savefig(sname + "." + save_format,format=save_format,bbox_inches = "tight", pad_inches = 0)
    plt.show()
    



def plot_line_graphs(vals, title="", xlabel="", ylabel="", labels=None,sname=None,divergence_graph = False,stds = None,save_format = "png"):
    N, num_lines = vals.shape
    fig = plt.figure(figsize=(12,10))
    if USE_SNS_THEME:
        sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    plt.grid(False)
    xs = np.arange(0,N)
    for i in range(num_lines):
        if labels is not None:
            plt.plot(xs,vals[:,i],label=labels[i],linewidth=3)
        else:
            plt.plot(xs,vals[:,i],linewidth=3)
        if stds is not None:
            std = stds[:,i]
            plt.fill_between(xs,vals[:,i] + std, vals[:,i] - std, alpha=0.4)
    sns.despine(left=False,top=True, right=True, bottom=False)
    plt.xlabel(xlabel,fontsize=24)
    plt.ylabel(ylabel,fontsize=24)
    if divergence_graph:
        # set 0 to bottom ylim
        plt.ylim([0, None])
    plt.title(title,fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    if labels is not None:
        plt.legend(fontsize=25)
    if sname is not None:
        plt.savefig(sname + "." + save_format,format=save_format, bbox_inches = "tight", pad_inches = 0)
    plt.show()

def plot_equilibrium_graph(vals,eq_val, title="", xlabel="", ylabel="", label=None,use_legend = False, sname=None,divergence_graph = False,save_format="png"):
    fig = plt.figure(figsize=(12,10))
    if USE_SNS_THEME:
        sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    plt.grid(False)
    eq_val = eq_val.reshape((len(eq_val),))
    print(eq_val.shape)
    print(len(vals))
    eq_vals = np.zeros([len(vals),len(eq_val)])
    for i in range(len(vals)):
        eq_vals[i,:] = eq_val
    xs = np.arange(0,len(vals))
    assert len(eq_vals) == len(xs), "must be same length"
    for i in range(len(vals[0])):
        plt.plot(xs, vals[:,i], label="Activity", linewidth=2, color=colors[i])
        plt.plot(xs, eq_vals[:,i], linewidth=3, linestyle="--", label="Equilibrium Value",color = colors[i])

    sns.despine(left=False,top=True, right=True, bottom=False)
    plt.xlabel(xlabel,fontsize=24)
    plt.ylabel(ylabel,fontsize=24)
    plt.title(title,fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    if use_legend:
        plt.legend(fontsize=25)
    if sname is not None:
        plt.savefig(sname + "." + save_format,format=save_format, bbox_inches = "tight", pad_inches = 0)
    plt.show()

def plot_fixed_point_comparison_graph(x2s, x2_fps, eq_val,title="", xlabel="", ylabel="", sname=None, use_legend = True, save_format="png"):
    fig,ax = plt.subplots(figsize=(12,10))#plt.figure(figsize=(12,10))
    if USE_SNS_THEME:
        sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    plt.grid(False)
    eq_val = eq_val.reshape((len(eq_val),))
    eq_vals = np.array([eq_val for i in range(len(x2s))])
    xs = np.arange(0, len(x2s))
    #dummy_lines = [ax.plot([],[], label="Fixed Point", linestyle="solid",color="black"), ax.plot([],[], label="Iterative", linestyle="dashed", color="black")]
    print(x2s.shape)
    for i in range(x2s.shape[1]):
        plt.plot(xs, x2_fps[:,i], linestyle="solid", label="Fixed Point",color=colors[i])
        plt.plot(xs, x2s[:,i], linestyle="dashed", label="Iterative",color=colors[i])
        #plt.plot(xs, eq_vals, linestyle="dotted", label="Equilibrium Value")
    sns.despine(left=False,top=True, right=True, bottom=False)
    plt.xlabel(xlabel,fontsize=30)
    plt.ylabel(ylabel,fontsize=30)
    plt.title(title,fontsize=34)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    #legend = ax.legend(dummy_lines,["Fixed Point", "Iterative"])
    #ax.add_artist(legend)
    #fig, ax = plt.subplots()
    line_up, = ax.plot([], label='Line 2',linestyle="solid",color="black")
    line_down, = ax.plot([], label='Line 1',linestyle="dashed",color="black")
    if use_legend:
        ax.legend([line_up, line_down], ['Fixed Point', 'Iterative'],loc=1)
    if sname is not None:
        plt.savefig(sname + "." + save_format, format = save_format, bbox_inches="tight", pad_inches = 0)
    plt.show()
    
    
### graphs for the pcn learning plotting experiment from data ###
def plot_loss_diff_in_inference():
    loss_diff_load_list = np.load("data/loss_diff_list_2.npy")
    N = len(loss_diff_load_list)
    print(loss_diff_load_list.shape)
    mean_loss_diffs = np.mean(loss_diff_load_list,axis=0).reshape(len(loss_diff_load_list[0,:]))
    std_loss_diffs = np.std(loss_diff_load_list, axis=0).reshape(len(loss_diff_load_list[0,:])) / np.sqrt(N)
    xs = np.arange(0, len(mean_loss_diffs))
    
    fig = plt.figure(figsize=(12,10))
    if USE_SNS_THEME:
        sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    plt.grid(False)
    plt.plot(xs,mean_loss_diffs, linewidth=2, label="Mean loss difference")
    plt.fill_between(xs, mean_loss_diffs - std_loss_diffs, mean_loss_diffs + std_loss_diffs, alpha=0.5)
    plt.hlines(0, xmin=xs[0], xmax=xs[-1], color="gray", linestyle="dashed",label="loss increasing")
    sns.despine(left=False,top=True, right=True, bottom=False)
    plt.xlabel("Training batch",fontsize=24)
    plt.ylabel("Mean loss change",fontsize=24)
    plt.title("Change in backprop loss during inference",fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=25)
    plt.tight_layout()
    plt.savefig("loss_differences_figure.png", bbox_inches = "tight", pad_inches = 0, save_format = "png")
    plt.show()
    
def plot_gradient_angles():
    angle_list = np.load("data/Gradient_angles_list_2.npy")
    N = len(angle_list)
    mean_angle_list = np.mean(angle_list, axis=0)
    std_angle_list = np.std(angle_list, axis=0) / np.sqrt(N)
    #print(std_angle_list.shape)
    xs = np.arange(0,angle_list.shape[-1])
    fig = plt.figure(figsize=(12,10))
    if USE_SNS_THEME:
        sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    plt.grid(False)
    for l in range(len(mean_angle_list)):
        angles = mean_angle_list[l,:]
        std_angles = std_angle_list[l,:]
        #print(angles.shape)
        #print(std_angles.shape)
        plt.plot(xs, angles, label="Layer " + str(l),alpha=0.5)
        plt.fill_between(xs, angles-std_angles, angles + std_angles, alpha=0.5)
    plt.xlabel("Training batch",fontsize=24)
    plt.ylabel("Mean gradient similarity",fontsize=24)
    sns.despine(left=False,top=True, right=True, bottom=False)
    plt.title("Gradient similarity by layer throughout training",fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=25)
    plt.tight_layout()
    plt.savefig("all_layer_gradient_angles.png", bbox_inches = "tight", pad_inches = 0, save_format = "png")
    plt.show()
    
    
def plot_convergence_full_batch():
    gradient_angles = np.load("data/full_batch_gradient_angles_list.npy")
    norms_bp = np.load("data/full_batch_gradient_norm_list_bp.npy")[0,:]
    norms_pc = np.load("data/full_batch_gradient_norm_list_pc.npy")[0,:]
    train_losses = np.load("data/full_batch_train_loss.npy")[0,:]
    gradient_angles_single = np.load("data/small_full_batch_gradient_angles_list.npy")
    norms_bp_single = np.load("data/small_full_batch_gradient_norm_list_bp.npy")[0,:]
    norms_pc_single = np.load("data/small_full_batch_gradient_norm_list_pc.npy")[0,:]
    train_losses_single = np.load("data/small_full_batch_train_loss.npy")[0,:]
    N_diff = len(norms_bp) - len(norms_bp_single)
    norms_bp_single = np.array(list(norms_bp_single) + [norms_bp_single[-1] for i in range(N_diff)]) # repeat fixed end till end of main one
    norms_pc_single = np.array(list(norms_pc_single) + [norms_pc_single[-1] for i in range(N_diff)])
    train_losses_single = np.array(list(train_losses_single) + [train_losses_single[-1] for i in range(N_diff)])

    print(norms_bp.shape)
    xs = np.arange(0, len(norms_bp))
    #xs_single = np.arange(0, len(norms_bp_single))
    fig = plt.figure(figsize=(12,10))
    if USE_SNS_THEME:
        sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    plt.grid(False)
    plt.plot(xs, train_losses, label="Training loss (500)",color = colors[1])
    plt.plot(xs, norms_bp, label="BP gradient norm (500)",color = colors[2])
    plt.plot(xs, norms_pc, label="PC gradient_norm (500)",color=colors[3])
    plt.plot(xs, train_losses_single, label="Training loss (1)",linestyle="dashed",color = colors[1])
    plt.plot(xs, norms_bp_single, label="BP gradient norm (1)",linestyle="dashed",color = colors[2])
    plt.plot(xs, norms_pc_single, label="PC gradient_norm (1)",linestyle="dashed",color = colors[3])
    plt.yscale('log')
    sns.despine(left=False,top=True, right=True, bottom=False)
    plt.xlabel("Epoch",fontsize=24)
    plt.ylabel("Log gradient norm",fontsize=24)
    plt.title("Gradient norm throughout full-batch training",fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=25, loc='lower right')
    plt.tight_layout()
    plt.savefig("single_full_batch_gradient_norm.png", bbox_inches = "tight", pad_inches = 0, save_format = "png")
    plt.show()
    
def pc_bp_accuracy_plot():
    pc_train_acc = np.load("data/reconf_train_accs_list_avg_2.npy")[0,:]
    pc_test_acc = np.load("data/reconf_test_accs_list_avg_2.npy")[0,:]
    bp_train_acc = np.load("data/backprop_train_accs_list_avg.npy")[0,:]
    bp_test_acc = np.load("data/backprop_test_accs_list_avg.npy")[0,:]
    print(pc_test_acc)
    print(bp_test_acc)
    xs_train = np.arange(0, len(pc_train_acc))
    fig = plt.figure(figsize=(12,10))
    if USE_SNS_THEME:
        sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    plt.grid(False)
    xs_test = np.array([(i+1) * 938 for i in range(10)])
    plt.plot(xs_train,pc_train_acc, alpha=0.9,label="PC train accuracy")
    #plt.plot(xs_test, pc_test_acc,color=colors[0], linestyle="dashed", label="PC test accuracy",linewidth=3)
    plt.plot(xs_train,bp_train_acc,alpha=0.9,label="BP train accuracy",)
    #plt.plot(xs_test, bp_test_acc,color=colors[1], linestyle="dashed", label="BP test accuracy",linewidth=3)
    sns.despine(left=False,top=True, right=True, bottom=False)
    plt.xlabel("Batch number",fontsize=24)
    plt.ylabel("Training accuracy",fontsize=24)
    plt.title("MNIST accuracy",fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=25, loc='lower right')
    plt.tight_layout()
    plt.savefig("mnist_accuracy_pc_bp.png", bbox_inches = "tight", pad_inches = 0, save_format = "png")
    plt.show()

def plot_energy_evolution():
    epochs = 15,
    DEVICE = "cpu"
    batch_size=1
    mu_dt=0.1
    lr=0.003
    N_reconf_steps = 100
    use_backprop = False
    clamp_val = 1000
    use_PPC = False
    store_gradient_angle = False
    momentum = 0.9
    trainloader, valloader = get_mnist_dataset(batch_size = batch_size)
    
    model = NN_model(input_size, hidden_sizes, output_size, batch_size,device=DEVICE)

    w1, b1 = list(model.fc1.parameters())
    w2, b2 = list(model.fc2.parameters())
    w3,b3 = list(model.fc3.parameters())

    pc_fc1 = PCLayer2_GPU("cpu", linear_base_function,[deepcopy(w1.detach()),deepcopy(b1.detach())],f=torch.relu)
    pc_fc2 = PCLayer2_GPU("cpu", linear_base_function,  [deepcopy(w2.detach()),deepcopy(b2.detach())],f=torch.relu)
    pc_fc3 = PCLayer2_GPU("cpu", linear_base_function, [deepcopy(w3.detach()),deepcopy(b3.detach())],f=nn.LogSoftmax(dim=1))
    pcnet = PC_Net2_GPU([pc_fc1, pc_fc2, pc_fc3],batch_size, mu_dt = mu_dt, lr=lr,N_reconf_steps =N_reconf_steps,use_backprop=use_backprop, clamp_val=clamp_val,use_PPC = use_PPC,store_gradient_angle = store_gradient_angle, device=DEVICE)

    #criterion = nn.NLLLoss()
    #criterion = nn.MSELoss()
    def mse_loss(x,y):
      return 0.5 * torch.sum(torch.sum(torch.square(x-y),dim=[1]),dim=[0])
  
    criterion = mse_loss

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer_pc = optim.SGD(pcnet.get_parameters(), lr=lr, momentum=momentum)
    pcnet.init_optimizer(optimizer_pc)
    optimizer.zero_grad()
    time0 = time()
    for (i, (images, labels)) in enumerate(trainloader):
        
        images = images.view(images.shape[0], -1).to(DEVICE)
        labels = F.one_hot(labels,10).to(DEVICE).float()
        print("images: ", images.shape)
        print("labels: ", labels.shape)

        # Training pass
        optimizer.zero_grad()
        output = pcnet.forward(images)
        loss = criterion(output, labels)
        print("loss: ", loss.item())
        #acc = accuracy(output, labels)
        es, dparamss, ess, loss_diff, diff_F = pcnet.gradient_infer(images, labels, loss_fn = criterion, store_errors = True)
        print(len(ess[0]))
        Fs = []
        out_Ls = []
        E_tildes = []
        for t in range(len(ess)):
            es = ess[t]
            E_tilde = torch.sum(torch.square(es[1])) + torch.sum(torch.square(es[2])) 
            out_l = torch.sum(torch.square(es[3])).item() - 58
            print(len(es))
            Fe = E_tilde.item() + out_l
            Fs.append(Fe)
            out_Ls.append(out_l)
            E_tildes.append(E_tilde.item())

        print("FS")
        fig = plt.figure(figsize=(12,10))
        if USE_SNS_THEME:
            sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
        plt.grid(False)
        plt.plot(Fs,label="Free energy", linewidth=3, linestyle="dotted")
        plt.plot(out_Ls,label = "Output loss", linewidth=3, linestyle="solid")
        plt.plot(E_tildes,label="Residual energy",linewidth=3, linestyle="dashed")
        sns.despine(left=False,top=True, right=True, bottom=False)
        plt.xlabel("Inference step",fontsize=24)
        plt.ylabel("Energy value",fontsize=24)
        plt.title("Evolution of the three energies during inference",fontsize=26)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(fontsize=25, loc='lower right')
        plt.tight_layout()
        plt.savefig("energies_evolution_proper.png", bbox_inches = "tight", pad_inches = 0, save_format = "png")
        plt.show()
        return
    

if __name__ == '__main__':
    plot_loss_diff_in_inference()
    plot_gradient_angles()
    plot_convergence_full_batch()
    pc_bp_accuracy_plot()
    plot_energy_evolution()