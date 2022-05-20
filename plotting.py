# General plotting functions
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import brewer2mpl

bmap = brewer2mpl.get_map("Set2", 'qualitative',7)
colors = bmap.mpl_colors

sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)


def plot_line_graph(vals, title="", xlabel="", ylabel="", label=None,sname=None,divergence_graph = False,save_format = "pdf"):
    fig = plt.figure(figsize=(12,10))
    sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    plt.grid(False)
    xs = np.arange(0,len(vals))
    if label is not None:
        plt.plot(xs,vals,label=label,linewidth="2")
    else:
        plt.plot(xs,vals,linewidth="2")
    sns.despine(left=False,top=True, right=True, bottom=False)
    plt.xlabel(xlabel,fontsize=25)
    plt.ylabel(ylabel,fontsize=25)
    if divergence_graph:
        # set 0 to bottom ylim
        plt.ylim([0, None])
    plt.title(title,fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    if label is not None:
        plt.legend(fontsize=25)
    plt.tight_layout()
    if sname is not None:
        plt.savefig(sname,format=save_format,bbox_inches = "tight", pad_inches = 0)
    plt.show()
    



def plot_line_graphs(vals, title="", xlabel="", ylabel="", labels=None,sname=None,divergence_graph = False,stds = None,save_format = "pdf"):
    N, num_lines = vals.shape
    fig = plt.figure(figsize=(12,10))
    sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    plt.grid(False)
    xs = np.arange(0,N)
    for i in range(num_lines):
        if labels is not None:
            plt.plot(xs,vals[:,i],label=labels[i],linewidth="2")
        else:
            plt.plot(xs,vals[:,i],linewidth="2")
        if stds is not None:
            std = stds[:,i]
            plt.fill_between(xs,vals[:,i] + std, vals[:,i] - std, alpha=0.4)
    sns.despine(left=False,top=True, right=True, bottom=False)
    plt.xlabel(xlabel,fontsize=25)
    plt.ylabel(ylabel,fontsize=25)
    if divergence_graph:
        # set 0 to bottom ylim
        plt.ylim([0, None])
    plt.title(title,fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    if labels is not None:
        plt.legend(fontsize=25)
    if sname is not None:
        plt.savefig(sname,format=save_format, bbox_inches = "tight", pad_inches = 0)
    plt.show()

def plot_equilibrium_graph(vals,eq_val, title="", xlabel="", ylabel="", label=None,sname=None,divergence_graph = False,save_format="pdf"):
    fig = plt.figure(figsize=(12,10))
    sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    plt.grid(False)
    eq_val = eq_val.reshape((len(eq_val),))
    eq_vals = np.array([eq_val for i in range(len(vals))])
    xs = np.arange(0,len(vals))
    assert len(eq_vals) == len(xs), "must be same length"
    for i in range(len(vals[0])):
        plt.plot(xs, vals[:,i], label="Activity", linewidth=2, color=colors[i])
        plt.plot(xs, eq_vals[:,i], linewidth="1", linestyle="--", label="Equilibrium Value",color = colors[i])

    sns.despine(left=False,top=True, right=True, bottom=False)
    plt.xlabel(xlabel,fontsize=25)
    plt.ylabel(ylabel,fontsize=25)
    plt.title(title,fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.legend(fontsize=25)
    if sname is not None:
        plt.savefig(sname,format=save_format, bbox_inches = "tight", pad_inches = 0)
    plt.show()

def plot_fixed_point_comparison_graph(x2s, x2_fps, eq_val,title="", xlabel="", ylabel="", sname=None, save_format="pdf"):
    fig,ax = plt.subplots(figsize=(12,10))#plt.figure(figsize=(12,10))
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
    plt.xlabel(xlabel,fontsize=25)
    plt.ylabel(ylabel,fontsize=25)
    plt.title(title,fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #legend = ax.legend(dummy_lines,["Fixed Point", "Iterative"])
    #ax.add_artist(legend)
    #fig, ax = plt.subplots()
    line_up, = ax.plot([], label='Line 2',linestyle="solid",color="black")
    line_down, = ax.plot([], label='Line 1',linestyle="dashed",color="black")
    ax.legend([line_up, line_down], ['Fixed Point', 'Iterative'],loc=1)
    if sname is not None:
        plt.savefig(sname, format = save_format, bbox_inches="tight", pad_inches = 0)
    plt.show()
    
    
    