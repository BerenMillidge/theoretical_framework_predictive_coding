# General plotting functions
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import brewer2mpl

bmap = brewer2mpl.get_map("Set2", 'qualitative',7)
colors = bmap.mpl_colors

USE_SNS_THEME = False


def plot_line_graph(vals, title="", xlabel="", ylabel="", label=None,sname=None,use_legend = False,divergence_graph = False,save_format = "png"):
    fig = plt.figure(figsize=(12,10))
    if USE_SNS_THEME:
        sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    plt.grid(False)
    xs = np.arange(0,len(vals))
    if label is not None:
        plt.plot(xs,vals,label=label,linewidth=3)
    else:
        plt.plot(xs,vals,linewidth=3)
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

    
    