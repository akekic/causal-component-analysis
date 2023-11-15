# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: CauCA
#     language: python
#     name: cauca
# ---

# # Plots for Nonparametric Identifiability of Causal Representations from Unknown Interventions

# +
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec

import numpy as np

BLUE = "#1A85FF"
RED = "#D0021B"
color_list = ["#5790fc", "#f89c20", "#e42536",  "#964a8b", "#9c9ca1", "#7a21dd"]

mm = 1/(10 * 2.54)  # millimeters in inches
SINGLE_COLUMN = 85 * mm
DOUBLE_COLUMN = 174 * mm
FONTSIZE = 9

style_modifications = {
    "font.size": FONTSIZE,
    "axes.titlesize": FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
    "font.family": "sans-serif", # set the font globally
    "font.sans-serif": "Helvetica", # set the font name for a font family
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "errorbar.capsize": 2.5, 
}
plt.rcParams.update(style_modifications)


# + code_folding=[0]
def format_violin(vp, facecolor=BLUE):
    for el in vp["bodies"]:
        el.set_facecolor(facecolor)
        el.set_edgecolor("black")
        el.set_linewidth(0.75)
        el.set_alpha(0.9)
    for pn in ["cbars", "cmins", "cmaxes", "cmedians"]:
        vp_ = vp[pn]
        vp_.set_edgecolor("black")
        vp_.set_linewidth(0.5)


# + code_folding=[0]
def create_violinplot(groups, xlabel, ylabel, xticklabels, filename=None, ax=None, colors=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        pass
#         ax = ax.twinx()

    vp = ax.violinplot(groups, showmedians=True)
    format_violin(vp, BLUE)
    if colors:
        for pc, color in zip(vp['bodies'], colors):
            pc.set_facecolor(color)
    
    if xticklabels is not None:
        ax.set_xticks(np.arange(1, len(xticklabels) + 1))
        ax.set_xticklabels(xticklabels)
    else:
        ax.tick_params(bottom=False, labelbottom=False)
    # ax.set_xticks(xticks)
    # plt.locator_params(axis='y', nbins=5)
    # plt.yticks(fontsize=24)
    # plt.ylim([0, 0.5])
    ax.set_ylabel(ylabel)
    # ax.set_xlabel(xlabel)
    if filename is not None:
        plt.savefig(f"{filename}.svg")
    return ax


# -

# ## Figure 3

# +
df_cc3_2vars = pd.read_csv("data/3_nonlin_cauca_2vars_10scm_min2.csv")
df_cc3_dm_2vars = pd.read_csv("data/3_nonlin_cauca_misspec_2vars_10scm_min2.csv")
df_cc3_im_2vars = pd.read_csv("data/3_nonlin_cauca_int_misspec_2vars_10scm_min2.csv")
df_cc3_bm_2vars = pd.read_csv("data/3_nonlin_cauca_both_misspec_2vars_10scm_min2.csv")

df_cc3_dm_2vars = df_cc3_dm_2vars.merge(
    df_cc3_2vars[["seed", "val_loss", "test_loss"]]
    .sort_values("val_loss", ascending=True)
    .drop_duplicates(["seed"])
    .rename(columns={"val_loss": "val_loss_cauca", "test_loss": "test_loss_cauca"}),
    on=["seed"],
)

df_cc3_im_2vars = df_cc3_im_2vars.merge(
    df_cc3_2vars[["seed", "val_loss", "test_loss"]]
    .sort_values("val_loss", ascending=True)
    .drop_duplicates(["seed"])
    .rename(columns={"val_loss": "val_loss_cauca", "test_loss": "test_loss_cauca"}),
    on=["seed"],
)

df_cc3_bm_2vars = df_cc3_bm_2vars.merge(
    df_cc3_2vars[["seed", "val_loss", "test_loss"]]
    .sort_values("val_loss", ascending=True)
    .drop_duplicates(["seed"])
    .rename(columns={"val_loss": "val_loss_cauca", "test_loss": "test_loss_cauca"}),
    on=["seed"],
)

# ground truth log prob
df_cc3_dm_2vars = df_cc3_dm_2vars.drop(columns=["val_loss_gt"]).merge(
    df_cc3_2vars[["seed", "training_seed", "val_loss_gt"]], on=["seed", "training_seed"]
)
df_cc3_im_2vars = df_cc3_im_2vars.drop(columns=["val_loss_gt"]).merge(
    df_cc3_2vars[["seed", "training_seed", "val_loss_gt"]], on=["seed", "training_seed"]
)
df_cc3_bm_2vars = df_cc3_bm_2vars.drop(columns=["val_loss_gt"]).merge(
    df_cc3_2vars[["seed", "training_seed", "val_loss_gt"]], on=["seed", "training_seed"]
)

# choose runs with lowest validation loss
df_cc3_2vars = df_cc3_2vars.loc[df_cc3_2vars.groupby("seed")["val_loss"].idxmin()]
df_cc3_dm_2vars = df_cc3_dm_2vars.loc[
    df_cc3_dm_2vars.groupby("seed")["val_loss"].idxmin()
]
df_cc3_im_2vars = df_cc3_im_2vars.loc[
    df_cc3_im_2vars.groupby("seed")["val_loss"].idxmin()
]
df_cc3_bm_2vars = df_cc3_bm_2vars.loc[
    df_cc3_bm_2vars.groupby("seed")["val_loss"].idxmin()
]

# +
from matplotlib.ticker import MaxNLocator

fig, axes = plt.subplots(
    1, 2, figsize=(DOUBLE_COLUMN, 0.3 * DOUBLE_COLUMN), dpi=500, tight_layout=True
)

data_mcc = [
    df_cc3_2vars["test_mcc"],
    df_cc3_im_2vars["test_mcc"],
    df_cc3_dm_2vars["test_mcc"],
    df_cc3_bm_2vars["test_mcc"],
]
data_loss = [
    (df_cc3_im_2vars["val_loss"] - df_cc3_im_2vars["val_loss_cauca"]),
    (df_cc3_dm_2vars["val_loss"] - df_cc3_dm_2vars["val_loss_cauca"]),
    (df_cc3_bm_2vars["val_loss"] - df_cc3_bm_2vars["val_loss_cauca"]),
]
xticklabels = [
    r"$G' = G$," + "\naligned\ntargets",
    r"$G' = G$," + "\nmisaligned\ntargets",
    r"$G' \ne G$," + "\naligned\ntargets",
    r"$G' \ne G$," + "\nmisaligned\ntargets",
]
axes[0].axhline(1, ls="--", color="gray")
axes[0] = create_violinplot(
    data_mcc,
    xlabel=None,
    ylabel="MCC",
    xticklabels=xticklabels,
    ax=axes[0],
    colors=color_list[: len(data_mcc)],
)
axes[0].tick_params(axis="x", labelrotation=0)
axes[0].set_title("Identifiability Scores")

axes[1] = create_violinplot(
    data_loss,
    xlabel=None,
    ylabel=r"$\Delta$ log-likelihood",
    xticklabels=xticklabels[1:],
    ax=axes[1],
    colors=color_list[1 : len(data_loss)+1],
)
axes[1].tick_params(axis="x", labelrotation=0)
axes[1].axhline(0, ls="--", color="gray")
axes[1].set_title("Model Fits")


# Set y-ticks to 5 for both subplots
axes[0].set_yticks([0.55, 0.7, 0.85, 1.00])
axes[1].yaxis.set_major_locator(MaxNLocator(5))

plt.savefig("plots/crl_experiments.pdf")
plt.show()
# -

# ## Figure 4

# +
df_3vars_012 = pd.read_csv("data/3var_snr10_012.csv")  # well-specified
df_3vars_021 = pd.read_csv("data/3var_snr10_021.csv")
df_3vars_102 = pd.read_csv("data/3var_snr10_102.csv")
df_3vars_120 = pd.read_csv("data/3var_snr10_120.csv")
df_3vars_201 = pd.read_csv("data/3var_snr10_201.csv")
df_3vars_210 = pd.read_csv("data/3var_snr10_210.csv")

misspecified_dfs = [df_3vars_021, df_3vars_102, df_3vars_120, df_3vars_201, df_3vars_210]
all_dfs = [df_3vars_012,] + misspecified_dfs

for i in range(len(misspecified_dfs)):
    df = misspecified_dfs[i]
    df = df.merge(
        df_3vars_012[["seed", "val_loss"]]
        .sort_values("val_loss", ascending=True)
        .drop_duplicates(["seed"])
        .rename(columns={"val_loss": "val_loss_cauca"}),
        on=["seed"],
    )
    # ground truth log prob
    df = df.drop(columns=["val_loss_gt"]).merge(
        df_3vars_012[["seed", "training_seed", "val_loss_gt"]], on=["seed", "training_seed"]
    )
    misspecified_dfs[i] = df

# choose runs with lowest validation loss
for i in range(len(all_dfs)):
    df = all_dfs[i]
    df = df.loc[df.groupby("seed")["val_loss"].idxmin()]
    all_dfs[i] = df

# +
fig, axes = plt.subplots(
    1, 2, figsize=(DOUBLE_COLUMN, 0.4 * DOUBLE_COLUMN), dpi=500, tight_layout=True
)

data_mcc = [df["test_mcc"] for df in all_dfs]
data_loss = [(df["val_loss"] - df["val_loss_cauca"]) for df in misspecified_dfs]

xticklabels = [
    r"123$^*$", r"132$^{(*)}$", r"213", r"231", r"312", r"321"
]
axes[0].axhline(1, ls="--", color="gray")
axes[0] = create_violinplot(
    data_mcc,
    xlabel=None,
    ylabel="MCC",
    xticklabels=xticklabels,
    ax=axes[0],
    colors=color_list[: len(data_mcc)],
)
axes[0].tick_params(axis="x", labelrotation=0)
axes[0].set_title("Identifiability Scores")
axes[0].set_xlabel("Intervention target permutation")

axes[1] = create_violinplot(
    data_loss,
    xlabel=None,
    ylabel=r"$\Delta$ log-likelihood",
    xticklabels=xticklabels[1:],
    ax=axes[1],
    colors=color_list[1 : len(data_loss)+1],
)
axes[1].tick_params(axis="x", labelrotation=0)
axes[1].axhline(0, ls="--", color="gray")
axes[1].set_title("Model Fits")
axes[1].set_xlabel("Intervention target permutation")
plt.savefig("plots/crl_experiments_3var.pdf")
plt.show()
# -


