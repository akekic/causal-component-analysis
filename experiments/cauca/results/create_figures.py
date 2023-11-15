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

# # CauCA Plots

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
color_list = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]

mm = 1 / (10 * 2.54)  # millimeters in inches
SINGLE_COLUMN = 85 * mm
DOUBLE_COLUMN = 174 * mm
FONTSIZE = 9

style_modifications = {
    "font.size": FONTSIZE,
    "axes.titlesize": FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
    "font.family": "sans-serif",  # set the font globally
    "font.sans-serif": "Helvetica",  # set the font name for a font family
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "errorbar.capsize": 2.5,
}
plt.rcParams.update(style_modifications)


# -

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


def create_violinplot(
    groups, xlabel, ylabel, xticklabels, filename=None, ax=None, colors=None
):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        pass

    vp = ax.violinplot(groups, showmedians=True)
    format_violin(vp, BLUE)
    if colors:
        for pc, color in zip(vp["bodies"], colors):
            pc.set_facecolor(color)

    if xticklabels is not None:
        ax.set_xticks(np.arange(1, len(xticklabels) + 1))
        ax.set_xticklabels(xticklabels)
    else:
        ax.tick_params(bottom=False, labelbottom=False)
    ax.set_ylabel(ylabel)
    if filename is not None:
        plt.savefig(f"{filename}.svg")
    return ax


# ## Figure 4

def read_and_select_best(path):
    df = pd.read_csv(path)
    df = df.loc[df.groupby("seed")["val_loss"].idxmin()]
    return df


# +
# CauCA 2 nonlinearities, 4 variables, SNR 1
df_cc2 = read_and_select_best("data/2_nonlin_cauca.csv")

# CauCA 3 nonlinearities, 4 variables, SNR 1
df_cc3 = pd.read_csv("data/3_nonlin_cauca.csv")  # well-specified
df_cc3_dm = pd.read_csv("data/3_nonlin_cauca_misspec.csv")  # dag misspecified
# need to merge gt_val_loss from well-specified, since it's broken for dag misspecification
df_cc3_dm = df_cc3_dm[
    ["Notes", "seed", "training_seed", "val_loss", "val_mcc", "test_mcc"]
]
df_cc3_dm = df_cc3_dm.merge(
    df_cc3[["seed", "training_seed", "val_loss_gt"]], on=["seed", "training_seed"]
)

df_cc3 = df_cc3.loc[df_cc3.groupby("seed")["val_loss"].idxmin()]
df_cc3_dm = df_cc3_dm.loc[df_cc3_dm.groupby("seed")["val_loss"].idxmin()]
df_cc3_fm = read_and_select_best(
    "data/3_nonlin_cauca_func_misspec.csv"
)  # function misspecified

# CauCA 5 nonlinearities, 4 variables, SNR 1
df_cc5 = read_and_select_best("data/5_nonlin_cauca.csv")

# CauCA 3 nonlinearities, 2-5 variables, SNR 1
df_cc3_2vars = read_and_select_best("data/3_nonlin_cauca_2vars.csv")  # 2 variables
df_cc3_3vars = read_and_select_best("data/3_nonlin_cauca_3vars.csv")  # 3 variables
df_cc3_5vars = read_and_select_best("data/3_nonlin_cauca_5vars.csv")  # 5 variables

# ICA, 3 nonlinearities, 4 variables, SNR 1
df_ica3 = read_and_select_best("data/3_nonlin_ica.csv")  # well-specified
df_ica3_fm = read_and_select_best(
    "data/3_nonlin_ica_func_misspec.csv"
)  # function misspecified
df_ica3_naive = read_and_select_best(
    "data/3_nonlin_ica_naive.csv"
)  # single environment/i.i.d. misspecified

# CauCA 3 nonlinearities, 4 variables, SNR 10
df_cc3_10 = read_and_select_best("data/3_nonlin_cauca_10scm.csv")  # well-specified
df_cc3_10_dm = read_and_select_best(
    "data/3_nonlin_cauca_misspec_10scm.csv"
)  # dag misspecified

# CauCA 3 nonlinearities, 4 variables, SNR 5
df_cc3_5 = read_and_select_best("data/3_nonlin_cauca_5scm.csv")  # well-specified
df_cc3_5_dm = read_and_select_best(
    "data/3_nonlin_cauca_misspec_5scm.csv"
)  # dag misspecified

# +
fig = plt.figure(
    figsize=(DOUBLE_COLUMN, 0.4 * DOUBLE_COLUMN), dpi=700, tight_layout=False
)

gs = gridspec.GridSpec(2, 4, wspace=0.35, hspace=0.1, height_ratios=[0.4, 0.6])

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[0, 3])

ax5 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax6 = fig.add_subplot(gs[1, 1], sharey=ax5, sharex=ax2)

gs2 = gridspec.GridSpecFromSubplotSpec(1, 10, subplot_spec=gs[1, 2:], hspace=0.05)
ax7 = fig.add_subplot(gs2[1:])


axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
for i, (label, ax) in enumerate(
    zip(["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)"], axes)
):
    if i < 4:
        trans = mtransforms.ScaledTranslation(10 / 72, -40 / 72, fig.dpi_scale_trans)
    elif i == 6:
        trans = mtransforms.ScaledTranslation(10 / 72, -35 / 72, fig.dpi_scale_trans)
    else:
        trans = mtransforms.ScaledTranslation(10 / 72, -10 / 72, fig.dpi_scale_trans)
    ax.text(
        0.0,
        1.0,
        label,
        transform=ax.transAxes + trans,
        fontsize="medium",
        verticalalignment="top",
        fontfamily="serif",
        bbox=dict(facecolor="0.7", edgecolor="none", pad=3.0, alpha=0.75),
    )

# Data axes 1 and 5: basic CauCA
data_mcc = [
    df_cc3["test_mcc"],
    df_cc3_fm["test_mcc"],
    df_cc3_dm["test_mcc"],
]
data_loss = [
    (df_cc3["val_loss"] - df_cc3["val_loss_gt"]),
    (df_cc3_fm["val_loss"] - df_cc3_fm["val_loss_gt"]),
    (df_cc3_dm["val_loss"] - df_cc3_dm["val_loss_gt"]),
]

## Axis 1
colors = [color_list[0], color_list[2], color_list[1]]
ax1 = create_violinplot(
    data_mcc,
    xlabel=None,
    ylabel="MCC",
    xticklabels=["", "", "", ""],
    ax=ax1,
    colors=colors,
)
ax1.tick_params(axis="x", labelrotation=30, labelbottom=False)

## Axis 5
xticklabels = [
    "CauCA   ",
    "linear",
    r"     $E(G){=}\emptyset$",
]
ax5 = create_violinplot(
    data_loss,
    xlabel=None,
    ylabel=r"$\Delta$ log prob.",
    xticklabels=xticklabels,
    ax=ax5,
    colors=colors,
)
ax5.tick_params(axis="x", labelrotation=0)
ax5.set_yscale("log")

# Axis 4: Ablation - number of variables
data_mcc = [
    df_cc3_2vars["test_mcc"],
    df_cc3_3vars["test_mcc"],
    df_cc3["test_mcc"],
    df_cc3_5vars["test_mcc"],
]
xticklabels = [
    "2",
    "3",
    "4",
    "5",
]
colors = color_list[:1] * len(xticklabels)
ax4 = create_violinplot(
    data_mcc,
    xlabel="Latent dimension",
    ylabel="",
    xticklabels=xticklabels,
    ax=ax4,
    colors=colors,
)
ax4.xaxis.tick_top()
ax4.set_xlabel("Latent dimension")
ax4.xaxis.set_label_position("top")
ax4.tick_params(axis="x", labelbottom=False)

# Axis 3: Ablation - number of nonlinearities
data_mcc = [
    df_cc2["test_mcc"],
    df_cc3["test_mcc"],
    df_cc5["test_mcc"],
]
xticklabels = [
    "2",
    "3",
    "5",
]
colors = color_list[:1] * len(xticklabels)
ax3 = create_violinplot(
    data_mcc,
    xlabel="No. nonlinearities",
    ylabel="",
    xticklabels=xticklabels,
    ax=ax3,
    colors=colors,
)
ax3.tick_params(axis="x", labelrotation=0)
ax3.xaxis.tick_top()
ax3.xaxis.set_label_position("top")
ax3.set_xlabel("No. nonlinearities")


# Data axes 2 and 6: ICA setting
data_mcc = [
    df_ica3["test_mcc"],
    df_ica3_fm["test_mcc"],
    df_ica3_naive["test_mcc"],
]
data_loss = [
    (df_ica3["val_loss"] - df_ica3["val_loss_gt"]),
    (df_ica3_fm["val_loss"] - df_ica3_fm["val_loss_gt"]),
    (df_ica3_naive["val_loss"] - df_ica3_naive["val_loss_gt"]),
]

## Axis 2
colors = [color_list[0], color_list[2], color_list[3]]
ax2 = create_violinplot(
    data_mcc,
    xlabel=None,
    ylabel="",
    xticklabels=["", "", "", ""],
    ax=ax2,
    colors=colors,
)
ax2.set_yticks([0.2, 0.6, 1.0])

## Axis 6
xticklabels = [
    "ICA",
    "linear",
    "i.i.d",
]
ax6 = create_violinplot(
    data_loss, xlabel=None, ylabel="", xticklabels=xticklabels, ax=ax6, colors=colors
)
ax6.tick_params(axis="x", labelrotation=0)
ax6.set_yscale("log")
ax6.tick_params(axis="y", labelleft=False)


# Axis 7: signal-to-noise ratio
data_mcc = [
    df_cc3["test_mcc"],
    df_cc3_dm["test_mcc"],
    df_cc3_5["test_mcc"],
    df_cc3_5_dm["test_mcc"],
    df_cc3_10["test_mcc"],
    df_cc3_10_dm["test_mcc"],
]
data_loss = [
    (df_cc3["val_loss"] - df_cc3["val_loss_gt"]),
    (df_cc3_dm["val_loss"] - df_cc3_dm["val_loss_gt"]),
    (df_cc3_5["val_loss"] - df_cc3_5["val_loss_gt"]),
    (df_cc3_5_dm["val_loss"] - df_cc3_5_dm["val_loss_gt"]),
    (df_cc3_10["val_loss"] - df_cc3_10["val_loss_gt"]),
    (df_cc3_10_dm["val_loss"] - df_cc3_10_dm["val_loss_gt"]),
]
xticklabels = [
    "CauCA_1",
    r"$E(G){=}\emptyset$_1",
    "CauCA_5",
    r"$E(G){=}\emptyset$_5",
    "CauCA_10",
    r"$E(G){=}\emptyset$_10",
]
mcc_cauca = np.array(
    [
        df_cc3["test_mcc"].quantile(0.5),
        df_cc3_5["test_mcc"].quantile(0.5),
        df_cc3_10["test_mcc"].quantile(0.5),
    ]
)
mcc_cauca_err_up_raw = np.array(
    [
        df_cc3["test_mcc"].quantile(1),
        df_cc3_5["test_mcc"].quantile(1),
        df_cc3_10["test_mcc"].quantile(1),
    ]
)
mcc_cauca_err_up = mcc_cauca_err_up_raw - mcc_cauca
mcc_cauca_err_low_raw = np.array(
    [
        df_cc3["test_mcc"].quantile(0),
        df_cc3_5["test_mcc"].quantile(0),
        df_cc3_10["test_mcc"].quantile(0),
    ]
)
mcc_cauca_err_low = mcc_cauca - mcc_cauca_err_low_raw

mcc_dm = np.array(
    [
        df_cc3_dm["test_mcc"].quantile(0.5),
        df_cc3_5_dm["test_mcc"].quantile(0.5),
        df_cc3_10_dm["test_mcc"].quantile(0.5),
    ]
)
mcc_dm_err_up_raw = np.array(
    [
        df_cc3_dm["test_mcc"].quantile(1),
        df_cc3_5_dm["test_mcc"].quantile(1),
        df_cc3_10_dm["test_mcc"].quantile(1),
    ]
)
mcc_dm_err_up = mcc_dm_err_up_raw - mcc_dm
mcc_dm_err_low_raw = np.array(
    [
        df_cc3_dm["test_mcc"].quantile(0),
        df_cc3_5_dm["test_mcc"].quantile(0),
        df_cc3_10_dm["test_mcc"].quantile(0),
    ]
)
mcc_dm_err_low = mcc_dm - mcc_dm_err_low_raw

x = [1, 5, 10]
ax7.plot(x, mcc_cauca, "-o", color=color_list[0], label="CauCA")
ax7.fill_between(
    x,
    y1=mcc_cauca_err_low_raw,
    y2=mcc_cauca_err_up_raw,
    alpha=0.25,
    color=color_list[0],
)

ax7.plot(x, mcc_dm, "-o", color=color_list[1], label=r"$E(G){=}\emptyset$")
ax7.fill_between(
    x, y1=mcc_dm_err_low_raw, y2=mcc_dm_err_up_raw, alpha=0.25, color=color_list[1]
)
ax7.set_ylabel("MCC")
ax7.set_xlabel("Signal-to-noise ratio SCM")
ax7.set_xticks([1, 5, 10])
ax7.legend(fontsize="small")


plt.savefig("plots/experiments.pdf")

plt.show()
# -

# ## Figure 6

df = read_and_select_best("data/locscale.csv")
df_lin = read_and_select_best("data/linear_scm.csv")
df_fm = read_and_select_best("data/func_misspec.csv")
df_n = read_and_select_best("data/naive.csv")

fig, axes = plt.subplots(
    1, 1, figsize=(0.5 * DOUBLE_COLUMN, 0.4 * DOUBLE_COLUMN), dpi=500, tight_layout=True
)
# plt.suptitle("location scale scm")
data_mcc = [
    df["test_mcc"],
    df_lin["test_mcc"],
    df_fm["test_mcc"],
    df_n["test_mcc"],
]
xticklabels = [
    "CauCA\n(loc. scale)",
    "CauCA\n(lin. SCM)",
    "linear",
    "i.i.d.",
]
colors = [color_list[0], color_list[0], color_list[2], color_list[3]]
axes = create_violinplot(
    data_mcc, xlabel=None, ylabel="MCC", xticklabels=xticklabels, ax=axes, colors=colors
)
axes.tick_params(axis="x", labelrotation=0)
plt.savefig("plots/nonparam_experiments.pdf")
plt.show()


