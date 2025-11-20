import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

def plot_bp_hist(df_clean, ax):
    """
    Histogram över systoliskt blodtryck.
    """
    ax.hist(df_clean["systolic_bp"], bins=20, edgecolor="black")
    ax.set_title("Fördelning av systoliskt blodtryck")
    ax.set_xlabel("Systoliskt blodtryck")
    ax.set_ylabel("Antal personer")
    ax.grid(True, axis="y")
    return ax

def plot_weight_by_sex_box(df_clean, ax):
    """
    Boxplot av vikt uppdelat på kön.
    """
    df_clean.boxplot(column="weight", by="sex", ax=ax)
    ax.set_title("Vikt per kön")
    ax.set_xlabel("Kön")
    ax.set_ylabel("Vikt")
    ax.grid(True, axis="y")
    return ax

def _age_bins(df_clean):
    """
    Hjälpfunktion ålderskategorier.
    """
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]
    labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    return pd.cut(df_clean["age"], bins=bins, labels=labels)

def plot_smoker_by_age(df_clean, ax):
    """
    Stapeldiagram över rökare vs icke-rökare per åldersgrupp.
    """
    age_bins = _age_bins(df_clean)
    smoker_by_age = pd.crosstab(age_bins, df_clean["smoker"])
    smoker_by_age.plot(kind="bar", ax=ax)
    ax.set_title("Rökare vs icke-rökare per åldersgrupp")
    ax.set_xlabel("Åldersgrupp")
    ax.set_ylabel("Antal personer")
    ax.grid(True, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=45)
    return ax

def plot_disease_by_age(df_clean, ax):
    """
    Stapeldiagram öve sjuka vs friska per åldersgrupp.
    """
    age_bins = _age_bins(df_clean)
    disease_by_age = pd.crosstab(age_bins, df_clean["disease"])
    disease_by_age.plot(kind="bar", ax=ax)
    ax.set_title("Sjuka vs friska per åldersgrupp")
    ax.set_xlabel("Åldersgrupp")
    ax.set_ylabel("Antal personer")
    ax.grid(True, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=45)
    return ax

def overview_grid(df_clean, figsize=(14, 9), suptitle="Översikt: fyra sätt att visa data"):
    """
    Visar alla grafer tillsammans.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=False, sharey=False)

    plot_bp_hist(df_clean, axes[0, 0])
    plot_weight_by_sex_box(df_clean, axes[0, 1])
    plot_smoker_by_age(df_clean, axes[1, 0])
    plot_disease_by_age(df_clean, axes[1, 1])

    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.3)
    return fig, axes

def plot_pca_scatter(X_pca, y):
    """
    Scatterplot för PCA
    """
    y_numeric = pd.factorize(y)[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:,1], c=y_numeric, cmap="coolwarm", edgecolor="k", s=80, alpha=0.6)
    ax.set_xlabel("Huvudkomponent 1")
    ax.set_ylabel("Huvudkomponent 2")
    ax.set_title("PCA: projektion på de två huvudkomponenterna")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Grupp (K/M)")
    plt.tight_layout()
    return fig, ax

def plot_height_by_sex_box(df_clean):
    """
    Boxplot av längd uppdelat på kön.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    df_clean.boxplot(column="height", by="sex", ax=ax)
    ax.set_title("Längd per kön")
    ax.set_xlabel("Kön")
    ax.set_ylabel("Längd")
    ax.grid(True, axis="y")
    fig.suptitle("")
    return fig, ax
