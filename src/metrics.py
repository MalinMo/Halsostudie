import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from math import sqrt
# from statsmodels.stats.power import TTestIndPower

# np.random.seed(42)

def summary_stats(df_clean):
    """
    Beräknar medel, median, min och max för: age, weight, height, systolic_bp, cholesterol.
    """
    return(df_clean[["age", "weight", "height", "systolic_bp", "cholesterol"]]
           .agg(["mean", "median", "min", "max"])
           .round(2)
    )

class HealthAnalyzer:
    """
    Beräkna andelen personer i datasetet som har sjukdomen och jämför med en simulering.
    """
    def __init__(self, df_clean, disease_col="disease"):
        self.df_clean = df_clean
        self.disease_col = disease_col

    def observed(self):
        """
        Observerad andel sjuka.
        """
        return float(self.df_clean[self.disease_col].mean())
    
    def simulated(self, n=1000):
        """
        Simulerat andel sjuka.
        """
        p = self.observed()
        sim = np.random.choice([0, 1], size=n, p=[1 - p, p])
        return float(sim.mean())

    def compare(self, n_sim=1000):
        """
        Jämför observerad och simulerad andel med 95% konfidensintervall.
        """
        p_obs = self.observed()
        n_obs = len(self.df_clean)

        p_sim = self.simulated(n=n_sim)

        z_crit = stats.norm.ppf(0.975)

        se_obs = np.sqrt(p_obs * (1 - p_obs) / n_obs)
        se_sim = np.sqrt(p_sim * (1 - p_sim) / n_sim)

        summary = pd.DataFrame({
            "grupp": ["Observerat", "Simulerat"],
            "n": [n_obs, n_sim],
            "andel": [p_obs, p_sim],
            "ci_low": [p_obs - z_crit * se_obs, p_sim - z_crit * se_sim],
            "ci_high": [p_obs + z_crit * se_obs, p_sim + z_crit * se_sim]
        })
        return summary.round(3)
    
    def plot_compare(self, summary_compare):
        """
        Visulaisering av jämförelsen.
        """
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(summary_compare))

        ax.bar(
            x,
            summary_compare["andel"],
            yerr=[summary_compare["andel"] - summary_compare["ci_low"],
                summary_compare["ci_high"] -summary_compare["andel"]],
                capsize=8
)

        ax.set_xticks(x)
        ax.set_xticklabels(summary_compare["grupp"])
        ax.set_ylim(0, 0.1)
        ax.set_ylabel("Andel sjuka")
        ax.set_title("Observerad och simulerad andel sjuka med 95% konfidensintervall")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        return fig, ax

class MeanCIAnalyzer:
    """
    Beräknar ett konfidensintervall för medelvärdet av systolic_bp med normalapproximation och bootstrap.
    """
    def __init__(self, x, name="värde"):
        self.x = np.asarray(x, dtype=float)
        self.name = name
    
    def ci_mean_normal(self):
        """
        95% CI med normalapproxiamtion.
        """        
        nmean = float(np.mean(self.x))
        s = float(np.std(self.x, ddof=1))
        n = len(self.x)

        z_critical = 1.96
        half_width = z_critical * s / sqrt(n)
        nlo, nhi = nmean - half_width, nmean + half_width
        
        return nlo, nhi, nmean, s, n
    
    def ci_mean_bootstrap(self, B=5000):
        """
        95% CI med Bootstrap.
        """
        n = len(self.x)
        boot_means = np.empty(B)
        for b in range(B):
            boot_sample = np.random.choice(self.x, size=n, replace=True)
            boot_means[b] = np.mean(boot_sample)

        alpha = 0.05 / 2 
        blo, bhi= np.percentile(boot_means, [100*alpha, 100*(1-alpha)])
        bmean = float(np.mean(self.x))

        return blo, bhi, bmean
    
    def plot_compare(self, nlo, nhi, nmean, blo, bhi, bmean):
        """
        Visulaisering av jämförelsen.
        """
        fig, ax = plt.subplots(figsize=(5, 3))

        methods = ["Normal", "Bootstrap"]
        means = [nmean, bmean]
        lower = [nmean - nlo, bmean - blo]
        upper = [nhi - nmean, bhi - bmean]

        ax.bar(methods, means, yerr=[lower, upper],
            capsize=8, color=["blue", "green"],alpha=0.7)
        ax.set_title("Jämförelse av konfidensintervall för {self.name}")
        ax.set_ylabel(self.name)
        ax.grid(True, axis="y")
        plt.tight_layout()
        return fig, ax
    
class SmokerAnalyzer:
    """
    Testar hypotesen att rökare har ett högre värde än icke-rökare.
    """
    def __init__(self, df_clean, value_col="systolic_bp"):
        self.df_clean = df_clean
        self.value_col = value_col

        self.values_rökare = df_clean.loc[df_clean["smoker"].str.lower() == "yes", self.value_col].values
        self.values_ickerökare = df_clean.loc[df_clean["smoker"].str.lower() == "no", self.value_col].values

        self.n_rökare = len(self.values_rökare)
        self.n_ickerökare = len(self.values_ickerökare)

        self.mean_rökare = self.values_rökare.mean()
        self.mean_ickerökare = self.values_ickerökare.mean()

        self.std_rökare = self.values_rökare.std()
        self.std_ickerökare = self.values_ickerökare.std()

        self.var_rökare = self.values_rökare.var(ddof=1)
        self.var_ickerökare = self.values_ickerökare.var(ddof=1)

        self.s_pooled = np.sqrt(
            ((self.n_rökare - 1) * self.var_rökare + (self.n_ickerökare - 1) * self.var_ickerökare)
            / (self.n_rökare + self.n_ickerökare - 2)
            )

    def ttests(self):
        """
        T-test och Welch-test.
        """
        t_stat,  p_val  = stats.ttest_ind(self.values_ickerökare, self.values_rökare, equal_var=True)
        t_stat_w, p_val_w = stats.ttest_ind(self.values_ickerökare, self.values_rökare, equal_var=False)
        return t_stat, p_val, t_stat_w, p_val_w
    
    def cohens_d(self):
        """
        Cohen's d.
        """
        return (self.mean_rökare - self.mean_ickerökare) / self.s_pooled