import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from math import sqrt
from statsmodels.stats.power import TTestIndPower

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
        Visualisering av jämförelsen.
        """
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(summary_compare))

        ax.bar(
            x,
            summary_compare["andel"],
            yerr=[summary_compare["andel"] - summary_compare["ci_low"],
                summary_compare["ci_high"] - summary_compare["andel"]],
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
        Visualisering av jämförelsen.
        """
        fig, ax = plt.subplots(figsize=(7, 4))

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
    
    def power_simulation(self, n_sims=8000, alpha=0.05):
        """
        Powersimulering med samma storlek på utval (n) och s_pooled som observerat.
        """
        detections = 0

        for _ in range(n_sims):
            sim_rökare = np.random.normal(self.mean_rökare, self.s_pooled, self.n_rökare)
            sim_ickerökare = np.random.normal(self.mean_ickerökare, self.s_pooled, self.n_ickerökare)
            _, p = stats.ttest_ind(sim_rökare, sim_ickerökare, equal_var=False)
            if p < alpha:
                detections += 1
        self.power_sim = detections / n_sims
        return self.power_sim
    
    def power_analytical(self, alpha=0.05):
        """
        Analytisk beräkning av power (ej simulering)
        """
        effect_size = abs(self.cohens_d())
        ratio = self.n_rökare / self.n_ickerökare
        self.power_ana = TTestIndPower().power(
            effect_size=effect_size,
            nobs1=self.n_ickerökare,
            alpha=alpha,
            ratio=ratio,
            alternative="two-sided",
        )
        return self.power_ana
    
    def power_n_required(self, power_req=0.80, alpha=0.05):
        """
        Beräkning av storlek på urval för att uppnå 80% power.
        """
        effect_size = abs(self.cohens_d())
        ratio = self.n_rökare / self.n_ickerökare
        n_needed = TTestIndPower().solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power_req,
            ratio=ratio,
            alternative="two-sided",
        )
        self.n_needed = n_needed
        self.ratio = ratio

        return n_needed, n_needed * ratio
    
    def plot_sample_size_power(self, n_sims=8000, alpha=0.05, ylim=(0, 1.0)):
        """
        Visualisering av power vs storlek på urval.
        """
        sample_sizes = [1000, 5000, 20_000, 50_000]
        ratio = self.n_rökare / self.n_ickerökare 
        ss_pwr = []

        for n_ickerökare in sample_sizes:
            if n_ickerökare < 2000:
                sims = n_sims
            elif n_ickerökare > 10_000:
                sims = n_sims // 2
            else:
                sims = n_sims // 4
            n_rökare = int(round(n_ickerökare*ratio))
            detections =0
            for _ in range(sims):
                sim_rökare = np.random.normal(self.mean_rökare, self.s_pooled, n_rökare)
                sim_ickerökare = np.random.normal(self.mean_ickerökare, self.s_pooled, n_ickerökare)
                _, p = stats.ttest_ind(sim_rökare, sim_ickerökare, equal_var=False)
                if p < alpha:
                    detections += 1
            ss_pwr.append(detections / sims)
    
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(sample_sizes, ss_pwr, marker="o")
        ax.axhline(0.8, color="gray", linestyle="--", label="Mål: 80% power")
        ax.set_xlabel("Stickprovsstorlek i gruppen icke-rökare")
        ax.set_ylabel("Skattad power (simulerad)")
        ax.set_title(f"Power vs storlek på urval({self.value_col}, ɑ={alpha})")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()

        return fig, ax