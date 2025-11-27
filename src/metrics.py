import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from math import sqrt
from statsmodels.stats.power import TTestIndPower
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
    Klass som beräknar andelen personer i datasetet som har sjukdomen och jämför med en simulering.
    """
    def __init__(self, df_clean, disease_col="disease"):
        self.df_clean = df_clean
        self.disease_col = disease_col
        self.p_sim = None
        self.n_sim = None

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
        self.p_sim = float(sim.mean())
        self.n_sim = n
        return self.p_sim

    def compare(self, n_sim=1000):
        """
        Jämför observerad och simulerad andel med 95% konfidensintervall.
        """
        p_obs = self.observed()
        n_obs = len(self.df_clean)

        p_sim = self.p_sim
        n_sim = self.n_sim

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
    Klass som beräknar ett konfidensintervall för medelvärdet av systolic_bp med normalapproximation och bootstrap.
    Samma medelvärde används i båda.
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
        ax.set_title(f"Jämförelse av konfidensintervall för {self.name}")
        ax.set_ylabel(self.name)
        min_ci = min(nlo, blo)
        max_ci = max(nhi, bhi)
        margin = 0.2
        ax.set_ylim(min_ci - margin, max_ci + margin)
        ax.grid(True, axis="y")
        plt.tight_layout()
        return fig, ax
    
class SmokerAnalyzer:
    """
    Klass som testar hypotesen att rökare har ett högre värde än icke-rökare.
    """
    def __init__(self, df_clean, value_col="systolic_bp"):
        self.df_clean = df_clean
        self.value_col = value_col

        self.values_smoker = df_clean.loc[df_clean["smoker"].str.lower() == "yes", self.value_col].values
        self.values_non_smoker = df_clean.loc[df_clean["smoker"].str.lower() == "no", self.value_col].values

        self.n_smoker = len(self.values_smoker)
        self.n_non_smoker = len(self.values_non_smoker)

        self.mean_smoker = self.values_smoker.mean()
        self.mean_non_smoker = self.values_non_smoker.mean()

        self.std_smoker = self.values_smoker.std()
        self.std_non_smoker = self.values_non_smoker.std()

        self.var_smoker = self.values_smoker.var(ddof=1)
        self.var_non_smoker = self.values_non_smoker.var(ddof=1)

        self.s_pooled = np.sqrt(
            ((self.n_smoker - 1) * self.var_smoker + (self.n_non_smoker - 1) * self.var_non_smoker)
            / (self.n_smoker + self.n_non_smoker - 2)
            )

    def ttests(self):
        """
        T-test och Welch-test.
        """
        t_stat,  p_val  = stats.ttest_ind(self.values_smoker, self.values_non_smoker, equal_var=True)
        t_stat_w, p_val_w = stats.ttest_ind(self.values_smoker, self.values_non_smoker, equal_var=False)
        return t_stat, p_val, t_stat_w, p_val_w
    
    def cohens_d(self):
        """
        Cohen's d.
        """
        return (self.mean_smoker - self.mean_non_smoker) / self.s_pooled
    
    def power_simulation(self, n_sims=8000, alpha=0.05):
        """
        Powersimulering med samma storlek på urval (n) och s_pooled som observerat.
        """
        detections = 0

        for _ in range(n_sims):
            sim_smoker = np.random.normal(self.mean_smoker, self.s_pooled, self.n_smoker)
            sim_non_smoker = np.random.normal(self.mean_non_smoker, self.s_pooled, self.n_non_smoker)
            _, p = stats.ttest_ind(sim_smoker, sim_non_smoker, equal_var=False)
            if p < alpha:
                detections += 1
        self.power_sim = detections / n_sims
        return self.power_sim
    
    def power_analytical(self, alpha=0.05):
        """
        Analytisk beräkning av power (ej simulering)
        """
        effect_size = abs(self.cohens_d())
        ratio = self.n_smoker / self.n_non_smoker
        self.power_ana = TTestIndPower().power(
            effect_size=effect_size,
            nobs1=self.n_non_smoker,
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
        ratio = self.n_smoker / self.n_non_smoker
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
        ratio = self.n_smoker / self.n_non_smoker 
        ss_pwr = []

        for n_non_smoker in sample_sizes:
            if n_non_smoker < 2000:
                sims = n_sims
            elif n_non_smoker > 10_000:
                sims = n_sims // 2
            else:
                sims = n_sims // 4
            n_smoker = int(round(n_non_smoker*ratio))
            detections =0
            for _ in range(sims):
                sim_smoker = np.random.normal(self.mean_smoker, self.s_pooled, n_smoker)
                sim_non_smoker = np.random.normal(self.mean_non_smoker, self.s_pooled, n_non_smoker)
                _, p = stats.ttest_ind(sim_smoker, sim_non_smoker, equal_var=False)
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
    
class BPAnalyzer1:
    """
    Klass för enkel linjär regression för y_col från x_col.
    """
    def __init__(self, df_clean, x_col="age", y_col="systolic_bp"):
        self.df_clean = df_clean
        self.x_col = x_col
        self.y_col = y_col

        self.X = df_clean[[x_col]].values
        self.y = df_clean[y_col].values

        self.model = LinearRegression()
        self.model.fit(self.X, self.y)

        self.intercept = float(self.model.intercept_)
        self.slope = float(self.model.coef_[0])
        self.y_hat = self.model.predict(self.X)
        self.residuals = self.y - self.y_hat 
        self.r2 = float(self.model.score(self.X, self.y))

    def prediction(self, x_pre):
        """
        Gör prediktioner för ålder
        """
        x_pre = np.array(x_pre).reshape(-1, 1)
        return self.model.predict(x_pre)
    
    def plot_model(self):
        """
        Visualisering av data med regressionslinje.
        """
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(self.X.ravel(), self.y, alpha=0.6, label="Observerade värden")
        x_min, x_max = float(self.X.min()), float(self.X.max())
        x_grid = np.linspace(x_min, x_max, 200).reshape(-1, 1)
        y_grid = self.model.predict(x_grid)
        ax.plot(x_grid.ravel(), y_grid, linewidth=2, color="orange", label="Regressionslinje")

        ax.set_xlabel(self.x_col)
        ax.set_ylabel(self.y_col)
        ax.set_title(f"{self.y_col} som funktion av {self.x_col}\n"
                     f"ŷ = {self.intercept:.2f} + {self.slope:.2f}·x   (R² = {self.r2:.3f})")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        return fig, ax
    
    def plot_residuals(self):
        """
        Visualisering av residualer - slumpmoln runt noll.
        """
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(self.y_hat, self.residuals, alpha=0.6)
        ax.axhline(0, color="black", linewidth=1)

        ax.set_xlabel("Förklarat värde (ŷ)")
        ax.set_ylabel("Residual (y - ŷ)")
        ax.set_title("Residualer - slumpmoln runt 0 = bra")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        return fig, ax

class BPAnalyzer2:
    """
    Klass för multipel linjär regression.
    """
    def __init__(self, df_clean):
        self.df_clean = df_clean

        self.X = df_clean[["age", "weight"]].values
        self.y = df_clean["systolic_bp"].values

        self.model = LinearRegression()
        self.model.fit(self.X, self.y)

        self.intercept = float(self.model.intercept_)
        self.slope_age, self.slope_weight = map(float, self.model.coef_)

        self.y_hat = self.model.predict(self.X)
        self.residuals = self.y - self.y_hat 
        self.r2 = float(self.model.score(self.X, self.y))

    def prediction(self, age, weight):
        """
        Prediktioner för ålder och vikt.
        """
        age = np.array(age)
        weight = np.array(weight)
        X_pre = np.column_stack((age, weight))
        return self.model.predict(X_pre)
    
    def plot_model_age(self):
        """
        Visualisering av data med regressionslinje.
        """
        mean_weight = self.df_clean["weight"].mean()

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(self.df_clean["age"], self.y, alpha=0.6, label="Observerade värden")
        age_min, age_max = self.df_clean["age"].min(), self.df_clean["age"].max()
        age_grid = np.linspace(age_min, age_max, 200)
        X_grid = np.column_stack((age_grid, np.full_like(age_grid, mean_weight)))
        y_grid = self.model.predict(X_grid)
        ax.plot(age_grid, y_grid, linewidth=2, color="orange", label=f"Regressionslinje (vikt = {mean_weight:.1f} kg)")

        ax.set_xlabel("Ålder (år)")
        ax.set_ylabel("Systoliskt blodtryck (mmHg)")
        ax.set_title(f"""
                     Systoliskt blodtryck som funktion av ålder
                     ŷ = {self.intercept:.2f} + {self.slope_age:.2f}·age + {self.slope_weight:.2f}
                     R² = {self.r2:.3f}
                    """)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        return fig, ax
    
    def plot_model_weight(self):
        """
        Visualisering av data med regressionslinje.
        """
        mean_age = self.df_clean["age"].mean()

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(self.df_clean["weight"], self.y, alpha=0.6, label="Observerade värden")
        weight_min, weight_max = self.df_clean["weight"].min(), self.df_clean["weight"].max()
        weight_grid = np.linspace(weight_min, weight_max, 200)
        X_grid = np.column_stack((np.full_like(weight_grid, mean_age), weight_grid))
        y_grid = self.model.predict(X_grid)
        ax.plot(weight_grid, y_grid, linewidth=2, color="orange", label=f"Regressionslinje (ålder = {mean_age:.1f} år)")

        ax.set_xlabel("Vikt (kg)")
        ax.set_ylabel("Systoliskt blodtryck (mmHg)")
        ax.set_title(f"""
                     Systoliskt blodtryck som funktion av vikt
                     ŷ = {self.intercept:.2f} + {self.slope_age:.2f}·age + {self.slope_weight:.2f}
                     R² = {self.r2:.3f}
                    """)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        return fig, ax
    
    def plot_residuals(self):
        """
        Visualisering av residualer - slumpmoln runt noll.
        """
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(self.y_hat, self.residuals, alpha=0.6)
        ax.axhline(0, color="black", linewidth=1)

        ax.set_xlabel("Förklarat värde (ŷ)")
        ax.set_ylabel("Residual (y - ŷ)")
        ax.set_title("Residualer - slumpmoln runt 0 = bra")
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        return fig, ax
    
def pc_analysis(df_clean, test_size=0.3, random_state=42, n_components=2):
    """
    Principal Component Analysis för att studera mönster i datan.
    PCA reducerar datan till två huvudkomponenter innan klassificering.
    """
    features = ["age", "height", "weight", "systolic_bp", "cholesterol"]
    X = df_clean[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, pca

# Extra
def get_pca_loadings(pca_model, feature_names):
    """
    Returnerar PCA-loadings (komponenternas riktning/viktning per variabel).
    """
    loadings = pca_model.components_.T
    return pd.DataFrame(loadings, index=feature_names, columns=[f"HK{i+1}" for i in range(loadings.shape[1])])
