import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
from rich.console import Console
from rich.table import Table
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import itertools
import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from arch import arch_model
from scipy.stats import norm, chi2
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

warnings.filterwarnings("ignore")

plt.style.use('dark_background')

def retrieve_data_from_yf(ticker, start_date = "", end_date = ""):

    if start_date == "":
        data = yf.download(ticker, period = "max", end = end_date)
    else:
        data = yf.download(ticker, start = start_date, end = end_date)

    close_prices = data['Close']
    close_prices.columns = [ticker]
    return close_prices


def plot_prices(prices_df, ticker,  start_date = "", end_date = "", title = "", show = True, window_size = 21):
    fig, ax = plt.subplots(figsize=(12, 6))
    if start_date!="" and end_date != "":
        prices_df = prices_df.loc[start_date:end_date]
    elif start_date!="":
        prices_df = prices_df.loc[start_date:],
    elif end_date!="":
        prices_df = prices_df.loc[:end_date],
    else:
        pass
    
    rolling_mean = prices_df[ticker].rolling(window=window_size).mean()
    rolling_std = prices_df[ticker].rolling(window=window_size).std()

    ax.plot(prices_df.index, prices_df[ticker], color='#00ffcc', linewidth=1.5)
    ax.plot(rolling_mean.index,rolling_mean.values, color = "#ff00f2", linewidth=1, label = f"{window_size} days rolling mean")
    ax.plot(rolling_std.index,rolling_std.values, color = "#ff0000", linewidth=1, label = f"{window_size} days rolling std")
    fig.autofmt_xdate()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    if title == "":
        ax.set_title(f"Historical Price: {ticker}", fontsize=14, color='white', pad=20)
    else:
        ax.set_title(title, fontsize=14, color='white', pad=20)


    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Daily Close Price (USD)", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.show()


def get_stl_decomposition(ts_df, ticker, title, period = 252):
    test = ts_df.dropna()
    stl = seasonal_decompose(test[ticker].values, model='additive', period=period)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"STL Decomposition of {title} : {ticker} ", fontsize=14, y=0.98)
    axes[0].plot(stl.trend)
    axes[0].set_title("Trend")

    axes[1].plot(stl.seasonal)
    axes[1].set_title("Seasonal")

    axes[2].plot(stl.resid)
    axes[2].set_title("Residual")

    plt.tight_layout()
    plt.show()


def run_adf_kss_test(list_of_df, names, ticker):
    console = Console()
    table = Table(title="Augmented Dickey Fuller Results")

    table.add_column("Name", justify="right", style="cyan", no_wrap=True)
    table.add_column("ADF Stat", style="magenta")
    table.add_column("P-Value", justify="center", style="green")
    table.add_column("KSS P-Value", justify="center", style="green")
    table.add_column("Lags Used", justify="center", style="red")
    table.add_column("Sample size", justify="center", style="yellow")
    for df, name in zip(list_of_df, names):
        adf_res = adfuller(df[ticker])
        kpss_res = kpss(df[ticker])
        table.add_row(name, f"{adf_res[0]:.4f}", f"{adf_res[1]:.4f}", f"{kpss_res[1]:.4f}",str(adf_res[2]),str(adf_res[3]))
    console.print(table)

def plot_acf_pacf(df_list, names, ticker, title = ""):


    fig, axes = plt.subplots(len(df_list), 2, figsize=(16, 5 * len(df_list)))
    fig.suptitle(f"ACF / PACF graph{title}", fontsize=20, weight='bold', y=0.95)
    for i, (df, name) in enumerate(zip(df_list, names)):
        plot_acf(
            df[ticker].dropna(), 
            lags=min(30, df.shape[0]/2 - 1 ), 
            ax=axes[i,0], 
            title=f"ACF: {name}",
            color= "#ccff00", 
            vlines_kwargs={"colors": "#ccff00", "linewidth": 1.5}
        )

        plot_pacf(
            df[ticker].dropna(), 
            lags=min(30, df.shape[0]/2 - 1 ), 
            ax=axes[i,1], 
            title=f"PACF: {name}",
            color= "#00ff44", 
            vlines_kwargs={"colors": "#00ff44", "linewidth": 1.5}
        )

        
        for collection in axes[i,0].collections:
            collection.set_facecolor("red")
            collection.set_alpha(0.9)

        for collection in axes[i,1].collections:
            collection.set_facecolor("red")
            collection.set_alpha(0.9)
        axes[i,0].grid(True, linestyle='--', alpha=0.3)
        axes[i,0].spines['top'].set_visible(False)
        axes[i,0].spines['right'].set_visible(False)

        axes[i,1].grid(True, linestyle='--', alpha=0.3)
        axes[i,1].spines['top'].set_visible(False)
        axes[i,1].spines['right'].set_visible(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust layout to make room for global title
    plt.show()



def grid_search_sarima(
    y,
    p_max=1,
    d_max=1,
    q_max=1,
    seasonal=False,
    s=5,
    P_max=1,
    D=0,
    Q_max=1,
    trend="c"
):
    y = y.dropna()
    rows = []

    p_range = range(p_max + 1)
    d_range = range(d_max + 1)
    q_range = range(q_max + 1)

    if seasonal:
        P_range = range(P_max + 1)
        Q_range = range(Q_max + 1)

        for p, d, q, P, Q in itertools.product(
            p_range, d_range, q_range, P_range, Q_range
        ):
            try:
                model = SARIMAX(
                    y,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    trend=trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                res = model.fit(disp=False)
                rows.append((p, d, q, P, D, Q, s, res.aic, res.bic))
            except Exception:
                pass

        cols = ["p", "d", "q", "P", "D", "Q", "s", "AIC", "BIC"]

    else:
        for p, d, q in itertools.product(p_range, d_range, q_range):
            try:
                model = SARIMAX(
                    y,
                    order=(p, d, q),
                    trend=trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                res = model.fit(disp=False)
                rows.append((p, d, q, res.aic, res.bic))
            except Exception:
                pass

        cols = ["p", "d", "q", "AIC", "BIC"]

    out = (
        pd.DataFrame(rows, columns=cols)
        .sort_values("AIC", ascending=True)
        .reset_index(drop=True)
    )
    return out


def present_gridsearch_result(res_log_prices, res_returns, res_log_returns):
    console = Console()
    table = Table(title="[bold blue]SARIMA Model Selection Tournament[/bold blue]", show_lines=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Best (p, d, q)", style="magenta")
    table.add_column("AIC", justify="right", style="green")
    table.add_column("BIC", justify="right", style="green")

    table.add_row("Log Prices", f"({res_log_prices.iloc[0,0]:.0f},{res_log_prices.iloc[0,1]:.4f},{res_log_prices.iloc[0,2]:.4f})", f"{res_log_prices.iloc[0, 3]:.4f}",f"{res_log_prices.iloc[0, 4]:.4f}" )
    table.add_row("Returns", f"({res_returns.iloc[0,0]:.0f},{res_returns.iloc[0,1]:.4f},{res_returns.iloc[0,2]:.4f})", f"{res_returns.iloc[0, 3]:.4f}",f"{res_returns.iloc[0, 4]:.4f}" )
    table.add_row("Log Returns", f"({res_log_returns.iloc[0,0]:.0f},{res_log_returns.iloc[0,1]:.4f},{res_log_returns.iloc[0,2]:.4f})", f"{res_log_returns.iloc[0, 3]:.4f}",f"{res_log_returns.iloc[0, 4]:.4f}" )
    console.print(table)

def fit_arima_model(data, p,d,q):
    res = SARIMAX(data.dropna(),
                order=(p,d,q),    
                trend="c",
                method="lbfgs",
                cov_type = "opg",
                enforce_stationarity=False,
                enforce_invertibility=False).fit(disp=False)

    return res 



def interpret_sarimax_results(res, alpha=0.05, lb_lags=10):
    print("\n=== MODEL DIAGNOSTIC REPORT ===\n")
    print("1. Coefficient Significance")
    for name, pval in res.pvalues.items():
        if name == "sigma2":
            print(f" - {name}: variance parameter (p={pval:.3g})")
            continue

        if pval < alpha:
            print(f" - {name}: significant (p={pval:.3g}) → contributes to mean dynamics")
        else:
            print(f" - {name}: NOT significant (p={pval:.3g}) → may be unnecessary")

    resid = res.resid
    resid = resid[~np.isnan(resid)]

    print("\n2. Residual Autocorrelation (Ljung–Box)")

    lb_test = acorr_ljungbox(resid, lags=[lb_lags], return_df=True)
    lb_pvalue = lb_test["lb_pvalue"].iloc[0]

    if lb_pvalue > alpha:
        print(f" - No residual autocorrelation (p={lb_pvalue:.3g}) ")
    else:
        print(f" - Residual autocorrelation detected (p={lb_pvalue:.3g}) ")

    print("\n3. Normality of Residuals (Jarque–Bera)")

    jb_stat, jb_pvalue, skew, kurt = jarque_bera(resid)

    if jb_pvalue > alpha:
        print(f" - Residuals approximately normal (p={jb_pvalue:.3g})")
    else:
        print(
            f" - Residuals non-normal (p={jb_pvalue:.3g}) "
            f"→ skew={skew:.2f}, kurtosis={kurt:.2f}"
        )

    print("\n4. Heteroskedasticity (ARCH)")

    arch_stat, arch_pvalue, _, _ = het_arch(resid)

    if arch_pvalue > alpha:
        print(f" - Homoskedastic residuals (p={arch_pvalue:.3g})")
    else:
        print(f" - Heteroskedasticity detected (p={arch_pvalue:.3g}) → volatility clustering")


    if "ar.L1" in res.params and abs(res.params["ar.L1"] - 1) < 0.02:
        print("\n AR coefficient close to 1 → unit root  random walk likely")

    print("\n=== END OF REPORT ===\n")


def interpret_ljungbox(residuals, lags=np.linspace(0,20, 5), alpha=0.05, title = ""):

    lb_df = acorr_ljungbox(residuals, lags=list(lags), return_df=True)

    print(f"\n=== Ljung–Box Residual Autocorrelation Test {title}===\n")

    autocorr_detected = False

    for lag, row in lb_df.iterrows():
        pval = row["lb_pvalue"]

        if pval < alpha:
            autocorr_detected = True
            print(
                f"Autocorrelation detected up to lag {lag} "
                f"(p = {pval:.3g})"
            )
        else:
            print(
                f"No autocorrelation up to lag {lag} "
                f"(p = {pval:.3g})"
            )
    print("\nConclusion:")
    if autocorr_detected:
        print(
            "→ Residuals are NOT white noise: "
            "mean model may be misspecified."
        )
    else:
        print(
            "→ No evidence of residual autocorrelation: "
            "mean model is adequately specified."
        )

    print("\n=== END OF TEST ===\n")

    return lb_df


def interpret_arch_test(residuals, lags=np.linspace(5, 20, 4, dtype=int), alpha=0.05, title=""):

    print(f"\n=== Engle's ARCH Test {title} ===\n")

    arch_detected = False

    for lag in lags:
        stat, pvalue, _, _ = het_arch(residuals, nlags=int(lag))

        if pvalue < alpha:
            arch_detected = True
            print(
                f"ARCH effects detected up to lag {lag} "
                f"(p = {pvalue:.3g})"
            )
        else:
            print(
                f"No ARCH effects detected up to lag {lag} "
                f"(p = {pvalue:.3g})"
            )

    print("\nConclusion:")
    if arch_detected:
        print(
            "→ Residual variance is NOT constant: "
            "volatility clustering is present."
        )
    else:
        print(
            "→ No evidence of ARCH effects: "
            "residual variance appears constant."
        )

    print("\n=== END OF TEST ===\n")

def grid_search_garch(
    y,
    p_max=2,
    q_max=2,
    mean="Zero",
    dist="normal"
):
    y = y.dropna()
    rows = []

    p_range = range(1, p_max + 1)
    q_range = range(1, q_max + 1)

    for p, q in itertools.product(p_range, q_range):
        try:
            model = arch_model(
                y,
                mean=mean,
                vol="GARCH",
                p=p,
                q=q,
                dist=dist
            )
            res = model.fit(disp="off")
            rows.append((p, q, res.aic, res.bic))
        except Exception:
            pass

    out = (
        pd.DataFrame(rows, columns=["p", "q", "AIC", "BIC"])
        .sort_values("AIC")
        .reset_index(drop=True)
    )

    return out

def present_garch_gridsearch_result(res_log_prices, res_returns, res_log_returns):
    console = Console()
    table = Table(
        title="[bold blue]GARCH Model Selection Tournament[/bold blue]",
        show_lines=True
    )

    table.add_column("Series", style="cyan", no_wrap=True)
    table.add_column("Best (p, q)", style="magenta")
    table.add_column("AIC", justify="right", style="green")
    table.add_column("BIC", justify="right", style="green")
    table.add_row(
        "Log Prices",
        f"({res_log_prices.iloc[0,0]:.0f},{res_log_prices.iloc[0,1]:.0f})",
        f"{res_log_prices.iloc[0,2]:.4f}",
        f"{res_log_prices.iloc[0,3]:.4f}"
    )
    table.add_row(
        "Returns",
        f"({res_returns.iloc[0,0]:.0f},{res_returns.iloc[0,1]:.0f})",
        f"{res_returns.iloc[0,2]:.4f}",
        f"{res_returns.iloc[0,3]:.4f}"
    )

    table.add_row(
        "Log Returns",
        f"({res_log_returns.iloc[0,0]:.0f},{res_log_returns.iloc[0,1]:.0f})",
        f"{res_log_returns.iloc[0,2]:.4f}",
        f"{res_log_returns.iloc[0,3]:.4f}"
    )

    console.print(table)

def fit_garch_model(data, p=1, q=1, mean="Zero", dist="normal"):
    model = arch_model(
        data.dropna(),
        mean=mean,
        vol="GARCH",
        p=p,
        q=q,
        dist=dist
    )

    res = model.fit(disp="off")
    return res


def interpret_garch_results(res, alpha=0.05, lb_lags=10):
    print("\n=== GARCH MODEL DIAGNOSTIC REPORT ===\n")

    print("1. Parameter Significance")
    for name, pval in res.pvalues.items():
        if pval < alpha:
            print(f" - {name}: significant (p={pval:.3g})")
        else:
            print(f" - {name}: NOT significant (p={pval:.3g})")

    std_resid = res.std_resid
    std_resid = std_resid[~np.isnan(std_resid)]

    print("\n2. Standardized Residual Autocorrelation (Ljung–Box)")

    lb_test = acorr_ljungbox(std_resid, lags=[lb_lags], return_df=True)
    lb_pvalue = lb_test["lb_pvalue"].iloc[0]

    if lb_pvalue > alpha:
        print(f" - No autocorrelation in standardized residuals (p={lb_pvalue:.3g})")
    else:
        print(f" - Autocorrelation detected in standardized residuals (p={lb_pvalue:.3g})")

    print("\n3. Normality of Standardized Residuals (Jarque–Bera)")

    jb_stat, jb_pvalue, skew, kurt = jarque_bera(std_resid)

    if jb_pvalue > alpha:
        print(f" - Standardized residuals approximately normal (p={jb_pvalue:.3g})")
    else:
        print(
            f" - Standardized residuals non-normal (p={jb_pvalue:.3g}) "
            f"→ skew={skew:.2f}, kurtosis={kurt:.2f}"
        )

    print("\n4. Remaining ARCH Effects")

    arch_stat, arch_pvalue, _, _ = het_arch(std_resid)

    if arch_pvalue > alpha:
        print(f" - No remaining ARCH effects (p={arch_pvalue:.3g})")
    else:
        print(f" - Remaining ARCH effects detected (p={arch_pvalue:.3g})")

    print("\n=== END OF REPORT ===\n")

def rolling_sarimax_garch(series, sarimax_order,garch_order, window_size, start_oos, end_oos):
    forecasts_mean = []
    forecasts_vol = []
    realized = []
    dates = []

    for t in pd.date_range(start_oos, end_oos, freq="B"):
        t = t.strftime("%Y-%m-%d")
        if t not in series.index:
            continue
        train = series.loc[:t].iloc[-window_size-1:-1]

        if len(train) < window_size:
            continue
        sarimax_res = fit_arima_model(train, sarimax_order[0], sarimax_order[1], sarimax_order[2])
    

        resid = sarimax_res.resid.dropna()

        garch_res = fit_garch_model(resid,garch_order[0], garch_order[1])

        mean_fc = sarimax_res.get_forecast(steps=1).predicted_mean.iloc[0]

        var_fc = garch_res.forecast(horizon=1).variance.iloc[-1, 0]
        vol_fc = np.sqrt(var_fc)

        forecasts_mean.append(mean_fc)
        forecasts_vol.append(vol_fc)
        realized.append(series.loc[t])
        dates.append(t)

    return pd.DataFrame(
        {
            "realized": realized,
            "mean_forecast": forecasts_mean,
            "vol_forecast": forecasts_vol
        },
        index=dates
    )

def plot_sarimax_garch_forecast(df, title=""):
    df = df.dropna()

    mu = df["mean_forecast"]
    y = df["realized"]
    sigma = df["vol_forecast"]

    upper = mu + 1.96 * sigma
    lower = mu - 1.96 * sigma

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"SARIMAX–GARCH Rolling Forecast : {title}",
        fontsize=16,
        y=0.95
    )

    axes[0].plot(y.index, y.values, color="#00ffcc", linewidth=1.5, label="Realized")
    axes[0].plot(mu.index, mu.values, color="#ff00f2", linewidth=1.2, label="Mean forecast")
    axes[0].fill_between(
        df.index,
        lower,
        upper,
        color="#ff00f2",
        alpha=0.25,
        label="95% confidence band"
    )
    axes[0].set_title("Conditional Mean Forecast")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].plot(
        df.index,
        y - mu,
        color="#ff4444",
        linewidth=1.2
    )
    axes[1].axhline(0, linestyle="--", color="yellow", linewidth=1)
    axes[1].set_title("Forecast Error")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    axes[2].plot(
        df.index,
        sigma,
        color="#cc66ff",
        linewidth=1.5
    )
    axes[2].set_title("Conditional Volatility Forecast")
    axes[2].grid(True, linestyle="--", alpha=0.3)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    axes[2].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()



def evaluate_sarimax_garch(df):
    df = df.dropna()

    y = df["realized"].values
    mu = df["mean_forecast"].values
    sigma = df["vol_forecast"].values

    errors = y - mu
    std_errors = errors / sigma

    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    bias = np.mean(errors)

    mse_std = np.mean(std_errors**2)

    return {
        "RMSE (mean)": rmse,
        "MAE (mean)": mae,
        "Bias (mean)": bias,
        "Mean(std. residual^2)": mse_std
    }


def plot_sarimax_garch_forecast(df, title=""):
    df = df.dropna().copy()

    df.index = pd.to_datetime(df.index)

    mu = df["mean_forecast"]
    y = df["realized"]
    sigma = df["vol_forecast"]

    upper = mu + 1.96 * sigma
    lower = mu - 1.96 * sigma

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"SARIMAX-GARCH Rolling Forecast: {title}",
        fontsize=16,
        y=0.95
    )

    axes[0].plot(y.index, y.values, color="#00ffcc", lw=1.5, label="Realized")
    axes[0].plot(mu.index, mu.values, color="#ff00f2", lw=1.5, label="Mean forecast")
    axes[0].fill_between(
        df.index,
        lower,
        upper,
        color="#ff00f2",
        alpha=0.25,
        label="95% confidence band"
    )
    axes[0].set_title("Conditional Mean Forecast")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].plot(
        df.index,
        y - mu,
        color="#ff4444",
        lw=1.2
    )
    axes[1].axhline(0, linestyle="--", color="yellow", lw=1)
    axes[1].set_title("Forecast Error")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    axes[2].plot(
        df.index,
        sigma,
        color="#aa66ff",
        lw=1.5
    )
    axes[2].set_title("Conditional Volatility Forecast")
    axes[2].grid(True, linestyle="--", alpha=0.3)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()


def compute_var_es(df, alpha=0.05):
    df = df.dropna().copy()

    z_alpha = norm.ppf(alpha)
    pdf_alpha = norm.pdf(z_alpha)

    df["VaR"] = df["mean_forecast"] + df["vol_forecast"] * z_alpha
    df["ES"] = (
        df["mean_forecast"]
        - df["vol_forecast"] * pdf_alpha / alpha
    )

    df["violation"] = (df["realized"] < df["VaR"]).astype(int)

    return df

def kupiec_test(violations, alpha):
    T = len(violations)
    x = violations.sum()

    pi_hat = x / T

    if x == 0:
        return np.nan, 1.0

    LR_uc = -2 * (
        (T - x) * np.log((1 - alpha) / (1 - pi_hat)) +
        x * np.log(alpha / pi_hat)
    )

    p_value = 1 - chi2.cdf(LR_uc, df=1)

    return LR_uc, p_value

def christoffersen_test(violations):
    v = violations.values

    n00 = np.sum((v[:-1] == 0) & (v[1:] == 0))
    n01 = np.sum((v[:-1] == 0) & (v[1:] == 1))
    n10 = np.sum((v[:-1] == 1) & (v[1:] == 0))
    n11 = np.sum((v[:-1] == 1) & (v[1:] == 1))

    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    if pi0 in [0, 1] or pi1 in [0, 1]:
        return np.nan, 1.0

    LR_cc = -2 * (
        (n00 + n10) * np.log(1 - pi) +
        (n01 + n11) * np.log(pi)
        -
        (n00 * np.log(1 - pi0) +
         n01 * np.log(pi0) +
         n10 * np.log(1 - pi1) +
         n11 * np.log(pi1))
    )

    p_value = 1 - chi2.cdf(LR_cc, df=2)

    return LR_cc, p_value


def backtest_var(df, alpha=0.05):
    df = df.dropna()

    violations = df["violation"]

    kupiec_stat, kupiec_p = kupiec_test(violations, alpha)
    christ_stat, christ_p = christoffersen_test(violations)

    return {
        "alpha": alpha,
        "observations": len(df),
        "violations": violations.sum(),
        "expected_violations": alpha * len(df),
        "kupiec_pvalue": kupiec_p,
        "christoffersen_pvalue": christ_p
    }



def grid_search_garch_ex5(
    y,
    ar_lags=(0, 1),
    p_max=2,
    q_max=2,
    dist="normal"
):
    """
    Grid search for the SARIMA–GARCH FILTER (Exercise 5).
    Mean: Constant or AR(1)
    Variance: GARCH(p,q)
    """

    y = y.dropna()
    results = []

    for lag in ar_lags:
        mean_type = "Constant" if lag == 0 else "AR"

        for p, q in itertools.product(range(1, p_max + 1),
                                      range(1, q_max + 1)):
            try:
                model = arch_model(
                    y * 100,
                    mean=mean_type,
                    lags=None if lag == 0 else lag,
                    vol="GARCH",
                    p=p,
                    q=q,
                    dist=dist
                )
                res = model.fit(disp="off")

                results.append({
                    "Mean": "Constant" if lag == 0 else f"AR({lag})",
                    "ARCH_p": p,
                    "GARCH_q": q,
                    "AIC": res.aic,
                    "BIC": res.bic
                })

            except Exception:
                continue

    return (
        pd.DataFrame(results)
        .sort_values("AIC")
        .reset_index(drop=True)
    )

def present_garch_gridsearch_ex5(
    res_daily,
    res_monthly,
    res_quarterly
):
    console = Console()

    table = Table(
        title="[bold blue]SARIMA–GARCH Filtering Model Selection (Exercise 5)[/bold blue]",
        show_lines=True
    )

    table.add_column("Series", style="cyan", no_wrap=True)
    table.add_column("Mean Model", style="magenta")
    table.add_column("GARCH (p, q)", style="yellow")
    table.add_column("AIC", justify="right", style="green")
    table.add_column("BIC", justify="right", style="green")

    # Daily
    table.add_row(
        "Daily Returns",
        res_daily.loc[0, "Mean"],
        f"({int(res_daily.loc[0, 'ARCH_p'])}, {int(res_daily.loc[0, 'GARCH_q'])})",
        f"{res_daily.loc[0, 'AIC']:.4f}",
        f"{res_daily.loc[0, 'BIC']:.4f}"
    )

    # Monthly
    table.add_row(
        "Monthly Returns",
        res_monthly.loc[0, "Mean"],
        f"({int(res_monthly.loc[0, 'ARCH_p'])}, {int(res_monthly.loc[0, 'GARCH_q'])})",
        f"{res_monthly.loc[0, 'AIC']:.4f}",
        f"{res_monthly.loc[0, 'BIC']:.4f}"
    )

    # Quarterly
    table.add_row(
        "Quarterly Returns",
        res_quarterly.loc[0, "Mean"],
        f"({int(res_quarterly.loc[0, 'ARCH_p'])}, {int(res_quarterly.loc[0, 'GARCH_q'])})",
        f"{res_quarterly.loc[0, 'AIC']:.4f}",
        f"{res_quarterly.loc[0, 'BIC']:.4f}"
    )

    console.print(table)


def fit_garch_ex5(
    y,
    mean_model="Constant",
    ar_lag=0,
    p=1,
    q=1,
    dist="normal"
):
    """
    Fit the selected SARIMA–GARCH filtering model for Exercise 5.
    This model is used ONLY as a filtering step before Markov switching.
    """

    y = y.dropna()

    if mean_model == "Constant":
        mean = "Constant"
        lags = None
    elif mean_model.startswith("AR"):
        mean = "AR"
        lags = ar_lag
    else:
        raise ValueError("mean_model must be 'Constant' or 'AR(k)'")

    model = arch_model(
        y * 100,
        mean=mean,
        lags=lags,
        vol="GARCH",
        p=p,
        q=q,
        dist=dist
    )

    res = model.fit(disp="off")
    return res


def interpret_garch_ex5_results(res, alpha=0.05, lb_lags=10):
    """
    Diagnostic interpretation of the SARIMA–GARCH FILTER (Exercise 5).
    Purpose: verify adequacy BEFORE Markov switching.
    """

    print("\n=== SARIMA–GARCH FILTER DIAGNOSTIC (Exercise 5) ===\n")

    print("1. Parameter Significance")
    for name, pval in res.pvalues.items():
        if pval < alpha:
            print(f" - {name}: significant (p={pval:.3g})")
        else:
            print(f" - {name}: NOT significant (p={pval:.3g})")

    std_resid = res.std_resid
    std_resid = std_resid[~np.isnan(std_resid)]

    print("\n2. Residual Autocorrelation (Ljung–Box)")

    lb = acorr_ljungbox(std_resid, lags=[lb_lags], return_df=True)
    lb_pvalue = lb["lb_pvalue"].iloc[0]

    if lb_pvalue > alpha:
        print(f" - No autocorrelation (p={lb_pvalue:.3g})")
    else:
        print(f" - Autocorrelation detected (p={lb_pvalue:.3g}) → filter may be insufficient")

    print("\n3. Remaining ARCH Effects")

    arch_stat, arch_pvalue, _, _ = het_arch(std_resid)

    if arch_pvalue > alpha:
        print(f" - No remaining ARCH effects (p={arch_pvalue:.3g})")
    else:
        print(f" - Remaining ARCH effects detected (p={arch_pvalue:.3g})")

    print("\n4. Volatility Persistence")

    alpha_terms = [v for k, v in res.params.items() if "alpha" in k]
    beta_terms = [v for k, v in res.params.items() if "beta" in k]

    persistence = sum(alpha_terms) + sum(beta_terms)

    print(f" - α + β = {persistence:.3f}")

    if persistence < 1:
        print("   → Stationary volatility")
    else:
        print("   → Highly persistent volatility (near-integrated)")

    print("\nConclusion:")
    if lb_pvalue > alpha and arch_pvalue > alpha:
        print("→ SARIMA–GARCH filter is adequate for Markov switching.")
    else:
        print("→ Filtering may be insufficient; reconsider specification.")

    print("\n=== END OF REPORT ===\n")

           



def fit_markov_switching_model(
    resid,
    k_regimes=2,
    order=1,              
    trend="c",
    switching_variance=True

):
    ms_mod = MarkovRegression(
            resid,
            k_regimes=k_regimes,
            trend="c",
            switching_variance=True
        )
    ms_res = ms_mod.fit(disp=False)
    return ms_res
    

def extract_markov_regression_params(ms_res):

    params = ms_res.params
    k = ms_res.k_regimes

    if any(f"const[{i}]" in params.index for i in range(k)):
        c = np.array([params[f"const[{i}]"] for i in range(k)])
    elif "const" in params.index:
        c = np.repeat(params["const"], k)
    else:
        c = np.zeros(k)

    if any(f"sigma2[{i}]" in params.index for i in range(k)):
        sigma2 = np.array([params[f"sigma2[{i}]"] for i in range(k)])
    elif "sigma2" in params.index:
        sigma2 = np.repeat(params["sigma2"], k)
    else:
        sigma2 = None

    P = ms_res.regime_transition
    p_last = ms_res.filtered_marginal_probabilities.iloc[-1].values

    return c, sigma2, P[:,:,-1], p_last


def extract_z_from_arch(res):
    eps = res.resid / 100.0
    sigma = res.conditional_volatility / 100.0
    z = eps / sigma
    return z.dropna()

def forecast_arch_mean_sigma(res_garch, h):
    f = res_garch.forecast(horizon=h)

    mu_f = f.mean.iloc[-1] / 100.0
    sigma_f = np.sqrt(f.variance.iloc[-1]) / 100.0

    return mu_f, sigma_f


def forecast_markov_z(ms_res, h):
    c, sigma2, P, p = extract_markov_regression_params(ms_res)
    zf = []
    for _ in range(h):
        p = p @ P
        z_next = np.sum(p * c)
        zf.append(z_next)
    return np.array(zf), sigma2, p

def mixture_variance(p, c, sigma2):
    Ez = np.sum(p * c)
    Ez2 = np.sum(p * (sigma2 + c**2))
    return Ez2 - Ez**2

def forecast_markov_z(ms_res, h):
    c, sigma2, P, p0 = extract_markov_regression_params(ms_res)

    p_paths = []
    zf = []

    p = p0.copy()
    for _ in range(h):
        p = p @ P
        p_paths.append(p.copy())
        zf.append(np.sum(p * c))

    return np.array(zf), sigma2, p0, np.array(p_paths)

def forecasts_garch_markov(mu_f, sigma_f, z_vals):
    z_f = pd.Series(z_vals, index=mu_f.index)
    r_f = mu_f + sigma_f * z_f

    return pd.DataFrame({
        "mu_hat": mu_f,
        "sigma_hat": sigma_f,
        "z_hat": z_f,
        "r_hat": r_f
    })

def rolling_markov_switching_forecast_ex5(
    series,
    start_oos,
    end_oos,
    h=20,
    k_regimes=2,
    alpha=0.05, p_arc = 2, q_arch=2
):

    forecasts = []
    lowers = []
    uppers = []
    realized = []
    regimes = []
    dates = []

    z_alpha = norm.ppf(1 - alpha / 2)

    idx = series.index
    start_pos = idx.get_loc(start_oos)
    end_pos = idx.get_loc(end_oos)

    t_pos = start_pos

    while t_pos < end_pos:

        train = series.iloc[:t_pos] * 100  # scale for arch


        am = arch_model(
            train,
            mean="AR", lags=1,
            vol="GARCH", p=p_arc, q=q_arch,
            dist="t"
        )
        res_garch = am.fit(disp="off")


        z_hat = extract_z_from_arch(res_garch)


        ms_mod = MarkovRegression(
            z_hat,
            k_regimes=k_regimes,
            trend="c",
            switching_variance=True
        )
        ms_res = ms_mod.fit(disp=False)


        z_vals, sigma2_regs, p0, p_paths = forecast_markov_z(ms_res, h)



        if sigma2_regs is not None:
            c, _, _, _ = extract_markov_regression_params(ms_res)
            var_z = np.array([mixture_variance(p_paths[j], c, sigma2_regs) for j in range(h)])
        else:
            var_z = np.ones(h)


        mu_f, sigma_f = forecast_arch_mean_sigma(res_garch, h)

        for j in range(h):
            pos = t_pos + j
            if pos >= len(series):
                break

            date = idx[pos]

            y_hat = mu_f.iloc[j] + sigma_f.iloc[j] * z_vals[j]
            var_hat = (sigma_f.iloc[j] ** 2) * var_z[j]

            forecasts.append(y_hat)
            lowers.append(y_hat - z_alpha * np.sqrt(var_hat))
            uppers.append(y_hat + z_alpha * np.sqrt(var_hat))
            realized.append(series.iloc[pos])

            regimes.append(np.argmax(p_paths[j]))
            dates.append(date)

        t_pos += h

    return pd.DataFrame(
        {
            "realized": realized,
            "forecast": forecasts,
            "lower": lowers,
            "upper": uppers,
            "regime": regimes,
        },
        index=pd.to_datetime(dates),
    )


def plot_markov_switching_forecast(df, title=""):
    df = df.dropna().copy()
    df.index = pd.to_datetime(df.index)

    y = df["realized"]
    mu = df["forecast"]
    lower = df["lower"]
    upper = df["upper"]
    reg = df["regime"]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        y.index, y.values,
        color="#00ffcc",
        lw=1.2,
        label="Realized returns"
    )

    ax.plot(
        mu.index, mu.values,
        color="#ff00f2",
        lw=1.4,
        label="MS forecast"
    )

    ax.fill_between(
        mu.index,
        lower,
        upper,
        color="#ff00f2",
        alpha=0.25,
        label="95% confidence band"
    )

    regimes = reg.values
    unique_regs = np.unique(regimes)

    colors = ["#2222ff", "#ff2222", "#22ff22", "#ffaa00"]

    for r in unique_regs:
        mask = regimes == r
        ax.fill_between(
            df.index,
            ax.get_ylim()[0],
            ax.get_ylim()[1],
            where=mask,
            color=colors[int(r) % len(colors)],
            alpha=0.5,
            transform=ax.get_xaxis_transform(),
            label=f"Regime {int(r)}"
        )

    ax.set_title(
        f"Markov Switching Rolling Forecast {title}",
        fontsize=14,
        pad=15
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Returns")

    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()
