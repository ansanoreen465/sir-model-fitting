# SIR model fitting mini-project
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def sir(y, t, beta, gamma):
    S, I, R = y
    N = S + I + R
    dS = -beta * S * I / N
    dI =  beta * S * I / N - gamma * I
    dR =  gamma * I
    return [dS, dI, dR]

def simulate(beta, gamma, days=60, N=1000, I0=10):
    t = np.arange(days)
    y0 = [N - I0, I0, 0]
    sol = odeint(sir, y0, t, args=(beta, gamma))
    return t, sol[:, 1] 

TRUE_BETA, TRUE_GAMMA = 0.4, 0.1
rng = np.random.default_rng(42)
t, I_true = simulate(TRUE_BETA, TRUE_GAMMA)
I_obs = rng.poisson(I_true)

def neg_log_likelihood(params):
    beta, gamma = params
    if beta <= 0 or gamma <= 0:
        return 1e10
    _, I_model = simulate(beta, gamma)
    return -np.sum(I_obs * np.log(I_model + 1e-9) - I_model)

result = minimize(neg_log_likelihood, x0=[0.2, 0.05], method="Nelder-Mead")
beta_hat, gamma_hat = result.x

print(f"True parameters:   beta = {TRUE_BETA}, gamma = {TRUE_GAMMA}")
print(f"Fitted parameters: beta = {beta_hat:.3f}, gamma = {gamma_hat:.3f}")

_, I_fit = simulate(beta_hat, gamma_hat)
plt.figure(figsize=(7, 4))
plt.scatter(t, I_obs, s=18, label="Synthetic observed data (Poisson noise)")
plt.plot(t, I_true, "--", label="True model")
plt.plot(t, I_fit, "-", label=f"Fitted model (beta={beta_hat:.2f}, gamma={gamma_hat:.2f})")
plt.xlabel("Day"); plt.ylabel("Infected individuals")
plt.title("SIR model fitted to synthetic outbreak data")
plt.legend(); plt.tight_layout()
plt.savefig("sir_fit.png", dpi=150)
plt.show()