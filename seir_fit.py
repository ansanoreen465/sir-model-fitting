import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# SEIR model: adds an 'Exposed' compartment
def seir(y, t, beta, sigma, gamma):
    S, E, I, R = y
    N = S + E + I + R
    dS = -beta * S * I / N
    dE =  beta * S * I / N - sigma * E
    dI =  sigma * E - gamma * I
    dR =  gamma * I
    return [dS, dE, dI, dR]

def simulate_seir(params, days=80, N=1000, I0=10):
    beta, sigma, gamma = params
    t = np.arange(days)
    y0 = [N - I0, 0, I0, 0]
    sol = odeint(seir, y0, t, args=(beta, sigma, gamma))
    return t, sol[:, 2] # Return infected only

# Fake "true" data using SEIR
TRUE_PARAMS = [0.6, 0.2, 0.1] # beta, sigma, gamma
t, I_true = simulate_seir(TRUE_PARAMS)
rng = np.random.default_rng(42)
I_obs = rng.poisson(I_true)

def loss(params):
    if any(p <= 0 for p in params): return 1e10
    _, I_model = simulate_seir(params)
    return -np.sum(I_obs * np.log(I_model + 1e-9) - I_model)

res = minimize(loss, x0=[0.4, 0.1, 0.05], method="Nelder-Mead")
beta_hat, sigma_hat, gamma_hat = res.x

print(f"True:  beta={TRUE_PARAMS[0]}, sigma={TRUE_PARAMS[1]}, gamma={TRUE_PARAMS[2]}")
print(f"Fitted: beta={beta_hat:.2f}, sigma={sigma_hat:.2f}, gamma={gamma_hat:.2f}")

_, I_fit = simulate_seir(res.x)
plt.scatter(t, I_obs, s=15, label="Data")
plt.plot(t, I_fit, color='red', label="SEIR Fit")
plt.legend(); plt.title("SEIR Model Fit")
plt.show()
