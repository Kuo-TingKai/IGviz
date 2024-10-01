import numpy as np
import matplotlib.pyplot as plt

def binomial_log_likelihood(n, p, k):
    return k * np.log(p) + (n - k) * np.log(1 - p)

def fisher_information(n, p):
    return n / (p * (1 - p))

# Set parameters
n = 50  # Fixed number of trials
p_range = np.linspace(0.01, 0.99, 100)  # Range of probability
k_range = np.arange(n + 1)  # Possible number of successes

# Create 2D meshgrid
p_mesh, k_mesh = np.meshgrid(p_range, k_range)

# Calculate log likelihood
log_likelihood = binomial_log_likelihood(n, p_mesh, k_mesh)

# Calculate Fisher information
fisher_info = fisher_information(n, p_range)

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Log likelihood contour
contour = ax1.contourf(p_mesh, k_mesh, log_likelihood, levels=20, cmap='viridis')
ax1.set_xlabel('Probability (p)')
ax1.set_ylabel('Number of successes (k)')
ax1.set_title('Binomial Distribution Log Likelihood')
plt.colorbar(contour, ax=ax1, label='Log Likelihood')

# Fisher information curve
ax2.plot(p_range, fisher_info)
ax2.set_xlabel('Probability (p)')
ax2.set_ylabel('Fisher Information')
ax2.set_title('Binomial Distribution Fisher Information')

# Log likelihood curves for different k values
k_values = [10, 25, 40]
for k in k_values:
    ll = binomial_log_likelihood(n, p_range, k)
    ax3.plot(p_range, ll, label=f'k={k}')
ax3.set_xlabel('Probability (p)')
ax3.set_ylabel('Log Likelihood')
ax3.set_title('Log Likelihood for Different k Values')
ax3.legend()

plt.tight_layout()
plt.show()