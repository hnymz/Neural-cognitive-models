import numpy as np
import pandas as pd
import pyjags
import scipy.io as sio
import matplotlib.pyplot as plt
import pyhddmjagsutils as phju
from scipy.stats import gaussian_kde, norm

#%%
n_parts = 2
N_obs = 0
N200 = []
y = []
participant = []

# Change the directory to the Linux form
working_dir = '/mnt/d/research/internship/data_analysis'
for i in range(1, n_parts+1):
    modelling_data = pd.read_csv(f'{working_dir}/modelling_data_N200_{i}.csv')
    n = (modelling_data['N200_yes_or_no'] == 1).sum()
    N_obs += n
    filtered_data = modelling_data[modelling_data['N200_yes_or_no'] == 1]
    N200.extend((filtered_data['latency'] * 0.001).tolist())
    y.extend((filtered_data['accu_rt'] * 0.001).tolist())
    participant.extend([i] * n)

#%%
# Input for mixture modeling
Ones = np.ones(N_obs)
Constant = 20

#%%
# JAGS code
to_jags = '''
model {
    
    ##########
    # Priors
    ##########
    delta ~ dnorm(2, pow(4, -2))
    alpha ~ dnorm(1, pow(2, -2)) T(0, 4)
    
    ndt_int_sd ~ dgamma(.2, 1)
    ndt_gamma_sd ~ dgamma(1, 1)
    
    ndt_int_mean ~ dnorm(.2, pow(.2, -2))
    ndt_gamma_mean ~ dnorm(1, pow(3, -2))
 
    # Probability of a lapse trial
    prob_lapse_sd ~ dgamma(.3, 1)
    prob_lapse_mean ~ dnorm(.3, pow(.15, -2))
    
    # Priors for each participant
    for (p in 1:n_parts) {
        prob_lapse[p] ~ dnorm(prob_lapse_mean, pow(prob_lapse_sd, -2))T(0, 1)
        prob_DDM[p] <- 1 - prob_lapse[p]

        ndt_int[p] ~ dnorm(ndt_int_mean, pow(ndt_int_sd, -2))T(0,.7)
        ndt_gamma[p] ~ dnorm(ndt_gamma_mean, pow(ndt_gamma_sd, -2))
    }

    ##########
    # Wiener likelihood and uniform mixture using Ones trick
    for (i in 1:N_obs) {

        # Log density for DDM process of rightward/leftward RT
        ld_comp[i, 1] <- dlogwiener(y[i], alpha, ndt_int[participant[i]] + ndt_gamma[participant[i]] * N200[i], .5, delta)

        # Log density for lapse trials (negative max RT to positive max RT)
        ld_comp[i, 2] <- logdensity.unif(y[i], -1.8, 1.8)
        
        # Probability of mind wandering trials (lapse trials)
        DDM_or_Lapse[i] ~ dcat( c(prob_DDM[participant[i]], prob_lapse[participant[i]]) )        
        
        # Select one of these two densities (Mixture of nonlapse and lapse trials)
        # Restrict selected_density to be in the range (0.01, 0.99)
        selected_density[i] <- max(min(exp(ld_comp[i, DDM_or_Lapse[i]] - Constant), 0.99), 0.01)

        # Generate a likelihood for the MCMC sampler using a trick to maximize density value
        Ones[i] ~ dbern(selected_density[i])
        
    }
}
'''

# pyjags code
# Make sure $LD_LIBRARY_PATH sees /usr/local/lib
# Make sure that the correct JAGS/modules-4/ folder contains wiener.so and wiener.la
pyjags.modules.load_module('wiener')
pyjags.modules.load_module('dic')
pyjags.modules.list_modules()

n_chains = 6
burnin = 4000
n_samps = 10000

model_file = 'hier_N200_DDM.jags'
f = open(model_file, 'w')
f.write(to_jags)
f.close()

# Track these variables
trackvars = ['delta', 'alpha', 'ndt_int', 'ndt_gamma',
             'prob_lapse', 'DDM_or_Lapse']

# Setting initial values for parameters
initials = []
for c in range(0, n_chains):
    chaininit = {
        'ndt_int': np.random.uniform(.1, .5, size=n_parts),
        'ndt_gamma': np.random.uniform(.1, .5, size=n_parts),
        'ndt_int_sd': np.random.uniform(.01, .2),
        'ndt_gamma_sd': np.random.uniform(.01, .2),
        'ndt_int_mean': np.random.uniform(.1, .5),
        'ndt_gamma_mean': np.random.uniform(.1, .5),
        'delta': np.random.uniform(-3., 3.),
        'alpha': np.random.uniform(.5, 1.),
        'prob_lapse_sd': np.random.uniform(.01, .5),
        'prob_lapse_mean': np.random.uniform(.01, .1),
        'prob_lapse': np.random.uniform(.01, .1, size=n_parts)
    }
    initials.append(chaininit)

#%%
# Set random seed
np.random.seed(23)

# Fitting model
print('Fitting model ...')
threaded = pyjags.Model(file=model_file, init=initials,
                        data=dict(y=y, N_obs=N_obs, N200=N200, Ones=Ones, Constant=Constant, n_parts=n_parts, participant=participant),
                        chains=n_chains, adapt=burnin, threads=6,
                        progress_bar=True)
samples = threaded.sample(n_samps, vars=trackvars, thin=10)
savestring = ('hier_genparam_N200_DDM.mat')
print('Saving results to: \n %s' % savestring)
sio.savemat(savestring, samples)

# Diagnostics
# The printed diagnostics report the minimum/maximum for all parameters (not just for a single parameter)
samples_load = sio.loadmat(savestring)
samples_diagrelevant = samples_load.copy()
samples_diagrelevant.pop('DDM_or_Lapse', None) #Remove variable DDMorLapse to obtain Rhat diagnostics
diags = phju.diagnostic(samples_diagrelevant)

#%%
# 95% and 5% percentiles
post_ndt_gamma = samples['ndt_gamma'].flatten()
percentile_5 = np.percentile(post_ndt_gamma, 5)
percentile_95 = np.percentile(post_ndt_gamma, 95)
print("5th Percentile:", percentile_5)
print("95th Percentile:", percentile_95)

# Median
median_value = np.median(post_ndt_gamma)
print("Median:", median_value)

# Calculate the Savage-Dickey ratio for ndt_gamma
gamma_kde = gaussian_kde(post_ndt_gamma)

estimate = 1
posterior_density_at = gamma_kde.evaluate(estimate)[0]

# Evaluate the prior density
prior_density_at = norm.pdf(estimate, loc=estimate, scale=3)
# Calculate the Savage-Dickey Density Ratio
savage_dickey_ratio = posterior_density_at / prior_density_at
print(f"Savage-Dickey Density Ratio at {estimate}:", savage_dickey_ratio)

#%%
#Posterior distributions
plt.figure()
phju.jellyfish(samples['ndt_int'])
plt.title('Posterior distributions of ndt_int')
plt.savefig(('ndt_int.png'), format='png',bbox_inches="tight")

plt.figure()
phju.jellyfish(samples['ndt_gamma'])
plt.title('Posterior distributions of ndt_gamma')
plt.savefig(('ndt_gamma.png'), format='png',bbox_inches="tight")

