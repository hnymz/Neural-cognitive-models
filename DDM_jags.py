import numpy as np
import pandas as pd
import pyjags
import scipy.io as sio
from scipy import stats
import matplotlib.pyplot as plt
import pyhddmjagsutils as phju
from scipy.stats import gaussian_kde, norm

#%%
participant_ID = 1
# Loading data
# Change the directory to the Linux form
working_dir = '/mnt/d/research/internship/data_analysis'
modelling_data = pd.read_csv(f'{working_dir}/modelling_data_N200_{participant_ID}.csv')

N_obs = (modelling_data['N200_yes_or_no'] == 1).sum()
N_mis = (modelling_data['N200_yes_or_no'] == 0).sum()
filtered_data = modelling_data[modelling_data['N200_yes_or_no'] == 1]
N200 = filtered_data['latency'] * 0.001
y = filtered_data['accu_rt'] * 0.001

#%%
# Set random seed
np.random.seed(2021)

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
    
    ndt_int ~ dnorm(.2, pow(.2, -2))
    ndt_gamma ~ dnorm(1, pow(3, -2))

    #Probability of a lapse trial
    prob_lapse ~ dnorm(.3, pow(.15, -2)) T(0, 1)
    prob_DDM <- 1 - prob_lapse

    ##########
    # Wiener likelihood and uniform mixture using Ones trick
    for (i in 1:N_obs) {

        # Log density for DDM process of rightward/leftward RT
        ld_comp[i, 1] <- dlogwiener(y[i], alpha, ndt_int + ndt_gamma * N200[i], .5, delta)

        # Log density for lapse trials (negative max RT to positive max RT)
        ld_comp[i, 2] <- logdensity.unif(y[i], -1.8, 1.8)
        
        # Probability of mind wandering trials (lapse trials)
        DDM_or_Lapse[i] ~ dcat( c(prob_DDM, prob_lapse) )        
        
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
burnin = 4000  # Note that scientific notation breaks pyjags
n_samps = 10000

model_file = 'N200_DDM.jags'
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
        'ndt_int': np.random.uniform(.1, .5),
        'ndt_gamma': np.random.uniform(.1, .5),
        'delta': np.random.uniform(-3., 3.),
        'alpha': np.random.uniform(.5, 1.),
        'prob_lapse': np.random.uniform(.01, .1)
    }
    initials.append(chaininit)

#%%
# Fitting model
print('Fitting model ...')
threaded = pyjags.Model(file=model_file, init=initials,
                        data=dict(y=y, N_obs=N_obs, N200=N200, Ones=Ones, Constant=Constant),
                        chains=n_chains, adapt=burnin, threads=6,
                        progress_bar=True)
samples = threaded.sample(n_samps, vars=trackvars, thin=10)
savestring = ('genparam_N200_DDM.mat')
print('Saving results to: \n %s' % savestring)
sio.savemat(savestring, samples)

# Diagnostics
# The printed diagnostics report the minimum/maximum for all parameters (not just for a single parameter)
samples_load = sio.loadmat(savestring)
samples_diagrelevant = samples_load.copy()
samples_diagrelevant.pop('DDM_or_Lapse', None) #Remove variable DDMorLapse to obtain Rhat diagnostics
diags = phju.diagnostic(samples_diagrelevant)

#%%
# Calculate the Savage-Dickey ratio for ndt_gamma at 1
post_ndt_gamma = samples['ndt_gamma'].flatten()
gamma_kde = gaussian_kde(post_ndt_gamma)

#%%
for i in np.arange(0, 3, 0.1):
    posterior_density_at = gamma_kde.evaluate(i)[0]

    # Evaluate the prior density
    prior_density_at = norm.pdf(i, loc=i, scale=3)
    # Calculate the Savage-Dickey Density Ratio
    savage_dickey_ratio = posterior_density_at / prior_density_at
    print(f"Savage-Dickey Density Ratio at {i}:", savage_dickey_ratio)

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

