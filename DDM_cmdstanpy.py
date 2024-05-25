import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import arviz as az
from cmdstanpy import CmdStanModel
from scipy.stats import gaussian_kde, norm
#cmdstanpy.install_cmdstan()
#cmdstanpy.install_cmdstan(compiler=True)  # only valid on Windows

#%%
participant_ID = 1
working_dir = 'D:/research/internship/data_analysis'
modelling_data = pd.read_csv(f'{working_dir}/modelling_data_N200_{participant_ID}.csv')

N_obs = (modelling_data['N200_yes_or_no'] == 1).sum()
N_mis = (modelling_data['N200_yes_or_no'] == 0).sum()
accu_rt = modelling_data['accu_rt'] * 0.001
filtered_data = modelling_data[modelling_data['N200_yes_or_no'] == 1]
latency_values = filtered_data['latency'] * 0.001

#%%
stan_file = 'D:/PycharmProjects/Internship_2024_EEG_DDM/N200_latency_model.stan'
model = CmdStanModel(stan_file=stan_file)
print(model)
print(model.exe_info())

#%%
# Create a dictionary storing the input data
my_data = {
    'N_obs': N_obs,
    'N_mis': N_mis,
    'y': accu_rt,
    'n200lat_obs': latency_values
}

#%%
# Not suggested, but
# if cmdstanpy cannot convert the dictionary into a json file
# This is used to manually create a json file

class CustomEncoder(json.JSONEncoder):
    """ Custom encoder for numpy and pandas data types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        return super(CustomEncoder, self).default(obj)

#file_path = 'D:/PycharmProjects/Internship_2024_EEG_DDM/ddm_data.json'

# Write the dictionary to a file
#with open(file_path, 'w') as json_file:
    #json.dump(my_data, json_file, cls=CustomEncoder, indent=4)


#%%
# Generate initial values
n_chains = 4
initials = []  # list to hold dictionaries of initial values
for _ in range(n_chains):
    chain_init = {
        'res': np.random.uniform(0.01, 0.02),
        'lambda': np.random.uniform(0.01, 0.02),
        'delta': np.random.uniform(1, 3),
        'alpha': np.random.uniform(0.5, 1),
        'n200sub': np.random.uniform(0.15, 0.25),
        'n200lat_mis': np.random.uniform(0.15, 0.25, size=N_mis)
    }
    initials.append(chain_init)

#%%
fit = model.sample(data=my_data,
                   chains=n_chains,
                   iter_warmup=1000,
                   iter_sampling=1000,
                   adapt_delta=0.95,
                   max_treedepth=15,
                   inits=initials)
                   # show_console=True)

#%%
print(fit.diagnose())

#%%
summary_df = fit.summary()
summary_df.to_csv('diagnostics.csv')

#%%
# Calculate the savage-dickey ratio
lambda_samples = fit.stan_variable('lambda')

# Fit KDE to the posterior samples of lambda
lambda_kde = gaussian_kde(lambda_samples)

# Evaluate the density at lambda = 1
posterior_density_at_1 = lambda_kde.evaluate(1)[0]

# Assuming a normal prior for lambda: N(mu, sigma)
mu = 1  # Mean of the prior distribution
sigma = 3  # Standard deviation of the prior distribution

# Evaluate the prior density at lambda = 1
prior_density_at_1 = norm.pdf(1, loc=mu, scale=sigma)

# Calculate the Savage-Dickey Density Ratio
savage_dickey_ratio = posterior_density_at_1 / prior_density_at_1

print("Savage-Dickey Density Ratio:", savage_dickey_ratio)

#%%
# Generating trace plots
# Convert CmdStanPy fit to an ArviZ InferenceData object
total_data_points = my_data['N_obs'] + my_data['N_mis']  # Total count of data points

coords = {
    "data_id": np.arange(total_data_points),
    "obs_id": np.arange(my_data["N_obs"])
}

dims = {
    "y": ["data_id"],  # 'y' spans all combined data points
    "n200lat_obs": ["obs_id"]  # Observed latency values only index observed data
}

inference_data = az.from_cmdstanpy(
    posterior=fit,
    observed_data={
        "y": my_data["y"],
        "n200lat_obs": my_data["n200lat_obs"]
    },
    log_likelihood="log_lik",
    coords=coords,
    dims=dims
)

# Generate trace plots
az.plot_trace(inference_data, var_names=['lambda'])
plt.show()

#%%
# Generating autocorrelation plots
autocorr = az.plot_autocorr(inference_data, var_names=['lambda'])
plt.show()

