install.packages("rstan", repos = "https://cloud.r-project.org/", dependencies = TRUE)
library("rstan")
example(stan_model, package = "rstan", run.dontrun = TRUE)
install.packages("brms")
library(brms)
library(bayesplot)
library(density)

# Import data and exclude outliers if necessary
data <- read.csv("behavioral_data_modelling.csv")
filtered_data <- subset(data, rt > 250 & rt < 1500)

# Specify the prior for the slope parameter (b_latency)
# Normal distribution with mean = 1 and sd = 3
slope_prior <- set_prior("normal(1, 3)", class = "b", coef = "latency")

# Fit the Bayesian linear regression model
fit1 <- brm(rt ~ latency,
            data = data,
            prior = slope_prior,
            family = gaussian(),
            seed = 123)

# View the summary of the model
summary(fit1)
plot(fit1)
# Plotting posterior distributions
tiff("mcmc_histograms.tiff", width = 2500, height = 1000, res = 500)
mcmc_hist(as.array(fit1), pars = c("b_Intercept", "b_latency"))
dev.off()

# Extract b_latency from each chain
posterior_draws <- as_draws(fit1)
latency_chain1 <- posterior_draws[[1]]$b_latency
latency_chain2 <- posterior_draws[[2]]$b_latency
latency_chain3 <- posterior_draws[[3]]$b_latency
latency_chain4 <- posterior_draws[[4]]$b_latency
# Combine the latency samples from all chains into one vector
combined_latency_samples <- c(latency_chain1, latency_chain2, latency_chain3, latency_chain4)

# Calculate the Savage-Dickey ratio where the slope parameter equals to 1
posterior_density <- density(combined_latency_samples, kernel="gaussian")
posterior_at_1 <- approx(posterior_density$x, posterior_density$y, xout = 1)$y
prior_at_1 <- dnorm(1, mean = 1, sd = 3)
bf <- posterior_at_1 / prior_at_1

# Fit the logistic regression model
fit2 <- brm(accuracy ~ latency,
            data = data,
            family = bernoulli(),
            seed = 123)

# View the summary of the model
summary(fit2)
plot(fit2)









