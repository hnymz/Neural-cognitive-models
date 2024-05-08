// Adapted from the code in Amin Ghaderi-Kangavari et al. (2022)
functions {
  /* Model 1
   * Ratcliff diffusion log-PDF for a single response (adapted from brms 1.10.2 and hddm 0.7.8)
   * Arguments:
   *   Y: acc*rt in seconds (negative and positive RTs for incorrect and correct responses respectively)
   *   boundary: boundary separation parameter > 0
   *   ndt: non-decision time parameter > 0
   *   bias: initial bias parameter in [0, 1]
   *   drift: mean drift rate parameter across trials
   * Returns:
   *   a scalar to be added to the log posterior
   */
   real ratcliff_lpdf(real Y, real boundary,
                              real ndt, real bias, real drift) {
    real X;
    X = (abs(Y) - ndt); // Remove non-decision time
    if (Y >= 0) {
    return wiener_lpdf( abs(Y) | boundary, ndt, bias, drift );
    } else {
    return wiener_lpdf( abs(Y) | boundary, ndt, 1-bias, -drift );
    }
   }
}
data {
    int<lower=1> N_obs;       // Number of trial-level observations
    int<lower=1> N_mis;       // Number of trial-level missing data
    array[N_obs + N_mis] real y;    // acc*rt in seconds (negative and positive RTs for incorrect and correct responses respectively)
    vector<lower=0>[N_obs] n200lat_obs;      // N200 Latency for observed trials
}
parameters {
    vector<lower=0.101, upper=0.248>[N_mis] n200lat_mis;  // vector of missing data for n200 latency

    real<lower=0> n200latstd;     // n200lat std

    /* main paameter*/
    real<lower=0, upper=6> delta;               // drift rate
    real<lower=0, upper=4> alpha;               // Boundary boundary
    real<lower=0, upper=.4> res;                // residual of Non-decision time
    real<lower=0, upper=.4> n200sub;            // n200 mu parameter
    real<lower=0, upper=3> lambda;              // coefficient paramter
}
transformed parameters {
   vector[N_obs + N_mis] n200lat = append_row(n200lat_obs, n200lat_mis);
}
model {

    n200latstd ~ gamma(.1,.1);

    /* main paameter*/
    delta ~ normal(2, 4) T[0,6];

    res ~ normal(.2, .2) T[0,.4];
    lambda ~ normal(.5, 2) T[0,3];

    n200sub ~ normal(.15,.1) T[0,.4];
    alpha ~ normal(1, 2) T[0,4];
    eta ~ normal(1, 2) T[0,3];

    for (i in 1:N_obs) {
        // Note that N200 latencies are censored between 150 and 350 ms for observed data
        n200lat_obs[i] ~ normal(n200sub, n200latstd) T[.151,.348];
    }
    for (i in 1:N_mis) {
        // Note that N200 latencies are censored between 150 and 350 ms for missing data
        n200lat_mis[i] ~ normal(n200sub, n200latstd) T[.151,.348];
    }

    // Wiener likelihood
    for (i in 1:N_obs + N_mis) {

        // Log density for DDM process
        y[i] ~ ratcliff(alpha, res + lambda*n200lat[i], .5, delta);
    }
}
generated quantities {
   vector[N_obs + N_mis] log_lik;
   vector[N_obs + N_mis] n200lat_lpdf;

    // n200lat likelihood
    for (i in 1:N_obs+N_mis) {
        // Note that N200 latencies are censored between 150 and 350 ms for observed data
        n200lat_lpdf[i] = normal_lpdf(n200lat[i] | n200sub, n200latstd);
    }

   // Wiener likelihood
    for (i in 1:N_obs+N_mis) {
        // Log density for DDM process
         log_lik[i] = ratcliff_lpdf(y[i] | alpha, res + lambda*n200lat[i], .5, delta) + n200lat_lpdf[i];
   }
}
