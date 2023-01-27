data {
  
  int <lower = 1> n_matches;         // Number of matches
  real season_days_fit [n_matches];      // Lr values used to fit model
  vector [n_matches] delta_xG_fit;   // Kr values used to fit model
  
  int <lower = 1> n_pred;           // number of posterior predictive distributions to sample from
  real season_days_pred [n_pred];   // Lr values at posterior predictive distributions

  real mu_rho;                  // Prior parameter on length-scale in co-variance matrix
  real <lower = 0> sigma_rho;   // Prior parameter on length-scale in co-variance matrix

  real mu_alpha;                // Prior parameter on max co-variance
  real <lower = 0> sigma_alpha; // Prior parameter on max co-variance

  real <lower = 0> sigma_rate;  // Prior on within-parameter (Kr) variance

}

transformed data{

  int <lower = 1> n_total = n_matches + n_pred;   // Total number of points to evaluate
  real season_days [n_total];                                     // Labelling all Lr values
  for (n1 in 1:n_matches) season_days[n1] = season_days_fit[n1];
  for (n2 in 1:n_pred) season_days[n_matches + n2] = season_days_pred[n2];

}

parameters {

  real <lower = 0> rho;   // Length-scale
  real <lower = 0> alpha; // Maximum co-variance
  real <lower = 0> sigma; // Kr variance parameter
  vector [n_total] eta;   // Standard normal parameter for non-centered reparameterisation
}

transformed parameters {

  vector [n_total] f;             // Parameters describing uncertainty in functions at each Lr
  matrix [n_total, n_total] K;    // Covariance matrix
  matrix [n_total, n_total] L_K;  // L_K = K * transpose(K)
  
  K = cov_exp_quad(season_days, alpha, rho) +     // Stan function for a exponential quadrtaic matrix
      diag_matrix(rep_vector(1e-9, n_total)); // Ensures matrix is +ve definite (a valid co-variance matrix)
      
  L_K = cholesky_decompose(K);          // Cholesky decomposition of co-variance matrix for computational efficiency
    
  f = L_K * eta;                        // Latent variable non-centered parameteriation
  
}

model {
  
  // Priors on parameters describing co-variance matrix, K
  target += lognormal_lpdf(rho | mu_rho, sigma_rho);
  target += lognormal_lpdf(alpha | mu_alpha, sigma_alpha);

  // Prior on Kr variance parameter
  target += exponential_lpdf(sigma | sigma_rate);

  // Non-centered parameterisation requires eta to be defined as a standard normal distribution
  eta ~ std_normal();

  // Each Kr is normally distributed with a mean equal to the parameter at the corresponding Lr, 
  // and standard deviation equal to sigma
  delta_xG_fit ~ normal(f[1:n_matches], sigma);

}

generated quantities {

  // The posterior predictive distribution is evaluated at each Lr_pred
  vector [n_pred] delta_xG_pred;
  // vector [N2] loglik;
  
  for(i in 1:n_pred) {
    
    delta_xG_pred[i] = normal_rng(f[n_matches + i], sigma);
    
    // loglik[n2] = normal_lpdf(y2[n2] | f[N1 + n2], sigma);
    
  }
}