// Fully Pooled Models

functions{

  real log_norm_sigma(real norm_mu, real norm_sigma){
    
    return log(1 + (norm_sigma^2 / norm_mu^2));
    
  }
  
  real log_norm_mu(real norm_mu, real norm_sigma){
    
    return log(norm_mu) - 0.5 * log_norm_sigma(norm_mu, norm_sigma)^2;
    
  }
  
}

data {
  
  int <lower = 1> N; // Number of data points
  int <lower = 1> n_A; // Number of anomalies
  int <lower = 0> n_M; // Number of anomalies missed

  int <lower = 1> ID [N]; // Defect identifier

  vector [n_A] depth_i1; // Measured corrosion depth
  vector [n_A] depth_i2; // Measured corrosion depth
  vector <lower = 0> [n_A] error_i1; // Measurement error parameter
  vector <lower = 0> [n_A] error_i2; // Measurement error parameter

  real d_years; // Time of measurement

  vector [n_A] ex_1; // Indicator: was the measurement made in normal conditions?
  vector [n_A] ex_2; // Indicator: was the measurement made in normal conditions?
  
  real mu_mu_beta; // Prior corrosion growth rate parameter
  real <lower = 0> sigma_mu_beta; // Prior corrosion growth rate parameter
  real <lower = 0> rate_sigma_beta; // Prior corrosion growth rate parameter
  
  real mu_depth_imp;  // Prior on missing data
  real <lower = 0> sigma_depth_imp;  // Prior on missing data
  
//  real <lower = 0> rate_ex_error;
  
}

transformed data{
  
}

parameters {

  real mu_beta;
  real <lower = 0> sigma_beta;
  
//  real <lower = 0> ex_error;
  
  vector <lower = 0> [n_A] depth_true_i1;
  vector <lower = 0> [n_A] delta_depth;
  
}

transformed parameters {

  vector [n_A] depth_true_i2 = depth_true_i1 + delta_depth;
  vector [n_A] growth; // Growth rate depth of corrosion
  
  for (i in 1:n_A){
    
    growth[i] = (depth_true_i2[i] - depth_true_i1[i]) / (d_years);
  
  }

}

model {
  
  // Model
  
  for (n in 1:n_A) {

    if (ex_1[n] == 0){
      
      depth_i1[n] ~ normal(depth_true_i1[n], error_i1[n]);
    
    }
    
    if (ex_1[n] == 1) {

      depth_true_i1[n] ~ lognormal(mu_depth_imp, sigma_depth_imp);
    
    }
    
    if (ex_2[n] == 0){
      
      depth_i2[n] ~ normal(depth_true_i2[n], error_i2[n]);
      
    }
    
    if (ex_2[n] == 1) {

      depth_true_i2[n] ~ lognormal(mu_depth_imp, sigma_depth_imp);
      
    }

  }
      
  growth ~ lognormal(mu_beta, sigma_beta);
  
//  Priors
  
  target += normal_lpdf(mu_beta | mu_mu_beta, sigma_mu_beta);
  target += exponential_lpdf(sigma_beta | rate_sigma_beta);
  
}

generated quantities {
  
   real CGR_pp = lognormal_rng(mu_beta, sigma_beta);

}
