functions {
  
  // Linear algebra functions in Stan: https://mc-stan.org/docs/2_28/functions-reference/linear-algebra-functions-and-solvers.html 
  
  vector gp_pred_rng(real[] season_days_pred,
                     vector delta_xg_fit, real[] season_days_fit,
                     real alpha, real rho, real sigma, real delta) {
                       
    int n_Kr = rows(delta_xg_fit);
    int n_Lr = size(season_days_pred);
    
    vector[n_Lr] f2;
    {
      
      matrix[n_Kr, n_Kr] K = gp_exp_quad_cov(season_days_fit, alpha, rho)+
                             diag_matrix(rep_vector(square(sigma), n_Kr));
      
      matrix[n_Kr, n_Kr] L_K = cholesky_decompose(K);

      vector[n_Kr] L_K_div_delta_xg = mdivide_left_tri_low(L_K, delta_xg_fit);                     // mdivide_left_tri_low(matrix_A, vector_B) = [A |> tri() |> inverse()] * B
      
      vector[n_Kr] K_div_delta_xg = mdivide_right_tri_low(L_K_div_delta_xg', L_K)';            // mdivide_right_tri_low(matrix_A, vector_B) = B * [A |> tri() |> inverse()]
                                                                                             // Where ' is the transpose operator
      matrix[n_Kr, n_Lr] k_season_days_fit_season_days_pred = gp_exp_quad_cov(season_days_fit, season_days_pred, alpha, rho);  
      
      vector[n_Lr] f2_mu = (k_season_days_fit_season_days_pred' * K_div_delta_xg);                             
      
      matrix[n_Kr, n_Lr] v_pred = mdivide_left_tri_low(L_K, k_season_days_fit_season_days_pred);              // mdivide_left_tri_low(matrix_A, matrix_B) = A |> tri()) |> inverse() * B
      
      matrix[n_Lr, n_Lr] cov_f2 = gp_exp_quad_cov(season_days_pred, alpha, rho) - v_pred' * v_pred +
                                  diag_matrix(rep_vector(delta, n_Lr));                    // add small delta to matrix is +ve definite (a valid co-variance matrix)
      
      f2 = multi_normal_rng(f2_mu, cov_f2);
      
    }
    return f2;
  }
}

data {
  
  int <lower = 1> n_matches;
  real season_days_fit [n_matches];
  vector [n_matches] delta_xg_fit;

  int<lower=1> n_pred;
  real season_days_pred [n_pred];
  
  real mu_rho;                  
  real <lower = 0> sigma_rho;   

  real mu_alpha;                
  real <lower = 0> sigma_alpha;

  real <lower = 0> sigma_rate;
  
}

parameters {
  
  real <lower = 0> rho;
  real <lower = 0> alpha;
  real <lower = 0> sigma;

}

model {
  // Define the exponentiated-quadratic covariance function
  matrix[n_matches, n_matches] cov = gp_exp_quad_cov(season_days_fit, alpha, rho)+
                                                     diag_matrix(rep_vector(square(sigma), n_matches));
                             
  matrix[n_matches, n_matches] L_cov = cholesky_decompose(cov);
  
  // Priors on parameters describing co-variance matrix, K
  target += lognormal_lpdf(rho | mu_rho, sigma_rho);
  target += lognormal_lpdf(alpha | mu_alpha, sigma_alpha);

  // Prior on Kr variance parameter
  target += exponential_lpdf(sigma | sigma_rate);
  
  delta_xg_fit ~ multi_normal_cholesky(rep_vector(0, n_matches), L_cov);
}

generated quantities {
  vector [n_pred] f_pred = gp_pred_rng(season_days_pred, delta_xg_fit, season_days_fit, alpha, rho, sigma, 1e-10);
  real delta_xg_pred [n_pred] = normal_rng(f_pred, sigma);
}
