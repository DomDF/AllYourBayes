data {

  int <lower = 0> N; // Defining the number of defects in the test dataset
  int <lower = 0, upper = 1> det [N]; // A variable that describes whether each defect was detected [1]or not [0]
  vector <lower = 0> [N] depth; // A variable that describes the corresponding depth of each defect
  
  int <lower = 0> K; // Defining the number of probabilistic predictions required from the model
  vector <lower = 0> [K] depth_pred;
  
}

parameters {
  
  // The (unobserved) model parameters that we want to recover
  real alpha;
  real beta;
  
}

model {

  // A logistic regression model relating the defect depth to whether it will be detected
  det ~ bernoulli_logit(alpha + beta * log(depth));
  
  // Prior models for the unobserved parameters
  alpha ~ normal(0, 1);
  beta ~ normal(1, 1);

}

generated quantities {
  
  // Using the fitted model for probabilistic prediction.
  // K posterior predictive distributions will be estimated for a corresponding crack depth
  vector [K] postpred_pr;
  
  for (k in 1:K) {
    
    postpred_pr[k] = inv_logit(alpha + beta * log(depth_pred[k]));
    
  }
  
}