# Q-learning approach to solve a Goal-based Wealth Management problem
TEC_QL_GBWM_solve <- function(w0, G, T, n_states, n_pf, mu_vals, sig_vals,
                                 epochs, epsilon, alpha, seed) {
  
  set.seed(seed)  # Seed control for the simulations
  
  # Grid of wealth, including cumulative transition probabilities
  res <- TEC_QL_GBWM_build_grid(w0, G, T, mu_vals, sig_vals, n_states, 3)
  W <- res$w_array
  start_idx <- res$start_idx
  start_ctpm <- res$start_ctpm
  ctpm <- res$ctpm
  
  # Initialize Q-table
  Q <- array(0, dim = c(n_states, T + 1, n_pf))
  
  for (epoch in 1:epochs) {
    idx <- start_idx
    for (t in 1:(T + 1)) {
      # choose action with epsilon-greedy policy
      a0 <- get_action(Q, idx, t, epsilon, n_pf)
      
      tnext <- t + 1
      if (t < (T + 1)) {
        # Random choice of the next state, using the appropriate transitions.
        if (t == 1) {
          idx1 <- sum(start_ctpm[, a0] < runif(1)) + 1
        } else {  # t > 1
          idx1 <- sum(ctpm[idx, , a0] < runif(1)) + 1
        }
        # Index of the Best Action associated to the next state (in tnext):
        # FILL IN  / LIGNE CAVIARDÉE:
        a_next <- which.max(Q[idx1, tnext, ])
        # Q-factor update :
        # FILL IN  / LIGNE CAVIARDÉE:
        Q[idx, t, a0] <- (1 - alpha) * Q[idx, t, a0] + alpha * Q[idx1, tnext, a_next]
      } else {  # i.e. t==T+1, the end of the horizon
        # Is the goal reached ?
        # FILL IN  / LIGNE CAVIARDÉE:
        reward <- as.numeric(W[idx] >= G)
        # Q-factor update :
        # FILL IN  / LIGNE CAVIARDÉE:
        Q[idx, t, ] <- (1 - alpha) * Q[idx, t, ] + alpha * reward
        
        #idx1 <- idx   This command removed, was redundant ! MD 11 2026
      }
      idx <- idx1  
    }
  }
  
  # Compute success rate and optimal policy
  success <- max(Q[start_idx, 1, ])
  optimal_policy <- apply(Q, c(1, 2), which.max)
  
  list(success = success, Q = Q, optimal_policy = optimal_policy)
}

# get_action uses an epsilon-greedy policy to select the next action
get_action <- function(Q, idx, t, epsilon, n_pf) {
  
  if (runif(1) < epsilon) {
    # Call randomaction to pick a random action
    action <- randomaction(n_pf)
  } else {
    q <- Q[idx, t, ]
    max_idx <- as.numeric(q == max(q))
    
    if (sum(max_idx) > 1) {
      possible_actions <- which(max_idx == 1)
      action_idx <- randomaction(length(possible_actions))
      action <- possible_actions[action_idx]
    } else {
      action <- which.max(q)
    }
  }
  action
}

# randomaction function
randomaction <- function(n_pf) {
  vals <- cumsum(rep(1 / n_pf, n_pf))
  rnd <- runif(1)
  sum(vals < rnd) + 1
}