# Outputs of the function: 
#  w_array: grid of wealths, in dollars
#  start_idx:  index of the grid point that is just below W0
#  start_ctpm: cumulative transition probability matrix for initial time
#  ctpm : cumulative transition probability matrix for all other times

##
## The rest of the function is relatively complex and irrelevant for the purpose
##  of the course. No need to either change nor understand it.
##

TEC_QL_GBWM_build_grid <- function(w0, G, T, mu_vals, sig_vals, n_states, n_stdev) {
  
  # Function that constructs the grid using the midpoint algorithm.
  mu_min <- min(mu_vals)
  mu_max <- max(mu_vals)
  sig <- max(sig_vals)
  
  w_max <- w0 * exp((mu_max - 0.5 * sig^2) * T + n_stdev * sig * sqrt(T))
  w_min <- w0 * exp((mu_min - 0.5 * sig^2) * T - n_stdev * sig * sqrt(T))
  lnw_array <- seq(log(w_min), log(w_max), length.out = n_states)
  
  # shift grid so that the w=ln(G) is exactly between two ticks
  lnG <- log(G)
  IG <- which.min(abs(lnw_array - lnG)) # indice de la valeur la plus proche de G
  # find second closest value to G
  if (abs(lnw_array[IG + 1] - lnG) > abs(lnw_array[IG - 1] - lnG)) {
    IwTop <- IG
    IwBottom <- IG - 1
  } else {
    IwTop <- IG + 1
    IwBottom <- IG
  }
  # log wealths around G
  top <- lnw_array[IwTop]
  bottom <- lnw_array[IwBottom]
  # calculate the midpoint between both grid points
  midpoint <- bottom + (top - bottom) / 2
  # calculate midpoint adjustment
  midpoint_diff <- midpoint - lnG
  # adjust log grid
  lnw_array <- lnw_array - midpoint_diff
  
  w_array <- exp(lnw_array)
  
  # Calculate the transition probabilities tensor
  n_pf <- length(mu_vals)
  tpm <- array(0, dim = c(n_states, n_states, n_pf))
  ctpm <- array(0, dim = c(n_states, n_states, n_pf))
  # because w0 is probably not on the grid, we need to calculate the
  # transition probabilities separately.
  start_ctpm <- matrix(0, nrow = n_states, ncol = n_pf)
  
  for (action in 1:n_pf) {
    mu <- mu_vals[action]
    sig <- sig_vals[action]
    for (i in 1:n_states) {
      z <- (log(w_array / w_array[i]) - (mu - 0.5 * sig^2)) / sig
      p <- dnorm(z)  # computes the prob. density function of a N(0,1)
      tpm[i, , action] <- p / sum(p)
      ctpm[i, , action] <- cumsum(tpm[i, , action])
    }
    
    z <- (log(w_array / w0) - (mu - 0.5 * sig^2)) / sig
    p <- dnorm(z)
    start_ctpm[, action] <- cumsum(p / sum(p))
  }
  
  # Point de la grille de richesse le plus proche de W0
  start_idx <- which.min(abs(w_array - w0))
  
  list(w_array = w_array, start_idx = start_idx,
       start_ctpm = start_ctpm, ctpm = ctpm)
}