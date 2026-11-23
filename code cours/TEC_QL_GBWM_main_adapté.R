# Main function to set the problem's parameter and call the function TEC_QL_GBWM_solve_QL

rm(list = ls())                                # clear all data and functions

source("TEC_QL_GBWM_build_grid.R")
source("TEC_QL_GBWM_solve_CANEVAS_complété2026.R")

w0 <- 100                                      # initial wealth
G <- 200                                       # goal
T <- 10                                        # horizon
nw <- 101                                      # nb of wealth points
np <- 15                                       # nb of portfolios considered
epsi <- 0.3                                    # level of exploration (vs exploitation)

# Parameters of the efficient portfolios
mu_vals <- c(0.0526, 0.0552, 0.0577, 0.0603, 0.0629, 0.0655, 0.0680,
             0.0706, 0.0732, 0.0757, 0.0783, 0.0809, 0.0835, 0.0860, 0.0886)
sig_vals <- c(0.0485, 0.0486, 0.0493, 0.0508, 0.0529, 0.0556, 0.0587,
              0.0623, 0.0662, 0.0705, 0.0749, 0.0796, 0.0844, 0.0894, 0.0945)

seed <- sample.int(999999, 1)   # Seed control for the simulations. Requires a positive integer.
                                # Replace by a fixed integer if desired (fixed behaviour then)

## Train and test QL
epochs <- 50000                                # Size of training set
alpha <- 0.1                                   # Fix learning rate

tic <- Sys.time()                              # start computing stopwatch

result <- TEC_QL_GBWM_solve(w0, G, T, nw, np, mu_vals, sig_vals,
                               epochs, epsi, alpha, seed)
suc_rl <- result[[1]]
Q_rl <- result[[2]]
op_rl <- result[[3]]

print(suc_rl)

toc <- Sys.time()                              # stop stopwatch
toc - tic