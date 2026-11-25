# Devoir 3 : GBWM par Q-learning et SARSA (Das et Varma, 2020)
#
# ROUTINE PRINCIPALE : c'est ce fichier qu'il faut exécuter. Il produit les tableaux
# (console) et les graphiques du rapport (onglet Plots de RStudio et PDF dans outputs/).
# Sous-programmes appelés :
#   code cours/TEC_QL_GBWM_build_grid.R : grille de richesse du cours (non modifiée)
#   src/functions.R                     : apprentissage QL/SARSA, tests hors échantillon, expériences
#   src/graphs.R                        : graphiques
#
# Durée d'exécution :
#   full_run <- FALSE (par défaut) : nombres d'épisodes divisés par 10, moins de répétitions,
#                                    environ 2 min. Sert à vérifier que tout fonctionne.
#   full_run <- TRUE               : réglages de l'article, résultats du rapport, environ 26 min.
#
# Exemples d'appels des fonctions (voir src/functions.R) :
#   grid <- build_grid(w0 = 100, G = 200, horizon = 10, mu_vals, sig_vals, n_w = 101)
#   Q    <- train(grid, G = 200, epochs = 50000, epsilon = 0.3, alpha = 0.1, method = "SARSA",
#                 schedule = "paquets")$Q
#   evaluate_policy(Q, grid, G = 200, n_sim = 10000, seed = 2026)
#   run_experiments(make_cases(method = c("QL", "SARSA"), epsilon = c(0.1, 0.3)), grid, G = 200,
#                   n_sim = 5000, oos_seed = 2026)

rm(list = ls())
library(here)                                  # chemins relatifs à la racine du projet (Devoir 3.Rproj)
source(here("code cours", "TEC_QL_GBWM_build_grid.R"))
source(here("src", "functions.R"))
source(here("src", "graphs.R"))
dir.create(here("outputs"), showWarnings = FALSE)
t_start <- Sys.time()

# ---- Mode d'exécution ----
full_run <- FALSE                              # TRUE : résultats complets du rapport (long)
scale <- if (full_run) 1 else 0.1              # facteur appliqué à tous les nombres d'épisodes
episodes <- function(n) as.integer(round(n * scale))

# ---- Graines ----
seed_train <- 123                              # entraînements (résultats reproductibles)
seed_oos <- 2026                               # tests hors échantillon : graine DIFFÉRENTE -> nouvelles trajectoires
set.seed(seed_train)

# ---- Paramètres du problème (article, tableau 1, p. 17) ----
w0 <- 100                                      # richesse initiale
G <- 200                                       # objectif
horizon <- 10                                  # horizon en années
n_w <- 101                                     # nombre de noeuds de la grille de richesse
mu_vals <- c(0.0526, 0.0552, 0.0577, 0.0603, 0.0629, 0.0655, 0.0680,
             0.0706, 0.0732, 0.0757, 0.0783, 0.0809, 0.0835, 0.0860, 0.0886)
sig_vals <- c(0.0485, 0.0486, 0.0493, 0.0508, 0.0529, 0.0556, 0.0587,
              0.0623, 0.0662, 0.0705, 0.0749, 0.0796, 0.0844, 0.0894, 0.0945)
dp_benchmark <- 0.72                           # solution DP de l'article (étalon)

# ---- Paramètres des expériences ----
n_sim <- if (full_run) 5000 else 2000          # trajectoires hors échantillon (erreur type <= 0,007 en mode complet)
n_sim_curve <- if (full_run) 2000 else 1000    # trajectoires à chaque point des courbes d'apprentissage
n_seeds <- if (full_run) 10 else 3             # graines pour la variabilité des réglages de l'article
n_reps <- if (full_run) 3 else 2               # répétitions par configuration (moyennes)

grid <- build_grid(w0, G, horizon, mu_vals, sig_vals, n_w)

# ---- 1. Réplication in-sample du tableau 1 (QL, alpha = 0.1 constant) + tests hors échantillon ----
# Mêmes lignes que le tableau 1 de l'article (sauf DP), avec la valeur publiée pour comparaison.
table1 <- data.frame(epsilon = c(0.10, 0.10, 0.20, 0.20, 0.25, 0.30, 0.30, 0.40, 0.40, 0.40, 0.40),
                     epochs  = c(50, 100, 50, 100, 100, 50, 100, 50, 100, 200, 500) * 1000,
                     article = c(0.65, 0.65, 0.69, 0.71, 0.71, 0.72, 0.72, 0.73, 0.77, 0.75, 0.71))
section("1. Réplication")
res_replication <- run_experiments(do.call(rbind, Map(function(e, n) make_cases(epsilon = e, epochs = episodes(n)),
                                                      table1$epsilon, table1$epochs)),
                                   grid, G, n_sim, seed_oos)
res_replication$article <- table1$article

# ---- 2. Variabilité selon la graine : réglages de l'article (QL, epsilon = 0.3, alpha = 0.1 constant) ----
section("2. Graines")
res_seeds <- run_experiments(make_cases(epochs = episodes(c(50000, 100000)), rep = 1:n_seeds),
                             grid, G, n_sim, seed_oos)

# ---- 3. QL vs SARSA : niveaux d'exploration et d'apprentissage (alpha par paquets) ----
section("3. QL vs SARSA")
res_comparison <- run_experiments(make_cases(method = c("QL", "SARSA"), epsilon = c(0.1, 0.3, 0.5),
                                             alpha = c(0.1, 0.5), schedule = "paquets",
                                             epochs = episodes(50000), rep = 1:n_reps),
                                  grid, G, n_sim, seed_oos)

# ---- 4. Schémas du taux d'apprentissage : courbes d'apprentissage ----
section("4. Schémas alpha")
runs <- run_learning_curves(make_cases(method = c("QL", "SARSA"), schedule = c("constant", "paquets", "1/k", "1/N"),
                                       epochs = episodes(100000)),
                            grid, G, checkpoint_every = episodes(5000), n_sim = n_sim_curve, oos_seed = seed_oos)

# ---- 5. Entraînement long (convergence), QL et SARSA, alpha par paquets ----
section("5. Entraînement long")
run_long <- run_learning_curves(make_cases(method = c("QL", "SARSA"), schedule = "paquets",
                                           epochs = episodes(500000)),
                                grid, G, checkpoint_every = episodes(25000), n_sim = n_sim_curve,
                                oos_seed = seed_oos)

# ---- 6. Exploration décroissante : epsilon passe de 0.3 à 0.05 (sans être éliminée) ----
section("6. Epsilon décroissant")
res_eps_decay <- run_experiments(make_cases(method = c("QL", "SARSA"), epsilon_min = c(NA, 0.05),
                                            schedule = "paquets", epochs = episodes(50000), rep = 1:n_reps),
                                 grid, G, n_sim, seed_oos)

# ---- 7. Initialisation optimiste (Q = 1) face à l'exploration epsilon, QL ----
section("7. Init. optimiste")
res_optimistic <- run_experiments(make_cases(epsilon = c(0.05, 0.3), q0 = c(0, 1), schedule = "paquets",
                                             epochs = episodes(50000), rep = 1:n_reps),
                                  grid, G, n_sim, seed_oos)

# ---- Résultats ----
cat("\n\n================ RÉSULTATS ================\n")
if (!full_run) cat("MODE RAPIDE : épisodes divisés par", 1 / scale, "(full_run <- TRUE pour le rapport)\n")
cat("Étalon DP (article) :", dp_benchmark, "   Trajectoires hors échantillon :", n_sim,
    "   Graines : entraînement", seed_train, "/ hors échantillon", seed_oos, "\n")
print_table(res_replication[c("epsilon", "epochs", "article", "in_sample", "oos_greedy", "oos_se",
                              "oos_explore", "time_s")],
            "1. Réplication du tableau 1 (QL, alpha = 0.1 constant)")
print_table(seed_summary(res_seeds),
            paste("2. Variabilité sur", n_seeds, "graines (QL, epsilon = 0.3, alpha = 0.1 constant)"))
print_table(mean_by(res_comparison, c("method", "epsilon", "alpha", "epochs")),
            paste("3. QL vs SARSA, alpha par paquets (moyennes sur", n_reps, "répétitions)"))
print_table(final_points(runs)[c("method", "schedule", "epochs", "in_sample", "oos", "time_s")],
            "4. Schémas du taux d'apprentissage, valeurs finales (epsilon = 0.3)")
print_table(final_points(run_long)[c("method", "schedule", "epochs", "in_sample", "oos", "time_s")],
            "5. Entraînement long, valeurs finales (epsilon = 0.3)")
print_table(mean_by(res_eps_decay, c("method", "epsilon_min", "epochs")),
            paste("6. Exploration décroissante de 0.3 vers epsilon final (moyennes sur", n_reps, "répétitions)"))
print_table(mean_by(res_optimistic, c("epsilon", "q0", "epochs")),
            paste("7. Initialisation optimiste, QL (moyennes sur", n_reps, "répétitions)"))

plot_learning_curves(runs, dp_benchmark, here("outputs", "1_courbes_apprentissage.pdf"))
plot_oos_vs_epsilon(res_comparison, dp_benchmark, here("outputs", "2_succes_vs_epsilon.pdf"))
plot_policy_maps(Filter(function(run) run$curve$schedule[1] %in% c("constant", "1/N"), runs),
                 grid, G, here("outputs", "3_cartes_politique.pdf"))
plot_seed_distribution(res_seeds, dp_benchmark, here("outputs", "4_distribution_graines.pdf"))
plot_learning_curves(run_long, dp_benchmark, here("outputs", "5_entrainement_long.pdf"))

cat(sprintf("\nGraphiques enregistrés dans outputs/. Durée totale : %.1f min\n",
            as.numeric(difftime(Sys.time(), t_start, units = "mins"))))
