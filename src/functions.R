# Fonctions du Devoir 3 : GBWM par Q-learning et SARSA (Das et Varma, 2020)
# Les indices de temps en R vont de 1 à horizon + 1 (t = 1 correspond à la date 0).
# Requiert la fonction du cours TEC_QL_GBWM_build_grid (code cours/TEC_QL_GBWM_build_grid.R).

# Grille de richesse et probabilités de transition : celles du cours, sans
# modification (TEC_QL_GBWM_build_grid, n_sd = 3 comme dans TEC_QL_GBWM_solve).
# Grille équidistante en log-richesse, décalée pour que G tombe entre deux noeuds.
# Retourne :
#   W          : richesses des noeuds de la grille
#   ctpm       : ctpm[i, , k] = probabilités cumulées du noeud i vers les noeuds 1..n_w (portefeuille k)
#   start_ctpm : start_ctpm[, k] = idem, depuis la richesse initiale exacte w0 (hors grille)
#   start_idx  : noeud le plus proche de w0 (ligne de Q utilisée pour la date 0)
#   horizon    : nombre d'années
build_grid <- function(w0, G, horizon, mu_vals, sig_vals, n_w, n_sd = 3) {
  g <- TEC_QL_GBWM_build_grid(w0, G, horizon, mu_vals, sig_vals, n_w, n_sd)
  list(W = g$w_array, ctpm = g$ctpm, start_ctpm = g$start_ctpm, start_idx = g$start_idx,
       horizon = horizon)
}

# Tire le noeud de richesse suivant (l'"environnement", inconnu de l'agent),
# par inversion de la fonction de répartition discrète, comme dans le code du cours.
next_state <- function(grid, idx, t, a) {
  cum_p <- if (t == 1) grid$start_ctpm[, a] else grid$ctpm[idx, , a]
  sum(cum_p < runif(1)) + 1
}

# Évalue `code` avec la graine `seed`, puis rétablit l'état du générateur aléatoire :
# les tirages hors échantillon ne perturbent pas la suite des entraînements.
# seed = NULL : aucun effet (on continue la suite aléatoire courante).
with_seed <- function(seed, code) {
  if (is.null(seed)) return(code)
  old <- if (exists(".Random.seed", envir = globalenv())) get(".Random.seed", envir = globalenv())
  on.exit(if (!is.null(old)) assign(".Random.seed", old, envir = globalenv()))
  set.seed(seed)
  code                                           # évalué ici (évaluation paresseuse de R)
}

# Politique epsilon-greedy : action au hasard avec probabilité epsilon, sinon
# l'action de Q maximal (égalités brisées au hasard). q = Q[idx, t, ].
choose_action <- function(q, epsilon) {
  if (runif(1) < epsilon) return(sample.int(length(q), 1))
  best <- which(q == max(q))
  best[sample.int(length(best), 1)]
}

# Taux d'apprentissage selon le schéma choisi (alpha = valeur initiale) :
#   "constant" : alpha, comme dans l'article
#   "paquets"  : épisodes divisés en 10 paquets ; alpha, alpha / 2, ..., alpha / 10
#   "1/k"      : min(alpha, 1 / k), k = numéro de l'épisode (suggestion de l'énoncé)
#   "1/N"      : min(alpha, 1 / N), N = nombre de visites de la paire (état, action)
learning_rate <- function(schedule, alpha, epoch, epochs, n_visits) {
  switch(schedule,
         "constant" = alpha,
         "paquets"  = alpha / ceiling(10 * epoch / epochs),
         "1/k"      = min(alpha, 1 / epoch),
         "1/N"      = min(alpha, 1 / n_visits))
}

# Apprentissage de la table Q par épisodes simulés.
#   method      : "QL" (cible = max_a' Q(s', a'), off-policy)
#                 ou "SARSA" (cible = Q(s', a') avec a' l'action réellement jouée, on-policy)
#   schedule    : schéma du taux d'apprentissage (voir learning_rate)
#   epsilon_min : epsilon décroît linéairement de epsilon à epsilon_min au fil des
#                 épisodes (epsilon_min = epsilon : exploration constante)
#   q0          : valeur initiale de Q (q0 = 1 : initialisation optimiste)
#   checkpoint_every : si fourni, la politique est évaluée tous les checkpoint_every
#                 épisodes (in-sample et hors échantillon sur n_sim trajectoires,
#                 avec la graine oos_seed : mêmes trajectoires à chaque point de contrôle).
# Retourne list(Q, curve), curve étant NULL sans points de contrôle.
train <- function(grid, G, epochs, epsilon, alpha, method = "QL", schedule = "constant",
                  epsilon_min = epsilon, q0 = 0, checkpoint_every = NULL, n_sim = 2000,
                  oos_seed = NULL) {
  horizon <- grid$horizon
  Q <- array(q0, dim = c(length(grid$W), horizon + 1, dim(grid$ctpm)[3]))
  N <- array(0, dim = dim(Q))                  # visites de chaque (état, action)
  curve <- NULL

  for (epoch in 1:epochs) {
    eps <- epsilon + (epsilon_min - epsilon) * epoch / epochs
    idx <- grid$start_idx
    a <- choose_action(Q[idx, 1, ], eps)

    for (t in 1:horizon) {
      idx1 <- next_state(grid, idx, t, a)
      a1 <- choose_action(Q[idx1, t + 1, ], eps)  # action jouée en s'
      target <- if (method == "SARSA") Q[idx1, t + 1, a1] else max(Q[idx1, t + 1, ])
      N[idx, t, a] <- N[idx, t, a] + 1
      rate <- learning_rate(schedule, alpha, epoch, epochs, N[idx, t, a])
      Q[idx, t, a] <- Q[idx, t, a] + rate * (target - Q[idx, t, a])  # récompense nulle avant T
      idx <- idx1
      a <- a1
    }

    # Date finale : récompense 1 si l'objectif est atteint, 0 sinon (toutes les actions)
    reward <- grid$W[idx] >= G
    N[idx, horizon + 1, ] <- N[idx, horizon + 1, ] + 1
    rate <- learning_rate(schedule, alpha, epoch, epochs, N[idx, horizon + 1, 1])
    Q[idx, horizon + 1, ] <- Q[idx, horizon + 1, ] + rate * (reward - Q[idx, horizon + 1, ])

    if (!is.null(checkpoint_every) && epoch %% checkpoint_every == 0)
      curve <- rbind(curve, data.frame(epoch,
                                       in_sample = max(Q[grid$start_idx, 1, ]),
                                       oos = evaluate_policy(Q, grid, G, n_sim, seed = oos_seed)))
  }
  list(Q = Q, curve = curve)
}

# Taux de succès hors échantillon : Q est figée, on simule n_sim nouvelles
# trajectoires en suivant la politique (epsilon = 0 : politique gloutonne pure).
# seed : graine propre aux tests hors échantillon (différente de celle de
# l'entraînement). Avec la même graine, toutes les politiques sont évaluées sur
# les mêmes nombres aléatoires, ce qui rend leurs écarts plus faciles à comparer.
evaluate_policy <- function(Q, grid, G, n_sim, epsilon = 0, seed = NULL) {
  with_seed(seed, {
    success <- logical(n_sim)
    for (s in 1:n_sim) {
      idx <- grid$start_idx
      for (t in 1:grid$horizon) idx <- next_state(grid, idx, t, choose_action(Q[idx, t, ], epsilon))
      success[s] <- grid$W[idx] >= G
    }
    mean(success)
  })
}

# Tableau des configurations à tester : toutes les combinaisons des valeurs données,
# les autres paramètres prenant leur valeur par défaut. rep = numéro de répétition.
make_cases <- function(method = "QL", epsilon = 0.3, alpha = 0.1, schedule = "constant",
                       epsilon_min = NA, q0 = 0, epochs = 50000L, rep = 1) {
  cases <- expand.grid(method = method, epsilon = epsilon, alpha = alpha, schedule = schedule,
                       epsilon_min = epsilon_min, q0 = q0, epochs = epochs, rep = rep,
                       stringsAsFactors = FALSE)
  no_decay <- is.na(cases$epsilon_min)
  cases$epsilon_min[no_decay] <- cases$epsilon[no_decay]  # par défaut : pas de décroissance
  cases
}

# Entraîne la politique décrite par une ligne de make_cases().
train_case <- function(case, grid, G, checkpoint_every = NULL, n_sim = 2000, oos_seed = NULL) {
  train(grid, G, case$epochs, case$epsilon, case$alpha, case$method, case$schedule,
        case$epsilon_min, case$q0, checkpoint_every, n_sim, oos_seed)
}

# ---- Affichage de la progression ----

# Étape en cours (titre et heure de départ) et barres des étapes terminées, pour la progression.
progress_state <- new.env()
progress_state$done_lines <- character(0)

# Démarre une étape du programme principal ; sa barre s'affiche au premier progress_start.
section <- function(title) {
  progress_state$title <- title
  progress_state$t0 <- Sys.time()
}

# Description d'une configuration, p. ex.
# "Méthode : QL  |  epsilon : 0.30  |  alpha : 0.10 (paquets)  |  Q initial : 0  |  Épisodes : 50K  |  Répétition : 2"
describe_case <- function(case) {
  eps <- if (case$epsilon_min == case$epsilon) sprintf("%.2f", case$epsilon)
         else sprintf("%.2f -> %.2f", case$epsilon, case$epsilon_min)
  sprintf("Méthode : %s  |  epsilon : %s  |  alpha : %.2f (%s)  |  Q initial : %g  |  Épisodes : %gK  |  Répétition : %d",
          case$method, eps, case$alpha, case$schedule, case$q0, case$epochs / 1000, case$rep)
}

# Efface la console et réaffiche les barres des étapes terminées, la barre de l'étape en cours
# (avancement, durée) et, en dessous, la configuration en cours.
# La console de RStudio ne sait pas remonter d'une ligne : on efface donc tout ("\014" = Ctrl+L).
draw_progress <- function(done, n, label) {
  filled <- round(10 * done / n)
  bar <- paste0(strrep("█", filled), strrep("░", 10 - filled))
  secs <- as.numeric(difftime(Sys.time(), progress_state$t0, units = "secs"))
  line <- sprintf("%s [%s] %-5s %6.1fs", format(progress_state$title, width = 22), bar,
                  paste0(done, "/", n), secs)
  if (done == n) progress_state$done_lines <- c(progress_state$done_lines, line)
  cat(if (Sys.getenv("RSTUDIO") == "1") "\014" else "\033[2J\033[H")
  cat(progress_state$done_lines, sep = "\n")
  if (done < n) cat(line, "\n   ", label, "\n", sep = "")
  flush.console()
}

# Affiche la configuration i (sur n) en cours ; retourne l'heure de départ.
progress_start <- function(i, n, case) {
  draw_progress(i - 1, n, describe_case(case))
  Sys.time()
}

# Fin de la configuration i (sur n) : complète la barre si c'est la dernière ; retourne sa durée (secondes).
progress_end <- function(i, n, t0) {
  if (i == n) draw_progress(n, n, "")
  invisible(as.numeric(difftime(Sys.time(), t0, units = "secs")))
}

# ---- Expériences ----

# Entraîne puis évalue une politique pour chaque ligne de `cases` (voir make_cases).
# Ajoute les colonnes :
#   in_sample   : V(W0, 0) = max_a Q(W0, 0, a), la mesure rapportée par l'article.
#                 En SARSA, Q estime la valeur de la politique epsilon-greedy (avec
#                 exploration) : l'in-sample y est donc naturellement plus bas.
#   oos_greedy  : succès hors échantillon sans exploration
#   oos_se      : erreur type de oos_greedy, sqrt(p (1 - p) / n_sim)
#   oos_explore : succès hors échantillon en gardant l'exploration de fin d'entraînement
#   time_s      : temps de calcul (entraînement + évaluations), en secondes
# Les deux évaluations hors échantillon utilisent la graine oos_seed (mêmes
# nombres aléatoires pour toutes les politiques et les deux modes).
run_experiments <- function(cases, grid, G, n_sim, oos_seed = NULL) {
  rows <- lapply(seq_len(nrow(cases)), function(i) {
    case <- cases[i, ]
    t0 <- progress_start(i, nrow(cases), case)
    Q <- train_case(case, grid, G)$Q
    oos <- evaluate_policy(Q, grid, G, n_sim, seed = oos_seed)
    row <- cbind(case,
                 in_sample   = max(Q[grid$start_idx, 1, ]),
                 oos_greedy  = oos,
                 oos_se      = sqrt(oos * (1 - oos) / n_sim),
                 oos_explore = evaluate_policy(Q, grid, G, n_sim, case$epsilon_min, seed = oos_seed))
    row$time_s <- progress_end(i, nrow(cases), t0)
    row
  })
  do.call(rbind, rows)
}

# Moyennes des taux de succès par groupe de configurations (moyenne sur les répétitions).
mean_by <- function(results, by) {
  aggregate(results[c("in_sample", "oos_greedy", "oos_explore", "time_s")], results[by], mean)
}

# Statistiques sur les graines : une ligne par nombre d'épisodes et par mesure.
seed_summary <- function(results) {
  do.call(rbind, lapply(split(results, results$epochs), function(d) {
    values <- list("in-sample" = d$in_sample, "hors éch." = d$oos_greedy)
    data.frame(epochs = d$epochs[1], measure = names(values),
               mean = sapply(values, mean), sd = sapply(values, sd),
               min = sapply(values, min), max = sapply(values, max), time_s = mean(d$time_s))
  }))
}

# Courbes d'apprentissage : entraîne une politique par ligne de `cases` avec
# points de contrôle. Retourne une liste de runs (Q, curve) ; curve$time_s est le
# temps de calcul total du run (entraînement + évaluations aux points de contrôle).
run_learning_curves <- function(cases, grid, G, checkpoint_every, n_sim, oos_seed = NULL) {
  lapply(seq_len(nrow(cases)), function(i) {
    t0 <- progress_start(i, nrow(cases), cases[i, ])
    run <- train_case(cases[i, ], grid, G, checkpoint_every, n_sim, oos_seed)
    run$curve <- cbind(cases[i, ], run$curve, row.names = NULL)
    run$curve$time_s <- progress_end(i, nrow(cases), t0)
    run
  })
}

# Dernier point de contrôle de chaque courbe d'apprentissage (valeurs finales).
final_points <- function(runs) {
  do.call(rbind, lapply(runs, function(run) tail(run$curve, 1)))
}

# ---- Affichage des tableaux ----

# Noms de colonnes lisibles pour les tableaux de résultats.
column_labels <- c(method = "Méthode", epsilon = "epsilon", epsilon_min = "epsilon final",
                   alpha = "alpha", schedule = "Schéma alpha", article = "Article (in-sample)", q0 = "Q initial", epochs = "Épisodes",
                   measure = "Mesure", mean = "Moyenne", sd = "Écart-type", min = "Min", max = "Max",
                   in_sample = "In-sample", oos = "Hors éch.", oos_greedy = "Hors éch.",
                   oos_se = "Erreur type", oos_explore = "Hors éch. (exploration)",
                   time_s = "Temps (s)")

# Affiche un tableau avec titre, colonnes renommées et nombres arrondis à 3 décimales.
print_table <- function(df, title) {
  cat("\n", title, "\n", strrep("-", nchar(title)), "\n", sep = "")
  num <- vapply(df, is.double, logical(1))
  df[num] <- lapply(df[num], round, 3)
  names(df) <- column_labels[names(df)]
  print(df, row.names = FALSE)
}
