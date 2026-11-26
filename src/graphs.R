# Graphiques du rapport : affichés dans l'onglet Plots de RStudio et enregistrés
# en PDF (format vectoriel).
# Couleurs : premières teintes d'une palette catégorielle validée pour les
# daltoniens, et une rampe bleu -> gris -> rouge pour les portefeuilles (prudent -> risqué).

col_series <- c("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # bleu, orange, turquoise, jaune
col_text <- "#52514e"                          # axes et libellés
col_grid <- "#e1e0d9"                          # lignes de grille discrètes
col_dp <- "grey30"                             # ligne de l'étalon DP

# Libellés des schémas du taux d'apprentissage (voir learning_rate dans functions.R)
schedule_labels <- c("constant" = "alpha constant", "paquets" = "alpha par paquets (10)",
                     "1/k" = "alpha = min(alpha, 1/k)", "1/N" = "alpha = min(alpha, 1/N(s,a))")

# Titre d'un run de courbe d'apprentissage, p. ex. "QL, alpha constant".
run_title <- function(run) {
  paste(run$curve$method[1], schedule_labels[run$curve$schedule[1]], sep = ", ")
}

# Ouvre un panneau vide avec grille horizontale et ligne de l'étalon DP.
setup_panel <- function(xlim, ylim, xlab, ylab, main, dp_benchmark, x_axis = TRUE) {
  plot(NA, xlim = xlim, ylim = ylim, xlab = xlab, ylab = ylab, main = main,
       xaxt = if (x_axis) "s" else "n")          # "n" : pas d'axe x automatique
  abline(h = pretty(ylim), col = col_grid, lwd = 0.5)
  abline(h = dp_benchmark, col = col_dp, lty = 3, lwd = 1.5)
}

# Style commun : texte gris, axes en L, libellés horizontaux.
set_style <- function(...) {
  par(las = 1, bty = "l", fg = col_text, col.axis = col_text, col.lab = col_text,
      col.main = col_text, tcl = -0.3, mgp = c(2.6, 0.6, 0), ...)
}

# Trace `draw()` à l'écran (session interactive seulement, p. ex. RStudio), puis
# dans un fichier PDF. Les réglages par() de l'écran sont restaurés ensuite.
show_and_save <- function(file, width, height, draw) {
  if (interactive()) {
    op <- par(no.readonly = TRUE)
    draw()
    par(op)
  }
  pdf(file, width = width, height = height)
  draw()
  invisible(dev.off())
}

# Légende horizontale commune, sous tous les panneaux (marge externe du bas).
bottom_legend <- function(...) {
  par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
  plot.new()
  legend("bottom", ncol = 3, bty = "n", cex = 0.85, text.col = col_text, ...)
}

# Graphique 1 : taux de succès selon le nombre d'épisodes. Un panneau par méthode,
# une couleur par schéma d'apprentissage ; in-sample V(W0, 0) en tirets,
# hors échantillon (politique gloutonne) en trait plein.
plot_learning_curves <- function(runs, dp_benchmark, file) {
  curves <- do.call(rbind, lapply(runs, `[[`, "curve"))
  methods <- unique(curves$method)
  schemes <- unique(curves$schedule)
  ylim <- range(curves$in_sample, curves$oos, dp_benchmark)

  show_and_save(file, 9, 4.5, function() {
    set_style(mfrow = c(1, length(methods)), mar = c(4.2, 4.2, 2.5, 1), oma = c(3, 0, 0, 0))
    for (m in methods) {
      setup_panel(range(curves$epoch) / 1000, ylim, "Épisodes d'entraînement (milliers)", "Taux de succès",
                  m, dp_benchmark)
      for (j in seq_along(schemes)) {
        d <- curves[curves$method == m & curves$schedule == schemes[j], ]
        lines(d$epoch / 1000, d$oos, col = col_series[j], lwd = 2)
        lines(d$epoch / 1000, d$in_sample, col = col_series[j], lwd = 2, lty = 2)
      }
    }
    bottom_legend(legend = c(schedule_labels[schemes], "hors échantillon", "in-sample", "étalon DP"),
                  col = c(col_series[seq_along(schemes)], col_text, col_text, col_dp),
                  lty = c(rep(1, length(schemes)), 1, 2, 3), lwd = 2)
  })
}

# Graphique 2 : succès hors échantillon selon epsilon, moyenne +/- un écart-type
# sur les répétitions. Un panneau par valeur de alpha, une couleur par méthode.
plot_oos_vs_epsilon <- function(results, dp_benchmark, file) {
  stats <- aggregate(oos_greedy ~ method + epsilon + alpha, results,
                     function(x) c(mean = mean(x), sd = sd(x)))
  m <- stats$oos_greedy[, "mean"]
  s <- stats$oos_greedy[, "sd"]
  methods <- unique(stats$method)
  ylim <- range(m - s, m + s, dp_benchmark, na.rm = TRUE)
  dodge <- 0.01 * c(-1, 1)                     # décalage horizontal pour ne pas superposer les méthodes

  show_and_save(file, 9, 4.5, function() {
    set_style(mfrow = c(1, length(unique(stats$alpha))), mar = c(4.2, 4.2, 2.5, 1), oma = c(2, 0, 0, 0))
    for (a in unique(stats$alpha)) {
      setup_panel(range(stats$epsilon) + c(-0.05, 0.05), ylim, expression("Exploration " * epsilon),
                  "Taux de succès hors échantillon", paste("alpha initial =", a), dp_benchmark)
      for (j in seq_along(methods)) {
        k <- stats$alpha == a & stats$method == methods[j]
        x <- stats$epsilon[k] + dodge[j]
        bar <- s[k] > 0                        # pas de barre si l'écart-type est nul
        arrows(x[bar], (m[k] - s[k])[bar], x[bar], (m[k] + s[k])[bar],
               angle = 90, code = 3, length = 0.03, col = col_series[j])
        lines(x, m[k], col = col_series[j], lwd = 1)
        points(x, m[k], col = col_series[j], pch = 19, cex = 1.2)
      }
    }
    bottom_legend(legend = c(methods, "étalon DP"), col = c(col_series[seq_along(methods)], col_dp),
                  lty = c(rep(NA, length(methods)), 3), pch = c(rep(19, length(methods)), NA), lwd = 1.5)
  })
}

# Graphique 3 : carte de la politique apprise (portefeuille choisi selon la date
# et la richesse), un panneau par run. Les cases blanches ont un Q nul : état
# jamais visité, ou objectif jamais atteint depuis cet état (aucune information).
plot_policy_maps <- function(runs, grid, G, file) {
  horizon <- grid$horizon
  K <- dim(grid$ctpm)[3]
  cols <- colorRampPalette(c("#2a78d6", "#d9d6cf", "#e34948"))(K)

  policies <- lapply(runs, function(run) {
    Q <- run$Q[, 1:horizon, ]                  # dates de décision 0 à horizon - 1
    policy <- apply(Q, c(1, 2), which.max)
    policy[apply(Q, c(1, 2), max) == 0] <- NA
    policy
  })
  # Même plage de richesse pour tous les panneaux : celle où au moins un Q est informatif
  informative <- range(which(rowSums(!is.na(do.call(cbind, policies))) > 0))
  rows <- informative[1]:informative[2]

  show_and_save(file, 9, 8, function() {
    layout(cbind(matrix(seq_along(runs), ncol = 2, byrow = TRUE), length(runs) + 1),
           widths = c(1, 1, 0.25))
    set_style(mar = c(4.2, 4.5, 2.5, 1), oma = c(2, 0, 0, 0))
    for (i in seq_along(runs)) {
      image(0:(horizon - 1), grid$W[rows], t(policies[[i]][rows, ]), col = cols, zlim = c(1, K), log = "y",
            xlab = "Date (années)", ylab = "Richesse (échelle log)", main = run_title(runs[[i]]))
      abline(h = G, lty = 2, col = col_text)
      box()
    }
    # Barre de couleurs des portefeuilles
    par(mar = c(4.2, 0.5, 2.5, 3.5))
    image(1, 1:K, t(1:K), col = cols, axes = FALSE, xlab = "", ylab = "", main = "Portef.")
    axis(4, at = c(1, K), labels = c("1\nprudent", paste0(K, "\nrisqué")), padj = 0.5)
    mtext("Blanc : Q nul (aucune information).  Tirets : objectif G.", side = 1, outer = TRUE,
          line = 0.5, cex = 0.8, col = col_text)
  })
}

# Graphique 4 : distribution, sur plusieurs graines, du succès in-sample et hors
# échantillon pour les réglages de l'article. Une boîte par mesure et par nombre
# d'épisodes ; chaque point est une graine. La ligne pointillée est le 0,72
# rapporté par l'article (in-sample) et par le DP.
plot_seed_distribution <- function(results, dp_benchmark, file) {
  # Deux boîtes par nombre d'épisodes : in-sample puis hors échantillon
  groups <- list()
  for (e in sort(unique(results$epochs))) {
    d <- results[results$epochs == e, ]
    groups[[paste0("in-sample\n", e / 1000, "K")]] <- d$in_sample
    groups[[paste0("hors éch.\n", e / 1000, "K")]] <- d$oos_greedy
  }
  cols <- rep(col_series[2:1], length.out = length(groups))   # orange : in-sample, bleu : hors éch.
  ylim <- range(unlist(groups), dp_benchmark)
  at   <- seq_along(groups)
  
  show_and_save(file, 7, 4.5, function() {
    set_style(mar = c(4.8, 4.2, 1, 1))                         # un peu plus de place pour 2 lignes
    setup_panel(c(0.5, length(groups) + 0.5), ylim, "", "Taux de succès", "", dp_benchmark,
                x_axis = FALSE)                             
    boxplot(groups, at = at, add = TRUE, col = adjustcolor(cols, 0.25), border = cols,
            outline = FALSE, axes = FALSE, boxwex = 0.5)
    stripchart(groups, at = at, add = TRUE, vertical = TRUE, method = "jitter", jitter = 0.08,
               pch = 19, cex = 0.7, col = cols)
    axis(1, at = at, labels = names(groups), tick = FALSE, lwd = 0,
         padj = 0.5, mgp = c(3, 1.3, 0))                        # étiquettes tracées une seule fois
  })
}
