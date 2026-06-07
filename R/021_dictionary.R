#- Dictionary curation: evaluate and improve a seed word list -----------------
# Two CONSTRUCT-SIDE tools that run BEFORE vectionary_builder(): they describe and
# improve the seed WORD LIST itself, not whether the learned axis scores documents
# correctly (that needs external labels). Everything is cosine similarity + PCA on
# the seed words read against the full embedding vocabulary.
#
#   dictionary_eval()    a report card: Coherence / Dimensionality / Distinctiveness,
#                        each compared to random word lists of the same size, plus
#                        Separation (AUC) and frequency/length confound checks.
#   dictionary_suggest() words to ADD (raise_coherence / broaden_coverage) and weak
#                        current members to consider removing, judged by the human.
#
# The reference axis used here is parameter-free (difference of pole means, or the
# seed centroid for a unipolar dimension) -- NOT the regularized axis that
# vectionary_builder() learns later; word selection must not use the axis it trains.

#- Internal: metric primitives ------------------------------------------------
# Coherence, unipolar = length of the average seed vector (~ avg pairwise cosine).
.coherence_length <- function(M) sqrt(sum(colMeans(M)^2))

# Coherence, bipolar = pole gap: distance between the pole means in within-pole
# standard deviations (Cohen's d along the axis).
.pole_gap <- function(M, is_pos, is_neg) {
  ax <- colMeans(M[is_pos, , drop = FALSE]) - colMeans(M[is_neg, , drop = FALSE])
  ax <- ax / sqrt(sum(ax^2))
  p  <- as.numeric(M %*% ax)
  n1 <- sum(is_pos); n0 <- sum(is_neg)
  s  <- sqrt(((n1 - 1) * var(p[is_pos]) + (n0 - 1) * var(p[is_neg])) / (n1 + n0 - 2))
  (mean(p[is_pos]) - mean(p[is_neg])) / s
}

# Dimensionality = variance share on PC1, via the d x d covariance eigenvalues
# (same answer as prcomp, no component vectors computed -> several times faster).
.pc1_share <- function(M) {
  M  <- sweep(M, 2, colMeans(M))
  ev <- eigen(crossprod(M), symmetric = TRUE, only.values = TRUE)$values
  ev[1] / sum(ev)
}

# Distinctiveness = how little the axis overlaps the generic direction of language.
.distinctiveness <- function(ax, generic) 1 - abs(sum((ax / sqrt(sum(ax^2))) * generic))

# Plain-language verdict bands (effect sizes, NOT p-values: with a ~400k-word
# vocabulary even a trivial correlation tests as "significant").
.auc_band <- function(a)
  if (a >= 0.8) "strong"      else if (a >= 0.7) "acceptable"  else
  if (a >= 0.6) "weak"        else if (a >= 0.4) "near chance" else "worse than random"

.conf_band <- function(r) {
  if (is.na(r)) return("not available (rows not frequency-sorted)")
  lab <- if (abs(r) < 0.1) "negligible" else if (abs(r) < 0.3) "minor" else "confound"
  sprintf("%s, %.1f%% var", lab, 100 * r^2)
}

#- Internal: input resolution -------------------------------------------------
# Embeddings -> unit-normalized matrix (rownames = vocabulary). Re-normalizing an
# already-unit matrix is a no-op, so it is always safe to call.
.resolve_embeddings <- function(embeddings) {
  if (!is.matrix(embeddings) || is.null(rownames(embeddings)))
    cli_abort(c("{.arg embeddings} must be a matrix with rownames (the vocabulary).",
      "i" = "Load a .vec/.txt file (e.g. with {.fn data.table::fread}) and set the word column as rownames."))
  embeddings / sqrt(rowSums(embeddings^2))
}

# Dictionary -> one-dimension seed table (word, score, stem). Stems ending in "*"
# are expanded against the vocabulary; words kept only if their sign is consistent.
.dict_seed_df <- function(dictionary, dimension, vocab) {
  if (!is.data.frame(dictionary) || !"word" %in% names(dictionary))
    cli_abort("{.arg dictionary} must be a data.frame with a {.field word} column.")
  dim_cols <- setdiff(names(dictionary), "word")
  if (length(dim_cols) == 0)
    cli_abort("{.arg dictionary} needs a numeric dimension column besides {.field word}.")
  if (is.null(dimension)) {
    if (length(dim_cols) > 1)
      cli_abort(c("{.arg dictionary} has several dimensions; choose one with {.arg dimension}.",
        "i" = "Available: {.val {dim_cols}}"))
    dimension <- dim_cols[1]
  } else if (!dimension %in% dim_cols) {
    cli_abort("Dimension {.val {dimension}} not found. Available: {.val {dim_cols}}")
  }

  words  <- as.character(dictionary$word)
  scores <- suppressWarnings(as.numeric(dictionary[[dimension]]))
  ok     <- !is.na(scores)
  words  <- words[ok]; scores <- scores[ok]
  n_requested <- length(unique(words))

  parts <- Map(function(w, s) {
    if (grepl("\\*$", w)) hits <- vocab[startsWith(vocab, sub("\\*$", "", w))]
    else                  hits <- w[w %in% vocab]
    if (length(hits)) data.frame(word = hits, score = s, stem = w, stringsAsFactors = FALSE)
    else NULL
  }, words, scores)
  seed_df <- do.call(rbind, parts)
  if (is.null(seed_df) || !nrow(seed_df))
    cli_abort("None of the dictionary words were found in the embedding vocabulary.")

  sign_by_word <- tapply(seed_df$score, seed_df$word,
                         function(x) if (length(unique(sign(x))) == 1) x[1] else NA)
  keep    <- names(sign_by_word)[!is.na(sign_by_word)]
  seed_df <- seed_df[seed_df$word %in% keep & !duplicated(seed_df$word), ]

  attr(seed_df, "dimension")     <- dimension
  attr(seed_df, "n_total_stems") <- n_requested
  seed_df
}

#- dictionary_eval(): the report card -----------------------------------------

#' Evaluate a seed dictionary (construct-side report card)
#'
#' @description
#' Describes a seed word list as a measurement axis, using only cosine similarity
#' and PCA read against the full embedding vocabulary. It reports three numbers --
#' Coherence, Dimensionality, and Distinctiveness -- each benchmarked against random
#' word lists of the same size, plus a Separation (AUC) check and frequency/length
#' confound checks.
#'
#' This is a \strong{construct-side} diagnostic: it judges the WORD LIST (is it tight,
#' one-dimensional, distinctive, separable, unconfounded?), not whether the eventual
#' axis scores documents correctly. Run it \emph{before} \code{\link{vectionary_builder}}
#' to curate the seeds.
#'
#' @param dictionary A data.frame with a \code{word} column and one or more numeric
#'   dimension columns (the same format \code{\link{vectionary_builder}} accepts).
#'   The \strong{sign} of the scores sets the poles: a mix of positive and negative
#'   scores is treated as bipolar; a single sign is unipolar. Words ending in
#'   \code{"*"} are treated as stems and expanded against the vocabulary.
#' @param embeddings A numeric matrix with one row per word and \code{rownames} equal
#'   to the vocabulary. Rows are unit-normalized internally, so dot products are
#'   cosine similarities.
#' @param dimension Name of the dimension column to evaluate. Required only when the
#'   dictionary has more than one dimension column.
#' @param n_random Number of random word lists used as the chance-level baseline
#'   (default 500). Larger is steadier but slower.
#' @param seed Random seed for the baseline draws (default 574), for reproducibility.
#'
#' @return An object of class \code{"dictionary_eval"}: a list with \code{card},
#'   \code{checks}, and \code{influential} data frames plus metadata. It prints a
#'   formatted report card.
#'
#' @seealso \code{\link{dictionary_suggest}} to propose words to add or remove,
#'   \code{\link{vectionary_builder}} to learn the axis once the list is curated.
#'
#' @examples
#' \dontrun{
#' # embeddings: matrix, rownames = words (e.g. read from a GloVe/FastText file)
#' dict <- data.frame(
#'   word      = c("happy", "joy", "delight*", "sad", "grief", "misery"),
#'   sentiment = c(1, 1, 1, -1, -1, -1)
#' )
#' rep <- dictionary_eval(dict, embeddings)
#' rep                       # prints the report card
#' }
#'
#' @importFrom stats var
#' @export
dictionary_eval <- function(dictionary, embeddings, dimension = NULL,
                            n_random = 500, seed = 574) {
  E         <- .resolve_embeddings(embeddings)
  vocab     <- rownames(E)
  seed_df   <- .dict_seed_df(dictionary, dimension, vocab)
  dimension <- attr(seed_df, "dimension")

  W       <- E[seed_df$word, , drop = FALSE]
  pos     <- seed_df$score > 0
  neg     <- seed_df$score < 0
  bipolar <- any(pos) & any(neg)
  if (bipolar && (sum(pos) < 2 || sum(neg) < 2))
    cli_abort(c("A bipolar dimension needs >= 2 seed words on each pole.",
      "i" = "Found {sum(pos)} on the high pole and {sum(neg)} on the low pole."))
  if (!bipolar && nrow(W) < 2)
    cli_abort("A dimension needs at least 2 seed words.")

  generic <- colMeans(E); generic <- generic / sqrt(sum(generic^2))

  # reference axis: pole-mean difference (bipolar) or seed centroid (unipolar)
  a0 <- if (bipolar) colMeans(W[pos, , drop = FALSE]) - colMeans(W[neg, , drop = FALSE]) else colMeans(W)
  a0 <- a0 / sqrt(sum(a0^2))

  coh  <- if (bipolar) .pole_gap(W, pos, neg) else .coherence_length(W)
  dimn <- .pc1_share(W)
  dist <- .distinctiveness(a0, generic)

  # compare to random word lists of the same size
  set.seed(seed)
  n1 <- sum(pos); n0 <- sum(neg); ntot <- nrow(W)
  rand <- replicate(n_random, {
    Wr <- E[sample(nrow(E), ntot), , drop = FALSE]
    if (bipolar) {
      rp <- c(rep(TRUE, n1), rep(FALSE, n0))
      ar <- colMeans(Wr[rp, , drop = FALSE]) - colMeans(Wr[!rp, , drop = FALSE])
      c(coh = .pole_gap(Wr, rp, !rp), dim = .pc1_share(Wr), dist = .distinctiveness(ar, generic))
    } else {
      c(coh = .coherence_length(Wr), dim = .pc1_share(Wr), dist = .distinctiveness(colMeans(Wr), generic))
    }
  })
  pct <- function(x, draws) mean(x > draws)

  # stability: drop each stem in turn
  stems <- unique(seed_df$stem)
  loo <- sapply(stems, function(s) {
    k  <- seed_df$stem != s
    Wk <- W[k, , drop = FALSE]; pk <- pos[k]; nk <- neg[k]
    if (bipolar && (sum(pk) < 2 || sum(nk) < 2)) return(c(NA, NA, NA))
    ak <- if (bipolar) colMeans(Wk[pk, , drop = FALSE]) - colMeans(Wk[nk, , drop = FALSE]) else colMeans(Wk)
    c(if (bipolar) .pole_gap(Wk, pk, nk) else .coherence_length(Wk),
      .pc1_share(Wk), .distinctiveness(ak, generic))
  })

  # the two checks
  proj_all <- as.numeric(E %*% a0)
  is_seed  <- vocab %in% (if (bipolar) seed_df$word[pos] else seed_df$word)
  rk       <- rank(proj_all)
  auc      <- (sum(rk[is_seed]) - sum(is_seed) * (sum(is_seed) + 1) / 2) /
              (sum(is_seed) * sum(!is_seed))
  # frequency confound assumes rows are frequency-sorted (as GloVe is); if the
  # vocabulary is alphabetical the file carries no frequency info -> report NA.
  cor_freq <- if (is.unsorted(vocab)) cor(proj_all, seq_len(nrow(E)), method = "spearman") else NA
  cor_len  <- cor(proj_all, nchar(vocab), method = "spearman")

  card <- data.frame(
    metric = c(if (bipolar) "Coherence (pole gap)" else "Coherence (mean length)",
               "Dimensionality (PC1 share)", "Distinctiveness"),
    value  = round(c(coh, dimn, dist), 3),
    random = round(c(mean(rand["coh", ]), mean(rand["dim", ]), mean(rand["dist", ])), 3),
    beats_random = round(c(pct(coh, rand["coh", ]), pct(dimn, rand["dim", ]),
                           pct(dist, rand["dist", ])), 3),
    loo_range = round(c(diff(range(loo[1, ], na.rm = TRUE)),
                        diff(range(loo[2, ], na.rm = TRUE)),
                        diff(range(loo[3, ], na.rm = TRUE))), 3),
    stringsAsFactors = FALSE)
  checks <- data.frame(
    check   = c("Separation (AUC)", "Confound: frequency", "Confound: length"),
    value   = round(c(auc, cor_freq, cor_len), 3),
    verdict = c(.auc_band(auc), .conf_band(cor_freq), .conf_band(cor_len)),
    stringsAsFactors = FALSE)
  full <- c(coh, dimn, dist)
  influential <- data.frame(
    metric = card$metric,
    stems  = vapply(1:3, function(m) paste(
      head(stems[order(abs(loo[m, ] - full[m]), decreasing = TRUE)], 3), collapse = ", "),
      character(1)),
    stringsAsFactors = FALSE)

  structure(list(
    card = card, checks = checks, influential = influential,
    dimension = dimension, bipolar = bipolar, n_random = n_random,
    n_seed = nrow(seed_df), n_pos = sum(pos), n_neg = sum(neg),
    n_stems_matched = length(stems), n_stems_total = attr(seed_df, "n_total_stems")),
    class = "dictionary_eval")
}

#' @rdname dictionary_eval
#' @param x A \code{dictionary_eval} object.
#' @param ... Unused.
#' @export
print.dictionary_eval <- function(x, ...) {
  cli_h1("Report card: {x$dimension}")
  cli_alert_info(
    "{x$n_seed} seed words ({x$n_pos} high, {x$n_neg} low) from {x$n_stems_matched} of {x$n_stems_total} stems matched")

  cli_h2("Metrics")
  print(x$card, row.names = FALSE)
  cli_text("")
  cli_ul(c(
    "{.strong Coherence}: how tightly the seeds pin down a single direction",
    "{.strong Dimensionality}: variance share on PC1 (high = one direction)",
    "{.strong Distinctiveness}: distance from generic language (1 = unrelated)"))
  cli_ul(c(
    "{.field value}: this dictionary's score on the metric",
    "{.field random}: the same score for random word lists of equal size (the chance level)",
    "{.field beats_random}: fraction of those {x$n_random} random lists it scores higher than",
    "{.field loo_range}: score movement when any one stem is dropped (small = stable)"))
  cli_text("")
  cli_text("Most load-bearing stems (largest swing when dropped):")
  cli_ul(sprintf("{.strong %s}: %s", x$influential$metric, x$influential$stems))

  cli_h2("Checks")
  lvl <- function(v)
    if (grepl("not available", v))                "na"    else
    if (grepl("strong|acceptable|negligible", v)) "ok"    else
    if (grepl("weak|near chance|minor",       v)) "watch" else "bad"
  for (i in seq_len(nrow(x$checks))) {
    r   <- x$checks[i, ]
    msg <- sprintf("%s = %s  (%s)", r$check, format(r$value, nsmall = 3), r$verdict)
    switch(lvl(r$verdict),
      ok    = cli_alert_success("{msg}"),
      watch = cli_alert_warning("{msg}"),
      na    = cli_alert_info("{msg}"),
      bad   = cli_alert_danger("{msg}"))
  }
  cli_text("")
  cli_text("{.emph Separation (AUC)}: <0.4 worse than random | 0.4-0.6 near chance | 0.6-0.7 weak | 0.7-0.8 acceptable | >=0.8 strong")
  cli_text("{.emph Confound (|rho|)}: <0.1 negligible | 0.1-0.3 minor | >0.3 confound (% = score-variance explained)")
  invisible(x)
}

#- dictionary_suggest(): words to add / remove --------------------------------

# One pole's add-suggestions (the body shared by both poles).
.suggest_one <- function(pole_words, opp_words, E, n, raise_coherence,
                         broaden_coverage, pool, rarity_words, spellcheck) {
  vocab <- rownames(E)
  if (length(pole_words) < 2)
    cli_abort("Each pole needs at least 2 words (got {length(pole_words)}).")
  cpole <- colMeans(E[pole_words, , drop = FALSE])
  copp  <- if (is.null(opp_words)) colMeans(E) else colMeans(E[opp_words, , drop = FALSE])
  axis  <- cpole - copp; axis <- axis / sqrt(sum(axis^2))      # difference of means (unit)
  pr    <- as.numeric(E %*% axis)                              # projection on the axis
  thr   <- median(pr[match(pole_words, vocab)])                # as far out as a typical seed

  known <- c(pole_words, opp_words)
  ok    <- grepl("^[a-z]+$", vocab) & nchar(vocab) >= 3 & !(vocab %in% known) & pr >= thr
  if (!is.null(rarity_words)) {
    if (!is.unsorted(vocab))
      cli_abort(c("Embedding rows are alphabetical, so {.arg rarity_words} has no frequency to use.",
        "i" = "Set {.code rarity_words = NULL} to skip the frequency filter."))
    ok <- ok & seq_along(vocab) <= round(rarity_words * nrow(E))
  }
  if (spellcheck) {
    ci <- which(ok); good <- hunspell::hunspell_check(vocab[ci]); ok[ci[!good]] <- FALSE
  }
  ord  <- order(pr, decreasing = TRUE); ord <- ord[ok[ord]]
  near <- function(i) pole_words[which.max(as.numeric(E[pole_words, , drop = FALSE] %*% E[i, ]))]
  res  <- list()

  if (raise_coherence) {
    idx <- head(ord, n)
    res$coherence <- data.frame(mode = "coherence", word = vocab[idx],
      axis_proj = round(pr[idx], 3), nearest_seed = vapply(idx, near, ""),
      stringsAsFactors = FALSE)
  }
  if (broaden_coverage && length(ord)) {
    poolidx <- head(ord, pool)
    maxsim  <- apply(E[poolidx, , drop = FALSE] %*% t(E[pole_words, , drop = FALSE]), 1, max)
    picks   <- integer(0)
    for (j in seq_len(min(n, length(poolidx)))) {     # greedy farthest-point: most novel each time
      k <- which.min(maxsim); picks <- c(picks, poolidx[k])
      maxsim <- pmax(maxsim, as.numeric(E[poolidx, , drop = FALSE] %*% E[poolidx[k], ]))
      maxsim[k] <- Inf
    }
    res$coverage <- data.frame(mode = "coverage", word = vocab[picks],
      axis_proj = round(pr[picks], 3), nearest_seed = vapply(picks, near, ""),
      stringsAsFactors = FALSE)
  }
  if (!length(res))
    return(data.frame(mode = character(), word = character(),
      axis_proj = numeric(), nearest_seed = character(), stringsAsFactors = FALSE))
  res <- lapply(res, function(d) d[order(-d$axis_proj), ])
  do.call(rbind, res)
}

# One pole's weak current members + each one's nearest non-seed neighbours.
.weak_one <- function(pole_words, opp_words, E, n_weak, n_near) {
  vocab <- rownames(E)
  cp <- colMeans(E[pole_words, , drop = FALSE]); cp <- cp / sqrt(sum(cp^2))
  co <- if (is.null(opp_words)) colMeans(E) else colMeans(E[opp_words, , drop = FALSE])
  co <- co / sqrt(sum(co^2))
  lean <- as.numeric(E[pole_words, , drop = FALSE] %*% cp) -
          as.numeric(E[pole_words, , drop = FALSE] %*% co)
  idx  <- order(lean)[seq_len(min(n_weak, length(pole_words)))]
  ok   <- grepl("^[a-z]+$", vocab) & nchar(vocab) >= 3 & !(vocab %in% c(pole_words, opp_words))
  near <- function(w) {
    sims <- as.numeric(E %*% E[w, ]); sims[!ok] <- -Inf
    paste(vocab[order(sims, decreasing = TRUE)[seq_len(n_near)]], collapse = ", ")
  }
  data.frame(word = pole_words[idx], lean = round(lean[idx], 3),
             neighbors = vapply(pole_words[idx], near, ""), stringsAsFactors = FALSE)
}

#' Suggest words to add to or remove from a seed dictionary
#'
#' @description
#' Proposes an annotated diff for a seed word list, judged by the human:
#' \itemize{
#'   \item \strong{words to add}, two ways -- \code{raise_coherence} (the strongest
#'     words on the pole, which sharpen the axis) and \code{broaden_coverage}
#'     (on-theme words in new directions, picked greedily farthest-first, which widen it);
#'   \item \strong{weak current members} that lean toward the other side, each shown
#'     with its nearest non-seed neighbours so a member worth keeping can be explored.
#' }
#' For a bipolar dimension both poles are handled (the contrast is the opposite pole);
#' for a unipolar dimension the contrast is the vocabulary average. Selection uses the
#' parameter-free difference-of-means axis, never a learned/regularized axis. It never
#' edits the dictionary.
#'
#' @inheritParams dictionary_eval
#' @param n Number of words to suggest per pole and per mode (default 20).
#' @param raise_coherence If \code{TRUE} (default), suggest the strongest on-pole words.
#' @param broaden_coverage If \code{TRUE} (default), suggest on-theme words in new
#'   directions (greedy farthest-point).
#' @param pool Size of the candidate pool the coverage search draws from (default 1500).
#' @param rarity_words Keep only the most frequent fraction of the vocabulary as
#'   candidates (e.g. \code{0.1} = top 10\%; lower is stricter). \code{NULL} skips the
#'   frequency filter. Assumes the embedding rows are frequency-sorted (as GloVe is);
#'   errors if the vocabulary is alphabetical.
#' @param spellcheck If \code{TRUE} (default), drop non-words via \code{hunspell}
#'   (the engine \code{\link{vectionary_builder}} uses). Requires the \code{hunspell}
#'   package.
#' @param n_weak Number of weak current members to list per pole (default 15).
#' @param n_near Number of nearest non-seed neighbours to show per weak member
#'   (default 5).
#'
#' @return An object of class \code{"dictionary_suggest"}: a list with an \code{add}
#'   data frame (columns \code{pole}, \code{mode}, \code{word}, \code{axis_proj},
#'   \code{nearest_seed}) and a \code{weak} data frame (columns \code{pole},
#'   \code{word}, \code{lean}, \code{neighbors}). It prints a formatted diff.
#'
#' @seealso \code{\link{dictionary_eval}} for the report card.
#'
#' @examples
#' \dontrun{
#' dict <- data.frame(
#'   word      = c("happy", "joy", "sad", "grief"),
#'   sentiment = c(1, 1, -1, -1)
#' )
#' dictionary_suggest(dict, embeddings)                 # both poles
#' dictionary_suggest(dict, embeddings, rarity_words = NULL, spellcheck = FALSE)
#' }
#'
#' @export
dictionary_suggest <- function(dictionary, embeddings, dimension = NULL, n = 20,
                               raise_coherence = TRUE, broaden_coverage = TRUE,
                               pool = 1500, rarity_words = 0.1, spellcheck = TRUE,
                               n_weak = 15, n_near = 5) {
  if (spellcheck && !requireNamespace("hunspell", quietly = TRUE))
    cli_abort(c("{.pkg hunspell} is required for {.code spellcheck = TRUE}.",
      "i" = "Install it, or set {.code spellcheck = FALSE}."))
  E         <- .resolve_embeddings(embeddings)
  vocab     <- rownames(E)
  seed_df   <- .dict_seed_df(dictionary, dimension, vocab)
  dimension <- attr(seed_df, "dimension")

  pos_words <- seed_df$word[seed_df$score > 0]
  neg_words <- seed_df$word[seed_df$score < 0]
  bipolar   <- length(pos_words) > 0 && length(neg_words) > 0
  poles <- if (bipolar)
    list(list(label = "high (+)", pole = pos_words, opp = neg_words),
         list(label = "low (-)",  pole = neg_words, opp = pos_words))
  else
    list(list(label = "seeds", pole = seed_df$word, opp = NULL))

  add <- list(); weak <- list()
  for (p in poles) {
    a <- .suggest_one(p$pole, p$opp, E, n, raise_coherence, broaden_coverage,
                      pool, rarity_words, spellcheck)
    w <- .weak_one(p$pole, p$opp, E, n_weak, n_near)
    if (nrow(a)) add[[p$label]]  <- cbind(pole = p$label, a, stringsAsFactors = FALSE)
    if (nrow(w)) weak[[p$label]] <- cbind(pole = p$label, w, stringsAsFactors = FALSE)
  }
  structure(list(
    add = do.call(rbind, add), weak = do.call(rbind, weak),
    dimension = dimension, bipolar = bipolar),
    class = "dictionary_suggest")
}

#' @rdname dictionary_suggest
#' @param x A \code{dictionary_suggest} object.
#' @param ... Unused.
#' @export
print.dictionary_suggest <- function(x, ...) {
  cli_h1("Suggestions: {x$dimension}")
  titles <- c(coherence = "Best matches to add (sharpen the axis)",
              coverage  = "Broaden coverage (on-theme, new directions)")
  poles <- unique(c(x$add$pole, x$weak$pole))
  for (pl in poles) {
    cli_h2("Pole: {pl}")
    sub <- x$add[x$add$pole == pl, ]
    for (m in intersect(names(titles), unique(sub$mode))) {
      cli_text("{.emph {titles[[m]]}}")
      print(sub[sub$mode == m, c("word", "axis_proj", "nearest_seed")], row.names = FALSE)
    }
    w <- x$weak[x$weak$pole == pl, ]
    if (nrow(w)) {
      cli_text("{.emph Weak current members (consider removing, or explore neighbours)}")
      cli_ul(sprintf("{.strong %s} (lean %s): %s", w$word, format(w$lean, nsmall = 3), w$neighbors))
    }
  }
  invisible(x)
}
