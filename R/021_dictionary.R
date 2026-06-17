#- Dictionary curation: evaluate and improve a seed word list -----------------
# Two CONSTRUCT-SIDE tools that run BEFORE vectionary_builder(): they describe and
# improve the seed WORD LIST itself, not whether the learned axis scores documents
# correctly (that needs external labels). Everything is cosine similarity + PCA on
# the seed words read against the full embedding vocabulary.
#
#   dictionary_eval()    a report card: Coherence and Dimensionality (each pole scored
#                        as a unipolar list, compared to random word lists of the same
#                        size; a bipolar dict is two cards plus a pole-gap contrast),
#                        plus a Separation (leave-one-stem-out AUC) check and
#                        frequency / length / generic-language confounds.
#   dictionary_suggest() words to ADD (raise_coherence / broaden_coverage) and weak
#                        current members to consider removing, judged by the human.
#
# The reference axis used here is parameter-free (difference of pole means, or the
# seed centroid for a unipolar dimension) -- NOT the regularized axis that
# vectionary_builder() learns later; word selection must not use the axis it trains.

#- Internal: metric primitives ------------------------------------------------
# Each pole of a dictionary is scored as a UNIPOLAR list (the primitive); a bipolar
# dictionary is two such cards plus a contrast (the pole gap). All numbers are read on
# unit-normalized vectors, so a dot product is a cosine similarity.

# Coherence = length of the average seed vector (~ avg pairwise cosine), in [0, 1].
.coherence_length <- function(M) sqrt(sum(colMeans(M)^2))

# Centered covariance eigenvalues of a seed cloud (variances along its principal
# directions); shared by the two dimensionality readings below. No eigenvectors, so
# this is the same answer as prcomp but several times faster.
.cloud_var <- function(M) {
  M  <- sweep(M, 2, colMeans(M))
  ev <- eigen(crossprod(M), symmetric = TRUE, only.values = TRUE)$values
  ev[ev > 0]
}

# Dimensionality, headline = variance share on PC1 (1 = one direction / one concept).
.pc1_share <- function(M) { ev <- .cloud_var(M); if (!length(ev)) NA_real_ else ev[1] / sum(ev) }

# Dimensionality, companion = how many principal directions to reach `thresh` of the
# spread ("looks like about k concepts"). 1 = tightly one-dimensional.
.pcs_to <- function(M, thresh = 0.8) {
  ev <- .cloud_var(M)
  if (!length(ev)) NA_integer_ else which(cumsum(ev) / sum(ev) >= thresh)[1]
}

# Pole gap (the bipolar CONTRAST, not a headline metric) = distance between the pole
# means in within-pole SDs (Cohen's d along the difference axis): how far apart the
# two poles sit.
.pole_gap <- function(M, is_pos, is_neg) {
  ax <- colMeans(M[is_pos, , drop = FALSE]) - colMeans(M[is_neg, , drop = FALSE])
  ax <- ax / sqrt(sum(ax^2))
  p  <- as.numeric(M %*% ax)
  n1 <- sum(is_pos); n0 <- sum(is_neg)
  s  <- sqrt(((n1 - 1) * var(p[is_pos]) + (n0 - 1) * var(p[is_neg])) / (n1 + n0 - 2))
  (mean(p[is_pos]) - mean(p[is_neg])) / s
}

# Generic alignment (a CONFOUND, not a metric) = |cos| between the operative axis and
# the generic direction of language (the vocabulary mean). High = the axis leans into
# everyday/frequent language, so it is likely tracking frequency rather than meaning.
# (This is the old "distinctiveness", 1 - |cos|, re-read as a confound.)
.generic_align <- function(ax, generic) abs(sum((ax / sqrt(sum(ax^2))) * generic))

# Separation = one-sided AUC: P(a random "hi" word outranks a random "lo" word),
# the Wilcoxon-Mann-Whitney statistic. NA when either group is empty.
.auc_one <- function(proj, hi, lo) {
  if (sum(hi) < 1 || sum(lo) < 1) return(NA_real_)
  rk <- rank(c(proj[hi], proj[lo])); n_hi <- sum(hi)
  (sum(rk[seq_len(n_hi)]) - n_hi * (n_hi + 1) / 2) / (n_hi * sum(lo))
}

# Leave-one-stem-out Separation AUC for one pole. For each stem, rebuild the reference
# axis from the OTHER seeds (the pole centroid, minus the opposite pole's centroid when
# bipolar) pointed toward this pole, project the held-out seeds and the background, and
# ask P(held-out seed > background). Averaging over stems removes the circularity of an
# in-sample AUC -- the axis never sees the word it is scored on. `opp_centroid` is NULL
# for a unipolar list (the axis is just the pole centroid).
.loo_auc <- function(words, stems, opp_centroid, E, is_bg) {
  vocab <- rownames(E)
  Wp    <- E[words, , drop = FALSE]
  ust   <- unique(stems)
  aucs  <- vapply(ust, function(s) {
    keep <- stems != s
    if (!any(keep)) return(NA_real_)
    cen <- colMeans(Wp[keep, , drop = FALSE])
    ax  <- if (is.null(opp_centroid)) cen else cen - opp_centroid
    ax  <- ax / sqrt(sum(ax^2))
    proj <- as.numeric(E %*% ax)
    held <- vocab %in% words[!keep]          # the held-out pole seeds
    .auc_one(proj, held, is_bg)
  }, numeric(1))
  mean(aucs, na.rm = TRUE)
}

# Plain-language verdict bands (effect sizes, NOT p-values: with a ~400k-word
# vocabulary even a trivial correlation tests as "significant").
.auc_band <- function(a)
  if (is.na(a))     "not available"               else
  if (a >= 0.8)     "strong"      else if (a >= 0.7) "acceptable"  else
  if (a >= 0.6)     "weak"        else if (a >= 0.4) "near chance" else "worse than random"

.conf_band <- function(r) {
  if (is.na(r)) return("not available -- rows not known to be frequency-sorted")
  lab <- if (abs(r) < 0.1) "negligible" else if (abs(r) < 0.3) "minor" else "confound"
  sprintf("%s, %.1f%% var", lab, 100 * r^2)
}

.disc_band <- function(x) {
  if (is.na(x)) return("not computed")
  if (x < 0.2) "distinct" else if (x < 0.4) "some overlap" else
  if (x < 0.6) "strong overlap" else "near-duplicate"
}

# One pole's intrinsic geometry: coherence + dimensionality, each with its beats-random
# percentile and leave-one-stem-out range, plus the load-bearing stems for each.
.pole_metrics <- function(words, stems, E, n_random, seed) {
  W   <- E[words, , drop = FALSE]
  coh <- .coherence_length(W)
  pc1 <- .pc1_share(W)
  pcs <- .pcs_to(W, 0.8)
  n   <- length(words)

  set.seed(seed)
  rand <- replicate(n_random, {
    Wr <- E[sample(nrow(E), n), , drop = FALSE]
    c(coh = .coherence_length(Wr), dim = .pc1_share(Wr))
  })

  ust <- unique(stems)
  if (length(ust) >= 2) {
    loo <- vapply(ust, function(s) {
      Wk <- W[stems != s, , drop = FALSE]
      c(.coherence_length(Wk), .pc1_share(Wk))
    }, numeric(2))
    top3     <- function(row, val)
      paste(head(ust[order(abs(loo[row, ] - val), decreasing = TRUE)], 3), collapse = ", ")
    coh_loo  <- diff(range(loo[1, ], na.rm = TRUE)); dim_loo <- diff(range(loo[2, ], na.rm = TRUE))
    coh_infl <- top3(1, coh); dim_infl <- top3(2, pc1)
  } else {
    coh_loo  <- dim_loo <- NA_real_
    coh_infl <- dim_infl <- if (length(ust)) ust[1] else ""
  }

  list(
    coherence = coh, coh_beats = mean(coh > rand["coh", ], na.rm = TRUE),
    coh_loo = coh_loo, coh_infl = coh_infl,
    pc1 = pc1, pcs80 = pcs, dim_beats = mean(pc1 > rand["dim", ], na.rm = TRUE),
    dim_loo = dim_loo, dim_infl = dim_infl,
    n = n, n_stems = length(ust))
}

# Build a unit reference axis from a competing dictionary (or accept a ready axis
# vector) for the optional discriminant-validity check.
.reference_axis <- function(against, E, vocab) {
  if (is.numeric(against) && is.null(dim(against))) {
    if (length(against) != ncol(E))
      cli_abort("An {.arg against} axis vector must have length {ncol(E)} (the embedding dimension).")
    return(against / sqrt(sum(against^2)))
  }
  sd <- .dict_seed_df(against, NULL, vocab)
  p  <- sd$score > 0; ng <- sd$score < 0
  W  <- E[sd$word, , drop = FALSE]
  ax <- if (any(p) && any(ng)) colMeans(W[p, , drop = FALSE]) - colMeans(W[ng, , drop = FALSE]) else colMeans(W)
  ax / sqrt(sum(ax^2))
}

#- Internal: input resolution -------------------------------------------------
# Embeddings -> unit-normalized matrix (rownames = vocabulary). Re-normalizing an
# already-unit matrix is a no-op, so it is always safe to call.
.resolve_embeddings <- function(embeddings) {
  # A file path -> read it the way the builder does: detect/skip a FastText header
  # with .parse_embeddings_header() (R/03_builder.R), then fread only the n_dims value
  # columns, so a trailing space cannot leak an extra all-NA column.
  if (is.character(embeddings) && length(embeddings) == 1L) {
    if (!file.exists(embeddings))
      cli_abort("Embeddings file not found: {.path {embeddings}}")
    if (!requireNamespace("data.table", quietly = TRUE))
      cli_abort(c("{.pkg data.table} is required to read an embeddings file path.",
        "i" = "Install it, or pass a pre-loaded matrix to {.arg embeddings}."))
    header <- .parse_embeddings_header(readLines(embeddings, n = 1, warn = FALSE))
    dt <- data.table::fread(
      embeddings, header = FALSE, skip = if (header$is_header) 1L else 0L,
      sep = " ", quote = "", colClasses = c("character", rep("numeric", header$n_dims)),
      showProgress = FALSE, data.table = TRUE)
    embeddings <- as.matrix(dt[, 2:(header$n_dims + 1), with = FALSE])
    rownames(embeddings) <- dt[[1]]
  }
  if (!is.matrix(embeddings) || is.null(rownames(embeddings)))
    cli_abort(c("{.arg embeddings} must be a matrix with rownames (the vocabulary), or a path to a FastText {.file .vec} / GloVe or word2vec {.file .txt} file.",
      "i" = "Pass the file path directly, or load it (e.g. with {.fn data.table::fread}) and set the word column as rownames."))
  # fread coerces tokens matching data.table's na.strings (e.g. the literal "NA") to NA
  # in the word column; drop any NA-word row -- it cannot match a seed, and such tokens
  # are stopwords dropped downstream anyway.
  na_word <- is.na(rownames(embeddings))
  if (any(na_word)) embeddings <- embeddings[!na_word, , drop = FALSE]
  embeddings / sqrt(rowSums(embeddings^2))
}

# Dictionary -> one-dimension seed table (word, score, stem). Stems ending in "*"
# are expanded against the vocabulary; words kept only if their sign is consistent.
# Normalize a dictionary argument to a data.frame with a `word` column and >= 1
# dimension column -- the shared front door for dictionary_eval(), dictionary_suggest(),
# and vectionary_builder(), so every entry reads the same inputs. A character vector (or
# factor), or a data.frame with only a `word` column, is a BINARY list: each word scores
# 1 on every requested dimension (default "score"). A graded data.frame passes through.
.as_dictionary_df <- function(dictionary, dim_names = "score") {
  if (is.character(dictionary) || is.factor(dictionary)) {
    if (length(dictionary) == 0) cli_abort("{.arg dictionary} cannot be empty.")
    dictionary <- data.frame(word = as.character(dictionary), stringsAsFactors = FALSE)
    for (nm in dim_names) dictionary[[nm]] <- 1
    return(dictionary)
  }
  if (!is.data.frame(dictionary) || !"word" %in% names(dictionary))
    cli_abort(c("{.arg dictionary} must be a data.frame with a {.field word} column, or a character vector of words.",
      "i" = "A character vector (or a {.field word}-only data.frame) is read as a binary list; add numeric column(s) for a graded dictionary."))
  if (length(setdiff(names(dictionary), "word")) == 0)   # word-only -> binary list
    for (nm in dim_names) dictionary[[nm]] <- 1
  dictionary
}

.dict_seed_df <- function(dictionary, dimension, vocab) {
  # Accept a character vector / word-only data.frame (binary) or a graded data.frame --
  # the same front door vectionary_builder() uses.
  dictionary <- .as_dictionary_df(dictionary)
  dim_cols <- setdiff(names(dictionary), "word")
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
#' Describes a seed word list as a measurement axis, using only cosine similarity and
#' PCA read against the full embedding vocabulary. Each pole is scored as a unipolar
#' list -- the primitive -- on two headline metrics, \strong{Coherence} and
#' \strong{Dimensionality}, each benchmarked against random word lists of the same size.
#' A \strong{bipolar} dictionary is reported as two such cards plus a \strong{contrast}
#' (the pole gap, a Cohen's d). The checks are a leave-one-stem-out \strong{Separation
#' (AUC)} and frequency / length / generic-language confounds, with an optional
#' discriminant-validity check against a competing axis.
#'
#' This is a \strong{construct-side} diagnostic: it judges the WORD LIST, not whether
#' the eventual axis scores documents correctly. Run it \emph{before}
#' \code{\link{vectionary_builder}} to curate the seeds.
#'
#' @details
#' \strong{Why two cards for a bipolar dictionary.} A bipolar list is two unipolar
#' concepts plus a contrast between them, so a single combined card mixes "is each pole
#' tight?" with "are the poles far apart?". Scoring each pole on its own centroid keeps
#' every metric on the unipolar \eqn{[0,1]} scale and asks the facet question of each
#' pole, while the pole gap (Cohen's d along the difference axis) reports the separation
#' as its own number.
#'
#' \strong{Separation is leave-one-stem-out.} The reference axis is built \emph{from} the
#' seeds, so an in-sample AUC reads optimistically. Instead each stem is held out in
#' turn, the axis is rebuilt from the rest, and the held-out seeds are ranked against a
#' seed-excluded background; the reported AUC averages those folds. For a bipolar
#' dimension each pole is tested toward its own end and the \emph{weaker} pole governs
#' (\code{auc_pos} / \code{auc_neg} are returned).
#'
#' @param dictionary A data.frame with a \code{word} column and one or more numeric
#'   dimension columns (the same format \code{\link{vectionary_builder}} accepts), or a
#'   plain character vector of words. A character vector -- or a data.frame with only a
#'   \code{word} column -- is read as a \strong{binary} (unipolar) list: every word
#'   scores 1. For a graded dictionary the \strong{sign} of the scores sets the poles (a
#'   mix of positive and negative is bipolar). Words ending in \code{"*"} are treated as
#'   stems and expanded against the vocabulary.
#' @param embeddings Either a numeric matrix with one row per word and \code{rownames}
#'   equal to the vocabulary, or a path to an embeddings file (FastText \code{.vec},
#'   GloVe or word2vec \code{.txt}). A path is read with the builder's header-aware
#'   loader. Rows are unit-normalized internally, so dot products are cosine similarities.
#' @param dimension Name of the dimension column to evaluate. Required only when the
#'   dictionary has more than one dimension column.
#' @param n_random Number of random word lists used as the chance-level baseline
#'   (default 500). Larger is steadier but slower.
#' @param seed Random seed for the baseline draws (default 574), for reproducibility.
#' @param freq_sorted Whether the embedding rows are ordered most-frequent-first, so the
#'   row index can stand in for a frequency rank in the frequency-confound check (the
#'   released GloVe, FastText and word2vec files are). \code{TRUE}/\code{FALSE} force the
#'   check on or off; the default \code{NULL} guesses from whether the vocabulary is
#'   non-alphabetical -- which only rules out an alphabetical dump, so set it explicitly
#'   if you know the ordering. Unit-normalization removes the vector-norm frequency
#'   signal, so row order is the only frequency information available.
#' @param against Optional discriminant-validity reference: a competing dictionary (same
#'   formats as \code{dictionary}) or a ready unit axis vector. The report adds the
#'   cosine between the operative axis and this competing axis -- e.g. a generic valence
#'   axis, to check an affective construct is not merely tracking negativity.
#'
#' @return An object of class \code{"dictionary_eval"} that prints a formatted report
#'   card. Its fields include the per-pole metric lists (\code{card_pos}, and
#'   \code{card_neg} when bipolar), the contrast (\code{pole_d}, \code{pole_d_beats}),
#'   the Separation AUCs (\code{auc}, \code{auc_pos}, \code{auc_neg}), the confounds
#'   (\code{cor_freq}, \code{cor_len}, \code{gen_align}), the discriminant cosine
#'   (\code{disc}), and metadata.
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
#'
#' # embeddings can also be a file path (read with the builder's loader):
#' rep <- dictionary_eval(dict, "path/to/cc.pt.300.vec")
#' }
#'
#' @importFrom stats var
#' @export
dictionary_eval <- function(dictionary, embeddings, dimension = NULL,
                            n_random = 500, seed = 574,
                            freq_sorted = NULL, against = NULL) {
  E         <- .resolve_embeddings(embeddings)
  vocab     <- rownames(E)
  seed_df   <- .dict_seed_df(dictionary, dimension, vocab)
  dimension <- attr(seed_df, "dimension")

  pos     <- seed_df$score > 0
  neg     <- seed_df$score < 0
  bipolar <- any(pos) & any(neg)
  if (bipolar && (sum(pos) < 2 || sum(neg) < 2))
    cli_abort(c("A bipolar dimension needs >= 2 seed words on each pole.",
      "i" = "Found {sum(pos)} on the high pole and {sum(neg)} on the low pole."))
  if (!bipolar && nrow(seed_df) < 2)
    cli_abort("A dimension needs at least 2 seed words.")

  generic <- colMeans(E); generic <- generic / sqrt(sum(generic^2))

  if (bipolar) {
    pos_w <- seed_df$word[pos]; pos_s <- seed_df$stem[pos]
    neg_w <- seed_df$word[neg]; neg_s <- seed_df$stem[neg]
    card_pos <- .pole_metrics(pos_w, pos_s, E, n_random, seed)
    card_neg <- .pole_metrics(neg_w, neg_s, E, n_random, seed + 1L)

    c_pos    <- colMeans(E[pos_w, , drop = FALSE])
    c_neg    <- colMeans(E[neg_w, , drop = FALSE])
    ref_axis <- c_pos - c_neg; ref_axis <- ref_axis / sqrt(sum(ref_axis^2))

    # contrast: pole gap (Cohen's d) vs a same-size bipolar random null
    W      <- E[seed_df$word, , drop = FALSE]
    pole_d <- .pole_gap(W, pos, neg)
    n1 <- sum(pos); n0 <- sum(neg)
    set.seed(seed + 2L)
    rd <- replicate(n_random, {
      Wr <- E[sample(nrow(E), n1 + n0), , drop = FALSE]
      rp <- c(rep(TRUE, n1), rep(FALSE, n0))
      .pole_gap(Wr, rp, !rp)
    })
    pole_d_beats <- mean(pole_d > rd)

    # Separation: each pole toward its own end vs a seed-excluded background, weaker wins
    is_bg   <- !(vocab %in% seed_df$word)
    auc_pos <- .loo_auc(pos_w, pos_s, c_neg, E, is_bg)
    auc_neg <- .loo_auc(neg_w, neg_s, c_pos, E, is_bg)
    auc     <- min(auc_pos, auc_neg, na.rm = TRUE)
  } else {
    words    <- seed_df$word; stems <- seed_df$stem
    card_pos <- .pole_metrics(words, stems, E, n_random, seed)
    card_neg <- NULL
    ref_axis <- colMeans(E[words, , drop = FALSE]); ref_axis <- ref_axis / sqrt(sum(ref_axis^2))
    pole_d   <- pole_d_beats <- NA_real_
    auc_pos  <- .loo_auc(words, stems, NULL, E, !(vocab %in% words))
    auc_neg  <- NA_real_
    auc      <- auc_pos
  }

  # checks read on the operative axis (pole difference if bipolar, else seed centroid)
  proj_all  <- as.numeric(E %*% ref_axis)
  use_freq  <- if (is.null(freq_sorted)) is.unsorted(vocab) else isTRUE(freq_sorted)
  cor_freq  <- if (use_freq) cor(proj_all, seq_len(nrow(E)), method = "spearman") else NA_real_
  cor_len   <- cor(proj_all, nchar(vocab), method = "spearman")
  gen_align <- .generic_align(ref_axis, generic)
  disc      <- if (is.null(against)) NA_real_ else
    abs(sum(ref_axis * .reference_axis(against, E, vocab)))

  structure(list(
    dimension = dimension, bipolar = bipolar,
    card_pos = card_pos, card_neg = card_neg,
    pole_d = pole_d, pole_d_beats = pole_d_beats,
    auc = auc, auc_pos = auc_pos, auc_neg = auc_neg,
    cor_freq = cor_freq, cor_len = cor_len, gen_align = gen_align, disc = disc,
    freq_used = use_freq, n_random = n_random,
    n_seed = nrow(seed_df), n_pos = sum(pos), n_neg = sum(neg),
    n_stems_total = attr(seed_df, "n_total_stems")),
    class = "dictionary_eval")
}

#' @rdname dictionary_eval
#' @param x A \code{dictionary_eval} object.
#' @param ... Unused.
#' @export
print.dictionary_eval <- function(x, ...) {
  cli_h1("Report card: {x$dimension}")
  if (x$bipolar)
    cli_alert_info("{x$n_seed} seed words ({x$n_pos} high, {x$n_neg} low) -- two unipolar cards plus a contrast")
  else
    cli_alert_info("{x$n_seed} seed words (unipolar)")

  pole_block <- function(card, title) {
    cli_h2(title)
    cli_ul(c(
      sprintf("Coherence: %.3f  (beats %.0f%% of random, loo range %.3f; load-bearing: %s)",
              card$coherence, 100 * card$coh_beats, card$coh_loo, card$coh_infl),
      sprintf("Dimensionality: PC1 %.3f, ~%s PCs to 80%%  (beats %.0f%% of random, loo range %.3f; load-bearing: %s)",
              card$pc1, card$pcs80, 100 * card$dim_beats, card$dim_loo, card$dim_infl)))
  }
  if (x$bipolar) {
    pole_block(x$card_pos, "High pole (+)")
    pole_block(x$card_neg, "Low pole (-)")
    cli_h2("Contrast")
    cli_ul(sprintf("Pole separation (Cohen's d): %.3f  (beats %.0f%% of random)",
                   x$pole_d, 100 * x$pole_d_beats))
  } else {
    pole_block(x$card_pos, "Metrics")
  }
  cli_text("")
  cli_ul(c(
    "{.strong Coherence}: how tightly a pole's seeds pin down one direction (1 = perfectly tight)",
    "{.strong Dimensionality}: variance share on PC1 (high = one concept; PCs-to-80% ~ how many sub-themes)",
    "{.field beats_random}: fraction of {x$n_random} equal-size random lists it out-scores",
    "{.field loo_range}: metric movement when any one stem is dropped (small = stable)"))

  cli_h2(if (x$bipolar) "Checks (Separation per pole; confounds on the contrast axis)" else "Checks")
  sep_band <- .auc_band(x$auc)
  sep_msg  <- sprintf("Separation (LOO-AUC) = %s  (%s)",
                      format(round(x$auc, 3), nsmall = 3), sep_band)
  if (grepl("strong|acceptable", sep_band))      cli_alert_success("{sep_msg}")
  else if (grepl("weak|near chance", sep_band))  cli_alert_warning("{sep_msg}")
  else                                           cli_alert_danger("{sep_msg}")
  if (x$bipolar && is.finite(x$auc_pos) && is.finite(x$auc_neg)) {
    weak <- if (x$auc_neg <= x$auc_pos) "low" else "high"
    cli_text("  {.emph by pole}: high = {format(round(x$auc_pos, 3), nsmall = 3)}, low = {format(round(x$auc_neg, 3), nsmall = 3)} (weaker {weak} pole governs)")
  }

  conf_line <- function(label, val) {
    v   <- .conf_band(val)
    msg <- sprintf("%s = %s  (%s)", label,
                   if (is.na(val)) "NA" else format(round(val, 3), nsmall = 3), v)
    if (grepl("not available", v))    cli_alert_info("{msg}")
    else if (grepl("negligible", v))  cli_alert_success("{msg}")
    else if (grepl("minor", v))       cli_alert_warning("{msg}")
    else                              cli_alert_danger("{msg}")
  }
  conf_line("Confound: frequency", x$cor_freq)
  conf_line("Confound: length", x$cor_len)
  conf_line("Confound: generic language", x$gen_align)
  if (is.finite(x$disc)) {
    b   <- .disc_band(x$disc)
    msg <- sprintf("Discriminant vs competing axis = %.3f  (%s)", x$disc, b)
    if (b == "distinct")          cli_alert_success("{msg}")
    else if (b == "some overlap") cli_alert_warning("{msg}")
    else                          cli_alert_danger("{msg}")
  }

  cli_text("")
  cli_text("{.emph Separation (AUC)}: <0.4 worse than random | 0.4-0.6 near chance | 0.6-0.7 weak | 0.7-0.8 acceptable | >=0.8 strong")
  cli_text("{.emph Confound (|rho| or |cos|)}: <0.1 negligible | 0.1-0.3 minor | >0.3 confound (% = variance shared)")
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
