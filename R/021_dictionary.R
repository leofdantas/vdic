#- Dictionary curation: evaluate, compare, and improve a seed word list --------
# Three CONSTRUCT-SIDE tools that run BEFORE vectionary_builder(): they describe and
# improve the seed WORD LIST itself, not whether the learned axis scores documents
# correctly (that needs external labels). Everything is cosine similarity on the seed
# words read against the full embedding vocabulary -- no random baselines, no learned axis.
#
#   dictionary_eval()    a report card: Coherence plus a per-WORD audit (redundant /
#                        weak-offcentre / ambiguous-generic / ambiguous-xpole) and
#                        frequency / length / generic-direction confounds.
#   dictionary_compare() how candidate operationalisations relate: heading cosine,
#                        cross-seed AUC, and the disagreement word-lists between two lists.
#   dictionary_suggest() words to ADD (raise_coherence / broaden_coverage), judged by the human.
#
# All three read the same parameter-free OPERATIVE axis -- the seed centroid (unipolar) or
# the pole-mean difference (bipolar) -- never the regularized axis vectionary_builder()
# learns later; word selection must not use the axis it trains.

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

# Normalize a dictionary argument to a data.frame with a `word` column and >= 1
# dimension column -- the shared front door for the dictionary_ tools and
# vectionary_builder(), so every entry reads the same inputs. A character vector (or
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
  # Resolve one dimension to a seed table (word, score, stem): coerce inputs via the
  # shared .as_dictionary_df() front door (the one vectionary_builder() also uses), then
  # expand stems ending in "*" against the vocabulary and keep each word only if its sign
  # is consistent across entries.
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

#- Internal: shared geometry primitives ---------------------------------------
# All numbers are read on unit-normalized vectors, so a dot product is a cosine.

# Coherence = length of the average seed vector (~ avg pairwise cosine), in [0, 1]. For a
# pole it equals the mean of that pole's per-word on_axis values (see .word_audit).
.coherence_length <- function(M) sqrt(sum(colMeans(M)^2))

# Unit reference axis of a seed table: pole difference if bipolar, else the raw centroid.
# All three functions read THIS axis (eval's confounds, compare's heading, suggest's
# selection) -- never the learned/regularized axis, and never a background-subtracted one.
.operative_axis <- function(W, pos, neg) {
  ax <- if (any(pos) && any(neg))
    colMeans(W[pos, , drop = FALSE]) - colMeans(W[neg, , drop = FALSE]) else colMeans(W)
  ax / sqrt(sum(ax^2))
}

# One-sided AUC: P(a random "hi" word outranks a random "lo" word), the Wilcoxon-Mann-
# Whitney statistic. NA when either group is empty.
.auc_one <- function(proj, hi, lo) {
  if (sum(hi) < 1 || sum(lo) < 1) return(NA_real_)
  rk <- rank(c(proj[hi], proj[lo])); n_hi <- sum(hi)
  (sum(rk[seq_len(n_hi)]) - n_hi * (n_hi + 1) / 2) / (n_hi * sum(lo))
}

# Plain-language band for a confound (effect size, NOT a p-value: with a ~400k-word
# vocabulary even a trivial correlation tests as "significant").
.conf_band <- function(r) {
  if (is.na(r)) return("not available -- rows not known to be frequency-sorted")
  lab <- if (abs(r) < 0.1) "negligible" else if (abs(r) < 0.3) "minor" else "confound"
  sprintf("%s, %.1f%% var", lab, 100 * r^2)
}

# A table printer that emits through cli (so cli_fmt() captures the tables when a card is
# captured to text; a bare print() would escape cli's sink).
.show <- function(...) cli_verbatim(capture.output(print(...)))

# hunspell spell-check that degrades to a no-op if the package is absent, checked at CALL
# time (a package-load-time conditional would bake in whatever was installed at build).
.hcheck <- function(w)
  if (requireNamespace("hunspell", quietly = TRUE)) hunspell::hunspell_check(w) else rep(TRUE, length(w))

# Walk a ranked vocabulary index and return the first `k` indices that are clean candidate
# words: alphabetic, >= 3 chars, not a stopword, a real dictionary word, not already a seed.
# LAZY -- it spellchecks only as far down the ranking as it must, never the whole vocabulary.
.take_clean <- function(idx, vocab, exclude, stop, k) {
  out <- integer(0)
  for (i in idx) {
    w <- vocab[i]
    if (w %in% exclude || nchar(w) < 3L || !grepl("^[a-z]+$", w) || w %in% stop) next
    if (!.hcheck(w)) next
    out <- c(out, i)
    if (length(out) >= k) break
  }
  out
}

# Per-word audit of one pole (the CORE of eval). For each seed w, all from primitives the
# package already ships: on_axis = cos to the pole centroid; z = robust z of on_axis WITHIN
# the pole ((on - median)/mad), so the weak flag is pole-relative and travels across
# embeddings; generic = |cos| to the vocabulary mean direction (embedding centrality, NOT
# usage frequency); twin = nearest other seed (redundancy); xpole = cos to the OPPOSITE
# pole's centroid -- same footing as on_axis (bipolar leakage). verdict is a soft hint.
.word_audit <- function(words, W, g, opp_W = NULL,
                        t_weak = 1.5, t_twin = 0.80, t_gen = 0.40) {
  cc  <- colMeans(W); cc <- cc / sqrt(sum(cc^2))
  on  <- as.numeric(W %*% cc)
  gen <- abs(as.numeric(W %*% g))
  s   <- mad(on); z <- if (s > 0) (on - median(on)) / s else rep(0, length(on))
  S   <- W %*% t(W); diag(S) <- -Inf
  ti  <- max.col(S, ties.method = "first")
  out <- data.frame(word = words, on_axis = round(on, 3), z = round(z, 2),
    generic = round(gen, 3), twin = round(S[cbind(seq_along(words), ti)], 3),
    twin_word = words[ti], stringsAsFactors = FALSE)
  xp <- rep(0, length(words))
  if (!is.null(opp_W)) {
    oc <- colMeans(opp_W); oc <- oc / sqrt(sum(oc^2))          # opposite-pole centroid
    xp <- as.numeric(W %*% oc)                                 # cos to it -- same footing as on
    xi <- max.col(W %*% t(opp_W), ties.method = "first")       # nearest opposite SEED (diagnostic)
    out$xpole <- round(xp, 3); out$xpole_word <- rownames(opp_W)[xi]
  }
  v <- rep("anchor", length(words))
  v[out$twin > t_twin]               <- "redundant"
  v[z < -t_weak]                     <- "weak-offcentre"
  v[gen > t_gen]                     <- "ambiguous-generic"   # absolute: generic is an
  if (!is.null(opp_W)) v[xp > on]    <- "ambiguous-xpole"     # embedding-comparable confound
  out$verdict <- v
  sev <- c(`ambiguous-xpole` = 1, `ambiguous-generic` = 2, `weak-offcentre` = 3,
           redundant = 4, anchor = 5)
  # within a verdict group, order by THAT verdict's decisive metric (worst first)
  key2 <- ifelse(v == "ambiguous-generic", -out$generic,
          ifelse(v == "weak-offcentre",    out$z,
          ifelse(v == "redundant",         -out$twin,
          ifelse(v == "ambiguous-xpole",   -xp, 0))))
  out[order(sev[v], key2), ]
}

#- dictionary_eval(): the report card -----------------------------------------
# One pole's whole card: coherence + per-word audit.
.dict_card <- function(words, stems, E, g, opp_words, params) {
  W     <- E[words, , drop = FALSE]
  opp_W <- if (length(opp_words)) E[opp_words, , drop = FALSE] else NULL
  list(
    coherence = .coherence_length(W),
    audit     = .word_audit(words, W, g, opp_W, params$t_weak, params$t_twin, params$t_gen),
    n         = length(words), n_stems = length(unique(stems)))
}

#' Evaluate a seed dictionary (construct-side report card)
#'
#' @description
#' Describes a seed word list as a measurement axis using only cosine similarity read
#' against the full embedding vocabulary. Reports each pole's \strong{Coherence} (how
#' tightly the seeds point one way -- the mean of the per-word \code{on_axis} values) and a
#' per-word \strong{audit} that flags each seed as \code{redundant} (a near-duplicate of
#' another seed), \code{weak-offcentre} (a low outlier within its pole),
#' \code{ambiguous-generic} (leaning into the vocabulary's average direction), or
#' \code{ambiguous-xpole} (closer to the other pole's centroid, bipolar only), plus
#' frequency / length / generic-direction \strong{confounds} on the operative axis.
#'
#' This is a \strong{construct-side} diagnostic: it judges the WORD LIST, not whether the
#' eventual axis scores documents correctly. Run it \emph{before}
#' \code{\link{vectionary_builder}} to curate the seeds. It reads the same parameter-free
#' operative axis -- the seed centroid (unipolar) or pole-mean difference (bipolar) -- that
#' \code{\link{dictionary_compare}} and \code{\link{dictionary_suggest}} use, never the
#' regularized axis the builder learns.
#'
#' @param dictionary A data.frame with a \code{word} column and one or more numeric
#'   dimension columns (the format \code{\link{vectionary_builder}} accepts), or a plain
#'   character vector of words. A character vector -- or a \code{word}-only data.frame -- is
#'   read as a \strong{binary} (unipolar) list (every word scores 1); for a graded
#'   dictionary the \strong{sign} of the scores sets the poles (a mix of positive and
#'   negative is bipolar). Words ending in \code{"*"} are stems, expanded against the vocabulary.
#' @param embeddings A numeric matrix with one row per word and \code{rownames} equal to
#'   the vocabulary, or a path to an embeddings file (FastText \code{.vec}, GloVe or
#'   word2vec \code{.txt}). Rows are unit-normalized internally.
#' @param dimension Name of the dimension column to evaluate. Required only when the
#'   dictionary has more than one dimension column.
#' @param freq_sorted Whether the embedding rows are ordered most-frequent-first, so the
#'   row index can stand in for a frequency rank in the frequency confound. \code{TRUE} /
#'   \code{FALSE} force the check on or off; the default \code{NULL} guesses from whether
#'   the vocabulary is non-alphabetical.
#' @param t_weak Robust-z cutoff (within a pole) below which a seed is flagged
#'   \code{weak-offcentre} (default 1.5).
#' @param t_twin Cosine above which a seed is flagged \code{redundant} with its nearest
#'   fellow seed (default 0.80).
#' @param t_gen \code{|cos|} to the vocabulary mean direction above which a seed is flagged
#'   \code{ambiguous-generic} (default 0.40).
#'
#' @return An object of class \code{"dictionary_eval"} that prints a formatted report card.
#'   Fields include the per-pole cards (\code{cards}, each with \code{coherence} and the
#'   per-word \code{audit} data frame), the bipolar \code{contrast} (cosine between the pole
#'   means), the confounds (\code{cor_freq}, \code{cor_len}, \code{gen_align}), and metadata.
#'
#' @seealso \code{\link{dictionary_compare}} to contrast candidate lists,
#'   \code{\link{dictionary_suggest}} to propose words to add,
#'   \code{\link{vectionary_builder}} to learn the axis once the list is curated.
#'
#' @examples
#' \dontrun{
#' dict <- data.frame(
#'   word      = c("happy", "joy", "delight", "sad", "grief", "misery"),
#'   sentiment = c(1, 1, 1, -1, -1, -1)
#' )
#' dictionary_eval(dict, embeddings)        # prints the report card
#' }
#'
#' @importFrom stats cor mad median
#' @export
dictionary_eval <- function(dictionary, embeddings, dimension = NULL,
                            freq_sorted = NULL, t_weak = 1.5, t_twin = 0.80, t_gen = 0.40) {
  E         <- .resolve_embeddings(embeddings)
  vocab     <- rownames(E)
  seed_df   <- .dict_seed_df(dictionary, dimension, vocab)
  dimension <- attr(seed_df, "dimension")
  params    <- list(t_weak = t_weak, t_twin = t_twin, t_gen = t_gen)

  pos <- seed_df$score > 0; neg <- seed_df$score < 0
  bipolar <- any(pos) && any(neg)
  if (bipolar && (sum(pos) < 2 || sum(neg) < 2))
    cli_abort("A bipolar dimension needs >= 2 seed words on each pole.")
  if (!bipolar && nrow(seed_df) < 2) cli_abort("A dimension needs at least 2 seed words.")

  g <- colMeans(E); g <- g / sqrt(sum(g^2))                       # vocabulary mean direction (centrality)
  if (bipolar) {
    pw <- seed_df$word[pos]; ps <- seed_df$stem[pos]
    nw <- seed_df$word[neg]; ns <- seed_df$stem[neg]
    cards <- list(`+` = .dict_card(pw, ps, E, g, nw, params),
                  `-` = .dict_card(nw, ns, E, g, pw, params))
    cpos <- colMeans(E[pw, , drop = FALSE]); cneg <- colMeans(E[nw, , drop = FALSE])
    contrast <- sum(cpos * cneg) / sqrt(sum(cpos^2) * sum(cneg^2))  # cos between pole means
  } else {
    cards <- list(`.` = .dict_card(seed_df$word, seed_df$stem, E, g, character(0), params))
    contrast <- NA_real_
  }
  ref <- .operative_axis(E[seed_df$word, , drop = FALSE], pos, neg)

  proj <- as.numeric(E %*% ref)
  use_freq <- if (is.null(freq_sorted)) is.unsorted(vocab) else isTRUE(freq_sorted)
  structure(list(
    dimension = dimension, bipolar = bipolar, cards = cards, contrast = contrast,
    gen_align = abs(sum(ref * g)),
    cor_freq  = if (use_freq) cor(proj, seq_len(nrow(E)), method = "spearman") else NA_real_,
    cor_len   = cor(proj, nchar(vocab), method = "spearman"), freq_used = use_freq,
    n_seed = nrow(seed_df), n_pos = sum(pos), n_neg = sum(neg),
    n_stems_total = attr(seed_df, "n_total_stems")),
    class = "dictionary_eval")
}

#' @rdname dictionary_eval
#' @param x A \code{dictionary_eval} object.
#' @param ... Unused.
#' @export
print.dictionary_eval <- function(x, ...) {
  cli_h1("Dictionary report card: {.field {x$dimension}}")
  matched <- sum(vapply(x$cards, function(card) card$n_stems, integer(1)))
  if (x$bipolar)
    cli_alert_info("{x$n_seed} seeds ({.strong {x$n_pos}} high / {.strong {x$n_neg}} low) from {matched}/{x$n_stems_total} stems matched")
  else
    cli_alert_info("{x$n_seed} seeds (unipolar) from {matched}/{x$n_stems_total} stems matched")

  # each flagged word rendered with its DECISIVE number, inserted as plain data (no cli
  # markup inside, so a stray brace in a token can never break the template)
  vfmt <- function(sub, v) switch(v,
    "ambiguous-generic" = sprintf("%s (|cos| %.2f)", sub$word, sub$generic),
    "weak-offcentre"    = sprintf("%s (z %.1f)",     sub$word, sub$z),
    "redundant"         = sprintf("%s (~%s %.2f)",   sub$word, sub$twin_word, sub$twin),
    "ambiguous-xpole"   = sprintf("%s (opp-centroid %.2f, near %s)", sub$word, sub$xpole, sub$xpole_word))
  vlab <- c(`ambiguous-xpole`   = "closer to the other pole",
            `ambiguous-generic` = "leans to the vocabulary's average (central / unmarked) direction",
            `weak-offcentre`    = "off-centre in their pole",
            redundant           = "near-duplicate of another seed")

  pole_block <- function(card, title) {
    cli_h2(title)
    cli_text("{.strong Coherence} {.val {round(card$coherence, 3)}} {.emph (= mean of the on-axis column below; tightness 0-1, size-sensitive)}")
    aud <- card$audit; n_ok <- sum(aud$verdict == "anchor")
    if (n_ok == nrow(aud)) {
      cli_alert_success("{.strong Word audit}: all {nrow(aud)} seeds are clean anchors")
    } else {
      cli_text("{.strong Word audit}: {.val {n_ok}}/{nrow(aud)} clean anchors -- the rest to review:")
      cli_ul()
      for (v in names(vlab)) {
        sub <- aud[aud$verdict == v, ]
        if (!nrow(sub)) next
        words <- paste(vfmt(sub, v), collapse = ", ")
        cli_li("{.strong {v}} {.emph ({vlab[[v]]})}: {words}")
      }
      cli_end()
    }
  }
  if (x$bipolar) {
    pole_block(x$cards[["+"]], "High pole (+)")
    pole_block(x$cards[["-"]], "Low pole (-)")
    cli_h2("Contrast")
    ctag <- if (x$contrast < 0.2) "poles point clearly apart" else
            if (x$contrast < 0.5) "poles partly opposed" else "poles still share much direction"
    cli_text("Pole-means cosine {.val {round(x$contrast, 3)}} {.emph ({ctag}; lower = more opposed)}")
  } else pole_block(x$cards[["."]], "Metrics")

  cd <- cli_div(theme = list(h2 = list("margin-bottom" = 0)))   # subtitle hugs the rule
  cli_h2("Confounds")
  cli_text("{.emph does the axis track a nuisance (not the construct)?}")
  cli_end(cd)
  cli_text("")                                                  # blank line before the bullets
  # bullet per confound: value, band, and a one-line read keyed to the SIGN (Frequency and
  # Length are signed Spearman correlations; Generic is a sign-less |cos|) and the band.
  conf_li <- function(lab, v, kind) {
    b <- .conf_band(v)
    if (is.na(v)) { cli_li("{.strong {lab}} NA {.emph ({b})}"); return(invisible()) }
    neg  <- v < 0
    expl <- switch(kind,
      freq = if (grepl("negligible", b)) "independent of how common a word is."
             else if (neg) "leans toward more frequent words; partly rewards common vocabulary over the construct."
             else "leans toward rarer words; partly tracks rarity over the construct.",
      len  = if (grepl("negligible", b)) "word length does not affect the score."
             else if (!neg) "longer words score higher; a possible length artifact."
             else "shorter words score higher; a possible length artifact.",
      generic = if (grepl("negligible", b)) "points away from the vocabulary's average direction; carves its own region."
                else if (grepl("minor", b)) "leans slightly toward the vocabulary's average direction."
                else "leans into the vocabulary's average direction; partly measures geometric centrality, not the construct.")
    cli_li("{.strong {lab}} {.val {round(v, 3)}} {.emph ({b})} -- {expl}")
  }
  cli_ul()
  conf_li("Frequency", x$cor_freq, "freq")
  conf_li("Length", x$cor_len, "len")
  conf_li("Generic direction", x$gen_align, "generic")
  cli_end()
  if (x$bipolar)
    cli_text("{.emph Bipolar axis: nuisances shared by both poles cancel on the difference, so these read low -- the per-word flags carry genericness.}")
  invisible(x)
}

#- dictionary_compare(): how candidate lists relate ---------------------------

#' Compare candidate seed dictionaries
#'
#' @description
#' Shows how a set of candidate operationalisations of the same construct relate, on the
#' parameter-free operative axis. Reports the \strong{heading cosine} between every pair of
#' axes, a threshold-free \strong{cross-seed AUC} (can each axis rank its own seeds above a
#' rival's -- the rival's seeds are the negatives, so there is no neighbourhood radius to
#' choose), and, for exactly two dictionaries, the \strong{disagreement word-lists}: the
#' vocabulary ranked by the difference of the two axis projections, so the top words are
#' what the first list captures and the second misses, and the bottom vice versa.
#'
#' Currently \strong{unipolar} dictionaries only: the cross-seed AUC and the signed
#' disagreement list assume every seed projects high on its own axis, which a bipolar
#' (pole-difference) axis violates. Compare each pole as its own unipolar list instead.
#'
#' @param dictionaries A named list of two or more dictionaries (each in any format
#'   \code{\link{dictionary_eval}} accepts).
#' @param embeddings A numeric matrix with vocabulary \code{rownames}, or a path to an
#'   embeddings file. Rows are unit-normalized internally.
#' @param dimension Name of the dimension column to read (when graded).
#' @param language Language code for the stopword list used to clean the disagreement
#'   word-lists (default \code{"en"}).
#' @param n_show Number of words to show on each side of a disagreement list (default 15).
#'
#' @return An object of class \code{"dictionary_compare"}: the \code{heading} cosine matrix,
#'   the cross-seed \code{auc} matrix, and (for two dictionaries) the \code{disagree} lists.
#'
#' @seealso \code{\link{dictionary_eval}}, \code{\link{dictionary_suggest}}.
#'
#' @examples
#' \dontrun{
#' dictionary_compare(list(weather = weather_words, chemistry = chem_words), embeddings)
#' }
#'
#' @export
dictionary_compare <- function(dictionaries, embeddings, dimension = NULL,
                               language = "en", n_show = 15) {
  if (!is.list(dictionaries) || length(dictionaries) < 2)
    cli_abort("{.arg dictionaries} must be a list of 2 or more dictionaries.")
  if (is.null(names(dictionaries)))
    names(dictionaries) <- paste0("dict", seq_along(dictionaries))
  E     <- .resolve_embeddings(embeddings)
  vocab <- rownames(E)
  stop  <- .get_stopwords(language)

  sds <- lapply(dictionaries, function(d) .dict_seed_df(d, dimension, vocab))
  # Cross-seed AUC and the signed disagreement list assume every seed projects HIGH on its own
  # axis -- true only for unipolar lists. A bipolar axis (pos - neg) sends its negative pole
  # low by construction, so both reads break. Refuse bipolar input rather than return nonsense.
  is_bip <- vapply(sds, function(sd) any(sd$score > 0) && any(sd$score < 0), logical(1))
  if (any(is_bip))
    cli_abort(c("{.fn dictionary_compare} currently supports unipolar dictionaries only.",
      "x" = "{sum(is_bip)} bipolar list{?s} supplied: {.val {names(sds)[is_bip]}}.",
      "i" = "Compare each pole as its own unipolar list, or split the construct."))
  axes <- lapply(sds, function(sd) {
    W <- E[sd$word, , drop = FALSE]; .operative_axis(W, sd$score > 0, sd$score < 0)
  })
  A <- do.call(cbind, axes)
  heading <- abs(crossprod(A))                                  # |cos| between every pair

  k      <- length(dictionaries)
  projs  <- lapply(axes, function(a) as.numeric(E %*% a))       # one projection per axis
  inseed <- lapply(sds, function(sd) vocab %in% sd$word)
  auc <- matrix(NA_real_, k, k, dimnames = list(names(axes), names(axes)))
  for (i in seq_len(k)) for (j in seq_len(k)) if (i != j)       # axis i ranks ITS seeds
    auc[i, j] <- .auc_one(projs[[i]], inseed[[i]], inseed[[j]]) # above j's seeds

  disagree <- NULL
  if (k == 2) {
    pa <- projs[[1]]; pb <- projs[[2]]
    excl <- c(sds[[1]]$word, sds[[2]]$word); d <- pa - pb
    mk <- function(idx) data.frame(word = vocab[idx], projA = round(pa[idx], 3),
      projB = round(pb[idx], 3), gap = round(d[idx], 3), row.names = NULL)
    disagree <- list(
      a_not_b = mk(.take_clean(order(d, decreasing = TRUE), vocab, excl, stop, n_show)),
      b_not_a = mk(.take_clean(order(d), vocab, excl, stop, n_show)))
  }
  structure(list(names = names(axes), heading = round(heading, 3), auc = round(auc, 3),
    disagree = disagree, dimension = attr(sds[[1]], "dimension")),
    class = "dictionary_compare")
}

#' @rdname dictionary_compare
#' @param x A \code{dictionary_compare} object.
#' @param ... Unused.
#' @export
print.dictionary_compare <- function(x, ...) {
  cli_h1("Dictionary comparison: {.field {x$dimension}}")
  if (!is.null(x$disagree)) {                                  # headline read for a pair
    nm <- x$names; h <- x$heading[1, 2]
    tag <- if (h < 0.3) "near-orthogonal -- distinct constructs"
           else if (h < 0.6) "partly overlapping" else "largely the same axis"
    cli_alert_info("{.strong {nm[1]}} vs {.strong {nm[2]}}: heading |cos| {.val {h}} -- {tag}")
  }
  cli_h2("Heading |cos| {.emph (1 = same axis, 0 = orthogonal)}")
  .show(round(x$heading, 3))
  cli_h2("Cross-seed AUC {.emph (row axis ranks ITS seeds above the column's; ~.5 = interchangeable, -- = self)}")
  am <- formatC(x$auc, format = "f", digits = 3)              # the i==i self-comparison is
  dim(am) <- dim(x$auc); dimnames(am) <- dimnames(x$auc)      # undefined -> show a dash, not NA
  am[is.na(x$auc)] <- "-"
  .show(am, quote = FALSE)
  if (!is.null(x$disagree)) {
    nm <- x$names
    cli_h2("{.strong {nm[1]}} captures, {.strong {nm[2]}} misses {.emph (ranked by score gap)}")
    .show(x$disagree$a_not_b, row.names = FALSE)
    cli_h2("{.strong {nm[2]}} captures, {.strong {nm[1]}} misses {.emph (ranked by score gap)}")
    .show(x$disagree$b_not_a, row.names = FALSE)
  }
  invisible(x)
}

#- dictionary_suggest(): words to add -----------------------------------------
# One pole's add-suggestions (the body shared by both poles). Selection reads the same
# operative axis as eval/compare -- the seed centroid (unipolar) or pole difference (bipolar).
.suggest_one <- function(pole_words, opp_words, E, n, raise_coherence, broaden_coverage,
                         pool, rarity_words, spellcheck, stop) {
  vocab <- rownames(E)
  if (length(pole_words) < 2) cli_abort("Each pole needs at least 2 words.")
  cpole <- colMeans(E[pole_words, , drop = FALSE])
  axis  <- if (is.null(opp_words)) cpole                          # unipolar: raw centroid (matches eval/compare)
           else cpole - colMeans(E[opp_words, , drop = FALSE])    # bipolar: pole difference
  axis  <- axis / sqrt(sum(axis^2))
  pr    <- as.numeric(E %*% axis)
  thr   <- median(pr[match(pole_words, vocab)])                   # as far out as a typical seed

  known <- c(pole_words, opp_words)
  ok <- grepl("^[a-z]+$", vocab) & nchar(vocab) >= 3 & !(vocab %in% known) &
        !(vocab %in% stop) & pr >= thr
  if (!is.null(rarity_words)) {
    if (!is.unsorted(vocab))
      cli_abort(c("Embedding rows are alphabetical, so {.arg rarity_words} has no frequency to use.",
        "i" = "Set {.code rarity_words = NULL} to skip the frequency filter."))
    ok <- ok & seq_along(vocab) <= round(rarity_words * nrow(E))
  }
  if (spellcheck) { ci <- which(ok); ok[ci[!hunspell::hunspell_check(vocab[ci])]] <- FALSE }
  ord  <- order(pr, decreasing = TRUE); ord <- ord[ok[ord]]
  near <- function(i) pole_words[which.max(as.numeric(E[pole_words, , drop = FALSE] %*% E[i, ]))]
  res  <- list()
  if (raise_coherence && length(ord)) {
    idx <- head(ord, n)
    res$coherence <- data.frame(mode = "coherence", word = vocab[idx],
      axis_proj = round(pr[idx], 3), nearest_seed = vapply(idx, near, ""), stringsAsFactors = FALSE)
  }
  if (broaden_coverage && length(ord)) {
    poolidx <- head(ord, pool)
    maxsim  <- apply(E[poolidx, , drop = FALSE] %*% t(E[pole_words, , drop = FALSE]), 1, max)
    picks   <- integer(0)
    for (j in seq_len(min(n, length(poolidx)))) {     # greedy farthest-point: most novel each time
      kk <- which.min(maxsim); picks <- c(picks, poolidx[kk])
      maxsim <- pmax(maxsim, as.numeric(E[poolidx, , drop = FALSE] %*% E[poolidx[kk], ]))
      maxsim[kk] <- Inf
    }
    res$coverage <- data.frame(mode = "coverage", word = vocab[picks],
      axis_proj = round(pr[picks], 3), nearest_seed = vapply(picks, near, ""), stringsAsFactors = FALSE)
  }
  if (!length(res)) return(data.frame(mode = character(), word = character(),
    axis_proj = numeric(), nearest_seed = character(), stringsAsFactors = FALSE))
  do.call(rbind, lapply(res, function(d) d[order(-d$axis_proj), ]))
}

#' Suggest words to add to a seed dictionary
#'
#' @description
#' Proposes on-theme words to ADD to a seed list, judged by the human: \code{raise_coherence}
#' returns the strongest words on the pole (which sharpen the axis) and \code{broaden_coverage}
#' returns on-theme words in new directions (greedy farthest-point, which widen it). For a
#' bipolar dimension both poles are handled. Selection reads the same parameter-free operative
#' axis as \code{\link{dictionary_eval}} -- the seed centroid (unipolar) or pole-mean
#' difference (bipolar) -- never a learned axis, and never edits the dictionary. (Spotting
#' weak / redundant existing members is now \code{\link{dictionary_eval}}'s per-word audit.)
#'
#' @inheritParams dictionary_eval
#' @param n Number of words to suggest per pole and per mode (default 20).
#' @param raise_coherence If \code{TRUE} (default), suggest the strongest on-pole words.
#' @param broaden_coverage If \code{TRUE} (default), suggest on-theme words in new directions
#'   (greedy farthest-point).
#' @param pool Size of the candidate pool the coverage search draws from (default 1500).
#' @param rarity_words Keep only the most frequent fraction of the vocabulary as candidates
#'   (e.g. \code{0.2} = top 20\%), using row order as the frequency rank; this removes rare
#'   off-theme real words that spellcheck passes. \code{NULL} skips the filter. Assumes the
#'   rows are frequency-sorted (errors on an alphabetical vocabulary). A genuinely sparse pole
#'   still returns few words \emph{and warns}, regardless of this filter.
#' @param spellcheck If \code{TRUE} (default), drop non-words via \code{hunspell} (the engine
#'   \code{\link{vectionary_builder}} uses). Requires the \code{hunspell} package.
#' @param language Language code for the stopword list (default \code{"en"}).
#'
#' @return An object of class \code{"dictionary_suggest"}: a list with an \code{add} data
#'   frame (columns \code{pole}, \code{mode}, \code{word}, \code{axis_proj},
#'   \code{nearest_seed}). It prints a formatted list of additions.
#'
#' @seealso \code{\link{dictionary_eval}} for the report card,
#'   \code{\link{dictionary_compare}} to contrast lists.
#'
#' @examples
#' \dontrun{
#' dict <- data.frame(word = c("happy", "joy", "sad", "grief"), sentiment = c(1, 1, -1, -1))
#' dictionary_suggest(dict, embeddings)
#' dictionary_suggest(dict, embeddings, rarity_words = NULL, spellcheck = FALSE)
#' }
#'
#' @export
dictionary_suggest <- function(dictionary, embeddings, dimension = NULL, n = 20,
                               raise_coherence = TRUE, broaden_coverage = TRUE,
                               pool = 1500, rarity_words = 0.2, spellcheck = TRUE,
                               language = "en") {
  if (spellcheck && !requireNamespace("hunspell", quietly = TRUE))
    cli_abort(c("{.pkg hunspell} is required for {.code spellcheck = TRUE}.",
      "i" = "Install it, or set {.code spellcheck = FALSE}."))
  E         <- .resolve_embeddings(embeddings)
  seed_df   <- .dict_seed_df(dictionary, dimension, rownames(E))
  dimension <- attr(seed_df, "dimension")
  stop      <- .get_stopwords(language)

  pos <- seed_df$word[seed_df$score > 0]; neg <- seed_df$word[seed_df$score < 0]
  bipolar <- length(pos) > 0 && length(neg) > 0
  poles <- if (bipolar)
    list(list(label = "high (+)", pole = pos, opp = neg), list(label = "low (-)", pole = neg, opp = pos))
  else list(list(label = "seeds", pole = seed_df$word, opp = NULL))

  add <- list()
  for (p in poles) {
    a  <- .suggest_one(p$pole, p$opp, E, n, raise_coherence, broaden_coverage,
                       pool, rarity_words, spellcheck, stop)
    nw <- length(unique(a$word))
    if (nw < n)
      cli_warn(c("Pole {.val {p$label}}: only {nw} on-theme candidate{?s} survived the filters (asked for {n}).",
        "i" = "The region is thin, so both modes overlap. Loosen with {.code rarity_words = NULL} or check {.arg spellcheck}."))
    if (nrow(a)) add[[p$label]] <- cbind(pole = p$label, a, stringsAsFactors = FALSE)
  }
  structure(list(add = do.call(rbind, add), dimension = dimension, bipolar = bipolar),
    class = "dictionary_suggest")
}

#' @rdname dictionary_suggest
#' @param x A \code{dictionary_suggest} object.
#' @param ... Unused.
#' @export
print.dictionary_suggest <- function(x, ...) {
  cli_h1("Words to add: {.field {x$dimension}}")
  titles <- c(coherence = "Sharpen the axis {.emph (strongest on-pole words)}",
              coverage  = "Broaden coverage {.emph (on-theme, new directions)}")
  for (pl in unique(x$add$pole)) {
    cli_h2("Pole: {.strong {pl}}")
    sub <- x$add[x$add$pole == pl, ]
    for (m in intersect(names(titles), unique(sub$mode))) {
      cli_text(titles[[m]])
      .show(sub[sub$mode == m, c("word", "axis_proj", "nearest_seed")], row.names = FALSE)
    }
  }
  invisible(x)
}
