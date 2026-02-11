#- Vec-tionary S3 class and text analysis ----
#
# This file defines the Vec-tionary S3 class and all functions for scoring
# text against a vectionary. A vectionary stores pre-computed "word projections":
# a numeric score per word per dimension, built from word embeddings by
# vectionary_builder() in 03_builder.R. Scoring a document means:
#
#   1. Tokenize the text (lowercase, strip punctuation, split on whitespace)
#   2. Look up each token's projection score in the vectionary's word_projections
#   3. Aggregate those scores into a single number per dimension (mean, rms, etc.)
#
# Two code paths exist for efficiency:
#   - Single-document path: .get_doc_scores() + per-dimension sapply
#   - Batch path: .tokenize_and_match() + matrix operations via .batch_metric()
# The per-metric wrapper functions (.mean, .rms, etc.) auto-dispatch between them.


#' Vec-tionary S3 Class
#'
#' @description
#' An S3 class representing a vector-based dictionary (vec-tionary).
#' Contains axes in embedding space and methods for analyzing text.
#'
#' @details
#' Vectionary objects are created by [vectionary_builder()].
#' They provide methods to compute various metrics on text:
#' \itemize{
#'   \item `$mean(text)` - Arithmetic mean of projections
#'   \item `$rms(text)` - Root mean square (emphasizes high-magnitude projections)
#'   \item `$sd(text)` - Standard deviation of projections
#'   \item `$se(text)` - Standard error of the mean
#'   \item `$top_10(text)` - Mean of 10 highest projections (strongest signals)
#'   \item `$top_20(text)` - Mean of 20 highest projections (strongest signals)
#'   \item `$metrics(text)` - All metrics at once
#'   \item `$diagnose(n, dimension)` - Diagnostic report (see [vectionary_diagnose()])
#' }
#'
#' Stopwords are automatically removed at build time based on the
#' \code{language} parameter passed to [vectionary_builder()].
#' Use \code{remove_stopwords = FALSE} to include stopwords.
#'
#' @name Vec-tionary
NULL


#-- S3 methods: print, summary, $ ----

#' Print method for Vec-tionary
#'
#' @param x A vectionary object
#' @param ... Additional arguments (ignored)
#'
#' @return Invisibly returns \code{x}.
#'
#' @export
`print.Vec-tionary` <- function(x, ...) {
  cat("Vectionary\n")
  cat("-------------------\n")
  cat("Dimensions:", paste(x$dimensions, collapse = ", "), "\n")

  # Metadata format changed between versions: older objects use
  # "dictionary_words_count", newer ones use "seed_words_count" + optional
  # "expanded_words". Handle both so old .rds files still print correctly.
  if (!is.null(x$metadata$seed_words_count)) {
    if (!is.null(x$metadata$expanded_words) && x$metadata$expanded_words > 0) {
      cat("Training words:", x$metadata$training_words_count,
          sprintf("(%d seed + %d expanded)\n", x$metadata$seed_words_count, x$metadata$expanded_words))
    } else {
      cat("Seed words:", x$metadata$seed_words_count, "\n")
    }
  } else if (!is.null(x$metadata$dictionary_words_count)) {
    cat("Dictionary words:", x$metadata$dictionary_words_count, "\n")
  }

  if (!is.null(x$metadata$vocab_size)) {
    cat("Vocabulary:", x$metadata$vocab_size, "words\n")
  }

  if (!is.null(x$metadata$build_date)) {
    cat("Built:", format(x$metadata$build_date, "%Y-%m-%d %H:%M"), "\n")
  } else if (!is.null(x$metadata$load_date)) {
    cat("Loaded:", format(x$metadata$load_date, "%Y-%m-%d %H:%M"), "\n")
  }

  cat("\nAvailable methods:\n")
  cat("  $mean(text)    - Arithmetic mean of projections\n")
  cat("  $rms(text)     - Root mean square (emphasizes high values)\n")
  cat("  $sd(text)      - Standard deviation\n")
  cat("  $se(text)      - Standard error of the mean\n")
  cat("  $top_10(text)  - Mean of top 10 projections (strongest signals)\n")
  cat("  $top_20(text)  - Mean of top 20 projections (strongest signals)\n")
  cat("  $metrics(text) - All metrics at once\n")

  invisible(x)
}


#' Summary method for Vec-tionary
#'
#' @param object A vectionary object
#' @param ... Additional arguments (ignored)
#'
#' @return Invisibly returns \code{object}.
#'
#' @export
`summary.Vec-tionary` <- function(object, ...) {
  cat("Vectionary Summary\n")
  cat("====================\n\n")

  cat("Dimensions (", length(object$dimensions), "):\n", sep = "")
  for (dim in object$dimensions) {
    cat("  -", dim, "\n")
  }
  cat("\n")

  if (!is.null(object$metadata)) {
    cat("Metadata:\n")
    if (!is.null(object$metadata$method)) {
      cat("  Method:", object$metadata$method, "\n")
    }
    if (!is.null(object$metadata$binary_word)) {
      dict_type <- if (object$metadata$binary_word) "binary" else "continuous"
      cat("  Dictionary type:", dict_type, "\n")
    }
    if (!is.null(object$metadata$embeddings_source)) {
      cat("  Embeddings:", object$metadata$embeddings_source, "\n")
    }
    # Handle both old and new metadata fields (see print method comment)
    if (!is.null(object$metadata$seed_words_count)) {
      cat("  Seed words:", object$metadata$seed_words_count, "\n")
      if (!is.null(object$metadata$expanded_words) && object$metadata$expanded_words > 0) {
        cat("  Expanded words:", object$metadata$expanded_words, "\n")
        cat("  Training words:", object$metadata$training_words_count, "\n")
      }
    } else if (!is.null(object$metadata$dictionary_words_count)) {
      cat("  Dictionary words:", object$metadata$dictionary_words_count, "\n")
    }
    if (!is.null(object$metadata$vocab_size)) {
      cat("  Vocabulary size:", object$metadata$vocab_size, "words\n")
    }
    if (!is.null(object$metadata$words_found)) {
      training_count <- if (!is.null(object$metadata$training_words_count)) {
        object$metadata$training_words_count
      } else {
        object$metadata$dictionary_words_count
      }
      if (!is.null(training_count)) {
        cat("  Words found:", object$metadata$words_found,
            sprintf("(%.1f%%)\n", 100 * object$metadata$words_found / training_count))
      }
    }
    if (!is.null(object$metadata$build_date)) {
      cat("  Built:", format(object$metadata$build_date, "%Y-%m-%d %H:%M:%S"), "\n")
    }
  }

  invisible(object)
}


#' Dollar method for Vec-tionary (enables $method syntax)
#'
#' @param x A vectionary object
#' @param name Method name
#'
#' @return A function (for metric methods and \code{diagnose}) or the field value
#'   (for \code{axes}, \code{word_projections}, \code{dimensions}, \code{metadata}).
#'
#' @export
`$.Vec-tionary` <- function(x, name) {

  # These are the method names that return a scoring function.
  # When the user writes vect$mean, this returns a *function* that they
  # then call with text: vect$mean("some document").
  valid_methods <- c("mean", "rms", "sd", "se", "top_10", "top_20", "metrics")

  if (name == "diagnose") {
    # $diagnose returns a function with different signature (n, dimension)
    # than the scoring methods (text), so it's handled separately
    return(function(n = 30, dimension = NULL) {
      vectionary_diagnose(x, n = n, dimension = dimension)
    })

  } else if (name %in% valid_methods) {
    # Return a closure that captures the vectionary's word_projections and
    # dimensions, so the user only needs to pass text when calling it.
    # Each closure dispatches to the corresponding internal metric function
    # (e.g., .mean, .rms), which handles both single-doc and batch paths.
    return(function(text) {
      switch(name,
        mean = .mean(text, x$word_projections, x$dimensions),
        rms = .rms(text, x$word_projections, x$dimensions),
        sd = .sd(text, x$word_projections, x$dimensions),
        se = .se(text, x$word_projections, x$dimensions),
        top_10 = .top_10(text, x$word_projections, x$dimensions),
        top_20 = .top_20(text, x$word_projections, x$dimensions),
        metrics = .metrics(text, x$word_projections, x$dimensions),
        stop("Method not implemented: ", name)
      )
    })

  } else if (name %in% names(x)) {
    # Fall through to regular list element access (e.g., vect$axes,
    # vect$word_projections, vect$dimensions, vect$metadata)
    return(x[[name]])

  } else {
    stop("Vec-tionary has no method or field named '", name, "'")
  }
}


#-- Exported analysis function ----

#' Analyze text with a vectionary
#'
#' @description
#' Analyzes one or more documents and returns scores per dimension.
#' Optionally applies a one-tailed t-test to classify documents as
#' matching each topic dimension.
#'
#' @param vect A vectionary object built with [vectionary_builder()] or loaded
#'   via [readRDS()].
#' @param text Character string or character vector of documents to analyze.
#' @param metric Which aggregation metric to calculate (default: "mean"):
#'   \itemize{
#'     \item \code{"mean"}: Arithmetic mean of word projections — general-purpose summary
#'     \item \code{"rms"}: Root mean square — emphasizes high-magnitude projections
#'     \item \code{"sd"}: Standard deviation — spread of projections within the document
#'     \item \code{"se"}: Standard error of the mean — precision of the mean estimate
#'     \item \code{"top_10"}: Mean of the 10 highest projections — strongest signals only
#'     \item \code{"top_20"}: Mean of the 20 highest projections — strongest signals only
#'     \item \code{"all"}: Returns a list with all six metrics at once
#'   }
#' @param alpha Significance level for one-tailed topic classification (e.g. 0.05).
#'   When set, appends logical `_topic` columns to the result. The test computes
#'   the mean projection per document and flags those exceeding
#'   \eqn{\bar{x} + t_{1-\alpha,\, n-1} \cdot s}, where \eqn{\bar{x}} and
#'   \eqn{s} are the corpus mean and SD of document-level means. Requires
#'   \code{length(text) >= 2}. Default \code{NULL} (no test).
#'
#' @return A named list with one element per dimension. Each element is a
#'   numeric scalar (single document) or numeric vector (multiple documents).
#'   When \code{metric = "all"}, returns a list of such lists (one per metric).
#'   When \code{alpha} is set, the result also contains logical
#'   \code{_topic} elements and a \code{"threshold"} attribute with the
#'   per-dimension cutoffs. Convert to a data frame with
#'   \code{as.data.frame(result)}.
#'
#' @examples
#' \dontrun{
#' my_vect <- readRDS("my_vectionary.rds")
#'
#' # Single document
#' vectionary_analyze(my_vect, "We must protect vulnerable people", metric = "rms")
#'
#' # Multiple documents
#' texts <- c(
#'   "We must protect vulnerable people",
#'   "Justice and equality for all citizens",
#'   "Loyal members stand together"
#' )
#' vectionary_analyze(my_vect, texts, metric = "mean")
#'
#' # Topic classification (one-tailed t-test, alpha = 0.05)
#' vectionary_analyze(my_vect, texts, metric = "mean", alpha = 0.05)
#' }
#'
#' @export
vectionary_analyze <- function(vect, text, metric = "mean", alpha = NULL) {

  if (!is.character(text) || length(text) == 0) {
    stop("text must be a character string or character vector")
  }

  # Validate alpha early (before spending time scoring) so that invalid values
  # error immediately. Also warn if alpha is set with a single document, since
  # the topic test needs a distribution of scores across >= 2 documents.
  if (!is.null(alpha)) {
    if (!is.numeric(alpha) || length(alpha) != 1 || alpha <= 0 || alpha >= 1) {
      stop("alpha must be a single number in (0, 1)")
    }
    if (length(text) < 2) {
      warning("alpha ignored: topic classification requires at least 2 documents")
      alpha <- NULL
    }
  }

  if (!inherits(vect, "Vec-tionary")) {
    stop("vect must be a Vec-tionary object (built with vectionary_builder or loaded via readRDS)")
  }

  metric <- match.arg(metric, c("mean", "rms", "sd", "se", "top_10", "top_20", "all"))

  # Call internal metric functions directly rather than going through the $
  # accessor. This avoids creating a closure on every call. Each .metric()
  # function auto-dispatches: single text -> .get_doc_scores() path, multiple
  # texts -> .batch_metric() path.
  wp <- vect$word_projections
  dims <- vect$dimensions

  result <- switch(metric,
    mean = .mean(text, wp, dims),
    rms = .rms(text, wp, dims),
    sd = .sd(text, wp, dims),
    se = .se(text, wp, dims),
    top_10 = .top_10(text, wp, dims),
    top_20 = .top_20(text, wp, dims),
    all = .metrics(text, wp, dims)
  )

  #-- Topic classification (one-tailed t-test) ----
  # When alpha is set, we classify each document as matching or not matching
  # each dimension. The idea: compute the mean projection score for every
  # document, then treat that distribution of scores as the baseline. Documents
  # in the upper tail (above mean + t_crit * sd) are flagged as topic matches.
  #
  # The test is always based on the mean projection scores, regardless of what
  # `metric` the user requested for the main output. If the user already
  # requested metric="mean", we reuse those scores; otherwise we compute them.

  if (!is.null(alpha)) {

    # Get or compute mean scores for the t-test
    if (metric == "mean") {
      mean_scores <- result
    } else if (metric == "all") {
      mean_scores <- result$mean
    } else {
      mean_scores <- .batch_metric(text, wp, dims, "mean")
    }

    # mean_scores is a named list: one numeric vector per dimension.
    # For each dimension, compute the threshold and classify documents.
    n <- length(mean_scores[[1]])
    thresholds <- setNames(numeric(length(dims)), dims)

    topic_list <- setNames(
      vector("list", length(dims)),
      paste0(dims, "_topic")
    )

    for (dim in dims) {
      vals <- mean_scores[[dim]]

      n_valid <- sum(!is.na(vals))
      if (n_valid < 2) {
        thresholds[[dim]] <- NA_real_
        topic_list[[paste0(dim, "_topic")]] <- rep(FALSE, n)
        next
      }

      mu <- mean(vals, na.rm = TRUE)
      sigma <- stats::sd(vals, na.rm = TRUE)
      t_crit <- stats::qt(1 - alpha, df = n_valid - 1)
      thresholds[[dim]] <- mu + t_crit * sigma
      topic_list[[paste0(dim, "_topic")]] <- !is.na(vals) & vals > thresholds[[dim]]
    }

    # Append topic elements to the result list.
    # For metric="all": add $topic as a named list alongside $mean, $rms, etc.
    # For single metrics: append topic elements directly.
    if (metric == "all") {
      result$topic <- topic_list
    } else {
      result <- c(result, topic_list)
    }

    # Store test parameters as attributes so the user can inspect them
    # (e.g., attr(result, "threshold") to see per-dimension cutoffs)
    attr(result, "threshold") <- thresholds
    attr(result, "alpha") <- alpha
  }

  return(result)
}


#-- Internal helpers: tokenization and lookup ----

#' Tokenize text
#'
#' @description
#' Tokenizes text by splitting on whitespace and converting to lowercase.
#'
#' @param text Character string to tokenize
#'
#' @return Character vector of lowercase tokens
#'
#' @keywords internal
.tokenize <- function(text) {
  # Lowercase first so that "Protect" matches "protect" in word_projections
  text_clean <- tolower(as.character(text))

  # Replace all punctuation characters with spaces. This handles contractions
  # (e.g., "don't" -> "don t"), hyphens ("well-being" -> "well being"), etc.
  text_clean <- gsub("[[:punct:]]", " ", text_clean)

  # Split on any whitespace (spaces, tabs, multiple spaces)
  tokens <- unlist(strsplit(text_clean, "\\s+", perl = TRUE))

  # strsplit can produce empty strings from leading/trailing whitespace
  tokens <- tokens[nchar(tokens) > 0]

  return(tokens)
}


#' Create empty score data frame (internal)
#'
#' @description
#' Returns a 0-row data frame with the correct dimension column names.
#' Used as a consistent return value when a document has no matching tokens.
#'
#' @param dimensions Character vector of dimension names
#' @return Empty data frame with correct column structure
#' @keywords internal
.empty_score_df <- function(dimensions) {
  as.data.frame(matrix(
    nrow = 0,
    ncol = length(dimensions),
    dimnames = list(NULL, dimensions)
  ))
}


#' Get word scores for a single document
#'
#' @description
#' Tokenizes a document and looks up each token's projection scores from
#' word_projections. Returns one row per token occurrence (if a word appears
#' 3 times in the text, its scores appear 3 times in the output).
#'
#' This is the single-document code path. For multiple documents, use
#' .batch_metric() which is more efficient (single match() call for all docs).
#'
#' @param text Character string (single document, not a vector)
#' @param word_projections Data frame with 'word' column and dimension columns.
#'   Each row is a word from the embedding vocabulary with its projection score
#'   onto each dimension axis.
#' @param dimensions Character vector of dimension names (column names in
#'   word_projections to extract)
#'
#' @return Data frame with dimension columns, one row per word occurrence found.
#'   Words not in the vectionary's vocabulary are silently dropped.
#'
#' @keywords internal
.get_doc_scores <- function(text, word_projections, dimensions) {

  if (length(text) != 1) {
    stop("text must be a single character string, not a vector")
  }

  tokens <- .tokenize(text)
  if (length(tokens) == 0) return(.empty_score_df(dimensions))

  if (!is.data.frame(word_projections)) {
    stop("word_projections must be a data frame with 'word' column")
  }

  # Lowercase the vocabulary for case-insensitive matching against tokens
  # (which are already lowercased by .tokenize). This is a safety net in case
  # the builder stored mixed-case words; could be removed if vectionary_builder()
  # guarantees lowercase output.
  wp_words <- tolower(word_projections$word)

  # match() returns the index of the first match in wp_words for each token.
  # Tokens not found in the vocabulary get NA. If a token appears multiple
  # times in the text, it gets looked up multiple times (preserving duplicates).
  matched_indices <- match(tokens, wp_words)

  # Drop tokens not found in the vectionary vocabulary
  found_indices <- matched_indices[!is.na(matched_indices)]
  if (length(found_indices) == 0) return(.empty_score_df(dimensions))

  # Extract the projection scores for each matched token occurrence.
  # This is a data frame subset: rows = matched word indices, cols = dimensions.
  result <- word_projections[found_indices, dimensions, drop = FALSE]
  rownames(result) <- NULL

  return(result)
}


#-- Internal helpers: batch processing pipeline ----

#' Tokenize, match, and group tokens by document (internal)
#'
#' @description
#' Shared pipeline for batch processing of multiple documents. The key
#' optimization is doing a single match() call across all tokens from all
#' documents at once, rather than one match() per document.
#'
#' The pipeline:
#'   1. Tokenize each document, tracking how many tokens each produces
#'   2. Concatenate all tokens into one long vector
#'   3. Run a single match() against the vocabulary
#'   4. Use split() to group matched token indices back by document
#'
#' @param texts Character vector of documents
#' @param word_projections Data frame with 'word' column and dimension columns
#' @param dimensions Character vector of dimension names
#'
#' @return List with:
#'   - `matched_scores`: matrix (n_matched_tokens x n_dims) of projection scores
#'     for every matched token across all documents
#'   - `doc_groups`: named list where each element is a vector of row indices
#'     into matched_scores belonging to that document. Names are document indices
#'     as character strings (e.g., "1", "5", "42"). Documents with zero matches
#'     are absent from this list.
#'   - `n_docs`: total number of input documents
#'   - `n_dims`: number of dimensions
#'
#' @keywords internal
.tokenize_and_match <- function(texts, word_projections, dimensions) {

  n_docs <- length(texts)
  n_dims <- length(dimensions)

  # Lowercase vocabulary for case-insensitive matching (see .get_doc_scores comment)
  wp_words <- tolower(word_projections$word)

  # Convert the projection scores to a matrix for fast row-indexing.
  # Data frame column access is slower than matrix indexing in tight loops.
  score_matrix <- as.matrix(word_projections[, dimensions, drop = FALSE])

  # Tokenize each document and record token counts.
  # Example: 3 documents with 5, 3, 7 tokens -> doc_lengths = c(5, 3, 7)
  all_tokens <- vector("list", n_docs)
  doc_lengths <- integer(n_docs)
  for (i in seq_len(n_docs)) {
    toks <- .tokenize(texts[i])
    all_tokens[[i]] <- toks
    doc_lengths[i] <- length(toks)
  }

  # Flatten all tokens into a single vector: c(doc1_tok1, doc1_tok2, ..., doc2_tok1, ...)
  all_tokens_vec <- unlist(all_tokens, use.names = FALSE)

  # Single match() call for ALL tokens across ALL documents.
  # This is the main performance optimization: match() uses a hash table
  # internally, and one call with N tokens is much faster than N separate calls.
  all_matched <- match(all_tokens_vec, wp_words)

  # Build a parallel vector that maps each token position to its document.
  # E.g., if doc_lengths = c(5, 3, 7), then doc_ids = c(1,1,1,1,1, 2,2,2, 3,3,3,3,3,3,3)
  doc_ids <- rep(seq_len(n_docs), doc_lengths)

  # Keep only matched tokens (drop NAs from match). After this:
  # - matched_indices: which rows in word_projections each token maps to
  # - matched_doc_ids: which document each matched token belongs to
  found <- !is.na(all_matched)
  matched_indices <- all_matched[found]
  matched_doc_ids <- doc_ids[found]

  # Look up projection scores for all matched tokens at once (matrix subset).
  # Result: one row per matched token, one column per dimension.
  matched_scores <- score_matrix[matched_indices, , drop = FALSE]

  # Group the row indices of matched_scores by document.
  # split() returns a named list: names are document indices (as strings),
  # values are integer vectors of row positions in matched_scores.
  # Documents with zero matches don't appear in the list.
  doc_groups <- if (length(matched_doc_ids) > 0) {
    split(seq_along(matched_doc_ids), matched_doc_ids)
  } else {
    list()
  }

  list(
    matched_scores = matched_scores,
    doc_groups = doc_groups,
    n_docs = n_docs,
    n_dims = n_dims
  )
}


#' Batch-compute a metric across multiple documents
#'
#' @description
#' Computes a single metric (mean, rms, sd, se, top_10, or top_20) for each
#' document in a batch. Uses [.tokenize_and_match()] for the shared
#' tokenize-match-group pipeline, then loops over documents and dimensions
#' to compute the requested aggregate.
#'
#' @param texts Character vector of documents
#' @param word_projections Data frame with 'word' column and dimension columns
#' @param dimensions Character vector of dimension names
#' @param metric One of "mean", "rms", "sd", "se", "top_10", "top_20"
#'
#' @return Named list with one element per dimension, each a numeric vector
#'   of length \code{length(texts)}. Documents with no matched words get NA.
#'
#' @keywords internal
.batch_metric <- function(texts, word_projections, dimensions, metric) {

  # Run the shared tokenize-match-group pipeline once
  tm <- .tokenize_and_match(texts, word_projections, dimensions)

  # Pre-allocate result as NA matrix. Documents with no matches stay NA.
  result_mat <- matrix(NA_real_, nrow = tm$n_docs, ncol = tm$n_dims)

  # Iterate only over documents that had at least one vocabulary match.
  # doc_groups names are document indices as strings (e.g., "1", "42").
  for (doc_str in names(tm$doc_groups)) {
    doc_i <- as.integer(doc_str)
    rows <- tm$doc_groups[[doc_str]]  # row indices into tm$matched_scores
    n <- length(rows)                 # number of matched words in this document

    for (j in seq_len(tm$n_dims)) {
      # vals = projection scores for this document's matched words on dimension j
      vals <- tm$matched_scores[rows, j]

      # Compute the requested metric.
      # SD and SE use sample standard deviation (Bessel-corrected, n-1 denominator)
      # and return NA when n <= 1 (can't estimate variability from one observation).
      result_mat[doc_i, j] <- switch(metric,
        mean = sum(vals) / n,
        rms = sqrt(sum(vals^2) / n),
        sd = {
          if (n <= 1L) NA_real_
          else {
            m <- sum(vals) / n
            sqrt(sum((vals - m)^2) / (n - 1))
          }
        },
        se = {
          if (n <= 1L) NA_real_
          else {
            m <- sum(vals) / n
            sqrt(sum((vals - m)^2) / (n - 1)) / sqrt(n)
          }
        },
        top_10 = {
          # Sort descending, take up to 10 highest, average them.
          # If doc has < 10 words, uses all of them (head handles short vectors).
          top_vals <- head(sort(vals, decreasing = TRUE), 10L)
          sum(top_vals) / length(top_vals)
        },
        top_20 = {
          top_vals <- head(sort(vals, decreasing = TRUE), 20L)
          sum(top_vals) / length(top_vals)
        }
      )
    }
  }

  result <- setNames(
    lapply(seq_len(ncol(result_mat)), function(j) result_mat[, j]),
    dimensions
  )
  return(result)
}


#' Batch-compute all metrics across multiple documents
#'
#' @description
#' Computes all 6 metrics (mean, rms, sd, se, top_10, top_20) in a single pass
#' over the tokenized data. More efficient than calling .batch_metric() 6 times
#' because tokenization and matching happen only once via [.tokenize_and_match()].
#'
#' @param texts Character vector of documents
#' @param word_projections Data frame with 'word' column and dimension columns
#' @param dimensions Character vector of dimension names
#'
#' @return Named list of 6 named lists (one per metric). Each inner list has
#'   one element per dimension, each a numeric vector of length \code{length(texts)}.
#'
#' @keywords internal
.batch_metrics_all <- function(texts, word_projections, dimensions) {

  tm <- .tokenize_and_match(texts, word_projections, dimensions)
  metric_names <- c("mean", "rms", "sd", "se", "top_10", "top_20")

  # One pre-allocated NA matrix per metric
  result_mats <- lapply(metric_names, function(m) matrix(NA_real_, nrow = tm$n_docs, ncol = tm$n_dims))
  names(result_mats) <- metric_names

  for (doc_str in names(tm$doc_groups)) {
    doc_i <- as.integer(doc_str)
    rows <- tm$doc_groups[[doc_str]]
    n <- length(rows)

    for (j in seq_len(tm$n_dims)) {
      vals <- tm$matched_scores[rows, j]

      # Pre-compute mean (used by multiple metrics below)
      s <- sum(vals)
      m <- s / n

      result_mats$mean[doc_i, j] <- m
      result_mats$rms[doc_i, j] <- sqrt(sum(vals^2) / n)

      # Sample SD and SE (n-1 denominator). NA for n <= 1.
      if (n <= 1L) {
        result_mats$sd[doc_i, j] <- NA_real_
        result_mats$se[doc_i, j] <- NA_real_
      } else {
        samp_sd <- sqrt(sum((vals - m)^2) / (n - 1))
        result_mats$sd[doc_i, j] <- samp_sd
        result_mats$se[doc_i, j] <- samp_sd / sqrt(n)
      }

      # Top-k: sort descending, take up to k highest, average
      top10 <- head(sort(vals, decreasing = TRUE), 10L)
      result_mats$top_10[doc_i, j] <- sum(top10) / length(top10)

      top20 <- head(sort(vals, decreasing = TRUE), 20L)
      result_mats$top_20[doc_i, j] <- sum(top20) / length(top20)
    }
  }

  # Convert each matrix to a named list of dimension vectors
  result <- lapply(result_mats, function(mat) {
    setNames(
      lapply(seq_len(ncol(mat)), function(j) mat[, j]),
      dimensions
    )
  })

  return(result)
}


#-- Per-metric wrapper functions ----
#
# Each function below handles both single-doc and multi-doc inputs:
#   - length(text) == 1  -> single-document path via .get_doc_scores()
#   - length(text) > 1   -> batch path via .batch_metric()
#
# Both paths return a named list (one element per dimension).
# Single-doc: list(dim1 = value, dim2 = value, ...)
# Multi-doc:  list(dim1 = c(val1, val2, ...), dim2 = c(val1, val2, ...), ...)

#' Compute mean metric (single document or vector)
#'
#' @param text Character string or vector of strings
#' @param word_projections Data frame with word scores
#' @param dimensions Character vector of dimension names
#'
#' @return Named list with one element per dimension.
#' @keywords internal
.mean <- function(text, word_projections, dimensions) {
  if (length(text) > 1) {
    return(.batch_metric(text, word_projections, dimensions, "mean"))
  }
  doc_scores <- .get_doc_scores(text, word_projections, dimensions)
  if (nrow(doc_scores) == 0) return(setNames(as.list(rep(NA_real_, length(dimensions))), dimensions))
  # Average projection across all matched words for each dimension
  setNames(lapply(dimensions, function(dim) mean(doc_scores[[dim]], na.rm = TRUE)), dimensions)
}


#' Compute RMS metric (single document or vector)
#'
#' @description
#' Root Mean Square: sqrt(mean(x^2)). Unlike the arithmetic mean, RMS is
#' always non-negative and gives more weight to high-magnitude projections
#' (both positive and negative). Useful when both positive and negative
#' projections are meaningful signals.
#'
#' @param text Character string or vector of strings
#' @param word_projections Data frame with word scores
#' @param dimensions Character vector of dimension names
#' @return Named list with one element per dimension.
#' @keywords internal
.rms <- function(text, word_projections, dimensions) {
  if (length(text) > 1) {
    return(.batch_metric(text, word_projections, dimensions, "rms"))
  }
  doc_scores <- .get_doc_scores(text, word_projections, dimensions)
  if (nrow(doc_scores) == 0) return(setNames(as.list(rep(NA_real_, length(dimensions))), dimensions))
  setNames(lapply(dimensions, function(dim) sqrt(mean(doc_scores[[dim]]^2, na.rm = TRUE))), dimensions)
}


#' Compute SD metric (single document or vector)
#'
#' @description
#' Sample standard deviation of a document's word projections (Bessel-corrected,
#' n-1 denominator). Measures how spread out the projections are within a
#' single document. Returns NA for documents with 0 or 1 matched words.
#'
#' @param text Character string or vector of strings
#' @param word_projections Data frame with word scores
#' @param dimensions Character vector of dimension names
#' @return Named list with one element per dimension.
#' @keywords internal
.sd <- function(text, word_projections, dimensions) {
  if (length(text) > 1) {
    return(.batch_metric(text, word_projections, dimensions, "sd"))
  }
  doc_scores <- .get_doc_scores(text, word_projections, dimensions)
  if (nrow(doc_scores) == 0) return(setNames(as.list(rep(NA_real_, length(dimensions))), dimensions))
  setNames(lapply(dimensions, function(dim) {
    vals <- doc_scores[[dim]]
    n <- length(vals)
    if (n <= 1) return(NA_real_)
    sqrt(sum((vals - mean(vals))^2) / (n - 1))
  }), dimensions)
}


#' Compute SE metric (single document or vector)
#'
#' @description
#' Standard error of the mean: sample SD / sqrt(n). Estimates how precisely
#' the document's mean projection is known. Smaller SE (more words, less
#' variance) means the mean estimate is more reliable.
#'
#' @param text Character string or vector of strings
#' @param word_projections Data frame with word scores
#' @param dimensions Character vector of dimension names
#' @return Named list with one element per dimension.
#' @keywords internal
.se <- function(text, word_projections, dimensions) {
  if (length(text) > 1) {
    return(.batch_metric(text, word_projections, dimensions, "se"))
  }
  doc_scores <- .get_doc_scores(text, word_projections, dimensions)
  if (nrow(doc_scores) == 0) return(setNames(as.list(rep(NA_real_, length(dimensions))), dimensions))
  setNames(lapply(dimensions, function(dim) {
    vals <- doc_scores[[dim]]
    n <- length(vals)
    if (n <= 1) return(NA_real_)
    sqrt(sum((vals - mean(vals))^2) / (n - 1)) / sqrt(n)
  }), dimensions)
}


#' Compute top-10 metric (single document or vector)
#'
#' @description
#' Mean of the 10 highest projection scores in the document. Captures the
#' strength of the strongest semantic signals, ignoring noise from low-scoring
#' words. If the document has fewer than 10 matched words, averages all of them.
#'
#' @param text Character string or vector of strings
#' @param word_projections Data frame with word scores
#' @param dimensions Character vector of dimension names
#' @return Named list with one element per dimension.
#' @keywords internal
.top_10 <- function(text, word_projections, dimensions) {
  if (length(text) > 1) {
    return(.batch_metric(text, word_projections, dimensions, "top_10"))
  }
  doc_scores <- .get_doc_scores(text, word_projections, dimensions)
  if (nrow(doc_scores) == 0) return(setNames(as.list(rep(NA_real_, length(dimensions))), dimensions))
  setNames(lapply(dimensions, function(dim) {
    top_vals <- head(sort(doc_scores[[dim]], decreasing = TRUE), 10)
    mean(top_vals, na.rm = TRUE)
  }), dimensions)
}


#' Compute top-20 metric (single document or vector)
#'
#' @description
#' Mean of the 20 highest projection scores. Broader than top_10: captures
#' moderate signals alongside the strongest ones.
#'
#' @param text Character string or vector of strings
#' @param word_projections Data frame with word scores
#' @param dimensions Character vector of dimension names
#' @return Named list with one element per dimension.
#' @keywords internal
.top_20 <- function(text, word_projections, dimensions) {
  if (length(text) > 1) {
    return(.batch_metric(text, word_projections, dimensions, "top_20"))
  }
  doc_scores <- .get_doc_scores(text, word_projections, dimensions)
  if (nrow(doc_scores) == 0) return(setNames(as.list(rep(NA_real_, length(dimensions))), dimensions))
  setNames(lapply(dimensions, function(dim) {
    top_vals <- head(sort(doc_scores[[dim]], decreasing = TRUE), 20)
    mean(top_vals, na.rm = TRUE)
  }), dimensions)
}


#' Compute all metrics (single document or vector)
#'
#' @description
#' Computes all 6 metrics at once. For a single document, tokenizes and looks
#' up scores once, then computes each metric from the same scores. For multiple
#' documents, delegates to .batch_metrics_all() which does a single
#' tokenize-match pass.
#'
#' @param text Character string or vector of strings
#' @param word_projections Data frame with word scores
#' @param dimensions Character vector of dimension names
#' @return Named list of 6 named lists (one per metric).
#' @keywords internal
.metrics <- function(text, word_projections, dimensions) {
  if (length(text) > 1) {
    return(.batch_metrics_all(text, word_projections, dimensions))
  }

  # Single document: get all word scores once, compute each metric from them
  doc_scores <- .get_doc_scores(text, word_projections, dimensions)

  if (nrow(doc_scores) == 0) {
    nas <- setNames(as.list(rep(NA_real_, length(dimensions))), dimensions)
    return(list(mean = nas, rms = nas, sd = nas, se = nas, top_10 = nas, top_20 = nas))
  }

  make_list <- function(fn) setNames(lapply(dimensions, fn), dimensions)

  list(
    mean = make_list(function(dim) mean(doc_scores[[dim]], na.rm = TRUE)),
    rms = make_list(function(dim) sqrt(mean(doc_scores[[dim]]^2, na.rm = TRUE))),
    sd = make_list(function(dim) {
      vals <- doc_scores[[dim]]; n <- length(vals)
      if (n <= 1) NA_real_ else sqrt(sum((vals - mean(vals))^2) / (n - 1))
    }),
    se = make_list(function(dim) {
      vals <- doc_scores[[dim]]; n <- length(vals)
      if (n <= 1) NA_real_ else sqrt(sum((vals - mean(vals))^2) / (n - 1)) / sqrt(n)
    }),
    top_10 = make_list(function(dim) {
      top_vals <- head(sort(doc_scores[[dim]], decreasing = TRUE), 10); mean(top_vals)
    }),
    top_20 = make_list(function(dim) {
      top_vals <- head(sort(doc_scores[[dim]], decreasing = TRUE), 20); mean(top_vals)
    })
  )
}


#-- Diagnostics ----

#' Diagnose a vec-tionary
#'
#' @description
#' Prints a diagnostic report for a vec-tionary, showing the top-scoring words
#' per dimension and whether the seed (dictionary) words rank near the top.
#' Useful for verifying that a vectionary is capturing the intended semantics.
#'
#' @param vectionary A vectionary object (from [vectionary_builder()])
#' @param n Number of top words to show per dimension (default: 30)
#' @param dimension Which dimension(s) to diagnose. If NULL (default), diagnoses
#'   all dimensions. Can be a character vector of dimension names or a single name.
#'
#' @return Invisibly returns a list of data frames (one per dimension) with columns:
#'   \code{rank}, \code{word}, \code{score}, \code{seed}.
#'
#' @examples
#' \dontrun{
#' vect <- vectionary_builder(dictionary, embeddings)
#' vectionary_diagnose(vect)
#' vectionary_diagnose(vect, n = 50, dimension = "care")
#' }
#'
#' @export
vectionary_diagnose <- function(vectionary, n = 30, dimension = NULL) {

  if (!inherits(vectionary, "Vec-tionary")) {
    stop("vectionary must be a Vec-tionary object (from vectionary_builder())")
  }

  wp <- vectionary$word_projections
  dims <- vectionary$dimensions

  # Allow diagnosing a subset of dimensions
  if (!is.null(dimension)) {
    bad <- setdiff(dimension, dims)
    if (length(bad) > 0) {
      stop("Unknown dimension(s): ", paste(bad, collapse = ", "),
           ". Available: ", paste(dims, collapse = ", "))
    }
    dims <- dimension
  }

  # Seed words are the original dictionary words used to train the vectionary.
  # Per-dimension lists (seed_words_per_dim) are available since v0.9.8;
  # older objects only have the flat seed_words vector.
  seed_per_dim <- vectionary$metadata$seed_words_per_dim
  seed_all <- vectionary$metadata$seed_words
  if (is.null(seed_per_dim) && is.null(seed_all)) {
    seed_all <- character(0)
    cli::cli_alert_warning("No seed words stored in this vectionary (built with older version?)")
  }

  results <- list()

  for (dim in dims) {
    # Use per-dimension seed words when available, otherwise fall back to flat list
    seed_words <- if (!is.null(seed_per_dim) && dim %in% names(seed_per_dim)) {
      seed_per_dim[[dim]]
    } else {
      seed_all
    }
    if (is.null(seed_words)) seed_words <- character(0)

    # Sort all words by their projection score on this dimension (highest first)
    scores <- wp[[dim]]
    ord <- order(scores, decreasing = TRUE)
    wp_sorted <- data.frame(
      rank = seq_along(ord),
      word = wp$word[ord],
      score = round(scores[ord], 4),
      seed = wp$word[ord] %in% seed_words,
      stringsAsFactors = FALSE
    )

    top_n <- wp_sorted[1:min(n, nrow(wp_sorted)), ]

    # Count how many seed words appear in the top N — a healthy vectionary
    # should have most seed words ranking high
    seed_rows <- wp_sorted[wp_sorted$seed, ]
    n_seed <- nrow(seed_rows)
    n_seed_in_top <- sum(top_n$seed)

    cli::cli_h1("Dimension: {dim}")

    if (n_seed > 0) {
      median_rank <- stats::median(seed_rows$rank)
      best_rank <- min(seed_rows$rank)
      worst_rank <- max(seed_rows$rank)
      pct_in_top <- round(100 * n_seed_in_top / n_seed, 1)

      cli::cli_alert_info("{n_seed_in_top}/{n_seed} seed words in top {n} ({pct_in_top}%)")
      cli::cli_alert_info("Seed word ranks: best = {best_rank}, median = {median_rank}, worst = {worst_rank}")
      cli::cli_alert_info("Seed word scores: max = {round(max(seed_rows$score), 4)}, median = {round(stats::median(seed_rows$score), 4)}, min = {round(min(seed_rows$score), 4)}")
    }

    cli::cli_h2("Top {min(n, nrow(wp_sorted))} words")

    # Mark seed words with an asterisk for easy visual identification
    display <- top_n
    display$word <- ifelse(display$seed, paste0(display$word, " *"), display$word)
    display$seed <- NULL
    print(display, row.names = FALSE)

    # Show seed words that didn't make it into the top N — these might
    # indicate dimension learning issues or unusual seed words
    if (n_seed > 0) {
      missed_seeds <- seed_rows[seed_rows$rank > n, ]
      if (nrow(missed_seeds) > 0) {
        cli::cli_h2("Seed words outside top {n}")
        missed_display <- missed_seeds
        missed_display$seed <- NULL
        print(missed_display, row.names = FALSE)
      }
    }

    cli::cli_text("")
    results[[dim]] <- wp_sorted
  }

  invisible(results)
}
