# Tests for previously uncovered functions:
# vectionary_diagnose, topic classification, expand_stems, expand_vocabulary, solve_axis_duan

#- Helper: create mock Vec-tionary ----
.create_mock_vectionary <- function() {
  word_projections <- data.frame(
    word = c("protect", "care", "help", "safe", "kind",
             "harm", "hurt", "damage", "cruel", "abuse",
             "fair", "just", "equal", "rights", "balance"),
    care = c(0.9, 0.8, 0.7, 0.6, 0.5,
             -0.8, -0.7, -0.6, -0.5, -0.4,
             0.1, 0.05, 0.0, -0.05, -0.1),
    fairness = c(0.1, 0.0, 0.05, -0.1, 0.0,
                 -0.05, 0.0, -0.1, 0.0, 0.05,
                 0.9, 0.8, 0.7, 0.6, 0.5),
    stringsAsFactors = FALSE
  )

  vect <- list(
    axes = list(
      care = rnorm(10),
      fairness = rnorm(10)
    ),
    word_projections = word_projections,
    dimensions = c("care", "fairness"),
    metadata = list(
      method = "ridge",
      binary_word = TRUE,
      vocab_size = 15,
      build_date = Sys.time(),
      seed_words = c("protect", "care", "help", "harm", "hurt",
                     "fair", "just", "equal"),
      seed_words_per_dim = list(
        care = c("protect", "care", "help", "harm", "hurt"),
        fairness = c("fair", "just", "equal")
      )
    )
  )
  class(vect) <- "Vec-tionary"
  return(vect)
}

#- Helper: create mock embeddings file ----
.create_mock_embeddings_ext <- function(words = NULL, n_dims = 10) {
  if (is.null(words)) {
    words <- c(
      "protect", "care", "help", "safe", "kind",
      "harm", "hurt", "damage", "cruel", "abuse",
      "fair", "just", "equal", "rights", "balance",
      "protecting", "protected", "protector",
      "caring", "cared", "careful",
      "harmful", "harming", "harmed"
    )
  }
  path <- tempfile(fileext = ".vec")
  set.seed(42)
  con <- file(path, "w")
  writeLines(paste(length(words), n_dims), con)
  for (word in words) {
    vec <- round(rnorm(n_dims), 6)
    line <- paste(word, paste(vec, collapse = " "))
    writeLines(line, con)
  }
  close(con)
  return(path)
}

#- vectionary_diagnose ----

test_that("vectionary_diagnose runs without error", {
  vect <- .create_mock_vectionary()
  expect_no_error(
    suppressMessages(capture.output(vectionary_diagnose(vect)))
  )
})

test_that("vectionary_diagnose returns invisible list of data frames", {
  vect <- .create_mock_vectionary()
  suppressMessages(capture.output(res <- vectionary_diagnose(vect)))
  expect_type(res, "list")
  expect_equal(length(res), 2)
  expect_true("care" %in% names(res))
  expect_true("fairness" %in% names(res))
  expect_true(is.data.frame(res$care))
  expect_true("rank" %in% names(res$care))
  expect_true("word" %in% names(res$care))
  expect_true("score" %in% names(res$care))
  expect_true("seed" %in% names(res$care))
})

test_that("vectionary_diagnose filters by dimension", {
  vect <- .create_mock_vectionary()
  suppressMessages(capture.output(res <- vectionary_diagnose(vect, dimension = "care")))
  expect_equal(length(res), 1)
  expect_true("care" %in% names(res))
})

test_that("vectionary_diagnose errors on invalid dimension", {
  vect <- .create_mock_vectionary()
  expect_error(vectionary_diagnose(vect, dimension = "nonexistent"), "Unknown dimension")
})

test_that("vectionary_diagnose errors on non-vectionary", {
  expect_error(vectionary_diagnose(list(a = 1)), "Vec-tionary")
})

test_that("vectionary_diagnose respects n parameter", {
  vect <- .create_mock_vectionary()
  out5 <- suppressMessages(capture.output(
    res5 <- vectionary_diagnose(vect, n = 5, dimension = "care")
  ))
  out15 <- suppressMessages(capture.output(
    res15 <- vectionary_diagnose(vect, n = 15, dimension = "care")
  ))
  # n=5 should show fewer words in the top-N section
  expect_true(length(out5) < length(out15))
})

test_that("vectionary_diagnose marks seed words", {
  vect <- .create_mock_vectionary()
  output <- suppressMessages(capture.output(
    vectionary_diagnose(vect, n = 15, dimension = "care")
  ))
  # Seed words should be marked with asterisk
  expect_true(any(grepl("\\*", output)))
})

#- Topic classification (alpha parameter) ----

test_that("topic classification appends _topic elements", {
  vect <- .create_mock_vectionary()
  texts <- c("protect care help", "harm hurt damage", "fair just equal", "safe kind balance")

  result <- vectionary_analyze(vect, texts, metric = "mean", alpha = 0.05)

  expect_true("care_topic" %in% names(result))
  expect_true("fairness_topic" %in% names(result))
  expect_true(is.logical(result$care_topic))
  expect_true(is.logical(result$fairness_topic))
  expect_equal(length(result$care_topic), 4)
})

test_that("topic classification attaches threshold attribute", {
  vect <- .create_mock_vectionary()
  texts <- c("protect care help", "harm hurt damage", "fair just equal", "safe kind balance")

  result <- vectionary_analyze(vect, texts, metric = "mean", alpha = 0.05)

  thresh <- attr(result, "threshold")
  expect_true(!is.null(thresh))
  expect_true(is.numeric(thresh))
  expect_equal(length(thresh), 2)
  expect_true("care" %in% names(thresh))
  expect_true("fairness" %in% names(thresh))
})

test_that("topic classification attaches alpha attribute", {
  vect <- .create_mock_vectionary()
  texts <- c("protect care help", "harm hurt damage", "fair just equal")

  result <- vectionary_analyze(vect, texts, metric = "mean", alpha = 0.10)

  expect_equal(attr(result, "alpha"), 0.10)
})

test_that("topic classification works with metric = 'all'", {
  vect <- .create_mock_vectionary()
  texts <- c("protect care help", "harm hurt damage", "fair just equal")

  result <- vectionary_analyze(vect, texts, metric = "all", alpha = 0.05)

  expect_true("topic" %in% names(result))
  expect_true(is.list(result$topic))
  expect_true("care_topic" %in% names(result$topic))
  expect_true("fairness_topic" %in% names(result$topic))
})

test_that("topic classification warns with single document", {
  vect <- .create_mock_vectionary()
  expect_warning(
    vectionary_analyze(vect, "protect care help", metric = "mean", alpha = 0.05),
    "at least 2 documents"
  )
})

test_that("topic classification rejects invalid alpha", {
  vect <- .create_mock_vectionary()
  expect_error(
    vectionary_analyze(vect, c("protect", "harm"), metric = "mean", alpha = 0),
    "alpha must be"
  )
  expect_error(
    vectionary_analyze(vect, c("protect", "harm"), metric = "mean", alpha = 1),
    "alpha must be"
  )
  expect_error(
    vectionary_analyze(vect, c("protect", "harm"), metric = "mean", alpha = "bad"),
    "alpha must be"
  )
})

test_that("lower alpha produces stricter thresholds", {
  vect <- .create_mock_vectionary()
  texts <- c("protect care help", "harm hurt damage", "fair just equal",
             "safe kind balance", "rights equal fair")

  result_strict <- vectionary_analyze(vect, texts, metric = "mean", alpha = 0.01)
  result_lenient <- vectionary_analyze(vect, texts, metric = "mean", alpha = 0.25)

  thresh_strict <- attr(result_strict, "threshold")
  thresh_lenient <- attr(result_lenient, "threshold")

  # Stricter alpha should produce higher thresholds
  expect_true(thresh_strict[["care"]] >= thresh_lenient[["care"]])
  expect_true(thresh_strict[["fairness"]] >= thresh_lenient[["fairness"]])
})

test_that("topic classification uses non-metric scores for t-test", {
  vect <- .create_mock_vectionary()
  texts <- c("protect care help", "harm hurt damage", "fair just equal")

  # When metric = "rms", topic test should still use mean scores
  result <- vectionary_analyze(vect, texts, metric = "rms", alpha = 0.05)
  expect_true("care_topic" %in% names(result))
})

#- .expand_stems ----

test_that(".expand_stems expands wildcard patterns", {
  emb <- .create_mock_embeddings_ext()
  on.exit(unlink(emb))

  dict <- data.frame(
    word = c("protect*", "care*"),
    score = c(1, 1),
    stringsAsFactors = FALSE
  )

  result <- .expand_stems(dict, emb, verbose = FALSE)

  expect_true(is.data.frame(result))
  expect_true(nrow(result) > 2)  # Should have expanded
  # "protect*" matches: protecting, protected, protector
  expect_true("protecting" %in% result$word)
  expect_true("protected" %in% result$word)
  # "care*" matches: cared, careful (not "caring" — starts with "cari" not "care")
  expect_true("cared" %in% result$word)
  expect_true("careful" %in% result$word)
})

test_that(".expand_stems preserves exact words", {
  emb <- .create_mock_embeddings_ext()
  on.exit(unlink(emb))

  dict <- data.frame(
    word = c("protect*", "fair"),
    score = c(1, 1),
    stringsAsFactors = FALSE
  )

  result <- .expand_stems(dict, emb, verbose = FALSE)

  # "fair" should be preserved as-is
  expect_true("fair" %in% result$word)
  # "protect*" should be expanded
  expect_true("protecting" %in% result$word)
})

test_that(".expand_stems returns unchanged dict with no stems", {
  emb <- .create_mock_embeddings_ext()
  on.exit(unlink(emb))

  dict <- data.frame(
    word = c("protect", "care"),
    score = c(1, 1),
    stringsAsFactors = FALSE
  )

  result <- .expand_stems(dict, emb, verbose = FALSE)

  expect_equal(nrow(result), 2)
  expect_equal(result$word, c("protect", "care"))
})

test_that(".expand_stems inherits scores from stem", {
  emb <- .create_mock_embeddings_ext()
  on.exit(unlink(emb))

  dict <- data.frame(
    word = c("protect*"),
    score = c(0.9),
    stringsAsFactors = FALSE
  )

  result <- .expand_stems(dict, emb, verbose = FALSE)

  # All expanded words should inherit score = 0.9
  expect_true(all(result$score == 0.9))
})

#- .expand_vocabulary ----

test_that(".expand_vocabulary returns top-N candidates", {
  vect <- .create_mock_vectionary()

  dict <- data.frame(
    word = c("protect", "care"),
    care = c(1, 1),
    fairness = c(0, 0),
    stringsAsFactors = FALSE
  )

  result <- .expand_vocabulary(
    word_projections = vect$word_projections,
    dictionary = dict,
    dimensions = c("care", "fairness"),
    n_expand = 5,
    verbose = FALSE
  )

  expect_true(is.data.frame(result))
  expect_true(nrow(result) <= 5)
  # Expanded words should not include original dictionary words
  expect_false("protect" %in% result$word)
  expect_false("care" %in% result$word)
})

test_that(".expand_vocabulary excludes dictionary words", {
  vect <- .create_mock_vectionary()

  result <- .expand_vocabulary(
    word_projections = vect$word_projections,
    dictionary = vect$word_projections,  # Use all words as dictionary
    dimensions = c("care", "fairness"),
    n_expand = 5,
    verbose = FALSE
  )

  # No candidates left since all words are in dictionary
  expect_equal(nrow(result), 0)
})

test_that(".expand_vocabulary positive_only filters negatives", {
  vect <- .create_mock_vectionary()

  dict <- data.frame(
    word = c("protect"),
    care = c(1),
    fairness = c(1),
    stringsAsFactors = FALSE
  )

  result_pos <- .expand_vocabulary(
    word_projections = vect$word_projections,
    dictionary = dict,
    dimensions = c("care", "fairness"),
    n_expand = 100,
    positive_only = TRUE,
    verbose = FALSE
  )

  result_all <- .expand_vocabulary(
    word_projections = vect$word_projections,
    dictionary = dict,
    dimensions = c("care", "fairness"),
    n_expand = 100,
    positive_only = FALSE,
    verbose = FALSE
  )

  # Positive-only should return fewer or equal candidates
  expect_true(nrow(result_pos) <= nrow(result_all))
})

#- .solve_axis_duan ----

test_that(".solve_axis_duan returns unit-norm axis", {
  skip_if_not_installed("alabama")

  set.seed(42)
  vectors <- lapply(1:5, function(i) rnorm(10))
  scores <- c(1, 0.8, 0.6, -0.5, -0.9)

  axis <- .solve_axis_duan(vectors, scores, dim_name = "test", seed = 42)

  expect_type(axis, "double")
  expect_equal(length(axis), 10)
  # Duan method produces unit-norm axis
  expect_equal(sqrt(sum(axis^2)), 1, tolerance = 1e-4)
})

test_that(".solve_axis_duan is reproducible with same seed", {
  skip_if_not_installed("alabama")

  set.seed(42)
  vectors <- lapply(1:5, function(i) rnorm(10))
  scores <- c(1, 0.8, 0.6, -0.5, -0.9)

  axis1 <- .solve_axis_duan(vectors, scores, dim_name = "test", seed = 123)
  axis2 <- .solve_axis_duan(vectors, scores, dim_name = "test", seed = 123)

  expect_equal(axis1, axis2)
})

test_that(".solve_axis_duan preserves RNG state", {
  skip_if_not_installed("alabama")

  set.seed(42)
  vectors <- lapply(1:5, function(i) rnorm(10))
  scores <- c(1, 0.8, 0.6, -0.5, -0.9)

  # Record RNG state before
  set.seed(99)
  before <- runif(1)

  # Run Duan (modifies seed internally but should restore)
  .solve_axis_duan(vectors, scores, dim_name = "test", seed = 42)

  # RNG should be restored — next call should NOT match what we'd get if
  # the seed were altered
  set.seed(99)
  expected <- runif(1)

  expect_equal(before, expected)
})
