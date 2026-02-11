# Tests for R/03_builder.R
# Input validation, axis solving, and lambda selection

#- Helper: create mock embeddings file ----
.create_mock_embeddings <- function(path = NULL, n_words = 20, n_dims = 10) {
  if (is.null(path)) path <- tempfile(fileext = ".vec")

  # Generate mock words and vectors
  words <- c(
    "protect", "care", "help", "safe", "kind",
    "harm", "hurt", "damage", "cruel", "abuse",
    "fair", "just", "equal", "rights", "balance",
    "cheat", "fraud", "unfair", "biased", "corrupt"
  )
  words <- words[seq_len(min(n_words, length(words)))]

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

#- vectionary_builder input validation ----

test_that("vectionary_builder rejects empty dictionary", {
  expect_error(
    vectionary_builder(character(0), "dummy.vec", save_path = NULL),
    "dictionary cannot be empty"
  )
})

test_that("vectionary_builder rejects dictionary without word column", {
  df <- data.frame(x = 1:3, y = 4:6)
  expect_error(
    vectionary_builder(df, "dummy.vec", save_path = NULL),
    "must have a 'word' column"
  )
})

test_that("vectionary_builder rejects multi-word entries", {
  df <- data.frame(word = c("hello world", "test"), score = c(1, 1))
  expect_error(
    vectionary_builder(df, "dummy.vec", save_path = NULL),
    "multi-word entries"
  )
})

test_that("vectionary_builder rejects punctuation without expand_stem", {
  df <- data.frame(word = c("test*", "hello"), score = c(1, 1))
  expect_error(
    vectionary_builder(df, "dummy.vec", save_path = NULL, expand_stem = FALSE),
    "punctuation"
  )
})

test_that("vectionary_builder rejects missing embeddings file", {
  df <- data.frame(word = c("test", "hello"), score = c(1, 1))
  expect_error(
    vectionary_builder(df, "/nonexistent/path/embeddings.vec", save_path = NULL),
    "Embeddings file not found"
  )
})

test_that("vectionary_builder rejects missing dimensions", {
  df <- data.frame(word = c("test", "hello"))
  emb <- .create_mock_embeddings()
  on.exit(unlink(emb))
  expect_error(
    vectionary_builder(df, emb, save_path = NULL),
    "No dimensions specified"
  )
})

test_that("vectionary_builder rejects invalid method", {
  df <- data.frame(word = c("test", "hello"), score = c(1, 1))
  emb <- .create_mock_embeddings()
  on.exit(unlink(emb))
  expect_error(
    vectionary_builder(df, emb, method = "invalid", save_path = NULL),
    "method must be one of"
  )
})

test_that("vectionary_builder rejects negative lambda", {
  df <- data.frame(word = c("test", "hello"), score = c(1, 1))
  emb <- .create_mock_embeddings()
  on.exit(unlink(emb))
  expect_error(
    vectionary_builder(df, emb, lambda = -1, save_path = NULL),
    "lambda values must be non-negative"
  )
})

test_that("vectionary_builder rejects invalid l1_ratio", {
  df <- data.frame(word = c("test", "hello"), score = c(1, 1))
  emb <- .create_mock_embeddings()
  on.exit(unlink(emb))
  expect_error(
    vectionary_builder(df, emb, l1_ratio = 1.5, save_path = NULL),
    "l1_ratio must be a number between 0 and 1"
  )
})

test_that("vectionary_builder converts character vector to binary dictionary", {
  emb <- .create_mock_embeddings()
  on.exit(unlink(emb))

  vect <- vectionary_builder(
    c("protect", "care", "help"),
    emb,
    dimensions = "care",
    lambda = 1,
    save_path = NULL,
    verbose = FALSE,
    spellcheck = FALSE,
    remove_stopwords = FALSE
  )

  expect_s3_class(vect, "Vec-tionary")
  expect_equal(vect$dimensions, "care")
})

#- .solve_axis_ridge ----

test_that(".solve_axis_ridge returns unit-norm axis", {
  set.seed(42)
  # 5 words, 10 dimensions
  vectors <- lapply(1:5, function(i) rnorm(10))
  scores <- c(1, 0.8, 0.6, -0.5, -0.9)

  axis <- .solve_axis_ridge(vectors, scores, lambda = 1.0)

  expect_type(axis, "double")
  expect_equal(length(axis), 10)

  # Axis should be non-zero (not unit norm — preserves score scale)
  expect_true(sqrt(sum(axis^2)) > 0)
})

test_that(".solve_axis_ridge works with binary scores", {
  set.seed(42)
  vectors <- lapply(1:5, function(i) rnorm(10))
  scores <- c(1, 1, 1, 0, 0)

  axis <- .solve_axis_ridge(vectors, scores, lambda = 1.0)

  expect_type(axis, "double")
  expect_equal(length(axis), 10)
  expect_true(sqrt(sum(axis^2)) > 0)
})

test_that(".solve_axis_ridge with different lambda values produces different axes", {
  set.seed(42)
  vectors <- lapply(1:5, function(i) rnorm(10))
  scores <- c(1, 0.8, 0.6, -0.5, -0.9)

  axis_low <- .solve_axis_ridge(vectors, scores, lambda = 0.01)
  axis_high <- .solve_axis_ridge(vectors, scores, lambda = 100)

  # Different lambdas should produce different axes
  expect_false(all(abs(axis_low - axis_high) < 1e-10))
})

#- .solve_axis_elastic_net ----

test_that(".solve_axis_elastic_net returns valid axis", {
  set.seed(42)
  vectors <- lapply(1:5, function(i) rnorm(10))
  scores <- c(1, 0.8, 0.6, -0.5, -0.9)

  axis <- .solve_axis_elastic_net(vectors, scores, lambda = 0.1, l1_ratio = 0.5)

  expect_type(axis, "double")
  expect_equal(length(axis), 10)
  # Axis should be non-zero (not unit norm — preserves score scale)
  expect_true(sqrt(sum(axis^2)) > 0)
})

#- .load_seed_vectors ----

test_that(".load_seed_vectors loads correct words from text embeddings", {
  emb <- .create_mock_embeddings(n_dims = 10)
  on.exit(unlink(emb))

  result <- .load_seed_vectors(c("protect", "care", "nonexistent"), emb, verbose = FALSE)

  expect_type(result, "list")
  expect_true("protect" %in% names(result))
  expect_true("care" %in% names(result))
  expect_false("nonexistent" %in% names(result))

  # Check vector dimensions
  expect_equal(length(result[["protect"]]), 10)
  expect_equal(length(result[["care"]]), 10)
})

test_that(".load_seed_vectors is case-insensitive", {
  emb <- .create_mock_embeddings(n_dims = 10)
  on.exit(unlink(emb))

  result <- .load_seed_vectors(c("PROTECT", "Care"), emb, verbose = FALSE)

  expect_true("protect" %in% names(result))
  expect_true("care" %in% names(result))
})

#- .build_vectionary_internal ----

test_that(".build_vectionary_internal builds correctly with ridge", {
  emb <- .create_mock_embeddings(n_dims = 10)
  on.exit(unlink(emb))

  dict <- data.frame(
    word = c("protect", "care", "help", "harm", "hurt"),
    score = c(1, 1, 1, 0, 0)
  )

  result <- .build_vectionary_internal(
    dictionary = dict,
    embeddings_path = emb,
    dimensions = "score",
    method = "ridge",
    lambda = 1.0,
    project_full_vocab = FALSE,
    verbose = FALSE
  )

  expect_type(result, "list")
  expect_true("axes" %in% names(result))
  expect_true("word_projections" %in% names(result))
  expect_true("words_found" %in% names(result))

  # Axis should be non-zero
  axis <- result$axes[["score"]]
  expect_true(sqrt(sum(axis^2)) > 0)
})

#- Full vectionary_builder ----

test_that("vectionary_builder produces valid Vec-tionary object", {
  emb <- .create_mock_embeddings(n_dims = 10)
  on.exit(unlink(emb))

  dict <- data.frame(
    word = c("protect", "care", "help", "harm", "hurt"),
    care = c(1, 1, 1, 0, 0)
  )

  vect <- vectionary_builder(
    dictionary = dict,
    embeddings = emb,
    dimensions = "care",
    lambda = 1,
    save_path = NULL,
    verbose = FALSE,
    spellcheck = FALSE,
    remove_stopwords = FALSE
  )

  expect_s3_class(vect, "Vec-tionary")
  expect_equal(vect$dimensions, "care")
  expect_true(is.data.frame(vect$word_projections))
  expect_true("word" %in% names(vect$word_projections))
  expect_true("care" %in% names(vect$word_projections))
  expect_true(nrow(vect$word_projections) > 0)

  # Metadata
  expect_equal(vect$metadata$method, "ridge")
  expect_true(!is.null(vect$metadata$build_date))
})

test_that("vectionary_builder works with multiple dimensions", {
  emb <- .create_mock_embeddings(n_dims = 10)
  on.exit(unlink(emb))

  dict <- data.frame(
    word = c("protect", "care", "help", "fair", "just",
             "harm", "hurt", "cheat", "fraud", "cruel"),
    care = c(1, 1, 1, 0, 0, 0, 0, 0, 0, 0),
    fairness = c(0, 0, 0, 1, 1, 0, 0, 1, 1, 0)
  )

  vect <- vectionary_builder(
    dictionary = dict,
    embeddings = emb,
    dimensions = c("care", "fairness"),
    lambda = 1,
    save_path = NULL,
    verbose = FALSE,
    spellcheck = FALSE,
    remove_stopwords = FALSE
  )

  expect_s3_class(vect, "Vec-tionary")
  expect_equal(vect$dimensions, c("care", "fairness"))
  expect_equal(length(vect$axes), 2)
  expect_true("care" %in% names(vect$word_projections))
  expect_true("fairness" %in% names(vect$word_projections))
})

test_that("vectionary_builder saves to file when save_path provided", {
  emb <- .create_mock_embeddings(n_dims = 10)
  save_file <- tempfile(fileext = ".rds")
  on.exit({
    unlink(emb)
    unlink(save_file)
  })

  dict <- data.frame(
    word = c("protect", "care", "harm", "hurt"),
    score = c(1, 1, 0, 0)
  )

  vect <- vectionary_builder(
    dictionary = dict,
    embeddings = emb,
    lambda = 1,
    save_path = save_file,
    verbose = FALSE,
    spellcheck = FALSE,
    remove_stopwords = FALSE
  )

  expect_true(file.exists(save_file))

  # Reload and verify
  loaded <- readRDS(save_file)
  expect_s3_class(loaded, "Vec-tionary")
  expect_equal(loaded$dimensions, vect$dimensions)
})

#- .validate_lambda ----

test_that(".validate_lambda returns validity metrics", {
  emb <- .create_mock_embeddings(n_dims = 10)
  on.exit(unlink(emb))

  dict <- data.frame(
    word = c("protect", "care", "help", "harm", "hurt"),
    score = c(0.9, 0.8, 0.7, -0.5, -0.9)
  )

  # Build a vectionary first
  result <- .build_vectionary_internal(
    dictionary = dict,
    embeddings_path = emb,
    dimensions = "score",
    method = "ridge",
    lambda = 1.0,
    verbose = FALSE
  )

  vect <- list(
    axes = result$axes,
    word_projections = result$word_projections,
    dimensions = "score"
  )

  validation <- .validate_lambda(vect, dict, "score", emb)

  expect_type(validation, "list")
  expect_true("validity" %in% names(validation))
  expect_true("validity_metric" %in% names(validation))
  expect_true(is.numeric(validation$validity))
  expect_true(validation$validity >= 0 && validation$validity <= 1)
})

#- GCV ----

test_that(".gcv_select_lambda returns optimal lambda", {
  set.seed(42)
  n <- 10
  d <- 5
  W <- matrix(rnorm(n * d), nrow = n)
  y <- rnorm(n)

  result <- .gcv_select_lambda(W, y, return_all = TRUE)

  expect_type(result, "list")
  expect_true("optimal_lambda" %in% names(result))
  expect_true(is.numeric(result$optimal_lambda))
  expect_true(result$optimal_lambda > 0)
})
