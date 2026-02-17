# Tests for R/04_vectionary.R
# S3 class, metric functions, and vectionary_analyze

#- Helper: create a mock Vec-tionary object ----
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
      build_date = Sys.time()
    )
  )
  class(vect) <- "Vec-tionary"
  return(vect)
}

#- Tokenization ----

test_that(".tokenize converts to lowercase and removes punctuation", {
  tokens <- .tokenize("Hello, World! This is a TEST.")
  expect_equal(tokens, c("hello", "world", "this", "is", "a", "test"))
})

test_that(".tokenize handles empty string", {
  tokens <- .tokenize("")
  expect_equal(tokens, character(0))
})

test_that(".tokenize handles multiple spaces", {
  tokens <- .tokenize("word1   word2     word3")
  expect_equal(tokens, c("word1", "word2", "word3"))
})

test_that(".tokenize handles special punctuation", {
  tokens <- .tokenize("it's a test—with dashes & symbols!")
  expect_true("it" %in% tokens)
  expect_true("test" %in% tokens)
  expect_true("dashes" %in% tokens)
})

#- .get_doc_scores ----

test_that(".get_doc_scores returns correct scores for matched words", {
  vect <- .create_mock_vectionary()
  scores <- .get_doc_scores("protect care harm", vect$word_projections, vect$dimensions)

  expect_true(is.data.frame(scores))
  expect_equal(nrow(scores), 3)
  expect_equal(ncol(scores), 2)  # care, fairness

  # Check protect's care score
  expect_equal(scores$care[1], 0.9)
})

test_that(".get_doc_scores preserves duplicates", {
  vect <- .create_mock_vectionary()
  scores <- .get_doc_scores("protect protect protect", vect$word_projections, vect$dimensions)

  # Three occurrences of "protect" = three rows
  expect_equal(nrow(scores), 3)
  expect_equal(scores$care[1], scores$care[2])
  expect_equal(scores$care[2], scores$care[3])
})

test_that(".get_doc_scores returns empty for no matches", {
  vect <- .create_mock_vectionary()
  scores <- .get_doc_scores("xyzzyx unknown", vect$word_projections, vect$dimensions)

  expect_true(is.data.frame(scores))
  expect_equal(nrow(scores), 0)
})

test_that(".get_doc_scores is case-insensitive", {
  vect <- .create_mock_vectionary()
  scores <- .get_doc_scores("PROTECT Care HeLp", vect$word_projections, vect$dimensions)

  expect_equal(nrow(scores), 3)
})

test_that(".get_doc_scores rejects vector input", {
  vect <- .create_mock_vectionary()
  expect_error(
    .get_doc_scores(c("text1", "text2"), vect$word_projections, vect$dimensions),
    "single character string"
  )
})

#- Metric Functions ----

test_that(".mean computes correct arithmetic mean", {
  vect <- .create_mock_vectionary()
  result <- .mean("protect care harm", vect$word_projections, vect$dimensions)

  expect_type(result, "list")
  expect_equal(length(result), 2)
  expect_true("care" %in% names(result))
  expect_true("fairness" %in% names(result))

  # Manual: mean(0.9, 0.8, -0.8) = 0.3
  expect_equal(result[["care"]], mean(c(0.9, 0.8, -0.8)), tolerance = 1e-10)
})

test_that(".mean returns NA for no matches", {
  vect <- .create_mock_vectionary()
  result <- .mean("xyzzyx", vect$word_projections, vect$dimensions)

  expect_true(all(is.na(unlist(result))))
})

test_that(".mse computes mean squared error", {
  vect <- .create_mock_vectionary()
  result <- .mse("protect care harm", vect$word_projections, vect$dimensions)

  # Manual: mean(0.9^2, 0.8^2, 0.8^2) = mean(0.81, 0.64, 0.64)
  expected_care <- mean(c(0.9^2, 0.8^2, (-0.8)^2))
  expect_equal(result[["care"]], expected_care, tolerance = 1e-10)
})

test_that(".sd computes sample standard deviation", {
  vect <- .create_mock_vectionary()
  result <- .sd("protect care harm", vect$word_projections, vect$dimensions)

  # Sample SD: sqrt(sum((x - mean)^2) / (n - 1))
  vals <- c(0.9, 0.8, -0.8)
  m <- mean(vals)
  expected <- sqrt(sum((vals - m)^2) / (length(vals) - 1))
  expect_equal(result[["care"]], expected, tolerance = 1e-10)
})

test_that(".se computes standard error", {
  vect <- .create_mock_vectionary()
  result <- .se("protect care harm", vect$word_projections, vect$dimensions)

  # SE = sample_sd / sqrt(n)
  vals <- c(0.9, 0.8, -0.8)
  m <- mean(vals)
  samp_sd <- sqrt(sum((vals - m)^2) / (length(vals) - 1))
  expected <- samp_sd / sqrt(length(vals))
  expect_equal(result[["care"]], expected, tolerance = 1e-10)
})

test_that(".top_10 returns mean of top projections", {
  vect <- .create_mock_vectionary()

  # Use text with many words so top-10 is a subset
  text <- "protect care help safe kind harm hurt damage cruel abuse fair just equal rights balance"
  result <- .top_10(text, vect$word_projections, vect$dimensions)

  # For care dimension: top 10 of 15 words
  all_scores <- .get_doc_scores(text, vect$word_projections, vect$dimensions)
  care_sorted <- sort(all_scores$care, decreasing = TRUE)
  expected <- mean(care_sorted[1:10])
  expect_equal(result[["care"]], expected, tolerance = 1e-10)
})

test_that(".top_10 uses all words if fewer than 10", {
  vect <- .create_mock_vectionary()
  result_top10 <- .top_10("protect care harm", vect$word_projections, vect$dimensions)
  result_mean_top3 <- mean(sort(c(0.9, 0.8, -0.8), decreasing = TRUE)[1:3])

  # With only 3 words, top_10 should use all 3
  expect_equal(result_top10[["care"]], result_mean_top3, tolerance = 1e-10)
})

test_that(".metrics returns all six metrics", {
  vect <- .create_mock_vectionary()
  result <- .metrics("protect care harm", vect$word_projections, vect$dimensions)

  expect_type(result, "list")
  expect_equal(length(result), 6)
  expect_true(all(c("mean", "mse", "sd", "se", "top_10", "top_20") %in% names(result)))

  # Each element should be a named list
  for (metric_name in names(result)) {
    expect_true(is.list(result[[metric_name]]))
    expect_equal(length(result[[metric_name]]), 2)
    expect_true(all(c("care", "fairness") %in% names(result[[metric_name]])))
  }
})

#- $ operator dispatch ----

test_that("$ operator returns callable functions for valid methods", {
  vect <- .create_mock_vectionary()

  mean_fn <- vect$mean
  expect_true(is.function(mean_fn))

  mse_fn <- vect$mse
  expect_true(is.function(mse_fn))

  sd_fn <- vect$sd
  expect_true(is.function(sd_fn))

  se_fn <- vect$se
  expect_true(is.function(se_fn))

  top10_fn <- vect$top_10
  expect_true(is.function(top10_fn))

  top20_fn <- vect$top_20
  expect_true(is.function(top20_fn))

  metrics_fn <- vect$metrics
  expect_true(is.function(metrics_fn))

})

test_that("$ operator returns results when called", {
  vect <- .create_mock_vectionary()

  result <- vect$mean("protect care")
  expect_true(is.list(result))
  expect_equal(length(result), 2)
})

test_that("$ operator accesses regular fields", {
  vect <- .create_mock_vectionary()

  expect_equal(vect$dimensions, c("care", "fairness"))
  expect_true(is.data.frame(vect$word_projections))
  expect_true(is.list(vect$metadata))
})

test_that("$ operator errors on invalid names", {
  vect <- .create_mock_vectionary()
  expect_error(vect$nonexistent, "no method or field")
})

#- vectionary_analyze ----

test_that("vectionary_analyze works with direct object", {
  vect <- .create_mock_vectionary()
  result <- vectionary_analyze(vect, "protect care", metric = "mean")

  expect_true(is.list(result))
  expect_equal(names(result), c("care", "fairness"))
})

test_that("vectionary_analyze works with all metric options", {
  vect <- .create_mock_vectionary()

  for (metric in c("mean", "mse", "sd", "se", "top_10", "top_20")) {
    result <- vectionary_analyze(vect, "protect care", metric = metric)
    expect_true(is.list(result), info = paste("Failed for metric:", metric))
  }

  all_result <- vectionary_analyze(vect, "protect care", metric = "all")
  expect_true(is.list(all_result))
  expect_equal(length(all_result), 6)
})

test_that("vectionary_analyze rejects non-character text", {
  vect <- .create_mock_vectionary()
  expect_error(vectionary_analyze(vect, 123), "character string or character vector")
  expect_error(vectionary_analyze(vect, character(0)), "character string or character vector")
})

test_that("vectionary_analyze rejects invalid object", {
  expect_error(vectionary_analyze(list(a = 1), "text"), "Vec-tionary")
})

#- print and summary ----

test_that("print.Vec-tionary runs without error", {
  vect <- .create_mock_vectionary()
  expect_output(print(vect), "Vectionary")
  expect_output(print(vect), "care")
  expect_output(print(vect), "fairness")
})

test_that("summary.Vec-tionary runs without error", {
  vect <- .create_mock_vectionary()
  expect_output(summary(vect), "Vectionary Summary")
  expect_output(summary(vect), "ridge")
})

#- Vector input (batch) ----

test_that("batch mean matches single-document loop", {
  vect <- .create_mock_vectionary()
  texts <- c(
    "protect care help",
    "harm hurt damage",
    "fair just equal",
    "xyzzyx unknown"
  )

  batch_result <- vect$mean(texts)
  expect_true(is.list(batch_result))
  expect_equal(length(batch_result[["care"]]), 4)

  # Compare each row to single-document result
  for (i in seq_along(texts)) {
    single <- vect$mean(texts[i])
    for (dim in vect$dimensions) {
      expect_equal(batch_result[[dim]][i], single[[dim]], tolerance = 1e-10,
                   info = paste("Doc", i, "dim", dim))
    }
  }
})

test_that("batch mse matches single-document loop", {
  vect <- .create_mock_vectionary()
  texts <- c("protect care harm", "fair just equal rights")
  batch_result <- vect$mse(texts)
  for (i in seq_along(texts)) {
    single <- vect$mse(texts[i])
    for (dim in vect$dimensions) {
      expect_equal(batch_result[[dim]][i], single[[dim]], tolerance = 1e-10)
    }
  }
})

test_that("batch sd matches single-document loop", {
  vect <- .create_mock_vectionary()
  texts <- c("protect care harm", "fair just equal rights")
  batch_result <- vect$sd(texts)
  for (i in seq_along(texts)) {
    single <- vect$sd(texts[i])
    for (dim in vect$dimensions) {
      expect_equal(batch_result[[dim]][i], single[[dim]], tolerance = 1e-10)
    }
  }
})

test_that("batch se matches single-document loop", {
  vect <- .create_mock_vectionary()
  texts <- c("protect care harm", "fair just equal rights")
  batch_result <- vect$se(texts)
  for (i in seq_along(texts)) {
    single <- vect$se(texts[i])
    for (dim in vect$dimensions) {
      expect_equal(batch_result[[dim]][i], single[[dim]], tolerance = 1e-10)
    }
  }
})

test_that("batch top_10 matches single-document loop", {
  vect <- .create_mock_vectionary()
  texts <- c(
    "protect care help safe kind harm hurt damage cruel abuse fair just equal rights balance",
    "protect harm fair"
  )
  batch_result <- vect$top_10(texts)
  for (i in seq_along(texts)) {
    single <- vect$top_10(texts[i])
    for (dim in vect$dimensions) {
      expect_equal(batch_result[[dim]][i], single[[dim]], tolerance = 1e-10)
    }
  }
})

test_that("batch top_20 matches single-document loop", {
  vect <- .create_mock_vectionary()
  texts <- c("protect care harm", "fair just equal rights")
  batch_result <- vect$top_20(texts)
  for (i in seq_along(texts)) {
    single <- vect$top_20(texts[i])
    for (dim in vect$dimensions) {
      expect_equal(batch_result[[dim]][i], single[[dim]], tolerance = 1e-10)
    }
  }
})

test_that("batch metrics (all) matches single-document loop", {
  vect <- .create_mock_vectionary()
  texts <- c("protect care harm", "fair just equal rights", "xyzzyx")
  batch_result <- vect$metrics(texts)

  expect_true(is.list(batch_result))
  expect_equal(length(batch_result), 6)

  for (metric_name in names(batch_result)) {
    expect_true(is.list(batch_result[[metric_name]]))
    expect_equal(length(batch_result[[metric_name]][["care"]]), 3)
  }

  # Verify against single-doc loop
  for (i in seq_along(texts)) {
    single <- vect$metrics(texts[i])
    for (metric_name in names(single)) {
      for (dim in vect$dimensions) {
        expect_equal(
          batch_result[[metric_name]][[dim]][i],
          single[[metric_name]][[dim]],
          tolerance = 1e-10,
          info = paste("Doc", i, "metric", metric_name, "dim", dim)
        )
      }
    }
  }
})

test_that("vectionary_analyze works with vector input", {
  vect <- .create_mock_vectionary()
  texts <- c("protect care", "harm hurt", "fair just")

  result <- vectionary_analyze(vect, texts, metric = "mean")
  expect_true(is.list(result))
  expect_equal(length(result[["care"]]), 3)

  result_all <- vectionary_analyze(vect, texts, metric = "all")
  expect_true(is.list(result_all))
  expect_equal(length(result_all$mean[["care"]]), 3)
})

test_that("batch handles duplicated words in text correctly", {
  vect <- .create_mock_vectionary()
  texts <- c("protect protect protect", "care care")
  batch_result <- vect$mean(texts)
  for (i in seq_along(texts)) {
    single <- vect$mean(texts[i])
    for (dim in vect$dimensions) {
      expect_equal(batch_result[[dim]][i], single[[dim]], tolerance = 1e-10)
    }
  }
})
