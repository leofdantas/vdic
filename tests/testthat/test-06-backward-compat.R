#- Backward Compatibility Tests ----
#
# These tests verify that vectionaries built before v1.1.0 (without modality,
# embedding_dim, or image_projections fields) continue to work unchanged after
# the multi-modal infrastructure was added in Phase 1.

# A minimal v1.0-style vectionary: no modality, embedding_dim, or
# image_projections fields — mirrors what users have saved as RDS files.
.create_legacy_vectionary <- function() {
  word_projections <- data.frame(
    word     = c("protect", "care", "harm", "hurt"),
    care     = c(0.9, 0.8, -0.8, -0.7),
    stringsAsFactors = FALSE
  )
  vect <- list(
    axes             = list(care = rnorm(10)),
    word_projections = word_projections,
    dimensions       = "care",
    metadata         = list(
      method     = "ridge",
      binary_word = TRUE,
      vocab_size  = 4,
      build_date  = Sys.time()
    )
  )
  class(vect) <- "Vec-tionary"
  vect
}


#-- print and summary handle missing fields gracefully ----

test_that("print() works on legacy vectionary (no modality field)", {
  vect <- .create_legacy_vectionary()
  expect_null(vect[["modality"]])
  expect_output(print(vect), "Vectionary")
  expect_output(print(vect), "care")
})

test_that("summary() works on legacy vectionary (no modality field)", {
  vect <- .create_legacy_vectionary()
  expect_output(summary(vect), "Vectionary Summary")
  expect_output(summary(vect), "care")
})

test_that("print() does not show modality line for legacy vectionaries", {
  vect <- .create_legacy_vectionary()
  out  <- capture.output(print(vect))
  expect_false(any(grepl("Modality", out)))
  expect_false(any(grepl("Embedding dimension", out)))
})


#-- text analysis methods unchanged ----

test_that("$mean() works on legacy vectionary", {
  vect   <- .create_legacy_vectionary()
  result <- vect$mean("protect care")
  expect_named(result, "care")
  expect_true(is.numeric(result$care))
})

test_that("$rms() works on legacy vectionary", {
  vect   <- .create_legacy_vectionary()
  result <- vect$rms("protect care")
  expect_named(result, "care")
  expect_true(is.numeric(result$care))
})

test_that("$metrics() works on legacy vectionary", {
  vect   <- .create_legacy_vectionary()
  result <- vect$metrics("protect care")
  expect_true(is.list(result))
  # $metrics() returns list(mean = list(care = ...), rms = ..., ...)
  expect_true("mean" %in% names(result))
  expect_true("care" %in% names(result$mean))
})


#-- modality parameter: "text" is the default ----

test_that("vectionary_builder() modality defaults to 'text'", {
  expect_error(
    vectionary_builder(data.frame(word = "test", care = 1),
                       embeddings = "nonexistent.vec"),
    "Embeddings file not found"   # hits text-path file check, not modality error
  )
})

test_that("vectionary_builder() rejects unknown modality values", {
  expect_error(
    vectionary_builder(data.frame(word = "test", care = 1),
                       embeddings = "nonexistent.vec",
                       modality   = "audio"),
    "modality must be either"
  )
})


#-- new v1.1.0 vectionaries carry modality fields ----

test_that("new text vectionaries have modality = 'text'", {
  # A v1.1.0-style mock with new fields
  vect              <- .create_legacy_vectionary()
  vect$modality     <- "text"
  vect$embedding_dim <- 10L
  vect$image_projections <- NULL

  expect_equal(vect[["modality"]], "text")
  expect_equal(vect[["embedding_dim"]], 10L)
  expect_null(vect[["image_projections"]])
  expect_output(print(vect), "Modality: text")
  expect_output(print(vect), "Embedding dimension: 10")
})


#-- analyze_image() rejects text vectionaries ----

test_that("analyze_image() errors on legacy (text) vectionary", {
  vect <- .create_legacy_vectionary()
  expect_error(
    analyze_image(vect, images = "photo.jpg"),
    "modality.*multimodal|multimodal.*modality"
  )
})

test_that("analyze_image() errors on non-vectionary input", {
  expect_error(analyze_image(list(), images = "photo.jpg"), "Vec-tionary")
})


#-- multimodal builder gives informative not-yet-implemented error ----

test_that("vectionary_builder() with modality='multimodal' gives informative Python error", {
  err <- tryCatch(
    vectionary_builder(
      dictionary = data.frame(word = c("protect", "harm"), care = c(1, 0)),
      embeddings = "siglip",
      modality   = "multimodal"
    ),
    error = function(e) conditionMessage(e)
  )
  # Phase 2: now correctly errors on missing Python / reticulate setup
  expect_true(
    grepl("reticulate", err) || grepl("Python", err) || grepl("transformers", err)
  )
})
