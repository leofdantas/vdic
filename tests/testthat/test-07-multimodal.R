#- Multimodal Tests ----
#
# Tests for the multimodal pipeline (Phase 2 / Phase 3).
# Organised into two tiers:
#
#   Tier 1 (no Python needed): input validation, S3 structure, analyze_text()
#   on text vectionaries, analyze_image() error paths.
#
#   Tier 2 (Python required): guarded by skip_if_not_siglip(). Tests real
#   SigLIP encoding and end-to-end multimodal analysis.

#-- Helpers ----

.create_text_vectionary <- function() {
  wp <- data.frame(
    word     = c("protect", "care", "harm", "hurt"),
    care     = c(0.9, 0.8, -0.8, -0.7),
    stringsAsFactors = FALSE
  )
  structure(
    list(
      axes             = list(care = rnorm(300)),
      word_projections = wp,
      dimensions       = "care",
      modality         = "text",
      embedding_dim    = 300L,
      metadata         = list()
    ),
    class = "Vec-tionary"
  )
}

.create_multimodal_vectionary <- function(dim = 1152L) {
  axes <- list(
    care = rnorm(dim),
    harm = rnorm(dim)
  )
  structure(
    list(
      axes             = axes,
      word_projections = NULL,
      dimensions       = c("care", "harm"),
      modality         = "multimodal",
      embedding_dim    = dim,
      metadata         = list(model_name = "google/siglip-so400m-patch14-384")
    ),
    class = "Vec-tionary"
  )
}

skip_if_not_siglip <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    testthat::skip("reticulate not installed")
  }
  if (!reticulate::py_available(initialize = FALSE)) {
    testthat::skip("Python not available")
  }
  for (pkg in c("transformers", "torch", "PIL", "sentencepiece")) {
    if (!reticulate::py_module_available(pkg)) {
      testthat::skip(paste("Python package not available:", pkg))
    }
  }
}


#-- Tier 1: analyze_image() input validation ----

test_that("analyze_image() rejects non-Vec-tionary input", {
  expect_error(analyze_image(list(), images = "x.jpg"), "Vec-tionary")
})

test_that("analyze_image() rejects text/legacy vectionary", {
  vect <- .create_text_vectionary()
  expect_error(analyze_image(vect, images = "x.jpg"),
               "multimodal|modality", ignore.case = TRUE)
})

test_that("analyze_image() rejects empty images vector", {
  vect <- .create_multimodal_vectionary()
  expect_error(analyze_image(vect, images = character(0)), "non-empty")
})


#-- Tier 1: analyze_text() ----

test_that("analyze_text() rejects non-Vec-tionary input", {
  expect_error(analyze_text(list(), text = "hello"), "Vec-tionary")
})

test_that("analyze_text() rejects empty text vector", {
  vect <- .create_text_vectionary()
  expect_error(analyze_text(vect, text = character(0)), "non-empty")
})

test_that("analyze_text() returns data frame for text vectionary", {
  vect   <- .create_text_vectionary()
  result <- analyze_text(vect, c("protect and care", "harm and hurt"))

  expect_s3_class(result, "data.frame")
  expect_equal(nrow(result), 2L)
  expect_named(result, c("text", "care"))
  expect_equal(result$text, c("protect and care", "harm and hurt"))
})

test_that("analyze_text() scores are in expected direction (text vectionary)", {
  vect <- .create_text_vectionary()
  result <- analyze_text(vect, c("protect and care", "harm and hurt"))

  # "protect care" words have positive care scores; "harm hurt" negative
  expect_gt(result$care[1], 0)
  expect_lt(result$care[2], 0)
})

test_that("analyze_text() scores match vectionary_analyze() mean (text vectionary)", {
  vect   <- .create_text_vectionary()
  texts  <- c("protect care", "harm hurt")

  df_result   <- analyze_text(vect, texts)
  list_result <- vectionary_analyze(vect, texts, metric = "mean")

  expect_equal(df_result$care, list_result$care)
})

test_that("analyze_text() handles unknown words (returns NA)", {
  vect   <- .create_text_vectionary()
  result <- analyze_text(vect, "xyznotaword")

  expect_true(is.na(result$care[1]))
})

test_that("analyze_text() rejects multimodal vectionary without Python (errors clearly)", {
  vect <- .create_multimodal_vectionary()
  # Should error at .check_siglip_deps() before any Python call
  err <- tryCatch(
    analyze_text(vect, "test text"),
    error = function(e) conditionMessage(e)
  )
  # Either reticulate/Python missing or SigLIP not available
  expect_true(
    grepl("reticulate|Python|transformers|sentencepiece", err,
          ignore.case = TRUE)
  )
})


#-- Tier 1: multimodal vectionary structure ----

test_that("multimodal vectionary has correct structure", {
  vect <- .create_multimodal_vectionary()

  expect_s3_class(vect, "Vec-tionary")
  expect_equal(vect[["modality"]], "multimodal")
  expect_equal(vect[["embedding_dim"]], 1152L)
  expect_null(vect[["word_projections"]])
  expect_length(vect$axes$care, 1152L)
  expect_length(vect$axes$harm, 1152L)
})

test_that("print() shows modality and embedding_dim for multimodal vectionary", {
  vect <- .create_multimodal_vectionary()
  out  <- capture.output(print(vect))

  expect_true(any(grepl("multimodal", out, ignore.case = TRUE)))
  expect_true(any(grepl("1152", out)))
})


#-- Tier 2: SigLIP encoding (Python required) ----

test_that("SigLIP model loads and caches correctly", {
  skip_if_not_siglip()

  mm  <- vdic:::.load_siglip_model()
  mm2 <- vdic:::.load_siglip_model()

  expect_true(is.list(mm))
  expect_named(mm, c("processor", "model", "torch"))
  expect_identical(mm$model, mm2$model)  # cache hit
})

test_that(".encode_text_siglip() returns unit-normalized (n x 1152) matrix", {
  skip_if_not_siglip()

  words <- c("protect", "harm", "care")
  emb   <- vdic:::.encode_text_siglip(words)

  expect_true(is.matrix(emb))
  expect_equal(nrow(emb), 3L)
  expect_equal(ncol(emb), 1152L)
  expect_equal(rownames(emb), words)

  norms <- sqrt(rowSums(emb^2))
  expect_true(all(abs(norms - 1) < 1e-5))
})

test_that(".encode_images_siglip() returns unit-normalized (n x 1152) matrix", {
  skip_if_not_siglip()

  img_files <- replicate(2, tempfile(fileext = ".png"))
  for (f in img_files) {
    png(f, width = 200, height = 200)
    par(mar = c(2, 2, 1, 1))
    plot(1:5, col = 1:5)
    dev.off()
  }
  on.exit(unlink(img_files))

  emb <- vdic:::.encode_images_siglip(img_files)

  expect_true(is.matrix(emb))
  expect_equal(nrow(emb), 2L)
  expect_equal(ncol(emb), 1152L)
  expect_equal(rownames(emb), img_files)

  norms <- sqrt(rowSums(emb^2))
  expect_true(all(abs(norms - 1) < 1e-5))
})

test_that(".encode_images_siglip() errors clearly on missing file", {
  skip_if_not_siglip()

  expect_error(
    vdic:::.encode_images_siglip("/no/such/image.jpg"),
    "not found"
  )
})

test_that("analyze_text() works with multimodal vectionary", {
  skip_if_not_siglip()

  dict <- data.frame(
    word = c("protect", "care", "harm", "hurt"),
    care = c(1, 1, 0, 0),
    stringsAsFactors = FALSE
  )
  vect <- vectionary_builder(dict, embeddings = "siglip",
                             modality = "multimodal", verbose = FALSE)

  result <- analyze_text(vect, c("a caring nurse", "an act of violence"))

  expect_s3_class(result, "data.frame")
  expect_equal(nrow(result), 2L)
  expect_named(result, c("text", "care"))
  expect_true(all(is.finite(result$care)))
})

test_that("analyze_image() returns data frame with correct structure", {
  skip_if_not_siglip()

  dict <- data.frame(
    word = c("protect", "care", "harm", "hurt"),
    care = c(1, 1, 0, 0),
    stringsAsFactors = FALSE
  )
  vect <- vectionary_builder(dict, embeddings = "siglip",
                             modality = "multimodal", verbose = FALSE)

  img_files <- replicate(2, tempfile(fileext = ".png"))
  for (f in img_files) {
    png(f, width = 200, height = 200)
    par(mar = c(2, 2, 1, 1))
    plot(1:5, col = 1:5)
    dev.off()
  }
  on.exit(unlink(img_files))

  result <- analyze_image(vect, images = img_files)

  expect_s3_class(result, "data.frame")
  expect_equal(nrow(result), 2L)
  expect_named(result, c("image", "care"))
  expect_equal(result$image, img_files)
  expect_true(all(is.finite(result$care)))
})
