#- Phase 2 Integration Test: SigLIP Multi-Modal Vectionary ----
#
# Run this script from the package root:
#   source("tests/test_phase2_siglip.R")
#
# Requirements:
#   install.packages("reticulate")
#   reticulate::py_install(c("transformers", "torch", "Pillow"))
#
# The script will print PASS / FAIL for each test and a final summary.
# Copy the full output and report any FAILs.

library(vdic)

#-- Helpers ----

.pass <- function(label) cat(sprintf("  PASS  %s\n", label))
.fail <- function(label, err) cat(sprintf("  FAIL  %s\n         %s\n", label, conditionMessage(err)))

run_test <- function(label, expr) {
  tryCatch({ force(expr); .pass(label) },
           error   = function(e) .fail(label, e),
           warning = function(w) cat(sprintf("  WARN  %s\n         %s\n", label, conditionMessage(w))))
}

section <- function(title) cat(sprintf("\n── %s ──\n", title))


#-- Section 1: Python environment ----

section("1. Python environment")

run_test("reticulate is installed", {
  if (!requireNamespace("reticulate", quietly = TRUE)) stop("reticulate not installed")
})

run_test("Python is available", {
  if (!reticulate::py_available(initialize = TRUE)) stop("No Python environment found")
  cat(sprintf("         Python: %s\n", reticulate::py_config()$python))
})

run_test("transformers available", {
  if (!reticulate::py_module_available("transformers")) stop("transformers not found")
  tr <- reticulate::import("transformers")
  cat(sprintf("         transformers: %s\n", tr$`__version__`))
})

run_test("torch available", {
  if (!reticulate::py_module_available("torch")) stop("torch not found")
  torch <- reticulate::import("torch")
  cat(sprintf("         torch: %s\n", torch$`__version__`))
})

run_test("Pillow (PIL) available", {
  if (!reticulate::py_module_available("PIL")) stop("Pillow not found")
})

run_test(".check_siglip_deps() passes", {
  vdic:::.check_siglip_deps()
})


#-- Section 2: Model loading ----

section("2. SigLIP model loading")

run_test("Load model (may download ~800 MB on first run)", {
  mm <<- vdic:::.load_siglip_model()
  if (!is.list(mm) || !all(c("processor", "model", "torch") %in% names(mm))) {
    stop("model list missing expected elements")
  }
})

run_test("Model cache works (second call is instant)", {
  mm2 <- vdic:::.load_siglip_model()
  if (!identical(mm$model, mm2$model)) stop("Cache returned different object")
})


#-- Section 3: Text encoding ----

section("3. Text encoding (.encode_text_siglip)")

run_test("Encode 5 words -> matrix (5 x 512)", {
  words <- c("protect", "care", "harm", "hurt", "justice")
  emb <<- vdic:::.encode_text_siglip(words, model = mm)
  if (!is.matrix(emb))          stop("not a matrix")
  if (!identical(dim(emb), c(5L, 512L))) stop(sprintf("dim is %s, expected 5 x 512", paste(dim(emb), collapse = " x ")))
  if (!identical(rownames(emb), words))  stop("rownames not set to words")
})

run_test("Embeddings are unit-normalized (L2 norm ~= 1)", {
  norms <- sqrt(rowSums(emb^2))
  if (any(abs(norms - 1) > 1e-5)) stop(sprintf("norms not 1: min=%.6f max=%.6f", min(norms), max(norms)))
})

run_test("Encode batch > batch_size (batching logic)", {
  words70 <- paste0("word", seq_len(70))
  emb70 <- vdic:::.encode_text_siglip(words70, model = mm, batch_size = 32L)
  if (!identical(dim(emb70), c(70L, 512L))) stop(sprintf("dim is %s", paste(dim(emb70), collapse = " x ")))
})


#-- Section 4: Build multimodal vectionary ----

section("4. Build multimodal vectionary")

dict <- data.frame(
  word = c("protect", "care", "help", "nurture", "harm", "hurt", "damage", "violence"),
  care = c(1, 1, 1, 1, 0, 0, 0, 0),
  harm = c(0, 0, 0, 0, 1, 1, 1, 1),
  stringsAsFactors = FALSE
)

run_test("vectionary_builder(modality='multimodal') returns Vec-tionary", {
  vect <<- vectionary_builder(
    dictionary = dict,
    embeddings = "siglip",
    modality   = "multimodal",
    save_path  = NULL,
    verbose    = FALSE
  )
  if (!inherits(vect, "Vec-tionary")) stop("not a Vec-tionary")
})

run_test("modality field is 'multimodal'", {
  if (vect[["modality"]] != "multimodal") stop(sprintf("modality is '%s'", vect[["modality"]]))
})

run_test("embedding_dim is 512", {
  if (vect[["embedding_dim"]] != 512L) stop(sprintf("embedding_dim is %s", vect[["embedding_dim"]]))
})

run_test("axes are 512-dim vectors", {
  for (dim in vect$dimensions) {
    ax <- vect$axes[[dim]]
    if (length(ax) != 512L) stop(sprintf("axis '%s' has length %d", dim, length(ax)))
    if (!is.numeric(ax))    stop(sprintf("axis '%s' is not numeric", dim))
  }
})

run_test("word_projections is NULL", {
  if (!is.null(vect$word_projections)) stop("word_projections should be NULL for multimodal")
})

run_test("dimensions are c('care', 'harm')", {
  if (!identical(vect$dimensions, c("care", "harm"))) {
    stop(sprintf("dimensions: %s", paste(vect$dimensions, collapse = ", ")))
  }
})

run_test("metadata contains model_name", {
  if (is.null(vect$metadata$model_name)) stop("model_name missing from metadata")
  cat(sprintf("         model: %s\n", vect$metadata$model_name))
})

run_test("print() works on multimodal vectionary", {
  out <- capture.output(print(vect))
  if (!any(grepl("multimodal", out))) stop("'multimodal' not in print output")
  if (!any(grepl("512",        out))) stop("'512' not in print output")
})

run_test("save/load round-trip preserves axes", {
  tmp <- tempfile(fileext = ".rds")
  saveRDS(vect, tmp)
  vect2 <- readRDS(tmp)
  for (dim in vect$dimensions) {
    if (!isTRUE(all.equal(vect$axes[[dim]], vect2$axes[[dim]]))) {
      stop(sprintf("axes for dim '%s' differ after round-trip", dim))
    }
  }
  unlink(tmp)
})


#-- Section 5: Image encoding ----

section("5. Image encoding (.encode_images_siglip)")

# Create three small temporary PNG test images using R's built-in png()
img_files <- replicate(3, tempfile(fileext = ".png"))
for (f in img_files) {
  png(f, width = 64, height = 64); plot(1:5, col = sample(1:6, 5)); dev.off()
}

run_test("Encode 3 images -> matrix (3 x 512)", {
  emb_img <<- vdic:::.encode_images_siglip(img_files, model = mm)
  if (!is.matrix(emb_img))                       stop("not a matrix")
  if (!identical(dim(emb_img), c(3L, 512L)))     stop(sprintf("dim is %s", paste(dim(emb_img), collapse = " x ")))
  if (!identical(rownames(emb_img), img_files))  stop("rownames not set to image paths")
})

run_test("Image embeddings are unit-normalized", {
  norms <- sqrt(rowSums(emb_img^2))
  if (any(abs(norms - 1) > 1e-5)) stop(sprintf("norms not 1: min=%.6f max=%.6f", min(norms), max(norms)))
})

run_test("Missing file gives clear error", {
  err <- tryCatch(
    vdic:::.encode_images_siglip(c(img_files[1], "/nonexistent/image.jpg"), model = mm),
    error = function(e) conditionMessage(e)
  )
  if (!grepl("not found", err)) stop(sprintf("unexpected error: %s", err))
})


#-- Section 6: analyze_image() ----

section("6. analyze_image()")

run_test("Returns data frame with correct dimensions", {
  result <<- analyze_image(vect, images = img_files)
  if (!is.data.frame(result))             stop("not a data frame")
  if (nrow(result) != 3L)                 stop(sprintf("nrow is %d, expected 3", nrow(result)))
  if (!"image" %in% names(result))        stop("no 'image' column")
  if (!"care"  %in% names(result))        stop("no 'care' column")
  if (!"harm"  %in% names(result))        stop("no 'harm' column")
})

run_test("image column contains original paths", {
  if (!identical(result$image, img_files)) stop("image paths not preserved")
})

run_test("Scores are finite numerics", {
  for (dim in c("care", "harm")) {
    if (!is.numeric(result[[dim]]))    stop(sprintf("'%s' is not numeric", dim))
    if (any(!is.finite(result[[dim]]))) stop(sprintf("'%s' contains non-finite values", dim))
  }
  cat(sprintf("         care: [%.3f, %.3f, %.3f]\n", result$care[1], result$care[2], result$care[3]))
  cat(sprintf("         harm: [%.3f, %.3f, %.3f]\n", result$harm[1], result$harm[2], result$harm[3]))
})

run_test("analyze_image() rejects text vectionary", {
  err <- tryCatch(
    analyze_image(list(class = "Vec-tionary"), img_files),
    error = function(e) conditionMessage(e)
  )
  if (!grepl("Vec-tionary", err)) stop(sprintf("unexpected error: %s", err))
})

run_test("analyze_image() rejects missing images", {
  err <- tryCatch(
    analyze_image(vect, images = c(img_files[1], "/no/such/file.jpg")),
    error = function(e) conditionMessage(e)
  )
  if (!grepl("not found", err)) stop(sprintf("unexpected error: %s", err))
})

# Cleanup temp images
unlink(img_files)


#-- Summary ----

cat("\n────────────────────────────────────────\n")
cat("Done. Paste all output above when reporting issues.\n")
