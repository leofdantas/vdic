#- SigLIP Embedding Utilities ----
#
# This file provides utilities for encoding text and images with SigLIP via
# Python and the reticulate package. SigLIP is a multi-modal model from Google
# (google/siglip-so400m-patch14-384) that produces 512-dimensional embeddings in
# a shared text-image space — the same vector space for both words and images.
#
# Unlike word2vec/FastText/GloVe, SigLIP is a neural network model and does NOT
# provide a pre-computed vocabulary file. All embeddings must be computed on-the-fly
# using the model. This requires:
#   - Python (>= 3.9 recommended)
#   - reticulate R package
#   - Python libraries: transformers, torch, Pillow
#
# Installation guide (run once in R):
#   install.packages("reticulate")
#   reticulate::install_miniconda()        # if no Python environment exists
#   reticulate::py_install(c("transformers", "torch", "Pillow"))
#
# Two functions are provided:
#   .encode_text_siglip()  — encodes words/phrases to 512-dim vectors (for building axes)
#   .encode_images_siglip() — encodes image files to 512-dim vectors (for analyzing images)
#
# Both return a unit-normalized matrix of shape (n x 512), ready for ridge regression.


#-- Internal: Check for reticulate and Python dependencies ----

#' Check Python environment and required libraries (internal)
#'
#' @description
#' Verifies that reticulate is installed, a Python environment is active, and
#' the required Python libraries (transformers, torch, Pillow) are available.
#' Provides clear, actionable error messages if anything is missing.
#'
#' @return Invisible TRUE if all checks pass. Stops with an informative error
#'   message if any dependency is missing.
#'
#' @keywords internal
.check_siglip_deps <- function() {

  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop(
      "The 'reticulate' package is required for multi-modal vectionaries.\n\n",
      "Install it with:\n",
      "  install.packages('reticulate')\n\n",
      "Then set up a Python environment with the required libraries:\n",
      "  reticulate::install_miniconda()\n",
      "  reticulate::py_install(c('transformers', 'torch', 'Pillow'))\n\n",
      "Text-based vectionaries do not require Python or reticulate."
    )
  }

  if (!reticulate::py_available(initialize = FALSE)) {
    stop(
      "No Python environment found. reticulate requires a working Python installation.\n\n",
      "Set up a Python environment with:\n",
      "  reticulate::install_miniconda()\n",
      "  reticulate::py_install(c('transformers', 'torch', 'Pillow'))\n\n",
      "Or point reticulate to an existing environment:\n",
      "  reticulate::use_virtualenv('/path/to/venv')\n",
      "  reticulate::use_condaenv('my_env')"
    )
  }

  required  <- c("transformers", "torch", "PIL")
  missing   <- required[!vapply(required, reticulate::py_module_available, logical(1L))]

  if (length(missing) > 0L) {
    pkg_names <- gsub("PIL", "Pillow", missing)
    stop(
      "The following Python libraries are required but not found: ",
      paste(missing, collapse = ", "), "\n\n",
      "Install them with:\n",
      "  reticulate::py_install(c(", paste0("'", pkg_names, "'", collapse = ", "), "))\n\n",
      "Or from a terminal inside your Python environment:\n",
      "  pip install ", paste(pkg_names, collapse = " ")
    )
  }

  invisible(TRUE)
}


#-- Internal: Load SigLIP Model ----

#' Load SigLIP model via reticulate (internal)
#'
#' @description
#' Loads the SigLIP processor and model from the Hugging Face hub using the
#' Python transformers library. The model is cached locally by the transformers
#' library on first download (usually to \file{~/.cache/huggingface/}).
#'
#' The loaded model is cached in \code{.vdic_env$siglip_model} for the R
#' session, so subsequent calls with the same \code{model_name} skip the
#' download and loading step.
#'
#' @param model_name Hugging Face model ID
#'   (default: \code{"google/siglip-so400m-patch14-384"})
#'
#' @return Named list with elements:
#'   \describe{
#'     \item{processor}{SiglipProcessor instance}
#'     \item{model}{SiglipModel instance (eval mode, no gradients)}
#'     \item{torch}{torch Python module}
#'   }
#'
#' @keywords internal
.load_siglip_model <- function(model_name = "google/siglip-so400m-patch14-384") {

  # Return cached model if already loaded for this model_name
  cache_key <- paste0("siglip_", model_name)
  if (!is.null(.vdic_env[[cache_key]])) {
    return(.vdic_env[[cache_key]])
  }

  cli::cli_progress_step("Loading SigLIP model ({model_name})", spinner = TRUE)

  transformers <- reticulate::import("transformers", delay_load = FALSE)
  torch        <- reticulate::import("torch",        delay_load = FALSE)

  processor <- transformers$AutoProcessor$from_pretrained(model_name)
  model     <- transformers$AutoModel$from_pretrained(model_name)

  # Evaluation mode — disables dropout and batch norm training behaviour
  model$eval()

  result <- list(processor = processor, model = model, torch = torch)

  # Cache for reuse within the session
  .vdic_env[[cache_key]] <- result

  cli::cli_progress_done()

  result
}


#-- Internal: Encode Text with SigLIP ----

#' Encode text strings using SigLIP text encoder (internal)
#'
#' @description
#' Encodes a character vector of text strings (typically dictionary words or
#' short phrases) into 512-dimensional vectors using the SigLIP text encoder.
#' The resulting vectors live in the shared text-image embedding space, which
#' means they can be used to build axes that score images.
#'
#' Requires Python with the transformers, torch, and Pillow libraries.
#' Call \code{.check_siglip_deps()} before this function.
#'
#' @param text_strings Character vector of words or phrases to encode
#' @param model Named list from \code{.load_siglip_model()}. If NULL (default),
#'   loads the model automatically.
#' @param model_name SigLIP model ID (default: \code{"google/siglip-so400m-patch14-384"}).
#'   Only used when \code{model = NULL}.
#' @param normalize Logical. If TRUE (default), normalizes each embedding to unit
#'   length. SigLIP embeddings are designed to be unit-normalized; this ensures
#'   projections are cosine-similarity-based.
#' @param batch_size Integer. Number of strings to encode per batch (default: 64).
#'   Larger batches are faster but use more memory.
#'
#' @return Numeric matrix of shape \eqn{n \times 512}. Row names are
#'   set to \code{text_strings}.
#'
#' @keywords internal
.encode_text_siglip <- function(
  text_strings,
  model      = NULL,
  model_name = "google/siglip-so400m-patch14-384",
  normalize  = TRUE,
  batch_size = 64L
) {

  if (is.null(model)) model <- .load_siglip_model(model_name)

  n       <- length(text_strings)
  batches <- split(text_strings, ceiling(seq_len(n) / batch_size))

  cli::cli_progress_bar(
    "Encoding {n} word{?s} with SigLIP",
    total = length(batches)
  )

  result_rows <- vector("list", length(batches))

  for (i in seq_along(batches)) {
    batch <- as.list(batches[[i]])

    inputs <- model$processor(
      text            = batch,
      padding         = TRUE,
      truncation      = TRUE,
      return_tensors  = "pt"
    )

    # Run without gradient tracking (inference mode, faster + less memory)
    with(model$torch$no_grad(), {
      features <- model$model$get_text_features(
        input_ids      = inputs$input_ids,
        attention_mask = inputs$attention_mask
      )
    })

    # Convert torch tensor (batch_size x 512) to R matrix
    feat_r <- reticulate::py_to_r(features$detach()$cpu()$numpy())
    result_rows[[i]] <- feat_r

    cli::cli_progress_update()
  }

  cli::cli_progress_done()

  emb <- do.call(rbind, result_rows)
  rownames(emb) <- text_strings

  if (normalize) emb <- .l2_normalize_rows(emb)

  emb
}


#-- Internal: Encode Images with SigLIP ----

#' Encode image files using SigLIP image encoder (internal)
#'
#' @description
#' Encodes a character vector of image file paths into 512-dimensional vectors
#' using the SigLIP image encoder. The resulting vectors live in the same shared
#' text-image embedding space as the text embeddings, so images can be scored by
#' vectionaries whose axes were learned from text.
#'
#' Requires Python with the transformers, torch, and Pillow libraries.
#' Call \code{.check_siglip_deps()} before this function.
#'
#' @param image_paths Character vector of paths to image files (JPEG, PNG, etc.)
#' @param model Named list from \code{.load_siglip_model()}. If NULL (default),
#'   loads the model automatically.
#' @param model_name SigLIP model ID (default: \code{"google/siglip-so400m-patch14-384"}).
#'   Only used when \code{model = NULL}.
#' @param normalize Logical. If TRUE (default), normalizes each embedding to unit
#'   length. SigLIP embeddings are designed to be unit-normalized; this ensures
#'   projections are cosine-similarity-based.
#' @param batch_size Integer. Number of images to encode per batch (default: 32).
#'   Reduce if running out of memory. Larger batches are faster on GPU.
#'
#' @return Numeric matrix of shape \eqn{n \times 512}. Row names are
#'   set to \code{image_paths}.
#'
#' @keywords internal
.encode_images_siglip <- function(
  image_paths,
  model      = NULL,
  model_name = "google/siglip-so400m-patch14-384",
  normalize  = TRUE,
  batch_size = 32L
) {

  if (is.null(model)) model <- .load_siglip_model(model_name)

  # Validate all paths before starting
  missing_files <- image_paths[!file.exists(image_paths)]
  if (length(missing_files) > 0L) {
    stop(
      length(missing_files), " image file(s) not found:\n",
      paste0("  ", head(missing_files, 5), collapse = "\n"),
      if (length(missing_files) > 5) paste0("\n  ... (", length(missing_files), " total)") else ""
    )
  }

  PIL     <- reticulate::import("PIL.Image", delay_load = FALSE)
  n       <- length(image_paths)
  batches <- split(image_paths, ceiling(seq_len(n) / batch_size))

  cli::cli_progress_bar(
    "Encoding {n} image{?s} with SigLIP",
    total = length(batches)
  )

  result_rows <- vector("list", length(batches))

  for (i in seq_along(batches)) {
    batch_paths  <- batches[[i]]
    pil_images   <- lapply(batch_paths, function(p) PIL$open(p)$convert("RGB"))

    inputs <- model$processor(
      images         = pil_images,
      return_tensors = "pt"
    )

    with(model$torch$no_grad(), {
      features <- model$model$get_image_features(
        pixel_values = inputs$pixel_values
      )
    })

    feat_r <- reticulate::py_to_r(features$detach()$cpu()$numpy())
    result_rows[[i]] <- feat_r

    cli::cli_progress_update()
  }

  cli::cli_progress_done()

  emb <- do.call(rbind, result_rows)
  rownames(emb) <- image_paths

  if (normalize) emb <- .l2_normalize_rows(emb)

  emb
}


#-- Internal: L2 row normalization ----

#' Normalize matrix rows to unit L2 norm (internal)
#'
#' @param m Numeric matrix
#' @return Matrix with each row scaled to unit Euclidean length. Rows with
#'   zero norm are left unchanged.
#' @keywords internal
.l2_normalize_rows <- function(m) {
  norms <- sqrt(rowSums(m^2))
  norms[norms == 0] <- 1  # avoid division by zero
  m / norms
}
