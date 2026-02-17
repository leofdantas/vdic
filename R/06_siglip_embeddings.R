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

  # Check for reticulate R package
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

  # Check that a Python environment is available and configured
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

  # Check for the three required Python libraries
  required <- c("transformers", "torch", "PIL")
  missing  <- required[!vapply(required, reticulate::py_module_available, logical(1L))]

  if (length(missing) > 0L) {
    pkg_names <- gsub("PIL", "Pillow", missing)  # PIL installs as Pillow
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
#' library on first download (usually to ~/.cache/huggingface/).
#'
#' @param model_name Hugging Face model ID (default: "google/siglip-so400m-patch14-384")
#'
#' @return Named list with elements:
#'   - processor: SiglipProcessor instance
#'   - model: SiglipModel instance (eval mode, no gradients)
#'
#' @keywords internal
.load_siglip_model <- function(model_name = "google/siglip-so400m-patch14-384") {

  transformers <- reticulate::import("transformers", delay_load = TRUE)
  torch        <- reticulate::import("torch",        delay_load = TRUE)

  processor <- transformers$AutoProcessor$from_pretrained(model_name)
  model     <- transformers$AutoModel$from_pretrained(model_name)

  # Evaluation mode — disables dropout, batch norm training behavior, etc.
  model$eval()

  list(processor = processor, model = model, torch = torch)
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
#' See [vectionary_builder()] for setup instructions.
#'
#' @param text_strings Character vector of words or phrases to encode
#' @param model Named list from [.load_siglip_model()]. If NULL (default), loads
#'   the model automatically using the default model name.
#' @param model_name SigLIP model ID (default: "google/siglip-so400m-patch14-384").
#'   Only used when model = NULL.
#' @param normalize Logical. If TRUE (default), normalizes each embedding to unit
#'   length (L2 norm = 1). SigLIP embeddings are designed to be unit-normalized;
#'   this ensures projections are cosine-similarity-based.
#' @param batch_size Integer. Number of strings to encode per batch (default: 64).
#'   Larger batches are faster but use more memory.
#'
#' @return Numeric matrix of shape (length(text_strings) x 512). Row names are
#'   set to text_strings.
#'
#' @keywords internal
.encode_text_siglip <- function(
  text_strings,
  model      = NULL,
  model_name = "google/siglip-so400m-patch14-384",
  normalize  = TRUE,
  batch_size = 64L
) {

  # Phase 1: Skeleton implementation
  # Full implementation in Phase 2

  stop(
    ".encode_text_siglip() is not yet implemented.\n",
    "Full implementation coming in Phase 2 (vdic v1.2.0)."
  )
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
#' See [vectionary_builder()] for setup instructions.
#'
#' @param image_paths Character vector of paths to image files (JPEG, PNG, etc.)
#' @param model Named list from [.load_siglip_model()]. If NULL (default), loads
#'   the model automatically using the default model name.
#' @param model_name SigLIP model ID (default: "google/siglip-so400m-patch14-384").
#'   Only used when model = NULL.
#' @param normalize Logical. If TRUE (default), normalizes each embedding to unit
#'   length (L2 norm = 1). SigLIP embeddings are designed to be unit-normalized;
#'   this ensures projections are cosine-similarity-based.
#' @param batch_size Integer. Number of images to encode per batch (default: 32).
#'   Reduce if running out of memory. Larger batches are faster on GPU.
#'
#' @return Numeric matrix of shape (length(image_paths) x 512). Row names are
#'   set to basename(image_paths).
#'
#' @keywords internal
.encode_images_siglip <- function(
  image_paths,
  model      = NULL,
  model_name = "google/siglip-so400m-patch14-384",
  normalize  = TRUE,
  batch_size = 32L
) {

  # Phase 1: Skeleton implementation
  # Full implementation in Phase 2

  stop(
    ".encode_images_siglip() is not yet implemented.\n",
    "Full implementation coming in Phase 2 (vdic v1.2.0)."
  )
}
