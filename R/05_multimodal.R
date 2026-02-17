#- Multi-Modal Vectionary Pipeline ----
#
# This file implements the multi-modal branch of vectionary_builder() and the
# analyze_image() function. It is called from 03_builder.R after all shared
# input validation has run.
#
# Pipeline:
#   1. Check Python/reticulate dependencies (.check_siglip_deps)
#   2. Encode dictionary words with SigLIP text encoder -> 512-dim matrix
#   3. Learn axes via the same solvers used by the text pipeline
#      (.solve_axis_ridge, .solve_axis_glmnet, .solve_axis_duan from 03_builder.R)
#   4. Package into Vec-tionary S3 object (modality = "multimodal", embedding_dim = 512)
#   5. word_projections = NULL (no fixed vocabulary in SigLIP)
#
# analyze_image() later encodes image files the same way and projects them onto
# the learned axes.
#
# Python dependency: SigLIP is a neural network model with no pre-computed
# vocabulary file. All embeddings are computed on-the-fly via reticulate +
# Python transformers. Text-only vectionaries (03_builder.R) require no Python.


#-- Internal: Build Multi-Modal Axes ----

#' Build multi-modal axes using SigLIP text encoder (internal)
#'
#' @description
#' Called from [vectionary_builder()] when `modality = "multimodal"`, after all
#' shared input validation has run. Encodes dictionary words using SigLIP's text
#' encoder, then calls the same axis solvers used by the text pipeline. Returns a
#' Vec-tionary object with `modality = "multimodal"` and `word_projections = NULL`
#' (since SigLIP has no fixed vocabulary; images are projected at analysis time).
#'
#' @param dictionary Data frame with 'word' and dimension columns, already validated
#'   and binary-converted by [vectionary_builder()]
#' @param dimensions Character vector of dimension names
#' @param binary_word Logical passed through from [vectionary_builder()]
#' @param method Regularization method ("ridge", "elastic_net", "lasso", "duan")
#' @param l1_ratio Elastic net mixing parameter
#' @param lambda Regularization parameter (numeric or NULL if use_gcv)
#' @param use_gcv Logical. If TRUE, select lambda via GCV
#' @param lambda_range Numeric vector of lambda candidates, or NULL
#' @param min_validity Minimum validity threshold for lambda selection
#' @param expand_vocab Integer or NULL
#' @param save_path Where to save the resulting vectionary
#' @param verbose Logical
#' @param seed Integer seed for reproducibility
#'
#' @return A Vec-tionary S3 object with:
#'   - `modality = "multimodal"`
#'   - `embedding_dim = 512`
#'   - `word_projections = NULL`
#'   - `image_projections = NULL` (populated by [analyze_image()] at analysis time)
#'
#' @keywords internal
.build_multimodal_axes <- function(
  dictionary,
  dimensions,
  binary_word,
  method,
  l1_ratio,
  lambda,
  use_gcv,
  lambda_range,
  min_validity,
  expand_vocab,
  save_path,
  verbose,
  seed
) {

  # Phase 1: Skeleton — full implementation in Phase 2 (vdic v1.2.0)
  # Phase 2 will:
  #   1. Call .check_siglip_deps()
  #   2. Load SigLIP model via .load_siglip_model()
  #   3. Encode dictionary$word with .encode_text_siglip() -> embedding matrix (n x 512)
  #   4. Select lambda via GCV or grid search on the matrix (not a file)
  #   5. Call .solve_axis_ridge() / .solve_axis_glmnet() / .solve_axis_duan()
  #      directly with the matrix — same functions as the text pipeline
  #   6. Package into Vec-tionary with modality = "multimodal"

  stop(
    "Multi-modal vectionaries (modality = 'multimodal') are not yet implemented.\n\n",
    "Full implementation coming in Phase 2 (vdic v1.2.0).\n\n",
    "This requires Python with the transformers library:\n",
    "  install.packages('reticulate')\n",
    "  reticulate::py_install(c('transformers', 'torch', 'Pillow'))"
  )
}


#- Analyze Images with Multi-Modal Vectionary ----

#' Analyze Images with a Multi-Modal Vec-tionary
#'
#' @description
#' Scores a set of image files on the semantic dimensions of a multi-modal
#' vectionary. Each image is encoded to a 512-dimensional SigLIP vector and
#' projected onto the vectionary's learned axes.
#'
#' The vectionary must have been built with `modality = "multimodal"` in
#' [vectionary_builder()], meaning its axes were learned from dictionary words
#' encoded via SigLIP's text encoder. Because SigLIP maps both text and images
#' into the same 512-dimensional space, images can be scored by those same axes
#' — no labeled image training data required.
#'
#' **Python required:** Encoding images requires Python with the `transformers`,
#' `torch`, and `Pillow` libraries, accessible via the `reticulate` package.
#' Set up once with:
#' ```r
#' install.packages("reticulate")
#' reticulate::install_miniconda()   # skip if Python is already configured
#' reticulate::py_install(c("transformers", "torch", "Pillow"))
#' ```
#'
#' @param vect A Vec-tionary object with `modality = "multimodal"`, built with
#'   [vectionary_builder()] using `modality = "multimodal"`.
#' @param images Character vector of image file paths to analyze. Accepts JPEG,
#'   PNG, BMP, and other formats supported by Pillow.
#' @param model_name SigLIP model ID on Hugging Face Hub (default:
#'   `"google/siglip-so400m-patch14-384"`). The model is downloaded and cached
#'   by the `transformers` library on first use (approximately 800 MB). Must
#'   match the model used when building the vectionary.
#' @param batch_size Integer. Number of images to encode per batch (default: 32).
#'   Reduce if you run out of memory; increase for faster processing on GPU.
#'
#' @return Data frame with one row per image and one column per dimension, plus
#'   an `image` column with the original file path. Scores are cosine-similarity-
#'   based projections — higher values indicate stronger semantic alignment with
#'   the dimension's dictionary words.
#'
#' @examples
#' \dontrun{
#' # Build a multi-modal vectionary from a text dictionary
#' dictionary <- data.frame(
#'   word = c("protect", "care", "help", "harm", "hurt", "violence"),
#'   care = c(1, 1, 1, 0, 0, 0),
#'   harm = c(0, 0, 0, 1, 1, 1)
#' )
#'
#' vect <- vectionary_builder(
#'   dictionary = dictionary,
#'   embeddings = "siglip",
#'   modality   = "multimodal"
#' )
#'
#' # Score image files
#' result <- analyze_image(vect, images = c("hospital.jpg", "war_scene.jpg"))
#'
#' result
#' #>           image  care  harm
#' #> 1   hospital.jpg 0.823 0.134
#' #> 2  war_scene.jpg 0.112 0.756
#' }
#'
#' @export
analyze_image <- function(
  vect,
  images,
  model_name = "google/siglip-so400m-patch14-384",
  batch_size = 32L
) {

  if (!inherits(vect, "Vec-tionary")) {
    stop("vect must be a Vec-tionary object")
  }

  if (is.null(vect$modality) || vect$modality != "multimodal") {
    stop(
      "analyze_image() requires a multi-modal vectionary.\n",
      "Current vectionary has modality: '", vect$modality %||% "text", "'\n\n",
      "Build one with:\n",
      "  vectionary_builder(..., modality = 'multimodal')"
    )
  }

  # Phase 1: Skeleton — full implementation in Phase 2 (vdic v1.2.0)
  # Phase 2 will:
  #   1. Call .check_siglip_deps()
  #   2. Load SigLIP model via .load_siglip_model()
  #   3. Encode images with .encode_images_siglip() -> matrix (n x 512)
  #   4. Project each row onto vect$axes via matrix multiplication
  #   5. Return data frame: image path + one column per dimension

  stop(
    "analyze_image() is not yet implemented.\n\n",
    "Full implementation coming in Phase 2 (vdic v1.2.0).\n\n",
    "This requires Python with the transformers library:\n",
    "  install.packages('reticulate')\n",
    "  reticulate::py_install(c('transformers', 'torch', 'Pillow'))"
  )
}
