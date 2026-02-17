#- Multi-Modal Vectionary Pipeline ----
#
# This file implements the multi-modal branch of vectionary_builder() and the
# analyze_image() function. It is called from 03_builder.R after all shared
# input validation has run.
#
# Pipeline for .build_multimodal_axes():
#   1. Check Python/reticulate dependencies (.check_siglip_deps)
#   2. Encode dictionary words with SigLIP text encoder -> n x 1152 matrix
#   3. Select lambda via GCV using .gcv_select_lambda(W, y) directly (no file)
#   4. Learn axes with the same solvers as the text pipeline:
#        .solve_axis_ridge(), .solve_axis_elastic_net(), .solve_axis_duan()
#   5. Package into Vec-tionary S3 object (modality = "multimodal", embedding_dim = 1152)
#      word_projections = NULL (no fixed vocabulary in SigLIP)
#
# Pipeline for analyze_image():
#   1. Check Python/reticulate dependencies
#   2. Encode images with SigLIP image encoder -> n x 1152 matrix
#   3. Project each row onto vect$axes via matrix-vector multiplication
#   4. Return data frame: image path + one column per dimension
#
# Python dependency: SigLIP is a neural network model with no pre-computed
# vocabulary file. All embeddings are computed on-the-fly via reticulate +
# Python transformers. Text-only vectionaries (03_builder.R) require no Python.


#-- Internal: Build Multi-Modal Axes ----

#' Build multi-modal axes using SigLIP text encoder (internal)
#'
#' @description
#' Called from [vectionary_builder()] when \code{modality = "multimodal"}, after
#' all shared input validation has run. Encodes dictionary words using SigLIP's
#' text encoder, selects lambda via GCV, then calls the same axis solvers used by
#' the text pipeline. Returns a Vec-tionary object with \code{modality =
#' "multimodal"} and \code{word_projections = NULL} (SigLIP has no fixed vocabulary;
#' images are projected at analysis time by [analyze_image()]).
#'
#' @param dictionary Data frame with 'word' and dimension columns, already
#'   validated and binary-converted by [vectionary_builder()]
#' @param dimensions Character vector of dimension names
#' @param binary_word Logical. Passed through for metadata recording only
#' @param method Regularization method ("ridge", "elastic_net", "lasso", "duan")
#' @param l1_ratio Elastic net mixing parameter
#' @param lambda Regularization parameter (numeric) or NULL when use_gcv = TRUE
#' @param use_gcv Logical. If TRUE, selects lambda via GCV per dimension
#' @param lambda_range Numeric vector of lambda candidates for grid search, or NULL
#' @param min_validity Minimum validity threshold for grid search lambda selection
#' @param expand_vocab Integer or NULL. Silently ignored for multimodal
#' @param save_path Where to save the resulting vectionary (character or NULL)
#' @param verbose Logical. If TRUE, prints progress messages
#' @param seed Integer seed for reproducibility (used by duan method)
#'
#' @return A Vec-tionary S3 object with \code{modality = "multimodal"}
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

  #-- Warn about text-specific arguments that don't apply ----
  if (!is.null(expand_vocab)) {
    cli::cli_alert_warning(
      "expand_vocab is ignored for modality='multimodal' (SigLIP has no fixed vocabulary)"
    )
  }

  #-- Step 1: Check Python dependencies ----
  .check_siglip_deps()

  model_name <- "google/siglip-so400m-patch14-384"

  if (verbose) {
    cli::cli_h1("Building Multi-Modal Vec-tionary")
    cli::cli_ul(c(
      "Dictionary: {nrow(dictionary)} word{?s}, {length(dimensions)} dimension{?s} ({paste(dimensions, collapse = ', ')})",
      "Model: {model_name}",
      "Method: {method}{if (method == 'duan') ' (Duan et al., 2025)' else ''}"
    ))
    cli::cli_text("")
  }

  t_start <- Sys.time()

  #-- Step 2: Load SigLIP model ----
  mm <- .load_siglip_model(model_name)

  #-- Step 3: Encode dictionary words ----
  if (verbose) cli::cli_h2("Encoding dictionary words with SigLIP")
  t_step <- Sys.time()

  words        <- unique(dictionary$word)
  emb_matrix   <- .encode_text_siglip(words, model = mm)  # n_words x 1152

  if (verbose) {
    t_elapsed <- round(as.numeric(difftime(Sys.time(), t_step, units = "secs")), 1)
    cli::cli_alert_success("Encoded {nrow(emb_matrix)} word{?s} ({t_elapsed}s)")
  }

  # Convert matrix rows to named list — format expected by axis solvers
  # (.solve_axis_ridge etc. do `W <- do.call(rbind, word_vectors)` internally)
  word_vectors <- lapply(seq_len(nrow(emb_matrix)), function(i) emb_matrix[i, ])
  names(word_vectors) <- words

  #-- Step 4: Lambda selection ----
  if (use_gcv && method == "ridge") {
    if (verbose) cli::cli_h2("Selecting lambda via GCV")
    t_step <- Sys.time()

    # Select per dimension then aggregate with median (mirrors text pipeline)
    lambdas_per_dim <- vapply(dimensions, function(dim) {
      scores <- dictionary[[dim]]
      names(scores) <- dictionary$word
      y <- as.numeric(scores[words])
      ok <- !is.na(y)
      .gcv_select_lambda(emb_matrix[ok, , drop = FALSE], y[ok])
    }, numeric(1L))

    lambda <- stats::median(lambdas_per_dim)

    if (verbose) {
      t_elapsed <- round(as.numeric(difftime(Sys.time(), t_step, units = "secs")), 1)
      cli::cli_alert_success("Lambda = {round(lambda, 4)} ({t_elapsed}s)")
    }
  } else if (!is.null(lambda_range)) {
    # Grid search: select best lambda per dimension, take median
    if (verbose) cli::cli_h2("Selecting lambda via grid search")
    lambdas_per_dim <- vapply(dimensions, function(dim) {
      scores <- dictionary[[dim]]
      names(scores) <- dictionary$word
      y  <- as.numeric(scores[words])
      ok <- !is.na(y)
      gcv_vals <- vapply(lambda_range, function(lam) {
        .gcv_select_lambda(emb_matrix[ok, , drop = FALSE], y[ok])
      }, numeric(1L))
      lambda_range[which.min(gcv_vals)]
    }, numeric(1L))
    lambda <- stats::median(lambdas_per_dim)
  }

  #-- Step 5: Learn axes ----
  if (verbose) {
    cli::cli_h2("Learning axes")
    cli::cli_alert_info("Method: {method}{if (!is.null(lambda)) paste0(' | Lambda: ', round(lambda, 4)) else ''}")
  }
  t_step <- Sys.time()

  axes <- vector("list", length(dimensions))
  names(axes) <- dimensions

  for (dim in dimensions) {
    scores <- dictionary[[dim]]
    names(scores) <- dictionary$word
    word_scores <- scores[words]

    # Filter to non-NA scores (should be all, but be safe)
    valid      <- !is.na(word_scores)
    wv_dim     <- word_vectors[valid]
    ws_dim     <- word_scores[valid]

    axes[[dim]] <- switch(method,
      ridge       = .solve_axis_ridge(wv_dim, ws_dim, lambda),
      elastic_net = .solve_axis_elastic_net(wv_dim, ws_dim, lambda, l1_ratio),
      lasso       = .solve_axis_elastic_net(wv_dim, ws_dim, lambda, l1_ratio = 1.0),
      duan        = .solve_axis_duan(wv_dim, ws_dim, dim_name = dim, seed = seed)
    )
  }

  if (verbose) {
    t_elapsed <- round(as.numeric(difftime(Sys.time(), t_step, units = "secs")), 1)
    cli::cli_alert_success("Done ({t_elapsed}s)")
  }

  #-- Step 6: Package into Vec-tionary ----
  vect <- structure(
    list(
      axes              = axes,
      word_projections  = NULL,          # no fixed vocabulary for SigLIP
      image_projections = NULL,          # populated by analyze_image() at analysis time
      dimensions        = dimensions,
      modality          = "multimodal",
      embedding_dim     = ncol(emb_matrix),
      metadata          = list(
        method          = method,
        l1_ratio        = if (method == "elastic_net") l1_ratio else if (method == "lasso") 1.0 else NULL,
        binary_word     = binary_word,
        lambda          = lambda,
        model_name      = model_name,
        seed_words      = words,
        seed_words_count = length(words),
        words_encoded   = nrow(emb_matrix),
        seed            = seed,
        build_date      = Sys.time(),
        package_version = as.character(utils::packageVersion("vdic"))
      )
    ),
    class = "Vec-tionary"
  )

  #-- Step 7: Save if requested ----
  if (!is.null(save_path)) {
    if (dir.exists(save_path)) {
      save_path <- file.path(save_path, "vectionary_multimodal.rds")
    }
    saveRDS(vect, save_path)
    if (verbose) cli::cli_alert_success("Saved to {save_path}")
  }

  if (verbose) {
    t_total <- round(as.numeric(difftime(Sys.time(), t_start, units = "secs")), 1)
    cli::cli_h2("Summary")
    cli::cli_alert_success("Multi-modal Vec-tionary built successfully! ({t_total}s)")
    cli::cli_alert_info("{length(words)} words encoded | {length(dimensions)} dimension{?s}")
  }

  vect
}


#- Analyze Images with Multi-Modal Vectionary ----

#' Analyze Images with a Multi-Modal Vec-tionary
#'
#' @description
#' Scores a set of image files on the semantic dimensions of a multi-modal
#' vectionary. Each image is encoded to a 1152-dimensional SigLIP vector and
#' projected onto the vectionary's learned axes.
#'
#' The vectionary must have been built with \code{modality = "multimodal"} in
#' [vectionary_builder()], meaning its axes were learned from dictionary words
#' encoded via SigLIP's text encoder. Because SigLIP maps both text and images
#' into the same 1152-dimensional space, images can be scored by those same axes
#' — no labeled image training data required.
#'
#' **Python required:** Encoding images requires Python with the \code{transformers},
#' \code{torch}, and \code{Pillow} libraries, accessible via the \code{reticulate}
#' package. Set up once with:
#' \preformatted{
#' install.packages("reticulate")
#' reticulate::install_miniconda()   # skip if Python is already configured
#' reticulate::py_install(c("transformers", "torch", "Pillow", "sentencepiece"))
#' }
#'
#' @param vect A Vec-tionary object with \code{modality = "multimodal"}, built with
#'   [vectionary_builder()] using \code{modality = "multimodal"}.
#' @param images Character vector of image file paths to analyze. Accepts JPEG,
#'   PNG, BMP, WebP, and other formats supported by Pillow.
#' @param model_name SigLIP model ID on Hugging Face Hub (default:
#'   \code{"google/siglip-so400m-patch14-384"}). The model is downloaded and
#'   cached by the \code{transformers} library on first use (approximately 800 MB).
#'   Must match the model used when building the vectionary.
#' @param batch_size Integer. Number of images to encode per batch (default: 32).
#'   Reduce if you run out of memory; increase for faster processing on GPU.
#' @param alpha Significance level for one-tailed topic classification (e.g.
#'   \code{0.05}). When set, appends logical \code{_topic} columns to the
#'   result: an image is flagged when its score exceeds
#'   \eqn{\bar{x} + t_{1-\alpha,\, n-1} \cdot s}, where \eqn{\bar{x}} and
#'   \eqn{s} are the corpus mean and SD of image scores for that dimension.
#'   Requires at least 2 images. Default \code{NULL} (no classification).
#'
#' @return Data frame with one row per image and one column per dimension, plus
#'   an \code{image} column with the original file path. Scores are
#'   cosine-similarity-based projections — higher values indicate stronger
#'   semantic alignment with the dimension's dictionary words.
#'   When \code{alpha} is set, logical \code{_topic} columns are appended and
#'   the data frame carries \code{"threshold"} and \code{"alpha"} attributes
#'   with the per-dimension cutoffs.
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
  batch_size = 32L,
  alpha      = NULL
) {

  if (!inherits(vect, "Vec-tionary")) {
    stop("vect must be a Vec-tionary object")
  }

  # Use [[ to bypass the custom $ dispatcher — old vectionaries lack modality
  if (is.null(vect[["modality"]]) || vect[["modality"]] != "multimodal") {
    stop(
      "analyze_image() requires a multi-modal vectionary.\n",
      "Current vectionary has modality: '", vect[["modality"]] %||% "text", "'\n\n",
      "Build one with:\n",
      "  vectionary_builder(..., modality = 'multimodal')"
    )
  }

  if (length(images) == 0L) stop("images must be a non-empty character vector")

  #-- Validate alpha ----
  if (!is.null(alpha)) {
    if (!is.numeric(alpha) || length(alpha) != 1L || alpha <= 0 || alpha >= 1) {
      stop("alpha must be a single number in (0, 1)")
    }
    if (length(images) < 2L) {
      warning("alpha ignored: topic classification requires at least 2 images")
      alpha <- NULL
    }
  }

  #-- Check dependencies ----
  .check_siglip_deps()

  #-- Load model ----
  mm <- .load_siglip_model(model_name)

  #-- Encode images ----
  emb_matrix <- .encode_images_siglip(
    image_paths = images,
    model       = mm,
    batch_size  = batch_size
  )  # n_images x 1152

  #-- Project onto learned axes ----
  # Each axis is a 1152-dim vector; dot product with image embedding = score
  dims   <- vect$dimensions
  scores <- lapply(dims, function(dim) as.numeric(emb_matrix %*% vect$axes[[dim]]))
  names(scores) <- dims

  #-- Build result data frame ----
  result        <- as.data.frame(scores)
  result$image  <- images
  col_order     <- c("image", dims)

  #-- Topic classification (one-tailed t-test) ----
  if (!is.null(alpha)) {
    thresholds <- setNames(numeric(length(dims)), dims)

    for (dim in dims) {
      vals    <- scores[[dim]]
      n_valid <- sum(!is.na(vals))

      if (n_valid < 2L) {
        thresholds[[dim]]                  <- NA_real_
        result[[paste0(dim, "_topic")]]    <- rep(FALSE, length(vals))
      } else {
        mu               <- mean(vals, na.rm = TRUE)
        sigma            <- stats::sd(vals, na.rm = TRUE)
        t_crit           <- stats::qt(1 - alpha, df = n_valid - 1L)
        thresholds[[dim]]                  <- mu + t_crit * sigma
        result[[paste0(dim, "_topic")]]    <- !is.na(vals) & vals > thresholds[[dim]]
      }
    }

    attr(result, "threshold") <- thresholds
    attr(result, "alpha")     <- alpha
    col_order <- c(col_order, paste0(dims, "_topic"))
  }

  result[col_order]
}


#- Analyze Text with a Vec-tionary ----

#' Analyze Text with a Vec-tionary
#'
#' @description
#' Scores a character vector of texts on the semantic dimensions of a
#' vec-tionary. Returns a data frame with one row per text and one column per
#' dimension — the same structure as [analyze_image()].
#'
#' Two analysis paths are supported depending on the vec-tionary's modality:
#'
#' \describe{
#'   \item{Text vectionary (\code{modality = "text"} or legacy)}{
#'     Each text is tokenized, tokens are looked up in the pre-computed
#'     \code{word_projections} table, and the mean projection score is returned.
#'     This path requires no Python and works offline.}
#'   \item{Multi-modal vectionary (\code{modality = "multimodal"})}{
#'     Each text is encoded to a 1152-dimensional SigLIP vector via the text
#'     encoder and projected onto the vectionary axes — the same axes used by
#'     [analyze_image()]. This enables direct comparison of text and image
#'     scores on the same scale. Requires Python with \code{transformers},
#'     \code{torch}, \code{Pillow}, and \code{sentencepiece} via
#'     \code{reticulate}.}
#' }
#'
#' For richer text analysis (multiple metrics, topic classification) use
#' [vectionary_analyze()] with a text vectionary.
#'
#' @param vect A Vec-tionary object built with [vectionary_builder()].
#' @param text Character vector of texts to analyze. Each element is treated as
#'   one document.
#' @param model_name SigLIP model ID on Hugging Face Hub (default:
#'   \code{"google/siglip-so400m-patch14-384"}). Only used for multimodal
#'   vectionaries. Must match the model used when building the vectionary.
#' @param batch_size Integer. Number of texts to encode per batch (default:
#'   64L). Only used for multimodal vectionaries.
#'
#' @return Data frame with one row per text and one column per dimension, plus
#'   a \code{text} column containing the original input strings.
#'
#' @seealso [vectionary_analyze()] for multi-metric text analysis with topic
#'   classification; [analyze_image()] for image analysis.
#'
#' @examples
#' \dontrun{
#' # ---- Text vectionary (no Python needed) ----
#' vect <- readRDS("my_vectionary.rds")
#' result <- analyze_text(vect, c("We must protect the vulnerable",
#'                                "Violence erupted in the streets"))
#' result
#' #>                             text  care  harm
#' #> 1  We must protect the vulnerab... 0.623 0.112
#' #> 2  Violence erupted in the stre... 0.089 0.741
#'
#' # ---- Multi-modal vectionary (Python required) ----
#' mm_vect <- vectionary_builder(dictionary, embeddings = "siglip",
#'                               modality = "multimodal")
#' result <- analyze_text(mm_vect, c("a caring nurse", "an act of violence"))
#' }
#'
#' @export
analyze_text <- function(
  vect,
  text,
  model_name = "google/siglip-so400m-patch14-384",
  batch_size = 64L
) {

  if (!inherits(vect, "Vec-tionary")) {
    stop("vect must be a Vec-tionary object")
  }

  if (!is.character(text) || length(text) == 0L) {
    stop("text must be a non-empty character vector")
  }

  modality <- vect[["modality"]] %||% "text"

  if (modality == "multimodal") {

    #-- Multimodal path: encode text via SigLIP then project ----
    .check_siglip_deps()

    mm         <- .load_siglip_model(model_name)
    emb_matrix <- .encode_text_siglip(text, model = mm, batch_size = batch_size)

    scores <- lapply(vect$dimensions, function(dim) {
      as.numeric(emb_matrix %*% vect$axes[[dim]])
    })
    names(scores) <- vect$dimensions

  } else {

    #-- Text path: word-lookup projection (mean score per document) ----
    if (is.null(vect$word_projections)) {
      stop(
        "This text vectionary has no word_projections table.\n",
        "Build it with vectionary_builder() using a word embeddings file."
      )
    }

    scores <- .batch_metric(text, vect$word_projections, vect$dimensions, "mean")
  }

  #-- Return data frame ----
  result       <- as.data.frame(scores)
  result$text  <- text
  result[c("text", vect$dimensions)]
}
