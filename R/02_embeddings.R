#' Download word or multi-modal embeddings
#'
#' @description
#' Downloads pre-trained word embeddings (FastText, word2vec, GloVe) from public
#' repositories, or triggers the download and local caching of the SigLIP
#' multi-modal model from the Hugging Face Hub.
#'
#' Word embedding files are required to build text-based vec-tionaries but are NOT
#' included in the package to keep it lightweight. SigLIP is downloaded and cached
#' once by the Python \code{transformers} library (usually to
#' \file{~/.cache/huggingface/}); subsequent calls reuse the cached copy.
#'
#' @param language Language code: \code{"pt"} (Portuguese), \code{"en"} (English),
#'   or \code{"es"} (Spanish). Ignored when \code{model = "siglip"}.
#' @param model Embedding model to download:
#'   \itemize{
#'     \item \code{"fasttext"}: FastText Common Crawl embeddings (all languages).
#'     \item \code{"word2vec"}: Google News word2vec (English) or PT2Vec (Portuguese).
#'     \item \code{"glove"}: GloVe 6B trained on Wikipedia + Gigaword (English only).
#'     \item \code{"siglip"}: Google SigLIP multi-modal model
#'       (\code{google/siglip-so400m-patch14-384}, 1152 dimensions). Downloads and
#'       caches the model via the Python \code{transformers} library. Requires
#'       \code{reticulate} and a Python environment with
#'       \code{transformers}, \code{torch}, \code{Pillow}, and
#'       \code{sentencepiece} installed. Returns the Hugging Face model ID string
#'       to pass directly to \code{\link{vectionary_builder}(embeddings = ...)}.
#'   }
#' @param dimensions Embedding vector dimensionality (default: 300). Only used for
#'   word embedding models; ignored for \code{"siglip"}.
#' @param destination Directory to save the downloaded embeddings file
#'   (default: \code{vdic_data/} in the current working directory). Only used for
#'   word embedding models; ignored for \code{"siglip"} (HF cache is used instead).
#' @param force If \code{TRUE}, re-download even if the file already exists.
#'   For \code{"siglip"}, clears the session-level model cache so the model is
#'   re-loaded from the HF cache on next use.
#'
#' @return For word embedding models: path to the downloaded (and decompressed)
#'   embeddings file. For \code{"siglip"}: the Hugging Face model ID string
#'   (\code{"google/siglip-so400m-patch14-384"}), which can be passed to
#'   \code{vectionary_builder(embeddings = ..., modality = "multimodal")}.
#' @export
#'
#' @examples
#' \dontrun{
#' # Download Portuguese FastText embeddings
#' download_embeddings("pt", "fasttext")
#'
#' # Download to specific directory
#' download_embeddings("pt", "fasttext", destination = "~/my_embeddings")
#'
#' # Download and cache SigLIP for multi-modal vectionaries (requires Python)
#' download_embeddings(model = "siglip")
#' }

#- Download Embeddings ----
download_embeddings <- function(
  language    = c("pt", "en", "es"),
  model       = c("fasttext", "word2vec", "glove", "siglip"),
  dimensions  = 300,
  destination = NULL,
  force       = FALSE
) {

  language <- match.arg(language)
  model    <- match.arg(model)

  ##- SigLIP (multi-modal, Hugging Face) ----
  if (model == "siglip") {
    model_name <- "google/siglip-so400m-patch14-384"

    cli_h1("Downloading SigLIP multi-modal model")
    cli_alert_info("Model: {.val {model_name}}")
    cli_alert_info("Dimensions: 1152 (shared text-image space)")
    cli_alert_info(paste0(
      "The model is downloaded once and cached by the Python ",
      "{.pkg transformers} library (usually in {.path ~/.cache/huggingface/})."
    ))
    cli_alert_warning(paste0(
      "This may take several minutes on the first download (~3 GB). ",
      "Subsequent calls reuse the local cache."
    ))

    # Validate Python / reticulate dependencies before attempting any download
    .check_siglip_deps()

    # Optionally clear the session-level cache so the model is re-loaded
    if (force) {
      cache_key <- paste0("siglip_", model_name)
      .vdic_env[[cache_key]] <- NULL
      cli_alert_info("Session-level model cache cleared.")
    }

    # .load_siglip_model() downloads and caches via transformers; prints progress
    .load_siglip_model(model_name)

    cli_alert_success(paste0(
      "SigLIP cached successfully. ",
      "Pass {.val {model_name}} as the {.arg embeddings} argument to ",
      "{.fn vectionary_builder} with {.code modality = \"multimodal\"}."
    ))

    return(invisible(model_name))
  }
  
  # Default destination: vdic_data/ in working directory
  if (is.null(destination)) {
    destination <- file.path(getwd(), "vdic_data")
    dir.create(destination, showWarnings = FALSE, recursive = TRUE)
  }

  ##- Construct URL based on source ----
  ###- FastText ----
  if (model == "fasttext") {
    url <- sprintf(
      "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.%s.%d.vec.gz",
      language, dimensions
    )
    filename <- sprintf("cc.%s.%d.vec.gz", language, dimensions)

  ###- Word2Vec ----
  } else if (model == "word2vec") {
    if (!language %in% c("en", "pt")) {
      stop("Word2Vec only available for English or Portuguese (PT2Vec). Use 'fasttext' for other languages.")
    }
    # Google News word2vec (3M words, 300d, ~1.5 GB compressed):
    url <- "https://huggingface.co/LoganKilpatrick/GoogleNews-vectors-negative300/resolve/main/GoogleNews-vectors-negative300.bin.gz"
    filename <- "GoogleNews-vectors-negative300.bin.gz"
    if (language == "pt") {
      # Using PT2Vec
      url <- "http://pt2vec.inesctec.pt/files/dataset.zip"
      filename <- "PT2Vec-300.bin.gz"
    }

  } else if (model == "glove") {
  
  ###- GloVe ----
    if (language != "en") {
      stop("glove only available for English. Use 'fasttext' for other languages.")
    }
    # Using GloVe 6B (trained on Wikipedia + Gigaword)
    url <- "https://nlp.stanford.edu/data/glove.6B.zip"
    filename <- "glove.6B.zip"
  }

  dest_file <- file.path(destination, filename)

  # Check if already exists
  if (file.exists(dest_file) && !force) {
    cli_alert_info("Embeddings already exist at: {.path {dest_file}}")
    cli_alert_info("Use {.code force = TRUE} to re-download")
    return(dest_file)
  }

  # Download with progress
  cli_h1("Downloading {model} {language} embeddings")
  cli_alert_info("Dimensions: {dimensions}")
  cli_alert_info("Destination: {.path {dirname(dest_file)}}")
  cli_alert_warning("This may take 2-10 minutes depending on your connection")

  cli_progress_step(
    "Downloading {.file {filename}}",
    msg_done = "Downloaded {.file {filename}}",
    spinner = TRUE
  )

  old_timeout <- getOption("timeout")
  options(timeout = 1800)
  on.exit(options(timeout = old_timeout), add = TRUE)

  tryCatch({
    download.file(
      url = url,
      destfile = dest_file,
      mode = "wb",
      method = "libcurl",
      quiet = TRUE
    )
  }, error = function(e) {
    cli_progress_done()
    cli_abort("Download failed: {e$message}. Please check your internet connection.")
  })

  cli_progress_done()

  ##- Verify download ---- 
  if (!file.exists(dest_file)) {
    cli_abort("Download failed. File not created.")
  }

  file_size_mb <- file.info(dest_file)$size / 1024^2
  cli_alert_success("Downloaded successfully: {.val {round(file_size_mb, 1)}} MB")

  # Decompress if .gz
  if (grepl("\\.gz$", dest_file)) {
    cli_progress_step(
      "Decompressing {.file {basename(dest_file)}}",
      msg_done = "Decompressed {.file {basename(dest_file)}}",
      spinner = TRUE
    )

    decompressed_file <- sub("\\.gz$", "", dest_file)

    # Use R.utils::gunzip
    R.utils::gunzip(dest_file, decompressed_file, remove = FALSE, overwrite = TRUE)

    cli_progress_done()

    decompressed_size_mb <- file.info(decompressed_file)$size / 1024^2
    cli_alert_success("Decompressed size: {.val {round(decompressed_size_mb, 1)}} MB")

    # Convert .bin to .vec (text format) for compatibility with text-based parsers
    if (grepl("\\.bin$", decompressed_file)) {
      vec_file <- sub("\\.bin$", ".vec", decompressed_file)
      .convert_bin_to_vec(decompressed_file, vec_file, verbose = TRUE)
      decompressed_file <- vec_file
    }

    cli_alert_info("Saved to: {.path {decompressed_file}}")
    return(decompressed_file)
  }

  # Handle zip files (for GloVe)
  if (grepl("\\.zip$", dest_file)) {
    cli_progress_step("Extracting archive", spinner = TRUE)
    unzip(dest_file, exdir = destination)
    cli_progress_done()

    cli_alert_success("Extracted to: {.path {destination}}")

    # Return path to specific dimension file
    extracted_file <- file.path(destination, sprintf("glove.6B.%dd.txt", dimensions))
    if (file.exists(extracted_file)) {
      return(extracted_file)
    } else {
      cli_alert_info("Available files:")
      for (f in list.files(destination, pattern = "glove")) {
        cli_alert_info("  {.file {f}}")
      }
      return(destination)
    }
  }

  return(dest_file)
}


#' Convert word2vec binary (.bin) to text (.vec) format
#'
#' @description
#' Converts a word2vec binary file to text-based .vec format (FastText-compatible).
#' The binary format stores vectors as raw floats, while .vec uses space-separated
#' text. This conversion allows all existing text-based parsers to work with
#' word2vec embeddings.
#'
#' Binary format: header line "vocab_size n_dims\\n", then for each word:
#' word bytes terminated by space (0x20), then n_dims x 4-byte little-endian floats,
#' then optional newline (0x0A).
#'
#' @param bin_path Path to the input .bin file
#' @param vec_path Path for the output .vec file
#' @param verbose Print progress messages
#'
#' @return Path to the converted .vec file (invisibly)
#'
#' @keywords internal
.convert_bin_to_vec <- function(bin_path, vec_path, verbose = TRUE) {

  if (verbose) {
    cli_progress_step(
      "Converting binary to text format (.bin -> .vec)",
      msg_done = "Converted to text format",
      spinner = TRUE
    )
  }

  #- Read binary header ----
  # Header is a text line: "vocab_size n_dims\n"
  bin_con <- file(bin_path, "rb")
  on.exit(close(bin_con), add = TRUE)

  # Read header line byte-by-byte until newline
  header_bytes <- raw(0)
  repeat {
    byte <- readBin(bin_con, "raw", n = 1)
    if (length(byte) == 0) stop("Unexpected end of file reading header")
    if (byte == as.raw(0x0A)) break  # newline
    header_bytes <- c(header_bytes, byte)
  }

  header_str <- rawToChar(header_bytes)
  header_parts <- strsplit(trimws(header_str), "\\s+")[[1]]
  n_words <- as.integer(header_parts[1])
  n_dims <- as.integer(header_parts[2])

  if (verbose) {
    cli_alert_info("Binary file: {n_words} words, {n_dims} dimensions")
  }

  #- Convert entries ----
  vec_con <- file(vec_path, "w", encoding = "UTF-8")
  on.exit(close(vec_con), add = TRUE)

  # Write header in FastText .vec format
  writeLines(paste(n_words, n_dims), vec_con)

  words_converted <- 0
  bytes_per_vector <- n_dims * 4  # 4 bytes per float32

  for (i in seq_len(n_words)) {
    # Read word: bytes until space (0x20)
    word_bytes <- raw(0)
    repeat {
      byte <- readBin(bin_con, "raw", n = 1)
      if (length(byte) == 0) break  # EOF
      if (byte == as.raw(0x20)) break  # space = word terminator
      if (byte == as.raw(0x0A)) next  # skip newlines between entries
      word_bytes <- c(word_bytes, byte)
    }

    if (length(word_bytes) == 0) next  # skip empty words

    word <- tryCatch(
      rawToChar(word_bytes),
      error = function(e) NULL
    )
    if (is.null(word)) next  # skip words with invalid encoding

    # Read vector: n_dims x 4-byte little-endian floats
    vec_raw <- readBin(bin_con, "numeric", n = n_dims, size = 4, endian = "little")

    if (length(vec_raw) < n_dims) break  # EOF mid-vector

    # Write text line: "word v1 v2 v3 ..."
    vec_str <- paste(vec_raw, collapse = " ")
    writeLines(paste(word, vec_str), vec_con)

    words_converted <- words_converted + 1

    # Progress update every 500k words
    if (verbose && words_converted %% 500000 == 0) {
      pct <- round(100 * words_converted / n_words, 1)
      cli_alert_info("Converted {words_converted}/{n_words} words ({pct}%)")
    }
  }

  if (verbose) {
    cli_progress_done()
    vec_size_mb <- file.info(vec_path)$size / 1024^2
    cli_alert_success("Converted {words_converted} words to text format ({round(vec_size_mb, 1)} MB)")
  }

  invisible(vec_path)
}
