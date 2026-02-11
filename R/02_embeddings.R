#' Download word embeddings 
#'
#' @description
#' Downloads pre-trained word embeddings (FastText, word2vec, etc.) from public
#' repositories. These embeddings are required to build vec-tionaries but are NOT
#' included in the package to keep it lightweight.
#'
#' @param language Language code: "pt" (Portuguese), "en" (English), or "es" (Spanish).
#'   Determines which pre-trained embedding file to download.
#' @param model Embedding model to download:
#'   \itemize{
#'     \item \code{"fasttext"}: FastText Common Crawl embeddings (available for all languages)
#'     \item \code{"word2vec"}: Google News word2vec (English) or PT2Vec (Portuguese only)
#'     \item \code{"glove"}: GloVe 6B embeddings trained on Wikipedia + Gigaword (English only)
#'   }
#' @param dimensions Embedding vector dimensionality (default: 300). This is the
#'   number of numeric values per word in the embedding file, not the number of
#'   dictionary dimensions. FastText models are available in 300 dimensions.
#' @param destination Directory to save the downloaded embeddings file
#'   (default: \code{vdic_data/} in the current working directory).
#'   The directory is created automatically if it does not exist.
#' @param force If TRUE, re-download even if the file already exists at the
#'   destination. Useful for updating corrupted or incomplete downloads.
#'
#' @return Path to the downloaded embeddings file
#' @export
#'
#' @examples
#' \dontrun{
#' # Download Portuguese FastText embeddings
#' download_embeddings("pt", "fasttext")
#'
#' # Download to specific directory
#' download_embeddings("pt", "fasttext", destination = "~/my_embeddings")
#' }

#- Download Embeddings ----
download_embeddings <- function(
  language = c("pt", "en", "es"),
  model = c("fasttext", "word2vec", "glove"),
  dimensions = 300,
  destination = NULL,
  force = FALSE
) {

  language <- match.arg(language)
  model <- match.arg(model)
  
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
#' Binary format: header line "vocab_size n_dims\n", then for each word:
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
