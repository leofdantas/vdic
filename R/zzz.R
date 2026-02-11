#' @keywords internal
.onAttach <- function(libname, pkgname) {

  packageStartupMessage(
    "vdic: Build and use vec-tionaries for text analysis\n",
    "------------------------------------------------------\n",
    "Quick start:\n",
    "  1. download_embeddings('pt', 'fasttext')  # Download embeddings\n",
    "  2. vectionary_builder(dictionary, embeddings)  # Build vec-tionary\n",
    "  3. my_vect$mean(text)  # Analyze text\n",
    "\n",
    "See ?vdic for documentation."
  )
}

#' @keywords internal
.onLoad <- function(libname, pkgname) {
  # Disable ANSI escape codes in cli output for better terminal compatibility
  options(cli.ansi = FALSE)
}
