# Tests for R/02_embeddings.R
# Download URLs, path resolution, and binary conversion

#- download_embeddings input validation ----

test_that("download_embeddings rejects unsupported language for word2vec", {
  expect_error(
    download_embeddings("es", "word2vec", destination = tempdir()),
    "Word2Vec only available for English or Portuguese"
  )
})

test_that("download_embeddings rejects unsupported language for glove", {
  expect_error(
    download_embeddings("pt", "glove", destination = tempdir()),
    "glove only available for English"
  )
})

#- .convert_bin_to_vec ----

test_that(".convert_bin_to_vec converts binary format to text", {
  # Create a small mock .bin file
  bin_path <- tempfile(fileext = ".bin")
  vec_path <- tempfile(fileext = ".vec")
  on.exit({
    unlink(bin_path)
    unlink(vec_path)
  })

  # Write mock binary file
  # Header: "3 4\n" (3 words, 4 dimensions)
  con <- file(bin_path, "wb")

  # Header (text line)
  writeBin(charToRaw("3 4\n"), con)

  # Word 1: "cat" + space + 4 floats
  writeBin(charToRaw("cat"), con)
  writeBin(as.raw(0x20), con)  # space terminator
  writeBin(c(0.1, 0.2, 0.3, 0.4), con, size = 4, endian = "little")

  # Word 2: "dog" + space + 4 floats
  writeBin(as.raw(0x0A), con)  # newline between entries
  writeBin(charToRaw("dog"), con)
  writeBin(as.raw(0x20), con)
  writeBin(c(0.5, 0.6, 0.7, 0.8), con, size = 4, endian = "little")

  # Word 3: "bird" + space + 4 floats
  writeBin(as.raw(0x0A), con)
  writeBin(charToRaw("bird"), con)
  writeBin(as.raw(0x20), con)
  writeBin(c(-0.1, -0.2, 0.9, 1.0), con, size = 4, endian = "little")

  close(con)

  # Convert
  result <- .convert_bin_to_vec(bin_path, vec_path, verbose = FALSE)

  # Check output file exists

  expect_true(file.exists(vec_path))

  # Read and verify text output
  lines <- readLines(vec_path)

  # Header line
  expect_equal(lines[1], "3 4")

  # Parse word lines
  expect_equal(length(lines), 4)  # header + 3 words

  # Check word 1
  parts1 <- strsplit(lines[2], " ")[[1]]
  expect_equal(parts1[1], "cat")
  expect_equal(length(parts1), 5)  # word + 4 dims
  vals1 <- as.numeric(parts1[2:5])
  expect_equal(vals1, c(0.1, 0.2, 0.3, 0.4), tolerance = 1e-5)

  # Check word 2
  parts2 <- strsplit(lines[3], " ")[[1]]
  expect_equal(parts2[1], "dog")
  vals2 <- as.numeric(parts2[2:5])
  expect_equal(vals2, c(0.5, 0.6, 0.7, 0.8), tolerance = 1e-5)

  # Check word 3
  parts3 <- strsplit(lines[4], " ")[[1]]
  expect_equal(parts3[1], "bird")
  vals3 <- as.numeric(parts3[2:5])
  expect_equal(vals3, c(-0.1, -0.2, 0.9, 1.0), tolerance = 1e-5)
})

test_that(".convert_bin_to_vec handles single word", {
  bin_path <- tempfile(fileext = ".bin")
  vec_path <- tempfile(fileext = ".vec")
  on.exit({
    unlink(bin_path)
    unlink(vec_path)
  })

  con <- file(bin_path, "wb")
  writeBin(charToRaw("1 3\n"), con)
  writeBin(charToRaw("hello"), con)
  writeBin(as.raw(0x20), con)
  writeBin(c(1.0, 2.0, 3.0), con, size = 4, endian = "little")
  close(con)

  .convert_bin_to_vec(bin_path, vec_path, verbose = FALSE)

  lines <- readLines(vec_path)
  expect_equal(lines[1], "1 3")
  parts <- strsplit(lines[2], " ")[[1]]
  expect_equal(parts[1], "hello")
  expect_equal(as.numeric(parts[2:4]), c(1.0, 2.0, 3.0), tolerance = 1e-5)
})
