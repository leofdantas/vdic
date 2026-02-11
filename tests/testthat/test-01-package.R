# Tests for R/01_package.R
# Stopwords, language detection, and package-level utilities

#- Stopwords ----

test_that(".get_stopwords returns correct lists for supported languages", {
  en <- .get_stopwords("en")
  pt <- .get_stopwords("pt")
  es <- .get_stopwords("es")

  expect_type(en, "character")
  expect_type(pt, "character")
  expect_type(es, "character")

  expect_true(length(en) > 50)
  expect_true(length(pt) > 50)
  expect_true(length(es) > 50)

  # Check some known stopwords are present

  expect_true("the" %in% en)
  expect_true("and" %in% en)
  expect_true("de" %in% pt)
  expect_true("el" %in% es)
})

test_that(".get_stopwords returns NULL for unsupported languages", {
  expect_null(.get_stopwords("zh"))
  expect_null(.get_stopwords("ja"))
  expect_null(.get_stopwords("nonexistent"))
})

