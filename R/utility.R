#' @description get the accuracy
#' @export

SMAPE = function(actual, predicted) {
  2 * mean(abs(actual - predicted) / (actual + predicted), na.rm = T)
}