output "bucket" {
  value = aws_s3_bucket.lake.id
}

output "state_table" {
  value = aws_dynamodb_table.state.name
}

output "secret_arn" {
  value = aws_secretsmanager_secret.sf.arn
}

output "extractor_function" {
  value = aws_lambda_function.extractor.function_name
}

output "query_function" {
  value = aws_lambda_function.query.function_name
}
