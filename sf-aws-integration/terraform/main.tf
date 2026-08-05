terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
  }
}

provider "aws" {
  region = var.region
}

#----------------------------------------------------------------------
# Data lake bucket
#----------------------------------------------------------------------
resource "aws_s3_bucket" "lake" {
  bucket_prefix = "${var.project}-lake-"
}

resource "aws_s3_bucket_versioning" "lake" {
  bucket = aws_s3_bucket.lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "lake" {
  bucket = aws_s3_bucket.lake.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "lake" {
  bucket                  = aws_s3_bucket.lake.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

#----------------------------------------------------------------------
# Sync-state table (delta watermarks)
#----------------------------------------------------------------------
resource "aws_dynamodb_table" "state" {
  name         = "${var.project}-sync-state"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "entity_name"

  attribute {
    name = "entity_name"
    type = "S"
  }
}

#----------------------------------------------------------------------
# SuccessFactors credentials (value set out-of-band, never in Terraform)
#----------------------------------------------------------------------
resource "aws_secretsmanager_secret" "sf" {
  name        = "${var.project}/sf-credentials"
  description = "SuccessFactors API user; set value with put-secret-value"
}

#----------------------------------------------------------------------
# Extraction Lambda
#----------------------------------------------------------------------
data "archive_file" "extractor" {
  type        = "zip"
  source_dir  = "${path.module}/../src"
  output_path = "${path.module}/.build/extractor.zip"
}

resource "aws_iam_role" "extractor" {
  name = "${var.project}-extractor"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "extractor_logs" {
  role       = aws_iam_role.extractor.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "extractor" {
  name = "least-privilege"
  role = aws_iam_role.extractor.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject", "s3:GetObject"]
        Resource = "${aws_s3_bucket.lake.arn}/*"
      },
      {
        Effect   = "Allow"
        Action   = ["dynamodb:GetItem", "dynamodb:PutItem"]
        Resource = aws_dynamodb_table.state.arn
      },
      {
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = aws_secretsmanager_secret.sf.arn
      },
    ]
  })
}

resource "aws_lambda_function" "extractor" {
  function_name    = "${var.project}-extractor"
  role             = aws_iam_role.extractor.arn
  runtime          = "python3.12"
  handler          = "lambda_function.handler"
  filename         = data.archive_file.extractor.output_path
  source_code_hash = data.archive_file.extractor.output_base64sha256
  timeout          = 300
  memory_size      = 256

  environment {
    variables = {
      SF_BASE_URL = var.sf_base_url
      SECRET_ARN  = aws_secretsmanager_secret.sf.arn
      BUCKET      = aws_s3_bucket.lake.id
      STATE_TABLE = aws_dynamodb_table.state.name
    }
  }
}

#----------------------------------------------------------------------
# Schedule
#----------------------------------------------------------------------
resource "aws_cloudwatch_event_rule" "schedule" {
  name                = "${var.project}-schedule"
  schedule_expression = var.schedule_expression
}

resource "aws_cloudwatch_event_target" "schedule" {
  rule = aws_cloudwatch_event_rule.schedule.name
  arn  = aws_lambda_function.extractor.arn
}

resource "aws_lambda_permission" "schedule" {
  statement_id  = "AllowEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.extractor.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.schedule.arn
}

#----------------------------------------------------------------------
# Telemetry: alarms on failures and on missed runs
#----------------------------------------------------------------------
resource "aws_cloudwatch_metric_alarm" "extractor_errors" {
  alarm_name          = "${var.project}-extractor-errors"
  alarm_description   = "Extraction Lambda reported errors"
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  dimensions          = { FunctionName = aws_lambda_function.extractor.function_name }
  statistic           = "Sum"
  period              = 3600
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
}

resource "aws_cloudwatch_metric_alarm" "extractor_stalled" {
  alarm_name          = "${var.project}-extractor-stalled"
  alarm_description   = "No extraction run in the last 24 hours"
  namespace           = "AWS/Lambda"
  metric_name         = "Invocations"
  dimensions          = { FunctionName = aws_lambda_function.extractor.function_name }
  statistic           = "Sum"
  period              = 86400
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "LessThanThreshold"
  treat_missing_data  = "breaching"
}

#----------------------------------------------------------------------
# Bedrock query Lambda
#----------------------------------------------------------------------
data "archive_file" "query" {
  type        = "zip"
  source_dir  = "${path.module}/../bedrock_query"
  output_path = "${path.module}/.build/query.zip"
}

resource "aws_iam_role" "query" {
  name = "${var.project}-query"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "query_logs" {
  role       = aws_iam_role.query.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "query" {
  name = "least-privilege"
  role = aws_iam_role.query.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "${aws_s3_bucket.lake.arn}/curated/*"
      },
      {
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel", "bedrock:Converse"]
        Resource = "*"
      },
    ]
  })
}

resource "aws_lambda_function" "query" {
  function_name    = "${var.project}-query"
  role             = aws_iam_role.query.arn
  runtime          = "python3.12"
  handler          = "lambda_function.handler"
  filename         = data.archive_file.query.output_path
  source_code_hash = data.archive_file.query.output_base64sha256
  timeout          = 60
  memory_size      = 256

  environment {
    variables = {
      BUCKET   = aws_s3_bucket.lake.id
      MODEL_ID = var.bedrock_model_id
    }
  }
}
