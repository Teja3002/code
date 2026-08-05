variable "project" {
  description = "Name prefix for all resources"
  type        = string
  default     = "sf-aws-integration"
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "sf_base_url" {
  description = "SuccessFactors OData V2 base URL"
  type        = string
  default     = "https://apisalesdemo2.successfactors.eu/odata/v2"
}

variable "schedule_expression" {
  description = "EventBridge schedule for the extraction Lambda"
  type        = string
  default     = "rate(1 hour)"
}

variable "bedrock_model_id" {
  description = "Bedrock model used by the query Lambda"
  type        = string
  default     = "anthropic.claude-3-haiku-20240307-v1:0"
}
