variable "container_tag" {
  type = string
}

variable "stage_name" {
  type        = string
  description = "The name of the stage that is being deployed"
}

variable "tags" {
  description = "The tags to apply to resources"
}
