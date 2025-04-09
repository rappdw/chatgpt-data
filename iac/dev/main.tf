terraform {
  backend "s3" {
    bucket = "inf-env-iac-799767506880-us-east-1"
    key    = "terraform/applications/chatgpt-data/dev"
    region = "us-east-1"
  }
}

data "aws_eks_cluster" "default" {
  name = "inf-dev"
}

provider "kubernetes" {
  host                   = data.aws_eks_cluster.default.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.default.certificate_authority[0].data)
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    args        = ["eks", "get-token", "--cluster-name", "inf-dev"]
    command     = "aws"
  }
}

provider "aws" {
  region              = "us-east-1"
  allowed_account_ids = ["799767506880"]
}
provider "null" {}

locals {
  stage_name = "dev"
  tags = {
    business_unit = "AI Forge"
    created_by    = "resero-terraform"
    product       = "chatgpt-data"
    component     = "application"
    support_level = "dev"
  }
}

module "application" {
  source        = "../shared"
  container_tag = var.build_version
  stage_name    = "dev"
  tags          = local.tags
}

variable "build_version" {
  type = string
}
