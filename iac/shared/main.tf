locals {
  project_name = "chatgpt-data"
}

# App Modules
module "foundation" {
  source                 = "s3::https://s3-us-east-1.amazonaws.com/inf-env-iac-685199955655-us-east-1/service_modules/mlplatform_service_modules-7.11.10.zip//application_foundation_eks"
  project_name           = local.project_name
  stage_name             = var.stage_name
  cleanroom_project_name = "none"
  tags                   = var.tags
}

module "bucket" {
  source              = "s3::https://s3-us-east-1.amazonaws.com/inf-env-iac-685199955655-us-east-1/service_modules/mlplatform_service_modules-7.11.10.zip//service_modules/s3_bucket"
  app_foundation      = module.foundation
  service_name        = "chatgpt-data"
  service_description = "The bucket for data that backs the ChatGPT-Data service"
}

module "api" {
  source              = "s3::https://s3-us-east-1.amazonaws.com/inf-env-iac-685199955655-us-east-1/service_modules/mlplatform_service_modules-7.11.10.zip//service_modules/k8s_deployment"
  app_foundation_eks  = module.foundation
  service_name        = "chatgpt-data"
  service_description = "ChatGPT-data page"
  load_balancer = {
    allowed_cidr_blocks = [
      "10.0.0.0/8",
      "100.64.0.0/16"
    ]
    health_check_path = "/"
  }
  containers = [
    {
      name           = "chatgpt-data"
      ecr_image_name = "chatgpt-data"
      ecr_image_tag  = var.container_tag
      cpu            = "500m"
      memory         = "512Mi"
    }
  ]
  dependencies = {
    bucket = module.bucket
  }
}
