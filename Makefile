AWS_REGION ?= $(shell aws configure get sso_region)
AWS_ACCOUNT ?= $(shell aws configure get sso_account_id)
S3_BUCKET = chatgpt-data-dev-chatgpt-data

ECR_PREFIX=$(AWS_ACCOUNT).dkr.ecr.$(AWS_REGION).amazonaws.com
CI_BUILD_VERSION="$(shell echo "$$(git rev-parse HEAD)")"

.PHONY: pipeline-pre-build
pipeline-pre-build:
	@echo "Running pipeline pre-build"
	@echo "CI_BUILD_VERSION=${CI_BUILD_VERSION}"

.PHONY: pipeline-build-version
pipeline-build-version:
	@echo "$(CI_BUILD_VERSION)"

.PHONY: pipeline-post-build
pipeline-post-build:
	@echo "Running pipeline post-build"

.PHONY: pipeline-build
pipeline-build: pipeline-docker-build create-ecr-repository
	docker tag chatgpt-data:latest $(ECR_PREFIX)/chatgpt-data:$(CI_BUILD_VERSION)
	docker push $(ECR_PREFIX)/chatgpt-data:$(CI_BUILD_VERSION)

.PHONY: pipeline-docker-build
pipeline-docker-build:
	docker build . -t chatgpt-data:latest -f pipeline.Dockerfile --build-arg REPOCACHE_HOST=$(ARTIFACT_REPOSITORY_HOST)

.PHONY: local-docker-build
local-docker-build:
	docker build . -t chatgpt-data:latest -f Dockerfile

# Create ECR repository if it doesn't exist
.PHONY: create-ecr-repository
create-ecr-repository: ecr-login
	( \
      set -e; \
      aws ecr describe-repositories --region ${AWS_REGION} --repository-name "chatgpt-data" || \
        aws ecr create-repository --region ${AWS_REGION} \
          --repository-name "chatgpt-data" --image-tag-mutability IMMUTABLE \
          --image-scanning-configuration scanOnPush=true \
          --tags \
            Key=Name,Value="chatgpt-data" \
            Key=business_unit,Value="AI Forge" \
            Key=product,Value=chatgpt-data \
            Key=created_by,Value="Staging deploy pipeline" \
            Key=service,Value=chatgpt-data \
            Key=support_level,Value=dev \
            Key=owner_email,Value="bwinterton@proofpoint.com" \
            Key=data_classification,Value=Internal\
    )

ecr-login:
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(ECR_PREFIX)