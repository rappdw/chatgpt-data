#!/bin/bash
#
# Deploy the ChatGPT Enterprise Engagement Metrics dashboard to AWS Fargate
#
# This script:
# 1. Builds the Docker image
# 2. Pushes the image to Amazon ECR
# 3. Creates or updates the AWS Fargate service using CloudFormation

set -e

# Default values
REGION="us-east-1"
REPOSITORY_NAME="chatgpt-dashboard"
STACK_NAME="chatgpt-dashboard"
PROFILE="aica"
CORPORATE_IP_RANGES="0.0.0.0/0"
FORCE_DEPLOY=false
DRY_RUN=false

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Deploy ChatGPT Dashboard to AWS Fargate"
    echo
    echo "Options:"
    echo "  --aws-profile PROFILE       AWS profile to use (default: aica)"
    echo "  --region REGION             AWS region for deployment (default: us-east-1)"
    echo "  --stack-name NAME           CloudFormation stack name (default: chatgpt-dashboard)"
    echo "  --repository-name NAME      ECR repository name (default: chatgpt-dashboard)"
    echo "  --vpc-id VPC_ID             VPC ID for Fargate deployment (required)"
    echo "  --subnet-ids SUBNET_IDS     Comma-separated list of subnet IDs (required)"
    echo "  --corporate-ip-ranges CIDR  Comma-separated list of CIDR blocks (default: 0.0.0.0/0)"
    echo "  --force-deploy              Force deployment even if there are no changes"
    echo "  --dry-run                   Perform a dry run without making changes"
    echo "  --help                      Display this help message"
    exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --aws-profile)
            PROFILE="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --stack-name)
            STACK_NAME="$2"
            shift 2
            ;;
        --repository-name)
            REPOSITORY_NAME="$2"
            shift 2
            ;;
        --vpc-id)
            VPC_ID="$2"
            shift 2
            ;;
        --subnet-ids)
            SUBNET_IDS="$2"
            shift 2
            ;;
        --corporate-ip-ranges)
            CORPORATE_IP_RANGES="$2"
            shift 2
            ;;
        --force-deploy)
            FORCE_DEPLOY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required parameters
if [ -z "$VPC_ID" ]; then
    echo "Error: --vpc-id is required"
    usage
fi

if [ -z "$SUBNET_IDS" ]; then
    echo "Error: --subnet-ids is required"
    usage
fi

# Function to ensure AWS SSO login is valid
ensure_aws_sso_login() {
    local profile="$1"
    echo "Checking AWS SSO login status for profile '$profile'..."
    
    # Check if credentials are valid and not expired
    if aws sts get-caller-identity --profile "$profile" &>/dev/null; then
        echo "AWS SSO login is active and valid"
        return 0
    else
        echo "AWS SSO session is expired or not found"
    fi
    
    # Try to login with AWS SSO
    echo "Attempting AWS SSO login for profile '$profile'..."
    if aws sso login --profile "$profile"; then
        echo "AWS SSO login successful"
        return 0
    else
        echo "AWS SSO login failed"
        return 1
    fi
}

# Get AWS account ID
get_aws_account_id() {
    aws sts get-caller-identity --profile "$PROFILE" --query "Account" --output text
}

# Build and push Docker image
build_and_push_image() {
    local account_id="$1"
    local repository_uri="${account_id}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY_NAME}"
    local image_tag="${repository_uri}:latest"
    
    echo "Building Docker image..."
    cd "$PROJECT_ROOT"
    
    if [ "$DRY_RUN" = false ]; then
        docker build -t "$REPOSITORY_NAME" .
        
        echo "Logging in to ECR..."
        aws ecr get-login-password --profile "$PROFILE" --region "$REGION" | \
        docker login --username AWS --password-stdin "${account_id}.dkr.ecr.${REGION}.amazonaws.com"
        
        # Create repository if it doesn't exist
        aws ecr describe-repositories --profile "$PROFILE" --region "$REGION" \
            --repository-names "$REPOSITORY_NAME" >/dev/null 2>&1 || \
        aws ecr create-repository --profile "$PROFILE" --region "$REGION" \
            --repository-name "$REPOSITORY_NAME"
        
        echo "Tagging and pushing image..."
        docker tag "$REPOSITORY_NAME" "$image_tag"
        docker push "$image_tag"
    else
        echo "[DRY RUN] Would build, tag, and push image to: $image_tag"
    fi
    
    echo "Image URI: $image_tag"
    echo
    
    echo "$image_tag"
}

# Deploy CloudFormation stack
deploy_cloudformation() {
    local image_uri="$1"
    local template_file="${PROJECT_ROOT}/cloudformation-fargate.yml"
    
    echo "Deploying CloudFormation stack: $STACK_NAME"
    
    # Check if template exists
    if [ ! -f "$template_file" ]; then
        echo "Error: CloudFormation template not found: $template_file"
        exit 1
    fi
    
    # Parameters for CloudFormation
    local params=(
        "ParameterKey=VpcId,ParameterValue=${VPC_ID}"
        "ParameterKey=SubnetIds,ParameterValue=${SUBNET_IDS}"
        "ParameterKey=ECRImageURI,ParameterValue=${image_uri}"
        "ParameterKey=CorporateIpRanges,ParameterValue=${CORPORATE_IP_RANGES}"
    )
    
    # Add timestamp parameter for force deploy
    if [ "$FORCE_DEPLOY" = true ]; then
        params+=("ParameterKey=DeploymentTimestamp,ParameterValue=$(date +%s)")
    fi
    
    # Convert parameters to proper format
    local parameters=""
    for param in "${params[@]}"; do
        parameters="${parameters} --parameter-overrides ${param}"
    done
    
    if [ "$DRY_RUN" = false ]; then
        # Check if stack exists
        if aws cloudformation describe-stacks --profile "$PROFILE" --region "$REGION" \
            --stack-name "$STACK_NAME" >/dev/null 2>&1; then
            # Update existing stack
            echo "Updating existing stack: $STACK_NAME"
            aws cloudformation deploy \
                --profile "$PROFILE" \
                --region "$REGION" \
                --stack-name "$STACK_NAME" \
                --template-file "$template_file" \
                --capabilities CAPABILITY_NAMED_IAM \
                ${parameters}
        else
            # Create new stack
            echo "Creating new stack: $STACK_NAME"
            aws cloudformation deploy \
                --profile "$PROFILE" \
                --region "$REGION" \
                --stack-name "$STACK_NAME" \
                --template-file "$template_file" \
                --capabilities CAPABILITY_NAMED_IAM \
                ${parameters}
        fi
        
        # Get outputs
        echo "Deployment completed! Stack outputs:"
        aws cloudformation describe-stacks \
            --profile "$PROFILE" \
            --region "$REGION" \
            --stack-name "$STACK_NAME" \
            --query "Stacks[0].Outputs" \
            --output table
    else
        echo "[DRY RUN] Would deploy CloudFormation stack with parameters:"
        for param in "${params[@]}"; do
            echo "  $param"
        done
    fi
}

# Main execution
echo "=== ChatGPT Dashboard Fargate Deployment ==="
echo "AWS Profile: $PROFILE"
echo "Region: $REGION"
echo "VPC ID: $VPC_ID"
echo "Subnet IDs: $SUBNET_IDS"
echo "Corporate IP Ranges: $CORPORATE_IP_RANGES"
echo

if [ "$DRY_RUN" = true ]; then
    echo "*** DRY RUN MODE - NO CHANGES WILL BE MADE ***"
fi

# Ensure AWS SSO login is valid
if ! ensure_aws_sso_login "$PROFILE"; then
    echo "Failed to authenticate with AWS SSO using profile '$PROFILE'"
    echo "Please run 'aws sso login --profile aica' manually and try again"
    exit 1
fi

ACCOUNT_ID=$(get_aws_account_id)
echo "AWS Account ID: $ACCOUNT_ID"

IMAGE_URI=$(build_and_push_image "$ACCOUNT_ID")
deploy_cloudformation "$IMAGE_URI"

echo "Deployment script completed successfully!"
