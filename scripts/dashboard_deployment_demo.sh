#!/bin/bash
#
# Demo script for the ChatGPT Enterprise Engagement Metrics dashboard deployment
# This script walks through the process of local testing and AWS Fargate deployment

set -e

echo "========================================================="
echo "ChatGPT Enterprise Engagement Metrics Dashboard Deployment"
echo "========================================================="
echo

# Check dependencies
echo "Checking dependencies..."
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting." >&2; exit 1; }
command -v aws >/dev/null 2>&1 || { echo "AWS CLI is required but not installed. Aborting." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not installed. Aborting." >&2; exit 1; }
echo "All dependencies found!"
echo

# Project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Step 1: Test locally with Docker
echo "STEP 1: Testing locally with Docker Compose"
echo "----------------------------------------"
echo "This will build and run the container locally."
echo "You can access the dashboard at http://localhost:8000"
echo
read -p "Do you want to run local testing? (y/n): " run_local
echo

if [[ "$run_local" == "y" ]]; then
    echo "Building and running the container..."
    cd "$PROJECT_ROOT"
    docker-compose up --build -d
    
    echo
    echo "Container is now running. Press Enter to continue the demo."
    echo "To stop the container later, run: docker-compose down"
    read
else
    echo "Skipping local testing."
fi
echo

# Step 2: Choose deployment method
echo "STEP 2: Choose AWS Fargate Deployment Method"
echo "----------------------------------------"
echo "1) Python script (recommended)"
echo "2) Bash script"
echo "3) Skip deployment demo"
echo
read -p "Select deployment method (1-3): " deploy_method
echo

if [[ "$deploy_method" == "1" || "$deploy_method" == "2" ]]; then
    # Get common parameters
    read -p "Enter your AWS VPC ID (e.g., vpc-12345678): " vpc_id
    read -p "Enter subnet IDs (comma-separated, e.g., subnet-123,subnet-456): " subnet_ids
    read -p "Enter AWS region (default: us-east-1): " region
    region=${region:-us-east-1}
    read -p "Enter AWS profile (default: aica): " profile
    profile=${profile:-aica}
    read -p "Enter corporate IP ranges (default: 0.0.0.0/0): " ip_ranges
    ip_ranges=${ip_ranges:-0.0.0.0/0}
    
    echo
    echo "Running in dry-run mode for demo purposes..."
    
    if [[ "$deploy_method" == "1" ]]; then
        echo "Executing Python deployment script:"
        echo "python3 $SCRIPT_DIR/deploy_to_fargate.py \\"
        echo "    --vpc-id $vpc_id \\"
        echo "    --subnet-ids $subnet_ids \\"
        echo "    --region $region \\"
        echo "    --aws-profile $profile \\"
        echo "    --corporate-ip-ranges \"$ip_ranges\" \\"
        echo "    --dry-run"
        echo
        echo "In a real deployment, remove the --dry-run flag."
        
        # Install boto3 if needed
        python3 -c "import boto3" >/dev/null 2>&1 || {
            echo "boto3 is required for the Python script."
            read -p "Do you want to install boto3 now? (y/n): " install_boto3
            if [[ "$install_boto3" == "y" ]]; then
                python3 -m pip install boto3
            fi
        }
        
        # Run the command in dry-run mode
        python3 "$SCRIPT_DIR/deploy_to_fargate.py" \
            --vpc-id "$vpc_id" \
            --subnet-ids "$subnet_ids" \
            --region "$region" \
            --aws-profile "$profile" \
            --corporate-ip-ranges "$ip_ranges" \
            --dry-run
        
    elif [[ "$deploy_method" == "2" ]]; then
        echo "Executing Bash deployment script:"
        echo "$SCRIPT_DIR/deploy_to_fargate.sh \\"
        echo "    --vpc-id $vpc_id \\"
        echo "    --subnet-ids $subnet_ids \\"
        echo "    --region $region \\"
        echo "    --aws-profile $profile \\"
        echo "    --corporate-ip-ranges \"$ip_ranges\" \\"
        echo "    --dry-run"
        echo
        echo "In a real deployment, remove the --dry-run flag."
        
        # Make sure the script is executable
        chmod +x "$SCRIPT_DIR/deploy_to_fargate.sh"
        
        # Run the command in dry-run mode
        "$SCRIPT_DIR/deploy_to_fargate.sh" \
            --vpc-id "$vpc_id" \
            --subnet-ids "$subnet_ids" \
            --region "$region" \
            --aws-profile "$profile" \
            --corporate-ip-ranges "$ip_ranges" \
            --dry-run
    fi
else
    echo "Skipping deployment demo."
fi

echo
echo "========================================================="
echo "Demo completed!"
echo "For more information, see FARGATE_DEPLOYMENT.md"
echo "========================================================="
