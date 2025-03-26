# Deploying ChatGPT Dashboard to AWS Fargate

This guide provides instructions for deploying the ChatGPT Enterprise Engagement Metrics dashboard to AWS Fargate.

## Prerequisites

- AWS CLI installed and configured with appropriate permissions
- Docker installed on your local machine
- An AWS account with permissions to create ECR repositories, CloudFormation stacks, and Fargate services
- Python 3.8+ or Bash shell (for running the deployment scripts)
- AWS SSO configured with the `aica` profile

## AWS SSO Authentication

The deployment scripts use AWS SSO for authentication with the `aica` profile by default. Before deploying:

```bash
# Log in to AWS SSO with the aica profile
aws sso login --profile aica
```

The scripts will automatically check if your SSO session is valid and prompt you to log in if needed.

## Local Testing

Before deploying to Fargate, you can test the container locally:

```bash
# Build and run the container locally
docker-compose up --build

# The dashboard should be available at http://localhost:8000
```

## Deployment to AWS Fargate

We provide two deployment scripts for AWS Fargate:
1. **Python script** (`scripts/deploy_to_fargate.py`) - Recommended for most users
2. **Bash script** (`scripts/deploy_to_fargate.sh`) - Alternative option

### Using the Python Deployment Script

The Python script handles the entire deployment process, including:
- AWS SSO authentication validation
- Building and pushing the Docker image to ECR
- Creating or updating the CloudFormation stack
- Configuring network and security settings

```bash
# Install required dependencies
pip install boto3

# Run the deployment script with required parameters
python scripts/deploy_to_fargate.py \
    --vpc-id vpc-12345678 \
    --subnet-ids subnet-1234567,subnet-7654321 \
    --corporate-ip-ranges "10.0.0.0/8,172.16.0.0/12" \
    --region us-east-1 \
    --aws-profile aica

# For a dry run without making changes
python scripts/deploy_to_fargate.py \
    --vpc-id vpc-12345678 \
    --subnet-ids subnet-1234567,subnet-7654321 \
    --dry-run
```

### Using the Bash Deployment Script

Alternatively, you can use the Bash script which provides similar functionality:

```bash
# Run the deployment script with required parameters
./scripts/deploy_to_fargate.sh \
    --vpc-id vpc-12345678 \
    --subnet-ids subnet-1234567,subnet-7654321 \
    --corporate-ip-ranges "10.0.0.0/8,172.16.0.0/12" \
    --region us-east-1 \
    --aws-profile aica

# For a dry run without making changes
./scripts/deploy_to_fargate.sh \
    --vpc-id vpc-12345678 \
    --subnet-ids subnet-1234567,subnet-7654321 \
    --dry-run
```

### Script Options

Both scripts support the following options:

| Option | Description | Default |
|--------|-------------|---------|
| `--aws-profile` | AWS profile to use | `aica` |
| `--region` | AWS region for deployment | `us-east-1` |
| `--stack-name` | CloudFormation stack name | `chatgpt-dashboard` |
| `--repository-name` | ECR repository name | `chatgpt-dashboard` |
| `--vpc-id` | VPC ID for Fargate deployment | *(required)* |
| `--subnet-ids` | Comma-separated list of subnet IDs | *(required)* |
| `--corporate-ip-ranges` | Comma-separated list of CIDR blocks for access control | `0.0.0.0/0` |
| `--force-deploy` | Force deployment even if no changes | `false` |
| `--dry-run` | Perform a dry run without making changes | `false` |

## Maintenance

### Updating the Dashboard

To update the dashboard with new data:

1. Update the local `data/` directory with new files
2. Run the deployment script again with the same parameters
3. Add the `--force-deploy` option if there are no CloudFormation changes

### Monitoring

Monitor the service using AWS CloudWatch:

```bash
# View logs
aws logs get-log-events \
  --log-group-name /ecs/chatgpt-dashboard \
  --log-stream-name <log-stream-name> \
  --region <your-region>
```

## Troubleshooting

Common issues and solutions:

- **Docker build fails**: Ensure Docker is running on your machine
- **ECR login fails**: Check your AWS credentials and permissions
- **CloudFormation deployment fails**: Check the Events tab in the CloudFormation console
- **Container fails to start**: Check the ECS service logs in CloudWatch
- **Cannot access dashboard**: Verify security group rules allow your IP address
