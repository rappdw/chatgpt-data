#!/usr/bin/env python3
"""
Deploy the ChatGPT Enterprise Engagement Metrics dashboard to AWS Fargate.

This script:
1. Builds the Docker image
2. Pushes the image to Amazon ECR
3. Creates or updates the AWS Fargate service using CloudFormation
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import boto3
from botocore.exceptions import ClientError, ProfileNotExistError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Deploy ChatGPT Dashboard to AWS Fargate"
    )
    
    parser.add_argument(
        "--aws-profile",
        help="AWS profile to use for deployment (default: aica)",
        default="aica",
    )
    
    parser.add_argument(
        "--region",
        help="AWS region for deployment (default: us-east-1)",
        default="us-east-1",
    )
    
    parser.add_argument(
        "--stack-name",
        help="CloudFormation stack name (default: chatgpt-dashboard)",
        default="chatgpt-dashboard",
    )
    
    parser.add_argument(
        "--repository-name",
        help="ECR repository name (default: chatgpt-dashboard)",
        default="chatgpt-dashboard",
    )
    
    parser.add_argument(
        "--vpc-id",
        help="VPC ID for Fargate deployment",
        required=True,
    )
    
    parser.add_argument(
        "--subnet-ids",
        help="Comma-separated list of subnet IDs for Fargate deployment",
        required=True,
    )
    
    parser.add_argument(
        "--corporate-ip-ranges",
        help="Comma-separated list of CIDR blocks for access control (default: 0.0.0.0/0)",
        default="0.0.0.0/0",
    )
    
    parser.add_argument(
        "--force-deploy",
        help="Force deployment even if there are no changes",
        action="store_true",
    )
    
    parser.add_argument(
        "--dry-run",
        help="Perform a dry run without making changes",
        action="store_true",
    )
    
    return parser.parse_args()


def run_command(command: List[str], check: bool = True) -> Tuple[int, str, str]:
    """Run a shell command and return the result.
    
    Args:
        command: Command to run as a list of strings
        check: Whether to check the return code
    
    Returns:
        Tuple of (return code, stdout, stderr)
    
    Raises:
        subprocess.CalledProcessError: If check is True and the command fails
    """
    logger.debug(f"Running command: {' '.join(command)}")
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    stdout, stderr = process.communicate()
    returncode = process.returncode
    
    if check and returncode != 0:
        logger.error(f"Command failed with code {returncode}:")
        logger.error(f"Command: {' '.join(command)}")
        logger.error(f"STDOUT: {stdout}")
        logger.error(f"STDERR: {stderr}")
        raise subprocess.CalledProcessError(returncode, command, stdout, stderr)
    
    return returncode, stdout, stderr


def get_aws_account_id(session: boto3.Session) -> str:
    """Get the AWS account ID.
    
    Args:
        session: AWS session
    
    Returns:
        AWS account ID
    """
    sts = session.client("sts")
    return sts.get_caller_identity()["Account"]


def ensure_ecr_repository_exists(
    ecr_client: boto3.client, repository_name: str
) -> None:
    """Ensure the ECR repository exists, creating it if necessary.
    
    Args:
        ecr_client: ECR client
        repository_name: Name of the ECR repository
    """
    try:
        ecr_client.describe_repositories(repositoryNames=[repository_name])
        logger.info(f"ECR repository {repository_name} already exists")
    except ClientError as e:
        if e.response["Error"]["Code"] == "RepositoryNotFoundException":
            logger.info(f"Creating ECR repository {repository_name}")
            ecr_client.create_repository(repositoryName=repository_name)
            logger.info(f"ECR repository {repository_name} created")
        else:
            raise


def ensure_aws_sso_login(profile: str) -> bool:
    """Ensure the user is logged in to AWS SSO.
    
    Args:
        profile: AWS profile to use
    
    Returns:
        True if SSO login is successful, False otherwise
    """
    logger.info(f"Checking AWS SSO login status for profile '{profile}'...")
    
    # Check if credentials are valid and not expired
    try:
        session = boto3.Session(profile_name=profile)
        sts = session.client('sts')
        sts.get_caller_identity()
        logger.info("AWS SSO login is active and valid")
        return True
    except Exception as e:
        if "expired" in str(e).lower() or "not found" in str(e).lower():
            logger.warning(f"AWS SSO session is expired or not found: {e}")
        else:
            logger.warning(f"AWS credentials check failed: {e}")
    
    # Try to login with AWS SSO
    try:
        logger.info(f"Attempting AWS SSO login for profile '{profile}'...")
        result, stdout, stderr = run_command(
            ["aws", "sso", "login", "--profile", profile],
            check=False
        )
        
        if result == 0:
            logger.info("AWS SSO login successful")
            return True
        else:
            logger.error(f"AWS SSO login failed: {stderr}")
            return False
    except Exception as e:
        logger.error(f"Error during AWS SSO login: {e}")
        return False


def build_and_push_docker_image(
    args: argparse.Namespace, session: boto3.Session, dry_run: bool = False
) -> str:
    """Build and push the Docker image to ECR.
    
    Args:
        args: Command line arguments
        session: AWS session
        dry_run: Whether to perform a dry run
    
    Returns:
        Full URI of the pushed image
    """
    account_id = get_aws_account_id(session)
    ecr_client = session.client("ecr")
    
    # Ensure the repository exists
    if not dry_run:
        ensure_ecr_repository_exists(ecr_client, args.repository_name)
    
    # Get the ECR repository URI
    repository_uri = f"{account_id}.dkr.ecr.{args.region}.amazonaws.com/{args.repository_name}"
    
    # Build the Docker image
    logger.info("Building Docker image...")
    
    # Find the Dockerfile location (project root)
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    
    if not dry_run:
        run_command(["docker", "build", "-t", args.repository_name, "."])
    
    # Tag the image
    tag = f"{repository_uri}:latest"
    if not dry_run:
        run_command(["docker", "tag", args.repository_name, tag])
    
    # Get ECR login command
    logger.info("Logging in to ECR...")
    if not dry_run:
        ecr_login_cmd = [
            "aws", "ecr", "get-login-password",
            "--region", args.region,
            "--profile", args.aws_profile
        ]
        _, password, _ = run_command(ecr_login_cmd)
        
        run_command([
            "docker", "login",
            "--username", "AWS",
            "--password-stdin",
            f"{account_id}.dkr.ecr.{args.region}.amazonaws.com"
        ], check=True)
    
    # Push the image
    logger.info(f"Pushing image to {tag}...")
    if not dry_run:
        run_command(["docker", "push", tag])
    
    logger.info("Docker image pushed successfully")
    return tag


def deploy_cloudformation_stack(
    args: argparse.Namespace, session: boto3.Session, image_uri: str, dry_run: bool = False
) -> None:
    """Deploy the CloudFormation stack for Fargate service.
    
    Args:
        args: Command line arguments
        session: AWS session
        image_uri: URI of the Docker image
        dry_run: Whether to perform a dry run
    """
    cf_client = session.client("cloudformation")
    
    # Find the CloudFormation template
    project_root = Path(__file__).resolve().parent.parent
    template_path = project_root / "cloudformation-fargate.yml"
    
    with open(template_path, "r") as f:
        template_body = f.read()
    
    # Prepare parameters
    parameters = [
        {"ParameterKey": "VpcId", "ParameterValue": args.vpc_id},
        {"ParameterKey": "SubnetIds", "ParameterValue": args.subnet_ids},
        {"ParameterKey": "ECRImageURI", "ParameterValue": image_uri},
        {"ParameterKey": "CorporateIpRanges", "ParameterValue": args.corporate_ip_ranges},
    ]
    
    logger.info(f"Deploying CloudFormation stack {args.stack_name}...")
    
    if dry_run:
        logger.info("Dry run - would deploy with parameters:")
        for param in parameters:
            logger.info(f"  {param['ParameterKey']}: {param['ParameterValue']}")
        return
    
    try:
        # Check if stack exists
        cf_client.describe_stacks(StackName=args.stack_name)
        stack_exists = True
    except ClientError as e:
        if "does not exist" in str(e):
            stack_exists = False
        else:
            raise
    
    if stack_exists:
        # Update existing stack
        try:
            cf_client.update_stack(
                StackName=args.stack_name,
                TemplateBody=template_body,
                Parameters=parameters,
                Capabilities=["CAPABILITY_NAMED_IAM"],
            )
            logger.info("Stack update initiated")
        except ClientError as e:
            if "No updates are to be performed" in str(e) and not args.force_deploy:
                logger.info("No changes detected in CloudFormation stack")
            elif "No updates are to be performed" in str(e) and args.force_deploy:
                # Force update by changing timestamp parameter
                parameters.append({
                    "ParameterKey": "DeploymentTimestamp",
                    "ParameterValue": str(int(time.time()))
                })
                cf_client.update_stack(
                    StackName=args.stack_name,
                    TemplateBody=template_body,
                    Parameters=parameters,
                    Capabilities=["CAPABILITY_NAMED_IAM"],
                )
                logger.info("Forced stack update initiated")
            else:
                raise
    else:
        # Create new stack
        cf_client.create_stack(
            StackName=args.stack_name,
            TemplateBody=template_body,
            Parameters=parameters,
            Capabilities=["CAPABILITY_NAMED_IAM"],
        )
        logger.info("Stack creation initiated")
    
    if not args.dry_run:
        # Wait for stack operation to complete
        logger.info("Waiting for stack operation to complete...")
        waiter = cf_client.get_waiter(
            "stack_create_complete" if not stack_exists else "stack_update_complete"
        )
        waiter.wait(StackName=args.stack_name)
        
        # Get the outputs
        response = cf_client.describe_stacks(StackName=args.stack_name)
        outputs = response["Stacks"][0].get("Outputs", [])
        
        for output in outputs:
            logger.info(f"{output['OutputKey']}: {output['OutputValue']}")


def main() -> int:
    """Main entry point for the script.
    
    Returns:
        Exit code
    """
    args = parse_args()
    
    try:
        # Ensure AWS SSO login is valid
        if not ensure_aws_sso_login(args.aws_profile):
            logger.error(f"Failed to authenticate with AWS SSO using profile '{args.aws_profile}'")
            logger.error("Please run 'aws sso login --profile aica' manually and try again")
            return 1
        
        # Create AWS session
        session = boto3.Session(profile_name=args.aws_profile, region_name=args.region)
        
        # Build and push Docker image
        image_uri = build_and_push_docker_image(args, session, args.dry_run)
        
        # Deploy CloudFormation stack
        deploy_cloudformation_stack(args, session, image_uri, args.dry_run)
        
        logger.info("Deployment completed successfully")
        return 0
    
    except ProfileNotExistError:
        logger.error(f"AWS profile '{args.aws_profile}' does not exist")
        return 1
    except ClientError as e:
        logger.error(f"AWS error: {e}")
        return 1
    except subprocess.CalledProcessError:
        logger.error("Command execution failed")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
