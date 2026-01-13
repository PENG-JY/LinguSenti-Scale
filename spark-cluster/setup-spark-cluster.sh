#!/bin/bash

################################################################################
# Apache Spark Cluster Automated Setup Script
#
# This script automates the complete setup of a 4-node Spark cluster on AWS EC2:
# - 1 Master node
# - 3 Worker nodes
#
# Usage: ./setup-spark-cluster.sh <LAPTOP_IP>
#   LAPTOP_IP: Your laptop's public IP address (get from https://ipchicken.com/)
#
# Example: ./setup-spark-cluster.sh 123.45.67.89
#
# Logs are saved to: cluster-setup.log
################################################################################

set -e  # Exit on any error

# Setup logging
LOG_FILE="cluster-setup-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging to: $LOG_FILE"
echo "Started at: $(date)"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_step() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

# Trap errors and log them
trap 'log_error "Script failed at line $LINENO. Check $LOG_FILE for details."' ERR

# Check if laptop IP is provided
if [ $# -ne 1 ]; then
    log_error "Usage: $0 <LAPTOP_IP>"
    log_error "Example: $0 123.45.67.89"
    log_error "Get your laptop IP from https://ipchicken.com/"
    exit 1
fi

LAPTOP_IP=$1
log_info "Laptop IP: $LAPTOP_IP"

# Validate IP format
if ! [[ $LAPTOP_IP =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    log_error "Invalid IP address format: $LAPTOP_IP"
    exit 1
fi

################################################################################
# Part 1: Prerequisites
################################################################################

log_step "Part 1: Checking Prerequisites"

# Get EC2 instance public IP
log_info "Getting EC2 instance public IP..."
export MY_EC2_IP=$(curl -s https://checkip.amazonaws.com)
log_success "EC2 Instance IP: $MY_EC2_IP"

export MY_LAPTOP_IP=$LAPTOP_IP
log_success "Laptop IP: $MY_LAPTOP_IP"

# Set AWS region
export AWS_REGION=us-east-1
log_info "AWS Region: $AWS_REGION"

# Check AWS CLI
log_info "Checking AWS CLI..."
if ! command -v aws &> /dev/null; then
    log_error "AWS CLI not found. Please install it first."
    exit 1
fi
log_success "AWS CLI found: $(aws --version)"

# Check if cluster-files directory exists
if [ ! -d "cluster-files" ]; then
    log_error "cluster-files directory not found. Please run this script from the lab-spark-cluster directory."
    exit 1
fi
log_success "cluster-files directory found"

# Auto-detect IAM Instance Profile
log_info "Auto-detecting IAM Instance Profile..."

# Get IMDSv2 token for instance metadata access
TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" -s 2>/dev/null)

if [ -n "$TOKEN" ]; then
    # Use IMDSv2 with token
    INSTANCE_ID=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null)
else
    # Fallback to IMDSv1 (older method)
    INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null)
fi

if [ -n "$INSTANCE_ID" ]; then
    log_info "Current instance ID: $INSTANCE_ID"

    # Get the IAM instance profile ARN from instance metadata
    PROFILE_ARN=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].IamInstanceProfile.Arn' \
        --output text \
        --region $AWS_REGION 2>/dev/null || echo "None")

    if [ "$PROFILE_ARN" != "None" ] && [ -n "$PROFILE_ARN" ]; then
        # Extract profile name from ARN (format: arn:aws:iam::account:instance-profile/profile-name)
        export IAM_INSTANCE_PROFILE=$(echo $PROFILE_ARN | awk -F'/' '{print $NF}')
        log_success "Detected IAM Instance Profile: $IAM_INSTANCE_PROFILE"
    else
        log_warn "No IAM Instance Profile detected on current instance"
        log_info "Attempting to find available instance profiles..."

        # Try to find any available instance profile
        AVAILABLE_PROFILE=$(aws iam list-instance-profiles \
            --query 'InstanceProfiles[0].InstanceProfileName' \
            --output text \
            --region $AWS_REGION 2>/dev/null || echo "None")

        if [ "$AVAILABLE_PROFILE" != "None" ] && [ -n "$AVAILABLE_PROFILE" ]; then
            export IAM_INSTANCE_PROFILE=$AVAILABLE_PROFILE
            log_warn "Using first available profile: $IAM_INSTANCE_PROFILE"
        else
            log_error "No IAM Instance Profile found. Cluster instances will launch without IAM role."
            log_error "S3 access may not work. Consider creating an instance profile with S3 access."
            export IAM_INSTANCE_PROFILE=""
        fi
    fi
else
    log_warn "Could not detect instance ID. Using default profile name."
    export IAM_INSTANCE_PROFILE="LabInstanceProfile"
fi

################################################################################
# Part 2: Create Security Group
################################################################################

log_step "Part 2: Creating Security Group"

SG_NAME="spark-cluster-sg-$(date +%s)"

# Check if security group with this pattern exists and delete it
log_info "Checking for existing security groups..."
EXISTING_SG=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=spark-cluster-sg-*" \
  --query 'SecurityGroups[0].GroupId' \
  --output text \
  --region $AWS_REGION 2>/dev/null || echo "None")

if [ "$EXISTING_SG" != "None" ] && [ "$EXISTING_SG" != "" ]; then
    log_warn "Found existing security group: $EXISTING_SG"
    log_info "Attempting to delete existing security group..."
    if aws ec2 delete-security-group --group-id $EXISTING_SG --region $AWS_REGION 2>/dev/null; then
        log_success "Deleted existing security group"
        sleep 2
    else
        log_warn "Could not delete existing security group (may be in use)"
    fi
fi

log_info "Creating security group..."
export SPARK_SG_ID=$(aws ec2 create-security-group \
  --group-name $SG_NAME \
  --description "Security group for Spark cluster" \
  --region $AWS_REGION \
  --query 'GroupId' \
  --output text)
log_success "Security Group ID: $SPARK_SG_ID"

# Wait a bit for security group to be ready
sleep 2

log_info "Configuring security group rules..."

# Allow all traffic within the security group
log_info "Allowing intra-cluster communication..."
aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol -1 \
  --source-group $SPARK_SG_ID \
  --region $AWS_REGION > /dev/null
log_success "Intra-cluster communication enabled"

# Allow SSH from EC2 instance
log_info "Allowing SSH from EC2 instance ($MY_EC2_IP)..."
aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr ${MY_EC2_IP}/32 \
  --region $AWS_REGION > /dev/null
log_success "SSH access from EC2 enabled"

# Allow SSH from laptop
log_info "Allowing SSH from laptop ($MY_LAPTOP_IP)..."
aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol tcp \
  --port 22 \
  --cidr ${MY_LAPTOP_IP}/32 \
  --region $AWS_REGION > /dev/null
log_success "SSH access from laptop enabled"

# Allow Spark Web UI from EC2
log_info "Allowing Spark Web UI access from EC2..."
aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol tcp \
  --port 8080-8081 \
  --cidr ${MY_EC2_IP}/32 \
  --region $AWS_REGION > /dev/null
log_success "Spark Web UI access from EC2 enabled"

# Allow Spark Web UI from laptop
log_info "Allowing Spark Web UI access from laptop..."
aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol tcp \
  --port 8080-8081 \
  --cidr ${MY_LAPTOP_IP}/32 \
  --region $AWS_REGION > /dev/null
log_success "Spark Web UI access from laptop enabled"

# Allow Spark Application UI from EC2
log_info "Allowing Spark Application UI from EC2..."
aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol tcp \
  --port 4040 \
  --cidr ${MY_EC2_IP}/32 \
  --region $AWS_REGION > /dev/null
log_success "Spark Application UI from EC2 enabled"

# Allow Spark Application UI from laptop
log_info "Allowing Spark Application UI from laptop..."
aws ec2 authorize-security-group-ingress \
  --group-id $SPARK_SG_ID \
  --protocol tcp \
  --port 4040 \
  --cidr ${MY_LAPTOP_IP}/32 \
  --region $AWS_REGION > /dev/null
log_success "Spark Application UI from laptop enabled"

################################################################################
# Part 3: Create SSH Key Pair
################################################################################

log_step "Part 3: Creating SSH Key Pair"

KEY_NAME="spark-cluster-key-$(date +%s)"
KEY_FILE="${KEY_NAME}.pem"

# Check if key pairs with this pattern exist and delete them
log_info "Checking for existing key pairs..."
EXISTING_KEYS=$(aws ec2 describe-key-pairs \
  --filters "Name=key-name,Values=spark-cluster-key-*" \
  --query 'KeyPairs[*].KeyName' \
  --output text \
  --region $AWS_REGION 2>/dev/null || echo "")

if [ -n "$EXISTING_KEYS" ]; then
    for KEY in $EXISTING_KEYS; do
        log_warn "Found existing key pair: $KEY"
        log_info "Attempting to delete existing key pair..."
        if aws ec2 delete-key-pair --key-name $KEY --region $AWS_REGION 2>/dev/null; then
            log_success "Deleted existing key pair: $KEY"
        else
            log_warn "Could not delete key pair: $KEY"
        fi
    done
    sleep 1
fi

log_info "Creating key pair: $KEY_NAME..."
aws ec2 create-key-pair \
  --key-name $KEY_NAME \
  --query 'KeyMaterial' \
  --output text \
  --region $AWS_REGION > $KEY_FILE

chmod 400 $KEY_FILE
log_success "Key pair created and saved to $KEY_FILE"

################################################################################
# Part 3.5: Download and Extract Spark Locally (Orchestrator Machine)
################################################################################

log_step "Part 3.5: Preparing Spark for Distribution"

SPARK_ARCHIVE="spark-3.4.4-bin-hadoop3.tgz"
SPARK_LOCAL_ARCHIVE="$PWD/$SPARK_ARCHIVE"
SPARK_LOCAL_DIR="$PWD/spark-3.4.4-bin-hadoop3"
export SPARK_DIST_DIR="$PWD/spark"  # Final directory name for distribution

# Check if Spark is already extracted locally
if [ -d "$SPARK_DIST_DIR" ]; then
    log_success "Spark already extracted locally: $SPARK_DIST_DIR"
else
    # Try downloading pre-extracted Spark from S3 first
    log_info "Attempting to download Spark from course S3 bucket..."
    S3_FAILED=0

    if timeout 3600 aws s3 sync \
        s3://dsan6000-datasets/spark-3.4.4-bin-hadoop3/ \
        "$SPARK_DIST_DIR" \
        --request-payer requester 2>/dev/null; then

        if [ -d "$SPARK_DIST_DIR" ] && [ -f "$SPARK_DIST_DIR/bin/spark-submit" ]; then
            # Fix permissions on Spark scripts after S3 sync (ensure executables have +x)
            chmod +x "$SPARK_DIST_DIR"/bin/* "$SPARK_DIST_DIR"/sbin/* 2>/dev/null || true
            log_success "Spark downloaded from S3: $SPARK_DIST_DIR"
        else
            log_warn "S3 sync incomplete, falling back to downloading archive from internet..."
            rm -rf "$SPARK_DIST_DIR"
            S3_FAILED=1
        fi
    else
        log_warn "S3 download failed, falling back to downloading archive from internet..."
        rm -rf "$SPARK_DIST_DIR"
        S3_FAILED=1
    fi

    # If S3 failed, download and extract from internet
    if [ "$S3_FAILED" = "1" ]; then
        log_info "Downloading Spark archive from Apache archives..."
        log_info "This may take 5-10 minutes..."

        # Check if archive exists, if not download it
        if [ ! -f "$SPARK_LOCAL_ARCHIVE" ] || [ ! -s "$SPARK_LOCAL_ARCHIVE" ]; then
            DOWNLOAD_ATTEMPT=1
            MAX_ATTEMPTS=3
            DOWNLOAD_TIMEOUT=3600  # 1 hour timeout

            while [ $DOWNLOAD_ATTEMPT -le $MAX_ATTEMPTS ]; do
                log_info "Download attempt $DOWNLOAD_ATTEMPT of $MAX_ATTEMPTS"
                rm -f "$SPARK_LOCAL_ARCHIVE"

                # Download with timeout and retries
                if timeout $DOWNLOAD_TIMEOUT wget \
                    --timeout=120 \
                    --tries=3 \
                    --progress=bar:force:noscroll \
                    -O "$SPARK_LOCAL_ARCHIVE" \
                    https://archive.apache.org/dist/spark/spark-3.4.4/spark-3.4.4-bin-hadoop3.tgz 2>&1 | tail -3; then

                    if [ -f "$SPARK_LOCAL_ARCHIVE" ] && [ -s "$SPARK_LOCAL_ARCHIVE" ]; then
                        FILESIZE=$(stat -c%s "$SPARK_LOCAL_ARCHIVE")
                        log_success "Spark downloaded: $(numfmt --to=iec-i --suffix=B $FILESIZE 2>/dev/null || echo $FILESIZE bytes)"
                        break
                    fi
                else
                    WGET_EXIT=$?
                    if [ $WGET_EXIT -eq 124 ]; then
                        log_warn "Download timed out"
                    fi
                fi

                if [ $DOWNLOAD_ATTEMPT -lt $MAX_ATTEMPTS ]; then
                    log_info "Retrying in 10 seconds..."
                    sleep 10
                fi
                DOWNLOAD_ATTEMPT=$((DOWNLOAD_ATTEMPT + 1))
            done

            if [ ! -f "$SPARK_LOCAL_ARCHIVE" ] || [ ! -s "$SPARK_LOCAL_ARCHIVE" ]; then
                log_error "Failed to download Spark from both S3 and internet"
                exit 1
            fi
        fi

        # Extract archive locally
        log_info "Extracting Spark archive locally..."
        tar -xzf "$SPARK_LOCAL_ARCHIVE" || {
            log_error "Failed to extract Spark archive"
            exit 1
        }

        # Rename to final directory name
        [ -d "$SPARK_DIST_DIR" ] && rm -rf "$SPARK_DIST_DIR"
        mv "$SPARK_LOCAL_DIR" "$SPARK_DIST_DIR" || {
            log_error "Failed to rename Spark directory"
            exit 1
        }

        # Fix permissions on Spark scripts after extraction (ensure executables have +x)
        chmod +x "$SPARK_DIST_DIR"/bin/* "$SPARK_DIST_DIR"/sbin/* 2>/dev/null || true

        log_success "Spark extracted locally: $SPARK_DIST_DIR"

        # Clean up archive
        rm -f "$SPARK_LOCAL_ARCHIVE"
    fi
fi

################################################################################
# Part 4: Launch EC2 Instances (Parallel)
################################################################################

log_step "Part 4: Launching EC2 Instances (Parallel)"

# Get Ubuntu 22.04 AMI (free tier eligible, no marketplace subscription needed)
log_info "Finding Ubuntu 22.04 AMI..."
export AMI_ID=$(aws ec2 describe-images \
  --owners 099720109477 \
  --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
            "Name=state,Values=available" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
  --output text \
  --region $AWS_REGION)
log_success "AMI ID: $AMI_ID"

# Helper function to launch an instance
_launch_instance() {
    local INSTANCE_TYPE=$1
    local INSTANCE_NAME=$2
    local INSTANCE_ROLE=$3

    # Build the run-instances command with optional IAM profile
    RUN_CMD="aws ec2 run-instances \
      --image-id $AMI_ID \
      --instance-type t3.large \
      --key-name $KEY_NAME \
      --security-group-ids $SPARK_SG_ID"

    if [ -n "$IAM_INSTANCE_PROFILE" ]; then
        RUN_CMD="$RUN_CMD --iam-instance-profile Name=$IAM_INSTANCE_PROFILE"
    fi

    RUN_CMD="$RUN_CMD \
      --count 1 \
      --block-device-mappings '[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":100,\"VolumeType\":\"gp3\"}}]' \
      --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME},{Key=Role,Value=$INSTANCE_ROLE}]' \
      --region $AWS_REGION \
      --query 'Instances[0].InstanceId' \
      --output text"

    eval $RUN_CMD
}

# Launch master and worker nodes in parallel
log_info "Launching master node in background..."
_launch_instance "t3.large" "spark-master" "master" > /tmp/master_instance_id.txt &
MASTER_LAUNCH_PID=$!

log_info "Launching 3 worker nodes in background..."
_launch_instance "t3.large" "spark-worker-1" "worker" > /tmp/worker1_instance_id.txt &
WORKER1_LAUNCH_PID=$!

_launch_instance "t3.large" "spark-worker-2" "worker" > /tmp/worker2_instance_id.txt &
WORKER2_LAUNCH_PID=$!

_launch_instance "t3.large" "spark-worker-3" "worker" > /tmp/worker3_instance_id.txt &
WORKER3_LAUNCH_PID=$!

# Wait for all instance launches to complete
log_info "Waiting for all instance launches to complete..."
wait $MASTER_LAUNCH_PID $WORKER1_LAUNCH_PID $WORKER2_LAUNCH_PID $WORKER3_LAUNCH_PID || {
    log_error "Failed to launch one or more instances"
    exit 1
}

# Retrieve instance IDs from background processes
MASTER_INSTANCE_ID=$(cat /tmp/master_instance_id.txt | tr -d '[:space:]')
WORKER1_INSTANCE_ID=$(cat /tmp/worker1_instance_id.txt | tr -d '[:space:]')
WORKER2_INSTANCE_ID=$(cat /tmp/worker2_instance_id.txt | tr -d '[:space:]')
WORKER3_INSTANCE_ID=$(cat /tmp/worker3_instance_id.txt | tr -d '[:space:]')

# Validate all instance IDs were retrieved
if [ -z "$MASTER_INSTANCE_ID" ] || [ -z "$WORKER1_INSTANCE_ID" ] || [ -z "$WORKER2_INSTANCE_ID" ] || [ -z "$WORKER3_INSTANCE_ID" ]; then
    log_error "Failed to retrieve one or more instance IDs"
    log_error "Master: $MASTER_INSTANCE_ID, Worker1: $WORKER1_INSTANCE_ID, Worker2: $WORKER2_INSTANCE_ID, Worker3: $WORKER3_INSTANCE_ID"
    exit 1
fi

log_success "Master instance launched: $MASTER_INSTANCE_ID"
log_success "Worker 1 instance launched: $WORKER1_INSTANCE_ID"
log_success "Worker 2 instance launched: $WORKER2_INSTANCE_ID"
log_success "Worker 3 instance launched: $WORKER3_INSTANCE_ID"

# Clean up temporary files
rm -f /tmp/master_instance_id.txt /tmp/worker1_instance_id.txt /tmp/worker2_instance_id.txt /tmp/worker3_instance_id.txt

# Wait for ALL instances to be in running state before proceeding
log_info "Waiting for all instances to be in running state..."
aws ec2 wait instance-running \
  --instance-ids $MASTER_INSTANCE_ID $WORKER1_INSTANCE_ID $WORKER2_INSTANCE_ID $WORKER3_INSTANCE_ID \
  --region $AWS_REGION
log_success "All instances are running"

# Wait a bit more for SSH to be ready
log_info "Waiting 30 seconds for SSH to be ready on all instances..."
sleep 30

################################################################################
# Part 5: Get Instance IP Addresses
################################################################################

log_step "Part 5: Getting Instance IP Addresses"

log_info "Retrieving IP addresses..."

export MASTER_PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $MASTER_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text \
  --region $AWS_REGION)

export MASTER_PRIVATE_IP=$(aws ec2 describe-instances \
  --instance-ids $MASTER_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' \
  --output text \
  --region $AWS_REGION)

export WORKER1_PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $WORKER1_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text \
  --region $AWS_REGION)

export WORKER1_PRIVATE_IP=$(aws ec2 describe-instances \
  --instance-ids $WORKER1_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' \
  --output text \
  --region $AWS_REGION)

export WORKER2_PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $WORKER2_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text \
  --region $AWS_REGION)

export WORKER2_PRIVATE_IP=$(aws ec2 describe-instances \
  --instance-ids $WORKER2_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' \
  --output text \
  --region $AWS_REGION)

export WORKER3_PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $WORKER3_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text \
  --region $AWS_REGION)

export WORKER3_PRIVATE_IP=$(aws ec2 describe-instances \
  --instance-ids $WORKER3_INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' \
  --output text \
  --region $AWS_REGION)

log_success "Master: $MASTER_PUBLIC_IP (public) / $MASTER_PRIVATE_IP (private)"
log_success "Worker 1: $WORKER1_PUBLIC_IP (public) / $WORKER1_PRIVATE_IP (private)"
log_success "Worker 2: $WORKER2_PUBLIC_IP (public) / $WORKER2_PRIVATE_IP (private)"
log_success "Worker 3: $WORKER3_PUBLIC_IP (public) / $WORKER3_PRIVATE_IP (private)"

# Save cluster configuration
log_info "Saving cluster configuration..."
cat > cluster-config.txt <<EOF
# Cluster Configuration - Created $(date)
AWS_REGION=$AWS_REGION
SPARK_SG_ID=$SPARK_SG_ID
KEY_NAME=$KEY_NAME
KEY_FILE=$KEY_FILE

# Instance IDs
MASTER_INSTANCE_ID=$MASTER_INSTANCE_ID
WORKER1_INSTANCE_ID=$WORKER1_INSTANCE_ID
WORKER2_INSTANCE_ID=$WORKER2_INSTANCE_ID
WORKER3_INSTANCE_ID=$WORKER3_INSTANCE_ID

# IP Addresses
MASTER_PUBLIC_IP=$MASTER_PUBLIC_IP
MASTER_PRIVATE_IP=$MASTER_PRIVATE_IP
WORKER1_PUBLIC_IP=$WORKER1_PUBLIC_IP
WORKER1_PRIVATE_IP=$WORKER1_PRIVATE_IP
WORKER2_PUBLIC_IP=$WORKER2_PUBLIC_IP
WORKER2_PRIVATE_IP=$WORKER2_PRIVATE_IP
WORKER3_PUBLIC_IP=$WORKER3_PUBLIC_IP
WORKER3_PRIVATE_IP=$WORKER3_PRIVATE_IP
EOF

# Save IP addresses for copying to nodes
cat > cluster-ips.txt <<EOF
MASTER_PUBLIC_IP=$MASTER_PUBLIC_IP
MASTER_PRIVATE_IP=$MASTER_PRIVATE_IP
WORKER1_PUBLIC_IP=$WORKER1_PUBLIC_IP
WORKER1_PRIVATE_IP=$WORKER1_PRIVATE_IP
WORKER2_PUBLIC_IP=$WORKER2_PUBLIC_IP
WORKER2_PRIVATE_IP=$WORKER2_PRIVATE_IP
WORKER3_PUBLIC_IP=$WORKER3_PUBLIC_IP
WORKER3_PRIVATE_IP=$WORKER3_PRIVATE_IP
EOF

log_success "Configuration saved to cluster-config.txt and cluster-ips.txt"

# Create SSH helper script for master node
log_info "Creating SSH helper script..."
cat > ssh_to_master_node.sh <<EOF
#!/bin/bash
# SSH to Spark Master Node
# Generated on $(date)
ssh -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP
EOF

chmod +x ssh_to_master_node.sh
log_success "SSH helper script created: ssh_to_master_node.sh"

################################################################################
# Part 6: Copy Files to All Nodes (Parallel)
################################################################################

log_step "Part 6: Copying Files to All Nodes (Parallel)"

SSH_OPTS="-i $KEY_FILE -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30 -o ServerAliveCountMax=20 -o TCPKeepAlive=yes -o ConnectTimeout=10"

# Function to copy files to a node
_copy_files_to_node() {
    local NODE_NAME=$1
    local NODE_IP=$2

    log_info "Copying files to $NODE_NAME ($NODE_IP)..."
    ssh $SSH_OPTS ubuntu@$NODE_IP "mkdir -p ~/spark-cluster" 2>/dev/null
    scp -r $SSH_OPTS cluster-files/* ubuntu@$NODE_IP:~/spark-cluster/ 2>/dev/null
    scp $SSH_OPTS cluster-ips.txt ubuntu@$NODE_IP:~/spark-cluster/ 2>/dev/null
    log_success "Files copied to $NODE_NAME"
}

# Copy files to all nodes in parallel
_copy_files_to_node "Master" $MASTER_PUBLIC_IP &
COPY_MASTER_PID=$!

_copy_files_to_node "Worker 1" $WORKER1_PUBLIC_IP &
COPY_WORKER1_PID=$!

_copy_files_to_node "Worker 2" $WORKER2_PUBLIC_IP &
COPY_WORKER2_PID=$!

_copy_files_to_node "Worker 3" $WORKER3_PUBLIC_IP &
COPY_WORKER3_PID=$!

# Wait for all copy operations to complete
log_info "Waiting for file copy operations to complete on all nodes..."
wait $COPY_MASTER_PID $COPY_WORKER1_PID $COPY_WORKER2_PID $COPY_WORKER3_PID || {
    log_error "Failed to copy files to one or more nodes"
    exit 1
}
log_success "Files copied to all nodes"

################################################################################
# Part 6.5: Copy Spark Directory to All Nodes (Parallel)
################################################################################

log_step "Part 6.5: Distributing Spark to All Nodes (Parallel)"

log_info "Copying pre-extracted Spark directory to all nodes (faster & more reliable)..."

# Function to copy Spark directory to a node
_copy_spark_directory() {
    local NODE_NAME=$1
    local NODE_IP=$2
    local SCP_ATTEMPTS=1
    local SCP_MAX_ATTEMPTS=3

    while [ $SCP_ATTEMPTS -le $SCP_MAX_ATTEMPTS ]; do
        log_info "Copying Spark directory to $NODE_NAME ($NODE_IP)... (attempt $SCP_ATTEMPTS/$SCP_MAX_ATTEMPTS)"

        # Copy entire spark directory recursively with retry logic
        if scp -r $SSH_OPTS "$SPARK_DIST_DIR" ubuntu@$NODE_IP:~/ 2>/dev/null; then
            # Verify directory exists on remote
            if ssh $SSH_OPTS ubuntu@$NODE_IP "[ -d ~/spark ]" 2>/dev/null; then
                # Fix permissions on Spark scripts after copying (ensure executables have +x)
                ssh $SSH_OPTS ubuntu@$NODE_IP 'chmod +x ~/spark/bin/* ~/spark/sbin/* 2>/dev/null' 2>/dev/null
                log_success "Spark directory copied to $NODE_NAME"
                return 0
            else
                log_warn "Spark directory verification failed on $NODE_NAME, retrying..."
            fi
        else
            log_warn "SCP to $NODE_NAME failed, retrying..."
        fi

        if [ $SCP_ATTEMPTS -lt $SCP_MAX_ATTEMPTS ]; then
            sleep 5
        fi
        SCP_ATTEMPTS=$((SCP_ATTEMPTS + 1))
    done

    log_error "Failed to copy Spark directory to $NODE_NAME after $SCP_MAX_ATTEMPTS attempts"
    return 1
}

# Copy Spark directory to all nodes in parallel
_copy_spark_directory "Master" $MASTER_PUBLIC_IP &
COPY_SPARK_MASTER_PID=$!

_copy_spark_directory "Worker 1" $WORKER1_PUBLIC_IP &
COPY_SPARK_WORKER1_PID=$!

_copy_spark_directory "Worker 2" $WORKER2_PUBLIC_IP &
COPY_SPARK_WORKER2_PID=$!

_copy_spark_directory "Worker 3" $WORKER3_PUBLIC_IP &
COPY_SPARK_WORKER3_PID=$!

# Wait for all Spark directory copies to complete
log_info "Waiting for Spark distribution to complete..."
wait $COPY_SPARK_MASTER_PID $COPY_SPARK_WORKER1_PID $COPY_SPARK_WORKER2_PID $COPY_SPARK_WORKER3_PID || {
    log_error "Failed to copy Spark directory to one or more nodes"
    exit 1
}
log_success "Spark directory copied to all nodes"

################################################################################
# Part 7: Setup Master and Worker Nodes (Parallel)
################################################################################

log_step "Part 7: Setting Up Master and Worker Nodes (Parallel)"

# Function to setup master node
_setup_master() {
    log_info "Configuring master node..."

    ssh $SSH_OPTS ubuntu@$MASTER_PUBLIC_IP 'bash -s' <<'MASTER_SETUP'
set -e

echo "[Master Setup] Installing system packages..."

# Wait for any existing apt processes to finish (cloud-init may still be running)
WAIT_TIME=0
MAX_WAIT=300  # 5 minutes max wait for apt lock
while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if sudo apt update -qq 2>/dev/null; then
        break
    fi
    echo "[Master Setup] Waiting for apt lock to be released... ($WAIT_TIME/$MAX_WAIT seconds)"
    sleep 5
    WAIT_TIME=$((WAIT_TIME + 5))
done

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo "ERROR: apt update failed after $MAX_WAIT seconds"
    exit 1
fi

sudo apt install -y openjdk-17-jdk-headless python3-pip python3-venv curl unzip wget 2>&1 | grep -v "^Selecting\|^Preparing\|^Unpacking" || true

echo "[Master Setup] Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | grep -v "^Downloading\|^Installing" || true
export PATH="$HOME/.local/bin:$PATH"

echo "[Master Setup] Installing Python dependencies..."
cd ~/spark-cluster || { echo "ERROR: spark-cluster directory not found"; exit 1; }
uv sync 2>&1 | tail -5

echo "[Master Setup] Verifying Spark directory..."
cd ~ || exit 1

# Spark directory was pre-copied from orchestrator in Part 6.5
# Just verify it exists
if [ ! -d ~/spark ]; then
    echo "ERROR: Spark directory not found"
    exit 1
fi

echo "  - Spark directory verified: ~/spark"

echo "[Master Setup] Downloading AWS JARs (Hadoop 3.3.4 for Spark 3.4.4)..."
cd ~/spark/jars || { echo "ERROR: spark/jars directory not found"; exit 1; }
echo "  - Downloading hadoop-aws 3.3.4..."
wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar || { echo "ERROR: Failed to download hadoop-aws"; exit 1; }
echo "  - Downloading aws-java-sdk-bundle 1.12.262..."
wget -q https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar || { echo "ERROR: Failed to download aws-java-sdk-bundle"; exit 1; }

echo "[Master Setup] Configuring environment variables..."
cat >> ~/.bashrc <<'EOF'
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export SPARK_HOME=$HOME/spark
export PATH="$HOME/.local/bin:$PATH"
export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
export PYSPARK_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python
export PYSPARK_DRIVER_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python
export PS1="[MASTER] \u@\h:\w\$ "
EOF

source ~/.bashrc

echo "[Master Setup] Master node setup complete!"
MASTER_SETUP

    log_success "Master node configured"
}

# Function to setup worker node
_setup_worker() {
    local WORKER_NUM=$1
    local WORKER_IP=$2

    log_info "Configuring Worker $WORKER_NUM ($WORKER_IP)..."

    ssh $SSH_OPTS ubuntu@$WORKER_IP "bash -s $WORKER_NUM" <<'WORKER_SETUP'
set -e

WORKER_NUM=$1

echo "[Worker $WORKER_NUM Setup] Installing system packages..."

# Wait for any existing apt processes to finish (cloud-init may still be running)
# Stagger the wait times to reduce lock contention across workers
STAGGER=$((WORKER_NUM * 10))  # 10, 20, 30 second stagger for workers 1, 2, 3
sleep $STAGGER

WAIT_TIME=0
MAX_WAIT=300  # 5 minutes max wait for apt lock
while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if sudo apt update -qq 2>/dev/null; then
        break
    fi
    echo "[Worker $WORKER_NUM Setup] Waiting for apt lock to be released... ($WAIT_TIME/$MAX_WAIT seconds)"
    sleep 5
    WAIT_TIME=$((WAIT_TIME + 5))
done

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo "ERROR: apt update failed after $MAX_WAIT seconds"
    exit 1
fi

sudo apt install -y openjdk-17-jdk-headless python3-pip python3-venv curl unzip wget 2>&1 | grep -v "^Selecting\|^Preparing\|^Unpacking" || true

echo "[Worker $WORKER_NUM Setup] Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | grep -v "^Downloading\|^Installing" || true
export PATH="$HOME/.local/bin:$PATH"

echo "[Worker $WORKER_NUM Setup] Installing Python dependencies..."
cd ~/spark-cluster || { echo "ERROR: spark-cluster directory not found"; exit 1; }
uv sync 2>&1 | tail -5

echo "[Worker $WORKER_NUM Setup] Verifying Spark directory..."
cd ~ || exit 1

# Spark directory was pre-copied from orchestrator in Part 6.5
# Just verify it exists
if [ ! -d ~/spark ]; then
    echo "ERROR: Spark directory not found"
    exit 1
fi

echo "  - Spark directory verified: ~/spark"

echo "[Worker $WORKER_NUM Setup] Downloading AWS JARs (Hadoop 3.3.4 for Spark 3.4.4)..."
cd ~/spark/jars || { echo "ERROR: spark/jars directory not found"; exit 1; }
wget -q https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar || { echo "ERROR: Failed to download hadoop-aws"; exit 1; }
wget -q https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar || { echo "ERROR: Failed to download aws-java-sdk-bundle"; exit 1; }

echo "[Worker $WORKER_NUM Setup] Configuring environment variables..."
cat >> ~/.bashrc <<EOF
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export SPARK_HOME=\$HOME/spark
export PATH="\$HOME/.local/bin:\$PATH"
export PATH=\$PATH:\$SPARK_HOME/bin:\$SPARK_HOME/sbin
export PYSPARK_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python
export PYSPARK_DRIVER_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python
export PS1="[WORKER-$WORKER_NUM] \u@\h:\w\$ "
EOF

source ~/.bashrc

echo "[Worker $WORKER_NUM Setup] Worker node setup complete!"
WORKER_SETUP

    log_success "Worker $WORKER_NUM configured"
}

# Setup master and worker nodes in parallel
_setup_master &
SETUP_MASTER_PID=$!

_setup_worker 1 $WORKER1_PUBLIC_IP &
SETUP_WORKER1_PID=$!

_setup_worker 2 $WORKER2_PUBLIC_IP &
SETUP_WORKER2_PID=$!

_setup_worker 3 $WORKER3_PUBLIC_IP &
SETUP_WORKER3_PID=$!

# Wait for all setup operations to complete before proceeding
log_info "Waiting for all node setup operations to complete..."
wait $SETUP_MASTER_PID $SETUP_WORKER1_PID $SETUP_WORKER2_PID $SETUP_WORKER3_PID || {
    log_error "Failed to setup one or more nodes"
    exit 1
}
log_success "All master and worker nodes configured"

################################################################################
# Part 8: Configure Spark on All Nodes (Parallel)
################################################################################

log_step "Part 8: Configuring Spark on All Nodes (Parallel)"

# Function to configure Spark on master
_configure_master_spark() {
    log_info "Configuring Spark on master..."

    ssh $SSH_OPTS ubuntu@$MASTER_PUBLIC_IP "bash -s" <<'SPARK_CONFIG'
set -e

source ~/spark-cluster/cluster-ips.txt

echo "[Spark Config] Creating spark-env.sh..."
cat > $HOME/spark/conf/spark-env.sh <<EOF
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export SPARK_MASTER_HOST=$MASTER_PRIVATE_IP
export SPARK_MASTER_PORT=7077
export PYSPARK_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python
export PYSPARK_DRIVER_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python
EOF

chmod +x $HOME/spark/conf/spark-env.sh

echo "[Spark Config] Creating workers file..."
cat > $HOME/spark/conf/workers <<EOF
$WORKER1_PRIVATE_IP
$WORKER2_PRIVATE_IP
$WORKER3_PRIVATE_IP
EOF

echo "[Spark Config] Spark configuration complete on master!"
SPARK_CONFIG

    log_success "Spark configured on master"
}

# Function to configure Spark on worker
_configure_worker_spark() {
    local WORKER_NUM=$1
    local WORKER_IP=$2

    log_info "Configuring Spark on Worker $WORKER_NUM..."

    ssh $SSH_OPTS ubuntu@$WORKER_IP 'bash -s' <<'WORKER_SPARK_CONFIG'
set -e

source ~/spark-cluster/cluster-ips.txt

cat > $HOME/spark/conf/spark-env.sh <<EOF
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PYSPARK_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python
export PYSPARK_DRIVER_PYTHON=/home/ubuntu/spark-cluster/.venv/bin/python
EOF

chmod +x $HOME/spark/conf/spark-env.sh
WORKER_SPARK_CONFIG

    log_success "Spark configured on Worker $WORKER_NUM"
}

# Configure Spark on master and workers in parallel
_configure_master_spark &
CONFIG_MASTER_PID=$!

_configure_worker_spark 1 $WORKER1_PUBLIC_IP &
CONFIG_WORKER1_PID=$!

_configure_worker_spark 2 $WORKER2_PUBLIC_IP &
CONFIG_WORKER2_PID=$!

_configure_worker_spark 3 $WORKER3_PUBLIC_IP &
CONFIG_WORKER3_PID=$!

# Wait for all configuration operations to complete
log_info "Waiting for all Spark configuration operations to complete..."
wait $CONFIG_MASTER_PID $CONFIG_WORKER1_PID $CONFIG_WORKER2_PID $CONFIG_WORKER3_PID || {
    log_error "Failed to configure Spark on one or more nodes"
    exit 1
}
log_success "Spark configured on all nodes"

################################################################################
# Part 9: Setup SSH Keys for Passwordless Access
################################################################################

log_step "Part 9: Setting Up SSH Keys for Passwordless Access"

log_info "Copying SSH key to master node..."
ssh $SSH_OPTS ubuntu@$MASTER_PUBLIC_IP "mkdir -p ~/.ssh && chmod 700 ~/.ssh" 2>/dev/null
scp $SSH_OPTS $KEY_FILE ubuntu@$MASTER_PUBLIC_IP:~/.ssh/ 2>/dev/null
ssh $SSH_OPTS ubuntu@$MASTER_PUBLIC_IP "chmod 400 ~/.ssh/$KEY_FILE" 2>/dev/null
log_success "SSH key copied to master"

log_info "Generating SSH key on master..."
ssh $SSH_OPTS ubuntu@$MASTER_PUBLIC_IP 'bash -s' <<'SSH_SETUP'
if [ ! -f ~/.ssh/id_rsa ]; then
    ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa -q
fi
SSH_SETUP
log_success "SSH key generated on master"

# Get master's public key
MASTER_PUB_KEY=$(ssh $SSH_OPTS ubuntu@$MASTER_PUBLIC_IP "cat ~/.ssh/id_rsa.pub")

log_info "Adding master's key to authorized_keys on all workers..."
for WORKER_IP in $WORKER1_PUBLIC_IP $WORKER2_PUBLIC_IP $WORKER3_PUBLIC_IP; do
    ssh $SSH_OPTS ubuntu@$WORKER_IP "echo '$MASTER_PUB_KEY' >> ~/.ssh/authorized_keys" 2>/dev/null
done
log_success "Passwordless SSH configured"

################################################################################
# Part 10: Start Spark Cluster and Verify All Nodes
################################################################################

log_step "Part 10: Starting Spark Cluster and Verifying All Nodes"

log_info "Starting Spark master..."
ssh $SSH_OPTS ubuntu@$MASTER_PUBLIC_IP 'cd $HOME/spark && ./sbin/start-master.sh' > /tmp/master_start.log 2>&1 || {
    log_warn "start-master.sh returned exit code $?, but checking if process started..."
}
sleep 5
log_success "Spark master started"

log_info "Waiting for master to be fully initialized..."
sleep 5

log_info "Verifying master process..."
MASTER_RUNNING=$(ssh $SSH_OPTS ubuntu@$MASTER_PUBLIC_IP 'jps | grep Master | wc -l' 2>/dev/null | tr -d '[:space:]' || echo 0)
if [ -z "$MASTER_RUNNING" ] || [ "$MASTER_RUNNING" = "" ]; then
    MASTER_RUNNING=0
fi
if [ "$MASTER_RUNNING" -lt 1 ]; then
    log_error "Master process is not running!"
    exit 1
fi
log_success "Master process is running"

log_info "Starting Spark workers on master..."
ssh $SSH_OPTS ubuntu@$MASTER_PUBLIC_IP 'cd $HOME/spark && ./sbin/start-workers.sh' > /tmp/workers_start.log 2>&1 || {
    log_warn "start-workers.sh returned exit code $?, but checking if workers started..."
}
sleep 5
log_success "Spark worker startup initiated"

log_info "Waiting for all workers to fully start up and register with master..."
WAIT_COUNT=0
MAX_WAIT=120  # Maximum 120 seconds to wait for workers to register with master
EXPECTED_WORKERS=3

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    # Count active workers by checking individual worker nodes
    REGISTERED_WORKERS=0
    for i in 1 2 3; do
        WORKER_IP_VAR="WORKER${i}_PUBLIC_IP"
        WORKER_IP=${!WORKER_IP_VAR}
        if ssh $SSH_OPTS ubuntu@$WORKER_IP 'jps | grep -q Worker' 2>/dev/null; then
            REGISTERED_WORKERS=$((REGISTERED_WORKERS + 1))
        fi
    done

    if [ "$REGISTERED_WORKERS" -ge "$EXPECTED_WORKERS" ]; then
        log_success "All $EXPECTED_WORKERS workers are running and registered"
        break
    fi

    WAIT_COUNT=$((WAIT_COUNT + 5))
    if [ $WAIT_COUNT -lt $MAX_WAIT ]; then
        log_info "Workers running: $REGISTERED_WORKERS/$EXPECTED_WORKERS. Waiting... ($WAIT_COUNT/$MAX_WAIT seconds)"
        sleep 5
    fi
done

if [ "$REGISTERED_WORKERS" -lt "$EXPECTED_WORKERS" ]; then
    log_error "Not all workers started. Only $REGISTERED_WORKERS/$EXPECTED_WORKERS workers are running"
    log_error "Checking individual worker statuses..."

    WORKER_COUNT=0
    for i in 1 2 3; do
        WORKER_IP_VAR="WORKER${i}_PUBLIC_IP"
        WORKER_IP=${!WORKER_IP_VAR}
        if ssh $SSH_OPTS ubuntu@$WORKER_IP 'jps | grep -q Worker' 2>/dev/null; then
            log_info "Worker $i ($WORKER_IP): Running"
            WORKER_COUNT=$((WORKER_COUNT + 1))
        else
            log_warn "Worker $i ($WORKER_IP): NOT running"
        fi
    done

    if [ $WORKER_COUNT -lt 3 ]; then
        log_error "Failed to start all workers. Please check logs and SSH to nodes for debugging."
        exit 1
    fi
fi

################################################################################
# Part 11: Final Verification
################################################################################

log_step "Part 11: Final Cluster Verification"

log_info "Performing final health checks..."

# Check master
log_info "Checking master health..."
if ! ssh $SSH_OPTS ubuntu@$MASTER_PUBLIC_IP 'jps | grep -q Master' 2>/dev/null; then
    log_error "Master process not running!"
    exit 1
fi
log_success "Master health check passed"

# Check all workers
log_info "Checking individual worker health..."
HEALTHY_WORKERS=0
for i in 1 2 3; do
    WORKER_IP_VAR="WORKER${i}_PUBLIC_IP"
    WORKER_IP=${!WORKER_IP_VAR}

    if ssh $SSH_OPTS ubuntu@$WORKER_IP 'jps | grep -q Worker' 2>/dev/null; then
        log_success "Worker $i health check passed"
        HEALTHY_WORKERS=$((HEALTHY_WORKERS + 1))
    else
        log_warn "Worker $i health check failed"
    fi
done

if [ $HEALTHY_WORKERS -lt 3 ]; then
    log_error "Not all workers are healthy. Only $HEALTHY_WORKERS/3 are running."
    exit 1
fi

log_success "All worker health checks passed"

################################################################################
# Cleanup: Delete Local Spark Archive
################################################################################

log_step "Part 12: Cleanup"

if [ -f "$SPARK_LOCAL_PATH" ]; then
    log_info "Deleting local Spark archive: $SPARK_LOCAL_PATH"
    rm -f "$SPARK_LOCAL_PATH"
    log_success "Local Spark archive cleaned up"
fi

################################################################################
# Final Summary
################################################################################

log_step "✓ Spark Cluster Setup Complete!"

echo ""
echo "=================================================="
echo "           CLUSTER INFORMATION"
echo "=================================================="
echo ""
echo "Master Node:"
echo "  Public IP:  $MASTER_PUBLIC_IP"
echo "  Private IP: $MASTER_PRIVATE_IP"
echo ""
echo "Worker Nodes:"
echo "  Worker 1: $WORKER1_PUBLIC_IP (private: $WORKER1_PRIVATE_IP)"
echo "  Worker 2: $WORKER2_PUBLIC_IP (private: $WORKER2_PRIVATE_IP)"
echo "  Worker 3: $WORKER3_PUBLIC_IP (private: $WORKER3_PRIVATE_IP)"
echo ""
echo "=================================================="
echo "           ACCESS INFORMATION"
echo "=================================================="
echo ""
echo "SSH to Master:"
echo "  ssh -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP"
echo ""
echo "Or use the helper script:"
echo "  ./ssh_to_master_node.sh"
echo ""
echo "Spark Master Web UI:"
echo "  http://$MASTER_PUBLIC_IP:8080"
echo ""
echo "Spark Application UI (when job running):"
echo "  http://$MASTER_PUBLIC_IP:4040"
echo ""
echo "=================================================="
echo "           CONFIGURATION FILES"
echo "=================================================="
echo ""
echo "Cluster configuration: cluster-config.txt"
echo "IP addresses: cluster-ips.txt"
echo "SSH key: $KEY_FILE"
echo ""
echo "=================================================="
echo "           NEXT STEPS"
echo "=================================================="
echo ""
echo "1. Access Spark Master Web UI to verify 3 workers are connected"
echo "2. SSH to master and run a test job:"
echo "   ssh -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP"
echo "   cd ~/spark-cluster"
echo "   source cluster-ips.txt"
echo "   uv run python nyc_tlc_problem1_cluster.py spark://\$MASTER_PRIVATE_IP:7077"
echo ""
echo "3. To stop the cluster later:"
echo "   ssh -i $KEY_FILE ubuntu@$MASTER_PUBLIC_IP '\$SPARK_HOME/sbin/stop-all.sh'"
echo ""
echo "4. To terminate all resources, use the cleanup script or:"
echo "   source cluster-config.txt"
echo "   aws ec2 terminate-instances --instance-ids \$MASTER_INSTANCE_ID \$WORKER1_INSTANCE_ID \$WORKER2_INSTANCE_ID \$WORKER3_INSTANCE_ID --region \$AWS_REGION"
echo "   aws ec2 delete-security-group --group-id \$SPARK_SG_ID --region \$AWS_REGION"
echo "   aws ec2 delete-key-pair --key-name \$KEY_NAME --region \$AWS_REGION"
echo ""
echo "=================================================="
echo "           LOG FILE"
echo "=================================================="
echo ""
echo "Full setup log saved to: $LOG_FILE"
echo "If you encounter issues, check this log file for detailed error messages."
echo ""
echo "=================================================="
echo "           SETUP SUMMARY"
echo "=================================================="
ELAPSED_MINUTES=$((SECONDS / 60))
ELAPSED_SECONDS=$((SECONDS % 60))
echo "Total setup time: $ELAPSED_MINUTES minutes $ELAPSED_SECONDS seconds"
echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="
echo ""
log_success "Setup script completed successfully!"