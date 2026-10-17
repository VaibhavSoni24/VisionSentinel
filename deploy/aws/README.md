# Vision Sentinel — AWS Cloud Deployment Guide

**Competition Track:** OpenCV AI Competition 2026, powered by AWS  
**Target Architecture:** AWS Graviton3 / Graviton2 (Arm64) or x86_64  
**Cost Governance:** Strict Zero-Unauthorized-Charges Policy  

---

## 1. Cloud Architecture Overview

Vision Sentinel executes substantive OpenCV 5 computer vision directly inside a lightweight container deployed on Amazon Web Services.

```
                  +----------------------------------------------+
                  |                 AWS Cloud                    |
                  |                                              |
[User / Camera] --+--> [Application Load Balancer / Port 8000]   |
                  |                       |                      |
                  |          [Amazon ECS / EC2 Service]          |
                  |     +----------------------------------+     |
                  |     |   Vision Sentinel Container      |     |
                  |     |   - OpenCV 5 Video Ingest        |     |
                  |     |   - MOG2 Motion Extraction       |     |
                  |     |   - Trajectory Tracking & Zones  |     |
                  |     |   - Autonomous Agent Decision    |     |
                  |     +----------------------------------+     |
                  |                       |                      |
                  |           [Amazon CloudWatch Logs]           |
                  +----------------------------------------------+
```

---

## 2. Zero-Cost & Anti-Charge Policy

> [!CAUTION]
> The repository owner has imposed a strict ban on unauthorized cloud expenditures.
> Before creating live AWS resources, verify whether your account holds an approved competition credit grant or operates under the standard 12-month AWS Free Tier.

### AWS Free Tier Allowances:
- **Compute:** 750 hours/month of `t2.micro` or `t3.micro` (in regions where t2 is unavailable). Certain regions also offer a 750-hour trial of `t4g.micro` (AWS Graviton).
- **Amazon ECR:** 500 MB/month of storage in the private registry.
- **Amazon CloudWatch:** 5 GB of log ingestion, 10 custom metrics, 3 dashboards.

---

## 3. Container Build & Packaging

### Local Multi-Architecture Build (using Docker Buildx)
```bash
# Initialize buildx builder
docker buildx create --name sentinels-builder --use
docker buildx inspect --bootstrap

# Build for both x86_64 and AWS Graviton (Arm64)
docker buildx build --platform linux/amd64,linux/arm64 -t vision-sentinel:latest .
```

---

## 4. Deployment Instructions

### Step 1: Authenticate and Push to Amazon ECR
```bash
# Retrieve login password
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com

# Create private repository
aws ecr create-repository --repository-name vision-sentinel --region ap-south-1

# Tag and push container image
docker tag vision-sentinel:latest <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/vision-sentinel:latest
docker push <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/vision-sentinel:latest
```

### Step 2: Minimal IAM Least-Privilege Execution Role
Attach this policy to the ECS Task Execution Role (`ecsTaskExecutionRole`):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    }
  ]
}
```

### Step 3: Deploy via Amazon ECS or EC2 Graviton Instance
Launch a `t4g.small` (Arm64 Graviton) instance using Amazon Linux 2023 or deploy via ECS Fargate with CPU architecture set to `ARM64`.

```bash
# Pull and execute on EC2 Graviton
docker run -d --restart unless-stopped -p 8000:8000 \
  -e APP_ENV=production \
  -e APP_HOST=0.0.0.0 \
  -e APP_PORT=8000 \
  --name vision-sentinel \
  <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/vision-sentinel:latest
```

---

## 5. Live Smoke-Testing & Verification

Verify the live cloud deployment by testing the healthcheck and diagnostic status endpoints:

```bash
# 1. Verify healthcheck and OpenCV 5 version
curl -s http://<PUBLIC_IP>:8000/health | jq .

# Expected Output:
# {
#   "status": "healthy",
#   "app": "Vision Sentinel",
#   "version": "0.1.0",
#   "opencv_version": "5.0.0",
#   "is_opencv_5": true
# }

# 2. Trigger the built-in synthetic verification demo remotely
curl -s -X POST http://<PUBLIC_IP>:8000/api/demo | jq .status
# Expected Output: "completed"
```

---

## 6. Teardown and Cleanup Runbook

To prevent recurring costs after competition demonstration:
```bash
# Stop and delete container
docker stop vision-sentinel && docker rm vision-sentinel

# Delete ECR repository and images
aws ecr delete-repository --repository-name vision-sentinel --force --region ap-south-1

# Terminate EC2 / ECS services
aws ecs delete-service --cluster vision-sentinel-cluster --service vision-sentinel-svc --force
```
