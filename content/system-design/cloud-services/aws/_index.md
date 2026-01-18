+++
title = "AWS Services"
description = "Amazon Web Services (AWS): Comprehensive guide to AWS services including EC2, Lambda, S3, RDS, DynamoDB, VPC, CloudFront, and more for building scalable cloud applications."
+++

Amazon Web Services (AWS) is the leading cloud provider offering a comprehensive suite of services for compute, storage, databases, networking, messaging, security, and monitoring. This section covers essential AWS services and how to use them in system design.

---

## 📖 AWS Services

### Compute
- **[AWS Compute Services]({{< ref "1-AWS_Compute_Services.md" >}})** - EC2, Lambda, ECS, EKS, Fargate, and container services

### Storage
- **[AWS Storage Services]({{< ref "2-AWS_Storage_Services.md" >}})** - S3, EBS, EFS, Glacier, and storage solutions

### Databases
- **[AWS Database Services]({{< ref "3-AWS_Database_Services.md" >}})** - RDS, DynamoDB, ElastiCache, Redshift, and database services

### Networking
- **[AWS Networking Services]({{< ref "4-AWS_Networking_Services.md" >}})** - VPC, CloudFront, Route 53, ALB/NLB, and networking services

### Messaging
- **[AWS Messaging Services]({{< ref "5-AWS_Messaging_Services.md" >}})** - SQS, SNS, EventBridge, Kinesis, and messaging services

### Security
- **[AWS Security Services]({{< ref "6-AWS_Security_Services.md" >}})** - IAM, Cognito, Secrets Manager, WAF, and security services

### Monitoring
- **[AWS Monitoring Services]({{< ref "7-AWS_Monitoring_Services.md" >}})** - CloudWatch, X-Ray, CloudTrail, and observability services

### Architecture
- **[AWS Architecture Patterns]({{< ref "8-AWS_Architecture_Patterns.md" >}})** - Well-Architected Framework, common patterns, and best practices

---

## 🎯 AWS Core Concepts

### Regions and Availability Zones
- **Regions**: Geographic locations (us-east-1, eu-west-1, etc.)
- **Availability Zones (AZs)**: Isolated data centers within a region
- **Multi-AZ**: Deploy across AZs for high availability

### Service Categories
- **Compute**: Virtual machines, containers, serverless
- **Storage**: Object, block, and file storage
- **Databases**: Relational, NoSQL, in-memory, data warehousing
- **Networking**: VPCs, load balancers, CDNs, DNS
- **Messaging**: Queues, pub/sub, event streaming
- **Security**: Identity, access control, encryption
- **Monitoring**: Metrics, logs, traces

### Pricing Models
- **On-Demand**: Pay as you go
- **Reserved Instances**: 1-3 year commitments, discounts
- **Spot Instances**: Bid on unused capacity, up to 90% discount
- **Savings Plans**: Flexible pricing for compute usage

---

## 🚀 Getting Started

1. **Create AWS Account**: Sign up for AWS account
2. **Set Up IAM**: Configure users, roles, and permissions
3. **Choose Region**: Select appropriate region for your use case
4. **Start with Core Services**: EC2, S3, RDS
5. **Explore Serverless**: Lambda, API Gateway
6. **Learn Networking**: VPC, security groups
7. **Implement Monitoring**: CloudWatch, logging

---

## 💡 Best Practices

- **Use Managed Services**: Reduce operational overhead
- **Design for Failure**: Multi-AZ, auto-scaling, health checks
- **Secure by Default**: IAM, encryption, VPCs
- **Monitor Everything**: CloudWatch, logging, alerts
- **Optimize Costs**: Right-sizing, reserved instances, spot instances
- **Follow Well-Architected Framework**: Operational excellence, security, reliability, performance, cost optimization

