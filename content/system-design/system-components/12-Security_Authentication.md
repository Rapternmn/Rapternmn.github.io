+++
title = "Security & Authentication"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 12
description = "Security, authentication, and authorization in system design: OAuth, JWT, RBAC, security best practices, encryption, and secrets management."
+++

---

## Introduction

Security is fundamental to system design. It protects data, services, and users from threats. Understanding authentication, authorization, and security best practices is essential for building secure systems.

---

## Security Fundamentals

### CIA Triad

**Confidentiality**: Data is accessible only to authorized users.

**Integrity**: Data is accurate and unmodified.

**Availability**: Data and services are accessible when needed.

---

## Authentication vs Authorization

### Authentication

**Definition**: Verifying user identity.

**Question**: "Who are you?"

**Methods**: 
- Username/password
- OAuth
- JWT
- API keys
- Biometrics

---

### Authorization

**Definition**: Determining what user can access.

**Question**: "What can you do?"

**Models**: 
- RBAC
- ABAC
- ACL

---

## Authentication Mechanisms

### 1. Username/Password

**How it works**: User provides username and password.

**Security**: 
- Hash passwords (bcrypt, Argon2)
- Salt passwords
- Rate limiting
- Password policies

**Use Cases**: Traditional web applications

---

### 2. OAuth 2.0

**Definition**: Industry-standard authorization framework.

**Flows**: 
- **Authorization Code**: Web applications
- **Client Credentials**: Server-to-server
- **Implicit**: Legacy (deprecated)
- **Resource Owner Password**: Trusted clients

**Use Cases**: Third-party authentication, API access

---

### 3. OpenID Connect (OIDC)

**Definition**: Identity layer on top of OAuth 2.0.

**Features**: 
- Authentication (not just authorization)
- User info endpoint
- ID tokens

**Use Cases**: User authentication, SSO

---

### 4. JWT (JSON Web Tokens)

**Definition**: Self-contained tokens with claims.

**Structure**: 
- Header: Algorithm, type
- Payload: Claims (user, permissions)
- Signature: Verification

**Advantages**: 
- Stateless
- Scalable
- Self-contained

**Disadvantages**: 
- Can't revoke easily
- Size limitations
- Security if not properly signed

**Use Cases**: API authentication, stateless sessions

---

### 5. API Keys

**Definition**: Simple authentication tokens.

**Advantages**: 
- Simple
- Easy to implement

**Disadvantages**: 
- Less secure
- Hard to revoke
- Can be leaked

**Use Cases**: API access, service-to-service

---

### 6. mTLS (Mutual TLS)

**Definition**: Both client and server authenticate.

**How it works**: 
- Client certificate
- Server certificate
- Mutual verification

**Use Cases**: Service-to-service authentication, high security

---

## Authorization Models

### 1. RBAC (Role-Based Access Control)

**Definition**: Permissions assigned to roles, users assigned to roles.

**Example**: 
- Role: Admin (all permissions)
- Role: User (limited permissions)
- User assigned to role

**Advantages**: 
- Simple
- Easy to manage
- Scalable

**Use Cases**: Most applications

---

### 2. ABAC (Attribute-Based Access Control)

**Definition**: Permissions based on attributes.

**Attributes**: 
- User attributes (department, location)
- Resource attributes (sensitivity, owner)
- Environmental attributes (time, location)

**Advantages**: 
- Fine-grained
- Flexible
- Context-aware

**Use Cases**: Complex authorization needs

---

### 3. ACL (Access Control Lists)

**Definition**: List of permissions for each resource.

**Example**: 
- File: user1 (read), user2 (read, write)
- Resource: list of users and permissions

**Use Cases**: File systems, simple resources

---

## Security Best Practices

### 1. Encryption at Rest

**Definition**: Encrypt data stored on disk.

**Methods**: 
- Database encryption
- File system encryption
- Object storage encryption

**Use Cases**: Sensitive data, compliance

---

### 2. Encryption in Transit (TLS/SSL)

**Definition**: Encrypt data during transmission.

**TLS Versions**: Use TLS 1.2+ (avoid older versions).

**Certificate Management**: 
- Valid certificates
- Certificate rotation
- Certificate pinning

**Use Cases**: All network communication

---

### 3. Secrets Management

**Definition**: Secure storage of sensitive data.

**Secrets**: 
- API keys
- Passwords
- Certificates
- Database credentials

**Tools**: 
- HashiCorp Vault
- AWS Secrets Manager
- Kubernetes Secrets

---

### 4. Input Validation

**Definition**: Validate all user input.

**Types**: 
- Type validation
- Range validation
- Format validation
- Business rule validation

**Prevents**: 
- SQL injection
- XSS
- Command injection

---

### 5. SQL Injection Prevention

**Problem**: Malicious SQL in user input.

**Solutions**: 
- Parameterized queries
- Prepared statements
- Input validation
- ORM frameworks

---

### 6. XSS Prevention

**Problem**: Malicious scripts in user input.

**Solutions**: 
- Input sanitization
- Output encoding
- Content Security Policy (CSP)
- HttpOnly cookies

---

### 7. CSRF Protection

**Problem**: Forged requests from other sites.

**Solutions**: 
- CSRF tokens
- SameSite cookies
- Origin validation

---

## Security Patterns

### 1. Zero-Trust Architecture

**Definition**: Never trust, always verify.

**Principles**: 
- Verify explicitly
- Least privilege
- Assume breach

**Use Cases**: High-security environments

---

### 2. Defense in Depth

**Definition**: Multiple security layers.

**Layers**: 
- Network security
- Application security
- Data security
- Monitoring

---

### 3. Principle of Least Privilege

**Definition**: Minimum permissions needed.

**Benefits**: 
- Reduce attack surface
- Limit damage
- Better security

---

## API Security

### Rate Limiting

**Definition**: Limit number of requests.

**Benefits**: 
- Prevent abuse
- DDoS protection
- Fair usage

---

### API Authentication

**Methods**: 
- API keys
- OAuth 2.0
- JWT
- mTLS

---

### API Versioning and Deprecation

**Versioning**: 
- URL versioning
- Header versioning
- Query parameter

**Deprecation**: 
- Announce deprecation
- Provide migration path
- Sunset old versions

---

## Network Security

### Firewalls

**Definition**: Control network traffic.

**Types**: 
- Network firewalls
- Application firewalls (WAF)
- Host-based firewalls

---

### VPNs

**Definition**: Encrypted network connection.

**Use Cases**: 
- Remote access
- Site-to-site connections
- Secure communication

---

### Private Networks (VPC)

**Definition**: Isolated network environment.

**Features**: 
- Private IP ranges
- Network isolation
- Security groups

**Use Cases**: Cloud deployments, isolation

---

## Security Compliance

### GDPR

**Definition**: European data protection regulation.

**Requirements**: 
- Data protection
- User rights
- Data breach notification
- Privacy by design

---

### HIPAA

**Definition**: US healthcare data protection.

**Requirements**: 
- Protected health information
- Access controls
- Audit logs
- Encryption

---

### SOC 2

**Definition**: Security and availability controls.

**Requirements**: 
- Security controls
- Availability
- Processing integrity
- Confidentiality
- Privacy

---

## Technologies

### OAuth 2.0 Providers

**Auth0**: Identity platform.

**Okta**: Enterprise identity.

**Features**: 
- OAuth/OIDC
- User management
- SSO
- MFA

---

### Keycloak

**Type**: Open source identity management.

**Features**: 
- OAuth/OIDC
- User federation
- Social login
- Admin UI

---

### HashiCorp Vault

**Type**: Secrets management.

**Features**: 
- Secrets storage
- Dynamic secrets
- Encryption as a service
- Access control

---

### AWS IAM

**Type**: AWS identity and access management.

**Features**: 
- User management
- Role-based access
- Policy management
- MFA

---

### Google Cloud IAM

**Type**: GCP identity and access management.

**Features**: 
- Resource access control
- Service accounts
- Policy management
- Audit logs

---

## Use Cases

### User Authentication

Verify user identity for application access.

---

### Service-to-Service Authentication

Secure communication between services.

---

### API Security

Protect APIs from unauthorized access.

---

### Data Protection

Encrypt and protect sensitive data.

---

## Best Practices

1. **Encrypt Everything**: At rest and in transit
2. **Least Privilege**: Minimum permissions
3. **Input Validation**: Validate all input
4. **Secrets Management**: Never hardcode secrets
5. **Regular Updates**: Keep software updated
6. **Monitoring**: Monitor for security events
7. **Incident Response**: Plan for security incidents

---

## Key Takeaways

- **Security** is fundamental to system design
- **Authentication** verifies identity, **authorization** controls access
- **OAuth/JWT** are common authentication mechanisms
- **RBAC/ABAC** are authorization models
- **Best practices** include encryption, validation, secrets management
- **Compliance** requirements vary by industry/region
- **Defense in depth** provides multiple security layers

---

## Related Topics

- **[API Gateway]({{< ref "4-API_Gateway.md" >}})** - API authentication and security
- **[Service Discovery & Service Mesh]({{< ref "6-Service_Discovery_Service_Mesh.md" >}})** - mTLS for service communication
- **[Monitoring & Observability]({{< ref "11-Monitoring_Observability.md" >}})** - Security monitoring

