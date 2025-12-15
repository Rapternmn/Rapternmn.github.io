+++
title = "API Design Guidelines"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 1
description = "Comprehensive guide to API design: REST principles, HTTP methods, status codes, request/response design, versioning, documentation, and best practices for building scalable APIs."
+++

---

## Introduction

API Design Guidelines provide best practices for designing RESTful APIs that are intuitive, consistent, scalable, and maintainable. Well-designed APIs improve developer experience, reduce integration time, and enable long-term maintainability.

---

## REST Principles

### What is REST?

**REST** (Representational State Transfer) is an architectural style for designing web services. RESTful APIs use HTTP methods to perform operations on resources.

### REST Principles

1. **Stateless**: Each request contains all information needed to process it
2. **Client-Server**: Separation of concerns between client and server
3. **Cacheable**: Responses should be cacheable when appropriate
4. **Uniform Interface**: Consistent interface for all resources
5. **Layered System**: System can be composed of hierarchical layers
6. **Code on Demand** (optional): Server can send executable code to client

---

## Resource Naming

### Use Nouns, Not Verbs

**Good**:
```
GET /users
GET /users/123
POST /users
PUT /users/123
DELETE /users/123
```

**Bad**:
```
GET /getUsers
GET /getUser/123
POST /createUser
PUT /updateUser/123
DELETE /deleteUser/123
```

### Use Plural Nouns

**Good**:
```
GET /users
GET /orders
GET /products
```

**Bad**:
```
GET /user
GET /order
GET /product
```

### Use Hierarchical Structure

**Good**:
```
GET /users/123/orders
GET /users/123/orders/456
GET /organizations/789/users
```

**Bad**:
```
GET /userOrders?userId=123
GET /getOrder?userId=123&orderId=456
```

### Use Hyphens, Not Underscores

**Good**:
```
GET /user-profiles
GET /order-items
```

**Bad**:
```
GET /user_profiles
GET /orderItems
```

---

## HTTP Methods

### GET

**Purpose**: Retrieve a resource or collection

**Characteristics**:
- Idempotent (safe to call multiple times)
- Should not modify server state
- Can be cached

**Examples**:
```
GET /users
GET /users/123
GET /users/123/orders
```

### POST

**Purpose**: Create a new resource

**Characteristics**:
- Not idempotent (may create duplicates)
- Request body contains resource data
- Returns created resource

**Examples**:
```
POST /users
Body: { "name": "John", "email": "john@example.com" }

POST /users/123/orders
Body: { "productId": 456, "quantity": 2 }
```

### PUT

**Purpose**: Update or replace a resource

**Characteristics**:
- Idempotent (same request = same result)
- Request body contains complete resource
- Creates resource if it doesn't exist

**Examples**:
```
PUT /users/123
Body: { "name": "John Doe", "email": "john@example.com" }
```

### PATCH

**Purpose**: Partial update of a resource

**Characteristics**:
- Idempotent (should be)
- Request body contains only fields to update
- More efficient than PUT for partial updates

**Examples**:
```
PATCH /users/123
Body: { "email": "newemail@example.com" }
```

### DELETE

**Purpose**: Delete a resource

**Characteristics**:
- Idempotent (deleting twice = same result)
- Usually no request body
- Returns 204 No Content or 200 OK

**Examples**:
```
DELETE /users/123
DELETE /users/123/orders/456
```

---

## HTTP Status Codes

### 2xx Success

- **200 OK**: Request succeeded (GET, PUT, PATCH)
- **201 Created**: Resource created (POST)
- **202 Accepted**: Request accepted, processing asynchronously
- **204 No Content**: Success, no content to return (DELETE)

### 3xx Redirection

- **301 Moved Permanently**: Resource moved permanently
- **302 Found**: Resource temporarily moved
- **304 Not Modified**: Resource not modified (caching)

### 4xx Client Error

- **400 Bad Request**: Invalid request syntax
- **401 Unauthorized**: Authentication required
- **403 Forbidden**: Authenticated but not authorized
- **404 Not Found**: Resource not found
- **409 Conflict**: Resource conflict (e.g., duplicate)
- **422 Unprocessable Entity**: Valid syntax but semantic errors
- **429 Too Many Requests**: Rate limit exceeded

### 5xx Server Error

- **500 Internal Server Error**: Generic server error
- **502 Bad Gateway**: Invalid response from upstream
- **503 Service Unavailable**: Service temporarily unavailable
- **504 Gateway Timeout**: Upstream timeout

---

## Request Design

### Query Parameters

**Use for**:
- Filtering
- Sorting
- Pagination
- Searching

**Examples**:
```
GET /users?status=active&role=admin
GET /users?sort=name&order=asc
GET /users?page=1&limit=20
GET /users?search=john
```

### Request Headers

**Common Headers**:
```
Authorization: Bearer <token>
Content-Type: application/json
Accept: application/json
X-Request-ID: <unique-id>
```

### Request Body

**Content-Type**: `application/json`

**Structure**:
```json
{
  "field1": "value1",
  "field2": "value2",
  "nested": {
    "field": "value"
  }
}
```

---

## Response Design

### Consistent Structure

**Standard Response**:
```json
{
  "data": {
    "id": "123",
    "name": "John",
    "email": "john@example.com"
  },
  "meta": {
    "timestamp": "2025-12-15T10:00:00Z"
  }
}
```

**Collection Response**:
```json
{
  "data": [
    { "id": "123", "name": "John" },
    { "id": "456", "name": "Jane" }
  ],
  "meta": {
    "total": 100,
    "page": 1,
    "limit": 20
  },
  "links": {
    "self": "/users?page=1",
    "next": "/users?page=2",
    "prev": null
  }
}
```

### Error Response

**Standard Error Format**:
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format"
      }
    ],
    "requestId": "req-123"
  }
}
```

---

## Pagination

### Offset-Based Pagination

**Query Parameters**:
```
GET /users?page=1&limit=20
```

**Response**:
```json
{
  "data": [...],
  "meta": {
    "page": 1,
    "limit": 20,
    "total": 100,
    "totalPages": 5
  }
}
```

**Pros**: Simple, easy to implement
**Cons**: Performance issues with large offsets

### Cursor-Based Pagination

**Query Parameters**:
```
GET /users?cursor=abc123&limit=20
```

**Response**:
```json
{
  "data": [...],
  "meta": {
    "limit": 20,
    "hasMore": true
  },
  "links": {
    "next": "/users?cursor=xyz789&limit=20"
  }
}
```

**Pros**: Better performance, consistent results
**Cons**: More complex, no direct page access

---

## Filtering and Sorting

### Filtering

**Query Parameters**:
```
GET /users?status=active&role=admin
GET /users?age[gte]=18&age[lte]=65
GET /users?tags=tag1,tag2
```

**Operators**:
- `eq`: equals
- `ne`: not equals
- `gt`: greater than
- `gte`: greater than or equal
- `lt`: less than
- `lte`: less than or equal
- `in`: in array
- `like`: pattern matching

### Sorting

**Query Parameters**:
```
GET /users?sort=name&order=asc
GET /users?sort=-createdAt  // descending
GET /users?sort=name,createdAt  // multiple fields
```

---

## API Versioning

### URL Versioning

**Example**:
```
GET /v1/users
GET /v2/users
```

**Pros**: Clear, explicit
**Cons**: URL pollution

### Header Versioning

**Example**:
```
GET /users
Headers: Accept: application/vnd.api.v1+json
```

**Pros**: Clean URLs
**Cons**: Less discoverable

### Query Parameter Versioning

**Example**:
```
GET /users?version=1
```

**Pros**: Simple
**Cons**: Not RESTful, can be ignored

**Recommendation**: **URL Versioning** (most common, clearest)

---

## Authentication and Authorization

### Authentication Methods

1. **API Keys**:
   ```
   Authorization: ApiKey <key>
   ```

2. **Bearer Tokens** (JWT):
   ```
   Authorization: Bearer <token>
   ```

3. **OAuth 2.0**:
   ```
   Authorization: Bearer <access_token>
   ```

### Authorization

- **RBAC**: Role-Based Access Control
- **ABAC**: Attribute-Based Access Control
- **Scopes**: Fine-grained permissions

---

## Error Handling

### Consistent Error Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": {...},
    "requestId": "req-123",
    "timestamp": "2025-12-15T10:00:00Z"
  }
}
```

### Error Codes

- **VALIDATION_ERROR**: Invalid input
- **NOT_FOUND**: Resource not found
- **UNAUTHORIZED**: Authentication required
- **FORBIDDEN**: Not authorized
- **RATE_LIMIT_EXCEEDED**: Too many requests
- **INTERNAL_ERROR**: Server error

---

## Rate Limiting

### Headers

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640000000
```

### Response (429 Too Many Requests)

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "retryAfter": 60
  }
}
```

---

## API Documentation

### OpenAPI/Swagger

**Benefits**:
- Machine-readable specification
- Interactive documentation
- Code generation
- Testing tools

### Documentation Should Include

1. **Endpoints**: All available endpoints
2. **Request/Response**: Examples
3. **Authentication**: How to authenticate
4. **Error Codes**: All possible errors
5. **Rate Limits**: Rate limiting information
6. **Changelog**: API version changes

---

## Best Practices

### 1. Use HTTPS

Always use HTTPS for APIs to encrypt data in transit.

### 2. Use JSON

Use JSON as the primary data format (unless specific use case requires otherwise).

### 3. Use Consistent Naming

- Use `camelCase` for JSON fields
- Use `kebab-case` for URLs
- Use `UPPER_SNAKE_CASE` for constants

### 4. Return Appropriate Status Codes

Use correct HTTP status codes for different scenarios.

### 5. Provide Meaningful Error Messages

Error messages should be helpful for debugging and user-friendly.

### 6. Support CORS

If serving web clients, configure CORS properly.

### 7. Implement Rate Limiting

Protect your API from abuse with rate limiting.

### 8. Use Pagination

Always paginate large collections.

### 9. Version Your API

Version your API to support backward compatibility.

### 10. Document Everything

Comprehensive documentation improves developer experience.

---

## Performance Optimization

### 1. Use Compression

Enable gzip compression for responses.

### 2. Implement Caching

Use HTTP caching headers (`Cache-Control`, `ETag`).

### 3. Use Field Selection

Allow clients to request specific fields:
```
GET /users?fields=id,name,email
```

### 4. Batch Operations

Support batch operations when appropriate:
```
POST /users/batch
Body: [{...}, {...}]
```

### 5. Use Async Operations

For long-running operations, return 202 Accepted and provide status endpoint.

---

## Security Best Practices

### 1. Validate Input

Always validate and sanitize input data.

### 2. Use Parameterized Queries

Prevent SQL injection with parameterized queries.

### 3. Implement Rate Limiting

Protect against DDoS and abuse.

### 4. Use HTTPS

Encrypt all communications.

### 5. Implement Authentication

Require authentication for sensitive endpoints.

### 6. Use Least Privilege

Grant minimum necessary permissions.

### 7. Log Security Events

Log authentication failures, rate limit hits, etc.

### 8. Regular Security Audits

Regularly audit APIs for security vulnerabilities.

---

## Key Takeaways

- **REST Principles**: Follow REST principles for consistency
- **Resource Naming**: Use nouns, plural, hierarchical structure
- **HTTP Methods**: Use appropriate HTTP methods
- **Status Codes**: Return correct HTTP status codes
- **Consistent Structure**: Use consistent request/response structure
- **Versioning**: Version your API for backward compatibility
- **Documentation**: Comprehensive documentation is essential
- **Security**: Implement security best practices
- **Performance**: Optimize for performance
- **Error Handling**: Consistent error handling improves UX

---

## Related Topics

- **[API Gateway]({{< ref "../system-components/4-API_Gateway.md" >}})** - API Gateway patterns and features
- **[Rate Limiter]({{< ref "../system-design-case-studies/3-Rate_Limiter.md" >}})** - Rate limiting implementation
- **[Security & Authentication]({{< ref "../system-components/12-Security_Authentication.md" >}})** - API security and authentication

