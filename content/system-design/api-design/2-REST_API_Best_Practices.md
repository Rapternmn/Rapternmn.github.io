+++
title = "REST API Best Practices"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 2
description = "Best practices for designing and implementing REST APIs: naming conventions, request/response patterns, error handling, security, performance, and real-world examples."
+++

---

## Introduction

REST API Best Practices provide actionable guidelines for building production-ready REST APIs. These practices improve API usability, maintainability, security, and performance.

---

## Naming Conventions

### Resource Names

**Use Plural Nouns**:
```
GET /users
GET /orders
GET /products
```

**Use Hyphens for Multi-Word Resources**:
```
GET /user-profiles
GET /order-items
GET /shipping-addresses
```

**Avoid Abbreviations**:
```
✅ GET /users
❌ GET /usr
```

### Field Names

**Use camelCase for JSON**:
```json
{
  "firstName": "John",
  "lastName": "Doe",
  "emailAddress": "john@example.com"
}
```

**Be Consistent**:
```json
{
  "userId": "123",
  "userName": "john",
  "userEmail": "john@example.com"
}
```

---

## URL Structure

### Keep URLs Simple and Intuitive

**Good**:
```
GET /users/123
GET /users/123/orders
GET /users/123/orders/456/items
```

**Bad**:
```
GET /getUserById/123
GET /userOrders?userId=123
GET /users/123/orders/456/getItems
```

### Use Query Parameters for Filtering

**Good**:
```
GET /users?status=active&role=admin
GET /products?category=electronics&price[gte]=100
```

**Bad**:
```
GET /users/active/admin
GET /products/electronics/price-greater-than-100
```

### Avoid Deep Nesting

**Good** (max 2-3 levels):
```
GET /users/123/orders
GET /organizations/789/departments/456/users
```

**Bad** (too deep):
```
GET /users/123/orders/456/items/789/products/101/reviews
```

**Alternative** (flatten):
```
GET /orders/456/items
GET /items/789/product
GET /products/101/reviews
```

---

## HTTP Methods Usage

### GET - Retrieve Resources

**Idempotent and Safe**:
```
GET /users
GET /users/123
GET /users/123/orders
```

**Should Not Modify State**:
```
✅ GET /users/123
❌ GET /users/123/delete
```

### POST - Create Resources

**Create New Resource**:
```
POST /users
Body: { "name": "John", "email": "john@example.com" }

Response: 201 Created
Location: /users/123
```

**Perform Actions** (when no resource created):
```
POST /users/123/activate
POST /orders/456/cancel
```

### PUT - Replace Resource

**Full Update**:
```
PUT /users/123
Body: { "name": "John Doe", "email": "john@example.com", "role": "admin" }

Response: 200 OK or 204 No Content
```

**Idempotent**: Same request produces same result

### PATCH - Partial Update

**Update Specific Fields**:
```
PATCH /users/123
Body: { "email": "newemail@example.com" }

Response: 200 OK
```

**Idempotent**: Should be idempotent

### DELETE - Remove Resource

**Delete Resource**:
```
DELETE /users/123

Response: 204 No Content or 200 OK
```

**Idempotent**: Deleting twice = same result

---

## Request Design

### Request Headers

**Required Headers**:
```
Content-Type: application/json
Accept: application/json
Authorization: Bearer <token>
```

**Optional but Recommended**:
```
X-Request-ID: <unique-id>  // For tracing
X-Client-Version: 1.0.0    // Client version
User-Agent: MyApp/1.0.0    // Client identification
```

### Request Body

**Use JSON**:
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "age": 30
}
```

**Validate Input**:
- Required fields
- Data types
- Format validation (email, URL, etc.)
- Range validation (min/max)

### Query Parameters

**Filtering**:
```
GET /users?status=active&role=admin
GET /products?price[gte]=100&price[lte]=500
```

**Sorting**:
```
GET /users?sort=name&order=asc
GET /users?sort=-createdAt  // descending
```

**Pagination**:
```
GET /users?page=1&limit=20
GET /users?offset=0&limit=20
GET /users?cursor=abc123&limit=20
```

---

## Response Design

### Success Responses

**Single Resource** (200 OK):
```json
{
  "data": {
    "id": "123",
    "name": "John Doe",
    "email": "john@example.com"
  }
}
```

**Collection** (200 OK):
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
  }
}
```

**Created** (201 Created):
```json
{
  "data": {
    "id": "789",
    "name": "New User",
    "email": "new@example.com"
  }
}
Headers: Location: /users/789
```

**No Content** (204 No Content):
```
DELETE /users/123
Response: 204 No Content (no body)
```

### Error Responses

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
    "requestId": "req-123",
    "timestamp": "2025-12-15T10:00:00Z"
  }
}
```

**Common Error Codes**:
- `VALIDATION_ERROR`: Invalid input
- `NOT_FOUND`: Resource not found
- `UNAUTHORIZED`: Authentication required
- `FORBIDDEN`: Not authorized
- `CONFLICT`: Resource conflict
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INTERNAL_ERROR`: Server error

---

## Pagination

### Offset-Based Pagination

**Request**:
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
  },
  "links": {
    "self": "/users?page=1&limit=20",
    "first": "/users?page=1&limit=20",
    "last": "/users?page=5&limit=20",
    "prev": null,
    "next": "/users?page=2&limit=20"
  }
}
```

**Use When**: Direct page access needed, small datasets

### Cursor-Based Pagination

**Request**:
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
    "self": "/users?cursor=abc123&limit=20",
    "next": "/users?cursor=xyz789&limit=20"
  }
}
```

**Use When**: Large datasets, consistent results needed

---

## Filtering and Sorting

### Filtering

**Simple Filters**:
```
GET /users?status=active
GET /users?role=admin&status=active
```

**Range Filters**:
```
GET /products?price[gte]=100&price[lte]=500
GET /users?age[gt]=18&age[lt]=65
```

**Array Filters**:
```
GET /products?tags=electronics,computers
GET /users?roles[]=admin&roles[]=user
```

**Search**:
```
GET /users?search=john
GET /products?q=laptop
```

### Sorting

**Single Field**:
```
GET /users?sort=name&order=asc
GET /users?sort=-createdAt  // descending
```

**Multiple Fields**:
```
GET /users?sort=name,createdAt
GET /users?sort=name,-createdAt
```

---

## Field Selection

### Allow Clients to Request Specific Fields

**Request**:
```
GET /users/123?fields=id,name,email
GET /users?fields=id,name
```

**Response**:
```json
{
  "data": {
    "id": "123",
    "name": "John Doe",
    "email": "john@example.com"
  }
}
```

**Benefits**:
- Reduce payload size
- Improve performance
- Better for mobile clients

---

## Batch Operations

### Batch Create

**Request**:
```
POST /users/batch
Body: {
  "items": [
    { "name": "John", "email": "john@example.com" },
    { "name": "Jane", "email": "jane@example.com" }
  ]
}
```

**Response**:
```json
{
  "data": [
    { "id": "123", "name": "John", ... },
    { "id": "456", "name": "Jane", ... }
  ],
  "meta": {
    "created": 2,
    "failed": 0
  }
}
```

### Batch Update

**Request**:
```
PATCH /users/batch
Body: {
  "items": [
    { "id": "123", "email": "newemail@example.com" },
    { "id": "456", "role": "admin" }
  ]
}
```

---

## Async Operations

### Long-Running Operations

**Request**:
```
POST /reports
Body: { "type": "sales", "dateRange": {...} }

Response: 202 Accepted
{
  "jobId": "job-123",
  "status": "processing",
  "statusUrl": "/jobs/job-123"
}
```

**Status Check**:
```
GET /jobs/job-123

Response: 200 OK
{
  "jobId": "job-123",
  "status": "completed",
  "resultUrl": "/reports/report-456"
}
```

---

## Security Best Practices

### Authentication

**Use Bearer Tokens**:
```
Authorization: Bearer <token>
```

**Use API Keys for Service-to-Service**:
```
Authorization: ApiKey <key>
```

### Authorization

**Check Permissions**:
- Verify user has permission for resource
- Return 403 Forbidden if not authorized
- Don't expose existence of resources (404 vs 403)

### Input Validation

**Validate All Input**:
- Required fields
- Data types
- Format (email, URL, etc.)
- Range (min/max)
- Length limits

**Sanitize Input**:
- Remove dangerous characters
- Escape special characters
- Prevent injection attacks

### Rate Limiting

**Implement Rate Limiting**:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640000000
```

**Return 429 on Exceed**:
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "retryAfter": 60
  }
}
```

### HTTPS

**Always Use HTTPS**:
- Encrypt all communications
- Protect sensitive data
- Prevent man-in-the-middle attacks

---

## Performance Optimization

### Caching

**Use HTTP Caching**:
```
Cache-Control: public, max-age=3600
ETag: "abc123"
Last-Modified: Wed, 15 Dec 2025 10:00:00 GMT
```

**Cache Headers**:
- `Cache-Control`: Control caching behavior
- `ETag`: Entity tag for validation
- `Last-Modified`: Last modification time

### Compression

**Enable Compression**:
```
Content-Encoding: gzip
```

**Compress Large Responses**:
- Reduce bandwidth
- Improve performance
- Better for mobile clients

### Database Optimization

**Use Indexes**:
- Index frequently queried fields
- Composite indexes for multi-field queries

**Limit Results**:
- Always paginate large collections
- Use limit on queries

**Avoid N+1 Queries**:
- Use eager loading
- Batch queries
- Use joins when appropriate

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

### Validation Errors

**Return Field-Level Errors**:
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Validation failed",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format",
        "code": "INVALID_FORMAT"
      },
      {
        "field": "age",
        "message": "Age must be between 18 and 100",
        "code": "OUT_OF_RANGE"
      }
    ]
  }
}
```

### Don't Expose Internal Details

**Bad**:
```json
{
  "error": {
    "message": "SQLException: Connection timeout at line 123..."
  }
}
```

**Good**:
```json
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "An error occurred. Please try again later.",
    "requestId": "req-123"
  }
}
```

---

## API Versioning

### URL Versioning (Recommended)

```
GET /v1/users
GET /v2/users
```

**Pros**: Clear, explicit, easy to understand
**Cons**: URL pollution

### Header Versioning

```
GET /users
Headers: Accept: application/vnd.api.v1+json
```

**Pros**: Clean URLs
**Cons**: Less discoverable

### Best Practice

**Use URL Versioning** for most cases:
- Clear and explicit
- Easy to understand
- Works with all clients

**Support Multiple Versions**:
- Maintain backward compatibility
- Deprecate old versions gradually
- Provide migration guides

---

## Documentation

### OpenAPI/Swagger

**Use OpenAPI Specification**:
- Machine-readable
- Interactive documentation
- Code generation
- Testing tools

### Documentation Should Include

1. **Endpoints**: All endpoints with examples
2. **Request/Response**: Request and response schemas
3. **Authentication**: How to authenticate
4. **Error Codes**: All possible errors
5. **Rate Limits**: Rate limiting information
6. **Changelog**: API version changes
7. **SDKs**: Available SDKs and libraries

---

## Testing

### Test Cases

**Test Scenarios**:
- Happy paths
- Error cases
- Edge cases
- Boundary conditions
- Authentication/authorization
- Rate limiting

### Test Tools

- **Postman**: API testing
- **curl**: Command-line testing
- **Automated Tests**: Unit, integration, E2E tests

---

## Monitoring and Logging

### Logging

**Log Important Events**:
- Request/response (with sanitization)
- Errors and exceptions
- Authentication failures
- Rate limit hits
- Performance metrics

**Include Context**:
- Request ID
- User ID
- Timestamp
- IP address

### Monitoring

**Monitor**:
- Response times
- Error rates
- Request rates
- API availability
- Resource usage

---

## Key Takeaways

- **Consistent Naming**: Use consistent naming conventions
- **Proper HTTP Methods**: Use correct HTTP methods
- **Status Codes**: Return appropriate status codes
- **Error Handling**: Consistent error format
- **Pagination**: Always paginate large collections
- **Security**: Implement authentication, authorization, rate limiting
- **Performance**: Optimize with caching, compression
- **Documentation**: Comprehensive documentation
- **Versioning**: Version your API
- **Testing**: Test thoroughly

---

## Related Topics

- **[API Design Guidelines]({{< ref "1-API_Design_Guidelines.md" >}})** - Core API design principles
- **[API Versioning]({{< ref "4-API_Versioning.md" >}})** - Detailed versioning strategies
- **[API Gateway]({{< ref "../system-components/4-API_Gateway.md" >}})** - API Gateway patterns

