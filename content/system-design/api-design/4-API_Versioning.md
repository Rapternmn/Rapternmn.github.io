+++
title = "API Versioning"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 4
description = "API versioning strategies: URL versioning, header versioning, semantic versioning, backward compatibility, deprecation, and migration strategies."
+++

---

## Introduction

API versioning is crucial for maintaining backward compatibility while evolving your API. Proper versioning allows you to introduce changes without breaking existing clients.

---

## Why Version APIs?

### Reasons to Version

1. **Backward Compatibility**: Support existing clients
2. **Breaking Changes**: Introduce breaking changes safely
3. **Gradual Migration**: Allow clients to migrate gradually
4. **Stability**: Provide stable API for production clients
5. **Innovation**: Enable rapid iteration for new features

### When to Version

**Version When**:
- Removing fields
- Changing field types
- Changing required fields
- Changing response structure
- Changing authentication
- Changing error formats

**Don't Version When**:
- Adding new fields (backward compatible)
- Adding new endpoints (backward compatible)
- Fixing bugs (backward compatible)
- Performance improvements (backward compatible)

---

## Versioning Strategies

### 1. URL Versioning

**Format**: `/v{version}/resource`

**Example**:
```
GET /v1/users
GET /v2/users
GET /v1/users/123
GET /v2/users/123
```

**Pros**:
- Clear and explicit
- Easy to understand
- Works with all clients
- Easy to cache
- Easy to route

**Cons**:
- URL pollution
- More endpoints to maintain

**Implementation**:
```javascript
// Express.js example
app.use('/v1', v1Router);
app.use('/v2', v2Router);
```

---

### 2. Header Versioning

**Format**: `Accept: application/vnd.api.v{version}+json`

**Example**:
```
GET /users
Headers: Accept: application/vnd.api.v1+json

GET /users
Headers: Accept: application/vnd.api.v2+json
```

**Pros**:
- Clean URLs
- RESTful approach
- Content negotiation

**Cons**:
- Less discoverable
- Requires header support
- Harder to test
- Caching complexity

**Implementation**:
```javascript
app.get('/users', (req, res) => {
  const version = req.headers['accept'].match(/v(\d+)/)[1];
  if (version === '1') {
    return v1Handler(req, res);
  } else if (version === '2') {
    return v2Handler(req, res);
  }
});
```

---

### 3. Query Parameter Versioning

**Format**: `?version={version}` or `?v={version}`

**Example**:
```
GET /users?version=1
GET /users?version=2
GET /users?v=1
```

**Pros**:
- Simple
- Easy to implement
- Works with all clients

**Cons**:
- Not RESTful
- Can be ignored
- URL pollution
- Caching issues

**Recommendation**: **Not recommended** (not RESTful, can be ignored)

---

### 4. Custom Header Versioning

**Format**: `X-API-Version: {version}`

**Example**:
```
GET /users
Headers: X-API-Version: 1

GET /users
Headers: X-API-Version: 2
```

**Pros**:
- Clean URLs
- Explicit versioning
- Easy to implement

**Cons**:
- Custom header (not standard)
- Less discoverable
- Requires header support

---

## Recommended Approach

### URL Versioning (Recommended)

**Why**:
- Most common and understood
- Clear and explicit
- Works with all clients
- Easy to cache and route
- Easy to test

**Implementation**:
```
/v1/users
/v2/users
/v3/users
```

---

## Semantic Versioning

### Version Format

**Format**: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

**Examples**:
- `v1.0.0`: Initial release
- `v1.1.0`: Added new endpoint (backward compatible)
- `v1.1.1`: Bug fix (backward compatible)
- `v2.0.0`: Breaking changes

### API Versioning

**For APIs, typically use**:
- **Major version only**: `v1`, `v2`, `v3`
- **Major.Minor**: `v1.0`, `v1.1`, `v2.0`

**Recommendation**: **Major version only** (simpler, sufficient)

---

## Backward Compatibility

### What is Backward Compatible?

**Compatible Changes**:
- ✅ Adding new fields
- ✅ Adding new endpoints
- ✅ Making optional fields required (with defaults)
- ✅ Adding new query parameters
- ✅ Adding new response fields

**Incompatible Changes**:
- ❌ Removing fields
- ❌ Changing field types
- ❌ Making required fields optional
- ❌ Changing field names
- ❌ Changing response structure
- ❌ Removing endpoints

### Maintaining Compatibility

**Strategy**:
1. **Add, Don't Remove**: Add new fields, keep old ones
2. **Deprecate First**: Deprecate before removing
3. **Gradual Migration**: Allow time for migration
4. **Documentation**: Document changes clearly

---

## Deprecation Strategy

### Deprecation Process

**Step 1: Announce Deprecation**
```json
{
  "field": "oldField",
  "deprecated": true,
  "deprecationDate": "2025-12-15",
  "removalDate": "2026-06-15",
  "replacement": "newField",
  "migrationGuide": "https://api.example.com/migration"
}
```

**Step 2: Add Warnings**
```
Headers: X-API-Deprecation: oldField will be removed on 2026-06-15
```

**Step 3: Support Both**
- Support old and new fields
- Log usage of deprecated fields
- Notify clients

**Step 4: Remove**
- Remove after deprecation period
- Provide migration guide
- Support migration tools

---

## Version Lifecycle

### Version Stages

1. **Alpha**: Early development, unstable
2. **Beta**: Testing, may have issues
3. **Stable**: Production-ready
4. **Deprecated**: Will be removed
5. **Sunset**: Removed, no longer available

### Version Support Policy

**Support Multiple Versions**:
- Current version: Full support
- Previous version: Security updates only
- Older versions: Deprecated, migration required

**Example**:
- `v3`: Current, full support
- `v2`: Previous, security updates
- `v1`: Deprecated, migration required

---

## Migration Strategies

### Gradual Migration

**Phase 1: Introduce New Version**
- Release new version alongside old
- Support both versions
- Document differences

**Phase 2: Encourage Migration**
- Provide migration guides
- Offer support
- Highlight benefits

**Phase 3: Deprecate Old Version**
- Announce deprecation
- Set removal date
- Provide migration tools

**Phase 4: Remove Old Version**
- Remove after deprecation period
- Provide final migration support

---

## Versioning Best Practices

### 1. Version Early

**Start with v1**:
- Even for initial release
- Sets expectation for versioning
- Easier to add versions later

### 2. Document Changes

**Changelog**:
- List all changes
- Breaking changes highlighted
- Migration guides
- Examples

### 3. Communicate Changes

**Announcements**:
- Email notifications
- Blog posts
- Documentation updates
- In-app notifications

### 4. Provide Migration Tools

**Tools**:
- Migration scripts
- Code examples
- Testing tools
- Support channels

### 5. Set Clear Policies

**Policies**:
- Support duration
- Deprecation timeline
- Removal schedule
- Migration support

---

## Implementation Examples

### Express.js

```javascript
// Version 1
const v1Router = express.Router();
v1Router.get('/users', (req, res) => {
  res.json({ users: [...] });
});

// Version 2
const v2Router = express.Router();
v2Router.get('/users', (req, res) => {
  res.json({ 
    data: { users: [...] },
    meta: { ... }
  });
});

// Mount routers
app.use('/v1', v1Router);
app.use('/v2', v2Router);
```

### Django REST Framework

```python
# urls.py
urlpatterns = [
    path('v1/', include('api.v1.urls')),
    path('v2/', include('api.v2.urls')),
]
```

### Spring Boot

```java
@RestController
@RequestMapping("/v1/users")
public class UserControllerV1 { ... }

@RestController
@RequestMapping("/v2/users")
public class UserControllerV2 { ... }
```

---

## Version Detection

### Default Version

**Strategy**:
- Default to latest stable version
- Or require explicit version
- Document default behavior

**Example**:
```
GET /users  # Defaults to v2
GET /v1/users  # Explicit v1
GET /v2/users  # Explicit v2
```

### Version Negotiation

**Content Negotiation**:
```
GET /users
Headers: 
  Accept: application/json
  X-API-Version: 2
```

---

## Testing Versions

### Test Each Version

**Test Cases**:
- All endpoints for each version
- Backward compatibility
- Migration paths
- Deprecation warnings

### Version-Specific Tests

```javascript
describe('API v1', () => {
  it('should return v1 format', async () => {
    const response = await request(app)
      .get('/v1/users')
      .expect(200);
    expect(response.body).toHaveProperty('users');
  });
});

describe('API v2', () => {
  it('should return v2 format', async () => {
    const response = await request(app)
      .get('/v2/users')
      .expect(200);
    expect(response.body).toHaveProperty('data');
  });
});
```

---

## Common Pitfalls

### 1. Over-Versioning

**Problem**: Creating new version for every change

**Solution**: Only version for breaking changes

### 2. Under-Versioning

**Problem**: Not versioning when needed

**Solution**: Version when making breaking changes

### 3. Inconsistent Versioning

**Problem**: Different versioning across endpoints

**Solution**: Consistent versioning strategy

### 4. Poor Communication

**Problem**: Not communicating changes

**Solution**: Clear communication and documentation

### 5. Short Deprecation Periods

**Problem**: Not enough time for migration

**Solution**: Provide adequate deprecation period (6-12 months)

---

## Key Takeaways

- **URL Versioning**: Recommended approach
- **Semantic Versioning**: Use major versions for APIs
- **Backward Compatibility**: Maintain compatibility when possible
- **Deprecation**: Deprecate before removing
- **Migration**: Provide migration guides and tools
- **Documentation**: Document all changes
- **Communication**: Communicate changes clearly
- **Testing**: Test all versions

---

## Related Topics

- **[API Design Guidelines]({{< ref "1-API_Design_Guidelines.md" >}})** - Core API design principles
- **[REST API Best Practices]({{< ref "2-REST_API_Best_Practices.md" >}})** - REST API practices
- **[API Gateway]({{< ref "../system-components/4-API_Gateway.md" >}})** - API Gateway for versioning

