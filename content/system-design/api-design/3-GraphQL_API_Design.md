+++
title = "GraphQL API Design"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 3
description = "Design GraphQL APIs: schema design, queries, mutations, subscriptions, resolvers, and best practices for building scalable GraphQL APIs."
+++

---

## Introduction

GraphQL is a query language and runtime for APIs that allows clients to request exactly the data they need. GraphQL provides a flexible, efficient alternative to REST APIs.

---

## What is GraphQL?

**GraphQL** is:
- **Query Language**: Clients specify what data they need
- **Type System**: Strongly typed schema
- **Runtime**: Executes queries against your data

**Key Benefits**:
- **Single Endpoint**: One endpoint for all operations
- **Client-Driven**: Clients request only needed fields
- **Strongly Typed**: Type-safe API
- **Introspection**: Self-documenting API

---

## GraphQL vs REST

### REST Limitations

**Multiple Requests**:
```
GET /users/123
GET /users/123/orders
GET /users/123/profile
```

**Over-fetching**:
```
GET /users/123
Response: { id, name, email, address, phone, ... }  // Only need name
```

**Under-fetching**:
```
GET /users/123
Response: { id, name }  // Need more data, make another request
```

### GraphQL Advantages

**Single Request**:
```graphql
query {
  user(id: "123") {
    name
    orders {
      id
      total
    }
    profile {
      bio
    }
  }
}
```

**Exact Data Needed**:
- Request only required fields
- No over-fetching
- No under-fetching

---

## Schema Design

### Type System

**Scalar Types**:
- `String`: Text
- `Int`: Integer
- `Float`: Decimal
- `Boolean`: True/false
- `ID`: Unique identifier

**Object Types**:
```graphql
type User {
  id: ID!
  name: String!
  email: String!
  age: Int
  orders: [Order!]!
}
```

**Type Modifiers**:
- `!`: Non-nullable (required)
- `[Type]`: List/array
- `[Type!]!`: Non-nullable list of non-nullable items

### Schema Example

```graphql
type Query {
  user(id: ID!): User
  users(limit: Int, offset: Int): [User!]!
  order(id: ID!): Order
}

type Mutation {
  createUser(input: CreateUserInput!): User!
  updateUser(id: ID!, input: UpdateUserInput!): User!
  deleteUser(id: ID!): Boolean!
}

type User {
  id: ID!
  name: String!
  email: String!
  orders: [Order!]!
  createdAt: String!
}

type Order {
  id: ID!
  userId: ID!
  user: User!
  items: [OrderItem!]!
  total: Float!
  status: OrderStatus!
  createdAt: String!
}

enum OrderStatus {
  PENDING
  PROCESSING
  SHIPPED
  DELIVERED
  CANCELLED
}

input CreateUserInput {
  name: String!
  email: String!
  password: String!
}

input UpdateUserInput {
  name: String
  email: String
}
```

---

## Queries

### Basic Query

```graphql
query {
  user(id: "123") {
    id
    name
    email
  }
}
```

### Query with Variables

```graphql
query GetUser($userId: ID!) {
  user(id: $userId) {
    id
    name
    email
  }
}

Variables:
{
  "userId": "123"
}
```

### Nested Queries

```graphql
query {
  user(id: "123") {
    id
    name
    orders {
      id
      total
      items {
        product {
          name
          price
        }
        quantity
      }
    }
  }
}
```

### Query Aliases

```graphql
query {
  user1: user(id: "123") {
    name
  }
  user2: user(id: "456") {
    name
  }
}
```

### Fragments

```graphql
fragment UserFields on User {
  id
  name
  email
}

query {
  user(id: "123") {
    ...UserFields
    orders {
      id
    }
  }
}
```

---

## Mutations

### Create Mutation

```graphql
mutation {
  createUser(input: {
    name: "John Doe"
    email: "john@example.com"
    password: "secret123"
  }) {
    id
    name
    email
  }
}
```

### Update Mutation

```graphql
mutation {
  updateUser(
    id: "123"
    input: {
      name: "John Smith"
      email: "johnsmith@example.com"
    }
  ) {
    id
    name
    email
  }
}
```

### Delete Mutation

```graphql
mutation {
  deleteUser(id: "123")
}
```

### Multiple Mutations

```graphql
mutation {
  createUser(input: {...}) {
    id
  }
  createOrder(input: {...}) {
    id
  }
}
```

---

## Subscriptions

### Real-Time Updates

```graphql
subscription {
  orderStatusChanged(orderId: "123") {
    id
    status
    updatedAt
  }
}
```

**Use Cases**:
- Real-time notifications
- Live updates
- Chat messages
- Status changes

---

## Resolvers

### Resolver Functions

**Query Resolver**:
```javascript
const resolvers = {
  Query: {
    user: async (parent, args, context) => {
      return await db.user.findById(args.id);
    },
    users: async (parent, args, context) => {
      return await db.user.findAll({
        limit: args.limit,
        offset: args.offset
      });
    }
  }
};
```

**Field Resolver**:
```javascript
const resolvers = {
  User: {
    orders: async (parent, args, context) => {
      return await db.order.findByUserId(parent.id);
    }
  }
};
```

### N+1 Problem

**Problem**: Multiple database queries for nested fields

**Solution**: DataLoader (batching and caching)

```javascript
const DataLoader = require('dataloader');

const userLoader = new DataLoader(async (userIds) => {
  const users = await db.user.findByIds(userIds);
  return userIds.map(id => users.find(u => u.id === id));
});

const resolvers = {
  Order: {
    user: async (parent) => {
      return await userLoader.load(parent.userId);
    }
  }
};
```

---

## Best Practices

### 1. Use Input Types

**Good**:
```graphql
mutation {
  createUser(input: CreateUserInput!) {
    id
  }
}
```

**Bad**:
```graphql
mutation {
  createUser(name: String!, email: String!) {
    id
  }
}
```

### 2. Use Enums for Fixed Values

**Good**:
```graphql
enum OrderStatus {
  PENDING
  PROCESSING
  SHIPPED
}
```

**Bad**:
```graphql
status: String  # "pending", "processing", "shipped"
```

### 3. Use Non-Null Sparingly

**Good**:
```graphql
type User {
  id: ID!
  name: String!
  email: String!
  bio: String  # Optional
}
```

**Bad**:
```graphql
type User {
  id: ID!
  name: String!
  email: String!
  bio: String!  # Should be optional
}
```

### 4. Paginate Lists

**Cursor-Based Pagination**:
```graphql
type Query {
  users(first: Int, after: String): UserConnection!
}

type UserConnection {
  edges: [UserEdge!]!
  pageInfo: PageInfo!
}

type UserEdge {
  node: User!
  cursor: String!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}
```

### 5. Use Field-Level Deprecation

```graphql
type User {
  id: ID!
  name: String!
  fullName: String! @deprecated(reason: "Use name instead")
}
```

### 6. Implement Rate Limiting

**Per-Query Complexity**:
- Limit query depth
- Limit query complexity
- Timeout long queries

**Per-Client Rate Limiting**:
- Limit requests per client
- Limit query cost

### 7. Handle Errors Properly

**GraphQL Errors**:
```json
{
  "errors": [
    {
      "message": "User not found",
      "path": ["user"],
      "extensions": {
        "code": "NOT_FOUND",
        "userId": "123"
      }
    }
  ],
  "data": {
    "user": null
  }
}
```

---

## Security

### Authentication

**Context-Based**:
```javascript
const resolvers = {
  Query: {
    user: async (parent, args, context) => {
      if (!context.user) {
        throw new Error('Unauthorized');
      }
      return await db.user.findById(args.id);
    }
  }
};
```

### Authorization

**Field-Level Authorization**:
```javascript
const resolvers = {
  User: {
    email: async (parent, args, context) => {
      if (context.user.id !== parent.id && !context.user.isAdmin) {
        return null; // Hide email
      }
      return parent.email;
    }
  }
};
```

### Query Depth Limiting

**Prevent Deep Queries**:
```javascript
const depthLimit = require('graphql-depth-limit');

app.use('/graphql', graphqlHTTP({
  validationRules: [depthLimit(5)]
}));
```

### Query Complexity Analysis

**Limit Complex Queries**:
```javascript
const { createComplexityLimitRule } = require('graphql-query-complexity');

const rule = createComplexityLimitRule(1000);
```

---

## Performance Optimization

### 1. Use DataLoader

**Batch and Cache**:
- Batch database queries
- Cache results
- Reduce N+1 queries

### 2. Implement Caching

**Response Caching**:
- Cache query results
- Invalidate on mutations
- Use CDN for public queries

### 3. Optimize Resolvers

**Efficient Queries**:
- Use database indexes
- Avoid N+1 queries
- Use joins when appropriate

### 4. Use Persisted Queries

**Pre-register Queries**:
- Reduce query size
- Improve security
- Better caching

---

## Schema Evolution

### Backward Compatibility

**Add Fields**:
```graphql
type User {
  id: ID!
  name: String!
  email: String!
  phone: String  # New optional field
}
```

**Deprecate Fields**:
```graphql
type User {
  id: ID!
  name: String!
  fullName: String! @deprecated(reason: "Use name")
}
```

**Versioning**:
- Use schema versioning
- Support multiple versions
- Gradual migration

---

## Testing

### Query Testing

```graphql
query {
  user(id: "123") {
    id
    name
  }
}
```

### Mutation Testing

```graphql
mutation {
  createUser(input: {
    name: "Test User"
    email: "test@example.com"
  }) {
    id
    name
  }
}
```

### Integration Testing

- Test resolvers
- Test error handling
- Test authentication/authorization

---

## Tools and Libraries

### GraphQL Servers

- **Apollo Server**: Popular GraphQL server
- **GraphQL Yoga**: Easy-to-use server
- **Express GraphQL**: Express middleware

### GraphQL Clients

- **Apollo Client**: React client
- **Relay**: Facebook's GraphQL client
- **urql**: Lightweight client

### Development Tools

- **GraphQL Playground**: Interactive IDE
- **GraphiQL**: GraphQL IDE
- **Apollo Studio**: GraphQL platform

---

## Key Takeaways

- **Schema-First**: Design schema carefully
- **Type Safety**: Use strong typing
- **Resolvers**: Implement efficient resolvers
- **N+1 Problem**: Use DataLoader
- **Pagination**: Always paginate lists
- **Security**: Implement authentication/authorization
- **Performance**: Optimize with caching and batching
- **Error Handling**: Consistent error format
- **Testing**: Test queries and mutations

---

## Related Topics

- **[API Design Guidelines]({{< ref "1-API_Design_Guidelines.md" >}})** - Core API design principles
- **[REST API Best Practices]({{< ref "2-REST_API_Best_Practices.md" >}})** - REST API practices
- **[API Gateway]({{< ref "../system-components/4-API_Gateway.md" >}})** - API Gateway for GraphQL

