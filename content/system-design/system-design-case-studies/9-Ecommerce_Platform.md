+++
title = "E-commerce Platform (Amazon/eBay)"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 9
description = "Design an e-commerce platform like Amazon or eBay. Covers product catalog, shopping cart, payment processing, order management, inventory, and scaling to millions of products."
+++

---

## Problem Statement

Design an e-commerce platform that allows users to browse products, add to cart, checkout, and process payments. The system should handle product catalog, inventory management, order processing, and payment integration.

**Examples**: Amazon, eBay, Shopify, Walmart

---

## Requirements Clarification

### Functional Requirements

1. **Product Catalog**: Browse and search products
2. **Shopping Cart**: Add/remove items from cart
3. **Checkout**: Process orders
4. **Payment Processing**: Handle payments (credit card, PayPal, etc.)
5. **Order Management**: Track order status
6. **Inventory Management**: Track product inventory
7. **User Accounts**: User registration, authentication
8. **Reviews & Ratings**: Product reviews and ratings
9. **Recommendations**: Product recommendations

### Non-Functional Requirements

- **Scale**: 
  - 100M users
  - 10M daily active users
  - 100M products
  - 10M orders/day
  - Peak: 100K orders/hour
- **Latency**: < 200ms for product pages, < 2 seconds for checkout
- **Availability**: 99.9% uptime
- **Consistency**: Strong consistency for inventory, eventual for recommendations

---

## Capacity Estimation

### Traffic Estimates

- **Product Views**: 10M DAU × 20 views/day = 200M views/day
- **Views/sec**: ~2,300 views/second
- **Orders**: 10M orders/day = ~116 orders/second
- **Peak orders**: 100K/hour = ~28 orders/second
- **Searches**: 10M DAU × 10 searches/day = 100M searches/day

### Storage Estimates

- **Products**: 100M products × 10 KB = 1 TB
- **Orders**: 10M orders/day × 5 KB = 50 GB/day
- **Yearly Orders**: ~18 TB/year
- **User Data**: 100M users × 1 KB = 100 GB
- **Images**: Assume 5 images per product × 500 KB = 250 TB

---

## API Design

### REST APIs

```
GET /api/v1/products
Query: ?category=electronics&page=1&limit=20
Response: {
  "products": [...],
  "total": 10000,
  "page": 1
}

GET /api/v1/products/{productId}
Response: {
  "productId": "prod123",
  "name": "Product Name",
  "price": 99.99,
  "inventory": 100,
  "images": [...],
  "reviews": [...]
}

POST /api/v1/cart
Request: {
  "productId": "prod123",
  "quantity": 2
}

POST /api/v1/orders
Request: {
  "items": [
    {"productId": "prod123", "quantity": 2}
  ],
  "shippingAddress": {...},
  "paymentMethod": "credit_card"
}
Response: {
  "orderId": "order456",
  "status": "pending",
  "total": 199.98
}

GET /api/v1/orders/{orderId}
Response: {
  "orderId": "order456",
  "status": "shipped",
  "items": [...],
  "trackingNumber": "TRACK123"
}
```

---

## Database Design

### Schema

**Products Table** (PostgreSQL):
```
productId (PK): UUID
name: VARCHAR
description: TEXT
price: DECIMAL
categoryId: UUID
inventory: INT
status: VARCHAR (active, inactive, out_of_stock)
createdAt: TIMESTAMP
```

**Categories Table** (PostgreSQL):
```
categoryId (PK): UUID
name: VARCHAR
parentCategoryId: UUID (nullable)
```

**Orders Table** (PostgreSQL):
```
orderId (PK): UUID
userId: UUID
status: VARCHAR (pending, paid, shipped, delivered, cancelled)
total: DECIMAL
shippingAddress: JSON
paymentMethod: VARCHAR
createdAt: TIMESTAMP
```

**Order Items Table** (PostgreSQL):
```
orderId (PK): UUID
productId (PK): UUID
quantity: INT
price: DECIMAL
```

**Shopping Cart Table** (Redis):
```
userId (PK): UUID
items: JSON [
  {"productId": "prod123", "quantity": 2}
]
```

**Inventory Table** (PostgreSQL):
```
productId (PK): UUID
quantity: INT
reserved: INT
available: INT (computed: quantity - reserved)
```

### Database Selection

**Products, Orders**: **PostgreSQL** (relational data, ACID)
**Shopping Cart**: **Redis** (ephemeral, fast)
**Product Search**: **Elasticsearch** (full-text search)
**Product Images**: **Object Storage** (S3)

---

## High-Level Design

### Architecture

![Ecommerce Architecture](/images/system-design/ecommerce-architecture.png)

### Components

1. **Product Service**: Product catalog, search
2. **Cart Service**: Shopping cart management
3. **Order Service**: Order processing
4. **Payment Service**: Payment processing
5. **Inventory Service**: Inventory management
6. **Search Service**: Product search (Elasticsearch)
7. **Recommendation Service**: Product recommendations
8. **Notification Service**: Order updates, emails

---

## Detailed Design

### Product Catalog

**Product Storage**:
- **PostgreSQL**: Product metadata
- **Elasticsearch**: Product search index
- **Object Storage**: Product images

**Product Search**:
- **Full-Text Search**: Search by name, description
- **Filters**: Category, price range, ratings
- **Sorting**: Price, rating, popularity

**Caching**:
- **CDN**: Cache product images
- **Redis**: Cache popular products
- **Cache Invalidation**: Invalidate on product update

---

### Shopping Cart

**Storage**: **Redis** (ephemeral, fast)

**Cart Operations**:
- **Add Item**: Add product to cart
- **Update Quantity**: Update item quantity
- **Remove Item**: Remove product from cart
- **Clear Cart**: Empty cart

**Cart Expiration**: TTL in Redis (e.g., 30 days)

**Cart Sync**: Sync cart across devices (same user)

---

### Order Processing Flow

1. **User** → Order Service: Create order
2. **Order Service**:
   - Validate cart items
   - Check inventory (reserve items)
   - Calculate total
   - Create order (status: "pending")
3. **Payment Service**: Process payment
   - If success: Update order status to "paid"
   - If failure: Release inventory, cancel order
4. **Order Service**: 
   - Update inventory (deduct reserved items)
   - Notify fulfillment service
   - Send confirmation email
5. **Fulfillment**: Ship order, update status

---

### Inventory Management

**Inventory Challenges**:
- **Race Conditions**: Multiple users buying same product
- **Overselling**: Selling more than available
- **Reservation**: Reserve items during checkout

**Solutions**:
1. **Database Locks**: Use row-level locks
2. **Optimistic Locking**: Version-based locking
3. **Reservation**: Reserve items during checkout, release if payment fails

**Inventory Operations**:
- **Reserve**: Reserve items (quantity - reserved)
- **Commit**: Deduct reserved items (on payment success)
- **Release**: Release reserved items (on payment failure/timeout)

---

### Payment Processing

**Payment Flow**:
1. **Order Service** → Payment Gateway: Initiate payment
2. **Payment Gateway**: Process payment (credit card, PayPal, etc.)
3. **Payment Gateway** → Order Service: Payment result (webhook)
4. **Order Service**: Update order status

**Payment Gateways**:
- **Stripe**: Credit cards
- **PayPal**: PayPal payments
- **Square**: Point of sale

**Security**:
- **PCI Compliance**: Don't store credit card details
- **Tokenization**: Use payment tokens
- **Encryption**: Encrypt sensitive data

---

### Order Status Tracking

**Order States**:
1. **Pending**: Order created, payment pending
2. **Paid**: Payment successful
3. **Processing**: Order being prepared
4. **Shipped**: Order shipped
5. **Delivered**: Order delivered
6. **Cancelled**: Order cancelled

**Status Updates**:
- **Automatic**: System updates (payment, shipping)
- **Manual**: Admin updates
- **Notifications**: Notify user on status change

---

## Scalability

### Horizontal Scaling

- **Stateless Services**: All services scale horizontally
- **Database Sharding**: Shard by userId or productId
- **Read Replicas**: Database read replicas for reads

### Read Scaling

- **Caching**: Cache products, popular items
- **CDN**: Serve product images
- **Read Replicas**: Database read replicas

### Write Scaling

- **Database Sharding**: Distribute writes
- **Async Processing**: Process orders asynchronously
- **Message Queue**: Queue order processing

---

## Reliability

### High Availability

- **Multiple Service Instances**: No single point of failure
- **Database Replication**: Master-slave replication
- **Payment Gateway Redundancy**: Multiple payment providers

### Data Consistency

- **Inventory**: Strong consistency (prevent overselling)
- **Orders**: Strong consistency (critical)
- **Recommendations**: Eventual consistency acceptable

### Fault Tolerance

- **Payment Failures**: Retry payment, handle gracefully
- **Inventory Failures**: Fallback to database, prevent overselling
- **Service Failures**: Queue operations, process when service recovers

---

## Trade-offs

### Consistency vs Availability

- **Inventory**: Strong consistency (CP) - prevent overselling
- **Product Catalog**: Eventual consistency acceptable (AP)
- **Recommendations**: Eventual consistency acceptable (AP)

### Latency vs Consistency

- **Product Pages**: Optimize for latency (caching)
- **Checkout**: Optimize for consistency (strong consistency)

---

## Extensions

### Additional Features

1. **Wishlist**: Save products for later
2. **Product Recommendations**: ML-based recommendations
3. **Price Tracking**: Track price changes
4. **Product Reviews**: User reviews and ratings
5. **Loyalty Program**: Rewards and points
6. **Multi-Vendor**: Support multiple sellers
7. **International**: Multi-currency, multi-language

---

## Key Takeaways

- **Microservices**: Separate services for products, cart, orders, payments
- **Inventory Management**: Prevent overselling with reservations
- **Payment Processing**: Use payment gateways, don't store card details
- **Caching**: Cache products and popular items
- **Database Sharding**: Shard for scalability
- **Async Processing**: Process orders asynchronously for reliability

---

## Related Topics

- **[API Gateway]({{< ref "../system-components/4-API_Gateway.md" >}})** - API routing and management
- **[Caching Strategies]({{< ref "../system-components/3-Caching_Strategies.md" >}})** - Product caching
- **[Databases]({{< ref "../databases/_index.md" >}})** - Database selection
- **[Message Queues & Message Brokers]({{< ref "../system-components/5-Message_Queues_Message_Brokers.md" >}})** - Order processing queue

