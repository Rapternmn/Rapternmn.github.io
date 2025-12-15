+++
title = "Load Balancing"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 2
description = "Comprehensive guide to load balancing: algorithms, types, architectures, health checks, and real-world implementations."
+++

---

## Introduction

Load balancing is the process of distributing incoming network traffic across multiple servers to ensure no single server becomes overwhelmed. It's a critical component for achieving high availability, scalability, and performance in distributed systems.

---

## What is Load Balancing?

Load balancing distributes client requests across multiple backend servers (also called server pools or server farms). This ensures:

- **High Availability**: If one server fails, traffic routes to healthy servers
- **Scalability**: Handle more traffic by adding servers
- **Performance**: Distribute load to prevent bottlenecks
- **Reliability**: Redundancy prevents single points of failure

---

## Why Load Balancing is Needed

### Problems Without Load Balancing

1. **Single Point of Failure**: One server handles all traffic
2. **Limited Scalability**: Can't handle traffic beyond server capacity
3. **Poor Performance**: Overloaded servers cause slow responses
4. **No Redundancy**: Server failure causes complete outage

### Benefits of Load Balancing

1. **Fault Tolerance**: Automatic failover to healthy servers
2. **Scalability**: Add servers to handle increased load
3. **Performance**: Optimal resource utilization
4. **Flexibility**: Easy to add/remove servers

---

## Load Balancing Algorithms

### 1. Round Robin

**How it works**: Distributes requests sequentially to each server in rotation.

**Example**: Server A → Server B → Server C → Server A (repeat)

**Advantages**:
- Simple to implement
- Even distribution over time
- No server state needed

**Disadvantages**:
- Doesn't consider server load or capacity
- Uneven if servers have different capabilities

**Use Cases**: Servers with similar capacity and load

---

### 2. Least Connections

**How it works**: Routes requests to the server with the fewest active connections.

**Advantages**:
- Considers current server load
- Better for long-lived connections
- Adapts to server capacity

**Disadvantages**:
- Requires tracking connection counts
- May not account for connection duration

**Use Cases**: Long-lived connections (WebSocket, database connections)

---

### 3. Weighted Round Robin

**How it works**: Round robin but assigns weights to servers based on capacity.

**Example**: Server A (weight 3) gets 3 requests, Server B (weight 1) gets 1 request

**Advantages**:
- Accounts for server capacity differences
- Flexible weight assignment
- Better resource utilization

**Disadvantages**:
- Requires manual weight configuration
- Weights may need adjustment as servers change

**Use Cases**: Servers with different capacities

---

### 4. IP Hash / Consistent Hashing

**How it works**: Uses hash of client IP to determine server. Same IP always goes to same server.

**Advantages**:
- Session persistence (sticky sessions)
- Predictable routing
- Good for caching

**Disadvantages**:
- Uneven distribution if IPs are clustered
- Doesn't adapt to server load

**Use Cases**: When session affinity is needed

---

### 5. Least Response Time

**How it works**: Routes to server with lowest response time and fewest active connections.

**Advantages**:
- Considers both latency and load
- Optimal performance
- Adapts to server conditions

**Disadvantages**:
- More complex to implement
- Requires response time monitoring

**Use Cases**: Performance-critical applications

---

## Load Balancer Types

### Layer 4 (Transport Layer) Load Balancing

**OSI Layer**: Transport Layer (TCP/UDP)

**How it works**: Makes routing decisions based on IP address and port number.

**Characteristics**:
- Faster (less processing)
- Lower latency
- Doesn't inspect application data
- Works with any protocol (TCP/UDP)

**Use Cases**: 
- High-throughput applications
- When application inspection isn't needed
- Simple routing requirements

**Examples**: AWS Network Load Balancer (NLB), HAProxy in TCP mode

---

### Layer 7 (Application Layer) Load Balancing

**OSI Layer**: Application Layer (HTTP/HTTPS)

**How it works**: Makes routing decisions based on HTTP headers, URLs, cookies, etc.

**Characteristics**:
- More intelligent routing
- Can inspect request content
- SSL/TLS termination
- Content-based routing

**Features**:
- URL-based routing
- Cookie-based session affinity
- Header-based routing
- Request/response manipulation

**Use Cases**:
- HTTP/HTTPS applications
- When content-based routing is needed
- Microservices with different endpoints

**Examples**: AWS Application Load Balancer (ALB), NGINX, F5

---

## Load Balancer Architectures

### 1. Hardware Load Balancers

**Definition**: Dedicated physical appliances for load balancing.

**Advantages**:
- High performance
- Dedicated resources
- Often includes security features

**Disadvantages**:
- Expensive
- Less flexible
- Requires physical maintenance
- Vendor lock-in

**Examples**: F5 BIG-IP, Citrix ADC

---

### 2. Software Load Balancers

**Definition**: Load balancing software running on standard servers.

**Advantages**:
- Cost-effective
- Flexible and configurable
- Can run on commodity hardware
- Open source options available

**Disadvantages**:
- Shares resources with other processes
- May require more configuration

**Examples**: NGINX, HAProxy, Apache HTTP Server

---

### 3. Cloud Load Balancers

**Definition**: Managed load balancing services from cloud providers.

**Advantages**:
- Fully managed (no maintenance)
- Auto-scaling
- Integrated with cloud services
- Pay-as-you-go pricing

**Disadvantages**:
- Vendor lock-in
- Less control over configuration
- Potential cost at scale

**Examples**: 
- **AWS**: Application Load Balancer (ALB), Network Load Balancer (NLB), Classic Load Balancer
- **GCP**: Cloud Load Balancing
- **Azure**: Azure Load Balancer, Application Gateway

---

## Health Checks

### What are Health Checks?

Health checks monitor server availability and automatically remove unhealthy servers from the pool.

### Health Check Types

1. **Active Health Checks**: Load balancer periodically checks server health
2. **Passive Health Checks**: Monitor actual request responses

### Health Check Parameters

- **Interval**: How often to check (e.g., every 30 seconds)
- **Timeout**: Maximum time to wait for response
- **Threshold**: Number of failures before marking unhealthy
- **Path**: Endpoint to check (e.g., `/health`)

### Health Check Endpoints

Common health check endpoints:
- `/health` - Basic health check
- `/health/ready` - Readiness probe (Kubernetes)
- `/health/live` - Liveness probe (Kubernetes)

---

## Session Persistence (Sticky Sessions)

### What are Sticky Sessions?

Session persistence ensures a client's requests always go to the same server.

### Why Use Sticky Sessions?

- **Session State**: Server stores session data
- **Caching**: Server caches user-specific data
- **Stateful Applications**: Applications that maintain state

### Implementation Methods

1. **Cookie-Based**: Load balancer sets cookie with server identifier
2. **IP Hash**: Hash client IP to determine server
3. **URL Rewriting**: Embed server ID in URL

### Trade-offs

**Advantages**:
- Maintains session state
- Better caching efficiency

**Disadvantages**:
- Uneven load distribution
- Server failure loses sessions
- Harder to scale

---

## Geographic Load Balancing

### What is Geographic Load Balancing?

Routes traffic to servers based on geographic location of the client.

### Benefits

- **Reduced Latency**: Route to nearest data center
- **Disaster Recovery**: Failover to different regions
- **Compliance**: Route to region-specific servers

### Implementation

- **DNS-Based**: Different DNS responses based on location
- **Anycast**: Same IP address in multiple locations
- **Global Load Balancer**: Routes to regional load balancers

---

## Load Balancer Placement

### 1. Client-Side Load Balancing

**Definition**: Client chooses which server to connect to.

**Advantages**:
- No single point of failure
- Direct connection to server

**Disadvantages**:
- Client complexity
- No centralized control

**Use Cases**: Mobile apps, client libraries

---

### 2. Server-Side Load Balancing

**Definition**: Load balancer sits between clients and servers.

**Advantages**:
- Centralized control
- Transparent to clients
- Can add features (SSL termination, caching)

**Disadvantages**:
- Single point of failure (mitigated with redundancy)
- Additional network hop

**Use Cases**: Web applications, APIs

---

### 3. DNS Load Balancing

**Definition**: DNS returns different IP addresses for load balancing.

**Advantages**:
- Simple to implement
- No additional infrastructure

**Disadvantages**:
- DNS caching can cause uneven distribution
- Limited control
- Slow failover (DNS TTL)

**Use Cases**: Simple load distribution, failover

---

## Real-World Examples

### AWS Application Load Balancer (ALB)

**Type**: Layer 7 load balancer

**Features**:
- Path-based routing
- Host-based routing
- Container support (ECS, EKS)
- WebSocket support
- SSL/TLS termination

**Use Cases**: Microservices, containerized applications

---

### AWS Network Load Balancer (NLB)

**Type**: Layer 4 load balancer

**Features**:
- Ultra-low latency
- High throughput
- Static IP addresses
- Preserves source IP

**Use Cases**: High-performance applications, TCP/UDP traffic

---

### NGINX

**Type**: Software load balancer (Layer 4 and Layer 7)

**Features**:
- Open source
- Highly configurable
- Reverse proxy
- SSL termination
- Caching

**Use Cases**: Self-hosted applications, high customization needs

---

### HAProxy

**Type**: Software load balancer (Layer 4 and Layer 7)

**Features**:
- High performance
- Advanced health checks
- Detailed statistics
- ACL-based routing

**Use Cases**: High-performance requirements, complex routing

---

## Use Cases

### 1. Web Server Load Balancing

Distribute HTTP/HTTPS traffic across multiple web servers.

**Components**: Layer 7 load balancer, health checks, session persistence (if needed)

---

### 2. Database Load Balancing

Distribute read queries across database replicas.

**Components**: Layer 4 load balancer, read replicas, health checks

**Note**: Write operations typically go to primary database

---

### 3. Microservices Load Balancing

Route API requests to appropriate microservice instances.

**Components**: API Gateway + Load Balancer, service discovery integration

---

### 4. API Load Balancing

Distribute API requests across API server instances.

**Components**: Layer 7 load balancer, rate limiting, authentication

---

## Best Practices

1. **Health Checks**: Always configure health checks
2. **Redundancy**: Use multiple load balancers (active-active or active-passive)
3. **Monitoring**: Monitor load balancer metrics and server health
4. **SSL/TLS**: Terminate SSL at load balancer for better performance
5. **Session Affinity**: Use only when necessary (stateful applications)
6. **Auto-scaling**: Integrate with auto-scaling for dynamic capacity
7. **Logging**: Enable access logs for debugging and analytics

---

## Key Takeaways

- **Load balancing** distributes traffic to prevent overload and ensure availability
- **Algorithms** vary by use case (round robin, least connections, weighted, etc.)
- **Layer 4** is faster, **Layer 7** is more intelligent
- **Health checks** are essential for automatic failover
- **Session persistence** should be used only when necessary
- **Cloud load balancers** offer managed, scalable solutions

---

## Related Topics

- **[Caching Strategies]({{< ref "3-Caching_Strategies.md" >}})** - Reduce load with caching
- **[API Gateway]({{< ref "4-API_Gateway.md" >}})** - API Gateway often includes load balancing
- **[Availability & Reliability]({{< ref "10-Availability_Reliability.md" >}})** - High availability patterns

