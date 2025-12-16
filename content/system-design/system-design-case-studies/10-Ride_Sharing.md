+++
title = "Ride-Sharing Service (Uber/Lyft)"
date = 2025-12-15T10:00:00+05:30
draft = false
weight = 10
description = "Design a ride-sharing service like Uber or Lyft. Covers real-time matching, geolocation, surge pricing, trip tracking, and scaling to millions of rides."
+++

---

## Problem Statement

Design a ride-sharing service that matches riders with nearby drivers in real-time. The system should handle geolocation tracking, dynamic pricing, trip management, and payment processing.

**Examples**: Uber, Lyft, Ola, Grab

---

## Requirements Clarification

### Functional Requirements

1. **Ride Request**: Request a ride from current location
2. **Driver Matching**: Match rider with nearby driver
3. **Real-Time Tracking**: Track driver/rider location in real-time
4. **Trip Management**: Start, complete, cancel trips
5. **Payment Processing**: Process payment after trip
6. **Surge Pricing**: Dynamic pricing based on demand
7. **Rating System**: Rate drivers and riders
8. **Driver Management**: Driver registration, verification

### Non-Functional Requirements

- **Scale**: 
  - 100M users
  - 10M daily active users
  - 1M rides/day
  - Peak: 100K rides/hour
  - Average trip duration: 20 minutes
- **Latency**: < 5 seconds to match driver
- **Real-Time**: Update location every 5 seconds
- **Availability**: 99.9% uptime

---

## Capacity Estimation

### Traffic Estimates

- **Rides**: 1M rides/day = ~12 rides/second
- **Peak rides**: 100K/hour = ~28 rides/second
- **Location Updates**: 
  - Drivers: 100K drivers × 1 update/5sec = 20K updates/second
  - Riders: 1M active riders × 1 update/5sec = 200K updates/second
  - Total: ~220K location updates/second

### Storage Estimates

- **Trips**: 1M trips/day × 2 KB = 2 GB/day
- **Yearly Trips**: ~730 GB/year
- **Location History**: 220K updates/sec × 100 bytes = 22 MB/sec
- **Daily Location Data**: ~2 TB/day (needs optimization)

---

## API Design

### REST APIs

```
POST /api/v1/rides/request
Request: {
  "riderId": "rider123",
  "pickupLocation": {
    "latitude": 37.7749,
    "longitude": -122.4194
  },
  "dropoffLocation": {
    "latitude": 37.7849,
    "longitude": -122.4294
  },
  "rideType": "standard"  // standard, premium, pool
}
Response: {
  "rideId": "ride456",
  "driverId": "driver789",
  "estimatedArrival": "5 minutes",
  "estimatedFare": 25.50
}

GET /api/v1/rides/{rideId}
Response: {
  "rideId": "ride456",
  "status": "in_progress",
  "driverLocation": {...},
  "riderLocation": {...},
  "estimatedArrival": "10 minutes"
}

POST /api/v1/rides/{rideId}/update-location
Request: {
  "latitude": 37.7750,
  "longitude": -122.4195
}

POST /api/v1/rides/{rideId}/complete
Response: {
  "rideId": "ride456",
  "fare": 28.50,
  "duration": 18,
  "distance": 5.2
}
```

---

## Database Design

### Schema

**Rides Table** (PostgreSQL):
```
rideId (PK): UUID
riderId: UUID
driverId: UUID
status: VARCHAR (requested, matched, in_progress, completed, cancelled)
pickupLocation: POINT (lat, lng)
dropoffLocation: POINT (lat, lng)
fare: DECIMAL
surgeMultiplier: DECIMAL
startedAt: TIMESTAMP
completedAt: TIMESTAMP
```

**Drivers Table** (PostgreSQL):
```
driverId (PK): UUID
name: VARCHAR
vehicleType: VARCHAR
licensePlate: VARCHAR
status: VARCHAR (available, busy, offline)
currentLocation: POINT (lat, lng)
rating: DECIMAL
```

**Location Updates Table** (Time-series DB - InfluxDB/TimescaleDB):
```
timestamp (PK): TIMESTAMP
driverId: UUID
latitude: DECIMAL
longitude: DECIMAL
```

**Ride History Table** (PostgreSQL):
```
rideId (PK): UUID
riderId: UUID
driverId: UUID
fare: DECIMAL
distance: DECIMAL
duration: INT
completedAt: TIMESTAMP
```

### Database Selection

**Rides, Drivers**: **PostgreSQL** (relational data, ACID, PostGIS for geospatial)
**Location Updates**: **Time-series Database** (InfluxDB, TimescaleDB) or **Redis** (recent locations)
**Ride History**: **PostgreSQL** (long-term storage)

---

## High-Level Design

### Architecture

![Ride Sharing Architecture](/images/system-design/ride-sharing-architecture.png)

### Components

1. **Ride Service**: Core ride management
2. **Matching Service**: Match riders with drivers
3. **Location Service**: Handle location updates
4. **Geospatial Database**: Store and query locations
5. **Pricing Service**: Calculate fares, surge pricing
6. **Payment Service**: Process payments
7. **Notification Service**: Notify riders/drivers

---

## Detailed Design

### Driver Matching

**Matching Algorithm**:
1. **Get Nearby Drivers**: Query drivers within radius (e.g., 5 km)
2. **Filter Available**: Only available drivers
3. **Rank Drivers**: 
   - Distance to pickup
   - Driver rating
   - Driver availability
4. **Select Best**: Choose top driver
5. **Notify Driver**: Send ride request to driver
6. **Driver Accepts**: Confirm match

**Geospatial Query**:
- Use PostGIS (PostgreSQL extension) for geospatial queries
- Query: Find drivers within X km of pickup location
- Index: Spatial index (R-tree) for fast queries

---

### Real-Time Location Tracking

**Location Update Flow**:
1. **Driver/Rider App**: Send location update every 5 seconds
2. **Location Service**: Receive update
3. **Store**: Store in time-series database (recent) and Redis (current)
4. **Broadcast**: Broadcast to relevant clients (rider/driver)

**Optimization**:
- **Redis**: Store current location (fast access)
- **Time-Series DB**: Store location history (analytics)
- **Throttling**: Don't update if location change < threshold

**Real-Time Updates**:
- **WebSocket**: Push location updates to clients
- **Server-Sent Events**: Alternative to WebSocket

---

### Surge Pricing

**Surge Pricing Factors**:
1. **Demand**: Number of ride requests
2. **Supply**: Number of available drivers
3. **Time**: Peak hours (rush hour)
4. **Location**: High-demand areas
5. **Weather**: Bad weather increases demand

**Surge Calculation**:
```
surgeMultiplier = (demand / supply) × timeFactor × locationFactor
fare = baseFare × surgeMultiplier
```

**Implementation**:
- **Real-Time Calculation**: Calculate surge in real-time
- **Geographic Zones**: Divide city into zones, calculate per zone
- **Update Frequency**: Update every few minutes

---

### Trip Lifecycle

**Trip States**:
1. **Requested**: Rider requests ride
2. **Matched**: Driver matched, driver notified
3. **Accepted**: Driver accepts ride
4. **Driver Arriving**: Driver heading to pickup
5. **In Progress**: Rider in vehicle, trip started
6. **Completed**: Trip completed, payment processed
7. **Cancelled**: Trip cancelled (by rider or driver)

**State Transitions**:
- **Requested** → **Matched**: Driver found
- **Matched** → **Accepted**: Driver accepts
- **Accepted** → **In Progress**: Trip started
- **In Progress** → **Completed**: Trip ended
- Any state → **Cancelled**: Cancellation

---

### Payment Processing

**Payment Flow**:
1. **Trip Completion**: Trip ends, calculate fare
2. **Payment Service**: Process payment
   - Charge rider's payment method
   - Split fare (platform fee, driver payout)
3. **Update Ride**: Mark ride as paid
4. **Notify**: Send receipt to rider, payout to driver

**Payment Methods**:
- **Credit Card**: Stored payment methods
- **Digital Wallet**: PayPal, Apple Pay, Google Pay
- **Cash**: Cash payment (driver collects)

---

## Scalability

### Horizontal Scaling

- **Stateless Services**: All services scale horizontally
- **Database Sharding**: Shard by geographic region
- **Geographic Partitioning**: Partition by city/region

### Read Scaling

- **Caching**: Cache driver locations in Redis
- **Read Replicas**: Database read replicas
- **CDN**: Serve static assets

### Write Scaling

- **Location Updates**: Use time-series database (optimized for writes)
- **Async Processing**: Process payments asynchronously
- **Message Queue**: Queue ride processing

---

## Reliability

### High Availability

- **Multiple Service Instances**: No single point of failure
- **Database Replication**: Master-slave replication
- **Geographic Redundancy**: Multiple data centers

### Fault Tolerance

- **Driver Matching Failures**: Retry matching, expand search radius
- **Location Update Failures**: Queue updates, process when service recovers
- **Payment Failures**: Retry payment, notify user

### Data Consistency

- **Ride Status**: Strong consistency (critical)
- **Location Updates**: Eventual consistency acceptable
- **Payment**: Strong consistency (critical)

---

## Trade-offs

### Consistency vs Availability

- **Ride Matching**: Strong consistency (prevent double-booking)
- **Location Updates**: Eventual consistency acceptable (AP)
- **Surge Pricing**: Eventual consistency acceptable (AP)

### Latency vs Accuracy

- **Location Updates**: Frequent updates (low latency, higher cost)
- **Matching**: Fast matching (may not be optimal)

---

## Extensions

### Additional Features

1. **Ride Pooling**: Share rides with other riders
2. **Scheduled Rides**: Book rides in advance
3. **Multiple Stops**: Multiple pickup/dropoff points
4. **Driver Analytics**: Track driver performance
5. **Route Optimization**: Optimize routes
6. **Surge Predictions**: Predict surge pricing
7. **Loyalty Program**: Rewards for frequent riders

---

## Key Takeaways

- **Geospatial Database**: Use PostGIS for location queries
- **Real-Time Updates**: WebSocket for real-time location updates
- **Matching Algorithm**: Match based on distance, rating, availability
- **Surge Pricing**: Dynamic pricing based on demand/supply
- **Time-Series Database**: Store location history efficiently
- **Scalability**: Geographic partitioning for scalability

---

## Related Topics

- **[Databases]({{< ref "../databases/_index.md" >}})** - Geospatial and time-series databases
- **[Message Queues & Message Brokers]({{< ref "../system-components/5-Message_Queues_Message_Brokers.md" >}})** - Async processing
- **[Real-Time Systems]({{< ref "../system-design-case-studies/4-Chat_System.md" >}})** - WebSocket patterns
- **[Scalability Patterns]({{< ref "../system-components/9-Scalability_Patterns.md" >}})** - Geographic partitioning

