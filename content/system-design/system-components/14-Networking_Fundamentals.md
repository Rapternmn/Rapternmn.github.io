+++
title = "Networking Fundamentals"
date = 2025-12-19T10:00:00+05:30
draft = false
weight = 14
description = "Networking fundamentals: OSI model, TCP/IP, protocols (TCP, UDP, HTTP, HTTPS), DNS, IP addressing, subnetting, routing, and network topologies essential for system design."
+++

---

## Introduction

Understanding networking fundamentals is crucial for system design. Networks enable communication between distributed system components, and knowledge of protocols, addressing, and network architecture is essential for designing scalable, reliable systems.

**Key Topics**:
- OSI and TCP/IP models
- Network protocols (TCP, UDP, HTTP, HTTPS)
- IP addressing and subnetting
- DNS and name resolution
- Routing and switching
- Network topologies
- Common network protocols

---

## OSI Model

### Overview

The **OSI (Open Systems Interconnection) Model** is a conceptual framework that standardizes the functions of a telecommunication or computing system into seven abstraction layers.

### The Seven Layers

**1. Physical Layer (Layer 1)**
- **Function**: Transmits raw bits over physical medium
- **Protocols**: Ethernet, USB, Bluetooth
- **Devices**: Cables, hubs, repeaters
- **Data Unit**: Bits

**2. Data Link Layer (Layer 2)**
- **Function**: Error detection/correction, MAC addressing, framing
- **Protocols**: Ethernet, Wi-Fi (802.11), PPP
- **Devices**: Switches, bridges
- **Data Unit**: Frames
- **Key Concepts**: MAC addresses, ARP (Address Resolution Protocol)

**3. Network Layer (Layer 3)**
- **Function**: Routing, logical addressing, path determination
- **Protocols**: IP (IPv4, IPv6), ICMP, ARP
- **Devices**: Routers
- **Data Unit**: Packets
- **Key Concepts**: IP addresses, routing tables, subnetting

**4. Transport Layer (Layer 4)**
- **Function**: End-to-end communication, error recovery, flow control
- **Protocols**: TCP, UDP
- **Devices**: Firewalls (stateful)
- **Data Unit**: Segments (TCP) or Datagrams (UDP)
- **Key Concepts**: Ports, connection-oriented vs connectionless

**5. Session Layer (Layer 5)**
- **Function**: Session management, synchronization
- **Protocols**: NetBIOS, PPTP
- **Key Concepts**: Session establishment, maintenance, termination

**6. Presentation Layer (Layer 6)**
- **Function**: Data translation, encryption, compression
- **Protocols**: SSL/TLS (partially), JPEG, MPEG
- **Key Concepts**: Data encoding, encryption, compression

**7. Application Layer (Layer 7)**
- **Function**: User interface, application services
- **Protocols**: HTTP, HTTPS, FTP, SMTP, DNS, SSH
- **Devices**: Application firewalls, proxies
- **Data Unit**: Messages
- **Key Concepts**: Application protocols, APIs

### Data Flow Through OSI Layers

```
Application Data
    ↓
Presentation (Encryption/Compression)
    ↓
Session (Session Management)
    ↓
Transport (TCP/UDP - Segmentation)
    ↓
Network (IP - Routing)
    ↓
Data Link (Ethernet - Framing)
    ↓
Physical (Bits on Wire)
```

---

## TCP/IP Model

### Overview

The **TCP/IP Model** is a practical, four-layer model that maps closely to the OSI model but is more aligned with actual network implementation.

### The Four Layers

**1. Network Interface Layer (Link Layer)**
- **Maps to**: OSI Layers 1-2
- **Function**: Physical transmission, MAC addressing
- **Protocols**: Ethernet, Wi-Fi, PPP

**2. Internet Layer (Network Layer)**
- **Maps to**: OSI Layer 3
- **Function**: Routing, IP addressing
- **Protocols**: IP, ICMP, ARP
- **Key Concepts**: IP addresses, routing

**3. Transport Layer**
- **Maps to**: OSI Layer 4
- **Function**: End-to-end communication
- **Protocols**: TCP, UDP
- **Key Concepts**: Ports, reliability

**4. Application Layer**
- **Maps to**: OSI Layers 5-7
- **Function**: Application protocols
- **Protocols**: HTTP, HTTPS, FTP, SMTP, DNS, SSH

### OSI vs TCP/IP Comparison

| OSI Model | TCP/IP Model | Protocols |
|-----------|--------------|-----------|
| Application | Application | HTTP, HTTPS, FTP, SMTP |
| Presentation | Application | SSL/TLS, JPEG |
| Session | Application | NetBIOS |
| Transport | Transport | TCP, UDP |
| Network | Internet | IP, ICMP |
| Data Link | Network Interface | Ethernet, Wi-Fi |
| Physical | Network Interface | Cables, Hubs |

---

## TCP (Transmission Control Protocol)

### Characteristics

**Connection-Oriented**: Establishes connection before data transfer

**Reliable**: 
- Guarantees delivery
- Error detection and correction
- Retransmission of lost packets
- Ordered delivery

**Flow Control**: Prevents sender from overwhelming receiver

**Congestion Control**: Adjusts transmission rate based on network conditions

### TCP Three-Way Handshake

**Connection Establishment**:

```
Client                          Server
  |                                |
  |---- SYN (seq=x) -------------->|
  |                                |
  |<--- SYN-ACK (seq=y, ack=x+1) --|
  |                                |
  |---- ACK (seq=x+1, ack=y+1) --->|
  |                                |
Connection Established
```

**1. SYN**: Client sends synchronization packet with initial sequence number
**2. SYN-ACK**: Server acknowledges and sends its own sequence number
**3. ACK**: Client acknowledges server's sequence number

### TCP Connection Termination

**Four-Way Handshake**:

```
Client                          Server
  |                                |
  |---- FIN (seq=x) -------------->|
  |                                |
  |<--- ACK (ack=x+1) -------------|
  |                                |
  |<--- FIN (seq=y) ---------------|
  |                                |
  |---- ACK (ack=y+1) ------------>|
  |                                |
Connection Closed
```

### TCP Features

**1. Sequence Numbers**
- Track byte order
- Enable retransmission
- Detect duplicates

**2. Acknowledgments (ACK)**
- Confirm received data
- Cumulative or selective

**3. Flow Control (Sliding Window)**
- Receiver advertises window size
- Sender adjusts transmission rate

**4. Congestion Control**
- Slow start
- Congestion avoidance
- Fast retransmit
- Fast recovery

### TCP Header Structure

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|          Source Port          |       Destination Port        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        Sequence Number                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    Acknowledgment Number                      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Data |           |U|A|P|R|S|F|                               |
| Offset| Reserved  |R|C|S|S|Y|I|            Window             |
|       |           |G|K|H|T|N|N|                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|           Checksum            |         Urgent Pointer        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

**Key Fields**:
- **Source/Destination Port**: Identify applications
- **Sequence Number**: Byte order
- **Acknowledgment Number**: Next expected byte
- **Flags**: SYN, ACK, FIN, RST, PSH, URG
- **Window Size**: Flow control
- **Checksum**: Error detection

### TCP Use Cases

- **HTTP/HTTPS**: Web browsing
- **FTP**: File transfer
- **SMTP**: Email
- **SSH**: Secure shell
- **Database connections**: MySQL, PostgreSQL
- **Any application requiring reliability**

---

## UDP (User Datagram Protocol)

### Characteristics

**Connectionless**: No connection establishment

**Unreliable**:
- No delivery guarantee
- No error recovery
- No retransmission
- No ordered delivery

**Low Overhead**: Smaller header, faster transmission

**No Flow/Congestion Control**: Sends at application rate

### UDP Header Structure

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|          Source Port          |       Destination Port        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|            Length             |           Checksum            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

**Key Fields**:
- **Source/Destination Port**: Identify applications
- **Length**: UDP header + data length
- **Checksum**: Error detection (optional)

### UDP Use Cases

- **DNS**: Domain name resolution (fast queries)
- **DHCP**: Dynamic IP assignment
- **Video Streaming**: Real-time video (some packet loss acceptable)
- **Voice over IP (VoIP)**: Real-time voice
- **Online Gaming**: Low latency critical
- **SNMP**: Network management
- **TFTP**: Trivial file transfer

### TCP vs UDP Comparison

| Feature | TCP | UDP |
|---------|-----|-----|
| Connection | Connection-oriented | Connectionless |
| Reliability | Reliable | Unreliable |
| Ordering | Ordered delivery | No ordering |
| Error Recovery | Yes | No |
| Flow Control | Yes | No |
| Congestion Control | Yes | No |
| Overhead | High (20 bytes header) | Low (8 bytes header) |
| Speed | Slower | Faster |
| Use Cases | Web, email, file transfer | DNS, video, gaming |

---

## HTTP (Hypertext Transfer Protocol)

### Overview

**HTTP** is an application-layer protocol for transmitting hypermedia documents (HTML, images, etc.) over the web.

### HTTP Versions

**HTTP/1.0**:
- One request per connection
- No persistent connections
- No pipelining

**HTTP/1.1**:
- Persistent connections (keep-alive)
- Pipelining (limited)
- Host header required
- Chunked transfer encoding

**HTTP/2**:
- Multiplexing (multiple requests on one connection)
- Header compression (HPACK)
- Server push
- Binary protocol
- Stream prioritization

**HTTP/3**:
- Uses QUIC (UDP-based)
- Faster connection establishment
- Improved multiplexing
- Better error recovery

### HTTP Request Structure

```
GET /api/users HTTP/1.1
Host: example.com
User-Agent: Mozilla/5.0
Accept: application/json
Content-Type: application/json
Content-Length: 123

{request body}
```

**Components**:
- **Request Line**: Method, URI, HTTP version
- **Headers**: Key-value pairs
- **Body**: Optional data

### HTTP Methods

**GET**: Retrieve resource (idempotent, safe)
**POST**: Create resource or submit data
**PUT**: Update/replace resource (idempotent)
**PATCH**: Partial update (idempotent)
**DELETE**: Delete resource (idempotent)
**HEAD**: Get headers only (idempotent, safe)
**OPTIONS**: Get allowed methods (idempotent, safe)

### HTTP Status Codes

**2xx Success**:
- `200 OK`: Request succeeded
- `201 Created`: Resource created
- `204 No Content`: Success, no content

**3xx Redirection**:
- `301 Moved Permanently`: Permanent redirect
- `302 Found`: Temporary redirect
- `304 Not Modified`: Use cached version

**4xx Client Error**:
- `400 Bad Request`: Invalid request
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Access denied
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded

**5xx Server Error**:
- `500 Internal Server Error`: Server error
- `502 Bad Gateway`: Invalid response from upstream
- `503 Service Unavailable`: Service temporarily unavailable
- `504 Gateway Timeout`: Upstream timeout

### HTTP Headers

**Request Headers**:
- `Host`: Server domain
- `User-Agent`: Client information
- `Accept`: Acceptable content types
- `Authorization`: Credentials
- `Content-Type`: Request body type
- `Content-Length`: Request body size

**Response Headers**:
- `Content-Type`: Response body type
- `Content-Length`: Response body size
- `Cache-Control`: Caching directives
- `Set-Cookie`: Set cookies
- `Location`: Redirect URL
- `ETag`: Entity tag for caching

---

## HTTPS (HTTP Secure)

### Overview

**HTTPS** is HTTP over TLS/SSL, providing encryption and authentication.

### How HTTPS Works

**1. TLS Handshake**:
```
Client                          Server
  |                                |
  |---- ClientHello -------------->|
  |                                |
  |<--- ServerHello ---------------|
  |<--- Certificate --------------|
  |<--- ServerHelloDone -----------|
  |                                |
  |---- ClientKeyExchange -------->|
  |---- ChangeCipherSpec --------->|
  |---- Finished ----------------->|
  |                                |
  |<--- ChangeCipherSpec ---------|
  |<--- Finished ------------------|
  |                                |
Encrypted Communication
```

**2. Certificate Validation**:
- Server sends certificate
- Client validates certificate chain
- Verifies certificate authority (CA)
- Checks expiration and revocation

**3. Key Exchange**:
- Symmetric key established
- Used for encryption/decryption

### TLS/SSL Versions

**SSL 2.0/3.0**: Deprecated (vulnerabilities)

**TLS 1.0/1.1**: Deprecated (weak ciphers)

**TLS 1.2**: Widely used, secure

**TLS 1.3**: Latest, improved security and performance

### HTTPS Benefits

- **Encryption**: Data encrypted in transit
- **Authentication**: Server identity verified
- **Integrity**: Data tampering detected
- **Privacy**: Prevents eavesdropping
- **SEO**: Search engines favor HTTPS

### HTTPS Port

- **Default Port**: 443 (HTTP uses 80)

---

## DNS (Domain Name System)

### Overview

**DNS** translates human-readable domain names to IP addresses.

### DNS Hierarchy

```
                    Root (.)
                     |
        +------------+------------+
        |            |            |
     .com          .org         .net
        |            |            |
    example.com   example.org  example.net
        |
    www.example.com
```

**Components**:
- **Root Servers**: Top of hierarchy (.)
- **Top-Level Domains (TLD)**: .com, .org, .net, .edu
- **Second-Level Domains**: example.com
- **Subdomains**: www.example.com

### DNS Record Types

**A Record**: Maps domain to IPv4 address
```
example.com → 192.0.2.1
```

**AAAA Record**: Maps domain to IPv6 address
```
example.com → 2001:db8::1
```

**CNAME Record**: Alias to another domain
```
www.example.com → example.com
```

**MX Record**: Mail exchange server
```
example.com → mail.example.com (priority 10)
```

**TXT Record**: Text information (SPF, DKIM, etc.)
```
example.com → "v=spf1 include:_spf.google.com ~all"
```

**NS Record**: Name server
```
example.com → ns1.example.com
```

**PTR Record**: Reverse DNS (IP to domain)
```
192.0.2.1 → example.com
```

### DNS Resolution Process

**1. Local Cache Check**
- Browser cache
- OS cache
- Router cache

**2. Recursive Query to DNS Resolver**
- ISP's DNS server
- Public DNS (8.8.8.8, 1.1.1.1)

**3. Iterative Query**
```
Resolver → Root Server (.): "Where is .com?"
Root → Resolver: "Ask .com TLD server"

Resolver → .com TLD: "Where is example.com?"
.com TLD → Resolver: "Ask example.com nameserver"

Resolver → example.com NS: "What is www.example.com?"
example.com NS → Resolver: "192.0.2.1"
```

**4. Response Cached**
- Result stored in cache (TTL)

### DNS Caching

**TTL (Time To Live)**: How long record is cached
- Shorter TTL: More frequent updates, more queries
- Longer TTL: Fewer queries, slower updates

### DNS Use Cases

- **Domain Resolution**: www.example.com → IP
- **Load Balancing**: Multiple A records
- **Failover**: Multiple A records with health checks
- **CDN**: Geographic routing
- **Email**: MX records for mail routing

---

## IP Addressing

### IPv4

**32-bit address** (4 octets)

**Format**: `192.168.1.1`

**Address Space**: 4.3 billion addresses (exhausted)

**Private Ranges**:
- `10.0.0.0/8` (10.0.0.0 - 10.255.255.255)
- `172.16.0.0/12` (172.16.0.0 - 172.31.255.255)
- `192.168.0.0/16` (192.168.0.0 - 192.168.255.255)

**Special Addresses**:
- `127.0.0.1`: Localhost
- `0.0.0.0`: All interfaces
- `255.255.255.255`: Broadcast

### IPv6

**128-bit address** (8 groups of 4 hex digits)

**Format**: `2001:0db8:85a3:0000:0000:8a2e:0370:7334`

**Shortened**: `2001:db8:85a3::8a2e:370:7334`

**Address Space**: 340 undecillion addresses

**Benefits**:
- Larger address space
- Simplified header
- Built-in security (IPsec)
- Better multicast
- Auto-configuration

### Subnetting

**Purpose**: Divide network into smaller subnets

**CIDR Notation**: `192.168.1.0/24`
- `/24` means 24 bits for network, 8 bits for hosts
- 256 addresses (254 usable, 1 network, 1 broadcast)

**Common Subnets**:
- `/24` (255.255.255.0): 256 addresses
- `/16` (255.255.0.0): 65,536 addresses
- `/8` (255.0.0.0): 16,777,216 addresses

**Subnet Calculation**:
```
Network: 192.168.1.0/24
Subnet Mask: 255.255.255.0
Network Address: 192.168.1.0
Broadcast: 192.168.1.255
Usable Range: 192.168.1.1 - 192.168.1.254
```

---

## Routing

### Overview

**Routing** determines the path packets take from source to destination.

### Routing Tables

**Contains**:
- Destination network
- Next hop (gateway)
- Interface
- Metric (cost)

**Example**:
```
Destination      Gateway          Interface    Metric
0.0.0.0/0        192.168.1.1     eth0         10
192.168.1.0/24   *               eth0         0
10.0.0.0/8       192.168.1.100  eth0         20
```

### Routing Protocols

**1. Static Routing**
- Manually configured
- Simple, predictable
- No automatic updates

**2. Dynamic Routing**
- Automatically updates
- Adapts to changes
- More complex

**Interior Gateway Protocols (IGP)**:
- **RIP**: Distance vector, simple
- **OSPF**: Link state, fast convergence
- **EIGRP**: Cisco proprietary, hybrid

**Exterior Gateway Protocols (EGP)**:
- **BGP**: Border Gateway Protocol, internet routing

### Default Gateway

**Router** that forwards packets to external networks

**Example**: `192.168.1.1` (home router)

---

## Network Topologies

### Star Topology

**Structure**: All devices connected to central hub/switch

**Advantages**:
- Easy to manage
- Failure of one device doesn't affect others
- Easy to add/remove devices

**Disadvantages**:
- Single point of failure (hub/switch)
- Requires more cabling

### Bus Topology

**Structure**: All devices on single cable

**Advantages**:
- Simple, inexpensive
- Less cabling

**Disadvantages**:
- Single point of failure (cable)
- Difficult to troubleshoot
- Limited scalability

### Ring Topology

**Structure**: Devices connected in circular ring

**Advantages**:
- Predictable performance
- No collisions

**Disadvantages**:
- Single point of failure
- Difficult to add/remove devices

### Mesh Topology

**Structure**: Every device connected to every other device

**Advantages**:
- High redundancy
- No single point of failure
- High reliability

**Disadvantages**:
- Expensive (lots of cabling)
- Complex to manage

### Hybrid Topology

**Structure**: Combination of multiple topologies

**Advantages**:
- Flexibility
- Scalability

**Disadvantages**:
- More complex

---

## Common Network Protocols

### FTP (File Transfer Protocol)

**Purpose**: File transfer

**Ports**: 21 (control), 20 (data)

**Modes**: Active, Passive

**Security**: FTP (insecure), FTPS (TLS), SFTP (SSH)

### SMTP (Simple Mail Transfer Protocol)

**Purpose**: Email transmission

**Port**: 25 (unencrypted), 587 (TLS), 465 (SSL)

**Use Cases**: Sending email

### POP3 (Post Office Protocol)

**Purpose**: Email retrieval

**Port**: 110 (unencrypted), 995 (SSL)

**Characteristics**: Downloads and deletes from server

### IMAP (Internet Message Access Protocol)

**Purpose**: Email retrieval

**Port**: 143 (unencrypted), 993 (SSL)

**Characteristics**: Keeps email on server, syncs across devices

### SSH (Secure Shell)

**Purpose**: Secure remote access

**Port**: 22

**Features**: Encryption, authentication, port forwarding

### Telnet

**Purpose**: Remote access (insecure)

**Port**: 23

**Note**: Deprecated, use SSH instead

### SNMP (Simple Network Management Protocol)

**Purpose**: Network management

**Ports**: 161 (UDP), 162 (UDP, traps)

**Use Cases**: Monitoring network devices

---

## Network Devices

### Hub

**Layer**: Physical (Layer 1)

**Function**: Broadcasts to all ports

**Characteristics**: No intelligence, collision domain

### Switch

**Layer**: Data Link (Layer 2)

**Function**: Forwards based on MAC addresses

**Characteristics**: Learns MAC addresses, reduces collisions

**Types**:
- **Unmanaged**: Plug and play
- **Managed**: Configurable, VLAN support

### Router

**Layer**: Network (Layer 3)

**Function**: Routes based on IP addresses

**Characteristics**: Connects different networks, NAT support

### Firewall

**Layer**: Network/Transport/Application

**Function**: Filters traffic based on rules

**Types**:
- **Packet Filtering**: Layer 3/4
- **Stateful**: Tracks connections
- **Application**: Layer 7 inspection

### Load Balancer

**Layer**: Transport/Application

**Function**: Distributes traffic across servers

**Types**: Layer 4 (TCP/UDP), Layer 7 (HTTP/HTTPS)

---

## Network Troubleshooting

### Common Tools

**ping**: Test connectivity
```bash
ping google.com
ping 8.8.8.8
```

**traceroute/tracert**: Trace route
```bash
traceroute google.com
```

**nslookup/dig**: DNS queries
```bash
nslookup example.com
dig example.com
```

**netstat**: Network connections
```bash
netstat -an
netstat -rn  # Routing table
```

**tcpdump/wireshark**: Packet capture
```bash
tcpdump -i eth0
```

**curl**: HTTP requests
```bash
curl -v https://example.com
```

### Common Issues

**1. Connectivity Problems**
- Check physical connections
- Verify IP configuration
- Test with ping
- Check routing table

**2. DNS Issues**
- Verify DNS server
- Test with nslookup
- Check DNS cache
- Verify DNS records

**3. Firewall Blocking**
- Check firewall rules
- Verify ports are open
- Test with telnet/nc

**4. High Latency**
- Check network congestion
- Verify routing path
- Test with traceroute
- Check bandwidth

---

## Network Security

### Firewalls

**Purpose**: Control network traffic

**Types**:
- **Network Firewall**: Between networks
- **Host Firewall**: On individual hosts
- **Application Firewall**: Layer 7 filtering

### VPN (Virtual Private Network)

**Purpose**: Encrypted tunnel over public network

**Types**:
- **Site-to-Site**: Connect networks
- **Remote Access**: Connect users
- **SSL VPN**: Browser-based

### NAT (Network Address Translation)

**Purpose**: Translate private IPs to public IPs

**Types**:
- **Static NAT**: One-to-one mapping
- **Dynamic NAT**: Pool of public IPs
- **PAT/NAPT**: Port-based translation

### VLAN (Virtual LAN)

**Purpose**: Logical network segmentation

**Benefits**:
- Security isolation
- Broadcast domain separation
- Flexible network design

---

## Summary

**Network Models**:
- **OSI Model**: 7 layers (conceptual)
- **TCP/IP Model**: 4 layers (practical)

**Key Protocols**:
- **TCP**: Reliable, connection-oriented
- **UDP**: Fast, connectionless
- **HTTP/HTTPS**: Web protocols
- **DNS**: Name resolution

**Addressing**:
- **IPv4**: 32-bit, exhausted
- **IPv6**: 128-bit, future
- **Subnetting**: Network segmentation

**Routing**: Path determination, routing tables, protocols

**Topologies**: Star, bus, ring, mesh, hybrid

**Security**: Firewalls, VPNs, NAT, VLANs

Understanding networking fundamentals is essential for designing distributed systems, troubleshooting issues, and ensuring secure, reliable communication between system components!

