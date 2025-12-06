# Analytics Setup Guide

This guide explains how to add viewership analytics to your Hugo blog. Multiple analytics options are supported, from traditional Google Analytics to privacy-friendly alternatives.

## Available Analytics Services

### 1. Google Analytics 4 (GA4) - Recommended for Most Users

**Pros:**
- Free and comprehensive
- Industry standard
- Detailed insights and reports
- Integration with Google services

**Cons:**
- Requires cookie consent in some regions (GDPR)
- More privacy concerns

**Setup Steps:**

1. **Create a Google Analytics account:**
   - Go to https://analytics.google.com/
   - Sign in with your Google account
   - Create a new property for your website

2. **Get your Measurement ID:**
   - In GA4, go to **Admin** → **Data Streams**
   - Click on your web stream
   - Copy your **Measurement ID** (format: `G-XXXXXXXXXX`)

3. **Add to `hugo.toml`:**
   ```toml
   [params.analytics.google]
     id = "G-XXXXXXXXXX"  # Your Measurement ID
   ```

4. **Verify:**
   - Deploy your site
   - Visit your blog
   - Check Google Analytics Real-Time reports to see if it's working

### 2. Plausible Analytics - Privacy-Friendly

**Pros:**
- Privacy-friendly (GDPR compliant, no cookies)
- Simple, clean dashboard
- Lightweight script
- Open source

**Cons:**
- Paid service (starts at $9/month)
- Less detailed than Google Analytics

**Setup Steps:**

1. **Sign up:**
   - Go to https://plausible.io/
   - Create an account and add your domain

2. **Add to `hugo.toml`:**
   ```toml
   [params.analytics.plausible]
     domain = "rapternmn.github.io"  # Your domain
   ```

### 3. Umami Analytics - Self-Hosted Privacy-Friendly

**Pros:**
- Completely free (self-hosted)
- Privacy-friendly
- Open source
- No cookies

**Cons:**
- Requires self-hosting setup
- More technical setup

**Setup Steps:**

1. **Self-host Umami** (or use a hosted service)
2. **Get your website ID** from your Umami dashboard
3. **Add to `hugo.toml`:**
   ```toml
   [params.analytics.umami]
     id = "your-website-id"
     src = "https://your-umami-instance.com/script.js"
   ```

### 4. GoatCounter - Privacy-Friendly & Free

**Pros:**
- Free tier available
- Privacy-friendly
- Simple setup
- Open source

**Cons:**
- Less features than Google Analytics
- Free tier has limitations

**Setup Steps:**

1. **Sign up:**
   - Go to https://www.goatcounter.com/
   - Create a free account

2. **Add to `hugo.toml`:**
   ```toml
   [params.analytics.goatcounter]
     code = "your-code"  # Your GoatCounter code
   ```

### 5. Cloudflare Web Analytics - Privacy-Friendly & Free

**Pros:**
- Free
- Privacy-friendly
- No cookies
- Easy setup if using Cloudflare

**Cons:**
- Requires Cloudflare account
- Less detailed than Google Analytics

**Setup Steps:**

1. **Enable in Cloudflare:**
   - Go to your Cloudflare dashboard
   - Navigate to **Analytics** → **Web Analytics**
   - Enable for your domain

2. **Get token and add to `hugo.toml`:**
   ```toml
   [params.analytics.cloudflare]
     token = "your-token"
   ```

## Quick Start: Google Analytics 4

The most common choice. Here's the fastest setup:

1. **Get your GA4 Measurement ID:**
   - Visit https://analytics.google.com/
   - Create property → Get Measurement ID (starts with `G-`)

2. **Update `hugo.toml`:**
   ```toml
   [params.analytics.google]
     id = "G-XXXXXXXXXX"  # Replace with your ID
   ```

3. **Deploy and verify:**
   - Push changes to GitHub
   - Visit your site
   - Check GA4 Real-Time reports

## Privacy Considerations

- **Google Analytics**: Requires cookie consent banners in EU (GDPR)
- **Plausible/Umami/GoatCounter/Cloudflare**: Privacy-friendly, no cookies needed

## Multiple Analytics

You can enable multiple analytics services simultaneously. Just configure them all in `hugo.toml`.

## Verification

After adding analytics:

1. Deploy your site
2. Visit a few pages on your blog
3. Check your analytics dashboard (usually takes a few minutes to show data)
4. Verify page views are being tracked

## Troubleshooting

- **Not seeing data?**
  - Wait 5-10 minutes (analytics can be delayed)
  - Check browser console for errors
  - Verify your ID/token is correct
  - Make sure you're in production mode (not localhost)

- **Script not loading?**
  - Check browser console for errors
  - Verify the analytics service is accessible
  - Check ad blockers (they may block analytics)

## Recommended for Your Blog

For a technical blog like yours, I'd recommend:
- **Google Analytics 4** - For comprehensive insights
- **Plausible** - As a privacy-friendly alternative

You can use both simultaneously if desired.

