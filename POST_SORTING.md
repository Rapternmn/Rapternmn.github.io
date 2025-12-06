# How to Control Blog Post Sorting in Hugo

Hugo provides several ways to control how your blog posts are sorted. Here are the main methods:

## Method 1: Using `date` in Front Matter (Default)

By default, Hugo sorts posts by **date** (newest first). You can control this by setting the `date` field in each post's front matter:

```toml
+++
title = "My Post"
date = 2025-11-22T10:00:00+05:30
+++
```

**Tips:**
- Use different times on the same day to control order
- Newer dates appear first (descending order)
- Format: `YYYY-MM-DDTHH:MM:SS+TZ`

## Method 2: Using `weight` Parameter

The `weight` parameter gives you explicit control over post order. **Lower numbers appear first**.

```toml
+++
title = "First Post"
date = 2025-11-22T10:00:00+05:30
weight = 1
+++
```

```toml
+++
title = "Second Post"
date = 2025-11-22T10:00:00+05:30
weight = 2
+++
```

**Sorting Priority:**
1. Posts with `weight` are sorted by weight (ascending: 1, 2, 3...)
2. Posts without `weight` are sorted by date (newest first)
3. Posts with same weight are sorted by date

## Method 3: Using `publishDate` and `expiryDate`

You can also use:
- `publishDate` - When the post should be published
- `expiryDate` - When the post should expire/hide

## Method 4: Global Sorting Configuration

You can configure default sorting in `hugo.toml`:

```toml
[params]
  # Sort posts by date (default)
  # Options: "date", "weight", "title", "lastmod"
  sortBy = "date"
  
  # Sort order: "asc" (oldest first) or "desc" (newest first)
  sortOrder = "desc"
```

## Current Sorting Behavior

Your blog currently sorts posts by **date** (newest first) by default. This is the standard Hugo behavior.

## Examples

### Sort by Date (Newest First) - Current Default
```toml
+++
title = "Latest Post"
date = 2025-11-22T15:00:00+05:30
+++
```

### Sort by Weight (Explicit Order)
```toml
+++
title = "Introduction Post"
date = 2025-11-22T10:00:00+05:30
weight = 1  # Appears first
+++
```

```toml
+++
title = "Advanced Topic"
date = 2025-11-22T10:00:00+05:30
weight = 10  # Appears later
+++
```

### Sort by Date with Same Time (Use Weight)
If multiple posts have the same date/time, use weight to control order:
```toml
+++
title = "Post A"
date = 2025-11-22T10:00:00+05:30
weight = 1
+++
```

```toml
+++
title = "Post B"
date = 2025-11-22T10:00:00+05:30
weight = 2
+++
```

## Best Practices

1. **For chronological blogs**: Use `date` only (current setup)
2. **For tutorial series**: Use `weight` to ensure proper sequence
3. **For mixed content**: Use `weight` for important posts, `date` for others
4. **For same-day posts**: Use different times or add `weight`

## Quick Fix: Reorder Posts by Date

If you want to change the order of existing posts:

1. **Option A**: Update the `date` field in front matter
   - Newer date = appears first
   - Older date = appears later

2. **Option B**: Add `weight` to front matter
   - Lower number = appears first
   - Higher number = appears later

## Example: Making a Post Appear First

```toml
+++
title = "Featured Post"
date = 2025-11-22T10:00:00+05:30
weight = 0  # Lower than others, appears first
+++
```

