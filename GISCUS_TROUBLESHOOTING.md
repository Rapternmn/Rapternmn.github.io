# Giscus Setup Troubleshooting Checklist

If you're getting "Cannot use giscus on this repository" error, check each item below:

## ✅ Step-by-Step Fix

### 1. Repository Visibility (MUST BE PUBLIC)
- [ ] Go to: https://github.com/Rapternmn/Rapternmn.github.io
- [ ] Click **Settings** → **General**
- [ ] Check if it says **"Public"** at the top
- [ ] If it says **"Private"**, you need to make it public:
  - Scroll to **Danger Zone** at the bottom
  - Click **Change repository visibility**
  - Select **Make public**
  - Type the repository name to confirm
  - Click **I understand, change repository visibility**

### 2. Enable Discussions
- [ ] Go to: https://github.com/Rapternmn/Rapternmn.github.io/settings
- [ ] Scroll to **Features** section
- [ ] Check the **Discussions** checkbox
- [ ] Click **Save changes** (if you just enabled it)
- [ ] Wait 30 seconds for changes to propagate

### 3. Install Giscus App (CRITICAL STEP!)
- [ ] Go to: https://github.com/apps/giscus
- [ ] Click the green **Install** button (or **Configure** if already installed)
- [ ] Select **Only select repositories**
- [ ] Choose `Rapternmn/Rapternmn.github.io` from the dropdown
- [ ] Click **Install** (or **Save** if configuring)
- [ ] Review permissions - make sure it has access to:
  - ✅ Discussions (read and write)
  - ✅ Metadata (read-only)
- [ ] Click **Approve & Install**

### 4. Verify Installation
- [ ] Go to: https://github.com/Rapternmn/Rapternmn.github.io/settings/installations
- [ ] You should see **giscus** in the list of installed apps
- [ ] If not, go back to step 3

### 5. Wait and Retry
- [ ] Wait 1-2 minutes after installing the app
- [ ] Go to: https://giscus.app
- [ ] Enter repository: `Rapternmn/Rapternmn.github.io`
- [ ] The form should now work without errors

### 6. Get Configuration
- [ ] Fill in the Giscus form:
  - Repository: `Rapternmn/Rapternmn.github.io`
  - Category: Choose "Announcements" or create new
  - Mapping: "Pathname"
  - Theme: Your preference
- [ ] Click **Generate**
- [ ] Copy the `data-repo-id` value
- [ ] Copy the `data-category-id` value

### 7. Update hugo.toml
- [ ] Open `hugo.toml`
- [ ] Find `[params.giscus]` section
- [ ] Paste `repoId` value
- [ ] Paste `categoryId` value
- [ ] Save the file

## Common Issues

### Issue: "Repository not found"
- **Fix**: Make sure the repository name is exactly `Rapternmn/Rapternmn.github.io` (case-sensitive)
- Check: https://github.com/Rapternmn/Rapternmn.github.io exists and is accessible

### Issue: "Discussions not enabled"
- **Fix**: Go to Settings → General → Features → Enable Discussions
- Wait 30 seconds and try again

### Issue: "Giscus app not installed"
- **Fix**: Install the app from https://github.com/apps/giscus
- Make sure you select the correct repository during installation

### Issue: Repository is Private
- **Fix**: Change repository visibility to Public
- Settings → General → Danger Zone → Change visibility

## Quick Test

After completing all steps:
1. Push your changes to GitHub
2. Wait for CI/CD to deploy
3. Visit a blog post on your live site
4. Scroll to the bottom - you should see the Giscus comments section

## Still Having Issues?

If you've checked all items above and it still doesn't work:
1. Double-check the repository name spelling
2. Make sure you're logged into the correct GitHub account
3. Try clearing your browser cache
4. Wait 5 minutes and try again (GitHub sometimes needs time to propagate changes)

