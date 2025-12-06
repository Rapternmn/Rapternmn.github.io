# Comments Setup Guide

Your blog now supports comments! You can choose from multiple comment systems. **Giscus** is recommended as it's free, privacy-friendly, and uses GitHub Discussions.

## Option 1: Giscus (Recommended) 🌟

Giscus uses GitHub Discussions to power comments. It's free, open-source, and privacy-friendly.

### Setup Steps (IMPORTANT - Follow in order):

1. **Verify Repository is Public:**
   - Go to your repository: https://github.com/Rapternmn/Rapternmn.github.io
   - Make sure the repository is **Public** (not Private)
   - Giscus only works with public repositories
   - If private, go to **Settings** → **General** → **Danger Zone** → **Change repository visibility**

2. **Enable Discussions on your GitHub repository:**
   - Go to your repository: https://github.com/Rapternmn/Rapternmn.github.io
   - Click **Settings** → **General**
   - Scroll down to **Features**
   - Check **Discussions** checkbox
   - Click **Save changes**

3. **Install Giscus GitHub App (REQUIRED - Do this FIRST!):**
   - Go to: https://github.com/apps/giscus
   - Click the green **Install** button
   - Select **Only select repositories**
   - Choose `Rapternmn/Rapternmn.github.io` from the dropdown
   - Click **Install**
   - Review and approve the permissions (it needs access to Discussions)

4. **Get your Giscus configuration:**
   - Visit https://giscus.app
   - Fill in the form:
     - **Repository**: `Rapternmn/Rapternmn.github.io`
     - **Discussion Category**: Choose "Announcements" or create a new one
     - **Page ↔️ Discussions Mapping**: Select "Pathname"
     - **Features**: Enable what you want (reactions, etc.)
     - **Theme**: Choose your preferred theme
   - Click **Generate** to get your configuration
   - If you still see an error, go back to step 3 and make sure the app is installed

3. **Update `hugo.toml`:**
   - Copy the `data-repo-id` value → paste in `params.giscus.repoId`
   - Copy the `data-category-id` value → paste in `params.giscus.categoryId`
   - Update `params.giscus.repo` if needed (should be `Rapternmn/Rapternmn.github.io`)
   - Update `params.giscus.category` to match the category name you chose

4. **Install Giscus app (if prompted):**
   - GitHub may ask you to install the Giscus app
   - Click the link and authorize it

5. **Test it:**
   - Push your changes
   - Visit a blog post
   - Comments section should appear at the bottom

## Option 2: Utterances (Alternative)

Utterances uses GitHub Issues for comments. Simpler setup but less features.

### Setup Steps:

1. **Install Utterances app:**
   - Go to https://github.com/apps/utterances
   - Click **Install** and select your repository
   - Authorize the app

2. **Update `hugo.toml`:**
   - Comment out the `[params.giscus]` section
   - Uncomment the `[params.utterances]` section
   - Update `repo` to `Rapternmn/Rapternmn.github.io`

## Option 3: Disqus (Traditional)

Disqus is a traditional commenting system but includes ads and tracking.

### Setup Steps:

1. **Create a Disqus account:**
   - Go to https://disqus.com
   - Sign up and create a site
   - Get your shortname

2. **Update `hugo.toml`:**
   - Comment out the `[params.giscus]` section
   - Uncomment the `[params.disqus]` section
   - Add your shortname

## Enabling/Disabling Comments

- **Enable comments globally**: Set `params.comments = true` in `hugo.toml` (already done)
- **Disable comments on a specific post**: Add `comments: false` to the post's front matter:
  ```yaml
  +++
  title = "My Post"
  comments = false
  +++
  ```

## Customization

You can customize the comment system appearance and behavior in `hugo.toml`:
- **Theme**: Match your site's theme (light/dark/auto)
- **Position**: Top or bottom of post
- **Reactions**: Enable/disable emoji reactions
- **Language**: Set the comment interface language

## Troubleshooting

### "Cannot use giscus on this repository" Error

This error means one of the requirements is not met. Check each:

1. **Repository must be PUBLIC:**
   - Go to your repo → Settings → General
   - Check if it says "Public" or "Private"
   - If Private, change it to Public (Settings → Danger Zone)

2. **Discussions must be ENABLED:**
   - Go to your repo → Settings → General → Features
   - Make sure "Discussions" checkbox is checked
   - Click "Save changes" if you just enabled it

3. **Giscus App must be INSTALLED:**
   - Go to: https://github.com/apps/giscus
   - Click "Configure" or "Install"
   - Make sure `Rapternmn/Rapternmn.github.io` is selected
   - If not installed, click "Install" and select your repository
   - Grant all requested permissions

4. **Wait a few minutes:**
   - After installing the app, wait 1-2 minutes
   - Then refresh https://giscus.app and try again

5. **Verify repository name:**
   - Make sure you're using: `Rapternmn/Rapternmn.github.io` (case-sensitive)
   - Check for typos

### Other Issues

- **Comments not showing?**
  - Check that `params.comments = true` in `hugo.toml`
  - Verify your Giscus configuration is correct
  - Make sure `repoId` and `categoryId` are filled in
  - Check browser console for errors (F12 → Console tab)

- **Giscus not loading on site?**
  - Verify `repoId` and `categoryId` are correct in `hugo.toml`
  - Make sure the Giscus app is installed on your repo
  - Check that Discussions are enabled
  - Clear browser cache and try again

## Privacy & Moderation

- **Giscus**: Comments are stored in GitHub Discussions, you can moderate them there
- **Utterances**: Comments are stored in GitHub Issues, you can moderate them there
- **Disqus**: Comments are stored on Disqus servers, moderate via Disqus dashboard

