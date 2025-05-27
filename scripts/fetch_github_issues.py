# scripts/fetch_github_issues.py

from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
import os
from dotenv import load_dotenv
import json
import asyncio
import sys

# Load environment variables
load_dotenv()

# Get GitHub token
print("\n🔍 Debug: Checking token sources...")
print(f"Token from environment: {'Yes' if os.environ.get('GITHUB_TOKEN') else 'No'}")
print(f"Token from .env: {'Yes' if os.getenv('GITHUB_TOKEN') else 'No'}")

token = os.getenv("GITHUB_TOKEN")
if not token:
    print("Error: GITHUB_TOKEN environment variable is not set.")
    print("Please create a GitHub Personal Access Token and add it to your .env file:")
    print("1. Go to https://github.com/settings/tokens")
    print("2. Click 'Generate new token (classic)'")
    print("3. Select 'repo' scope")
    print("4. Copy the token and add it to your .env file as:")
    print("   GITHUB_TOKEN=your_token_here")
    sys.exit(1)

# Debug: Print token info (first 4 chars only for security)
print(f"Token loaded: {'Yes' if token else 'No'}")
print(f"Token starts with: {token[:4]}...")
print(f"Token length: {len(token)} characters")

# Configure transport with SSL verification and proper headers
transport = AIOHTTPTransport(
    url="https://api.github.com/graphql",
    headers={
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Kubernetes-Issue-Analysis"  # Add User-Agent header
    },
    ssl=True  # Enable SSL verification
)

# Create GraphQL client
client = Client(
    transport=transport,
    fetch_schema_from_transport=True,
    execute_timeout=30  # 30 second timeout
)

# First, let's verify we can access the repository
verify_query = gql("""
query {
  repository(owner: "kubernetes", name: "kubernetes") {
    name
    description
    url
  }
}
""")

async def verify_access():
    """Verify we can access the repository"""
    try:
        print("\n🔍 Verifying repository access...")
        result = await client.execute_async(verify_query)
        repo = result["repository"]
        print(f"✅ Successfully accessed repository:")
        print(f"   Name: {repo['name']}")
        print(f"   URL: {repo['url']}")
        print(f"   Description: {repo['description']}")
        return True
    except Exception as e:
        print(f"\n❌ Repository access failed: {str(e)}")
        return False

# GraphQL query to fetch issues
query = gql("""
query {
  repository(owner: "kubernetes", name: "kubernetes") {
    issues(first: 100, states: OPEN, orderBy: {field: CREATED_AT, direction: DESC}) {
      nodes {
        number
        title
        body
        url
        createdAt
        state
        labels(first: 5) {
          nodes {
            name
          }
        }
        comments(first: 10) {
          nodes {
            body
            createdAt
            author {
              login
            }
          }
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
}
""")

async def fetch_issues():
    """Fetch Kubernetes issues from GitHub"""
    try:
        # First verify we can access the repository
        if not await verify_access():
            print("\n❌ Cannot proceed with fetching issues due to repository access failure")
            sys.exit(1)

        print("\n🔍 Fetching Kubernetes issues from GitHub...")
        result = await client.execute_async(query)
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Save the issues
        issues = result["repository"]["issues"]["nodes"]
        output_file = "data/k8s_issues_sample.json"
        
        with open(output_file, "w") as f:
            json.dump(issues, f, indent=2)
            
        print(f"✅ Successfully fetched {len(issues)} issues")
        print(f"💾 Saved to: {output_file}")
        
        # Print some stats
        total_comments = sum(len(issue["comments"]["nodes"]) for issue in issues)
        print(f"\n📊 Statistics:")
        print(f"   - Total issues: {len(issues)}")
        print(f"   - Total comments: {total_comments}")
        print(f"   - Average comments per issue: {total_comments/len(issues):.1f}")
        
    except Exception as e:
        if "401" in str(e):
            print("\n❌ Authentication failed. Please check your GitHub token:")
            print("1. Make sure your token is valid and not expired")
            print("2. Verify the token has the 'repo' scope")
            print("3. Check that the token is correctly set in your .env file")
            print("\nAdditional troubleshooting:")
            print("- Try accessing https://github.com/kubernetes/kubernetes in your browser")
            print("- Verify your token has not expired")
            print("- Try creating a new token with these scopes:")
            print("  * repo (Full control of private repositories)")
            print("  * read:org (Read organization data)")
            print("  * read:user (Read user data)")
        elif "403" in str(e):
            print("\n❌ Rate limit exceeded. Please try again later or use a different token")
        else:
            print(f"\n❌ Error fetching issues: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(fetch_issues())
