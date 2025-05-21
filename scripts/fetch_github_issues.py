# scripts/fetch_github_issues.py

from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
import os
from dotenv import load_dotenv
import json

load_dotenv()
token = os.getenv("GITHUB_TOKEN")

transport = AIOHTTPTransport(
    url="https://api.github.com/graphql",
    headers={"Authorization": f"Bearer {token}"}
)

client = Client(transport=transport, fetch_schema_from_transport=True)

query = gql("""
query {
  repository(owner: "kubernetes", name: "kubernetes") {
    issues(first: 5, states: OPEN) {
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
        comments(first: 5) {
          nodes {
            body
            createdAt
            author {
              login
            }
          }
        }
      }
    }
  }
}
""")

async def fetch_issues():
    result = await client.execute_async(query)
    issues = result["repository"]["issues"]["nodes"]
    with open("data/k8s_issues_sample.json", "w") as f:
        json.dump(issues, f, indent=2)
    print("Saved 5 sample issues to data/k8s_issues_sample.json")

if __name__ == "__main__":
    import asyncio
    asyncio.run(fetch_issues())
