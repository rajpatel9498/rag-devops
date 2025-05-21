# scripts/preprocess_issues.py

import json
import os

input_path = "data/k8s_issues_sample.json"
output_path = "data/k8s_issues_preprocessed.json"

def preprocess_issue(issue):
    title = issue.get("title", "").strip()
    body = issue.get("body", "").strip()
    comments = issue.get("comments", {}).get("nodes", [])
    
    comment_texts = [
        comment.get("body", "").strip()
        for comment in comments
        if comment.get("body")
    ]
    
    full_text = "\n\n".join([title, body] + comment_texts)
    
    return {
        "number": issue.get("number"),
        "url": issue.get("url"),
        "text": full_text
    }

def main():
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return

    with open(input_path, "r") as infile:
        issues = json.load(infile)

    processed = [preprocess_issue(issue) for issue in issues]

    with open(output_path, "w") as outfile:
        json.dump(processed, outfile, indent=2)

    print(f"✅ Preprocessed {len(processed)} issues")
    print(f"💾 Saved to: {output_path}")

if __name__ == "__main__":
    main()