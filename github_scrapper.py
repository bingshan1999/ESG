from github import Github
import csv
from cred import GITHUB_TOKEN

# Replace with your GitHub token
token = GITHUB_TOKEN
repo_name = 'ethereum/go-ethereum'  # Replace with the repository in the format 'owner/repo'
filename = 'data/github_eth.csv'

# Authenticate to GitHub
g = Github(token)

# Get the repository
repo = g.get_repo(repo_name)

# Get all issues (closed, all, open)
issues = repo.get_issues(state='all')

print(issues.totalCount)

# Prepare the CSV file
with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['issue_number', 'issue_title', 'issue_body_and_comments', 'number_of_comments', 'created_at', 'closed_at', 'labels', 'reactions', 'user']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    count = 0
    # Iterate through the grouped issues
    for issue in issues:
        issue_body_and_comments = issue.body if issue.body else ''
        number_of_comments = issue.comments
        # Get comments for each issue
        comments = issue.get_comments()
        for comment in comments:
            issue_body_and_comments += f"\n\nComment by {comment.user.login} at {comment.created_at}:\n{comment.body}"
        
        labels = ', '.join(label.name for label in issue.labels)
        reactions = issue.get_reactions()
        
        issue_data = {
            'issue_number': issue.number,
            'issue_title': issue.title,
            'issue_body_and_comments': issue_body_and_comments,
            'number_of_comments': number_of_comments,
            'created_at': issue.created_at,
            'closed_at': issue.closed_at,
            'labels': labels,
            'reactions': reactions.totalCount,
            'user': issue.user.login
        }

        count += 1
        if count%100==0: print("issue count: ", count)
        writer.writerow(issue_data)

print("Downloaded Github data")