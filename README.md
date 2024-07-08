# How to re-create virtual env
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

# Updating env
pip freeze > requirements.txt

# Credentials for scrapping
create cred.py in the root directory
```
REDDIT_CLIENT_ID = ''
REDDIT_CLIENT_SECRET = ''
REDDIT_USER_AGENT = ''
REDDIT_USERNAME = ''
REDDIT_PASSWORD = ''
GITHUB_TOKEN = ''
```