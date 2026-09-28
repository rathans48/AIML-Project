# config.example.py
# DEPRECATED Python-literal pattern — do not put real secrets here.
# Secrets now live in a git-ignored `.env` file loaded via python-dotenv.
#
# Setup:
#   1. Copy `.env.example` to `.env`
#   2. Fill in your own real API keys/credentials in `.env`
#   3. `config.py` loads them with `load_dotenv()` + `os.environ["NAME"]`
#
# Example `.env` content (see `.env.example` for the canonical template):
#   YOUTUBE_API_KEY=your_youtube_data_api_v3_key_here
#   NEWS_API_KEY=your_newsapi_key_here
#   REDDIT_CLIENT_ID=your_reddit_app_client_id_here
#   REDDIT_CLIENT_SECRET=your_reddit_app_client_secret_here
#   REDDIT_USER_AGENT=your_app_name_here
#   REDDIT_USERNAME=your_reddit_username_here
#   REDDIT_PASSWORD=your_reddit_password_here
