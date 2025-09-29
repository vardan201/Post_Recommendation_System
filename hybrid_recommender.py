import numpy as np
import pandas as pd

# ------------------------------
# 1. Load precomputed matrices
# ------------------------------

# Content-based
similarity_matrix = np.load("similarity_matrix.npy")
user_ids = np.load("user_ids.npy", allow_pickle=True).tolist()
post_ids = np.load("post_ids.npy", allow_pickle=True).tolist()

# Collaborative filtering
cf_score_matrix = np.load("cf_score_matrix.npy")
user_ids_cf = np.load("user_ids_cf.npy", allow_pickle=True).tolist()
post_ids_cf = np.load("post_ids_cf.npy", allow_pickle=True).tolist()

# ------------------------------
# 2. Convert CF scores to DataFrame for easy alignment
# ------------------------------
cf_score_df = pd.DataFrame(cf_score_matrix, index=user_ids_cf, columns=post_ids_cf)

# ------------------------------
# 3. Hybrid Recommendation
# ------------------------------
alpha = 0.5  # weight for content-based
top_k = 3
hybrid_recommendations = {}

for i, user in enumerate(user_ids):
    # Content-based scores
    content_scores = similarity_matrix[i]  # aligned with post_ids

    # CF scores aligned to content post_ids
    if user in cf_score_df.index:
        cf_scores_series = cf_score_df.loc[user]
        cf_scores = cf_scores_series[post_ids].values  # align to content post order
    else:
        cf_scores = np.zeros(len(post_ids))  # new user, no CF

    # Hybrid score
    hybrid_scores = alpha * content_scores + (1 - alpha) * cf_scores

    # Top-k posts
    top_indices = hybrid_scores.argsort()[::-1][:top_k]
    top_posts = [post_ids[j] for j in top_indices]
    hybrid_recommendations[user] = top_posts



    # Example: test for user_id "U17"
test_user = "user_id"

if test_user in hybrid_recommendations:
    print(f"Top {top_k} hybrid recommended posts for {test_user}: {hybrid_recommendations[test_user]}")
else:
    print(f"User {test_user} not found in the dataset.")
