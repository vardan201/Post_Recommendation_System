📖 Hybrid Recommendation System (Content-Based + Collaborative Filtering)
🚀 Project Overview

This project implements a Hybrid Recommendation System that combines the strengths of:

Content-Based Filtering (CBF): Recommends posts based on similarity between user embeddings and post embeddings.

Collaborative Filtering (CF): Recommends posts based on engagement patterns of similar users.

The hybrid system balances both approaches to provide personalized and diverse recommendations for each user.

📂 Dataset

We use two datasets:

Posts Dataset
Each post has features (e.g., embeddings, text, or metadata).
Example:

post_id, title, embedding_vector
P1, "Post about AI", [...]
P2, "Post about Sports", [...]


Engagements Dataset
Captures user-post interactions.
Example:

user_id, post_id, engagement
U1, P52, 1
U1, P44, 0
U1, P1, 1
U2, P4, 1
...


1 → engaged (liked/clicked)

⚙️ Workflow
1️⃣ Content-Based Filtering

Represent users as vectors (e.g., average of engaged post embeddings).

Represent posts as vectors (precomputed embeddings).

Compute cosine similarity between user and post vectors:

similarity_matrix = cosine_similarity(user_vectors, post_vectors)


Save outputs:

np.save("similarity_matrix.npy", similarity_matrix)
np.save("user_ids.npy", np.array(user_ids))
np.save("post_ids.npy", np.array(post_ids))

2️⃣ Collaborative Filtering

Construct user-post engagement matrix:

user_post_matrix = engagements_df.pivot(
    index="user_id", columns="post_id", values="engagement"
).fillna(0)


Apply matrix factorization (SVD):

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50)
user_factors = svd.fit_transform(user_post_matrix)
post_factors = svd.components_.T
predicted_scores = np.dot(user_factors, post_factors.T)


Save outputs:

np.save("cf_score_matrix.npy", predicted_scores)
np.save("user_ids_cf.npy", np.array(user_ids_cf))
np.save("post_ids_cf.npy", np.array(post_ids_cf))

3️⃣ Hybrid Recommender

Load both models’ results:

similarity_matrix = np.load("similarity_matrix.npy")
cf_score_matrix = np.load("cf_score_matrix.npy")


Align CF scores with CBF posts.

Compute hybrid score:

hybrid_scores = alpha * content_scores + (1 - alpha) * cf_scores


Get top-k recommendations for each user.

🧪 Example Usage
# Run hybrid recommender
alpha = 0.5  # balance: 0=only CF, 1=only CBF
top_k = 3

# Example test for user "U17"
test_user = "U17"
if test_user in hybrid_recommendations:
    print(f"Top {top_k} hybrid recommended posts for {test_user}: {hybrid_recommendations[test_user]}")


✅ Output:

Top 3 hybrid recommended posts for U17: ['P12', 'P47', 'P88']

⚡ Features

Handles new users (falls back to content-based).

Handles new posts (falls back to collaborative filtering).

Adjustable alpha for tuning the contribution of each method.

Modular: can extend with deep learning models later.


🎯 Future Improvements

Use Neural Collaborative Filtering (NCF) for deeper CF.

Use BERT embeddings for post content.

Add Streamlit/Flask app to serve recommendations interactively.  

0 → not engaged
