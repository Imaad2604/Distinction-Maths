import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds


def build_toy_ratings():
    """6 users × 8 movies. Ratings 1-5, 0 means unrated."""
    rows = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5]
    cols = [0, 1, 2, 0, 1, 3, 2, 4, 5, 3, 4, 0, 5, 1]
    data = [5, 4, 3, 5, 4, 2, 4, 5, 3, 4, 5, 3, 4, 2]
    return sp.csr_matrix((data, (rows, cols)), shape=(6, 8))


def item_item_recommend(R, target_user, k=5):
    """Predict ratings for a user using item-item similarity."""
    S = (R.T @ R).tocsr()
    user_ratings = R[target_user, :].toarray().ravel()
    rated_items = np.where(user_ratings > 0)[0]
    predictions = np.zeros(R.shape[1])
    for i in range(R.shape[1]):
        score = 0.0
        for j in rated_items:
            score += S[i, j] * user_ratings[j]
        predictions[i] = score
    predictions[rated_items] = 0
    return predictions


def svd_recommend(R, target_user, k=2):
    """Predict ratings using truncated SVD."""
    U, sigma, Vt = svds(R.astype(float), k=k)
    predictions = (U[target_user, :] * sigma) @ Vt
    user_ratings = R[target_user, :].toarray().ravel()
    predictions[user_ratings > 0] = 0
    return predictions


R = build_toy_ratings()
print("Ratings matrix (0 = unrated):")
print(R.toarray())
print()

preds_ii = item_item_recommend(R, target_user=0)
preds_svd = svd_recommend(R, target_user=0, k=2)

print("Top-3 recommendations for user 0:")
print("Item-item:")
for i in np.argsort(preds_ii)[::-1][:3]:
    print(f"  item {i}: score {preds_ii[i]:.3f}")

print("SVD:")
for i in np.argsort(preds_svd)[::-1][:3]:
    print(f"  item {i}: score {preds_svd[i]:.3f}")