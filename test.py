from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


boston = load_boston()

y = boston.target

print(y[:5])

print(sorted(y)[:10])

print(sorted(y)[:-10])