
b = SelectKBest(score_func=mutual_info_regression, k=2)
X_new = b.fit_transform(X, y)
mask = b.get_support()

new_features = [] # The list of your K best features

for bool, feature in zip(mask, X_valid):
    if bool:
        new_features.append(feature)
print(new_features)

#new_features should be the sames as X_new
