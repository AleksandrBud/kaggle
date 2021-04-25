from pandas import DataFrame, Series
from numpy import zeros
from numpy.random   import permutation
from tqdm import tqdm


def calc_mean_scoring_rating(raiting_1: float, raiting_2: float, raiting_3: float) -> float:
    count_rating = 3
    if raiting_1 == 0:
        count_rating = count_rating - 1
    if raiting_2 == 0:
        count_rating = count_rating - 1
    if raiting_3 == 0:
        count_rating = count_rating - 1

    rating = 0.5
    if count_rating != 0:
        rating = round((raiting_1 + raiting_2 + raiting_3) / count_rating, 3)

    return rating


def merge_new_feature(main_df: DataFrame, tmp_df: DataFrame, col_name: str) -> DataFrame:
    main_df = main_df.merge(DataFrame(tmp_df.values, index=tmp_df.index, columns=[col_name]),
                            on='APPLICATION_NUMBER',
                            how='left')
    return main_df


def custom_one_hot_encoding(i_df_main, i_df, columns, join_index):
    for column in columns:
        for val in i_df[column].unique():
            i_df_main[f'EXISTS_{val}'] = i_df_main[join_index].isin(i_df.loc[i_df[column] == val,
                                                                             join_index].values
                                                                    ).astype(int)

    return i_df_main


def calc_amount_annuity(amount_credit: float) -> float:
    return amount_credit * 0.0142 * (1 + 0.0142) ** 36 / ((1 + 0.0142) ** 36 - 1)


def create_std_features(i_df: DataFrame, grouped_column: str):
    stats = ["count", "mean", "min", "max", "std"]
    groped_df = i_df.groupby(grouped_column).agg(stats).reset_index()
    col = []
    for column in groped_df.columns:
        if column[0] == grouped_column:
            col.append(column[0])
        else:
            col.append(f'{column[0]}_{column[1]}')
    groped_df = DataFrame(groped_df.values, index=groped_df.index, columns=col)
    return groped_df


def calc_permutation_importance(estimator,
                                metric: callable,
                                x_valid: DataFrame,
                                y_valid: Series
                                ) -> Series:
    scores = {}
    y_pred = estimator.predict_proba(x_valid)[:, 1]
    base_score = metric(y_valid, y_pred)
    for feature in tqdm(x_valid.columns):
        x_val_copy = x_valid.copy()
        x_val_copy[feature] = permutation(x_val_copy[feature])

        y_pred = estimator.predict_proba(x_val_copy)[:, 1]
        score = metric(y_valid, y_pred)
        scores[feature] = base_score - score

    scores = Series(scores)
    scores = scores.sort_values(ascending=False)

    return scores


def validation(x, y, cv, model, params, metric, categorical=None, features=[]):
    estimators, fold_scores = [], []
    oof_preds = zeros(x.shape[0])
    calc_p_imp = False

    if not features:
        calc_p_imp = True

    x[categorical] = x[categorical].astype(str)
    for fold, (train_idx, valid_idx) in enumerate(cv.split(x, y)):
        x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        if not calc_p_imp:
            model_feat = features[fold][features[fold] > 0].index.tolist()
            x_train = x_train[model_feat]
            x_valid = x_valid[model_feat]

        internal_model = model(**params)

        ctg = []
        for i in categorical:
            if i in x_train.columns:
                ctg.append(i)
        internal_model.fit(x_train, y_train, ctg,
                           eval_set=[(x_train, y_train), (x_valid, y_valid)]
                           )
        oof_preds[valid_idx] = internal_model.predict_proba(x_valid)[:, 1]
        score = metric(y_valid, oof_preds[valid_idx])
        print(f'Fold {fold}, Valid score = {round(score, 5)}')
        fold_scores.append(round(score, 5))
        estimators.append(internal_model)
        if calc_p_imp:
            feat = calc_permutation_importance(internal_model, metric, x_valid, y_valid)
            features.append(feat)
    print(f'Score by each fold: {fold_scores}')
    print(f'Mean scores {Series(fold_scores).mean()}')
    print('=' * 65)
    return estimators, oof_preds, features
