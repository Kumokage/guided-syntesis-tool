import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")

with app.setup:
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from pathlib import Path
    from matplotlib.offsetbox import AnchoredText
    import matplotlib.pyplot as plt
    import xgboost as xgb
    import marimo as mo
    import polars as pl
    import shap
    import numpy as np


@app.cell
def _():
    sheet_name = mo.ui.text(label="Sheet name", placeholder="Experiment_data")
    table_name = mo.ui.text(label="Table name", placeholder="Table1")
    file_browser = mo.ui.file_browser(
        initial_path=Path("./data/"), multiple=False, filetypes=[".xlsx"]
    )

    mo.vstack([sheet_name, table_name, file_browser])
    return file_browser, sheet_name, table_name


@app.cell
def _(file_browser, sheet_name, table_name):
    def read_data():
        is_input_data = (
            sheet_name.value is not None
            and table_name.value is not None
            and len(sheet_name.value) != 0
            and len(table_name.value) != 0
            and len(file_browser.value) != 0
        )
        if is_input_data:
            try:
                data = pl.read_excel(
                    file_browser.value[0].path,
                    table_name=table_name.value,
                    sheet_name=sheet_name.value,
                )
            except ValueError:
                return None

            # TODO: In general application should be moved
            composition_mixtures_dict = {
                "H2O + HNO3 + TiIPO + ButOH": 1,
                "H2O + HNO3 + TiIPO + IPOОН": 2,
                "H2O + HNO3 + ТiBut + ButOH": 3,
                "H2O + TiOSO₄´xH2SO4´yH2O": 4,
            }
            data = data.rename(
                {
                    "t, °С": "T, °С",
                    "с(Ti4+), M": "с(Ti$^{4+}$), M",
                    "Composition mixtures": "MC",
                }
            )
            data = data.with_columns(pl.col("MC").replace(composition_mixtures_dict))
            data = data.with_columns(
                pl.col(pl.String).str.strip_chars().cast(pl.Float64, strict=False),
            )
            data = data.with_columns(
                pl.col("MC").cast(pl.UInt16),
            )
            return data
        else:
            return None

    data = read_data()

    # TODO: Check ValueError
    (
        data
        if data is not None
        else mo.callout(mo.md("### Input data first!"), kind="danger")
    )
    return (data,)


@app.cell
def _(data):
    X = None
    objectives = []

    if data is not None:
        X = data.select(["MC", "T, °С", "t, min", "с(acid), M", "с(Ti$^{4+}$), M"])
        objectives.append(data.select(["Stability of sols, days"]))
        objectives.append(data.select(["d, nm"]))
        objectives.append(data.select(["Contents, %"]))
    return X, objectives


@app.cell
def _(X, availiable_models, objectives):
    select_objective = mo.ui.dropdown(
        options={objective.columns[0]: objective for objective in objectives},
        label="Pick an objective",
    )

    select_models = mo.ui.multiselect(
        options={model.__name__: model for model in availiable_models},
        label="Select model",
    )

    start_training_button = mo.ui.run_button(label="Start learning", kind="success")
    (
        mo.vstack([select_objective, select_models, start_training_button])
        if X is not None
        else None
    )
    return select_models, select_objective, start_training_button


@app.cell
def _(X, select_models, select_objective, start_training_button):
    trained_models = None
    if (
        X is not None
        and select_objective.value is not None
        and len(select_models.value) != 0
        and start_training_button.value
    ):

        trained_models = {}
        for model in mo.status.progress_bar(
            select_models.value,
            title="Learning models",
            subtitle="Please wait",
            show_eta=True,
            show_rate=True,
        ):
            name = f"{select_objective.value.columns[0]}, {model.__name__}"
            trained_models[name] = model(
                X, select_objective.value.to_numpy().ravel(), name
            )
    return (trained_models,)


@app.cell
def _(available_explainers, trained_models):
    select_explainer = mo.ui.dropdown(
        options={explainer.__name__: explainer for explainer in available_explainers},
        value=available_explainers[1].__name__,
        label="Pick an explainer",
    )
    select_explainer if trained_models is not None else None
    return (select_explainer,)


@app.cell
def _(X, draw_plots, select_explainer, trained_models):
    (
        draw_plots(trained_models, X, select_explainer.value)
        if trained_models is not None
        else None
    )
    return


@app.cell
def _():
    def draw_plots(models, X, explainer):
        plt.figure(figsize=(10, 6))

        for index, key in enumerate(models):
            plt.subplot(len(models), 1, index + 1)
            _txt = AnchoredText(
                chr(ord("a") + index),
                loc="upper left",
                pad=0.20,
                borderpad=0,
                prop=dict(fontsize="xx-large"),
            )
            plt.gca().add_artist(_txt)

            explainer(models[key], X, key)
            plt.tick_params(axis="both", labelsize=16)

        plt.subplots_adjust(top=(0.4 + 0.7 * len(models)))
        return plt.gca()

    def explain_shap(model, X, name, color_bar=True):
        explainer = shap.Explainer(model.predict, X.to_pandas())
        shap_values = explainer(X.to_pandas())
        shap.plots.beeswarm(
            shap_values, plot_size=[10, 6], s=26, show=False, color_bar=color_bar
        )
        return plt.gca()

    def explain_importance(model, X, model_name):
        def hex_to_RGB(hex_str):
            """#FFFFFF -> [255,255,255]"""
            return [int(hex_str[i : i + 2], 16) for i in range(1, 6, 2)]

        def get_color_gradient(c1, c2, n):
            """
            Given two hex colors, returns a color gradient
            with n colors.
            """
            assert n > 1
            c1_rgb = np.array(hex_to_RGB(c1)) / 255
            c2_rgb = np.array(hex_to_RGB(c2)) / 255
            mix_pcts = [_x / (n - 1) for _x in range(n)]
            rgb_colors = [(1 - mix) * c1_rgb + mix * c2_rgb for mix in mix_pcts]
            return [
                "#" + "".join([format(int(round(val * 255)), "02x") for val in item])
                for item in rgb_colors
            ]

        features = {}
        columns = X.columns
        importances = model.feature_importances_
        color1 = "#2D466D"
        color2 = "#A2B0C5"
        for i, feature in enumerate(columns):
            features[f"f{i+1}"] = feature

        indices = np.argsort(importances)[::-1]
        num_to_plot = len(columns)
        feature_indices = [ind + 1 for ind in indices[:num_to_plot]]

        print("Feature ranking:")
        for f in range(num_to_plot):
            print(
                "%d. %s %f "
                % (
                    f + 1,
                    features["f" + str(feature_indices[f])],
                    importances[indices[f]],
                )
            )
        bars = plt.bar(
            range(num_to_plot),
            importances[indices[:num_to_plot]],
            color=get_color_gradient(color1, color2, num_to_plot),
            align="center",
        )
        plt.yticks(fontsize=18)
        plt.xlim([-1, num_to_plot])
        plt.legend(
            bars,
            ["".join(features["f" + str(i)]) for i in feature_indices],
            fontsize="16",
        )
        plt.title(f"Feature importance in {model_name}", fontsize=18)
        return plt.gca()

    available_explainers = [explain_shap, explain_importance]
    return available_explainers, draw_plots


@app.cell
def _():
    #  FUnctions for learning models
    def random_foreste(X, y, name):
        parameters = {
            "n_estimators": range(100, 500, 100),
            "max_depth": [None] + list(range(3, 11, 2)),
        }
        model = RandomForestRegressor(random_state=42)
        clf = GridSearchCV(
            model,
            parameters,
            cv=5,
            scoring="neg_mean_absolute_error",
            refit=True,
            n_jobs=-1,
        )
        clf.fit(X, y)
        best_random_forest = clf.best_estimator_
        print(name)
        print(clf.best_score_)
        print(clf.best_params_)
        print()
        return best_random_forest

    def gradient_boosting(X, y, name):
        parameters = {
            "learning_rate": [0.5, 0.25, 0.1, 0.05, 0.01],
            "n_estimators": [4, 8, 16, 32, 64, 128],
            "max_depth": range(1, 18, 2),
        }
        model = GradientBoostingRegressor(random_state=42)
        clf = GridSearchCV(
            model,
            parameters,
            cv=5,
            scoring="neg_mean_absolute_error",
            refit=True,
            n_jobs=-1,
        )
        clf.fit(X, y)
        best_gradient_boost = clf.best_estimator_
        print(name)
        print(clf.best_score_)
        print(clf.best_params_)
        print()
        return best_gradient_boost

    def xgboost(X, y, name):
        parameters = {
            "min_child_weight": [1, 5, 7, 10],
            "gamma": [0.5, 1, 1.5, 2, 2.5],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "max_depth": [3, 4, 5],
        }
        model = xgb.XGBRegressor(
            learning_rate=0.02, n_estimators=600, nthread=1, seed=0
        )
        clf = GridSearchCV(
            model,
            parameters,
            cv=5,
            scoring="neg_mean_absolute_error",
            refit=True,
            n_jobs=-1,
        )
        clf.fit(X, y)
        best_xgboost = clf.best_estimator_
        print(name)
        print(clf.best_score_)
        print(clf.best_params_)
        print()
        return best_xgboost

    availiable_models = [random_foreste, gradient_boosting, xgboost]
    return (availiable_models,)


if __name__ == "__main__":
    app.run()
