import marimo

__generated_with = "0.16.5"
app = marimo.App()


@app.cell
def _():
    # (use marimo's built-in package management features instead) !pip install plotly==5.14.1
    # (use marimo's built-in package management features instead) !pip install "jupyterlab>=3" "ipywidgets>=7.6"
    # (use marimo's built-in package management features instead) !pip install jupyter-dash
    # (use marimo's built-in package management features instead) !pip install -U kaleido
    # (use marimo's built-in package management features instead) !pip install opentsne
    # (use marimo's built-in package management features instead) !pip install nbformat
    # (use marimo's built-in package management features instead) !pip install pandas
    # (use marimo's built-in package management features instead) !pip install seaborn
    # (use marimo's built-in package management features instead) !pip install openpyxl
    # (use marimo's built-in package management features instead) !pip install matplotlib
    # (use marimo's built-in package management features instead) !pip install scikit-learn
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import openpyxl as xl
    import FILibExcel
    import seaborn as sns
    import matplotlib.pyplot as plt
    return FILibExcel, np, plt, sns


@app.cell
def _():
    excel_path = r"../../static/mock/titanium sols.xlsx"
    artifacts_path = r"../../static/mock/soles_artifacts/first_dataset/"
    return artifacts_path, excel_path


@app.cell
def _(FILibExcel, excel_path):
    tables_dict = FILibExcel.get_all_tables(file_name=excel_path)
    df = tables_dict["Table1"]['dataframe']
    df = df.fillna(value=0)
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.drop('Composition mixtures', axis=1).astype('float').describe().T
    return


@app.cell
def _(df, plt, sns):
    _fig, _axes = plt.subplots(ncols=2, nrows=4)
    _fig.set_size_inches(20, 35)
    for _col, _ax in zip(df.drop('Composition mixtures', axis=1), _axes.flat):
        sns.histplot(df[_col].astype('float'), ax=_ax).set(title=f'{_col} distribution')
    _fig.subplots_adjust(hspace=0.35)
    plt.show()
    return


@app.cell
def _(df):
    df_clear =  df.drop('Composition mixtures', axis=1)
    df_clear = df_clear.astype('float')
    return (df_clear,)


@app.cell
def _(df_clear):
    df_clear.columns
    return


@app.cell
def _(df_clear):
    df_plot = df_clear.copy(deep=True)

    def replace_func_days(x):
        if x < 20:
            return "< 20"
        else:
            return str(x)

    df_plot['Stability of sols, days'] = df_plot['Stability of sols, days'].apply(replace_func_days)
    return (df_plot,)


@app.cell
def _():
    25 /3
    return


@app.cell
def _(artifacts_path, df_plot, plt, sns):
    _fig, _axes = plt.subplots(ncols=2, nrows=2)
    _fig.set_size_inches(20, 17)
    for _col, _ax in zip(['t, °С', 't, min', 'с(acid), mol/l', 'с(Ti4+), mol/l'], _axes.flat):
        sns.histplot(df_plot, x=_col, hue='Stability of sols, days', hue_order=['90.0', '60.0', '40.0', '< 20'], multiple='stack', palette='mako', ax=_ax).set(title=f'{_col} distribution')
    _fig.subplots_adjust(hspace=0.15)
    plt.savefig(artifacts_path + 'distibution_with_stability_days.png', format='png', dpi=600)
    plt.show()
    return


@app.cell
def _(artifacts_path, df_plot, plt, sns):
    _fig, _axes = plt.subplots(ncols=2, nrows=1)
    _fig.set_size_inches(20, 5)
    for _col, _ax in zip(['t, °С', 't, min'], _axes.flat):
        sns.histplot(df_plot, x=_col, hue='Stability of sols, days', multiple='stack', palette='mako', binwidth=1, ax=_ax).set(title=f'{_col} distribution')
    plt.savefig(artifacts_path + 'distibution_with_stability_days_bin3.pdf', format='pdf', dpi=600)
    plt.show()
    return


@app.cell
def _(artifacts_path, df_plot, plt, sns):
    _fig, _axes = plt.subplots(ncols=2, nrows=1)
    _fig.set_size_inches(20, 5)
    for _col, _ax in zip(['с(acid), mol/l', 'ultrasound '], _axes.flat):
        sns.histplot(df_plot, x=_col, hue='Stability of sols, days', multiple='stack', palette='mako', ax=_ax, binwidth=0.05).set(title=f'{_col} distribution')
    plt.savefig(artifacts_path + 'distibution_with_stability_days_bin005.pdf', format='pdf', dpi=600)
    plt.show()
    return


@app.cell
def _(artifacts_path, df_plot, plt, sns):
    _col = 'с(Ti4+), mol/l'
    sns.histplot(df_plot, x=_col, hue='Stability of sols, days', multiple='stack', palette='mako', binwidth=0.01).set(title=f'{_col} distribution')
    plt.savefig(artifacts_path + 'distibution_with_stability_days_bin001.pdf', format='pdf', dpi=600)
    return


@app.cell
def _(df_clear):
    from sklearn.preprocessing import StandardScaler
    sca = StandardScaler()
    sca.fit(df_clear)
    df_standardized = sca.transform(df_clear)
    df_standardized.shape
    return (df_standardized,)


@app.cell
def _(df_standardized):
    from sklearn.decomposition import PCA
    pca = PCA(5)
    X_pca5 = pca.fit_transform(df_standardized)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.sum())
    return X_pca5, pca


@app.cell
def _(df_clear, np, pca):
    _components = dict(zip(df_clear.columns, pca.components_[0]))
    _components = sorted(_components.items(), key=lambda x: x[1], reverse=True)
    for _c, _w in _components:
        print(_c, np.round(_w, 3))
    return


@app.cell
def _(df_clear, np, pca):
    _components = dict(zip(df_clear.columns, pca.components_[1]))
    _components = sorted(_components.items(), key=lambda x: x[1], reverse=True)
    for _c, _w in _components:
        print(_c, np.round(_w, 3))
    return


@app.cell
def _(X_pca5):
    from openTSNE import TSNE

    embedding = TSNE(perplexity=23.33).fit(X_pca5)
    return (embedding,)


@app.cell
def _(df_clear, embedding):
    x_plot = df_clear.copy(deep=True)
    x_plot['tsne1'] = embedding[:, 0]
    x_plot['tsne2'] = embedding[:, 1]
    return (x_plot,)


@app.cell
def _(x_plot):
    x_plot.head()
    return


@app.cell
def _(x_plot):
    x_plot_1 = x_plot.astype('float')
    return (x_plot_1,)


@app.cell
def _(artifacts_path, df_clear, x_plot_1):
    import plotly.express as px
    _fig = px.scatter(x_plot_1, x='tsne1', y='tsne2', color='Stability of sols, days', hover_data=df_clear.columns, width=900, height=600)
    _fig.update_traces(marker=dict(size=8, line=dict(width=0.5)), selector=dict(mode='markers'))
    _fig.update_layout(title='TSNE plot of sols stability', xaxis_title='x', yaxis_title='y')
    _fig.write_html(artifacts_path + 'TSNE_plot_stability_days.html')
    _fig.show()
    return (px,)


@app.cell
def _(artifacts_path, df_clear, px, x_plot_1):
    _fig = px.scatter(x_plot_1, x='tsne1', y='tsne2', color='d, nm', hover_data=df_clear.columns, width=900, height=600)
    _fig.update_traces(marker=dict(size=8, line=dict(width=0.5)), selector=dict(mode='markers'))
    _fig.update_layout(title='TSNE plot of sols stability', xaxis_title='x', yaxis_title='y')
    _fig.write_html(artifacts_path + 'TSNE_plot_d.html')
    _fig.show()
    return


if __name__ == "__main__":
    app.run()
