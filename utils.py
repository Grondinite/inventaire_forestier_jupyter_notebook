import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm


def import_tree_dataset(path):

    df_fueillus_original = pd.read_csv(path, sep=';', decimal=',')
    df_feuillus = df_fueillus_original.copy()

    # Find and replace in the `species` column
    df_feuillus['species'] = df_feuillus['species'].str.replace("Érable rouge", "érable_rouge", case=False, regex=False)
    # Find and replace in the `species` column
    df_feuillus['species'] = df_feuillus['species'].str.replace("Épinette noire", "épinette_noire", case=False,
                                                                regex=False)
    # Find and replace in the `species` column
    df_feuillus['species'] = df_feuillus['species'].str.replace("Bouleau blanc", "bouleau_blanc", case=False,
                                                                regex=False)
    # Drop Column: `notes`
    df_feuillus = df_feuillus.drop(columns=['notes'])
    # Remove Na
    df_feuillus.dropna(how="any", inplace=True)
    return df_feuillus


def generate_normal_tree_distribution(df_tree, label=""):
    dph = df_tree["dph"]
    height = df_tree["height"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x_dph = np.linspace(dph.min(), dph.max(), 200)
    mu_dph, std_dph = dph.mean(), dph.std()

    n_dph, bins_dph, _ = axes[0].hist(
        dph,
        bins=10,
        density=False,
        alpha=0.6,
        color="skyblue",
        edgecolor="black"
    )
    bin_width_dph = bins_dph[1] - bins_dph[0]
    axes[0].plot(
        x_dph,
        norm.pdf(x_dph, mu_dph, std_dph) * len(dph) * bin_width_dph,
        "r-",
        linewidth=2
    )
    axes[0].set_title(f"Distribution DHP avec ajustement normal - {label}")
    axes[0].set_xlabel("DHP (cm)")
    axes[0].set_ylabel("Nombre d'arbres")

    x_height = np.linspace(height.min(), height.max(), 200)
    mu_height, std_height = height.mean(), height.std()

    n_height, bins_height, _ = axes[1].hist(
        height,
        bins=10,
        density=False,
        alpha=0.6,
        color="lightgreen",
        edgecolor="black"
    )
    bin_width_height = bins_height[1] - bins_height[0]
    axes[1].plot(
        x_height,
        norm.pdf(x_height, mu_height, std_height) * len(height) * bin_width_height,
        "r-",
        linewidth=2
    )
    axes[1].set_title(f"Distribution hauteur avec ajustement normal - {label}")
    axes[1].set_xlabel("Hauteur (m)")
    axes[1].set_ylabel("Nombre d'arbres")

    plt.tight_layout()
    plt.show()

def generate_biomass_estimation(df_tree):
    df_trees_model_original = pd.read_csv('./data/trees_model_parameters.csv', sep=',', decimal='.')
    df_tree_model = df_trees_model_original.copy()
    df_tree_model = df_tree_model.fillna(0)

    ## TODO: Add SE and CF to estimate
    for index, row in df_tree.iterrows():
        params = df_tree_model.loc[df_tree_model['species'] == row['species']]

        params = params.drop('species', axis=1).set_index('param').T
        params_estimate = params.loc['estimate']

        y_wood = params_estimate['wood1'] * (row['dph'] ** params_estimate['wood2']) * (
                    row['height'] ** params_estimate['wood3'])
        y_bark = params_estimate['bark1'] * (row['dph'] ** params_estimate['bark2']) * (
                    row['height'] ** params_estimate['bark3'])
        y_foliage = params_estimate['foliage1'] * (row['dph'] ** params_estimate['foliage2']) * (
                    row['height'] ** params_estimate['foliage3'])
        y_branches = params_estimate['branches1'] * (row['dph'] ** params_estimate['branches2']) * (
                    row['height'] ** params_estimate['branches3'])
        total_biomass = y_wood + y_bark + y_foliage + y_branches

        # print(index, row, "wood biomass: ", total_biomass, " kg/ha")
        # print(row)
        df_tree.loc[index, 'biomass'] = total_biomass
        # print('----------')

    return df_tree


def get_biomass_summary(df_tree):
    biomass_summary = df_tree['biomass'].agg(['mean', 'median', 'sum', 'count']).to_frame(name='biomass')
    return biomass_summary


def get_biomass_summary_by_species(df_tree):
    biomass_by_species = df_tree.groupby('species')['biomass'].agg(['mean', 'median', 'sum', 'count'])
    return biomass_by_species
