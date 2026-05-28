import shap
import matplotlib.pyplot as plt


def explain(type: str, model, masker):
    """ """

    if type == "linear":
        explainer = shap.LinearExplainer(model, masker)
        shap_values = explainer(masker)

        return shap_values

    if type == "tree":
        explainer = shap.TreeExplainer(model, masker)
        raise Exception("Not supported")
    if type == "kernal":
        raise Exception("Not supported")
    if type == "deep":
        raise Exception("Not supported")


def plot_beeswarm(shap_values):
    fig, ax = plt.subplots()
    shap.plots.beeswarm(shap_values, ax=ax, plot_size=None)

    return fig
