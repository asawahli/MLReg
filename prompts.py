## generate prompts
import base64
import io
import matplotlib.pyplot as plt


def encode_image(
    figs,
):
    if not isinstance(figs, list):
        figs = [figs]
    encoded_list = []
    for fig in figs:
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        base64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
        encoded_list.append(base64_image)
        buf.close()

    return encoded_list


def ai_prompts(inputs, analysis_type, figs=None):
    # Base constraints that apply to all prompts
    formatting_rules = "Write exactly two paragraphs, totaling around 500 words. Be professional, insightful, and avoid using complex technical jargon without a brief explanation."

    if analysis_type == "describe":
        prompt = f"""
            You are a senior data scientist performing exploratory data diagnostics before regression modeling.

            I am providing summary statistics (count, mean, std, min, max, percentiles).

            Your task is NOT only to describe the data, but to generate actionable modeling insights.

            Specifically:

            1. Scale Assessment
            Identify variables with very different magnitudes.
            Quantify the scale differences and recommend whether normalization or standardization is required.
            Explain the risk to regression coefficients if scaling is ignored.

            2. Outlier Risk Analysis
            Identify variables where the gap between the 75th percentile and the max value suggests extreme outliers.
            Estimate the severity (low, moderate, high).
            Recommend specific actions:
            - keep
            - cap (winsorize)
            - transform (log/sqrt)
            - remove

            3. Distribution Shape Diagnosis
            Determine which variables are likely skewed.
            Explain how skewness could affect:
            - regression stability
            - coefficient interpretation
            - residual behavior

            4. Priority Ranking
            Rank the top 3 variables that pose the greatest risk to modeling performance.
            Justify each ranking.

            5. Recommended Next Steps
            Suggest specific preprocessing actions before running regression.

            Write exactly two paragraphs totaling around 500 words.
            Focus on practical modeling decisions rather than textbook definitions.

            Here is the data:
            {inputs}
            """
        content = [{"type": "text", "text": prompt}]

    elif analysis_type == "corr":
        prompt = f"""
            You are a senior data scientist evaluating feature relationships before building a regression model.

            I am providing a correlation matrix.

            Your objective is to identify modeling risks and feature selection opportunities.

            Specifically:

            1. Target Relationships
            Identify the top 5 strongest predictors of the target variable.
            Rank them by absolute correlation strength.
            Explain whether each relationship is positive or negative and what that means practically.

            2. Multicollinearity Detection
            Identify pairs of independent variables with correlations greater than 0.8 (or less than -0.8).
            Explain why these relationships create instability in regression models.
            State which variables should potentially be removed or combined.

            3. Feature Redundancy Analysis
            Determine whether groups of features carry similar information.
            Recommend dimensionality reduction strategies if necessary:
            - feature removal
            - PCA
            - regularization methods

            4. Modeling Risk Summary
            Identify the biggest correlation-related risk to model reliability.

            5. Recommended Feature Strategy
            Provide a clear plan:
            - Keep
            - Remove
            - Combine
            - Transform

            Write exactly two paragraphs totaling around 500 words.
            Focus on decisions that affect model performance.

            Here is the data:
            {inputs}
            """
        content = [{"type": "text", "text": prompt}]

    elif analysis_type == "residuals":
        prompt = """
            You are a senior data scientist interpreting regression diagnostic figures.

            I am providing residual diagnostic plots. Your responsibility is to explain what is visually happening in EACH figure before making modeling judgments.

            The figures include:
            - Actual vs Predicted
            - Residuals vs Fitted
            - Residual Histogram
            - Q-Q Plot

            Your analysis MUST follow this structure:

            1. Actual vs Predicted Plot (Visual Explanation Required)
            Describe:
            - Whether points lie tightly along the diagonal or scatter widely
            - Whether prediction accuracy changes at low vs high values
            - Whether systematic under- or over-prediction is visible
            Explain what these patterns mean for model accuracy.

            2. Residuals vs Fitted Plot (Visual Explanation Required)
            Describe:
            - The visible shape of the residual cloud (random, curved, funnel-shaped, clustered)
            - Whether residual spread increases with prediction magnitude
            - Whether patterns or curvature exist
            Explain what this implies about:
            - homoscedasticity
            - missing nonlinear relationships

            3. Residual Histogram (Visual Explanation Required)
            Describe:
            - Whether the histogram is symmetric or skewed
            - Whether extreme tails exist
            - Whether residuals cluster around zero
            Explain what this indicates about:
            - prediction bias
            - error stability

            4. Q-Q Plot (Visual Explanation Required)
            Describe:
            - Whether points follow the straight reference line
            - Where deviations occur (tails, center, both)
            Explain whether residuals are normally distributed.

            5. Model Health Verdict
            Choose one:
            - Reliable
            - Usable with adjustments
            - Not reliable

            Explain your reasoning based on the visual evidence.

            6. Recommended Fixes
            Suggest specific modeling improvements based on what is seen in the figures.

            Write each point in one paragraphs each paragraph around 150 words. no introduction of yourself required

            IMPORTANT:
            Do not jump directly to conclusions.
            First describe what is visually observable in the figures.
            Then interpret the statistical meaning.
            """

        content = [{"type": "text", "text": prompt}]
        try:
            base64_strings = encode_image(figs)
            for b64 in base64_strings:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
        except Exception as e:
            return e

    elif analysis_type == "summary":
        prompt = f"""
            You are a senior data analyst evaluating multiple regression models using numerical performance metrics.

            I am providing regression performance metrics for both training and testing datasets.

            Metrics include:
            - R² (coefficient of determination)
            - MSE (mean squared error)
            - RMSE (root mean squared error)
            - MAE (mean absolute error)

            Your task is to compare the models strictly based on statistical performance.
            Do not refer to stakeholders, deployment, business impact, or real-world users.
            Focus only on numerical behavior and model generalization.

            Perform the following analysis:

            1. Model Ranking Based on Test Metrics
            Rank all models using Test-set performance.
            Use primarily:
            - Highest Test R²
            - Lowest Test RMSE
            - Lowest Test MAE

            Explain clearly which model shows the strongest predictive performance based on these values.

            2. Generalization Behavior
            Compare Train and Test metrics for each model.
            Identify signs of:

            Overfitting:
            - Train performance much better than Test performance

            Underfitting:
            - Both Train and Test performance weak

            Well-Generalized Model:
            - Train and Test metrics close to each other

            Quantify the differences where possible.

            3. Error Magnitude Analysis
            Interpret RMSE and MAE numerically.
            Estimate the typical prediction error magnitude using MAE.
            Discuss whether RMSE is significantly larger than MAE, which may indicate large residual errors.

            4. Stability Assessment
            Identify which model shows the most consistent behavior between Train and Test datasets.
            Explain which models exhibit unstable or inconsistent performance.

            5. Final Statistical Conclusion
            State which model demonstrates the most favorable statistical performance overall.
            Justify this conclusion strictly using numerical comparisons between metrics.

            Write exactly two paragraphs totaling around 500 words.

            IMPORTANT:
            Base all conclusions only on numerical values.
            Avoid references to real-world users, deployment, or decision-making contexts.
            Use metric comparisons as the primary evidence.

            Here is the training data:
            {inputs[0]}

            Here is the testing data:
            {inputs[1]}
            """
        content = [{"type": "text", "text": prompt}]

    else:
        return "Error: Invalid analysis type provided."

    # Here you would call your LLM API (e.g., OpenAI, Gemini, Claude)
    # response = llm_client.generate(prompt)
    # return response

    return content  # Returning the prompt for testing purposes
