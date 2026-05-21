import streamlit as st
import pandas as pd

# Page Configuration
st.set_page_config(page_title="Guidelines", layout="wide")

st.title("App Guide & Data Requirements")

with st.expander("1. Dataset Formatting Guidelines", expanded=False):
    # st.subheader("1. Input File Requirements")
    st.markdown(
        """
    To ensure the application processes your data smoothly, your uploaded file must follow these rules:

    * **File Format:** The dataset must be a standard **CSV** (`.csv`) file.
    * **Column Headers:** The **first row** of your file must contain the names of your columns. The app will use these headers as labels in all exploratory graphs, feature lists, and model summaries.
    * **Numerical Data Only:** The model training process currently only accepts numerical data (numbers, floats, decimals). Ensure your features are numeric. 
    * **Example of csv file format:**          
                """,
        unsafe_allow_html=True,
    )
    _, col, _ = st.columns([1, 2, 1])
    col.table(
        {
            "feature1": [1.5, 2.1, 0.9],
            "feature2": [3.2, 1.4, 2.8],
            "feature3": [0.8, 4.5, 3.3],
            "target": [10.0, 15.0, 8.0],
        },
        border=True,
    )

with st.expander("2. Target Variable Selection", expanded=False):
    st.markdown(
        """
    * The target variable is the specific column in your dataset that you want the model to predict. 
    * By default, the application assumes that the target variable is the **last column** in your CSV file. 
    * if your target variable is not the last column, you can select it **manually** from a dropdown menu during the preprocessing step.
    """,
        unsafe_allow_html=True,
    )

with st.expander("3. Recommended Workflow", expanded=False):
    st.markdown(
        """
    For the best results, navigate through the app in this order:
    1. **Home:** Upload your `.csv` file or load a demo dataset.
    2. **Exploratory Data Analysis (EDA):** Visualize your data and check for patterns or missing values.
    3. **Preprocessing:** Handle missing data, scale your numbers, and confirm your target variable.
    4. **Train Models:** Select your algorithm, train the model, store model and download the resulting file.
    5. **Model Summary:** Review the performance metrics of your trained models and manage stored models.
    6. **Predict:** Predict based on your trained model or Upload your trained model to make predictions on new data.
    
    :red[** if you need AI assistance, configure the AI assistant in the settings.]
    """,
        unsafe_allow_html=True,
    )

with st.expander("4. AI Assistance Settings", expanded=False):
    st.markdown(
        """
    The application includes an AI assistant to help you with data analysis and model interpretation. To use this feature, you need to configure the AI assistant in the **settings**.
    
    The app support gemini and logfare.ai. You can choose your preferred provider and input your API key to enable AI support across the app. Once configured, you can use AI assistance button through out the app.
    """,
        unsafe_allow_html=True,
    )
    with st.expander("logfare.ai settings", expanded=False):
        st.markdown("""
        To use logfare.ai as your AI assistant provider, follow these steps:
        * Sign up for an account at [logfare.ai](https://logfare.ai/) and obtain your API key.
        * In the app settings, select logfare.ai as your provider and enter your API key in the designated field.
        * click the "Save Settings" button to enable AI support. You can now use the AI assistance features throughout the app for data analysis and model interpretation.         
                    """)
        st.markdown("""
        in Dashboard, you can find the API, copy to MLreg app settings.
                    

        """)
        st.image(
            "assets/logfare_api.jpg",
            clamp=True,
        )

        st.markdown("""
        by default, logfare.ai provides a free tier with limited models.
        To access more powerful models, you can subscribe to the premium tier. select **opt in premium tier** in Logfare.ai Dashboard.
        The premium tier is free. however, the data shared with logfare.ai may sell the data shared with logfare.ai.
""")
        st.image(
            "assets/logfare_premium.jpg",
            clamp=True,
        )
    with st.expander("gemini settings", expanded=False):
        st.markdown("""
        To use Gemini as your AI assistant provider, follow these steps:
        * Sign up for an account at [Google AI Studio](https://aistudio.google.com/).
        * from sidemenu select **Get API Key**
        * Create a new API key and copy it to ML-reg app settings.
        * List of models will be retrieved from google API key. \\
        :red[**Note:** Gemini API may have usage limits and costs associated with it, so be sure to review the pricing details on the Google AI Studio website.]
        * In the MLreg app settings, select Gemini as your provider and enter your API key in the designated field.
        * click the "Save Settings" button to enable AI support. You can now use the AI assistance features throughout the app for data analysis and model interpretation.
                        """)


@st.dialog("Demo Datasets")
def load_demo_data(n):
    st.session_state.df = pd.read_csv(f"demos/demo{n}.csv")
    st.success(
        "Demo dataset loaded successfully! you can now navigate to **Explore Data** and explore the data and train models."
    )


with st.expander("5. Demos", expanded=False):
    st.write("Here are some example datasets you can use to test the application:")
    st.markdown("""
Select a pre-loaded sample dataset below to test the application features instantly without uploading your own file.
""")

    st.button(
        "Load demo dataset 1",
        on_click=lambda: load_demo_data(1),
    )
    st.button("Load demo dataset 2", on_click=lambda: load_demo_data(2))
# Optional: Add a friendly button to jump back home
st.write("---")
if st.button("Ready? Go back to Home"):
    st.switch_page("pages/0_home.py")
