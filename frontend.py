import streamlit as st
import pandas as pd
import requests
import io
import base64

# Function to add background image
def add_background_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{image_url}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to fetch predictions from your FastAPI backend
def get_predictions(file):
    files = {'file': file.getvalue()}
    # Replace the URL with your FastAPI URL for prediction endpoint
    response = requests.post("http://localhost:8000/predict/", files=files)
    if response.status_code == 200:
        return pd.read_csv(io.StringIO(response.text))
    return None

# Function to fetch the plot from your FastAPI backend
def get_plot(file):
    files = {'file': file.getvalue()}
    # Replace the URL with your FastAPI URL for plot endpoint
    response = requests.post("http://localhost:8000/plot/", files=files)
    if response.status_code == 200:
        return response.content
    return None

# Streamlit app configuration
st.set_page_config(page_title="Dev's Heart Complexity Predictor 🫀", page_icon="🫀")

# Add background image (Replace URL with your image URL)
background_image_url = "https://static.vecteezy.com/system/resources/previews/036/752/971/non_2x/ai-generated-a-heart-shape-with-the-words-free-love-wallpapers-free-photo.jpg"
add_background_image(background_image_url)

# Main App Title
st.title("Dev's Heart Complexity Predictor 🫀")

# Add instructions container
st.markdown(
    """
    <div style="background-color: black; padding: 20px; color: white; border-radius: 10px;">
        <h3>Instructions for the Uploaded CSV File:</h3>
        <p>The CSV file should contain the following columns:</p>
        <ul>
            <li><strong>Age</strong> - The age of the individual.</li>
            <li><strong>Sex</strong> - The sex of the individual (1 for male, 0 for female).</li>
            <li><strong>Chest_pain</strong> - The type of chest pain.</li>
            <li><strong>trestbps</strong> - Resting blood pressure.</li>
            <li><strong>cholestrol</strong> - Serum cholesterol.</li>
            <li><strong>fbs</strong> - Fasting blood sugar.</li>
            <li><strong>restecg</strong> - Resting electrocardiographic results.</li>
            <li><strong>thalach</strong> - Maximum heart rate achieved.</li>
            <li><strong>exang</strong> - Exercise induced angina.</li>
            <li><strong>oldpeak</strong> - ST depression induced by exercise.</li>
            <li><strong>slope</strong> - Slope of the peak exercise ST segment.</li>
            <li><strong>ca</strong> - Number of major vessels colored by fluoroscopy.</li>
            <li><strong>thal</strong> - Thalassemia.</li>
        </ul>
        <p>Please ensure that the file is formatted correctly before uploading.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# File Upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    # Predict endpoint
    with st.spinner("Generating predictions..."):
        predicted_df = get_predictions(uploaded_file)
        if predicted_df is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Original CSV")
                uploaded_df = pd.read_csv(uploaded_file)
                st.dataframe(uploaded_df)
            with col2:
                st.write("### Predicted CSV")
                st.dataframe(predicted_df)
        else:
            st.error("Failed to generate predictions. Please try again.")

    # Plot button and display
    if st.button("Plot the Linear Regression"):
        with st.spinner("Generating plot..."):
            plot_content = get_plot(io.BytesIO(predicted_df.to_csv(index=False).encode()))
            if plot_content:
                # Encode the plot_content (binary data) to base64
                base64_pdf = base64.b64encode(plot_content).decode('utf-8')  # Correct encoding to base64
                
                # Display the PDF in an iframe
                pdf_viewer = f"""
                <iframe src="data:application/pdf;base64,{base64_pdf}" 
                        width="100%" height="600px" style="border: none;"></iframe>
                """
                st.markdown(pdf_viewer, unsafe_allow_html=True)

                # Provide a download button for the PDF
                st.download_button(
                    label="Download PDF",
                    data=plot_content,
                    file_name="linear_regression_plot.pdf",
                    mime="application/pdf"
                )
            else:
                st.error("Failed to generate the plot. Please try again.")
