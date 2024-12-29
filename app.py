import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import BytesIO

# Load model and vectorizer
model = joblib.load('lRM.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# App title and configuration
st.set_page_config(page_title="AIU Sentiment Analysis", page_icon=":bar_chart:", layout="wide")

# Sidebar with logo
st.sidebar.image("/Users/ismailyousif/Documents/Sem1Y3/Natural Language Processing/project/AIU.png", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction"])

# Home Page
if page == "Home":
    st.title("AIU Sentiment Analysis Project")
    st.image("/Users/ismailyousif/Documents/Sem1Y3/Natural Language Processing/project/AIU.png", width=200)
    st.subheader("Objective")
    st.write(
        "This project focuses on sentiment analysis of comments or text data related to various topics. "
        "Users can input individual comments or upload files for batch predictions. The system utilizes "
        "machine learning models to classify sentiments into Positive, Neutral, or Negative categories. "
        "The application is designed for efficiency and accuracy in sentiment predictions."
    )
    st.subheader("Features")
    st.write(
        "- Predict sentiment for individual comments.\n"
        "- Upload a batch of comments in an XLSX file for sentiment predictions.\n"
        "- Visualize the sentiment distribution with interactive charts.\n"
        "- Powered by AIU and advanced Natural Language Processing techniques."
    )

# Prediction Page
elif page == "Prediction":
    st.title("Prediction Page")

    # User input for single comment
    st.subheader("Predict Sentiment for a Single Comment")
    user_input = st.text_area("Enter a comment:", "")

    if st.button("Predict Sentiment"):
        if user_input.strip():
            input_vectorized = vectorizer.transform([user_input])
            prediction = model.predict(input_vectorized)[0]

            st.success(f"Predicted Sentiment: **{prediction}**")

            # Generate bar chart for single comment
            st.subheader("Sentiment Distribution")
            counts = {"Positive": 0, "Neutral": 0, "Negative": 0}

            # Update the counts based on the prediction
            if prediction == "Positive":
                counts["Positive"] = 1
            elif prediction == "Neutral":
                counts["Neutral"] = 1
            elif prediction == "Negative":
                counts["Negative"] = 1

            # Debugging counts
            st.write("Debugging Counts:", counts)

            if sum(counts.values()) > 0:
                plt.figure(figsize=(6, 4))
                plt.bar(counts.keys(), counts.values(), color=["green", "blue", "red"])
                plt.title("Sentiment Distribution", fontsize=16)
                plt.ylabel("Count", fontsize=14)
                plt.ylim(0, 1.5)  # Adjusted y-axis for single prediction
                st.pyplot(plt)
            else:
                st.error("No valid data to display in the bar chart.")
        else:
            st.warning("Please enter a valid comment.")

    # File upload for batch predictions
    st.subheader("Predict Sentiments from a File")
    uploaded_file = st.file_uploader("Upload an XLSX file:", type=["xlsx"])

    if uploaded_file:
        # Load the file
        df = pd.read_excel(uploaded_file)

        if "comments" in df.columns:
            # Preprocess and predict
            df["Sentiment"] = model.predict(vectorizer.transform(df["comments"].astype(str)))

            # Display results
            st.write("Predictions:")
            st.dataframe(df)

            # Download results
            output = BytesIO()
            df.to_excel(output, index=False)
            st.download_button(
                label="Download Predictions",
                data=output.getvalue(),
                file_name="sentiment_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # Generate bar chart for batch predictions
            st.subheader("Sentiment Distribution")
            sentiment_counts = df["Sentiment"].value_counts()
            plt.figure(figsize=(8, 6))
            sentiment_counts.plot(kind="bar", color=["green", "blue", "red"])
            plt.title("Sentiment Distribution", fontsize=16)
            plt.ylabel("Count", fontsize=14)
            plt.xticks(rotation=0, fontsize=12)
            st.pyplot(plt)
        else:
            st.error("The uploaded file must contain a column named 'comments'.")
