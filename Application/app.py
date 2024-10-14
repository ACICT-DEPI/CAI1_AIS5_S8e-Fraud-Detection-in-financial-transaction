import streamlit as st
import joblib

# Load text models
tfidf_model = joblib.load(r'E:\DEPI\Tech-Project\APP\text_models\tfidf_vectorizer.pkl')
log_model = joblib.load(r'E:\DEPI\Tech-Project\APP\text_models\Log_model.pkl')
# KNN_model = joblib.load(r'E:\DEPI\Tech-Project\APP\text_models\KNN_model.pkl')
# decisionTree_model = joblib.load(r'E:\DEPI\Tech-Project\APP\text_models\DecisionTree_model.pkl')

# Load features model
AdaBoost_model = joblib.load(r'E:\DEPI\Tech-Project\APP\features_models\Ada_boost.pkl')

# First page for model selection
def main_page():
    st.title('Welcome to the Fraud Detection System')

    # Create two buttons for model selection
    col1, col2 = st.columns(2)

    with col1:
        if st.button('Using Features'):
            st.session_state['page'] = 'select_features_model'

    with col2:
        if st.button('Using Mail Text'):
            st.session_state['page'] = 'select_text_model'
            
            
def select_text_model_page():
    st.title('Select Text Model for Fraud Detection')

    # Dropdown to select the model
    model_choice = st.selectbox(
        'Select the model to use for prediction',
        ('Logistic Regression', 'KNN', 'Decision Tree')
    )

    # Initialize prediction_model
    prediction_model = None

    # Load the selected model based on user's choice
    if model_choice == 'Logistic Regression':
        prediction_model = log_model
    elif model_choice == 'KNN':
        # Uncomment this when KNN_model is available
        # prediction_model = KNN_model
        st.write('KNN model is currently unavailable.')
        return
    elif model_choice == 'Decision Tree':
        # Uncomment this when decision_tree_model is available
        # prediction_model = decision_tree_model
        st.write('Decision Tree model is currently unavailable.')
        return

    # Check if prediction_model is defined before saving to session state
    if prediction_model is not None:
        # Save the selected model and tfidf model to session state
        st.session_state['selected_model'] = model_choice
        st.session_state['tfidf_model'] = tfidf_model
        st.session_state['prediction_model'] = prediction_model

        # Proceed to prediction text page
        st.session_state['page'] = 'prediction_text'


# Define a function for the main logic of the prediction page using email/text
def prediction_text_page():
    # Load the selected model and tfidf model from session state
    tfidf_model = st.session_state['tfidf_model']
    prediction_model = st.session_state['prediction_model']

    # Create a simple web interface for prediction
    st.title(f'Credit Card Fraud Detection ({st.session_state["selected_model"]}) - Text Input')
    st.write('Enter the email or transaction message text to check if it is fraudulent:')

    # Get user input (text)
    input_text = st.text_area('Input email text', '')

    # Create two columns for the Predict and Back buttons
    col1, col2 = st.columns(2)

    with col1:
        # Predict button
        if st.button('Predict'):
            if input_text:  # Ensure the input text is not empty
                try:
                    # Transform the input text using the tfidf model
                    transformed_text = tfidf_model.transform([input_text])

                    # Make prediction using the selected model
                    prediction = prediction_model.predict(transformed_text)

                    # Display result based on prediction
                    if prediction[0] == 1:
                        st.write('This email or message is **fraudulent**.')
                    else:
                        st.write('This email or message is **NON fraudulent**.')
                except Exception as e:
                    st.write('Error:', e)
            else:
                st.write('Please input a valid text message.')

    with col2:
        # Back button
        if st.button('Back'):
            st.session_state['page'] = 'selection'




# Define a function to select the features model
def select_features_model_page():
    st.title('Select Features Model for Fraud Detection')

    # Dropdown to select the model
    model_choice = st.selectbox(
        'Select the model to use for prediction',
        ('AdaBoost', 'Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest')
    )


    # Initialize prediction_model
    prediction_model = None

    # Load the selected model based on user's choice
    if model_choice == 'AdaBoost':
        prediction_model = AdaBoost_model
    elif model_choice == 'Logistic Regression':
        # Uncomment this when decision_tree_model is available
        # prediction_model = decision_tree_model
        st.write('Logistic Regression model is currently unavailable.')
        return
    elif model_choice == 'KNN':
        # Uncomment this when KNN_model is available
        # prediction_model = KNN_model
        st.write('KNN model is currently unavailable.')
        return
    elif model_choice == 'Decision Tree':
        # Uncomment this when decision_tree_model is available
        # prediction_model = decision_tree_model
        st.write('Decision Tree model is currently unavailable.')
        return
    
    # Check if prediction_model is defined before saving to session state
    if prediction_model is not None:
        # Save the selected model and features to session state
        st.session_state['selected_model'] = model_choice
        st.session_state['prediction_model'] = prediction_model

    # Proceed to prediction features page
    st.session_state['page'] = 'prediction_features'
        
        
    
# Define a function for the main logic of the prediction page using features
def prediction_features_page():
    st.title(f'Credit Card Fraud Detection ({st.session_state["selected_model"]}) - Feature Input')
    st.write('Enter the transaction features to check if it is fraudulent:')

    # Create inputs for features (for example, amount, time, and other numeric features)
    Year = st.number_input('Year', min_value=0, max_value=2024)
    Month = st.number_input('Month', min_value=0, max_value=12)
    Day = st.number_input('Day', min_value=0, max_value=31)
    Hours = st.number_input('Hours', min_value=0.0, max_value=24.0)
    Amount = st.number_input('Amount', min_value=0.0, max_value=1000000.0) 
    
    Use_Chip = st.text_input('Use Chip')# string
    Merchant_City = st.text_input('Merchant_City')  # string
    Merchant_State = st.text_input('Merchant_State')  # string
    
    Zip = st.number_input('Zip', min_value=0.0, max_value=10000.0)
    MCC = st.number_input('MCC', min_value=0.0, max_value=10000.0)
    
    Notes = st.text_input('Notes')    # string
    
    
    # Collect all features in a list or array
    features = [Year, Month, Day, Hours, Amount, Use_Chip, Merchant_City, Merchant_State, Zip, MCC, Notes]

    # Create two columns for the Predict and Back buttons
    col1, col2 = st.columns(2)

    with col1:
        # Predict button
        if st.button('Predict'):
            if all(f >= 0 for f in features):  # Ensure features are valid
                try:
                    # Make prediction using the selected model
                    prediction_model = st.session_state['prediction_model']
                    prediction = prediction_model.predict([features])

                    # Display result based on prediction
                    if prediction[0] == 1:
                        st.write('This transaction is **fraudulent**.')
                    else:
                        st.write('This transaction is **NON fraudulent**.')
                except Exception as e:
                    st.write('Error:', e)
            else:
                st.write('Please input valid feature values.')

    with col2:
        # Back button
        if st.button('Back'):
            st.session_state['page'] = 'selection'


# Main function to control page navigation
def main():
    # Check the current page in session state, default is 'selection'
    if 'page' not in st.session_state:
        st.session_state['page'] = 'selection'
    
    # Render the appropriate page based on session state
    if st.session_state['page'] == 'selection':
        main_page()
    elif st.session_state['page'] == 'select_features_model':
        select_features_model_page()
    elif st.session_state['page'] == 'select_text_model':
        select_text_model_page()
    elif st.session_state['page'] == 'prediction_text':
        prediction_text_page()
    elif st.session_state['page'] == 'prediction_features':
        prediction_features_page()

# Run the app
if __name__ == "__main__":
    main()




import streamlit as st
import joblib

# Load text models
tfidf_model = joblib.load(r'E:\DEPI\Tech-Project\APP\text_models\tfidf_vectorizer.pkl')
log_model = joblib.load(r'E:\DEPI\Tech-Project\APP\text_models\Log_model.pkl')
# KNN_model = joblib.load(r'E:\DEPI\Tech-Project\APP\text_models\KNN_model.pkl')
# decisionTree_model = joblib.load(r'E:\DEPI\Tech-Project\APP\text_models\DecisionTree_model.pkl')

# Load features model
AdaBoost_model = joblib.load(r'E:\DEPI\Tech-Project\APP\features_models\Ada_boost.pkl')

# Main Home page for model selection
def home_page():
    st.title('Fraud Detection in Financial Transaction')
    
    # Dropdown to Select The Transaction Type
    option = st.selectbox(
        "Select The Transaction Type:",
        ("", "Predict With Text Mail", "Predict With Transaction Credit Card")
    )

            
    # Button to navigate to the selected page
    if st.button("Go to Prediction"): 
        # Disable the button if no transaction type is selected
        if not option:
            st.error("Please select a transaction type to proceed.")
        else:
            # Store selected option in session state to navigate to the corresponding page
            if option == "Predict With Text Mail":
                st.session_state["page"] = "text_mail"
            elif option == "Predict With Transaction Credit Card":
                st.session_state["page"] = "credit_card"
        
        st.experimental_rerun()  # Refresh the app to switch the page based on the selected option


def predict_with_text_mail():
    st.subheader("Predict With Text Mail")
    
    # Text input for user to paste email content
    email_content = st.text_area("Enter the email content for fraud prediction:")
    
    # # Select model for prediction
    # model_choice = st.selectbox(
    #     "Choose a model for prediction:",
    #     ("", "Logistic Regression", "K-Nearest Neighbors", "Decision Tree")
    # )

    if st.button("Predict"):
        # Validate email content and model selection
        if not email_content.strip():
            st.error("Please enter email content.")
        # elif not model_choice:
        #     st.error("Please select a model for prediction.")
        else:
            # If inputs are valid, proceed with the prediction
            transformed_text = tfidf_model.transform([email_content])
            
            prediction = None  # Initialize prediction variable
            prediction = log_model.predict(transformed_text)
            
            # # Predict based on selected model
            # if model_choice == "Logistic Regression":
            #     prediction = log_model.predict(transformed_text)
            # elif model_choice == "K-Nearest Neighbors":
            #     st.warning("KNN model is currently unavailable.")
            # elif model_choice == "Decision Tree":
            #     st.warning("Decision Tree model is currently unavailable.")
            
            # Display the result if prediction exists
            if prediction is not None:
                if prediction[0] == 1:
                    st.success("This transaction is likely to be fraudulent.")
                else:
                    st.success("This transaction seems legitimate.")


# Function to handle predictions using credit card transaction data  
def predict_with_transaction_credit_card():
    st.subheader("Predict With Transaction Credit Card")
    
    # Example features: amount, transaction type, etc.
    transaction_amount = st.number_input("Enter the transaction amount:")
    transaction_type = st.selectbox("Select the transaction type:",
                                    ("Online Purchase", "Point of Sale", "ATM Withdrawal"))
    
    if st.button("Predict"):
        if transaction_amount > 0:
            # Dummy features example, replace this with actual feature extraction
            features = [[transaction_amount]]  # Replace with actual features
            
            # Prediction using AdaBoost model
            prediction = AdaBoost_model.predict(features)
            
            # Display the result
            if prediction[0] == 1:
                st.success("This transaction is likely to be fraudulent.")
            else:
                st.success("This transaction seems legitimate.")
        else:
            st.error("Please enter valid transaction details.")


# Main function to control page navigation
def main():
    # Initialize session state if not already present
    if "page" not in st.session_state:
        st.session_state["page"] = "home"
    
    # Navigate between pages based on session state
    if st.session_state["page"] == "home":
        home_page()
    elif st.session_state["page"] == "text_mail":
        predict_with_text_mail()
    elif st.session_state["page"] == "credit_card":
        predict_with_transaction_credit_card()

    # # Button to go back to the home page
    # if st.session_state["page"] != "home":
    #     if st.button("Back to Home"):
    #         st.session_state["page"] = "home"
    #         st.experimental_rerun()


# Run the app
if __name__ == "__main__":
    main()
