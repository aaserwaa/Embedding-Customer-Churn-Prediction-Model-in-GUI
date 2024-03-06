import streamlit as st
# Import the other pages 
from pages.dashboard import show_dashboard_page
#from pages.data import show_data_page
from pages.predict import show_predict_page
from pages.history import show_history_page
#from pages.contact import show_contact_page

# Function to create call-to-action button
def create_cta_button(cta_text, cta_link):
    st.button(cta_text, key=cta_link)

# Function to create header section
def create_header_section(title):
    st.title(title)

# Function to create hero section
def create_hero_section(header, subheader, image_url):
    st.header(header)
    st.subheader(subheader)
    st.image(image_url, use_column_width=True)

# Function to create features section
def create_features_section(features):
    st.header("Key Features")
    st.write("List of key features:")
    for feature in features:
        st.write(f"- {feature}")

# Function to create how it works section
def create_how_it_works_section(steps):
    st.header("How It Works")
    st.write("Step-by-step guide:")
    for step in steps:
        st.write(step)

# Function to create testimonials section
def create_testimonials_section(testimonials):
    st.header("Testimonials")
    st.write("Customer reviews and trust signals:")
    for testimonial in testimonials:
        st.write(testimonial)
        
# Function to create footer section
def create_footer_section(contact_info, legal_info):
    st.sidebar.header("Contact Information")
    st.sidebar.write("Contact details:")
    for contact_detail in contact_info:
        st.sidebar.write(contact_detail)

    st.sidebar.header("Legal")
    for legal_detail in legal_info:
        st.sidebar.write(legal_detail)

# Function to display the home page content
def show_home_page():
    # Call to Action Button
    cta_text = "Start Predicting Churn"
    cta_link = "/predict-churn"
    create_cta_button(cta_text, cta_link)
    

    # Header Section
    header_title = "Stay - Predict Customer Churn"
    create_header_section(header_title)

    # Hero Section
    hero_header = "Reduce Churn and Boost Retention with Stay"
    hero_subheader = "Empower Your Business with Predictive Analytics to Keep Customers Happy and Loyal."
    hero_image_url = "https://www.touchpoint.com/wp-content/uploads/2023/02/5.-Customer-churn-article.png"
    create_hero_section(hero_header, hero_subheader, hero_image_url)
    
    # Introduce the app and its purpose
    st.markdown("---")
    st.subheader("What is this app?")
    st.markdown(
      """
      STAY uses machine learning models to predict the likelihood of customer churn. 
      By providing relevant information about your customers, you can gain valuable insights 
      to identify at-risk customers and take proactive measures to retain them.
      """
    )
# Highlight app benefits
    st.subheader("Why use this app?")
    st.markdown(
      """
      * **Early identification:** Predict churn before it happens, allowing you to focus resources on retention efforts.
      * **Data-driven insights:** Gain a deeper understanding of factors contributing to churn.
      * **Improved decision-making:** Make informed decisions to target effective retention strategies.
      """
    )
    # Features Section
    features_list = [
        "Advanced Machine Learning Models",
        "KPI's Dashboards",
        "Real-time Predictions",
        "Actionable Insights"
    ]
    create_features_section(features_list)

    # How It Works Section
    steps_list = [
        "1. Input Customer Information",
        "2. Choose a Model",
        "3. View Results"
    ]
    create_how_it_works_section(steps_list)

    # Testimonials Section
    testimonials_list = [
        "Customer review 1",
        "Customer review 2",
        "Customer review 3"
    ]
    create_testimonials_section(testimonials_list)

    # Footer Section
    contact_info = [
        "Email: info@stayapp.com",
        "Phone: +233-50-456-7890"
    ]
    legal_info = [
        "Privacy Policy",
        "Terms of Service"
    ]
    
    create_footer_section(contact_info, legal_info)

if __name__ == '__main__':
    show_home_page()
