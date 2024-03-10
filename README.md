# Embedding Customer Churn Prediction Model in GUI

## Overview
This project employs pre-trained models stored using joblib, trained on Telco Churn customer data from Vodafone. Its objective is to develop a user-friendly interface suitable for both non-technical and tech-savvy users, enabling seamless prediction of customer churn. This will be accomplished by constructing a web application using Streamlit.

## Table of Contents
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Project Structure
The project is organized into the following main components:

 1. `home.py/`
   - Contains the code for the home page of the streamlit app.
  
 2. `saved models/`
   - This folder contained the saved models that will be used for prediction.
   
 3. `encoder/`
   - Saves encoders or transformers used during preprocessing.
   
 5. `README.md`
   - The main documentation file providing an overview of the project.

 6. `requirements.txt`
   - Lists all project dependencies for easy installation.

 7. `images/`
   - Contains images, charts, or diagrams used in documentation.


## Dependencies
To run this project, you need to have the following dependencies installed. You can install them using `pip`:
streamlit: framework for building the app

pandas: Data manipulation and analysis.

scikit-learn: Machine learning tools and utilities.

matplotlib: Data visualization.

seaborn: Statistical data visualization.

plotly: visualization tool

joblib: Joblib is used for parallelizing code, particularly during model training.

numpy: Mathematical functions for numerical operations.

jupyter: Jupyter notebooks for interactive data exploration.

all other requirements are named in the requirement.txt file

## Installation

Follow these steps to set up and run the project on your local machine.

### Clone the Repository

git clone https://github.com/aaserwaa/Customer-Churn-Analysis.git
cd your-project

## Usage

Run the cells in your pyhthon enviroments to execute the streamlit app interface. This includes home, dashboard, history, predict and data pages.

## Conclusion

The project successfully addresses the challenge of predicting customer churn in a telecommunications company and also provides a user freindly web app interface for doing so. Key insights include:

- Utilizing both Random Forest and SVM models for their high accuracy and balanced recall.
- Identifying areas for improvement in precision for churned customers.

## Contributing

Contributions to this project are welcome. To contribute, please follow these steps:
1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the [MIT License](LICENSE).

Here is a link to my [published article on LinkedIn](https://www.linkedin.com/pulse/customer-churn-analysis-my-journey-predictive-analytics-serwaa-akoto-wnvke)
