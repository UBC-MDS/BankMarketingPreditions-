# Bank Marketing Campaign Prediction

**Authors**: Hala Arar, Fazeeia Mohammad, Rong Wan 

## Project Summary

This project aims to enhance bank marketing campaigns by using machine learning to predict whether a customer will subscribe to a term deposit based on demographic and campaign-related data. The goal is to develop a model that can more effectively target potential customers, improving resource allocation and reducing marketing costs by excluding unlikely subscribers.

The project explores several machine learning models, including Logistic Regression and Decision Trees. These models are trained on customer data, with preprocessing techniques like feature scaling and one-hot encoding applied to prepare the data for analysis. The outcome of this work demonstrates how banks can leverage machine learning to implement more effective, data-driven marketing strategies, which can lead to better customer acquisition and optimized campaign performance.

The Logistic Regression model achieved an accuracy of 88.5%, with a focus on minimizing false positives, resulting in a precision of 0.70 and recall of 0.20. Despite its high precision, the model's low recall indicates that it misses a significant portion of actual subscribers. The Decision Tree model, with an accuracy of 89.7%, demonstrated better recall (0.23) but at the cost of increased false positives (126). Both models highlight the class imbalance in the dataset, where non-subscribers are far more prevalent than subscribers. The Logistic Regression model is more suitable when minimizing false positives is prioritized, whereas the Decision Tree model is more effective in identifying potential subscribers but may require further regularization to reduce overfitting. 

Strategic recommendations for targeted marketing, personalized offers, and campaign timing are proposed to optimize resource allocation and improve conversion rates. Future model iterations should focus on improving both precision and recall to enhance marketing efforts and increase return on investment.


## How to Run the Analysis

### Option 1: Run Using Docker

1. **Install Docker**

Download and install Docker Desktop for your operating system and ensure it is running.

2. **Clone the repository**

Open your terminal and run the following command to clone the project repository to your local machine:

git clone https://github.com/mindy001/BankMarketingPreditions-.git

Navigate to the project directory

cd BankMarketingPreditions-


3. **Pull the Docker Image**

Run the following command in your terminal to pull the project’s Docker image:

docker pull fazeeia/dsci522-dockerfile-bank:latest

4. **Run the Docker Container**

Run the following command in your terminal to start the Docker container and mount your local directory:

docker run -p 8888:8888 -it -v /$(pwd):/home/jovyan/work fazeeia/dsci522-d

-p 8888:8888: This maps port 8888 inside the container to port 8888 on your local machine (where JupyterLab will be running).
-v $(pwd):/home/jovyan/work: This mounts your current working directory to the container’s working directory, allowing you to access your project files inside the container.

5. **Access JupyterLab**

You will see a URL in the terminal. Open the link in your browser to access the JupyterLab environment. 

### Option 2: Run Locally

1. **Clone the repository**

git clone https://github.com/mindy001/Group37DSCI522.git

2. **Set up the environment**

conda env create -f env/environment.yml
conda activate bankenv

3. **Run the Analysis**

After activating the environment, you can run the analysis script or Jupyter notebook called bank_marketing_analysis.ipynb

4. **Open the Report**

The final report is available as a PDF. You can view the completed analysis by opening the bank_marketing_analysis.pdf file.

## Dependencies

To run the analysis and work with the code, you will need to install the following Python packages. These are automatically included in the environment.yml file, but here is the full list for reference:

altair
numpy
pandas
scikit-learn (includes tools like train_test_split, GridSearchCV, StandardScaler, OneHotEncoder, KNeighborsClassifier, LogisticRegression, DecisionTreeClassifier, etc.)
matplotlib
seaborn
ucimlrepo
altair_ally

These dependencies are necessary for data processing, model building, evaluation, and visualization.


## License

This project is licensed under CC0 1.0 Universal (Creative Commons Public Domain Dedication). By applying this license, the creator voluntarily waives all copyright and related rights, allowing anyone to use, modify, distribute, or build upon the work for any purpose, including commercial purposes, without the need for permission or attribution. 

## References

Meshref, H. (2020). Predicting loan approval of bank direct marketing data using ensemble machine learning algorithms. International Journal of Circuits, Systems and Signal Processing, 14, 117. https://doi.org/10.46300/9106.2020.14.117

Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

Wang, D. (2020). Research on bank marketing behavior based on machine learning. AIAM2020: Proceedings of the 2nd International Conference on Artificial Intelligence and Advanced Manufacture, 150–154. https://doi.org/10.1145/3421766.3421800

Xie, C., Zhang, J.-L., Zhu, Y., Xiong, B., & Wang, G.-J. (2023). How to improve the success of bank telemarketing? Prediction and interpretability analysis based on machine learning. Computers & Industrial Engineering, 175, 108874. https://doi.org/10.1016/j.cie.2022.108874

Zaki, A. M., Khodadadi, N., Lim, W. H., & Towfek, S. K. (2024). Predictive analytics and machine learning in direct marketing for anticipating bank term deposit subscriptions. American Journal of Business and Operations Research, 11(1), 79-88. https://doi.org/10.54216/AJBOR.110110

