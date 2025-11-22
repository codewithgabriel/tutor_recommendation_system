# Quick useful links

Acces Code [Source Code](https://github.com/codewithgabriel/tutor_recommendation_system/blob/main/tutor-recommendation-system.ipynb)

Access to  [recommendation.csv file](https://github.com/codewithgabriel/tutor_recommendation_system/blob/main/recommendations.csv)

Access to [engineered_features.csv](https://github.com/codewithgabriel/tutor_recommendation_system/blob/main/engineered_features.csv)

# Introduction

In this notebook, a Tutor Recommendation System is built using the provided documents in the email for the assessment which include  `students.pdf` and `tutors.pdf` files, the methological approach employed is a combination of a content base filtering with scoring system for feature engineering and  LightGBM (Gradient Boost Model) and ANN (Artificial Neural Network) model  for recommendation and ranking. 

# Why LightGBM and ANN
LightGBM was choosen because it is  optimized for speed and memory usage which makes it suitable for relatively small datasets like this one, It natively supports ranking tasks which aligns perfectly with the recommendation system’s goal of ranking tutors for each student also LightGBM can efficiently handle categorical variables after encoding, ensuring strong performance on structured tabular data another reason is that feature importance scores from LightGBM provide insights into which student-tutor attributes drive recommendations.

Initially for this project, I only want to use LightGBM, but the choice of ANN came because it allows  exploration of deep learning approaches for recommendation, which can scale to larger datasets in the future. ANN can capture complex, non-linear patterns between student preferences and tutor attributes that may not be obvious in tabular features. Neural networks adapt well to diverse feature types and can learn intricate interactions beyond simple scoring rules also ANN provides a complementary perspective to LightGBM.

# The methodology steps carried out in brief:
1. Data Preprocessing : In this step, the given dataset was converted from pdf file format to pandas DataFrame datatype which is machine learning friendly using `camelot-py` library,  data cleaning is not carried out because the dataset is short and there is no missing/null/nan values.
2. Feature Engineering: The content base filtering approach uses a scoring system to include additional features (column) named `target_label` that capture the similarities between students and tutors, this method generate the column by filtering different content base logical and idea conditions such as if the student's hobby intercept with tutor's hobby, another example is a case in-which the student's prefered programming language matches the tutor's language and so on. By applying content-based method, the process creates context awareness and pattern for the models to learn and adapt with by giving a `target_label` which serves as a rank feature for each observation in the dataset.
3. Encoding Categorical or Dummy variables: In this step each columns containing categorical or dummy variables were handled using `LabelEncoder` from  `sklearn.preprocessing` module. Additionally, `hobbies` column in both students and tutors dataset was tokenized and converted into set datatype, this is carried out because each observation in the students `hobbies` column was use to filter intersection with tutors `columns`.
4. Feature Seperation: After succesful data preprocessing and feature engineering, features and target label were extracted.
5. Data Split: The dataset is then split into train set and test set use 80% and 20% respectively, then each set was grouped according to `student_id`, this ensure that the dataset is ready for LightGBM and ANN model training.
6. Model Training: During this phase LightGBM and ANN model was trained using the train set and test set from 5 above.
7. Model Evaluation and Testing: The Performance of the LightGBM  and ANN model was evaluted using metrics such as normalized discounted cummalative gain `(NDCG@k)` and discounted cummulative gain `(dcg)`, the higher metrics the more accurate prediction for the models,  these two metrics are then used to calculate the relevance score of each tutor to student, the higher this relevance score the more similarity and recommendation.