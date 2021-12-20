#1. Import Necessary libraries
import pandas as pd
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #Lemmattization

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer



#2. Import Dataset
spam_ham_data = pd.read_csv('SMSSpamCollection',sep = '\t',names = ['label','messages'])



#3. Data Preparation
### Step 1: Prepare a pipeline for Text Preprocessing
corpus = []
#ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

for i in range(0,len(spam_ham_data)):
     review = re.sub('[^a-zA-Z]', ' ', spam_ham_data['messages'][i])
     review = review.lower()
     review = review.split()
     #review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
     review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
     review = ' '.join(review)
     corpus.append(review)

### Step 2: Create a Document Matrix
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer #To implement BOW
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

### Step 3: Create X and y
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(spam_ham_data['label'])


#4. Model Building
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,stratify=y,random_state=12)

#5. Model Training
from sklearn.naive_bayes import MultinomialNB
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train,y_train)

#5. Model Testing
y_pred = nb_classifier.predict(X_test)

#6. Model Evaluation
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score
confusion_mat = confusion_matrix(y_test,y_pred)
acc_score     = accuracy_score(y_test,y_pred)
precision     = precision_score(y_test,y_pred)













