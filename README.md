# Spam-SMS-Classifier
A tutorial on how to perform preprocessing of text data, vectorization, choosing a Machine Learning model and optimizing its hyperparameters.The dataset can be downloaded from [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset/data)

## Getting Started
The Spam Classifier aims at classifiying SMS as spam or ham. Detailed explanation can be found in the IPython Notebook.

#### Split into train and test dataset
Use read_csv() function in Pandas to load the dataset and then split the dataframe into train and test dataset using train_test_split() function in sklearn library.
```python
from sklearn.model_selection import train_test_split
xtrain,xval,ytrain,yval= train_test_split(data_prepared,encoded_label,test_size=0.2, random_state=42)
```
#### Preprocess data
Here we use functions available in NLTK toolkit like tokenizers and lemmatizers(WordNetLemmatizer) to preprocess the textual data and then convert it into vectors using vectorizers(CountVectorizer). The target variables are converted to one hot vectors(LabelBinarizer). All these custom transformers are written as Pipelines.

###### Custom Transformer for preprocessing text data

```python
class NLTK_Preprocessing_Module(BaseEstimator, TransformerMixin):
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        return (self.helperFunction(X))
    
    def lemmatize_all(self,sentence):
        
        wnl = WordNetLemmatizer()
        for word, tag in pos_tag(word_tokenize(sentence)):
            if tag.startswith("NN"):
                yield wnl.lemmatize(word, pos='n')
            elif tag.startswith('VB'):
                yield wnl.lemmatize(word, pos='v')
            elif tag.startswith('JJ'):
                yield wnl.lemmatize(word, pos='a')
            elif tag.startswith('R'):
                yield wnl.lemmatize(word, pos='r')
            else:
                yield word
            
    def msgProcessing(self,raw_msg):
        
        meaningful_words=[]
        words2=[]
        raw_msg = str(raw_msg.lower())
        raw_msg=re.sub(r'[^a-z\s]', ' ', raw_msg)
        words=raw_msg.split()
        """Remove words with length lesser than 2"""
        for i in words:
            if len(i)>=2:
                words2.append(i)
        stops=set(stopwords.words('english'))
        meaningful_words=" ".join([w for w in words2 if not w in stops])
        return(" ".join(self.lemmatize_all(meaningful_words)))


    def helperFunction(self,df):
        
        print ("Data Preprocessing!!!")
        cols=['Message']
        df=df[cols]
        df.Message.replace({r'[^\x00-\x7F]+':''},regex=True,inplace=True)
        num_msg=df[cols].size
        clean_msg=[]
        for i in range(0,num_msg):
            clean_msg.append(self.msgProcessing(df['Message'][i]))
        df['Processed_msg']=clean_msg
        X=df['Processed_msg']
        print ("Data Preprocessing Ends!!!")
        return X
```
#### ML Models

I have chosen four models.
1. Multinomial Naive Bayes
2. Decision Tree
3. Random Forest
4. Support Vector Machine

#### Plotting ROC Curve and choosing best performig classifier

Train the models in default hyperparameters and check how well they perform using k cross validation(k=5). Once the models are trained, plot the ROC (Receiver Output Characteristics) curve to see which classifier performs the best. The best classifier can be identified by computing the ROC AUC score. An ideal model will have an ROC AUC score=1.

```python
from sklearn.metrics import roc_auc_score
auc_MNB=roc_auc_score(ytrain, MNB_scores1)
```
## Prerequisites
1. Python (>=3.4)
2. Pandas
3. Scikit
4. Matplotlib
5. Numpy
6. NLTK

## Built With
1. Jupyter Notebook
