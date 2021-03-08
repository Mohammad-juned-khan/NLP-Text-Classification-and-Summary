import streamlit as st

#package for spacy summary
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
nlp = spacy.load('en_core_web_sm')

from heapq import nlargest
punctuation=punctuation+"\n"+" "+"  "

#summy summary package
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

#library fro classification
import re
import pickle
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

#function for summarization
def sumy_summarize(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result

#function for spacy summary
def spacy_process(text):
    #nlp = en_core_web_sm.load() #changed for 8 march
    #nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    #stopword = list(STOP_WORDS)


    # create Tokens of words
    tokens = [token.text for token in doc]
    # remove punctuations

    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1
    max_frequency = max(word_frequencies.values())
    # normalize the frequencies of words
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency
    # sentence tokenization
    sentence_tokens = [sent for sent in doc.sents]
    sentence_score = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_score.keys():
                    sentence_score[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_score[sent] += word_frequencies[word.text.lower()]
    select_length = int(len(sentence_tokens) * 0.4)
    summary = nlargest(select_length, sentence_score, key=sentence_score.get)
    final_summary = [word.text for word in summary]
    summary = " ".join(final_summary)

    return summary

#loding model for classification
#load model
modell = pickle.load(open('final_model', 'rb'))
tfidf = pickle.load(open('tfidf', 'rb'))


#function for classification
#function for preprocessing
def preprocess_text(text):
    # remove all punctuation
    text = re.sub(r'[^\w\d\s]', ' ', text)
    # collapse all white spaces
    text = re.sub(r'\s+', ' ', text)
    # convert to lower case
    text = re.sub(r'^\s+|\s+?$', '', text.lower())
    #taking only alphabetic words
    text = re.sub('[^a-zA-Z]',' ',text)
    # removing only single letters in the
    text= ' '.join( [w for w in text.split() if len(w)>1])
    # remove stop words and perform stemming
    stop_words = nltk.corpus.stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    return ' '.join(
        lemmatizer.lemmatize(term)
        for term in text.split()
        if term not in set(stop_words)
    )
#function for input prediction
def input_predict(text):
    # preprocess the text
    text = preprocess_text(text)
    # convert text to a list
    yh = [text]
    # transform the input
    inputpredict = tfidf.transform(yh)
    # predict the user input text
    y_predict_userinput = modell.predict(inputpredict)
    output = int(y_predict_userinput)

    return output

def main():
    html_temp1 = """<div style="background-color:#6D7B8D;padding:10px">
                    		<h4 style="color:white;text-align:center;">Summary Maker And Text Classifier Natural Language Processing(NLP) Application</h4>
                    		</div>
                    		</br>"""
    st.markdown(html_temp1, unsafe_allow_html=True)
    #st.title("Summary Maker And Text Classifier Natural Language Processing(NLP)  Application")
    activities = ["Home", "Summarize", "Classify"]
    st.sidebar.title("NLP Application")
    choice = st.sidebar.selectbox("Select Activity", activities)
    st.sidebar.title("About APP")
    st.sidebar.info( """This Application Helps You Classification Text/News Article Category and Helps Summarization .""")

    #Link for code
    st.sidebar.markdown("""[Github] (https://github.com/Mohammad-juned-khan) [LinkedIn] (https://www.linkedin.com/in/md-juned-khan) [Website] (https://www.mohammadjunedkhan.com/)""")

    if choice == "Home":
        #color codes  ff1a75  6D7B8D
        html_temp2 = """
                		<div style="background-color:#98AFC7;padding:10px">
                		<h4 style="color:white;text-align:center;">This is a Natural Language Processing(NLP) based application useful for Text/News Article Classification using Multinomial Naive Bayes Machine Learning model trained using TFIDF technique and summary generation task using Spacy and Sumy. This NLP functionality is implemented using Streamlit Framework.</h4>
                		</div>
                		<br></br>
                		<br></br>"""

        
        #st.markdown("""This is a Natural Language Processing(NLP) based application useful for Text/News Article Classification using Multinomial Naive Bayes Machine Learning model trained using TFIDF technique and summary generation task using Spacy and Sumy. This NLP functionality is implemented using Streamlit Framework.""")
        st.markdown(html_temp2, unsafe_allow_html=True)

        #st.info("This is the information of application")

    elif choice == "Summarize":
        st.subheader("Summary with NLP")
        raw_text = st.text_area("Enter Your Text", "Type Here or Paste")
        summary_choice = st.selectbox("Summary Choice", ["Spacy", "Sumy Lex Rank"])

        if st.button("Summarize"):

            if summary_choice == "Spacy":
                summary_result = spacy_process(raw_text)
            elif summary_choice == "Sumy Lex Rank":
                summary_result = sumy_summarize(raw_text)

            st.success(summary_result)
        else:
            return None
    elif choice == "Classify":
        st.subheader("Classify Text with NLP Model")
        raw_text = st.text_area("Enter Your Text", "Type Here or Paste")
        if st.button("Classify"):
            predict_input = input_predict(raw_text)
            if predict_input == 0:
                category = "Business"
            elif predict_input == 1:
                category = "Entertainment"
            elif predict_input == 2:
                category = "Politics"
            elif predict_input == 3:
                category = "Sports"
            elif predict_input == 4:
                category = "Technology"
            else:
                category = "Error"

            st.success(category)

        else:
            return None

    return None



if __name__ == "__main__":
    main()
