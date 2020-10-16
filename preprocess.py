#store the resume in a variable
def process_resume (file):
    import re #regular expression for editing text
    import nltk #nltk library for hand
    from textblob import TextBlob
    from nltk.corpus import stopwords #to remove stopwords
    from nltk.tokenize import word_tokenize #to split text into minimal meaningful units
    from textblob import Word #to extract a root word by considering the vocabulary
    from nltk.stem import PorterStemmer #to extract the root word of a text
    
    from nltk.util import ngrams  
    
    resume = file
    
    #convert to lowercase
    resume = resume.lower()
    
    #remove hyphen
    resume = resume.replace('_','')
    
    #remove punctuation
    resume = re.sub(r'[^\w\s]','', resume) 
    
    #Remove additional white spaces
    resume = re.sub('[\s]+', ' ', resume)
    resume = re.sub('[\n]+', ' ', resume)
    
    #Remove not alphanumeric symbols white spaces
    resume = re.sub(r'[^\w]', ' ', resume)
    
    #extract email address
    addresses = re.findall(r'[\w\.-]+@[\w\.-]+', resume)
    
    #correct wrong spelling
    #resume = ''.join(str(TextBlob(resume).correct()))
    
    #lemmatize text    
    resume = " ".join([Word(word).lemmatize() for word in resume.split()])
    
    #tokenize text
    resume = nltk.word_tokenize(resume)
    
    #remove stopwords
    stop = stopwords.words('english')
    resume = [i for i in resume if not i in stop]
    
    #stemming
    #stemmer = PorterStemmer()
    #resume=" ".join([stemmer.stem(word) for word in resume.split()]))
    
    #trim
    #resume = resume.strip('\'"')
    
    resume = " ".join(resume)
        
    file = resume
    
    return file

