#21995445

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

f = open('all_sentiment_shuffled.txt', 'r', encoding='utf-8')
lines = f.read().splitlines() #read the file line by line

file = []
for i in range(len(lines)):
    file.append(lines[i].split(" ", 3)) #split lines into 4 columns

sentiment = []
review = []
for i in range(len(file)):
    sentiment.append(file[i][1]) #get the 2nd column as sentiments
    review.append(file[i][3])    #get the 4th column as reviews

train_sentiment, test_sentiment, train_review, test_review = train_test_split(sentiment, review, test_size=0.2,
                                                                              shuffle=True) #split data into train and test data

pos_rev = []
neg_rev = []
for i in range(len(train_sentiment)):
    if train_sentiment[i] == 'pos':
        pos_rev.append(train_review[i])     #get the positive reviews and hold them in an array
    elif train_sentiment[i] == 'neg':
        neg_rev.append(train_review[i])     #get the negative reviews and hold them in an array


def createDict(rev_arr, feature):
    if feature == 'unigram':
        sBoW = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None, ngram_range=(1, 1))     #ngram as unigram
    elif feature == 'bigram':
        sBoW = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None, ngram_range=(2, 2))     #ngram as bigram
    elif feature == 'both':
        sBoW = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None, ngram_range=(1, 2))     #ngram as unigram+bigram

    sen_BoW = sBoW.fit_transform(rev_arr)
    sen_word_count = sen_BoW.toarray().sum(axis=0)
    sZip = zip(sBoW.get_feature_names(), sen_word_count)
    sen_dict = dict(sZip)                                               #create dictionary for neg/pos words

    sBow_features = sBoW.get_feature_names()                            #this is used for creating vocab dictionary
    sBow_sum = sen_BoW.sum()                                            #number of words in the neg/pos dict used in calculating probability

    return sen_dict, sBow_features, sBow_sum


def NaiveBayes(feature):
    pos_dict, pBow_features, pBow_sum = createDict(pos_rev, feature)  #create pos dictionary
    neg_dict, nBow_features, nBow_sum = createDict(neg_rev, feature)  #create neg dictionary

    vocab_count = len(set(pBow_features + nBow_features))               #create vocabulary length

    prob_of_pos = np.log10(len(pos_rev) / len(train_review))            # p(positive)
    prob_of_neg = np.log10(len(neg_rev) / len(train_review))            # p(negative)

    correct_prediction = 0
    for i in range(len(test_review)):                                   #loop to predict for every review in the test data

        pos_prob_of_rev = 0
        neg_prob_of_rev = 0

        if feature == 'unigram':
            tBoW = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None, ngram_range=(1, 1))  # ngram as unigram
        elif feature == 'bigram':
            tBoW = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None, ngram_range=(2, 2))  # ngram as bigram
        elif feature == 'both':
            tBoW = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None, ngram_range=(1, 2))  # ngram as unigram+bigram
        try:
            test_BoW = tBoW.fit_transform(test_review[i:i + 1])         #get the line(review)
        except:
            continue

        test_word_count = test_BoW.toarray().sum(axis=0)
        tZip = zip(tBoW.get_feature_names(), test_word_count)
        test_dict = dict(tZip)                                          #create dictionary fot the review

        for word in test_dict.keys():
            try:
                pos_prob_of_rev += np.log10((pos_dict[word] + 1) / (pBow_sum + vocab_count) + 1)        #if word exist in positive dictionary calculate probability laplace smoothing applied
            except KeyError:
                pos_prob_of_rev += np.log10(1 / (pBow_sum + vocab_count))                               #if word doesn't exist in positive dictionary calculate probability laplace smoothing applied
            try:
                neg_prob_of_rev += np.log10((neg_dict[word] + 1) / (nBow_sum + vocab_count) + 1)  #if word exist in negative dictionary calculate probability laplace smoothing applied
            except KeyError:
                neg_prob_of_rev += np.log10(1 / (nBow_sum + vocab_count))                         #if word doesn't exist in negative dictionary calculate probability laplace smoothing applied

        pos_prob_of_rev += prob_of_pos   #sum of  pos probabilities of all the words (not multipication bcs i used log probabilities)
        neg_prob_of_rev += prob_of_neg   #sum of  neg probabilities of all the words (not multipication bcs i used log probabilities)

        if pos_prob_of_rev > neg_prob_of_rev and test_sentiment[i] == 'pos': #if pos prob of review is bigger than neg prob and the sentiment of review is pos correct prediction
            correct_prediction += 1
        elif neg_prob_of_rev > pos_prob_of_rev and test_sentiment[i] == 'neg': #if neg prob of review is bigger than pos prob and the sentiment of review is neg correct prediction
            correct_prediction += 1

        acc = (correct_prediction / len(test_sentiment)) * 100  #calculate accuracy

    return acc

def tf_idf(feature,pos_rev,neg_rev):

    if feature == 'unigram':
        p_tfidf = TfidfVectorizer(stop_words=None, ngram_range=(1, 1))  # ngram as unigram
        n_tfidf = TfidfVectorizer(stop_words=None, ngram_range=(1, 1))
    elif feature == 'bigram':
        p_tfidf = TfidfVectorizer(stop_words=None, ngram_range=(2, 2))  # ngram as bigram
        n_tfidf = TfidfVectorizer(stop_words=None, ngram_range=(2, 2))
    elif feature == 'both':
        p_tfidf = TfidfVectorizer(stop_words=None, ngram_range=(1, 2))  # ngram as both unigram+bigram
        n_tfidf = TfidfVectorizer(stop_words=None, ngram_range=(1, 2))

    pos_tfidf = p_tfidf.fit_transform(pos_rev)
    neg_tfidf = n_tfidf.fit_transform(neg_rev)

    pos_count = pos_tfidf.toarray().sum(axis=0)
    pZip = zip(p_tfidf.get_feature_names(),  pos_count)
    positive = dict(pZip)                                   #create positive word dictionary with tf-idf

    neg_count = neg_tfidf.toarray().sum(axis=0)
    nZip = zip(n_tfidf.get_feature_names(), neg_count)      #create negative word dictionary with tf-idf
    negative = dict(nZip)

    vocab = set(p_tfidf.get_feature_names()).intersection(n_tfidf.get_feature_names()) #the words that exist both in neg and pos dictionaries

    presence_pos=[]
    pos_presence_count=[]
    presence_neg=[]
    neg_presence_count=[]
    for word in vocab:
        if positive[word] > negative[word]:                 #if a word's frequency is bigger in pos it's appereance means the review is probably positive
            presence_pos.append(word)                       #save the word
            pos_presence_count.append(positive[word])       #save the frequency of the word
        elif positive[word] < negative[word]:
            presence_neg.append(word)
            neg_presence_count.append(negative[word])

    pos_pres_dict = dict(zip(presence_pos, pos_presence_count))                 #create dictionary with word and frequency
    pos_sort = sorted(pos_pres_dict, key= pos_pres_dict.get, reverse=True)[:10] #sort the dictionary and find the biggest 10 frequencies
    neg_pres_dict = dict(zip(presence_neg, neg_presence_count))
    neg_sort = sorted(neg_pres_dict, key=neg_pres_dict.get, reverse=True)[:10]

    print("10 words whose presence most strongly predicts that the review is positive: ", pos_sort)
    print("10 words whose presence most strongly predicts that the review is negative: ", neg_sort)

    pos_set = set(p_tfidf.get_feature_names())
    neg_set = set(n_tfidf.get_feature_names())

    only_pos = list(pos_set.difference(neg_set))        #words that exist only in positive dictionary
    only_neg = list(neg_set.difference(pos_set))        #words that exist only in negative dictionary

    pos_only_words_count=[]
    for i in range(len(only_pos)):
        pos_only_words_count.append(positive[only_pos[i]])      #find the words' frequencies and hold

    pos_only_dict = dict(zip(only_pos, pos_only_words_count))
    pos_only_sort = sorted(pos_only_dict, key=pos_only_dict.get, reverse=True)[:10]     #create dictionary with word and its frequency, sort and find the 10 biggest frequencies

    neg_only_words_count = []
    for i in range(len(only_neg)):
        neg_only_words_count.append(negative[only_neg[i]])

    neg_only_dict = dict(zip(only_neg, neg_only_words_count))
    neg_only_sort = sorted(neg_only_dict, key=neg_only_dict.get, reverse=True)[:10]

    print("\n\n10 words whose absence most strongly predicts that the review is positive: ", neg_only_sort)
    print("10 words whose absence most strongly predicts that the review is negative: ", pos_only_sort)


feature = "bigram"  #change to "bigram" for bigram
                    #change to "unigram" for unigram
                    #change to "both" to apply both unigram and bigram
accuracy = NaiveBayes(feature)
print("Accuracy for " + feature + " is", accuracy,"\n\n")
tf_idf(feature, pos_rev, neg_rev) #prints 10 words takes a while to execute(like 10 secs)
                                  # I added these to my report so you can comment this line.
