import re
import sys 
import math 
import random
from collections import defaultdict

smoothing = sys.argv[1]
corpus = sys.argv[2]

def read_file():
    with open(corpus, 'r') as f:
        data = f.readlines()
    return data

def sent_tokenization(data):
    # realised that readlines isn't enough to tokenize the data into sequences of sentences because of uneven spacing present throughtout the text - tried my best to tokenize properly - still have doubts about the accuracy of the tokenization
    big_data = ""

    for sentence in data:
        sentence = re.sub(r'\n', ' ', sentence)
        if sentence != " ":
            big_data += sentence
    
    sentences = []

    big_data = big_data.strip()
    big_data = " " + big_data + " "
    
    # taking care of sentences where they aren't really ending because of words like Mr or Mrs or Ms or any other word that starts with a capital letter
    big_data = re.sub(r'(Mr|Ms|Mrs)\.', r'\1<prd>', big_data)

    # converting all things in the form of an url to the URL tag - so that websites are addressed in a uniform manner
    big_data = re.sub(r'\w+:\/\/\S+', '<URL>', big_data)

    # taking care of numbers with decimals
    big_data = re.sub(r'([0-9])\.([0-9])', r'\1<prd>\2', big_data)

    # taking care of ellipses in the text if any
    big_data = re.sub(r'\.\+', ' ', big_data)

    # taking care of names such as R. B. William and all
    big_data = re.sub(r'\s([A-Z])\.\ ', r' \1<prd>\ ', big_data)

    # convering the words to denote italics to remove the italics underscores
    big_data = re.sub(r'\_(\w+)\_', r'\1', big_data)

    big_data = re.sub(r'(\w)\.(\w)\.(\w)\.', r'\1<prd>\2<prd>\3<prd>', big_data)
    big_data = re.sub(r'(\w)\.(\w)\.', r'\1<prd>\2<prd>', big_data)
    big_data = re.sub(r'\ (\w)\.', r' \1<prd>', big_data)

    if "\"" in big_data: 
        big_data = big_data.replace(".\"","\".")
    
    if "!" in big_data:
        big_data = big_data.replace("!\"","\"!")
    
    if "?" in big_data:
        big_data = big_data.replace("?\"","\"?")
    
    big_data = big_data.replace(".",".<stop>")
    big_data = big_data.replace("?","?<stop>")
    big_data = big_data.replace("!","!<stop>")

    big_data = big_data.replace("<prd>",".")

    sentences = big_data.split("<stop>")
    sentences = sentences[:-1]

    sentences = [s.strip() for s in sentences]

    return sentences

def tokenization(text):
    # this is a function which will tokenize the sentences and actually give us a list of words in the sentence 
    clean_text = text

    # converting the text to lower case to introduce uniformity
    clean_text = clean_text.lower()

    # converting all things in the form of a hashtag to the HASHTAG tag
    clean_text = re.sub(r'\#\w+', '<HASHTAG>', clean_text)

    # converting all things in the form of a mention to the MENTION tag
    clean_text = re.sub(r'\@\w+', '<MENTION>', clean_text)

    # removing the chapter numbers from text - this is a very special case and is sort of a text cleaning work that is being done in this case for the given piece of text
    clean_text = re.sub(r'\[ [0-9]+ \]', '', clean_text)

    # convering anything that is a number to the number tag
    clean_text = re.sub(r'\d+\.*\d*', '<NUM>', clean_text)

    # convering things in the form of NUM(st, nd, rd, th) to just NUM - improving on the num tag
    clean_text = re.sub(r'<NUM>(?:st|nd|rd|th)', '<NUM>', clean_text)

    # taking care of &c that is seen in a number of places - don't really know what it does, so removing it here 
    clean_text = re.sub(r'(?:\.|\,)\ \&c', '', clean_text)

    # taking care of words with 'll form 
    clean_text = re.sub(r'\'ll\ ', ' will ', clean_text)

    # taking care of the word can't and won't 
    clean_text = re.sub(r'\ can\'t\ ', 'can not', clean_text)
    clean_text = re.sub(r'\ won\'t\ ', 'would not', clean_text)

    # taking care of words with n't form 
    clean_text = re.sub(r'n\ t\ ', ' not ', clean_text)

    # taking care of words with 're form 
    clean_text = re.sub(r'\'re\ ', ' are ', clean_text)

    # taking care of words with 'm form 
    clean_text = re.sub(r'\'m\ ', ' am ', clean_text)

    # taking care of the 've form 
    clean_text = re.sub(r'\'ve\ ', ' have ', clean_text)

    # taking care of the 'd form
    clean_text = re.sub(r'\'d\ ', ' would ', clean_text)

    # didn't really do anything for the 's form cause it might denote possession and in this corpus it is more common for 's to denote possession - so just separated the 's forms 
    clean_text = re.sub(r'\'s\ ', ' \'s', clean_text)

    # hyphenated words - while playing around with the corpus, realised it'd make more sense to combine them than to break them down due to presence of words like head-ache or to-day - doing it twice for words like mother-in-law : tried fancy stuff, didn't really work out ono
    clean_text = re.sub(r'(\w+)\-(\w+)', r'\1\2', clean_text)
    clean_text = re.sub(r'(\w+)\-(\w+)', r'\1\2', clean_text)

    # excess hypens need to go now 
    clean_text = re.sub(r'\-+', ' ', clean_text)

    # padding punctuation characters to ensure cute tokenization
    # clean_text = re.sub(r'\s*([,.?!;:"()])\s*', r' \1 ', clean_text)

    # so in some n gram models, punctuation is included while in some it is not (like google's model), and we can also choose to remove it completely - we would just need to change the substitution string
    clean_text = re.sub(r'\s*([,.?!;:"()—_\\])\s*', r' ', clean_text)

    # getting rid of all the extra spaces that there might be present
    clean_text = re.sub(r'\s+', ' ', clean_text)

    # getting rid of trainling spaces 
    clean_text.strip()

    tokens = [token for token in clean_text.split() if token != ""]
    tokens = ['<s>'] * (3) + tokens + ['</s>'] * (3)

    return tokens

unigram = defaultdict(int)
bigram = defaultdict(lambda : defaultdict(int))
trigram = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
fourgram = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(int))))

bigram_rev = defaultdict(lambda : defaultdict(int))
trigram_rev = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))
fourgram_rev = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(int)))) 

uni_count = 0
bi_count = 0
tri_count = 0
four_count = 0

def generate_grams(sentence):
    # this function takes a sentence and generates a unigram, bigram, trigram and fourgram for the sentence and then adds it to the global dictionaries
    global uni_count, bi_count, tri_count, four_count

    for word in sentence:
        if word not in unigram:
            uni_count += 1

        unigram[word] += 1
    
    for i in range(len(sentence) - 1):
        bigram_key = sentence[i]
        next_word = sentence[i + 1]

        if bigram_key not in bigram:
            bi_count += 1
        elif next_word not in bigram[bigram_key]:
            bi_count += 1

        bigram[bigram_key][next_word] += 1
        bigram_rev[next_word][bigram_key] += 1

    for i in range(len(sentence) - 2):
        trigram_key = sentence[i]
        next_word = sentence[i + 1]
        next_next_word = sentence[i + 2]

        if trigram_key not in trigram:
            tri_count += 1
        elif next_word not in trigram[trigram_key]:
            tri_count += 1
        elif next_next_word not in trigram[trigram_key][next_word]:
            tri_count += 1

        trigram[trigram_key][next_word][next_next_word] += 1
        trigram_rev[next_next_word][next_word][trigram_key] += 1

    for i in range(len(sentence) - 3):
        fourgram_key = sentence[i]
        next_word = sentence[i + 1]
        next_next_word = sentence[i + 2]
        next_next_next_word = sentence[i + 3]

        if fourgram_key not in fourgram:
            four_count += 1
        elif next_word not in fourgram[fourgram_key]:
            four_count += 1
        elif next_next_word not in fourgram[fourgram_key][next_word]:
            four_count += 1 

        fourgram[fourgram_key][next_word][next_next_word][next_next_next_word] += 1
        fourgram_rev[next_next_next_word][next_next_word][next_word][fourgram_key] += 1

vocabulary = set()
word_count = {}

def count_words(clean_sent):
    for sentence in clean_sent:
        for word in sentence:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

def create_vocabulary():
    global unkcount, newunkcount
    unkcount = 0
    newunkcount = 0

    for word in word_count:
        if word_count[word] >= 1:
            vocabulary.add(word)
        if word_count[word] == 1:
            unkcount += 1
        if word_count[word] == 2:
            newunkcount += 1

def out_of_vocabulary(word):
    # this function checks if a word is present in the vocabulary or not
    if word in vocabulary:
        return word
    else:
        return '<UNK>'

def replace_oov(sentence):
    # this function replaces all the words that are not present in the vocabulary with the <UNK> token
    for i in range(len(sentence)):
        sentence[i] = out_of_vocabulary(sentence[i])

    return sentence

kn_prob = {}

def kneser_ney(n, tokens):
    # we know that for highest order n = 4 and for the rest we need to do the stuff for lower order things and not the highest order wala thing 
    if tokens in kn_prob:
        return kn_prob[tokens]
    else:
        tokens = tokens.split()
        word = tokens[-1]
        context = tokens[:-1]

        # discount = unkcount / (unkcount + 2 * newunkcount)
        discount = 0.75

        if (n == 4):
            p_num = max(fourgram[context[0]][context[1]][context[2]][word] - discount, 0)
            p_den = trigram[context[0]][context[1]][context[2]]
            try:
                p = p_num / p_den
            except ZeroDivisionError:
                p = 0
            
            lmd_num = discount * len(fourgram[context[0]][context[1]][context[2]])
            lmd_den = trigram[context[0]][context[1]][context[2]]
            try:
                lmd = lmd_num / lmd_den
            except ZeroDivisionError:
                lmd = 1

            prob = p + (lmd * kneser_ney(n - 1, ' '.join(tokens[1:])))
        elif (n == 3):
            count = len(fourgram_rev[word][context[0]][context[1]])
            p_num = max(count - discount, 0)
            p_den = four_count
            try:
                p = p_num / p_den
            except ZeroDivisionError:
                p = 0

            lmd_num = discount * len(trigram[context[0]][context[1]])
            lmd_den = bigram[context[0]][context[1]]
            try:
                lmd = lmd_num / lmd_den
            except ZeroDivisionError:
                lmd = 1

            prob = p + (lmd * kneser_ney(n - 1, ' '.join(tokens[1:])))
        elif (n == 2):
            count = len(trigram_rev[word][context[0]])
            p_num = max(count - discount, 0)
            p_den = tri_count
            try:
                p = p_num / p_den
            except ZeroDivisionError:
                p = 0

            lmd_num = discount * len(bigram[context[0]])
            lmd_den = unigram[context[0]]
            try:
                lmd = lmd_num / lmd_den
            except ZeroDivisionError:
                lmd = 1

            prob = p + (lmd * kneser_ney(n - 1, ' '.join(tokens[1:])))
        else:
            # unigram 
            p_num = len(bigram_rev[word])
            p_den = bi_count
            try:
                p = p_num / p_den
            except ZeroDivisionError:
                p = 0

            correction = discount / word_count
            prob = p + correction
        
    kn_prob[' '.join(tokens)] = prob
    return prob

wb_prob = {}

def witten_bell(n, tokens):
    if tokens in wb_prob:
        return wb_prob[tokens]
    else:
        tokens = tokens.split()
        word = tokens[-1]
        context = tokens[:-1]

        if(n == 4):
            pseq = fourgram[context[0]][context[1]][context[2]][word]
            total_cases = trigram[context[0]][context[1]][context[2]]
            n1 = len(fourgram[context[0]][context[1]][context[2]])

            prob = (pseq + (n1 * witten_bell(n-1, ' '.join(tokens[1: ])))) / (total_cases + n1)
        elif(n == 3):
            pseq = trigram[context[0]][context[1]][word]
            total_cases = bigram[context[0]][context[1]]
            n1 = len(trigram[context[0]][context[1]])

            prob = (pseq + (n1 * witten_bell(n-1, ' '.join(tokens[1: ])))) / (total_cases + n1)
        elif(n == 2):
            pseq = bigram[context[0]][word]
            total_cases = unigram[context[0]]
            n1 = len(bigram[context[0]])

            prob = (pseq + (n1 * witten_bell(n-1, ' '.join(tokens[1: ])))) / (total_cases + n1)
        elif(n == 1):
            pseq = unigram[word]
            total_cases = word_count
            n1 = 0

            prob = pseq / total_cases

        wb_prob[''.join(tokens)] = prob
        return prob

def create_data(clean_sent):
    clean_sent.sort()
    sentences = len(clean_sent)
    percent = 1 - (1000 / sentences) 

    random.seed(23)
    random.shuffle(clean_sent)

    train = clean_sent[:int(percent * sentences)]
    test = clean_sent[int(percent * sentences):]

    return train, test

def calculate_probability(tokens):
    n = 4

    probability = 1

    for i in range(len(tokens) - n + 1):
        token = tokens[i:i+n]
        token = " ".join(token)
        if smoothing == "w":
            probability *=  witten_bell(n, token)
        else:
            probability *= kneser_ney(n, token)

    return probability

def calculate_perplexity(sentence):

    probability = calculate_probability(sentence)
    try:
        perplexity = pow(1/probability, 1/len(sentence))
    except ZeroDivisionError:
        perplexity = 0

    return perplexity

def main():
    data = read_file()
    sentences = sent_tokenization(data)

    clean_sent = []
    
    for sentence in sentences:
        clean_sent.append(tokenization(sentence))
    
    train, test = create_data(clean_sent)

    count_words(train)
    create_vocabulary()

    global word_count
    word_count = 0

    for sentence in train:
        sentence = replace_oov(sentence)
        generate_grams(sentence)
        word_count += len(sentence)
    
    unigram['<UNK>'] = unkcount
    
    # avg_perp = 0
    # sent = 0

    # output = ""
    # for sentence in train:
    #     perp = calculate_perplexity(sentence)
    #     avg_perp += perp
    #     sent += 1
    #     output += (" ".join(sentence)) + "\t" + str(perp) + "\n"
    
    # avg_perp = avg_perp / sent
    # output = str(avg_perp) + "\n" + output

    # f = open("2021114002_LM4_train-perplexity", "w")
    # f.write(output)
    # f.close()

    # avg_perp = 0
    # sent = 0

    # output = ""
    # for sentence in test:
    #     sentence = replace_oov(sentence)
    #     perp = calculate_perplexity(sentence)
    #     if perp != 0 and perp != float('inf'): #uhh ok why the fuck is this happening but it works if i just ignore it so yay?
    #         avg_perp += perp
    #         sent += 1
    #         output += (" ".join(sentence)) + "\t" + str(perp) + "\n"

    # avg_perp = avg_perp / sent
    # output = str(avg_perp) + "\n" + output

    # f = open("2021114002_LM4_test-perplexity", "w")
    # f.write(output)
    # f.close()
    
    sentence = input("input sentence: ")
    sentence = tokenization(sentence)
    sentence = replace_oov(sentence)
    print(calculate_probability(sentence))
    
if __name__ == "__main__":
    main()