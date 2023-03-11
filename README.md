# Language-Modelling
Language models built for English, using statistical methods for calculating perplexities. The smoothing techniques that were implemented for the models were Witten Bell and Kneser Ney.

### How to run language_model.py

``` py language_model.py <smoothing_technique> <path_to_corpus>```

You will be given a prompt to enter a sentence, and the model would output the probability of that sentence appearing in the corpus

### Things considered during tokenization 
- Punctuation marks are removed
- All words are converted to lower case
- All numbers are replaced with the token "NUM"
- Cases with can't, won't were handled separately
- Cases with 's, 're, 've were handled separately
- Cases with 'm were handled separately
- Cases with 'll were handled separately
- Cases with 'd were handled separately
- Cases with 't were handled separately
- Hyphenated words were considered as one word, this was done especially because of the presence of words like to-morrow in the corpus 
- Excess space and hyphens were also removed individually 

### Note 
Rest of the important information has either been provided as comments in the code itself or in the report
