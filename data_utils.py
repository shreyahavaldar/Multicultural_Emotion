##### Modules
import numpy as np 
import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import os
import fasttext
import fasttext.util
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification


#------ set prefix string to parent directory of fasttext models, i.e. "/sandata/Fasttext/fastText/" 
PATH_PREFIX = "/sandata/Fasttext/fastText/"

######### TODOs:(from Langchen)
#### Currently the code needs a very specific csv structure, i.e. number of columns
#### What I have in mind: combine the load_ekman_emotion and load_more_emotion into one function, Let's name it load_emotion
#### Add a feature that automatically detects the number of columns and the column names
#### So far we don't have headers for any of the csv files, so we need to add headers to the csv files
#### Let's name the headers as "semantic_translation", "official_translation", "English_correspondence", "Category"
#### The "Category" column is for the category of the emotion used for fixing projection axes, e.g. High arousal, Low arousal etc. 
#### Then we can combine the folders "Ekman_emotions" and "more_emotions" into one folder "Emotion", and put all the csv files in there.

#### Note: We might also have files like animals_P.csv, which is a list of animals in English instead of emotions. These emotions only have one columns
#### I think it is better we just gave up the usage of the current dictionary structure, and use a customized class instead.

class Emotion_Category:
    '''
    A class used to represent an emotion dimension. It has the following attributes: 
    - language: the language(or maybe the category, i.e. Russel, animals) of the emotion
    - unofficial_words_list: a list of strings of the unofficial words in some language
    - official_words_list: a list of strings of the official words in some language
    - English_correspondence_list: a list of strings of the English translation of emotion words
    - category_list: a list of strings of the category of the emotion words, this is used for generating axes of projection
    Do remember to name the class properly when you use it. For example, if you want to use the emotion words in English, you can name it as "English_emotion"
    If you want to use the emotion words in Russel, you can name it as "Russel_emotion"
    If you want to load some words of animals, you can name it as "animals"
    '''
    def __init__(self, filename, filepath=None, roberta_model_name=None, sent_model_name=None):
        dir_path = os.path.dirname(os.path.realpath('__file__'))
        if(filepath):
          path = dir_path + '/' + filepath + '{}.csv'.format(filename)
        else:
          path = dir_path + '/Dimension_emotions/' + '{}.csv'.format(filename)
        df_bonus = pd.read_csv(path, header= None)
        df_bonus = df_bonus.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        unofficial_words_list = None
        official_words_list = None
        English_correspondence_list = None
        category_list = None
        if len(df_bonus.columns) >= 1: 
            unofficial_words_list = np.array(df_bonus.iloc[:,0].values)   
        if len(df_bonus.columns) == 3:
            English_correspondence_list = np.array(df_bonus.iloc[:,1].values)         
        if len(df_bonus.columns) == 4:
            official_words_list = np.array(df_bonus.iloc[:,1].values)
            English_correspondence_list = np.array(df_bonus.iloc[:,2].values)
        category_list = np.array(df_bonus.iloc[:,-1].values)
        
        self.roberta_model_name = roberta_model_name ## name of roberta model (String)
        self.sent_model_name = sent_model_name ## name of sentence transformer model (String)
        self.language = filename ## String
        self.unofficial_words_list = unofficial_words_list ## np.array of strings
        self.official_words_list = official_words_list ## np.array of strings
        self.English_correspondence_list = English_correspondence_list ## np.array of strings
        self.category_list = category_list ## np.array of strings
        self.fasttext_official_embedding = None ## np.array of size(len(emotion_list), latent_dimension)
        self.fasttext_unofficial_embedding = None ## np.array of size(len(emotion_list), latent_dimension)
        self.bert_official_embedding = None ## np.array of size(len(emotion_list), latent_dimension)
        self.bert_unofficial_embedding = None ## np.array of size(len(emotion_list), latent_dimension)
        self.roberta_official_embedding = None ## np.array of size(len(emotion_list), latent_dimension)
        self.roberta_unofficial_embedding = None ## np.array of size(len(emotion_list), latent_dimension)
        self.phrases_unofficial = None ## np.array of strings
        self.phrases_official = None ## np.array of strings
        for item in [self.official_words_list, self.English_correspondence_list, self.category_list]:
            if item is not None:
                assert len(item) == len(self.unofficial_words_list) # The length of the emotion list is not consistent with the length of the category list
    
    
    
    def contextualization(self, buffer = None, official = False, ) -> None:
        '''
        Contextualize the emotion words using some certain buffer and returns some phrases.
        If the input buffer is None, then the contextualization will be based on hard-coded rules for each language
        The flag official determines in which attribute will the contextualizated phrases be stored
        If the flag official is True, then the phrases will be stored in the attribute self.phrases_official
        If the flag official is False, then the phrases will be stored in the attribute self.phrases_unofficial
        '''   
        phrase_list = []
        if buffer is None:
            if self.language == "English":
                buffer = "I feel "
            elif self.language == "Japanese":
                buffer = "私は"
            elif self.language == "Chinese":
                buffer = "我感到"
            elif self.language == "Spanish":
                buffer = "estoy "
            elif self.language == 'Hindi':
                buffer = "मैं {} हूं"
            else:
                raise NotImplementedError("The language {} is not supported yet".format(self.language))
        if official:
            for word in self.official_words_list:
                # if(self.language == 'Hindi'):
                #   print(buffer.format(word))
                #   phrase_list.append(buffer.format(word))  
                # else:
                #   phrase_list.append(buffer + word)
                phrase_list.append(buffer + word)
            self.phrases_official = np.array(phrase_list)
        else:
            for word in self.unofficial_words_list:
                # if(self.language == 'Hindi'):
                #   print(buffer.format(word))
                #   phrase_list.append(buffer.format(word))  
                # else:
                #   phrase_list.append(buffer + word)
                phrase_list.append(buffer + word)
            self.phrases_unofficial = np.array(phrase_list)



    def new_encoding(self, mode: str, mean_centered = True, official = False):
        '''
        Encode the phrases using SentenceTransformer, or encode the raw emotion words using fastText.
    
        WARNING: According to the source https://www.sbert.net/examples/training/multilingual/README.html, 
        'distiluse-base-multilingual-cased-v1' has better performence than the 'distiluse-base-multilingual-cased-v2'
        backend. However, v1 backend doesn't support Hindi and Japanese. If Hindi and Japanese are excluded in one round, 
        I suggest using the 'distiluse-base-multilingual-cased-v1' backend. 
    
        WARNING: For fasttext, I have used the reference here https://fasttext.cc/docs/en/crawl-vectors.html
        It is worth-noting that fasttext don't have a 'universal' model for all of the languaegs. Instead, I downloaded
        the pretrained model for each of the language and the word vectors will be generated per language.
    
        Parameters
        ----------
        mode : str
          A string that represents the encoding mode. Now support 'distiluse-base-multilingual-cased-v1' and
          'distiluse-base-multilingual-cased-v2' for phrases, and 'fastText' for emotions.
        centralized : bool
          A boolean that determines whether the embeddings are centralized or not.
          Note that the centralization is performed within each language. The method for centralization over all languages
          is not implemented.
        Returns
        -------
        embeddings_output: np.ndarray
          A numpy array of the encoded sentences
        language_list_for_output: list(str*)
          A list that contains the language of each encoded sentence
        emotions_list: list(str*)
          A list that contains the emotion of each encoded sentence
        '''
        if mode == 'bert':
            if(self.sent_model_name is None):
              embedding = new_Phrase_encoding(self, 'distiluse-base-multilingual-cased-v2', mean_centered, official)
            else:
              embedding = new_Phrase_encoding(self, self.sent_model_name, mean_centered, official)
            if official:
                self.bert_official_embedding = embedding
            else:
                self.bert_unofficial_embedding = embedding
        elif mode == 'distiluse-base-multilingual-cased-v1' or mode == 'distiluse-base-multilingual-cased-v2':
            embedding = new_Phrase_encoding(self, mode, mean_centered, official)
            if official:
                self.bert_official_embedding = embedding
            else:
                self.bert_unofficial_embedding = embedding
        elif mode == 'fastText':
            embedding = new_Word_encoding(self, mean_centered, official)
            if official:
                self.fasttext_official_embedding = embedding
            else:
                self.fasttext_unofficial_embedding = embedding
        elif mode == 'roberta':
            embedding = new_Roberta_encoding(self, mean_centered, official)
            if official:
                self.roberta_official_embedding = embedding
            else:
                self.roberta_unofficial_embedding = embedding
        else:
            raise ValueError('Encoding mode {} currently not supported'.format(mode))
    
def new_Phrase_encoding(Emotion: Emotion_Category, mode: str, mean_centered: bool , official: bool) -> np.ndarray:
    '''
    A subroutine for the encoding method.
    Encode the phrases using SentenceTransformer.
    
    WARNING: According to the source https://www.sbert.net/examples/training/multilingual/README.html, 
    'distiluse-base-multilingual-cased-v1' has better performence than the 'distiluse-base-multilingual-cased-v2'
    backend. However, v1 backend doesn't support Hindi and Japanese. If Hindi and Japanese are excluded in one round, 
    I suggest using the 'distiluse-base-multilingual-cased-v1' backend. 
    
    Parameters
    ----------
    Emotion : Emotion
      An Emotion object
    mode : str
      A string that represents the encoding mode. Now support 'distiluse-base-multilingual-cased-v1' and
      'distiluse-base-multilingual-cased-v2' for phrases.
    centralized : bool
      A boolean that determines whether the embeddings are centralized or not.
      Note that the centralization is performed within each language. The method for centralization over all languages
      is not implemented.
    official : bool
      A boolean that determines whether the official phrases or the unofficial phrases will be encoded.
    Returns
    -------
    embeddings_output: np.ndarray
      A numpy array of the encoded sentences
    '''
    model = SentenceTransformer(mode)
    sentences = list(Emotion.phrases_official) if official else list(Emotion.phrases_unofficial)
    embeddings = model.encode(sentences)
    if mean_centered:
        embeddings = embeddings - np.mean(embeddings, axis=0)
    return embeddings
  
def new_Word_encoding(Emotion: Emotion_Category, mean_centered: bool , official: bool) -> np.ndarray:
    '''
    A subroutine for the encoding method.
    Encode the emotion words using fastText.
    
    WARNING: For fasttext, I have used the reference here https://fasttext.cc/docs/en/crawl-vectors.html
    It is worth-noting that fasttext don't have a 'universal' model for all of the languaegs. Instead, I downloaded
    the pretrained model for each of the language and the word vectors will be generated per language.
    
    Parameters
    ----------
    Emotion : Emotion
      An Emotion object
    centralized : bool
      A boolean that determines whether the embeddings are centralized or not.
      Note that the centralization is performed within each language. The method for centralization over all languages
      is not implemented.
    official : bool
      A boolean that determines whether the official phrases or the unofficial phrases will be encoded.
    Returns
    -------
    embeddings_output: np.ndarray
      A numpy array of the encoded sentences
    '''
    language = Emotion.language
    embeddings = []
    if 'English' in language or language == 'jingle_jangle_processed':
        pre_trained_model = PATH_PREFIX + 'cc.en.300.bin'
    elif 'Hindi' in language:
        pre_trained_model = PATH_PREFIX + 'cc.hi.300.bin'
    elif 'Chinese' in language:
        pre_trained_model = PATH_PREFIX + 'cc.zh.300.bin'
    elif 'Spanish' in language:
        pre_trained_model = PATH_PREFIX + 'cc.es.300.bin'
    elif 'Japanese' in language:
        pre_trained_model = PATH_PREFIX + 'cc.ja.300.bin'
    else:
        raise ValueError('No pre-trained model for language: {}'.format(language))
        
    ft = fasttext.load_model(pre_trained_model)
    words = list(Emotion.official_words_list) if official else list(Emotion.unofficial_words_list)
    embeddings = np.concatenate([np.reshape(ft.get_word_vector(word), (1,300)) for word in words])
    if mean_centered:
        embeddings = embeddings -  np.mean(embeddings, axis=0)
    return embeddings    

def new_Roberta_encoding(Emotion: Emotion_Category, mean_centered: bool, official: bool) -> np.ndarray:  
    try:
      roberta_model = AutoModelForMaskedLM.from_pretrained(Emotion.roberta_model_name)
      roberta_tokenizer = AutoTokenizer.from_pretrained(Emotion.roberta_model_name)
    except:
      roberta_model = AutoModelForSequenceClassification.from_pretrained(Emotion.roberta_model_name)
      roberta_tokenizer = AutoTokenizer.from_pretrained(Emotion.roberta_model_name)
    # tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
    
    sentences = list(Emotion.phrases_official) if official else list(Emotion.phrases_unofficial)
    embeddings = []
    for sent in sentences:
      sent_embedding = roberta_encoding(sent, roberta_model, roberta_tokenizer)
      embeddings.append(sent_embedding)
        
    if mean_centered:
        embeddings = torch.concat([e - torch.mean(torch.stack(embeddings), dim=0) for e in embeddings], dim=0).numpy()
    else:
        embeddings = torch.concat(embeddings, dim=0).numpy()
    return embeddings
  
def roberta_encoding(sentence, roberta_model, roberta_tokenizer):
  '''
    Generate a mean-pooled sentence embedding using a pre-trained RoBERTa model.
    Args:
        sentence (str): The input sentence to be embedded.
        model (RobertaModel): The pre-trained RoBERTa model.
        tokenizer (RobertaTokenizer): The tokenizer corresponding to the RoBERTa model. 
    Returns:
        mean_pooled_embedding (torch.Tensor): The mean-pooled sentence embedding.
  '''
  inputs = roberta_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
  with torch.no_grad():
      outputs = roberta_model(**inputs, output_hidden_states=True)
  token_embeddings = outputs.hidden_states[-1]
  input_mask = inputs['attention_mask']
  sum_embeddings = torch.sum(token_embeddings * input_mask.unsqueeze(-1), dim=1)
  total_tokens = torch.clamp(input_mask.sum(1), min=1e-9)
  mean_pooled_embedding = sum_embeddings / total_tokens.unsqueeze(-1)
  return mean_pooled_embedding

def generate_averaged_points(category: str, mode: str, official: bool, args: list) -> np.ndarray:
    '''
    Generate the averaged points of the averaged embeddings of the category of emotions.
    
    Parameters
    ----------
    category : str
      A string that represents the first category of emotions. The string should be one of the following:
      'High arousal', 'Low arousal', 'High valence', 'Low valence', 'High dominance', 'Low dominance'
    mode : str
      A string that determines whether the bert, fasttext, or roberta embeddings will be used.
    official : bool
      A boolean that determines whether the official phrases or the unofficial phrases will be encoded.
    args : Dimension
      Emotion Dimension objects that may or may not have the same language.
    Returns
    -------
    projection_axes: np.ndarray
      A numpy array of the projection axes of the averaged embeddings of the four categories of emotions.
    '''
    embeddings = []
    for Dimension in args:
        loc = np.where(Dimension.category_list == category)
        if mode == 'bert':
            embedding = Dimension.bert_official_embedding[loc] if official else Dimension.bert_unofficial_embedding[loc]
        elif mode == 'fasttext':
            embedding = Dimension.fasttext_official_embedding[loc] if official else Dimension.fasttext_unofficial_embedding[loc]
        elif mode == 'roberta':
            embedding = Dimension.roberta_official_embedding[loc] if official else Dimension.roberta_unofficial_embedding[loc]
        embeddings.append(embedding)
    embeddings = np.concatenate(embeddings)
    return np.mean(embeddings, axis = 0)
  
def projection_helper(embeddings: np.ndarray, axes1: np.ndarray, axes2: np.ndarray):
    
    ## Compute the cosine 
    cos = np.dot(axes1,axes2) / np.linalg.norm(axes1, ord =2) / np.linalg.norm(axes2, ord=2)
    
    # Compute the projection
    x_axis = []
    tilted_y_axis = []
    for vector in embeddings:
        projection_1 = np.dot(vector, axes1) / np.linalg.norm(axes1, ord =2)
        projection_2 = np.dot(vector, axes2) / np.linalg.norm(axes2, ord =2)
        x_axis.append(projection_1)
        tilted_y_axis.append(projection_2)
    return np.asarray(x_axis), np.asarray(tilted_y_axis)



def project_points_onto_axes(points, x_point1, x_point2, y_point1, y_point2):
    """
    Projects a 2D numpy array of n-dimensional points onto orthogonal axes defined by two pairs of n-dimensional points,
    while ensuring that the axes are orthogonal and the intersection point of these two axes is the origin.

    Parameters:
    -----------
    points : numpy array
        A 2D numpy array with shape (N, D), where N is the number of points and D is the number of dimensions.
    x_point1 : numpy array
        The first n-dimensional point that defines the x-axis.
    x_point2 : numpy array
        The second n-dimensional point that defines the x-axis.
    y_point1 : numpy array
        The first n-dimensional point that defines the y-axis.
    y_point2 : numpy array
        The second n-dimensional point that defines the y-axis.

    Returns:
    --------
    numpy array
        A 1D numpy array with shape (N,), where each element contains the magnitude of the n-dimensional point
        projected onto the x-axis defined by the input points.
    numpy array
        A 1D numpy array with shape (N,), where each element contains the magnitude of the n-dimensional point
        projected onto the y-axis defined by the input points.
    """
    # Compute the unit vectors along the axes
    x_axis_vector = (x_point2 - x_point1) / 2
    y_axis_vector = (y_point2 - y_point1) / 2
    x_axis_vector = x_axis_vector / np.linalg.norm(x_axis_vector, ord=2)
    y_axis_vector = y_axis_vector / np.linalg.norm(y_axis_vector, ord=2)
    # Now length of the vector is 1
    cos = np.dot(x_axis_vector,y_axis_vector)
    # Project each point onto the x-axis and y-axis
    x_projection = []
    y_projection = []
    x_dist = []
    y_dist = []
    x_middle = (x_point1 + x_point2) / 2
    y_middle = (y_point1 + y_point2) / 2
    x1x = np.dot(x_point1 - x_middle , x_axis_vector) 
    x2x = np.dot(x_point2 - x_middle, x_axis_vector) 
    y1y = np.dot(y_point1 - y_middle, y_axis_vector) 
    y2y = np.dot(y_point2 - y_middle, y_axis_vector) 
    x1y = np.dot(x_point1 - y_middle, y_axis_vector) 
    x2y = np.dot(x_point2 - y_middle, y_axis_vector) 
    y1x = np.dot(y_point1 - x_middle, x_axis_vector) 
    y2x = np.dot(y_point2 - x_middle, x_axis_vector) 
    x1xtrue = x1x - x1y*cos
    x1ytrue = x1y - x1x*cos
    x2xtrue = x2x - x2y*cos
    x2ytrue = x2y - x2x*cos
    y1xtrue = y1x - y1y*cos
    y1ytrue = y1y - y1x*cos
    y2xtrue = y2x - y2y*cos
    y2ytrue = y2y - y2x*cos
    xorigin, yorigin = line_intersection((x1xtrue,x2xtrue,y1xtrue,y2xtrue), (x1ytrue,x2ytrue,y1ytrue,y2ytrue))
    x_negative_scale = np.abs(x1xtrue - xorigin)
    x_positive_scale = np.abs(x2xtrue - xorigin)
    y_negative_scale = np.abs(y1ytrue - yorigin)
    y_positive_scale = np.abs(y2ytrue - yorigin)
    for point in points:
        x_proj = np.dot(point - x_middle , x_axis_vector)  
        y_proj = np.dot(point - y_middle , y_axis_vector) 
        true_y = (y_proj - x_proj*cos) - yorigin
        true_x = (x_proj - y_proj*cos) - xorigin
        if true_x < 0:
            true_x = true_x / x_negative_scale
        else:
            true_x = true_x / x_positive_scale
        if true_y < 0:
            true_y = true_y / y_negative_scale
        else:
            true_y = true_y / y_positive_scale
        x_projection.append(true_x)
        y_projection.append(true_y)
        x_dist.append(np.linalg.norm(point - x_proj*x_axis_vector, ord=2))
        y_dist.append(np.linalg.norm(point - y_proj*y_axis_vector, ord=2))

    # Return the magnitudes of the projections as numpy arrays
    return np.array(x_projection), np.array(y_projection), np.array(x_dist), np.array(y_dist)
      
      
def Phrase_encoding(Phrase_Dict: dict, encoding_mode: str, centralized: bool) -> tuple:
    '''
    A subroutine for the encoding method.
    Encode the phrases using SentenceTransformer.
    
    WARNING: According to the source https://www.sbert.net/examples/training/multilingual/README.html, 
    'distiluse-base-multilingual-cased-v1' has better performence than the 'distiluse-base-multilingual-cased-v2'
    backend. However, v1 backend doesn't support Hindi and Japanese. If Hindi and Japanese are excluded in one round, 
    I suggest using the 'distiluse-base-multilingual-cased-v1' backend. 
    
    Parameters
    ----------
    Phrase_Dict : dict
      A Phrase dictionary of form Dict[(str, Tuple[List(str*), List(str*)])].
    encoding_mode : str
      A string that represents the encoding mode. Now support 'distiluse-base-multilingual-cased-v1' and
      'distiluse-base-multilingual-cased-v2' for phrases.
    centralized : bool
      A boolean that determines whether the embeddings are centralized or not.
      Note that the centralization is performed within each language. The method for centralization over all languages
      is not implemented.
    Returns
    -------
    embeddings_output: np.ndarray
      A numpy array of the encoded sentences
    language_list_for_output: list(str*)
      A list that contains the language of each encoded sentence
    emotions_list: list(str*)
      A list that contains the emotion of each encoded sentence
    '''
    model = SentenceTransformer(encoding_mode)
    embeddings_list = []
    emotions_list = []
    language_list_for_output = []
    language_list = list(Phrase_Dict.keys())
    for language in language_list:
        sentences = Phrase_Dict[language][0]
        embeddings = model.encode(sentences)
        if centralized:
            embeddings = embeddings -  embeddings.mean()
        ### store the variables in lists
        embeddings_list.append(embeddings)
        emotions_list = emotions_list + list(Phrase_Dict[language])[1]
        for i in range(len(sentences)):
            language_list_for_output.append(language)
    embeddings_output = np.concatenate(embeddings_list)
    total_points = embeddings_output.shape[0]
    assert total_points == len(emotions_list)
    assert total_points == len(language_list_for_output)
    return embeddings_output, np.array(language_list_for_output), emotions_list
  

def Word_encoding(Emotion_Dict: dict, centralized: bool) -> tuple:
    '''
    A subroutine for the encoding method.
    Encode the emotion words using fastText.
    
    WARNING: For fasttext, I have used the reference here https://fasttext.cc/docs/en/crawl-vectors.html
    It is worth-noting that fasttext don't have a 'universal' model for all of the languaegs. Instead, I downloaded
    the pretrained model for each of the language and the word vectors will be generated per language.
    
    Parameters
    ----------
    Emotion_Dict : dict
      A dictionary of Ekman emotions. The key-value pair is of format (str, pandas.DataFrame)
      A dictionary of language-emotion pairs. The DataFrame Object should have three columns. The 0th column is 
      the semantic translation of the Ekman emotions, the 1st column is the official translation of the Ekman 
      emotions, the 2nd column is the English ekman emotions.
    centralized : bool
      A boolean that determines whether the embeddings are centralized or not.
      Note that the centralization is performed within each language. The method for centralization over all languages
      is not implemented.
    Returns
    -------
    embeddings: np.ndarray
      A numpy array of the encoded sentences
    languages: list(str*)
      A list that contains the language of each encoded sentence
    emotions: list(str*)
      A list that contains the emotion of each encoded sentence
    '''
    embeddings = []
    languages = []
    emotions = []
    for language, df in Emotion_Dict.items():
        if language == 'English' or language == 'jingle_jangle_processed':
            pre_trained_model = 'cc.en.300.bin'
        elif language == 'Hindi':
            pre_trained_model = 'cc.hi.300.bin'
        elif language == 'Chinese':
            pre_trained_model = 'cc.zh.300.bin'
        elif language == 'Spanish':
            pre_trained_model = 'cc.es.300.bin'
        elif language == 'Japanese':
            pre_trained_model = 'cc.ja.300.bin'
        else:
            raise ValueError('No pre-trained model for language: {}'.format(language))
        
        ft = fasttext.load_model(pre_trained_model)
        embed, emo = Fasttext_encoding(df, centralized, ft)
        embeddings.append(embed)
        emotions = emotions + emo
        for i in range(len(emo)):
            languages.append(language)
    embeddings = np.concatenate(embeddings)
    total_points = embeddings.shape[0]
    assert total_points == len(emotions)
    assert total_points == len(languages)
    return embeddings, np.array(languages), emotions


def Fasttext_encoding(df: pd.DataFrame, centralized: bool, ft) -> tuple:
    '''
    A subroutine for the encoding method.
    Encode the emotion words using fastText.
    
    WARNING: For fasttext, I have used the reference here https://fasttext.cc/docs/en/crawl-vectors.html
    It is worth-noting that fasttext does not have a 'universal' model for all of the languaegs. Instead, I downloaded
    the pretrained model for each of the language and the word vectors will be generated per language.
    
    Parameters
    ----------
    df : pd.DataFrame
      A DataFrame object with three columns. The 0th column is the semantic translation of the 
      emotions, the 1st column is the official translation of emotions, the 2nd column is the English emotions.
    centralized : bool
      A boolean that determines whether the embeddings are centralized or not.
      Note that the centralization is performed within each language. The method for centralization over all languages
      is not implemented.
    ft : fasttext.model
      A pre-trained fasttext model
    Returns
    -------
    embeddings: np.ndarray
      A numpy array of the encoded words
    emotions: list(str*)
      A list that contains the emotion of each encoded words
    '''
    ## The english emotions
    emotions = list(df.loc[:,2])
    ## The coversational words
    words = list(df.loc[:,0])
    embeddings = np.concatenate([np.reshape(ft.get_word_vector(word), (1,300)) for word in words])
    if centralized:
        embeddings = embeddings - np.mean(embeddings)
    total_points = embeddings.shape[0]
    assert total_points == len(words)
    assert total_points == len(emotions)
    return embeddings, emotions

def line_intersection(x_pts, y_pts):
    
    line1 = ((x_pts[0], y_pts[0]), (x_pts[1], y_pts[1]))
    line2 = ((x_pts[2], y_pts[2]), (x_pts[3], y_pts[3]))

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y