import torch
import torch.nn.functional as F
from torch.utils.data import Dataset 
import json 
from music_preprocessing import preprocess_midi_files, SEQUENCE_LENGTH 


DICTIONARY = 'music_to_integer_dictionary.json'


############################################################
############# DO NOT CHANGE ANYTHING BELOW  ################ 
############################################################


def create_single_file_dataset(preprocessed_output_dictionary):
    '''
    Concatenates preprocessed music pieces into a single string with delimiters. 
    Parameters:
        preprocessed_output_dictionary (dict): A dictionary with preprocessed music pieces.
        Note that by construction, this function takes the output from preprocess_midi_files
        in music_preprocessing.py

    Output:
        str: A single string of concatenated music pieces.
    ''' 
    music_delimiter = "/ " * SEQUENCE_LENGTH
    output = ""
    ## load through encoaded music from 
    for key, music in preprocessed_output_dictionary.items():
        output = output + music + " " + music_delimiter
    output = output[:-1] 
    return output 


def get_music_to_int_dictionary(all_music): 
    '''
    Creates a dictionary that maps musical notes to integers.
    
    Parameters:
        all_music (str): A string of all music concatenated together.
    
    Output:
        dict: A dictionary mapping musical notes to integers. 
    ''' 
    dictionary_from_music_to_integers = {} 
    all_music = all_music.split() 
    list_of_vocabulary = list(set(all_music)) ## get the size of vocabulary
    dictionary_from_music_to_integers = {symbol: index for index, symbol in enumerate(list_of_vocabulary)}  
    path = DICTIONARY 
    with open(path, 'w') as file:
        json.dump(dictionary_from_music_to_integers, file, indent=4)
    return dictionary_from_music_to_integers


def convert_music_to_int(all_music, music_to_int_dictionary):
    '''
    Converts a string of music notes to their integer representations.
    
    Parameters:
        all_music (str): A string of all music concatenated together.
        music_to_int_dictionary (dict): A dictionary mapping musical notes to integers.
    
    Output:
        list: A list of integers representing the music notes.
    ''' 
    dictionary_from_music_to_integers = music_to_int_dictionary 
    list_of_all_music = all_music.split()
    final_output = [] 
    for character in list_of_all_music: 
        final_output.append(dictionary_from_music_to_integers[character])
    return final_output 


def generate_training_data(sequence_length = SEQUENCE_LENGTH):
    '''
    Generates training and target sequences from the encoded music data.
    
    Parameters:
        sequence_length (int): The length of each training sequence.
    
    Returns:
        tuple: Training data (one-hot encoded), target data, number of vocabulary,
               and the music-to-integer dictionary.
    ''' 
    preprocessed_output_dictionary = preprocess_midi_files() ## first preprocess the data 
    all_music = create_single_file_dataset(preprocessed_output_dictionary) ## second, create an entire string of music  
    music_to_int_dictionary = get_music_to_int_dictionary(all_music) ## then passing them through to create encodings 
    list_of_music_encoded = convert_music_to_int(all_music, music_to_int_dictionary) ## this gives encoding 

    inputs = []
    targets = []

    num_sequences = len(list_of_music_encoded) - sequence_length
    for i in range(num_sequences):
        inputs.append(list_of_music_encoded[i:sequence_length+i])
        targets.append(list_of_music_encoded[sequence_length+i])
    print(f'there are {num_sequences} of training data')
    return inputs, targets, music_to_int_dictionary 


class trainset(Dataset):
    def __init__(self, output_size, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.output_size = output_size 

    def __getitem__(self, index):
        input_data = self.inputs[index]
        train_data = torch.tensor(input_data)
        train_data = train_data.reshape(SEQUENCE_LENGTH, 1)
        train_data = torch.zeros(SEQUENCE_LENGTH, self.output_size).scatter_(1, traindata, 1)
        target_data = self.targets[index]
        target_data = torch.tensor(target_data)
        return train_data, target_data

    def __len__(self):
        return len(self.targets)

