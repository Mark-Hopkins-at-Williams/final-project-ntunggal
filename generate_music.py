import torch 
import torch.nn.functional as F
import numpy as np 
import music21 
import json
from buildLSTM import  SEQUENCE_LENGTH
from buildLSTM import LSTMModel
from get_data import DICTIONARY 

MODEL_PATH = "model_final.pkl" 
OUTPUT_SIZE = 38 

class GenerateMelody():
    '''
    Class for generating melodies using a trained LSTM model. 
    ''' 
    def __init__(self, music_to_int_dictionary): 
        int_to_music_dictionary = {value: key for key, value in music_to_int_dictionary.items()}
        self.music_to_int_dictionary = music_to_int_dictionary
        self.start_delimiters = ['/'] * SEQUENCE_LENGTH  
        self.int_to_music_dictionary = int_to_music_dictionary

    def generate(self, seed, number_of_steps, max_sequence_length, temperature): 
        '''
        Generates a melody from a given seed using pre-trained model.
        
        Parameters:
            seed (str): The initial sequence to start the generation.
            number_of_steps (int): The number of steps to generate.
            max_sequence_length (int): The maximum length of the input sequence.
            temperature (float): Controls the randomness of predictions by scaling the logits before applying softmax.
        
        Returns:
            melody (list of str): The generated melody as a list of musical symbols.
        '''

        seed = seed.split()
        melody = seed   
        seed = self.start_delimiters + seed
        seed = [self.music_to_int_dictionary[symbol] for symbol in seed]

        model = LSTMModel()
        state_dict = torch.load(MODEL_PATH) 
        model.load_state_dict(state_dict) 
        model.eval()

        print("We are now generating music... ")
        for current_step in range(number_of_steps):
            seed = seed[-max_sequence_length:]
            seed_tensor = torch.tensor(seed)
            seed_tensor = seed_tensor.reshape(max_sequence_length, 1).long() 
            onehoted_seed = torch.zeros(max_sequence_length, OUTPUT_SIZE).scatter_(1, seed_tensor, 1)
            one_batch_onehot_seed = onehoted_seed.reshape(1, max_sequence_length, OUTPUT_SIZE)
            probabilities = model(one_batch_onehot_seed)
            index_output_from_sampling = self.sample_with_temperature(probabilities, temperature)
            seed.append(index_output_from_sampling)
            output_symbol = self.int_to_music_dictionary[index_output_from_sampling]
            if output_symbol == "/": # if model predicts character "/", then we stop  
                continue 
            melody.append(output_symbol)
        return melody 


    def sample_with_temperature(self, probabilites, temperature):
        '''
        Samples an index from a probability array using temperature sampling.
        
        Parameters:
            probabilities (torch.Tensor): Tensor of probabilities for each possible output.
            temperature (float): Controls the randomness of predictions by scaling the logits before applying softmax.
        
        Returns:
            index (int): The selected index based on the sampling.
        '''
        probabilites = probabilites.squeeze() 
        predictions = torch.log(probabilites) / temperature
        probabilites = F.softmax(predictions, dim = -1)
        choices = range(len(probabilites))
        probabilites = probabilites.detach().numpy()
        index = np.random.choice(choices, p = probabilites)
        return index
    

    def save_melody(self, melody, step_duration = 0.5, 
                    format="midi", file_name="mel.mid"):
        '''
        Converts a melody into a MIDI file.
        
        Parameters:
            melody (list of str): The melody to convert.
            step_duration (float): Duration of each time step in quarter length.
            format (str): The format to save the file in (default is MIDI).
            file_name (str): The name of the file to save.
        '''
        stream = music21.stream.Stream()
        start_symbol = None
        step_counter = 1
        for i, symbol in enumerate(melody):
            if symbol != "_" or i + 1 == len(melody):
                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1
                    if start_symbol == "r": ## rest handling 
                        music21_event = music21.note.Rest(quarterLength=quarter_length_duration)
                    else: # note handling 
                        music21_event = music21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)
                    stream.append(music21_event)
                    step_counter = 1
                start_symbol = symbol
            else:
                step_counter += 1
        stream.write(format, file_name)

if __name__ == "__main__":    
    with open(DICTIONARY, 'r') as file:
        music_to_int_dictionary = json.load(file)

    melody_generator = GenerateMelody(music_to_int_dictionary)
    seed = "65 _ _ _ 65 _ 69"
    melody = melody_generator.generate(seed, 100, SEQUENCE_LENGTH, 100)
    melody_generator.save_melody(melody) 
