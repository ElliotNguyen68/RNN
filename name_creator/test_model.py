from utils import *
from rnn import RNN
rnn=RNN(n_categories,n_letters,128)
rnn.load_state_dict(torch.load("D:/visual studio/my_python/RNN/name_creator/rnn(w2w).pth"))
max_length = 20

# Sample from a category and starting letter
def sample(category, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = rnn.create_init_hidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name

# Get multiple samples from one category and multiple starting letters
def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))

samples('Japanese', 'RUS')

samples('German', 'GER')

samples('Vietnamese', 'LMTT')

samples('Chinese', 'CHI')