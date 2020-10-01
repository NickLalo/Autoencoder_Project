import math
import os


# change directory to Basic_PyTorch_Model for easier access to contents
os.chdir("../")


"""Generate Binary Data"""
# zero up to input including input.  Ex: 15 means 16 values will be produced with 4 bits.  From zero to input.
zero_up_to = 15

number_of_bits = math.ceil(math.log2(zero_up_to + 0.1))
data_list = []
input_number_list = [x for x in range(zero_up_to + 1)]
max_val = 2 ** number_of_bits - 1

for input_number in input_number_list:
    bit_list = [0 for _ in range(number_of_bits)]

    index = number_of_bits - 1
    while input_number > 0.5:
        remainder = input_number % 2
        input_number = int(input_number / 2)
        if remainder == 1:
            bit_list[index] = 1
        index -= 1

    data_list.append(bit_list)

# check to make sure we did it right even though the index might not be needed for writing to file
for index, bit_list in enumerate(data_list):
    print(f"{bit_list}, {index}")

"""Write the binary dataset to csv for unsupervised learning"""
if "Data" not in os.listdir():
    os.makedirs('Data')

with open("Data/binary_dataset_no_labels.csv", "w") as outfile:
    for index, bit_list in enumerate(data_list):
        for num in range(len(bit_list)):
            outfile.write(f"input_{num},")
        outfile.write(f"target")
        outfile.write(f"\n")
        break

    # inflate the number of data points even though they are copies
    # attempting to train a model with not enough training data can cause errors in creating batches
    for num in range(15):
        for index, bit_list in enumerate(data_list):
            bit_list = [str(x) for x in bit_list]
            outfile.write(f"{','.join(bit_list)},{index}\n")
