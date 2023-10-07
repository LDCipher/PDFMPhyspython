class DataSet(object):
    def __init__(self):
        self.x = []
        self.Qs = []
        self.flavours = []
        self.F = []

def get_array(line):
    temp = line.strip()
    temp = temp.split(' ')
    return temp


def process_file(file_path):
    All_x = []
    All_Qs = []
    x = []
    Qs = []
    ## if (line == '---')
    AllDatasets = []
    k = 0
    H = 0

    with open(file_path, 'r') as file:
        i = 0
        j = -1
        temp_F = []
        tempDataset = None
        specialLine = ''
        for line in file:
            # process each line
            if (i == 2):
                specialLine = line

            if (i in [0, 1]):
                print("ignore line")
            else:
                if (line == specialLine):
                    # make a dataset
                    print(line)
                    if tempDataset != None:
                        print("Update")
                        AllDatasets.append(tempDataset)
                        temp_F = []
                    tempDataset = DataSet()
                    j = 0
                if j == 1:
                    tempDataset.x = get_array(line)
                elif (j == 2):
                    tempDataset.Qs = get_array(line)
                elif (j == 3):
                    tempDataset.flavours = get_array(line)
                elif j > 3:
                    temp_F += get_array(line)
                    if ((j - 3) % (len(tempDataset.Qs)) == 0):
                        tempDataset.F.append(temp_F)
                        temp_F = []

            j += 1
            i += 1
    return AllDatasets


def main():
    file_path = 'MSHT20lo_as130_0000.dat'
    DataSets = []
    DataSets = process_file(file_path)

    for dataset in DataSets:
        #print(f'x is : {dataset.x}')
        #print(f'Q is : {dataset.Qs}')
        #print(f'flavour is : {dataset.flavours}')
        #print(f'F is: {dataset.F}')
        print(len(dataset.x))
        print(len(dataset.F)) # 127
        print(len(dataset.F[0])) # 44
        print(len(dataset.F[1]))
        print(len(dataset.F[2]))
        print(" ")

        print(dataset.F[0][:20])
        print(dataset.F[1][:20])
        print(dataset.F[2][:20])


main()
