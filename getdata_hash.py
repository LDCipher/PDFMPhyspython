import h5py


class DataSet(object):
    def __init__(self):
        self.x = []
        self.Qs = []
        self.flavours = []
        self.F = []

    def split_data(self, q_in: float, flv_in: float):
        # given Q value and flavour value give all x
        # flavour is defined by:
        flv_index = self.flavours.index(flv_in)
        q_index = self.Qs.index(q_in)

        x_values = []
        # loop over F
        for i in range(q_index, len(self.F), len(self.Qs)):
            x_values.append(self.F[i][flv_index])
        return x_values


def get_array(line):
    temp = line.strip()
    temp = temp.split(' ')
    temp_out = []
    for i in range(0, len(temp)):
        temp_out.append(float(temp[i]))
    return temp_out


def process_file(file_path):
    all_datasets = []

    with open(file_path, 'r') as file:
        i = 0
        j = -1
        temp_dataset = None
        special_line = ''
        for line in file:
            # process each line
            if i == 2:
                special_line = line

            if i in [0, 1]:
                pass

            else:
                if line == special_line:
                    # make a dataset
                    j = 0
                    if temp_dataset is not None:
                        all_datasets.append(temp_dataset)
                    temp_dataset = DataSet()
                if j == 1:
                    temp_dataset.x = get_array(line)
                elif j == 2:
                    temp_dataset.Qs = get_array(line)
                elif j == 3:
                    temp_dataset.flavours = get_array(line)
                elif j > 3:
                    temp_dataset.F.append(get_array(line))

            j += 1
            i += 1
    return all_datasets


def store_datasets_in_h5py(file_path, datasets):
    with h5py.File(file_path, 'w') as hdf_file:
        hdf_file.create_dataset('x_values', data=datasets[0].x)
        for flavour in datasets[0].flavours:
            group = hdf_file.create_group(str(flavour))
            for dataset in datasets:
                for Q in dataset.Qs:
                    try:
                        group.create_dataset(str(Q), data=dataset.split_data(Q, flavour))
                    except ValueError:
                        pass


def main():
    file_path = 'MSHT20lo_as130/MSHT20lo_as130_0000.dat'
    DataSets = process_file(file_path)
    store_datasets_in_h5py('c:/Users/LD_Ci/Documents/Pythonshit/PDFMPhyspython/hp5y_data.h5', DataSets)
    print(DataSets[2].Qs)


main()
