import numpy as np
import h5py


class DataSet(object):
    def __init__(self):
        self.x = []
        self.Qs = []
        self.flavours = []
        self.min = []
        self.max = []

    def split_data_min(self, q_in: float, flv_in: float):
        # given Q value and flavour value give all x
        # flavour is defined by:
        flv_index = self.flavours.index(flv_in)
        q_index = self.Qs.index(q_in)

        x_values = []
        # loop over F
        for i in range(q_index, len(self.min), len(self.Qs)):
            x_values.append(self.min[i][flv_index])
        return x_values

    def split_data_max(self, q_in: float, flv_in: float):
        # given Q value and flavour value give all x
        # flavour is defined by:
        flv_index = self.flavours.index(flv_in)
        q_index = self.Qs.index(q_in)

        x_values = []
        # loop over F
        for i in range(q_index, len(self.max), len(self.Qs)):
            x_values.append(self.max[i][flv_index])
        return x_values


def get_array(line):
    temp = line.strip()
    temp = temp.split(' ')
    temp_out = []
    for i in range(0, len(temp)):
        temp_out.append(float(temp[i]))
    return temp_out


def process_file(file_path):
    dataset = []
    with open(file_path[0], 'r') as file:
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
                        dataset.append(temp_dataset)
                    temp_dataset = DataSet()
                if j == 1:
                    temp_dataset.x = get_array(line)
                elif j == 2:
                    temp_dataset.Qs = get_array(line)
                elif j == 3:
                    temp_dataset.flavours = get_array(line)
                elif j > 3:
                    temp_dataset.min.append(get_array(line))
                    temp_dataset.max.append(get_array(line))

            j += 1
            i += 1

    for err_file in file_path:
        with open(err_file, 'r') as file:
            i = 0
            j = -1
            n = -1
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
                        n += 1
                    elif j > 3:
                        a = np.minimum(get_array(line), dataset[n].min[j - 4])
                        dataset[n].min[j - 4] = a
                        b = np.maximum(get_array(line), dataset[n].max[j - 4])
                        dataset[n].max[j - 4] = b

                j += 1
                i += 1
    return dataset


def store_datasets_in_h5py(file_path, datasets):
    with (h5py.File(file_path, 'w') as hdf_file):
        for flavour in datasets[0].flavours:
            group1 = hdf_file.create_group(str(flavour))
            for dataset in datasets:
                for Q in dataset.Qs:
                    try:
                        group2 = group1.create_group(str(Q))
                        group2.create_dataset('min', data=dataset.split_data_min(Q, flavour))
                        group2.create_dataset('max', data=dataset.split_data_max(Q, flavour))
                    except ValueError:
                        pass


def main():
    file_path = []
    for x in np.linspace(1, 60, 60):
        file_path.append('MSHT20lo_as130/MSHT20lo_as130_00{:02d}.dat'.format(int(x)))
    datasets = process_file(file_path)
    store_datasets_in_h5py('c:/Users/LD_Ci/Documents/Pythonshit/PDFMPhyspython/hp5y_error.h5', datasets)


main()
