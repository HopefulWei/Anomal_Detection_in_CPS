from bitarray import bitarray
import numpy
import mmh3
import pandas as pd


global i,j,k,l
i = 0
j = 0
k = 0
l = 0
m = 0
Y = []
my_matrix1 = numpy.loadtxt(open("D:\python file\paper\Bloom\TEXT7-3-1.csv","r"),delimiter=",",skiprows=0)
my_matrix2 = numpy.loadtxt(open("D:\python file\paper\Bloom\TEXT7-3-2.csv","r"),delimiter=",",skiprows=0)
class BloomFilter(set):

        def __init__(self, size, hash_count):
            super(BloomFilter, self).__init__()
            self.bit_array = bitarray(size)
            self.bit_array.setall(0)
            self.size = size
            self.hash_count = hash_count

        def __len__(self):
            return self.size

        def __iter__(self):
            return iter(self.bit_array)

        def add(self, item):
            for ii in range(self.hash_count):
                index = mmh3.hash(item, ii) % self.size
                self.bit_array[index] = 1

            return self

        def __contains__(self, item):
            out = True
            for ii in range(self.hash_count):
                index = mmh3.hash(item, ii) % self.size
                if self.bit_array[index] == 0:
                    out = False

            return out


def main():
        bloom = BloomFilter(1000000, 10)
        animals = my_matrix1
        # First insertion of animals into the bloom filter
        for animal in animals:
            bloom.add(animal)

        # Membership existence for already inserted animals
        # There should not be any false negatives
        for animal in animals:
            if animal in bloom:
                global i
                i = i + 1
                #print('{} is in bloom filter as expected'.format(animal))
            else:
                global j
                j = j + 1
                # print('Something is terribly went wrong for {}'.format(animal))
                #print('FALSE NEGATIVE!')

        # Membership existence for not inserted animals
        # There could be false positives
        other_animals = my_matrix2
        for other_animal in other_animals:
            if other_animal in bloom:
                global k
                k = k + 1
                Y.append(0)
                #print('{} is not in the bloom, but a false positive'.format(other_animal))
            else:
                global l
                l = l + 1
                Y.append(1)

                #print('{} is not in the bloom filter as expected'.format(other_animal))


if __name__ == '__main__':

    main()
    print(i,j,k,l)
    file = open('data.txt', 'w')
    file.write(str(Y));
    dataframe = pd.DataFrame({'result': Y})
    dataframe.to_csv("data2.csv", index=False, sep=',')
    file.close()
    file = open('data1.txt', 'w')
    file.write(str(l));
    file.close()