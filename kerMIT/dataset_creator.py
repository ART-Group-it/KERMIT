__author__ = 'lorenzo'

import dtk
import pickle
import sentence_encoder
import dataset_reader
from tree import Tree
import numpy

class DatasetCreator:

    def __init__(self, file = "/Users/lorenzo/Documents/Universita/PHD/Lavori/DSTK/RTE/RTE3_dev_processed.xml.svm",
                 dtk_params={"LAMBDA":1.0, "dimension":1024},
                 encoder_params=[1024, 1, "pos"]):

        self.file = file

        self.dtk_params = dtk_params
        self.encoder_params = encoder_params
        dt = dtk.DT(**dtk_params)

        #dir2 = "/Users/lorenzo/Documents/Universita/PHD/Lavori/DSTK/RTE"
        #file = "RTE3_dev_processed.xml.svm"
        #test = "RTE3_test_processed.xml"

        filename = "dtk " + str(sorted(list(dtk_params.items())))
        filename2 = "encoder " + str(list(encoder_params))

        try:
            D1 = pickle.load(open(filename, "rb"))
            self.D1 = D1
        except:
            print("No file found for dtk, generating one")

            Data = dataset_reader.Dataset(self.file)
            trees = [Tree(string=pairs[0]) for pairs in Data.pairs]

            D1 = []
            for t in trees[1:]:
                D1.append(dt.dt(t))


            D1 = numpy.array(D1)
            pickle.dump(D1, open(filename, "wb"))
            self.D1 = D1
            #raise


        try:
            self.D2 = pickle.load(open(filename2, "rb"))
        except:
            print("No file found for sentence encoder, generating one")

            Data = dataset_reader.Dataset(self.file)
            trees = [Tree(string=pairs[0]) for pairs in Data.pairs]
            D2 = []

            for t in trees[1:]:
                D2.append(sentence_encoder.encoder(t.sentence, *encoder_params))

            D2 = numpy.array(D2)
            pickle.dump(D2, open(filename2, "wb"))
            self.D2 = D2
            #raise

    def get_d(self):
        return [self.D1, self.D2]
