import numpy as np
import pandas as pd
import random
#import unittest
from twisted.trial import unittest
import tensorflow as tf
from tensorflow import keras
from assg_tasks import load_dataset
from assg_tasks import TfStackWrapper
#from assg_tasks import make_date_sequence_vectorizor
#from assg_tasks import make_date_sequence_dataset
#from assg_tasks import get_lstm_translator_model
#from assg_tasks import AttentionLayer
#from assg_tasks import get_attention_translator_model


class test_make_date_sequence_vectorizor(unittest.TestCase):

    def setUp(self):
        pass

    def test_example_case(self):
        sequence_length = 15
        texts = ['Months on ye at by esteem 04/20/1960 desire warmth former.'
                'Sure that that way gave any fond now!'
                'His boy middleton, sir nor engrossed affection excellent.'
                'Dissimilar compliment cultivated preference eat sufficient may 4th 2026.']

        vectorizor = make_date_sequence_vectorizor(sequence_length, texts)
        v0 = vectorizor(texts[0])
        expected_vocabulary = ['', '[UNK]', ' ', 'e', 't', 'n', 'i', 'a', 's', 'r', 'o', 'm', 'f', 'd', 'y', 'l', 'h', 'c', '0', 'w', 'u', '2', '.', 'v', 'p', 'g', 'b', '6', '4', '/', 'x', '9', '1', ',', '!']
        expected_v0 = np.array([11, 10,  5,  4, 16,  8,  2, 10,  5,  2, 14,  3,  2,  7,  4])

        self.assertEqual(vectorizor.vocabulary_size(), 35)
        self.assertEqual(vectorizor.get_vocabulary(), expected_vocabulary)
        self.assertEqual(len(v0), 15)
        self.assertTrue(np.allclose(v0.numpy(), expected_v0))


class test_make_date_sequence_dataset(unittest.TestCase):

    def setUp(self):
        # load the sequence text pairs
        num_samples = 1000
        text_pairs = load_dataset(num_samples)
        random.seed(42)
        random.shuffle(text_pairs)
        num_train_samples = 700
        self.train_pairs = text_pairs[:num_train_samples]

        # first extract the seperate source and target texts
        train_source_texts = [pair[0] for pair in self.train_pairs]
        train_target_texts = [pair[1] for pair in self.train_pairs]

        # create vectorizors for our sequences for these tests
        source_sequence_length = 25
        target_sequence_length = 15
        self.source_vectorizor = make_date_sequence_vectorizor(source_sequence_length, train_source_texts)
        self.target_vectorizor = make_date_sequence_vectorizor(target_sequence_length + 1, train_target_texts)

    def test_ds(self):
        batch_size = 50
        train_ds = make_date_sequence_dataset(self.train_pairs, batch_size, self.source_vectorizor, self.target_vectorizor)

        # pull out a batch of samples
        for inputs, targets in train_ds:
            break

        self.assertIsInstance(inputs, dict)
        self.assertTrue('source_inputs' in inputs.keys())
        self.assertTrue('target_inputs' in inputs.keys())
        self.assertEqual(inputs['source_inputs'].shape, (50, 25, 37))
        self.assertEqual(inputs['target_inputs'].shape, (50, 15, 15))
        self.assertEqual(targets.shape, (50, 15, 15))


class test_get_lstm_translator_model(unittest.TestCase):

    def setUp(self):
        pass

    def test_model(self):
        source_sequence_length = 30
        source_vocab_size = 37
        target_sequence_length = 12
        target_vocab_size = 15
        latent_dim = 16
        model = get_lstm_translator_model(source_sequence_length, source_vocab_size, 
                                          target_sequence_length, target_vocab_size,
                                          latent_dim)
        
        self.assertEqual(len(model.layers), 5)

        l = model.layers[0]
        self.assertEqual(l.name, 'source_inputs')
        self.assertEqual(l.batch_shape, (None, 30, 37))

        l = model.layers[1]
        self.assertEqual(l.name, 'target_inputs')
        self.assertEqual(l.batch_shape, (None, 12, 15))

        l = model.layers[2]
        self.assertEqual(l.name, 'encoder')
        self.assertIsInstance(l, keras.layers.Bidirectional)
        self.assertEqual(l.output.shape, (None, 32))

        # its important that the decoder has 3 inputs, the target inputs and the
        # encoder state as initial state
        l = model.layers[3]
        self.assertEqual(l.name, 'decoder')
        self.assertIsInstance(l, keras.layers.LSTM)
        self.assertEqual(l.output.shape, (None, 12, 32))
        self.assertEqual(l.output.shape, (None, 12, 32))
        self.assertEqual(l.input[0].shape, (None, 12, 15))
        self.assertEqual(l.input[1].shape, (None, 32))
        self.assertEqual(l.input[2].shape, (None, 32))

        l = model.layers[4]
        self.assertEqual(l.name, 'decoder_output')
        self.assertIsInstance(l, keras.layers.Dense)
        self.assertEqual(l.output.shape, (None, 12, 15))


class test_AttentionLayer(unittest.TestCase):

    def setUp(self):
        pass

    def test_example_case(self):
        random.seed(10)
        np.random.seed(10)
        tf.random.set_seed(10)
        num_samples = 10
        rnn_dim = 64
        source_sequence_length = 30

        encoder_states = np.random.uniform(0, 1, (num_samples, source_sequence_length, rnn_dim))
        decoder_state = np.random.uniform(0, 1, (num_samples, rnn_dim))

        layer = AttentionLayer(source_sequence_length)
        context = layer({'encoder_states': encoder_states,
                        'decoder_state': decoder_state})

        self.assertEqual(context.shape, (10, 1, 64))

        expected_c0 = np.array([
            0.5037994 , 0.43870404, 0.55076516, 0.5320272 , 0.4497775 ,
            0.4148697 , 0.5386045 , 0.5177513 , 0.40090263, 0.42972448,
            0.4572997 , 0.47683048, 0.49402452, 0.536242  , 0.52213925,
            0.46703514, 0.59789056, 0.49478093, 0.46939936, 0.542826  ,
            0.4635692 , 0.44132176, 0.601612  , 0.554757  , 0.49511957,
            0.54176253, 0.58370805, 0.53299934, 0.49154517, 0.51990664,
            0.49009022, 0.4908493 , 0.46393758, 0.3727612 , 0.5021915 ,
            0.5757127 , 0.48293594, 0.6114995 , 0.494828  , 0.51572114,
            0.61215955, 0.5642143 , 0.48494598, 0.43998736, 0.5452579 ,
            0.5203981 , 0.49647036, 0.4862917 , 0.5122664 , 0.4700712 ,
            0.527557  , 0.4904838 , 0.48257697, 0.5416058 , 0.4744855 ,
            0.46109793, 0.5608495 , 0.41660565, 0.48337755, 0.525952  ,
            0.470785  , 0.48944175, 0.49040702, 0.5216422            
        ])
        self.assertTrue(np.allclose(context[0,0].numpy(), expected_c0))

        expected_c4 = np.array([
            0.4122765 , 0.4247996 , 0.5710289 , 0.47901043, 0.51949656,
            0.5382855 , 0.42713532, 0.47985557, 0.5384437 , 0.42234477,
            0.5086542 , 0.57953066, 0.5340299 , 0.5697076 , 0.64763224,
            0.33723468, 0.48433146, 0.4767087 , 0.47345132, 0.34579113,
            0.46548566, 0.43608657, 0.55177134, 0.6076619 , 0.5382657 ,
            0.4935463 , 0.5228719 , 0.43273377, 0.4780051 , 0.5377238 ,
            0.5700406 , 0.5088542 , 0.4861575 , 0.5502521 , 0.41935536,
            0.5548004 , 0.43846667, 0.5375339 , 0.58796966, 0.55285776,
            0.46397898, 0.52084565, 0.45830855, 0.4047341 , 0.5349217 ,
            0.47174186, 0.5206208 , 0.48786578, 0.45320332, 0.5559688 ,
            0.50591165, 0.5455824 , 0.47035566, 0.56832373, 0.42370018,
            0.4313404 , 0.5853108 , 0.51793563, 0.49121392, 0.5155988 ,
            0.4699644 , 0.48710757, 0.4667536 , 0.54809517
        ])
        self.assertTrue(np.allclose(context[4,0].numpy(), expected_c4))


    def test_bigger_case(self):
        random.seed(12)
        np.random.seed(12)
        tf.random.set_seed(12)
        num_samples = 15
        rnn_dim = 100
        source_sequence_length = 55

        encoder_states = np.random.uniform(0, 1, (num_samples, source_sequence_length, rnn_dim))
        decoder_state = np.random.uniform(0, 1, (num_samples, rnn_dim))

        layer = AttentionLayer(source_sequence_length)
        context = layer({'encoder_states': encoder_states,
                        'decoder_state': decoder_state})

        self.assertEqual(context.shape, (15, 1, 100))

        expected_c5 = np.array([
            0.4824803 , 0.52045363, 0.49605545, 0.42335936, 0.40190142,
            0.47469595, 0.4534002 , 0.57123286, 0.5391433 , 0.4959829 ,
            0.463723  , 0.5239859 , 0.58327055, 0.53622526, 0.45712134,
            0.50546944, 0.48162782, 0.45022878, 0.46286964, 0.543607  ,
            0.48943305, 0.49486652, 0.47215316, 0.490042  , 0.49467123,
            0.51099247, 0.5093629 , 0.55477333, 0.49455673, 0.5347117 ,
            0.4491515 , 0.49931422, 0.57094604, 0.4904277 , 0.51301867,
            0.5314199 , 0.5484701 , 0.45745933, 0.48902327, 0.49053833,
            0.5233914 , 0.5549382 , 0.535002  , 0.43000916, 0.57895774,
            0.5989527 , 0.5036072 , 0.5279543 , 0.5860045 , 0.5006825 ,
            0.49213973, 0.5038005 , 0.5017473 , 0.4669185 , 0.46040633,
            0.4637619 , 0.4731999 , 0.55878794, 0.50088674, 0.4742914 ,
            0.50182647, 0.49206668, 0.48937818, 0.5082405 , 0.49593794,
            0.49640253, 0.41153562, 0.49388164, 0.4841841 , 0.5389934 ,
            0.48189947, 0.50244206, 0.52511275, 0.47781777, 0.5207989 ,
            0.49521896, 0.49587125, 0.4919609 , 0.4758168 , 0.52417505,
            0.52802676, 0.5009975 , 0.48447508, 0.53057015, 0.54087824,
            0.5289161 , 0.46522823, 0.48101223, 0.46777385, 0.49900773,
            0.4673733 , 0.4934601 , 0.5037107 , 0.52829677, 0.5496456 ,
            0.5091248 , 0.49277282, 0.4978058 , 0.538621  , 0.52156764
        ])
        self.assertTrue(np.allclose(context[5,0].numpy(), expected_c5))

        expected_c10 = np.array([
            0.46214673, 0.38601068, 0.49689522, 0.47591272, 0.5033806 ,
            0.51456857, 0.557174  , 0.52478105, 0.47496122, 0.46930167,
            0.53722   , 0.5352524 , 0.53837925, 0.53671   , 0.5225972 ,
            0.4723938 , 0.4829255 , 0.52433115, 0.46499574, 0.46930355,
            0.48171252, 0.55876607, 0.57798773, 0.47121027, 0.52059627,
            0.5287397 , 0.51659787, 0.41074824, 0.5107225 , 0.5220437 ,
            0.5145898 , 0.4460358 , 0.46390903, 0.53028494, 0.50731075,
            0.49124104, 0.59769505, 0.47139168, 0.4579718 , 0.47210345,
            0.5535866 , 0.4560656 , 0.5126957 , 0.51088995, 0.47447607,
            0.56398046, 0.5467942 , 0.49649826, 0.51798826, 0.54547316,
            0.5070276 , 0.584814  , 0.5336334 , 0.51270294, 0.48264822,
            0.57242316, 0.4252058 , 0.432759  , 0.52234805, 0.49464306,
            0.4705031 , 0.5056908 , 0.45389193, 0.50740564, 0.4359356 ,
            0.5428313 , 0.6046993 , 0.49000302, 0.44685644, 0.4648462 ,
            0.5097128 , 0.51453215, 0.54529566, 0.4565653 , 0.49589872,
            0.5007672 , 0.5151037 , 0.43161842, 0.51945335, 0.57456815,
            0.50422597, 0.44501942, 0.5146318 , 0.4848802 , 0.54194176,
            0.5183116 , 0.48091495, 0.6035262 , 0.45947677, 0.58256394,
            0.5118599 , 0.5504369 , 0.48546726, 0.47643316, 0.5149813 ,
            0.50656736, 0.51173955, 0.4600672 , 0.4941591 , 0.5487105
        ])
        self.assertTrue(np.allclose(context[10,0].numpy(), expected_c10))


class test_get_attention_translator_model(unittest.TestCase):

    def setUp(self):
        pass

    def test_model(self):
        source_sequence_length = 30
        source_vocab_size = 37
        target_sequence_length = 12
        target_vocab_size = 15
        latent_dim = 16
        batch_size = 25
        model = get_attention_translator_model(source_sequence_length, source_vocab_size, 
                                          target_sequence_length, target_vocab_size,
                                          latent_dim, batch_size)
        
        self.assertEqual(len(model.layers), 6)

        l = model.layers[0]
        self.assertEqual(l.name, 'source_inputs')
        self.assertEqual(l.batch_shape, (None, 30, 37))

        l = model.layers[1]
        self.assertEqual(l.name, 'encoder')
        self.assertIsInstance(l, keras.layers.Bidirectional)
        self.assertEqual(l.output.shape, (None, 30, 32))

        l = model.layers[2]
        self.assertEqual(l.name, 'attention')
        self.assertIsInstance(l, AttentionLayer)
        self.assertEqual(l.output.shape, (25, 1, 32))

        # its important that the decoder has 3 inputs, the target inputs and the
        # encoder state as initial state
        l = model.layers[3]
        self.assertEqual(l.name, 'decoder')
        self.assertIsInstance(l, keras.layers.LSTM)
        self.assertEqual(l.output[0].shape, (25, 32))
        self.assertEqual(l.output[1].shape, (25, 32))
        self.assertEqual(l.output[2].shape, (25, 32))
        self.assertEqual(l.input.shape, (25, 1, 32))


        l = model.layers[4]
        self.assertEqual(l.name, 'decoder_output')
        self.assertIsInstance(l, keras.layers.Dense)
        self.assertEqual(l.output.shape, (25, 15))

        l = model.layers[5]
        self.assertEqual(l.name, 'outputs')
        self.assertIsInstance(l, TfStackWrapper)
        self.assertEqual(l.output.shape, (25, 12, 15))

