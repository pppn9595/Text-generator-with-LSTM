import numpy as np, os, re, smtplib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from email.mime.text import MIMEText
from email.header import Header
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, Callback
from keras.utils import np_utils

def format_text(text):
    text = text.lower()
    format_items = [
        {'from': '\n+', 'to': ' '},
        {'from': '\r+', 'to': ' '},
        {'from': '\t+', 'to': ' '},
        {'from': ' +', 'to': ' '},
    ]
    for format_item in format_items:
        text = re.sub(format_item['from'], format_item['to'], text)
    return text

def send_mail(fromaddr, username, password, smtp, to, subject, message):
    mail = MIMEText(message.encode('utf-8'), 'plain', 'utf-8')
    mail['Subject'] = Header(subject, 'utf-8')
    server = smtplib.SMTP(smtp)
    server.ehlo()
    server.starttls()
    server.login(username, password)
    server.sendmail(fromaddr, to, mail.as_string())
    server.quit()

class report_callback(Callback):
    def __init__(self, fromaddr, username, password, smtp, to, subject):
        self.fromaddr = fromaddr
        self.username = username
        self.password = password
        self.smtp = smtp
        self.to = to
        self.subject = subject
    def on_epoch_end(self, epoch, logs={}):
        text = TG.generate(test_texts[0])
        send_mail(self.fromaddr, self.username, self.password, self.smtp, self.to, self.subject, text)

class TextGenerator:
    def __init__(self, chars, seq_length, model_file = None):
        self.chars = sorted(chars)
        self.vocab_number = len(self.chars)
        self.char_to_int = dict( (char, i) for i, char in enumerate(self.chars) )
        self.int_to_char = dict( (i, char) for i, char in enumerate(self.chars) )
        self.seq_length = seq_length
        self.build_model()
        if model_file != None:
            self.load_model(model_file)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    def build_model(self):
        self.model = Sequential([
            LSTM(256, input_shape=(self.seq_length, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(256),
            Dropout(0.2),
            Dense( len(self.chars) , activation='softmax'),
        ])

    def load_model(self, model_file):
        self.model.load_weights(model_file)

    def train(self, text, epochs=50, batch_size=256, callbacks=[]):
        dataX = []
        dataY = []
        for i in range(0, len(text) - self.seq_length):
            seq_in = text[i:i + self.seq_length]
            seq_out = text[i + self.seq_length]
            dataX.append([self.char_to_int[char] for char in seq_in])
            dataY.append(self.char_to_int[seq_out])

        X = np.reshape(dataX, (len(dataX), self.seq_length, 1))
        X = X / float(self.vocab_number)
        y = np_utils.to_categorical(dataY)

        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    def generate(self, sentence, length=100, only_generated=False):
        start_seq = [self.char_to_int[char] for char in sentence]
        if only_generated:
            sentence = ''
        for i in range(length):
            x = np.reshape(start_seq, (1, self.seq_length, 1))
            x = x / float(self.vocab_number)
            prediction = self.model.predict(x, verbose=0)
            index = np.argmax(prediction)
            result = self.int_to_char[index]
            seq_in = [self.int_to_char[value] for value in start_seq]
            start_seq.append(index)
            start_seq = start_seq[1:len(start_seq)]
            sentence += result
        return sentence

file = 'datas/robinson_crusoe.txt'
train = False
epochs = 500
batch_size = 256
seq_length = 100
model_file = 'weights-improvement-35-1.5867.hdf5'
checkpoint_file = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
test_texts = [
    'egy szép napon arra lettem figyelmes, hogy két igen szép sirály szállt le a sziget tetejére. nem tud',
    'hol volt, hol nem volt, volt egyszer egy hajótörött, aki egy lakatlan sziget fogságába került, amibő',
    'és béviszlek titeket a földre, a mely felől esküre emeltem fel kezemet, hogy ábrahámnak, izsáknak és',
    'a neurális hálózat biológiai neuronok összekapcsolt csoportja. modern használatban a szó alatt a mes',
    'adott egy parciális differenciálegyenlet és peremértékfeladat. például az elektrosztatika peremérték',
]


text = open(file).read()
text = format_text(text)
chars = list(set(text))

callbacks = []
callbacks.append( ModelCheckpoint(checkpoint_file, monitor='loss', verbose=0, save_best_only=True, mode='min') )
#callbacks.append( report_callback('', '', '', '', '', 'Report') )

TG = TextGenerator(chars, seq_length, model_file)

if train:
    TG.train(text, epochs, batch_size, callbacks)
else:
    for test_text in test_texts:
        generated_text = TG.generate(test_text, 100, True)
        print('\n%s|%s\n' % (test_text, generated_text))
