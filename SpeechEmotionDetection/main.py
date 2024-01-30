import re
import librosa.display
import numpy as np
from keras.src.layers import Activation, Flatten
from keras.src.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D
from keras.models import model_from_json
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# reading all wav files from dataset
def find_wav_files(root_folder):
    wav_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.wav'):
                full_path = os.path.join(root, file)
                wav_files.append(full_path)
    return wav_files


# parsing filenames and setting weights of that audio file
def parse_file_name(file_name):
    file_name = file_name.replace("wav", "")
    parts = file_name.split('-')
    emotion_code = parts[2]
    actor_id_str = parts[6]
    actor_id_str = re.sub(r'\D', '', actor_id_str)
    actor_id = int(actor_id_str)
    gender = 'male' if actor_id % 2 == 1 else 'female'
    emotions = {
        '01': 'neutral', '02': 'calm', '03': 'happy',
        '04': 'sad', '05': 'angry', '06': 'fearful',
        '07': 'disgust', '08': 'surprised'
    }
    emotion = emotions.get(emotion_code, 'unknown')
    return f"{gender}_{emotion}"


# creating model that is CNN
def create_model():
    model = Sequential()
    model.add(Conv1D(256, 5, padding='same', input_shape=(216, 1)))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5, padding='same', ))
    model.add(Activation('relu'))
    model.add(Conv1D(128, 5, padding='same', ))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(12))
    model.add(Activation('softmax'))
    return model


# collecting all wav files in one list
def process_audio_files(wav_files):
    feeling_list = []
    for item in wav_files:
        label = parse_file_name(item)
        feeling_list.append(label)
    return feeling_list


# reading filenames and giving weights based on that
def extract_features_from_files(file_list):
    df = pd.DataFrame(columns=['feature'])
    bookmark = 0
    for index, y in enumerate(file_list):
        if file_list[index][6:-16] != '01' and file_list[index][6:-16] != '07' and file_list[index][
                                                                                   6:-16] != '08' and \
                file_list[index][:2] != 'su' and file_list[index][:1] != 'n' and file_list[index][
                                                                                 :1] != 'd':
            X, sample_rate = librosa.load(y, res_type='kaiser_fast', duration=2.5, sr=22050 * 2, offset=0.5)
            sample_rate = np.array(sample_rate)
            mfccs = np.mean(librosa.feature.mfcc(y=X,
                                                 sr=sample_rate,
                                                 n_mfcc=13),
                            axis=0)
            feature = mfccs
            df.loc[bookmark] = [feature]
            bookmark = bookmark + 1
    return df


# preparing data for model, based on df, creating X_train Y_train and X_test and Y_test
def prepare_data_for_model(features_df, labels_df, lb, test_size=0.2):
    combined_df = pd.concat([features_df, labels_df], axis=1)
    combined_df = shuffle(combined_df).fillna(0)
    X = combined_df.iloc[:, :-1]
    y = combined_df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    y_train = np_utils.to_categorical(lb.fit_transform(y_train.ravel()))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test.ravel()))
    x_traincnn = np.expand_dims(X_train, axis=2)
    x_testcnn = np.expand_dims(X_test, axis=2)
    return x_traincnn, x_testcnn, y_train, y_test


# training model and plotting
def train_and_visualize_model(model, x_train, y_train, x_test, y_test, batch_size=16, epochs=25):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='lower left')
    plt.show()

    return history


# saving trained model for easier predictions later
def save_model(model):
    model_name = 'Emotion_Voice_Detection_Model'
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)


# loadint pre-trained model that is saved
def load_model(path, x_test, y_test):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path)
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = loaded_model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
    return loaded_model


# prediction emotions from file
def predict_emotion_from_file(file_path, model, label_encoder):
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=2.5, sr=22050 * 2, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    feature_live = mfccs.reshape(1, -1)
    feature_live = np.expand_dims(feature_live, axis=2)
    live_preds = model.predict(feature_live, batch_size=32, verbose=1)
    live_preds_class_index = live_preds.argmax(axis=1)
    live_preds_label = label_encoder.inverse_transform(live_preds_class_index)

    return live_preds_label


# path to dataset folder
root_folder = 'RawData'
all_wav_files = find_wav_files(root_folder)
labels = pd.DataFrame(process_audio_files(all_wav_files))

df = extract_features_from_files(all_wav_files)
features_df = pd.DataFrame(df['feature'].values.tolist())
labels_df = pd.DataFrame(labels)
lb = LabelEncoder()
x_train_cnn, x_test_cnn, y_train, y_test = prepare_data_for_model(features_df, labels_df, lb)

model = create_model()
print(model.summary())
train_and_visualize_model(model, x_train_cnn, y_train, x_test_cnn, y_test)

save_model(model)
loaded_model = load_model("saved_models/Emotion_Voice_Detection_Model.h5", x_test_cnn, y_test)
prediction = predict_emotion_from_file('output10.wav', loaded_model, lb)
print(prediction)
