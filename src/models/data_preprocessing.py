from common_imports import

def preprocess_data(dataframe):
    # Assuming 'label' is the column to encode
    label_encoder = LabelEncoder()
    dataframe['label'] = label_encoder.fit_transform(dataframe['label'])
    
    x = dataframe.drop(columns=['label'])
    y = dataframe['label']
    
    x = x.values.astype('float32')
    y = y.values.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    
    return x_train, x_test, y_train, y_test, label_encoder

def load_data(filename):
    dataframe = pd.read_csv(filename)  # Update based on your file format
    return dataframe

