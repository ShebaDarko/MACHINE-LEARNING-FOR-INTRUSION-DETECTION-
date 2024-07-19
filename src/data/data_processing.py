from import_libraries import *

# Load your dataset 
# Separate features (x) and labels (y)
x_columns = df.columns.drop(['label'])
x = df[x_columns]

# Perform one-hot encoding for categorical features
categorical_columns = x.select_dtypes(include='object').columns
encoder = OneHotEncoder()
encoded_columns = encoder.fit_transform(x[categorical_columns]).toarray()

# Create column names for the one-hot encoded features
encoded_column_names = [f"{col}_{val}" for col, vals in zip(categorical_columns, encoder.categories_) for val in vals]
encoded_df = pd.DataFrame(encoded_columns, columns=encoded_column_names)
x = pd.concat([x, encoded_df], axis=1)
x = x.drop(categorical_columns, axis=1)

# Convert the target labels to integers using LabelEncoder
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])
y = df['label']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

