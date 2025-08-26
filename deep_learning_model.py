import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

data = pd.DataFrame()

# Data
data = pd.read_csv([data, pd.read_csv("C:\\Users\\***\\data.csv")], ignore_index=True)

drop_columns = [
    "id", "listing_url", "scrape_id", "last_scraped", "source", 'neighbourhood_group_cleansed',
    "name", "description", "neighborhood_overview", "picture_url", 'bathrooms', 
    "host_id", "host_url", "host_name", "host_since", "host_location",
    "host_about", "host_response_time", "host_response_rate",
    "host_acceptance_rate", "host_thumbnail_url", "host_picture_url",
    "host_neighbourhood", "host_listings_count", "host_total_listings_count",
    "host_verifications", "host_has_profile_pic", "host_identity_verified",
    "calendar_updated", "has_availability", "calendar_last_scraped",
    "first_review", "last_review", "license", "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes", "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms", 'neighbourhood', 
]

data.drop(columns=drop_columns, inplace=True)

data['bedrooms'] = data['bedrooms'].fillna(data['bedrooms'].mean())
data['minimum_minimum_nights'] = data['minimum_minimum_nights'].fillna(data['minimum_minimum_nights'].mean())
data['maximum_minimum_nights'] = data['maximum_minimum_nights'].fillna(data['maximum_minimum_nights'].mean())
data['minimum_maximum_nights'] = data['minimum_maximum_nights'].fillna(data['minimum_maximum_nights'].mean())
data['maximum_maximum_nights'] = data['maximum_maximum_nights'].fillna(data['maximum_maximum_nights'].mean())
data['minimum_nights_avg_ntm'] = data['minimum_nights_avg_ntm'].fillna(data['minimum_nights_avg_ntm'].mean())
data['maximum_nights_avg_ntm'] = data['maximum_nights_avg_ntm'].fillna(data['maximum_nights_avg_ntm'].mean())

data['price'] = data['price'].replace('[\$,]', '', regex=True).astype(float)


# 1. Extract the number and create a new column of type `float`
data['bathrooms'] = data['bathrooms_text'].str.extract(r'(\d+\.?\d*)').astype(float)

# 2. Create a new column - bath_type (0 = not specified, 1 = shared, 2 = private)
def get_bath_type(x):
    if pd.isnull(x):
        return 0
    x = x.lower()
    if 'shared' in x:
        return 1
    elif 'private' in x:
        return 2
    else:
        return 0

data['bathroom_type'] = data['bathrooms_text'].apply(get_bath_type)
data.drop(columns=['bathrooms_text'], inplace=True)

amenities_list = ['Bathtub', 'Hair dryer', 'Cleaning products', 'Shampoo', 'Conditioner', 'Body soap', 'Bidet', 'Hot water', 'Shower gel', 'Washer', 'Essentials', 'Hangers',
             'Bed linens', 'Extra pillows and blankets', 'Room-darkening shades', 'Iron', 'Drying rack for clothing', 'Safe', 'Clothing storage', 'TV', 'Sound system', 'Exercise equipment',
             'Pool table', 'Books and reading material', 'Crib', 'Air conditioning', 'Heating', 'Smoke alarm', 'Carbon monoxide alarm', 'First aid kit', 'Wifi', 'Dedicated workspace', 'Kitchen',
             'Refrigerator', 'Microwave', 'Cooking basics', 'Dishes and silverware', 'Freezer', 'Dishwasher', 'Stove', 'Oven', 'Hot water kettle', 'Wine glasses', 'Dining table', 'Coffee',
             'Private entrance', 'Laundromat nearby', 'Patio or balcony', 'Outdoor furniture', 'Free parking on premises', 'Pool', 'Hot tub', 'Elevator', 'Paid parking off premises',
             'Single level home', 'Pets allowed', 'Long term stays allowed', 'Self check-in', 'Building staff', 'Cleaning available during stay', 'Exterior security cameras on property']


for amenity in amenities_list:
    data[amenity] = data['amenities'].apply(lambda x: 1 if amenity in x else 0)

data.drop(columns=['amenities'], inplace=True)
data.dropna(inplace=True)

from sklearn.preprocessing import LabelEncoder

# Label encoding for categorical variables
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoders = {}

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le

# Məlumatların yüklənməsi və numpy formatına çevrilməsi
X = data.drop(columns=["price"]).values
y = data["price"].values.reshape(-1, 1)

# Data partitioning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tabnet_model = TabNetRegressor()
tabnet_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], max_epochs=100, patience=10)
y_pred = tabnet_model.predict(X_test)


def print_metrics(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)

    print("📊 Modelin nəticələri:")
    print(f"✔️ R²: {r2:.4f}")
    print(f"✔️ MSE: {mse:.4f}")
    print(f"✔️ RMSE: {rmse:.4f}")
    print(f"✔️ MAE: {mae:.4f}")


print_metrics(y_test, y_pred)