import pandas as pd
import random

# Load the data
df = pd.read_csv('data/cars_data.csv')

# Find coupes
coupes = df[df['Vehicle Style'].str.contains('Coupe', case=False, na=False)]

# Basic info
print(f"Total cars in dataset: {len(df)}")
print(f"Total coupes found: {len(coupes)}")
print()

# Show some examples of popular coupes
print("Popular coupe models:")
popular_coupes = coupes.sort_values(by='Popularity', ascending=False).head(10)
for _, row in popular_coupes.iterrows():
    print(f"Make: {row['Make']}, Model: {row['Model']}, Year: {row['Year']}, Popularity: {row['Popularity']}")
print()

# Check if specific coupe models exist
check_models = ["Nissan 200SX", "Nissan 300ZX", "BMW 4 Series", "Alfa Romeo 4C"]
print("Checking for specific models:")
for model in check_models:
    make, model_name = model.split(" ", 1)
    exists = len(df[(df['Make'] == make) & (df['Model'].str.contains(model_name)) & 
                  (df['Vehicle Style'].str.contains('Coupe', case=False, na=False))]) > 0
    print(f"{model}: {'Found' if exists else 'Not found'}")
print()

# More detailed check for Alfa Romeo 4C
print("Checking all Alfa Romeo cars:")
alfa_cars = df[df['Make'] == 'Alfa Romeo']
for _, row in alfa_cars.iterrows():
    print(f"Model: {row['Model']}, Style: {row['Vehicle Style']}, Year: {row['Year']}")
print()

# Compare with the models mentioned in the response
mentioned_models = ["Mazda 3", "FIAT 124 Spider", "Audi A4"]
print("Checking models mentioned in response:")
for model in mentioned_models:
    make, model_name = model.split(" ", 1)
    is_coupe = len(df[(df['Make'] == make) & (df['Model'].str.contains(model_name)) & 
                     (df['Vehicle Style'].str.contains('Coupe', case=False, na=False))]) > 0
    print(f"{model}: {'Is a coupe' if is_coupe else 'Not a coupe'}")
    
    # Show the actual style of this model
    style_rows = df[(df['Make'] == make) & (df['Model'].str.contains(model_name))]
    if not style_rows.empty:
        print(f"  Actual styles for {model}:")
        for _, style_row in style_rows.iterrows():
            print(f"  - {style_row['Vehicle Style']}") 