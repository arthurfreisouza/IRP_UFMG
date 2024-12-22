import numpy as np
import pandas as pd
from faker import Faker

def creating_np_array(rows_, fake):
    # Define the data type for the structured array
    dtype = [
        ('name', 'U100'),       # String with max length 100
        ('address', 'U200'),    # String with max length 200
        ('email', 'U100'),      # String with max length 100
        ('birthdate', 'datetime64[D]'),  # Date
        ('job', 'U100')         # String with max length 100
    ]
    
    # Create an empty structured NumPy array
    arr = np.empty(rows_, dtype=dtype)
    
    for i in range(rows_):
        arr[i] = (
            fake.name(),
            fake.address().replace('\n', ', '),  # Replace newline with a comma for better readability
            fake.email(),
            np.datetime64(fake.date_of_birth()),  # Convert to NumPy datetime
            fake.job()
        )
    
    return arr

if __name__ == "__main__":
    rows_ = int(input("Type here how many rows do you want: "))
    fake = Faker()

    # Generate the structured NumPy array
    ret = creating_np_array(rows_=rows_, fake=fake)
    
    # Convert the structured NumPy array to a Pandas DataFrame for better visualization
    my_df = pd.DataFrame(ret)
    
    # Display the DataFrame
    print(my_df)
