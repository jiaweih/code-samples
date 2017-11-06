''' Use Google's Geocode API to find coordinates of S&P 1500's addresses,
    and compute each firms' distances with the White House (38.8976763, 77.0387185).
'''
import time
import numpy as np
import pandas as pd
import googlemaps
from vincenty import vincenty

# Coordiates of the White House.
WHITE_HOUSE = (38.8976763, 77.0387185)
# Create googlemaps instance with a valid API key.
API_KEY = input("Enter your Geocoding API: ")
GMAPS = googlemaps.Client(key=API_KEY)


def get_lat_lon(df):
    ''' Get the latitude and longitude of each firm
        and its distance to the White House.
    '''
    time.sleep(1)
    address = df['address']
    # Create googlemaps instance with a valid API key.
    # Some addresses are not valid; for example, one address is (Universal
    # Corporate Center, 367 South Gulph Road, PO Box 615, King Of Prussia,
    # PA, 19406). gmaps.geocode would return an empty string for this address.
    # I looked it up online, the right address should have "PO Box 61558"
    # instead of "PO Box 615".
    geocode_result = GMAPS.geocode(address)
    # If the address is invalid, geocode_result would be empty;
    # NAs will be returned.
    if len(geocode_result) == 0:
        print("Address not valid: {}".format(address))
        df['lat'], df['lng'] = np.nan, np.nan
        df['distance(km)'] = np.nan
    else:
        # Extract the location of the firm and its latitude, longitude.
        locations = geocode_result[0]['geometry']['location']
        lat, lng = locations['lat'], locations['lng']
        df['lat'], df['lng'] = lat, lng
        # (latitude, longitude) of the firm.
        firm = (lat, lng)
        # Calculate the distance(km) from each firm to the White House.
        df['distance(km)'] = vincenty(firm, WHITE_HOUSE)
    return df


def main():
    # Read the Excel file and drop rows with NAs. (Two rows have empty addresses.)
    df = pd.read_excel("coname_addresses.xlsx").dropna()
    print("Start calculating")
    # Apply the function get_lat_lon to each row of the dataframe to get the
    # latitude and longitude of the firms, and their distances to the White House.
    df_output = df.apply(get_lat_lon, axis=1)
    # Write the outputs to an Excel file. NAs in the data indicate that
    # the corresponding addresses are not accurate.
    print("Start writing")
    writer = pd.ExcelWriter("output.xlsx")
    df_output.to_excel(writer, sheet_name="lat_lng_distance")
    writer.save()
    print("Exit")


if __name__ == '__main__':
    main()
