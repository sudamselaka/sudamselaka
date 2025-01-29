# Introducing the packages

# Array package
import numpy as np
import pandas as pd
import xarray as xr
# For mathematical functions
import math
# Shapefile loading package
import geopandas as gpd
# Projection plot package
import cartopy as crs
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
# Ploting package
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.dates as mdates

# Defining the plotting function
def plot(shapefile, lon, lat, data, **kwargs):
    
    # Change the font type as Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Loading keywords
    title = kwargs.get('title',None)
    vmax  = kwargs.get('vmax',None)
    vmin  = kwargs.get('vmin',None)
    cmap  = kwargs.get('cmap','Reds')
    save_name = kwargs.get('save_name', None)

    # This is to adjust the interval of value.
    num_levels = 20
    levels = np.linspace(vmin, vmax, num_levels+1)
    norm = BoundaryNorm(levels, ncolors=256)

    # Ploting
    fig = plt.figure(figsize=(6,6),dpi=300)
    # According to the requirement of assignment, the projection (EPSG:3031) is set.
    ax = fig.add_subplot(111, projection=ccrs.SouthPolarStereo()) 
    # Set mapping print out range
    ax.set_extent([-180, 180, -90, -65], ccrs.PlateCarree())

    # Make ocean color
    ax.add_feature(crs.feature.OCEAN, facecolor='#7cb0d3') 
    ax.gridlines(ls='--',color='grey',lw=0.5)
    
    # Applying the shapefile to map
    shape_feature = ShapelyFeature(shapefile.geometry, ccrs.SouthPolarStereo(), edgecolor='k', facecolor='white',lw=0.5)
    ax.add_feature(shape_feature)
    
    cs = ax.pcolormesh(lon, lat, data, norm=norm, cmap=cmap,transform=crs.crs.PlateCarree())
    cb = fig.colorbar(cs, ticks=np.linspace(vmin,vmax,11),orientation='horizontal', shrink=0.7, pad=0.05)   
    cb.ax.set_xlabel('Melt days',fontsize=10)
    ax.set_title(title,weight='bold',fontsize=18,pad=10)

    # Plot legend for coastline
    # plt.plot([-59.9,-60],[-59.9,-60],c='k',label='Coastline',transform=crs.crs.PlateCarree())
    # plt.legend(fontsize=10)
    continent_patch = Patch(facecolor='white', edgecolor='black', label='Antarctic Ice Shelfs')
    sea_patch = Patch(facecolor='#7cb0d3', edgecolor='black', label='Ocean')
    plt.legend(handles=[continent_patch, sea_patch], fontsize=10, loc='lower left', frameon=True, framealpha=1, edgecolor='black', fancybox=False)
    # Add the scale bar
    scalebar = ScaleBar(1, location='lower right', length_fraction=0.1)  # 1 unit = 1 degree
    ax.add_artist(scalebar)

    if save_name:
        fig.savefig(save_name, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()
    plt.close()

# Task 1: Load the shapefile
shapefile = 'env_pro_data/IceShelf_Antarctica_v02.shp'
gdf = gpd.read_file(shapefile)

# Task 2: Load the NetCDF file
data = xr.open_dataset('env_pro_data/CumJour-Antarctic-ssmi-1979-2023-H19.nc')

# Task 3: Extract data and preprocess with user-defined time period
# Define the preferred time period (e.g., from 1979 to 2023)
start_year = int(input("Enter the start year: "))
end_year = int(input("Enter the end year: "))

# Filter the dataset for the preferred time period
filtered_data = data.sel(time=slice(f"{start_year}-01-01", f"{(end_year)}-12-31"))

# Extract filtered data
lon = filtered_data['lon'].values #if 'lon' in filtered_data else filtered_data['x'].values
lat = filtered_data['lat'].values #if 'lat' in filtered_data else filtered_data['y'].values
melt = filtered_data['melt']
time = filtered_data['time']

# Group data by year within the preferred time period
melt_year = melt.groupby('time.year').sum(dim='time')

# Task 4: Annual average and maximum cumulative melt days
avg_melt_days = melt_year.mean(dim='year', skipna=True) # to ignore Nan values, if False; it would return NaN if any value along the dimension is NaN
avg_melt_days = avg_melt_days.where(avg_melt_days != 0, np.nan)  # Replace zeros with NaN

# Plot the annual average
plot(gdf, lon, lat, avg_melt_days,title=f'Annual Average of Cumulative Melt Days ({start_year}-{end_year})',vmax=math.ceil(avg_melt_days.max() / 10) * 10,vmin=math.floor(avg_melt_days.min() / 10) * 10,save_name=f'Annual_Average_of_Cumulative_Melt_Days_{start_year}_{end_year}.png')

# Task 5: Maximum melt days and maximum melt extent
max_melt_days = melt_year.max(dim=['y', 'x'], skipna=True).to_dataframe(name='Max_melt_days')
max_melt_days.index.name = None # Remove index name
max_melt_days['Year'] = max_melt_days.index # extract the index as Year
max_melt_days = max_melt_days.reset_index(drop=True)

# Identify the maximum melt days value and year
max_melt_days_value = max_melt_days['Max_melt_days'].max()
max_melt_days_years = max_melt_days[max_melt_days['Max_melt_days'] == max_melt_days_value]['Year'].tolist()
print(f'Max Melt days is {int(max_melt_days_value)} in the following year/years: {max_melt_days_years}')

# Calculate melt extent (number of pixels with non-NaN values) for each year in melt_year
melt_year_filtered = melt_year.where(melt_year != 0)
melt_extent = melt_year_filtered.notnull().sum(dim=['y', 'x']) # to create a boolean mask (.notnull())

# Find the year(s) with the maximum melt extent
max_melt_extent_value = melt_extent.max()
max_melt_extent_years = melt_extent.sel(year=melt_extent == max_melt_extent_value)

print(f'Max Melt Extent is {int(max_melt_extent_value)} in the following year/years: {", ".join(map(str, max_melt_extent_years.coords["year"].values))}')

# Task 6: Figures of task 5
# Plot the year with the maximum melt days
for year in max_melt_days_years:
    tmp = melt_year.sel(year=year)
    tmp = tmp.where(tmp != 0, np.nan)
    
    plot(gdf, lon, lat, tmp,title=f'Max Melt days is {int(max_melt_days_value)} in {year}',vmax=math.ceil(tmp.max() / 10) * 10,vmin=math.floor(tmp.min() / 10) * 10,save_name=f'Max Melt days is {int(max_melt_days_value)} in {year}.png')
    
# Plot the difference of annual average melt days and the year with the maximum melt days
for year in max_melt_days_years:  # Iterate through the years in max_melt_days_years
    tmp = melt_year.sel(year=year)
    tmp = tmp.where(tmp != 0, np.nan)
    
    difference = avg_melt_days - tmp
    plot(gdf, lon, lat, difference,title=f'Difference to Average Annual Melt Days in Year {year}',vmax=math.ceil(difference.max() / 10) * 10,vmin=math.floor(difference.min() / 10) * 10,cmap='bwr',save_name=f'Difference to Average Annual Melt Days in Year {year}.png')

year_value = max_melt_extent_years['year'].values.item()
tmp = melt_year.sel(year=year_value)
tmp = tmp.where(tmp != 0, np.nan)
plot(gdf,lon,lat, tmp,title=f'Largest melt extent is {int(max_melt_extent_value)} in {year_value}',vmax=math.ceil(tmp.max() / 10) * 10,vmin=math.floor(tmp.min() / 10) * 10,save_name=f'Largest melt extent is {int(max_melt_extent_value)} in {year_value}.png')


# Task 7: Convert your xarray to a Pandas DataFrame
melt_data_flattened = melt.values.reshape(melt.shape[0], -1)
pd_melt_extent = np.nansum(melt_data_flattened > 0, axis=1)
days_of_year = time.dt.dayofyear

# Creating Pandas dataframe
df_melt_extent = pd.DataFrame({"datetime": time['time'],"day_of_year": days_of_year,"melt_extent": pd_melt_extent})
print(df_melt_extent)

# Task 8: Plot melt extent timeseries
# Adding the 'year_group' column to df_melt_extent
df_melt_extent['year_group'] = df_melt_extent['datetime'].apply(lambda x: x.year if x.month >= 6 else x.year - 1)
df_melt_extent = df_melt_extent.loc[(df_melt_extent['year_group'] >= start_year) & (df_melt_extent['year_group'] <= end_year-1)] # added (end_year-1) to remove the incomplete melt data for the last year (considering the new year period)

# Add a 'custom_date' column to adjust dates for plotting
def adjust_date(date):
    if date.month >= 6:
        return pd.Timestamp(year=1999, month=date.month, day=date.day)
    else:
        return pd.Timestamp(year=2000, month=date.month, day=date.day) # here the years should include FEB 29th, otherwise it cannot handle leap year dates

df_melt_extent = df_melt_extent.copy()  # Avoid SettingWithCopyWarning
df_melt_extent['custom_date'] = df_melt_extent['datetime'].apply(adjust_date)

# Calculate the average Extent values by 'custom_date'
average_extent = df_melt_extent.groupby('custom_date')['melt_extent'].mean()

# Plot grey lines for each year
plt.figure(figsize=(12, 6), dpi=300)

for year, group in df_melt_extent.groupby('year_group'):
    color = 'red' if year == year_value else 'grey'
    alpha = 1.0 if year == year_value else 0.5
    label = f"Max Melt Extent ({year})" if color == 'red' else None  # Label the red line
    plt.plot(group['custom_date'], group['melt_extent'], color=color, alpha=alpha, label=label)

# Plot the average extent as a black line
plt.plot(average_extent.index, average_extent.values, color='black', linewidth=2, label=f'Average Melt Extent for {start_year}-{end_year} period')

# Customize the plot
plt.title(f'Melt Extent Data for {start_year}-{end_year} period')
plt.ylabel('Melt Extent / (Pixels)')
plt.grid(True, linestyle='--', alpha=0.6)

# Format the x-axis to show months
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.legend()

# Save and show the plot
plt.savefig(f'Melt Extent for {start_year}-{end_year} period', dpi=300)
plt.show()

# Task 9: 15 largest ice shelves and average melt extent
gdf = gdf.to_crs(epsg=3031)
gdf["area_calculated"] = gdf.geometry.area
sorted_gdf = gdf.sort_values("area_calculated", ascending=False)
top_15_gdf = sorted_gdf.head(15) # change the number as your interested no of top ice shelves

melt = melt.rio.write_crs("EPSG:3031", inplace=True)

# Create an empty list to store results
results = []

# Loop through all areas in 'top_15_gdf'
for area_name in top_15_gdf['NAME']:
    # Select the region for the current area
    region = top_15_gdf[top_15_gdf["NAME"] == area_name]
    
    # Clip the data for the current region
    clipped = melt.rio.clip(region.geometry, region.crs)
    
    # Mask the clipped data where the values are equal to 1
    melt_masked = clipped.where(clipped == 1)
    
    # Resample the data by year and calculate the mean
    yearly_average_melt = melt_masked.resample(time="1YE").mean()
    yearly_average_melt['time'] = yearly_average_melt['time'].dt.year
    yearly_average_melt = yearly_average_melt.rename({"time": "year"})
    
    # Sum the values over the spatial dimensions x and y
    yearly_sum = yearly_average_melt.sum(dim=["x", "y"])
    
    # Convert to pandas DataFrame and calculate the mean of the yearly sums
    df_yearly_sum = yearly_sum.to_pandas()
    average_yearly_sum = df_yearly_sum.mean()
    
    # Append the results (area name and average yearly sum) to the list
    results.append({'NAME': area_name, 'average_yearly_melt_extent': average_yearly_sum})

# Convert the list of results into a pandas DataFrame
df_top_15_melt = pd.DataFrame(results)

# Print and save the DataFrame
print(df_top_15_melt)
df_top_15_melt.to_csv(f'df_top_15_melt_{start_year}to{end_year}.csv', index=False)

# Plotting a pie chart for the results
plt.figure(figsize=(8, 8),dpi=300)

# Create the pie chart
plt.pie(df_top_15_melt['average_yearly_melt_extent'], labels=df_top_15_melt['NAME'], autopct='%1.1f%%', startangle=90)
plt.title(f'Average Melting Extent for the Largest 15 Ice Shelves from {start_year} to {end_year}', pad=20)
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
plt.savefig(f'Average Melting Extent for the Largest 15 Ice Shelves from {start_year} to {end_year}.png', bbox_inches='tight')  # Save as PNG file
plt.show()
print('/// End of the program ///')
