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

# Plotting package
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.dates as mdates

# Tkinter package
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Defining the plotting function
def plot(shapefile, lon, lat, data, **kwargs):
    plt.rcParams['font.family'] = 'Times New Roman'
    
    title = kwargs.get('title', None)
    vmax = kwargs.get('vmax', None)
    vmin = kwargs.get('vmin', None)
    cmap = kwargs.get('cmap', 'Reds')
    save_name = kwargs.get('save_name', None)

    num_levels = 20
    levels = np.linspace(vmin, vmax, num_levels + 1)
    norm = BoundaryNorm(levels, ncolors=256)

    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111, projection=ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -90, -65], ccrs.PlateCarree())
    ax.add_feature(crs.feature.OCEAN, facecolor='#7cb0d3')
    ax.gridlines(ls='--', color='grey', lw=0.5)

    shape_feature = ShapelyFeature(shapefile.geometry, ccrs.SouthPolarStereo(), edgecolor='k', facecolor='white', lw=.5)
    ax.add_feature(shape_feature)

    cs = ax.pcolormesh(lon, lat, data, norm=norm, cmap=cmap, transform=crs.crs.PlateCarree())
    cb = fig.colorbar(cs, ticks=np.linspace(vmin, vmax, 11), orientation='horizontal', shrink=0.7, pad=0.05)
    cb.ax.set_xlabel('Melt days', fontsize=10)
    ax.set_title(title, weight='bold', fontsize=18, pad=10)

    plt.plot([-59.9, -60], [-59.9, -60], c='k', label='Coastline', transform=crs.crs.PlateCarree())
    plt.legend(fontsize=10)
    continent_patch = Patch(facecolor='white', edgecolor='black', label='Antarctic Ice Shelfs')
    sea_patch = Patch(facecolor='#7cb0d3', edgecolor='black', label='Ocean')
    plt.legend(handles=[continent_patch, sea_patch], fontsize=10, loc='lower left', frameon=True, framealpha=1, edgecolor='black', fancybox=False)

    scalebar = ScaleBar(1, location='lower right', length_fraction=0.1)
    ax.add_artist(scalebar)

    if save_name:
        fig.savefig(save_name, dpi=100, bbox_inches='tight')

    plt.tight_layout()
    plt.show()
    plt.close()

# Tkinter GUI setup
root = tk.Tk()
root.title('BESUAL Melt Data Analyser')

# Global variables
data = None
gdf = None
start_year = None
end_year = None

def show_instructions():
    # Create a new window for instructions
    instructions_window = tk.Toplevel(root)
    instructions_window.title('Help')
    
    # Add a label with your instructions text
    instructions_text = '''
    Welcome to the BESUAL Melt Data Analyzer!
    
    This GUI allows you to analyze and visualize cumulative melt data from Antarctic regions. Follow the steps below to perform tasks:
    
    1. Load NetCDF File:
       - Click 'Load NetCDF' to upload your melt data in NetCDF format.
    
    2. Load Shapefile:
       - Click 'Load Shapefile' to load a compatible shapefile of your study region.
    
    3. Set Analysis Years:
       - Enter the start and end years to filter the dataset for the desired time range.
    
    4. Calculate Annual Average Cumulative Melt Days:
       - Compute the annual average of cumulative melt days over the selected period.
    
    5. Calculate the Year of Max Melt Days:
       - Identify and visualize the year with the highest cumulative melt days.
    
    6. Calculate the Year of Max Melt Extent:
       - Identify the year with the highest melt extent.
    
    7. Difference of Max Melt Days with Annual Average:
       - Plot the difference between the annual average melt days and the melt days of the year with maximum melt days.
    
    8. Plot Melt Extent Timeseries:
       - Examine trends in melt extent across the selected period, with both individual year and average data visualized.
    
    9. Melting Data of Top 15 Ice Shelves:
       - Automatically analyze and rank the 15 largest ice shelves based on their area and melt data.
    
    Happy analyzing!
    - Team BESUAL -
    '''
    
    # Display instructions in a label
    instructions_label = tk.Label(instructions_window, text=instructions_text, padx=10, pady=10, justify='left')
    instructions_label.pack()

    # Button to close the instructions window
    close_button = tk.Button(instructions_window, text='Close', command=instructions_window.destroy)
    close_button.pack(pady=10)
    
# Task 1: Load the shapefile
def load_shapefile():
    global gdf
    file_path = filedialog.askopenfilename(filetypes=[('Shapefiles', '*.shp')])
    if file_path:
        shapefile_label.config(text=file_path)
        gdf = gpd.read_file(file_path)
        messagebox.showinfo('File Loaded', 'Shapefile loaded successfully!')

# Task 2: Load the NetCDF file
def load_netcdf():
    global data
    file_path = filedialog.askopenfilename(filetypes=[('NetCDF Files', '*.nc')])
    if file_path:
        netcdf_label.config(text=file_path)
        data = xr.open_dataset(file_path)
        messagebox.showinfo('File Loaded', 'NetCDF file loaded successfully!')

# Assign start and end years
def set_years():
    try:
        global start_year, end_year, filtered_data, lon, lat, melt, time
        
        # Get input years
        start_year = int(start_year_entry.get())
        end_year = int(end_year_entry.get())

        # Get the range of years in the NetCDF time variable
        min_year = pd.to_datetime(data['time'].values.min()).year
        max_year = pd.to_datetime(data['time'].values.max()).year

        # Validate the input years
        if start_year < min_year or end_year > max_year:
            messagebox.showerror('Date Range Error', f'Year range out of bounds. Valid range: {min_year} to {max_year}.')
            return
        
        # Filter data based on the years
        filtered_data = data.sel(time=slice(f'{start_year}-01-01', f'{end_year}-12-31'))

        # Extract variables
        lon = filtered_data['lon'].values if 'lon' in filtered_data else filtered_data['x'].values
        lat = filtered_data['lat'].values if 'lat' in filtered_data else filtered_data['y'].values
        melt = filtered_data['melt']
        time = filtered_data['time']

        # Confirmation message
        messagebox.showinfo('Years Set', f'Start year: {start_year}, End year: {end_year} \nReady for calculations')
    except ValueError:
        # Handle invalid year input
        messagebox.showerror('Input Error', 'Please enter valid years.')

# Task 4: Process data for annual average melt
def annual_average_melt():
    try:
        if data is None or gdf is None:
            messagebox.showerror('Error', 'Please load both NetCDF and shapefile files.')
            return

        # Group data by year
        melt_year = melt.groupby('time.year').sum(dim='time')

        # Calculate annual average melt
        avg_melt_days = melt_year.mean(dim='year', skipna=True)
        avg_melt_days = avg_melt_days.where(avg_melt_days != 0, np.nan)

        # Call your plot function
        plot(gdf, lon, lat, avg_melt_days,title=f'Annual Average of Cumulative Melt Days ({start_year}-{end_year})',vmax=math.ceil(avg_melt_days.max() / 10) * 10,vmin=math.floor(avg_melt_days.min() / 10) * 10,save_name=f'Annual_Average_of_Cumulative_Melt_Days_{start_year}_{end_year}.png')
        
        messagebox.showinfo('Processing Complete', 'Annual average cumulative melt days plotted successfully.')

    except Exception as e:
        messagebox.showerror('Error', f'An error occurred: {e}')

# Task 5/6: Process max melt days
def max_melt_days():
    try:
        if data is None or gdf is None:
            messagebox.showerror('Error', 'Please load both NetCDF and shapefile files.')
            return

        melt_year = melt.groupby('time.year').sum(dim='time')

        max_melt_days = melt_year.max(dim=['y', 'x'], skipna=True).to_dataframe(name='Max_melt_days')
        max_melt_days_value = max_melt_days['Max_melt_days'].max()
        max_melt_days_years = max_melt_days[max_melt_days['Max_melt_days'] == max_melt_days_value].index.tolist()

        for year in max_melt_days_years:
            tmp = melt_year.sel(year=year)
            tmp = tmp.where(tmp != 0, np.nan)

            plot(gdf, lon, lat, tmp,title=f'Max Melt days is {int(max_melt_days_value)} in {year}',vmax=math.ceil(tmp.max() / 10) * 10,vmin=math.floor(tmp.min() / 10) * 10,save_name=f'Max Melt days is {int(max_melt_days_value)} in {year}.png')

            messagebox.showinfo('Max Melt Days', f'Max Melt days is {int(max_melt_days_value)} in {year}')
            
    except Exception as e:
        messagebox.showerror('Error', f'An error occurred: {e}')
        
# Task 5/6: Process max melt extent
def max_melt_extent():
    try:
        if data is None or gdf is None:
            messagebox.showerror('Error', 'Please load both NetCDF and shapefile files.')
            return

        melt_year = melt.groupby('time.year').sum(dim='time')

        # Calculate melt extent (number of pixels with non-NaN values) for each year in melt_year
        melt_year_filtered = melt_year.where(melt_year != 0)
        melt_extent = melt_year_filtered.notnull().sum(dim=['y', 'x'])

        # Find the year(s) with the maximum melt extent
        max_melt_extent_value = melt_extent.max()
        max_melt_extent_years = melt_extent.sel(year=melt_extent == max_melt_extent_value)

        year_value = max_melt_extent_years['year'].values.item()
        tmp = melt_year.sel(year=year_value)
        tmp = tmp.where(tmp != 0, np.nan)
        plot(gdf,lon,lat, tmp,title=f'Largest melt extent is {int(max_melt_extent_value)} in {year_value}',vmax=math.ceil(tmp.max() / 10) * 10,vmin=math.floor(tmp.min() / 10) * 10,save_name=f'Largest melt extent is {int(max_melt_extent_value)} in {year_value}.png')

        messagebox.showinfo('Max Melt Extent', f'Max Melt Extent is {int(max_melt_extent_value)} in the following year/years: {", ".join(map(str, max_melt_extent_years.coords["year"].values))}')

    except Exception as e:
        messagebox.showerror('Error', f'An error occurred: {e}')

# Plot the difference of annual average melt days and the year with the maximum melt days
def diff_melt_days():
    try:
        if data is None or gdf is None:
            messagebox.showerror('Error', 'Please load both NetCDF and shapefile files.')
            return
        
        # Identify the maximum melt days value and year
        melt_year = melt.groupby('time.year').sum(dim='time')
        max_melt_days = melt_year.max(dim=['y', 'x'], skipna=True).to_dataframe(name='Max_melt_days')
        max_melt_days_value = max_melt_days['Max_melt_days'].max()
        max_melt_days_years = max_melt_days[max_melt_days['Max_melt_days'] == max_melt_days_value].index.tolist()

        # Group data by year within the preferred time period
        melt_year = melt.groupby('time.year').sum(dim='time')
        avg_melt_days = melt_year.mean(dim='year', skipna=True)
        
        for year in max_melt_days_years:  # Iterate through the years in max_melt_days_years
            tmp = melt_year.sel(year=year)
            tmp = tmp.where(tmp != 0, np.nan)
            
            difference = avg_melt_days - tmp
            plot(gdf, lon, lat, difference,title=f'Difference to Average Annual Melt Days in Year {year}',vmax=math.ceil(difference.max() / 10) * 10,vmin=math.floor(difference.min() / 10) * 10,cmap='bwr',save_name=f'Difference to Average Annual Melt Days in Year {year}.png')
            
            messagebox.showinfo('Difference', f'Difference to Average Annual Melt Days in Year {year} plotted successfully')

    except Exception as e:
        messagebox.showerror('Error', f'An error occurred: {e}')

# Task 8: Plot melt extent
def plot_melt_extent():
    try:
        if data is None or gdf is None:
            messagebox.showerror('Error', 'Please load both NetCDF and shapefile files.')
            return

        melt_data_flattened = melt.values.reshape(melt.shape[0], -1)
        melt_extent = np.nansum(melt_data_flattened > 0, axis=1)
        days_of_year = time.dt.dayofyear
        
        # Creating Pandas dataframe
        df_melt_extent = pd.DataFrame({'datetime': time['time'],'day_of_year': days_of_year,'melt_extent': melt_extent})

        # Adding the 'year_group' column to df_melt_extent
        df_melt_extent['year_group'] = df_melt_extent['datetime'].apply(lambda x: x.year if x.month >= 6 else x.year - 1)
        df_melt_extent = df_melt_extent.loc[(df_melt_extent['year_group'] >= start_year) & (df_melt_extent['year_group'] <= end_year)]
        
        # Add a 'custom_date' column to adjust dates for plotting
        def adjust_date(date):
            if date.month >= 6:
                return pd.Timestamp(year=1979, month=date.month, day=date.day)
            else:
                return pd.Timestamp(year=1980, month=date.month, day=date.day)
        
        df_melt_extent = df_melt_extent.copy()  # Avoid SettingWithCopyWarning
        df_melt_extent['custom_date'] = df_melt_extent['datetime'].apply(adjust_date)
        
        # Calculate the average Extent values by 'custom_date'
        average_extent = df_melt_extent.groupby('custom_date')['melt_extent'].mean()
        
        # Plot grey lines for each year
        plt.figure(figsize=(12, 6))
        
        # Variable to define the year with the largest extent
        melt_year = melt.groupby('time.year').sum(dim='time')
        melt_year_filtered = melt_year.where(melt_year != 0)
        melt_extent = melt_year_filtered.notnull().sum(dim=['y', 'x'])
        max_melt_extent_value = melt_extent.max()
        max_melt_extent_years = melt_extent.sel(year=melt_extent == max_melt_extent_value)
        max_melt_extent_year_value = int(max_melt_extent_years['year'].values[0])
        
        for year, group in df_melt_extent.groupby('year_group'):
            color = 'red' if year == max_melt_extent_year_value else 'grey'
            alpha = 1.0 if year == max_melt_extent_year_value else 0.5
            label = f'Max Melt Extent ({year})' if color == 'red' else None  # Label the red line
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
        plt.savefig(f'Melt Extent for {start_year}-{end_year} period', dpi=100)
        plt.show()

        messagebox.showinfo('Melt Extent', f'Melt Extent for {start_year}-{end_year} period plotted successfully')

    except Exception as e:
        messagebox.showerror('Error', f'An error occurred: {e}')

# Task 9: 15 largest ice shelves and average melt extent
def top_15():
    try:
        if data is None or gdf is None:
            messagebox.showerror('Error', 'Please load both NetCDF and shapefile files.')
            return

        gdf2 = gdf.to_crs(epsg=3031)
        gdf2['area_calculated'] = gdf2.geometry.area
        sorted_gdf = gdf2.sort_values('area_calculated', ascending=False)
        top_15_gdf = sorted_gdf.head(15)

        melt2 = melt.rio.write_crs('EPSG:3031', inplace=True)

        # Create an empty list to store results
        results = []

        # Loop through all areas in 'top_15_gdf'
        for area_name in top_15_gdf['NAME']:
            # Select the region for the current area
            region = top_15_gdf[top_15_gdf['NAME'] == area_name]
            
            # Clip the data for the current region
            clipped = melt2.rio.clip(region.geometry, region.crs)
            
            # Mask the clipped data where the values are equal to 1
            melt_masked = clipped.where(clipped == 1)
            
            # Resample the data by year and calculate the mean
            yearly_average_melt = melt_masked.resample(time='1YE').mean()
            
            # Sum the values over the spatial dimensions x and y
            yearly_sum = yearly_average_melt.sum(dim=['x', 'y'])
            
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
        plt.figure(figsize=(8, 8), dpi=80)

        # Create the pie chart
        plt.pie(df_top_15_melt['average_yearly_melt_extent'], labels=df_top_15_melt['NAME'], autopct='%1.1f%%', startangle=90)
        plt.title(f'Average Melting Extent for the Largest 15 Ice Shelves from {start_year} to {end_year}', pad=20)
        plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
        plt.savefig(f'Average Melting Extent for the Largest 15 Ice Shelves from {start_year} to {end_year}.png', bbox_inches='tight')  # Save as PNG file
        plt.show()

        messagebox.showinfo('Melting Extent for the Largest 15 Ice Shelves', f'Average Melting Extent for the Largest 15 Ice Shelves from {start_year} to {end_year} plotted successfully and data exported to df_top_15_melt_{start_year}to{end_year}.csv.')

    except Exception as e:
        messagebox.showerror('Error', f'An error occurred: {e}')

def on_closing():
    # Show a thank-you message when the window is closed
    messagebox.showinfo('Thank You', 'Thank you for using the application!')
    root.destroy()  # Destroy the root window and close the application
    
# Tkinter Frame
frame = tk.Frame(root)
frame.config(width=750, height=420)  # Adjust as needed
frame.pack(padx=10, pady=10)

# Buttons
help_button = tk.Button(root, text='Help', command=show_instructions)
help_button.place(x=30, y=405)

load_netcdf_button = tk.Button(frame, text='Load NetCDF File', command=load_netcdf)
load_netcdf_button.place(x=20, y=10)

load_shapefile_button = tk.Button(frame, text='Load Shapefile', command=load_shapefile)
load_shapefile_button.place(x=20, y=50)

set_year_button = tk.Button(frame, text='Set Years', command=set_years)
set_year_button.place(x=480, y=90)

task4_button = tk.Button(frame, text='Calculate Annual Average Cumulative Melt Days', command=annual_average_melt)
task4_button.place(x=350, y=150)

task5a_button = tk.Button(frame, text='Calculate the Year of Max Melt Days', command=max_melt_days)
task5a_button.place(x=350, y=190)

task5b_button = tk.Button(frame, text='Calculate the Year of Max Melt Extent', command=max_melt_extent)
task5b_button.place(x=350, y=230)

task6b_button = tk.Button(frame, text='Difference of Max Melt Days with Annual Average', command=diff_melt_days)
task6b_button.place(x=350, y=270)

task8_button = tk.Button(frame, text='Plot Melt Extent Timeseries', command=plot_melt_extent)
task8_button.place(x=350, y=310)

task9_button = tk.Button(frame, text='Melting Data of Top 15 Ice Shelves', command=top_15)
task9_button.place(x=350, y=350)

# Add a label to display the file path
netcdf_label = tk.Label(frame, text='No file selected', fg='black', anchor='w')
netcdf_label.place(x=150, y=15)  # Adjust the `x, y` coordinates as needed

shapefile_label = tk.Label(frame, text='No file selected', fg='black', anchor='w')
shapefile_label.place(x=150, y=55)  # Adjust the `x, y` coordinates as needed

start_year_label = tk.Label(root, text='Start Year:')  # Directly in the root window
start_year_label.place(x=40, y=105)  # Adjust coordinates relative to the window
start_year_entry = tk.Entry(root)  # Directly in the root window
start_year_entry.place(x=100, y=105)  # Adjust coordinates relative to the window

end_year_label = tk.Label(root, text='End Year:')
end_year_label.place(x=250, y=105) 
end_year_entry = tk.Entry(root)
end_year_entry.place(x=310, y=105) 

# Load the image using Pillow
image_path = 'sit_antarctica_map.png'  # Replace with the path to your image
image = Image.open(image_path)
# Resize the image if needed
image = image.resize((300, 250))  # Resize the image to fit the GUI
# Convert the image to Tkinter format
image_tk = ImageTk.PhotoImage(image)
# Create a label to display the image
image_label = tk.Label(root, image=image_tk)
image_label.place(x=30, y=150)

logo_path = 'logo.webp'  # Replace with the path to your image
logo = Image.open(logo_path)
# Resize the image if needed
logo = logo.resize((110, 110))  # Resize the image to fit the GUI
# Convert the image to Tkinter format
logo_tk = ImageTk.PhotoImage(logo)
# Create a label to display the image
logo_label = tk.Label(root, image=logo_tk)
logo_label.place(x=650, y=15)
    
root.protocol('WM_DELETE_WINDOW', on_closing)

root.mainloop()
