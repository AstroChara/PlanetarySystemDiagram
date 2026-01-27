import pandas as pd
import numpy as np
from scipy.stats import gmean
from astropy import constants as const
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

import src.utils as utils



# INPUTS

input_folder = 'Input/'
output_folder = 'Output/'

input_filename = 'SolarSystem.csv'
diagram_title = 'Solar System'
output_filetype = '.png'

x_mode = 'd'
# 'd': distance
# 'P': orbital period
# 'S': irradiance

plot_unit_d = 'au'
plot_unit_t = 'd'
plot_unit_irradiance = 'So'

# Render primary flag
render_primaries = True

# Set secondaries' scale
secondary_standard_mass = const.M_earth.value * 1
secondary_standard_markersize = 8



# IMPORT DATA

df = pd.read_csv(input_folder + input_filename)

# https://stackoverflow.com/questions/32072076/find-the-unique-values-in-a-column-and-then-sort-them
system_names = df['System'].unique()
N_systems = len(system_names)



# DRAW DIAGRAM

fig, ax = plt.subplots()

plt.title(diagram_title)
fig.set_size_inches(15, (N_systems*2)+1)
plt.grid(True, which="both", alpha=0.25)
plt.style.use('dark_background')

# https://stackoverflow.com/questions/49471424/add-custom-tick-with-matplotlib
ticks = range(N_systems)
ax.set_yticks(ticks)
ax.set_yticklabels(system_names)

plt.ylim(np.min(ticks)-0.5, np.max(ticks)+0.5)
plt.gca().invert_yaxis()

ax.set_xscale('log')
if x_mode == 'd':
    plt.xlabel('Semi-major axis ({0})'.format(plot_unit_d))
elif x_mode == 'P':
    plt.xlabel('Orbital period ({0})'.format(plot_unit_t))
elif x_mode == 'S':
    plt.gca().invert_xaxis()
    plt.xlabel('Irradiance ({0})'.format(plot_unit_irradiance))

plt.tight_layout()



# Render objects
for system_index in ticks:
    df_StarSys = df[df['System'] == system_names[system_index]].reset_index(drop=True)
    
    
    # Handle primaries
    df_primaries = df_StarSys[df_StarSys['ObjectType'] == 'Primary'].reset_index(drop=True)
    N_primaries = len(df_primaries)
    
    M_pri = 0
    unit_M_pri = 'M_sun'
    L_pri = 0
    unit_L_pri = 'L_sun'
    
    for index in range(N_primaries):
        M, unit_M = df_primaries['Mass'][index], df_primaries['unit_M'][index]
        M = utils.mass_conversion(M,unit_M,unit_M_pri)
        M_pri += M
        
        L = df_primaries['Luminosity'][index]  # Unit is always L_sun
        L_pri += L
        
        if render_primaries & (x_mode == 'd'):
            # Name
            name = df_primaries['Name'][index]
            
            # Radius
            R, unit_R = df_primaries['Radius'][index], df_primaries['unit_R'][index]
            R = utils.length_conversion(R,unit_R,plot_unit_d)
            
            # Colour
            colour = df_primaries['Colour'][index]
            
            # Draw object itself
            y_pos = system_index-0.5 + (index/N_primaries)
            plt.gca().add_patch(Rectangle((0, y_pos), R, 1/N_primaries,
                                          facecolor=colour))
            # Draw name
            x_pos_name = R * 0.9
            plt.text(x_pos_name,y_pos + 0.5/N_primaries,
                     name,
                     rotation='vertical',
                     horizontalalignment='center',verticalalignment='center',
                     color='#000000',alpha=0.75)
    
    
    # Handle planetary system
    df_PlanetSys = df_StarSys[df_StarSys['ObjectType'] != 'Primary'].reset_index(drop=True)
    
    N_secondaries = 0
    
    for index in range(len(df_PlanetSys)):
        objectType = df_PlanetSys['ObjectType'][index]
        
        name = df_PlanetSys['Name'][index]
        
        # MASS: used by Secondary
        M, unit_M = df_PlanetSys['Mass'][index], df_PlanetSys['unit_M'][index]
        if not np.isnan(M):
            M = utils.mass_conversion(M,unit_M,'kg')
        
        # RADIUS: used by Secondary
        R, unit_R = df_PlanetSys['Radius'][index], df_PlanetSys['unit_R'][index]
        
        # DISTANCES: used by all
        #   SemiMajorAxis, InnerDistance, OuterDistance, unit_d, Eccentricity
        unit_d = df_PlanetSys['unit_d'][index]
        # This scheme assumes an object has either
        # - both SemiMajorAxis and Eccentricity
        # - both InnerDistance and OuterDistance
        a = utils.length_conversion(df_PlanetSys['SemiMajorAxis'][index],unit_d,plot_unit_d)
        e = df_PlanetSys['Eccentricity'][index]
        if (not np.isnan(a)) and (not np.isnan(e)):
            d_inner = a * (1-e)
            d_outer = a * (1+e)
        else:
            d_inner = utils.length_conversion(df_PlanetSys['InnerDistance'][index],unit_d,plot_unit_d)
            d_outer = utils.length_conversion(df_PlanetSys['OuterDistance'][index],unit_d,plot_unit_d)
        
        # ORBITAL PERIOD: used by all
        P, unit_P = df_PlanetSys['OrbitalPeriod'][index], df_PlanetSys['unit_P'][index]
        if not np.isnan(P):
            P = utils.time_conversion(P,unit_P,plot_unit_t)
            P_min, P_max = P, P  # placeholder values
        if np.isnan(P):
            P = utils.orbital_period(a,plot_unit_d, M_pri,unit_M_pri, plot_unit_t)
            P_min = utils.orbital_period(d_inner,plot_unit_d, M_pri,unit_M_pri, plot_unit_t)
            P_max = utils.orbital_period(d_outer,plot_unit_d, M_pri,unit_M_pri, plot_unit_t)
        
        # INCLINATION: used by Secondary
        # denotes that the provided mass value is minimum mass
        inc = df_PlanetSys['Inclination'][index] * np.pi/180
        if not np.isnan(inc):
            M = M / np.sin(inc)
        
        # COLOUR: used by all
        colour = df_PlanetSys['Colour'][index]
        
        
        # Handle modes
        if x_mode == 'd':
            x_min = d_inner
            x_med = a
            x_max = d_outer
        elif x_mode == 'P':
            x_min = P_min
            x_med = P
            x_max = P_max
        elif x_mode == 'S':
            x_min = utils.distance_to_irradiance(d_outer,L_pri)  # used by Secondary and Belt
            x_med = utils.distance_to_irradiance(a,L_pri)  # used by Secondary and Boundary
            x_max = utils.distance_to_irradiance(d_inner,L_pri)  # used by Secondary and Belt
        
        
        # Draw objects
        if objectType == 'Secondary':
            N_secondaries += 1
            # Draw object itself
            markersize = np.cbrt(M/secondary_standard_mass) * secondary_standard_markersize
            plt.plot(x_med,system_index,'.',
                     markersize=markersize,
                     markeredgewidth=0,
                     color=colour)
            # Draw name
            name_y_offset = (markersize/fig.dpi) / 4 + 0.1
            if N_secondaries % 2 == 1:  # odd, name below object
                name_y_offset = name_y_offset
                va_mode = 'top'
            else:  # even, name above object
                name_y_offset = -name_y_offset
                va_mode = 'bottom'
            plt.text(x_med,system_index + name_y_offset,
                     name,
                     horizontalalignment='center',verticalalignment=va_mode)
            # Draw pericentre and apocentre range
            if x_mode != 'P':
                plt.plot([x_min,x_max],[system_index,system_index],
                         linewidth=1,
                         color='#ffffff',alpha=0.5)
        
        if objectType == 'Belt':
            width = x_max - x_min
            y_pos = system_index-0.5
            # Draw belt itself
            plt.gca().add_patch(Rectangle((x_min, y_pos), width, 1,
                                          facecolor=colour,alpha=0.75))
            # Draw name
            name_pos = gmean([x_max,x_min])
            plt.text(name_pos,system_index,
                     name,
                     rotation='vertical',
                     horizontalalignment='center',verticalalignment='center',
                     color='#ffffff',alpha=0.5)
        
        if objectType == 'Boundary':
            y_pos = system_index-0.5
            plt.plot([x_med,x_med],[y_pos,y_pos+1],
                     '--',
                     color=colour,alpha=0.5)



# SAVE OUTPUT FILE

output_filename_suffix = ''
if x_mode == 'd':
    output_filename_suffix = '_distance_{0}'.format(plot_unit_d)
elif x_mode == 'P':
    output_filename_suffix = '_period_{0}'.format(plot_unit_t)
elif x_mode == 'S':
    output_filename_suffix = '_irradiance_{0}'.format(plot_unit_irradiance)

output_filepath = Path(output_folder + diagram_title + output_filename_suffix + output_filetype)
output_filepath.parent.mkdir(exist_ok=True, parents=True)
plt.savefig(output_filepath)
