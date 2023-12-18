from utils import par_for
import folium
import numpy as np
import haversine as hs
from dataclasses import dataclass,field
from haversine import inverse_haversine, Unit, Direction
from folium.plugins import HeatMap
import requests
import urllib
import pandas as pd
import time
from scipy.spatial.distance import cdist
from functools import lru_cache
import os

def show_map(hiker, trail=None, add_missing_radius=False, zoom=15, sims=None, trail_points=False,
             trail_heat=True, grid=None, probs=None, trail_enumerate=False, plot_origin=False, probs_point=True):
    map = folium.Map(location=hiker.loc, zoom_start=zoom)
    tile = folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='Esri Satellite', overlay=False, control=True).add_to(map)

    if plot_origin:
        folium.Marker(location=hiker.loc, popup="Last known location", icon=folium.Icon(color="red")).add_to(map)

    if add_missing_radius:
        folium.Circle(hiker.loc, radius=hiker.missing_radius).add_to(map)
    if trail is not None:
        if isinstance(trail, pd.DataFrame):
            trail_list = trail[['latitude', 'longitude']].values
            p = trail[['prob']].values if 'prob' in trail else None
        else:
            trail_list = trail
        folium.PolyLine(trail_list, tooltip="trail", color='red').add_to(map)
        if trail_enumerate:
            for i, loc in enumerate(trail_list):
                c = 'green' if i == 0 else None
                c = 'blue' if (i + 1) == len(trail) else c
                txt = str(i) if p is None else str(p[i])
                if c is not None:
                    folium.Marker(location=loc, popup=txt, icon=folium.Icon(color=c)).add_to(map)

    if sims is not None:
        if trail_heat:
            HeatMap(data=sims[['latitude', 'longitude']], min_opacity=0.1).add_to(map)
        if trail_points:
            for i, row in sims.iterrows():
                folium.CircleMarker([row['latitude'], row['longitude']], radius=1, fill=True).add_to(map)

    if probs is not None:
        HeatMap(data=probs, min_opacity=0.1).add_to(map)
        if probs_point:
            for i, row in probs.iterrows():
                folium.CircleMarker([row['latitude'], row['longitude']], popup=row['prob'],
                                    radius=1, fill=True, color='red').add_to(map)

    if grid is not None:
        grid['elev_meters']=(grid['elev_meters']-grid['elev_meters'].min())/(grid['elev_meters'].max()-grid['elev_meters'].min())
        HeatMap(data=grid[['latitude', 'longitude','elev_meters']], min_opacity=0.1).add_to(map)
        #folium.PolyLine(grid[['latitude', 'longitude']], color='blue').add_to(map)
        #folium.PolyLine(grid[['latitude', 'longitude']].sort_values(by='latitude'), color='blue').add_to(map)
        # for i,row in grid.iterrows():
        #    folium.CircleMarker([row['latitude'], row['longitude']],radius=0.5,fill=True,color='red').add_to(map)

    return map

#https://towardsdatascience.com/calculating-distance-between-two-geolocations-in-python-26ad3afe287b
def random_walk_simple(hiker,N=1,end_point_only=False):
    distance5min = hiker.max_speed*60*5 # Speed in m/5 minutes
    trails=[]
    for n in range(N):
        trail= [hiker.loc]
        for t in range(12*hiker.hours_missing):
            bearing= np.random.uniform(0,2*np.pi,1)
            new_loc=hs.inverse_haversine(trail[-1], distance5min, bearing,unit=Unit.METERS)
            trail.append(new_loc)
        if end_point_only:
            trails.append(trail[-1])
        else:
            trails.append(trail)
    return trails


def get_elevation_grid(hiker,grid_range=6000,grid_resolution=10,recache=False):
    """Query service using lat, lon. add the elevation values as a new column.
        Note that this API only allows 100 locations per request. Hence, for a higher number of rows,
        it must be broken down into different requests.
        Source (modified from):
        https://gis.stackexchange.com/questions/338392/getting-elevation-for-multiple-lat-long-coordinates-in-python
        Getting 401 /402 next batch
        Finished querying in 392.135259 minutes
    """
    url = r'https://epqs.nationalmap.gov/v1/json?'
    #cache_name = './PycharmProjects/medium/scripts/elevation_res10m_v2.csv'
    #cache_name = 'elevation_res10m_v2.csv'
    cache_name = 'elevation_res10m_v3.csv'

    grid_N = int(grid_range / grid_resolution)
    if not os.path.isfile(cache_name) and os.path.isfile('./PycharmProjects/medium/scripts/' + cache_name):
        cache_name='./PycharmProjects/medium/scripts/' + cache_name

    if os.path.isfile(cache_name):
        df = pd.read_csv(cache_name, index_col=None)
    else:
        # Define grid for elevation data based on function parameters
        grid = []
        origin = hs.inverse_haversine(hiker.loc, grid_range / 2, Direction.WEST, unit=Unit.METERS)
        origin = hs.inverse_haversine(origin, grid_range / 2, Direction.NORTH, unit=Unit.METERS)
        for east_offset in range(grid_N):
            lat, long = hs.inverse_haversine(origin, east_offset * grid_resolution, Direction.EAST, unit=Unit.METERS) # returns lat/lon
            grid.append([lat, long, east_offset * grid_resolution, 0])
            for south_offset in range(grid_N):
                _,long = hs.inverse_haversine((lat,long), grid_resolution, Direction.SOUTH, unit=Unit.METERS)
                grid.append([lat,long, east_offset * grid_resolution, south_offset * grid_resolution])
        df = pd.DataFrame(grid, columns=['latitude', 'longitude', 'east_offset', 'south_offset'])
        df['elev_meters'] = np.nan
        df['el_query'] = np.floor(df.index / 100)
    print("Total latitude points in the grid =%s"%df.latitude.nunique())
    print("Total longitude points in the grid =%s" % df.longitude.nunique())
    assert df.latitude.nunique()==df.longitude.nunique(), "Lat/Longitude mismatch !"

    if recache:
        C = df['el_query'].nunique()
        init = 0
        print("Querying elevation data for grid size=%s 100-batch queries=%s grid_N=%s (grid_N^2= %s ) " % (len(df), C,grid_N,grid_N**2))
        start = time.time()
        q = 0
        for q, query_df in df[df['elev_meters'].isna()].groupby('el_query'):
            elevations = []
            for lat, lon in zip(query_df['latitude'], query_df['longitude']):
                params = {'output': 'json', 'x': lon, 'y': lat, 'units': 'Meters'}
                result = requests.get((url + urllib.parse.urlencode(params)))
                rdict = result.json()
                if 'value' in rdict:
                    elevations.append(rdict['value'])
                else:
                    elevations.append(np.nan)
                init = 1
            if init:
                time.sleep(1)
                df.to_csv(cache_name)
                print("Getting %s /%s next batch" % (q, C))
            df.loc[query_df.index, 'elev_meters'] = elevations
            missing = df['elev_meters'].count() / len(df)
            print("Got %0.3f valid elevation data (N=%s) from query %s" % (missing, len(query_df), q))

        print("Finished querying in %3f minutes" % ((time.time() - start) / 60))
        missing = df['elev_meters'].count() / len(df)
        print("Saving %0.3f valid elevation data (N=%s) from query %s" % (missing, len(df), q))
        df.to_csv(cache_name, index=False)

    rem_cols = [x for x in df.columns if x.find('Unnamed') > -1]
    df.drop(columns=rem_cols, inplace=True)
    xy_grid=df.dropna().drop_duplicates(subset=['east_offset','south_offset']).pivot(index='south_offset',columns='east_offset',values='elev_meters')
    return df, xy_grid

@lru_cache
def old_get_elevation(loc,df_elevation):
    # TODO: Add cache
    def closest_point(point, points):
        """ Find closest point from a list of points.
            From: https://stackoverflow.com/questions/38965720/find-closest-point-in-pandas-dataframes
        """
        return points[cdist([point], points).argmin()]

    def match_value(df, col1, x, col2):
        """ Match value x from col1 row to value in col2. """
        return df[df[col1] == x][col2].values[0]

    row = closest_point(loc, list(df_elevation['point']))
    elevation = float(match_value(df_elevation, 'point', row, 'elev_meters'))

    return elevation


def get_rescue_grid(hiker, grid_range=500, grid_resolution=10, simulation_data=None):
    grid_N = int(grid_range / grid_resolution)
    grid = []
    origin = hs.inverse_haversine(hiker.loc, grid_range / 2, Direction.WEST, unit=Unit.METERS)
    origin = hs.inverse_haversine(origin, grid_range / 2, Direction.NORTH, unit=Unit.METERS)
    for east_offset in range(grid_N):
        grid_point = hs.inverse_haversine(origin, east_offset * grid_resolution, Direction.EAST, unit=Unit.METERS)
        grid.append(grid_point)
        for _ in range(grid_N):
            grid_point = hs.inverse_haversine(grid_point, grid_resolution, Direction.SOUTH, unit=Unit.METERS)
            grid.append(grid_point)
            d = (grid_point[0] - hiker.loc[0]) ** 2 + (grid_point[1] - hiker.loc[1]) ** 2

    df = pd.DataFrame(np.squeeze(np.array(grid)), columns=['latitude', 'longitude'])
    df['prob'] = np.nan
    prob = []
    N = len(data)
    bound_width = grid_resolution / 2
    lats = df.latitude.unique()
    lats.sort
    longs = df.longitude.unique()
    longs.sort
    check = 0
    for index, row in df.iterrows():
        lat_ind = np.argmin(np.abs(lats - row.latitude))
        long_ind = np.argmin(np.abs(longs - row.longitude))
        lat_ub = np.mean([lats[0] if lat_ind == 0 else lats[lat_ind - 1], row.latitude])
        lat_lb = np.mean([lats[-1] if (lat_ind + 1) == len(lats) else lats[lat_ind + 1], row.latitude])
        long_lb = np.mean([longs[0] if long_ind == 0 else longs[long_ind - 1], row.longitude])
        long_ub = np.mean([longs[-1] if (long_ind + 1) == len(longs) else longs[long_ind + 1], row.longitude])
        mask1 = (data.latitude <= lat_ub) & (data.latitude >= lat_lb)
        mask2 = (data.longitude <= long_ub) & (data.longitude >= long_lb)
        n = len(data[mask1 & mask2])
        df.loc[index, 'prob'] = n / N
        check += n
    if check < N:
        print("****Warning: Search grid is too small, %s points outside the grid" % (N - check))
    return df


def greedy_search(rescuer, rescue_grid, sims, start_at=None):
    # Perform greedy search by traversing through points of highest probabilty
    # in the neighboorhood

    lats = rescue_grid.latitude.unique()
    lats.sort
    longs = rescue_grid.longitude.unique()
    longs.sort
    if start_at is None:
        trail = rescue_grid.iloc[rescue_grid['prob'].idxmax()].copy().to_frame().T.reset_index(drop=True)
    else:
        trail = start_at
    # Start with the highest prob point and proceed according to probs of highest neighbors
    p = trail.iloc[-1].prob * rescuer.prob_find
    origin = trail.iloc[-1]
    patience_limit = 2
    patience = 0
    for i in range(rescuer.max_grid_points - 1):
        row = trail.iloc[-1]
        lat_ind = np.argmin(np.abs(lats - row.latitude))
        long_ind = np.argmin(np.abs(longs - row.longitude))
        long_south = np.min([lat_ind + 1, len(lats) - 1])
        long_north = np.max([lat_ind - 1, 0])
        long_west = np.min([long_ind + 1, len(longs) - 1])
        long_east = np.max([long_ind - 1, 0])
        hood = rescue_grid[(rescue_grid.latitude >= lats[long_south]) &
                           (rescue_grid.latitude <= lats[long_north]) &
                           (rescue_grid.longitude >= longs[long_east]) &
                           (rescue_grid.longitude <= longs[long_west])
                           ].copy()

        hood = pd.merge(hood, trail, on=['latitude', 'longitude', 'prob'], how='left', indicator='Exist')
        hood['Exist'] = np.where(hood.Exist == 'both', True, False)
        hood = hood[hood.Exist == 0].sort_values(by='prob')
        if len(hood) == 0 or patience > patience_limit:
            break
        if hood[['prob']].tail(1).values == 0:
            patience += 1
            if len(hood) > 1:
                # If all probs are zero, pick one that will return us to the original location (mode of the dist)
                hood['distance'] = (hood['latitude'] - origin['latitude']) ** 2 + (
                            hood['longitude'] - origin['longitude']) ** 2
                hood = hood.sort_values(by='distance', ascending=False)
        else:
            patience = 0

        trail = pd.concat([trail, hood[['latitude', 'longitude', 'prob']].tail(1)], axis=0)
        p += (trail.iloc[-1].prob * rescuer.prob_find)

    trail['prob_find'] = rescuer.prob_find
    trail = trail.reset_index(drop=True)
    return trail, p


def get_post_rescue_grid(rescue_grid, rescue_plan):
    # Updating the posterior using:
    # https://en.wikipedia.org/wiki/Bayesian_search_theory

    # Operate with log in order to make computations stable
    rescue_grid_post = rescue_grid.copy()
    rescue_grid_post['prob'] = np.log(rescue_grid_post['prob'] + 1e-9)
    for i, row in rescue_plan.iterrows():
        select_point = (rescue_grid_post.latitude == row.latitude) & (rescue_grid_post.longitude == row.longitude)
        ind = rescue_grid_post[select_point].index.values
        p = rescue_grid_post.loc[ind, 'prob'].values[0]
        q = np.log(row.prob_find + 1e-9)
        q1 = np.log(1 - row.prob_find + 1e-9)
        den = np.log(1 - np.exp(p + q))  # 1-p*q
        rescue_grid_post.loc[ind, 'prob'] = p + q1 - den  # p*(1-q)/den
        mask = rescue_grid_post.index.values == ind
        rescue_grid_post.loc[mask, 'prob'] = rescue_grid_post.loc[mask, 'prob'] - den
        # P=rescue_grid_post['prob'].sum()
        # if P>1:
        #    raise Exception("P=%s>1"%P)
    rescue_grid_post['prob'] = np.exp(rescue_grid_post['prob'])
    return rescue_grid_post


def random_walk_with_elevation(hiker, N=1, end_point_only=True, verbose=False):
    '''
        Updated with following assumptions:
        1. Velocity is a function of elevation
        2. Hiker cannot go beyond a positive elevation limit (will try another direction before yielding an iteration)
        3. If Hiker goes beyond a negative elevation limit it will stay there (terminal point, simulating a fall)
    '''
    distance5min = hiker.max_speed * 60 * 5  # Speed in m/5 minutes
    trails = []
    max_grad = 10 * 2 * np.pi / 180  # Assuming max 45 deg climbing gradient
    fall_grad = -30 * 2 * np.pi / 180  # Assuming max 45 deg climbing gradient
    maxg = 0
    for n in range(N):
        trail = [hiker.loc]
        past_elevation = get_elevation(hiker.loc)
        terminate = False
        t = 0
        while t < (12 * hiker.hours_missing):
            bearing = np.random.uniform(0, 2 * np.pi, 1)
            # Make distance and speed towards new location function of elevation
            target_loc = hs.inverse_haversine(trail[-1], distance5min, bearing, unit=Unit.METERS)
            target_elevation = get_elevation(target_loc)
            elevation_diff = past_elevation - target_elevation
            gradient = np.arctan(elevation_diff / distance5min)
            if gradient > max_grad:  # Dont move if we hit a wall
                new_loc = trail[-1]
                continue
            elif gradient < fall_grad:  # Fall, terminal state
                terminate = True
                new_loc = trail[-1]
                step = 0
            else:
                # Gradient with physical limit, set velocity according to grad and calculate location
                step = distance5min * np.cos(gradient) if gradient > 0 else distance5min
                new_loc = hs.inverse_haversine(trail[-1], step, bearing, unit=Unit.METERS)

            if gradient * 180 / (2 * np.pi) > maxg:
                maxg = gradient * 180 / (2 * np.pi)
                if verbose:
                    print("angle=%0.3f step=%0.3f (%0.3f) max=%0.3f" % (maxg, step, step / distance5min, distance5min))

            trail.append(new_loc)
            past_elevation = target_elevation
            t += 1
            if terminate:
                break
        if end_point_only:
            trails.append(trail[-1])
        else:
            trails.append(trail)
    return trails

