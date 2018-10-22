"""
all-to-one Version 3.

script for Rome data, 14_days, go Cluster!

"""


import numpy as np
import pickle 
import argparse


#import matplotlib.pyplot as plt
#plt.ion()


""" Some important functions... """

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    All args must be of equal length.    
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    m = 6367 * c * 1000
    haversine_distance_metres = m.astype(int)

    return haversine_distance_metres


def SourceArrayBuilder(source_long_list, source_lat_list, max_num_taxis):
    """ function to build sink array, for easy vectorised distance matrix calc later in taxi message flooding simulation
    """
    source_longitude_array = np.ones((max_num_taxis,len(source_long_list)))
    source_latitude_array = np.ones((max_num_taxis,len(source_lat_list)))
    for i in range(0,source_longitude_array.shape[1]):
        source_longitude_array[:,i] = source_longitude_array[:,i]*source_long_list[i]
        source_latitude_array[:,i] = source_latitude_array[:,i]*source_lat_list[i]

    return source_longitude_array, source_latitude_array


""" User inputs, for cluster simulation runs """
parser = argparse.ArgumentParser()
parser.add_argument('-s', action='store', type=int, dest='sink_id', help='Which sink to use for simulation?')
#parser.add_argument('-t', action='store', type=int, dest='start_time', help='When should simulation start, hours [0-23]?')
results = parser.parse_args()


"""
#C207 - NO PANDAS!!!!!!!!!!!!!
sink_id = 43
city = 'Rome' #SF
num_days = 14
simulation_time_limit_hours = 4 #limit at which simulation stops

path_to_data_files  = ("/home/user/ClusterSetUp-All-to-One/%s-Data/" % city)
sources_location_file_path = ("/home/user/ClusterSetUp-All-to-One/%s-Data/" % city)

VANET_dict_filename =  "split_dict_15_till_24_hrs_No_Pandas_VANET_august_edit_combined_14_days_rome.pickle"
Position_dict_filename = "split_dict_15_till_24_hrs_No_Pandas_august_edit_taxi_positions_combined_14_days_rome.pickle"

sources_locations_filename = ("%s_random_500_sinks_locations.pickle" % city)

results_data_file_path = ("/home/user/ClusterSetUp-All-to-One/%s-Results/%i_days/" % (city, num_days))
results_filename = ('%i_sink_id_all_to_1_%s_%i_days_results.pickle' % (sink_id, city, num_days))
"""


"""
#NiGB - NO PANDAS!!!!!!!!!!!!!
path_to_data_files  = "/home/elizabeth/ClusterSims/no-pd-pre-computed-data/Rome/"
VANET_dict_filename =  "No_Pandas_VANET_rome_august_edit_combined_7_days.pickle"
Position_dict_filename = "No_Pandas_august_edit_taxi_positions_combined_7_days.pickle"
sources_location_file_path = "/home/elizabeth/ClusterSims/sink-locations/"
sources_locations_filename = 'Rome_random_500_sinks_locations.pickle'
results_data_file_path = '/home/elizabeth/ClusterSims/all-1/all-1-results/'
results_filename = ('actually_non_continous_broadcasting_%i_sink_id_7_days_all_to_1_ROME_results.pickle' % sink_id)
"""


#Eclipse1 Set-Up:
sink_id = results.sink_id
city = 'Rome' #SF
num_days = 14
simulation_time_limit_hours = 4 #limit at which simulation stops

path_to_data_files  = ("/export/home/pietro.carnelli/ClusterSetUp-All-to-One/%s-Data/" % city)
sources_location_file_path = ("/export/home/pietro.carnelli/ClusterSetUp-All-to-One/%s-Data/" % city)

VANET_dict_filename =  "split_dict_15_till_24_hrs_No_Pandas_VANET_august_edit_combined_14_days_rome.pickle"
Position_dict_filename = "split_dict_15_till_24_hrs_No_Pandas_august_edit_taxi_positions_combined_14_days_rome.pickle"

sources_locations_filename = ("%s_random_500_sinks_locations.pickle" % city)

results_data_file_path = ("/export/home/pietro.carnelli/ClusterSetUp-All-to-One/%s-Results/%i_days/" % (city, num_days))
results_filename = ('%i_sink_id_all_to_1_%s_%i_days_results.pickle' % (sink_id, city, num_days))



### start of simulation... loading VANET/taxi data.
VANET_dict = pickle.load( open( (path_to_data_files + VANET_dict_filename), 'rb'))
Position_dict = pickle.load( open( (path_to_data_files + Position_dict_filename), 'rb'))
import_sources_locations = pickle.load( open( (sources_location_file_path + sources_locations_filename), 'rb'))

#initial set-up
max_num_taxis = 1500
source_range = 100 #metres
num_sources = 499
timestamps = np.array(sorted(list(Position_dict.keys())))

sink_location = import_sources_locations[sink_id]
#remove single sink from sources array
sources_locations = np.delete(import_sources_locations, (sink_id), axis=0)

sink_longitude_array = np.ones((max_num_taxis,))*sink_location[0]
sink_latitude_array = np.ones_like(sink_longitude_array)*sink_location[1]
unvisited_sources = sources_locations
num_sources = len(sources_locations)

# initialise source array for easier computation later on
source_longitude_array, source_latitude_array = SourceArrayBuilder(unvisited_sources[:,0],unvisited_sources[:,1],max_num_taxis)

# initialise within loop lists/dicts
taxis_with_messages_dict = {}
num_delivered_messages = [0]
num_active_taxis = []
num_unvisited_sources = []
num_taxis_in_vanet = []
visited_sources_dict = {}
sink_delivered_messages_set = set()
sink_received_messages_dict = {}

yet_to_be_visited_sources_index_set = set(list(range(0,num_sources)))

min_start_time_seconds = (17*60**2) #(15*60**2) #i.e. 15.00
max_start_time_seconds = (18*60**2) #(16*60**2) #i.e. 16.00


T_start = int(np.around([(np.random.randint(low=min_start_time_seconds,high=max_start_time_seconds,size=1)[0])], decimals=-1))
T_start_time = divmod(T_start,60*60)
T_start_hours = T_start_time[0]
T_start_mins = T_start_time[1]/60
T_end_time_seconds = T_start + simulation_time_limit_hours*60*60

print('all-to-1 sim, %s, sink ID = %i' % (city,sink_id))
print('VANET pre-computed data file: %s' % VANET_dict_filename)
print('starting time, %i: %i' % (T_start_hours,T_start_mins))

T = T_start

continuous_source_broadcast_on = False #True


while num_delivered_messages[-1]<num_sources and T<T_end_time_seconds:


    if continuous_source_broadcast_on is True:

        num_unvisited_sources.append(num_sources) #len(unvisited_sources))
        unvisited_sources_index_list = list(range(0,num_sources))#num_unvisited_sources[-1]))
    else:
        num_unvisited_sources.append(len(yet_to_be_visited_sources_index_set))
        unvisited_sources_index_list = list(yet_to_be_visited_sources_index_set)


    active_taxis_ids_list = Position_dict[T]['taxi_id']

    num_active_taxis.append(len(active_taxis_ids_list))

    sources_taxis_dist_mat = np.zeros((num_active_taxis[-1],num_unvisited_sources[-1]))

    active_taxis_longitude, active_taxis_latitude = zip(*Position_dict[T]['longlat'])


    for i in range(len(unvisited_sources_index_list)):

    #Evaluate linear distance between all active taxis and all sources
        sources_taxis_dist_mat[:,i] = haversine_np(active_taxis_longitude, active_taxis_latitude, source_longitude_array[0:num_active_taxis[-1],unvisited_sources_index_list[i]], source_latitude_array[0:num_active_taxis[-1],unvisited_sources_index_list[i]])

    #Update taxis dict accordingly
    taxis_within_sources_range_index = np.where(sources_taxis_dist_mat<source_range)[0]
    sources_within_range = np.where(sources_taxis_dist_mat<source_range)[1]

    taxis_within_sources_range_ids = [Position_dict[T]['taxi_id'][l] for l in taxis_within_sources_range_index]

    for source in sources_within_range:
        if source not in visited_sources_dict:
            visited_sources_dict[source] = T
            yet_to_be_visited_sources_index_set -={source}


    #update taxis_with_messages_dict, by adding new sources/taxi_ids to dict... 
    for j in range(0,len(taxis_within_sources_range_index)):

        taxi_id = taxis_within_sources_range_ids[j]
        if taxi_id in taxis_with_messages_dict:
            taxis_with_messages_dict[taxi_id].update([sources_within_range[j]])
        else:
            taxis_with_messages_dict[taxi_id] = set(())
            taxis_with_messages_dict[taxi_id].update([sources_within_range[j]])



    #set up VANET dict, check for intersection between active taxis and those within comms range
    vanet_dict = VANET_dict[T]

    active_taxis_set_ids = set((active_taxis_ids_list))
    vanet_taxis_set_ids  = set((vanet_dict['taxiAid']))
    vanet_taxis_set_ids.update(vanet_dict['taxiBid'])

    active_vanet_taxis_set_ids = vanet_taxis_set_ids.intersection(active_taxis_set_ids)


# iterate over all comunicating pairs of taxis, updating each set of visited sources...

    for k in range(0, len(vanet_dict['taxiAid'])):

        taxi_a_id = vanet_dict['taxiAid'][k]
        taxi_b_id = vanet_dict['taxiBid'][k]

        if (taxi_a_id in taxis_with_messages_dict) and (taxi_b_id in taxis_with_messages_dict):
            taxis_with_messages_dict[taxi_a_id].update(taxis_with_messages_dict[taxi_b_id])
            taxis_with_messages_dict[taxi_b_id].update(taxis_with_messages_dict[taxi_a_id])

        elif (taxi_a_id in taxis_with_messages_dict) and (taxi_b_id not in taxis_with_messages_dict):
            taxis_with_messages_dict[taxi_b_id] = taxis_with_messages_dict[taxi_a_id]

        elif (taxi_b_id in taxis_with_messages_dict) and (taxi_a_id not in taxis_with_messages_dict):
            taxis_with_messages_dict[taxi_a_id] = taxis_with_messages_dict[taxi_b_id]


# update sink delivery of messages, 

    sink_taxis_dist_mat = haversine_np(active_taxis_longitude, active_taxis_latitude, sink_longitude_array[0:num_active_taxis[-1]], sink_latitude_array[0:num_active_taxis[-1]])
    taxis_within_sink_range_index = np.where(sink_taxis_dist_mat<source_range)[0]

#update sink delivered messages, taxis need to a) be in range and b) have messages to deliver...
    if taxis_within_sink_range_index.size > 0:

        taxi_ids_within_sink_range_set = set()
        for taxi_index in taxis_within_sink_range_index:
            taxi_ids_within_sink_range_set.update([active_taxis_ids_list[taxi_index]])

        taxi_ids_with_messages_set = set(taxis_with_messages_dict.keys())
        taxi_ids_with_messages_and_within_sink_range_list = list(taxi_ids_with_messages_set.intersection(taxi_ids_within_sink_range_set))

        for g in taxi_ids_with_messages_and_within_sink_range_list:
            sink_delivered_messages_set.update(taxis_with_messages_dict[g])
            for source in list(taxis_with_messages_dict[g]):
                if source not in sink_received_messages_dict:
                    sink_received_messages_dict[source] = T
            
            #if sink_received_messages_dict


    num_delivered_messages.append(len(sink_delivered_messages_set))
    #num_active_taxis_plotting.append(active_taxis_ids_list.shape[0])
    
    #num_taxis_in_vanet.append(vanet_df.shape[0])
    taxis_a = set(vanet_dict['taxiAid'])
    taxis_b = set(vanet_dict['taxiBid'])

    num_taxis_in_vanet.append(len(taxis_a.union(taxis_b)))
    


    print('Time = %i' % T)
    print('sink_progress = %f' % (max(num_delivered_messages)/num_sources))

    print('Num. Active = %i' % num_active_taxis[-1])
    print('Num. in VANET = %i' % num_taxis_in_vanet[-1])

#    time_index+=1
    T+=10


transit_delay = []
total_delay = []
received_sources_list = []

for key, values in sink_received_messages_dict.items():

    transit_delay.append(values - visited_sources_dict[key])
    total_delay.append(values-T_start)

    if key >= sink_id:
        real_key = key+1
        received_sources_list.append(real_key)
    else:
        received_sources_list.append(key)

sources_zero_delay_index = np.where(np.array(transit_delay)<0.1)[0]
#sources_messages_received_index = list(sink_received_messages_dict.keys())

sources_never_received_set = set(range(0,499))
sources_never_received_set -= {sink_id}
sources_never_received_index = list(sources_never_received_set.symmetric_difference(set(received_sources_list)))


ALL_to_1_RESULTS_dict = {'transit_delay':np.array(transit_delay), 'total_delay':np.array(total_delay), 'received_sources':received_sources_list, 'non_received_sources':sources_never_received_index, 'num_delivered_messages':np.array(num_delivered_messages), 'num_active_taxis':np.array(num_active_taxis), 'num_vanet_taxis':np.array(num_taxis_in_vanet), 'sink_id':sink_id, 'sim_time':np.array(range(T_start,T_end_time_seconds,10))}

#saving results data
with open((results_data_file_path+results_filename),'wb') as handle:
    pickle.dump(ALL_to_1_RESULTS_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('num received messages = %i' % (num_delivered_messages[-1]))
print('saved results to: %s' % (results_data_file_path+results_filename))

if num_delivered_messages[-1]>0:
    print('mean total delay/seconds: %i' % (np.mean(ALL_to_1_RESULTS_dict['total_delay'])))
else:
    print('nothing was received...')

print('gentle reminder, sink id: %i, num_days: %i' % (sink_id, num_days))








"""
import matplotlib.pyplot as plt
plt.plot(ALL_to_1_RESULTS_dict['num_active_taxis'],'ok')
plt.show()

#### plotting for checking.....

figure_font = {'family':'normal','weight':'bold','size':20}

transit_delay_array = np.array(transit_delay)
total_delay_array = np.array(total_delay)
sources_longitude, sources_latitude = zip(*sources_locations)
sources_longitude_array = np.array(sources_longitude)
sources_latitude_array = np.array(sources_latitude)

sources_zero_delay_index = np.where(transit_delay_array<0.1)[0]
sources_messages_received_index = list(sink_received_messages_dict.keys())

sources_never_received_set = set(range(0,499))
sources_never_received_index = list(sources_never_received_set.symmetric_difference(set(sources_messages_received_index)))


cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(sources_longitude_array[sources_messages_received_index], sources_latitude_array[sources_messages_received_index], c=transit_delay_array, vmin=10, vmax=max(transit_delay), s=100, cmap=cm)
plt.colorbar(sc)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.plot(sink_location[0], sink_location[1],'db',markersize=15)
plt.plot(sources_longitude_array[sources_zero_delay_index], sources_latitude_array[sources_zero_delay_index], 'sb',markersize=10)
plt.plot(sources_longitude_array[sources_never_received_index], sources_latitude_array[sources_never_received_index], 'vk', markersize=10)
plt.title('ROME All-to-One, VANET Transit Delay, Sink ID: %i ' % (sink_id))
plt.show()












#extra material:

plt.plot(received_sources_list, transit_delay, 'ok')
plt.xlabel('Source ID/Number')
plt.ylabel('Transit Delay/[s]')
plt.title('All-to-1, sink_id = %i' % sink_id)
plt.show()

plot_time_hours = timestamps/(60**2)
num_delivered_messages_array = np.array(num_delivered_messages[0:86400])
plt.plot(plot_time_hours,num_delivered_messages_array/num_sources,'-ob')
plt.xlabel('Time/[Hrs]')
plt.ylabel('Normalised, num. received messages from sources')
plt.title('Overall message delivery, All-to-1, sink_id=%i' % sink_id)
plt.show()




plot_sink_id = 1#125 #8#488
non_visited_sinks = np.where(sink_transit_delay_matrix[:,plot_sink_id]<0)[0]
visited_sinks = np.where(sink_transit_delay_matrix[:,plot_sink_id]>0)[0]
#non_visited_sinks.astype(int)


plt.scatter(all_sinks_longitudes[visited_sinks],all_sinks_latitudes[visited_sinks],c=sink_transit_delay_matrix[visited_sinks,plot_sink_id ],s=100, cmap=plt.cm.Set1) #Oranges)
cb = plt.colorbar()
cb.set_label('Transit Delay/[Seconds]')

plt.plot(all_sinks_longitudes[non_visited_sinks],all_sinks_latitudes[non_visited_sinks],'ok')
plt.plot(all_sinks_longitudes[plot_sink_id],all_sinks_latitudes[plot_sink_id],'db',markersize=13)
plt.title('SF Taxi Network Transit Delay, Source ID: %i ' % (plot_sink_id))
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.show()


moar crap
plt.scatter(sources_longitude, sources_latitude, c=transit_delay_array, s=100, cmap=plt.cm.Set1)
cb = plt.colorbar()
cb.set_label('Transit Delay/[Seconds]')
plt.plot(sink_location[0], sink_location[1],'db',markersize=13)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('ROME All-to-One, VANET Transit Delay, Sink ID: %i ' % (sink_id))


Ataxi = set((1,2,3,5,7,11))
Btaxi = set((0,1,2,4,6,8))

In [8]: Ataxi.intersection(Btaxi)
Out[8]: {1, 2}

In [9]: Ataxi.union(Btaxi)
Out[9]: {0, 1, 2, 3, 4, 5, 6, 7, 8, 11}

In [46]: Ataxi.symmetric_difference(Btaxi)
Out[46]: {0, 3, 4, 5, 6, 7, 8, 11}


"""


