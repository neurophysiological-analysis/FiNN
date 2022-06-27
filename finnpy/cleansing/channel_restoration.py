'''
Created on Jun 2, 2020

This module implements functions to restore channels as an approach to deal with missing and/or bad channels.

:author: voodoocode
'''

import numpy as np
import finnpy.cleansing.neighboring_channels as nc

def _get_neighbor_channel_ids(ch_names):
    """
    
    Determines the neighboring channels of a specific channel
    
    :param ch_names: Channel list whose neighbors are to be identified.
    
    :return: List of neighboring channels (between two and four per channel of interest).
    
    """
    neigh_list = nc.neighbor_channels
    
    neigh_idx_list = dict()
    for (ch_name_idx, ch_name) in enumerate(ch_names):
        neigh_idx_list[ch_name_idx] = list()
        for neighbor in neigh_list[ch_name]:
            if (neighbor in ch_names):
                neigh_idx_list[ch_name_idx].append(ch_names.index(neighbor))
    return neigh_idx_list

def run(data, ch_names, bad_ch_idx_list):
    """
    Restores channel by averaging signals from their respective neighbors. In case neighboring channels are flagged as bad, channels get iteratively restored, from the channel with the most valid neighbors to the channel with the least. Restored channels are considered valid candidates for channel reconstruction.
    
    :param data: Input data in the format channels x samples.
    :param ch_names: Names of the channles from the input data. Order needs to be aligned with the channel order of ch_names.
    :param bad_ch_idx_list: List of bad channels with are selected for substitution by their neighbors.
    
    :return: Restored channels in the format channels x samples.
    
    """
    
    
    sub_data = np.copy(data)

    neigh_list = _get_neighbor_channel_ids(ch_names)

    while(len(bad_ch_idx_list) > 0):
        # Update neighbor count per bad channel
        # Has to be updated each iteration, as a channel is restored during each iteration
        available_neigh_cnt = np.zeros((len(bad_ch_idx_list)))
        for (bad_Id, bad_Ch_Idx) in enumerate(bad_ch_idx_list):
            available_neigh_cnt[bad_Id] = len(neigh_list[bad_Ch_Idx])
            
            for neighChIdx in neigh_list[bad_Ch_Idx]:
                if (neighChIdx in bad_ch_idx_list):
                    available_neigh_cnt[bad_Id] -= 1

        sorted_bad_Ch_Idx = (np.argsort(available_neigh_cnt)[::-1]).tolist()

        # Get neighboring channel for channel with most non-faulty neighbors
        bad_Idx = sorted_bad_Ch_Idx[0]
        loc_data = np.zeros(data[bad_ch_idx_list[bad_Idx]].shape)
        neighCnt = 0
        for neigh in neigh_list[bad_ch_idx_list[bad_Idx]]:
            # In case the channel has valid data, use it for reconstruction   
            if (neigh not in bad_ch_idx_list):
                loc_data += sub_data[neigh]
                neighCnt += 1

        # In case there are only faulty neighbors, raise an exception
        if (neighCnt == 0):
            raise Exception("Error, cannot restore channels due to too many missing channels")
        else:
            loc_data /= neighCnt
            sub_data[bad_ch_idx_list[bad_Idx]] = loc_data

            # After a channel is 'restored' it is not bad anymore
            del bad_ch_idx_list[bad_Idx]

    return sub_data





