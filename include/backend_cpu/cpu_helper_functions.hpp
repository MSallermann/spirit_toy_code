#pragma once
#ifndef CPU_HELPER_FUNCTIONS

inline void tupel_from_idx( int & idx, int * tupel, int * maxVal, int n )
{
    int idx_diff = idx;
    int div      = 1;
    for( int i = 0; i < n - 1; i++ )
        div *= maxVal[i];
    for( int i = n - 1; i > 0; i-- )
    {
        tupel[i] = idx_diff / div;
        idx_diff -= tupel[i] * div;
        div /= maxVal[i - 1];
    }
    tupel[0] = idx_diff / div;
}

#endif