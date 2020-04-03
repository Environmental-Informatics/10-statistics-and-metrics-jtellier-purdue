#!/bin/env python
# Created on March 25, 2020
#  by Keith Cherkauer
#
# This script servesa as the solution set for assignment-10 on descriptive
# statistics and environmental informatics.  See the assignment documention 
# and repository at:
# https://github.com/Environmental-Informatics/assignment-10.git for more
# details about the assignment.
#
""" Script modified from the template script by Joshua Tellier of Purdue University
on 4/3/2020 to fulfill the requirements for the lab 10 github assignment."""
import pandas as pd
import scipy.stats as stats
import numpy as np

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    global DataDF
    global MissingValues
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    
    # quantify the number of missing values
    for i in range(0,len(DataDF)-1): #checks for any values below zero, then replaces it with nan if outside the range
        if 0 > DataDF['Discharge'].iloc[i]:
            DataDF['Discharge'].iloc[i]=np.nan
    
    MissingValues = DataDF["Discharge"].isna().sum()
    
    return( DataDF, MissingValues )

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
    DataDF = DataDF.loc[startDate:endDate] #indexing for a specific date range (we use .loc because the dates are already the index)
    MissingValues = DataDF["Discharge"].isna().sum() #recalculating the number of missing values after trimming the dataset
    return( DataDF, MissingValues )

def CalcTqmean(Qvalues):
    """This function computes the Tqmean of a series of data, typically
       a 1 year time series of streamflow, after filtering out NoData
       values.  Tqmean is the fraction of time that daily streamflow
       exceeds mean streamflow for each year. Tqmean is based on the
       duration rather than the volume of streamflow. The routine returns
       the Tqmean value for the given data array."""
    Tqmean = ((Qvalues > Qvalues.mean()).sum()/len(Qvalues))
    return ( Tqmean )

def CalcRBindex(Qvalues):
    """This function computes the Richards-Baker Flashiness Index
       (R-B Index) of an array of values, typically a 1 year time
       series of streamflow, after filtering out the NoData values.
       The index is calculated by dividing the sum of the absolute
       values of day-to-day changes in daily discharge volumes
       (pathlength) by total discharge volumes for each year. The
       routine returns the RBindex value for the given data array."""
    a=0
    Qvalues = Qvalues.dropna()
    if sum(Qvalues) > 0:
        for i in range(1,len(Qvalues)):
            a=a+abs(Qvalues[i-1]-Qvalues[i])
        RBindex=a/sum(Qvalues)
    else:
        RBindex = np.nan
    return ( RBindex )

def Calc7Q(Qvalues):
    """This function computes the seven day low flow of an array of 
       values, typically a 1 year time series of streamflow, after 
       filtering out the NoData values. The index is calculated by 
       computing a 7-day moving average for the annual dataset, and 
       picking the lowest average flow in any 7-day period during
       that year.  The routine returns the 7Q (7-day low flow) value
       for the given data array."""
    Qvalues = Qvalues.dropna()
    val7Q=min(Qvalues.resample('7D').mean())
    return ( val7Q )

def CalcExceed3TimesMedian(Qvalues):
    """This function computes the number of days with flows greater 
       than 3 times the annual median flow. The index is calculated by 
       computing the median flow from the given dataset (or using the value
       provided) and then counting the number of days with flow greater than 
       3 times that value.   The routine returns the count of events greater 
       than 3 times the median annual flow value for the given data array."""
    Qvalues = Qvalues.dropna()
    median3x = (Qvalues > (Qvalues.median()*3)).sum()
    return ( median3x )

def GetAnnualStatistics(DataDF):
    """This function calculates annual descriptive statistcs and metrics for 
    the given streamflow time series.  Values are retuned as a dataframe of
    annual values for each water year.  Water year, as defined by the USGS,
    starts on October 1."""
    colnames = ['site_no','Mean Flow', 'Peak Flow','Median Flow','Coeff Var', 'Skew','Tqmean','R-B Index','7Q','3xMedian']
    annualdata=DataDF.resample('AS-OCT').mean()
    WYDataDF = pd.DataFrame(0, index=annualdata.index,columns=colnames)
    WYDataDF['site_no']=DataDF.resample('AS-OCT')['site_no'].mean()
    WYDataDF['Mean Flow']=DataDF.resample('AS-OCT')['Discharge'].mean()
    WYDataDF['Peak Flow']=DataDF.resample('AS-OCT')['Discharge'].max()
    WYDataDF['Median Flow']=DataDF.resample('AS-OCT')['Discharge'].median()
    WYDataDF['Coeff Var']=(DataDF.resample('AS-OCT')['Discharge'].std()/DataDF.resample('AS-OCT')['Discharge'].mean())*100
    WYDataDF['Skew']=DataDF.resample('AS-OCT').apply({'Discharge':lambda x: stats.skew(x,nan_policy='omit',bias=False)},raw=True)
    WYDataDF['Tqmean']=DataDF.resample('AS-OCT').apply({'Discharge':lambda x: CalcTqmean(x)})
    WYDataDF['R-B Index']=DataDF.resample('AS-OCT').apply({'Discharge':lambda x: CalcRBindex(x)})
    WYDataDF['7Q']=DataDF.resample('AS-OCT').apply({'Discharge':lambda x: Calc7Q(x)})
    WYDataDF['3xMedian']=DataDF.resample('AS-OCT').apply({'Discharge':lambda x: CalcExceed3TimesMedian(x)})
    return ( WYDataDF )

def GetMonthlyStatistics(DataDF):
    """This function calculates monthly descriptive statistics and metrics 
    for the given streamflow time series.  Values are returned as a dataframe
    of monthly values for each year."""
    colnames = ['site_no','Mean Flow','Coeff Var','Tqmean','R-B Index']
    monthdata=DataDF.resample('M').mean()
    MoDataDF = pd.DataFrame(0, index=monthdata.index,columns=colnames)
    MoDataDF['site_no']=DataDF.resample('M')['site_no'].mean()
    MoDataDF['Mean Flow']=DataDF.resample('M')['Discharge'].mean()
    MoDataDF['Coeff Var']=(DataDF.resample('M')['Discharge'].std()/DataDF.resample('M')['Discharge'].mean())*100
    MoDataDF['Tqmean']=DataDF.resample('M').apply({'Discharge':lambda x: CalcTqmean(x)})
    MoDataDF['R-B Index']=DataDF.resample('M').apply({'Discharge':lambda x: CalcRBindex(x)})
    return ( MoDataDF )

def GetAnnualAverages(WYDataDF):
    """This function calculates annual average values for all statistics and
    metrics.  The routine returns an array of mean values for each metric
    in the original dataframe."""
    AnnualAverages=WYDataDF.mean(axis=0)
    return( AnnualAverages )

def GetMonthlyAverages(MoDataDF):
    """This function calculates annual average monthly values for all 
    statistics and metrics.  The routine returns an array of mean values 
    for each metric in the original dataframe."""
    colnames = ['site_no','Mean Flow','Coeff Variation','TQmean','R-B Index']
    MonthlyAverages = pd.DataFrame(0, index=[1,2,3,4,5,6,7,8,9,10,11,12],columns=colnames)
    MonthlyAverages.iloc[0,0]=MoDataDF['site_no'][::12].mean()
    MonthlyAverages.iloc[1,0]=MoDataDF['site_no'][::12].mean()
    MonthlyAverages.iloc[2,0]=MoDataDF['site_no'][::12].mean()
    MonthlyAverages.iloc[3,0]=MoDataDF['site_no'][::12].mean()
    MonthlyAverages.iloc[4,0]=MoDataDF['site_no'][::12].mean()
    MonthlyAverages.iloc[5,0]=MoDataDF['site_no'][::12].mean()
    MonthlyAverages.iloc[6,0]=MoDataDF['site_no'][::12].mean()
    MonthlyAverages.iloc[7,0]=MoDataDF['site_no'][::12].mean()
    MonthlyAverages.iloc[8,0]=MoDataDF['site_no'][::12].mean()
    MonthlyAverages.iloc[9,0]=MoDataDF['site_no'][::12].mean()
    MonthlyAverages.iloc[10,0]=MoDataDF['site_no'][::12].mean()
    MonthlyAverages.iloc[11,0]=MoDataDF['site_no'][::12].mean()
    
    MonthlyAverages.iloc[0,1]=MoDataDF['Mean Flow'][3::12].mean()
    MonthlyAverages.iloc[1,1]=MoDataDF['Mean Flow'][4::12].mean()
    MonthlyAverages.iloc[2,1]=MoDataDF['Mean Flow'][5::12].mean()
    MonthlyAverages.iloc[3,1]=MoDataDF['Mean Flow'][6::12].mean()
    MonthlyAverages.iloc[4,1]=MoDataDF['Mean Flow'][7::12].mean()
    MonthlyAverages.iloc[5,1]=MoDataDF['Mean Flow'][8::12].mean()
    MonthlyAverages.iloc[6,1]=MoDataDF['Mean Flow'][9::12].mean()
    MonthlyAverages.iloc[7,1]=MoDataDF['Mean Flow'][10::12].mean()
    MonthlyAverages.iloc[8,1]=MoDataDF['Mean Flow'][11::12].mean()
    MonthlyAverages.iloc[9,1]=MoDataDF['Mean Flow'][::12].mean()
    MonthlyAverages.iloc[10,1]=MoDataDF['Mean Flow'][1::12].mean()
    MonthlyAverages.iloc[11,1]=MoDataDF['Mean Flow'][2::12].mean()
    
    MonthlyAverages.iloc[0,2]=MoDataDF['Coeff Var'][3::12].mean()
    MonthlyAverages.iloc[1,2]=MoDataDF['Coeff Var'][4::12].mean()
    MonthlyAverages.iloc[2,2]=MoDataDF['Coeff Var'][5::12].mean()
    MonthlyAverages.iloc[3,2]=MoDataDF['Coeff Var'][6::12].mean()
    MonthlyAverages.iloc[4,2]=MoDataDF['Coeff Var'][7::12].mean()
    MonthlyAverages.iloc[5,2]=MoDataDF['Coeff Var'][8::12].mean()
    MonthlyAverages.iloc[6,2]=MoDataDF['Coeff Var'][9::12].mean()
    MonthlyAverages.iloc[7,2]=MoDataDF['Coeff Var'][10::12].mean()
    MonthlyAverages.iloc[8,2]=MoDataDF['Coeff Var'][11::12].mean()
    MonthlyAverages.iloc[9,2]=MoDataDF['Coeff Var'][::12].mean()
    MonthlyAverages.iloc[10,2]=MoDataDF['Coeff Var'][1::12].mean()
    MonthlyAverages.iloc[11,2]=MoDataDF['Coeff Var'][2::12].mean()
    
    MonthlyAverages.iloc[0,3]=MoDataDF['Tqmean'][3::12].mean()
    MonthlyAverages.iloc[1,3]=MoDataDF['Tqmean'][4::12].mean()
    MonthlyAverages.iloc[2,3]=MoDataDF['Tqmean'][5::12].mean()
    MonthlyAverages.iloc[3,3]=MoDataDF['Tqmean'][6::12].mean()
    MonthlyAverages.iloc[4,3]=MoDataDF['Tqmean'][7::12].mean()
    MonthlyAverages.iloc[5,3]=MoDataDF['Tqmean'][8::12].mean()
    MonthlyAverages.iloc[6,3]=MoDataDF['Tqmean'][9::12].mean()
    MonthlyAverages.iloc[7,3]=MoDataDF['Tqmean'][10::12].mean()
    MonthlyAverages.iloc[8,3]=MoDataDF['Tqmean'][11::12].mean()
    MonthlyAverages.iloc[9,3]=MoDataDF['Tqmean'][::12].mean()
    MonthlyAverages.iloc[10,3]=MoDataDF['Tqmean'][1::12].mean()
    MonthlyAverages.iloc[11,3]=MoDataDF['Tqmean'][2::12].mean()
    
    MonthlyAverages.iloc[0,4]=MoDataDF['R-B Index'][3::12].mean()
    MonthlyAverages.iloc[1,4]=MoDataDF['R-B Index'][4::12].mean()
    MonthlyAverages.iloc[2,4]=MoDataDF['R-B Index'][5::12].mean()
    MonthlyAverages.iloc[3,4]=MoDataDF['R-B Index'][6::12].mean()
    MonthlyAverages.iloc[4,4]=MoDataDF['R-B Index'][7::12].mean()
    MonthlyAverages.iloc[5,4]=MoDataDF['R-B Index'][8::12].mean()
    MonthlyAverages.iloc[6,4]=MoDataDF['R-B Index'][9::12].mean()
    MonthlyAverages.iloc[7,4]=MoDataDF['R-B Index'][10::12].mean()
    MonthlyAverages.iloc[8,4]=MoDataDF['R-B Index'][11::12].mean()
    MonthlyAverages.iloc[9,4]=MoDataDF['R-B Index'][::12].mean()
    MonthlyAverages.iloc[10,4]=MoDataDF['R-B Index'][1::12].mean()
    MonthlyAverages.iloc[11,4]=MoDataDF['R-B Index'][2::12].mean()
    
    return( MonthlyAverages )

# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define filenames as a dictionary
    # NOTE - you could include more than jsut the filename in a dictionary, 
    #  such as full name of the river or gaging site, units, etc. that would
    #  be used later in the program, like when plotting the data.
    fileName = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
                 "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    
    # define blank dictionaries (these will use the same keys as fileName)
    DataDF = {}
    MissingValues = {}
    WYDataDF = {}
    MoDataDF = {}
    AnnualAverages = {}
    MonthlyAverages = {}
    
    # process input datasets
    for file in fileName.keys():
        
        print( "\n", "="*50, "\n  Working on {} \n".format(file), "="*50, "\n" )
        
        DataDF[file], MissingValues[file] = ReadData(fileName[file])
        print( "-"*50, "\n\nRaw data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # clip to consistent period
        DataDF[file], MissingValues[file] = ClipData( DataDF[file], '1969-10-01', '2019-09-30' )
        print( "-"*50, "\n\nSelected period data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # calculate descriptive statistics for each water year
        WYDataDF[file] = GetAnnualStatistics(DataDF[file])
        
        # calcualte the annual average for each stistic or metric
        AnnualAverages[file] = GetAnnualAverages(WYDataDF[file])
        
        print("-"*50, "\n\nSummary of water year metrics...\n\n", WYDataDF[file].describe(), "\n\nAnnual water year averages...\n\n", AnnualAverages[file])

        # calculate descriptive statistics for each month
        MoDataDF[file] = GetMonthlyStatistics(DataDF[file])

        # calculate the annual averages for each statistics on a monthly basis
        MonthlyAverages[file] = GetMonthlyAverages(MoDataDF[file])
        
        print("-"*50, "\n\nSummary of monthly metrics...\n\n", MoDataDF[file].describe(), "\n\nAnnual Monthly Averages...\n\n", MonthlyAverages[file])

#Testing each function for each dataset      
ReadData("WildcatCreek_Discharge_03335000_19540601-20200315.txt")
DataDF, MissingValues = ClipData(DataDF,'1969-10-01','2019-09-30')
Wild_WYDataDF = GetAnnualStatistics(DataDF)
Wild_MoDataDF = GetMonthlyStatistics(DataDF)
Wild_AnnualAverages = GetAnnualAverages(Wild_WYDataDF)
Wild_MonthlyAverages = GetMonthlyAverages(Wild_MoDataDF)

ReadData("TippecanoeRiver_Discharge_03331500_19431001-20200315.txt") 
DataDF, MissingValues = ClipData(DataDF,'1969-10-01','2019-09-30')
Tippe_WYDataDF = GetAnnualStatistics(DataDF)
Tippe_MoDataDF = GetMonthlyStatistics(DataDF)
Tippe_AnnualAverages = GetAnnualAverages(Tippe_WYDataDF)
Tippe_MonthlyAverages = GetMonthlyAverages(Tippe_MoDataDF)

AnnualStats = Wild_WYDataDF
AnnualStats = AnnualStats.append(Tippe_WYDataDF)
AnnualStats.to_csv('Annual_Metrics.csv', sep=',', index=True) #writing the corrected data to a csv

MonStats = Wild_MoDataDF
MonStats = MonStats.append(Tippe_MoDataDF)
MonStats.to_csv('Monthly_Metrics.csv', sep=',', index=True) #writing the corrected data to a csv

AnnMet = Wild_AnnualAverages
AnnMet = AnnMet.append(Tippe_AnnualAverages)
AnnMet.to_csv('Average_Annual_Metrics.txt', sep='\t', index=True) #writing the corrected data to a tab delimited file

MonMet = Wild_MonthlyAverages
MonMet = MonMet.append(Tippe_MonthlyAverages)
MonMet.to_csv('Average_Monthly_Metrics.txt', sep='\t', index=True) #writing the corrected data to a tab delimited file