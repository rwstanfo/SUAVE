## @ingroup Methods-Power-Battery-Discharge
# chin_discharge.py
# 
# Created:  Oct 2019, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
from SUAVE.Core import Data , Units 
import numpy as np
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline
from scipy.integrate import odeint

def chin_discharge(battery,numerics): 
    """Thevenin Model
    
     Assumptions:  
     
       Source: 
       
       Inputs:
       battery.
        resistance                      [Ohms]
        max_energy                      [Joules]
        current_energy (to be modified) [Joules]
        inputs.
            current                     [amps]
            power_in                    [Watts]
       
       Outputs:
       battery.
        current energy                  [Joules]
        resistive_losses                [Watts]
        voltage_open_circuit            [Volts]
        voltage_under_load              [Volts]
        
    """
 
    I_bat             = battery.inputs.current
    P_bat             = battery.inputs.power_in
    V_ul              = battery.inputs.V_ul
    R_bat             = battery.resistance
    T_bat             = battery.temperature 
    T_cell            = battery.cell_temperature
    time              = numerics.time.control_points
    I                 = numerics.time.integrate
    D                 = numerics.time.differentiate    
    max_energy        = battery.max_energy
    cell_mass         = battery.mass_properties.mass                
    Cp                = battery.cell.specific_heat_capacity    
    h                 = battery.heat_transfer_coefficient   
    t                 = battery.time_in_days
    T_ambient         = battery.ambient_temperature     
    cell_surface_area = battery.cell.surface_area
    current_energy    = battery.current_energy 
    max_energy        = battery.max_energy
    T_init            = battery.current_temperature
    Q                 = battery.charge_throughput
    T_cell            = battery.cell_temperature 
    battery_data      = battery_performance_maps()     
    
    # calculate the current going into one cell 
    n_series   = battery.module_config[0]  
    n_parallel = battery.module_config[1]
    n_total    = n_series * n_parallel 
    I_cell      = I_bat/n_parallel
    
    #state of charge of the battery
    initial_discharge_state = np.dot(I,P_bat) + current_energy[0]
    SOC =  np.divide(initial_discharge_state,max_energy)
    SOC[SOC < 0.] = 0.    
    DOD = 1 - SOC 
    
    # aging model 
    rms_V_ul = np.sqrt(np.mean(V_ul**2))
    
    alpha_cap = ((7.542*V_ul - 23.75)*1E6) * np.exp(-6976/T_cell)
    alpha_res = ((5.270*V_ul - 16.32)*1E5) * np.exp(-5986/T_cell)
    beta_cap  = 7.348E-3 * (rms_V_ul - 3.667)**2 +  7.60E-4 + 4.081E-3*DOD
    beta_res  = 2.153E-4 * (rms_V_ul - 3.725)**2 - 1.521E-5 + 2.798E-4*DOD
    
    # aging model 
    max_energy = max_energy*(1 - alpha_cap*(t**0.75) - beta_cap*np.sqrt(Q))  
    
    # look up tables 
    V_oc = np.zeros_like(I_cell)
    R_Th = np.zeros_like(I_cell)  
    C_Th = np.zeros_like(I_cell)  
    R_0  = np.zeros_like(I_cell) 
    for i in range(len(SOC)): 
        V_oc[i] = battery_data.V_oc_interp(T_cell[i], SOC[i])[0]
        C_Th[i] = battery_data.C_Th_interp(T_cell[i], SOC[i])[0]
        R_Th[i] = battery_data.R_Th_interp(T_cell[i], SOC[i])[0]
        R_0[i]  =  battery_data.R_0_interp(T_cell[i], SOC[i])[0] 
     
    V_Th = I_cell/(1/R_Th + C_Th*np.dot(D,np.ones_like(R_Th)))
    
    # resistive grown
    R_0  = R_0 *(1 + alpha_res*(t**0.75) + beta_res*Q)
    
    # Calculate resistive losses
    P_heat = (I_cell**2)*(R_0 + R_Th)
    
    #Implement model for heat 
    P_net  = P_heat - h*cell_surface_area*(T_cell - T_ambient)
    dT_dt  = P_net/(cell_mass*Cp)
    current_temperature  = T_init[0] + np.dot(I,dT_dt)
    
    # Power going into the battery accounting for resistance losses
    P_loss = n_total*P_heat
    P = P_bat - np.abs(P_loss) 
   
    ebat = np.dot(I,P)
    
    # Add this to the current state
    if np.isnan(ebat).any():
        ebat=np.ones_like(ebat)*np.max(ebat)
        if np.isnan(ebat.any()): #all nans; handle this instance
            ebat=np.zeros_like(ebat)
            
    current_energy = ebat + current_energy[0]
    
    new_SOC = np.divide(current_energy, max_energy)
    new_SOC[new_SOC<0] = 0. 
    
    # Voltage under load:
    voltage_under_load   = (V_oc - V_Th - (I_cell * R_0))
    
    # determine new charge throughput 
    Joules_to_Wth = 0.000277778
    Q  = Q +  ebat*Joules_to_Wth/voltage_under_load
    
    # if SOC is negative, voltage under load goes to zero 
    voltage_under_load[new_SOC == 0.] = 0.
    V_oc[new_SOC == 0.]   = 0. 
    P_heat[new_SOC == 0.] = 0.
    V_Th[new_SOC == 0.]   = 0.
    
    # Pack outputs
    battery.current_energy           = current_energy
    battery.current_temperature      = current_temperature
    battery.resistive_losses         = P_loss
    battery.load_power               = voltage_under_load*n_series*I_bat
    battery.current                  = I_bat
    battery.voltage_open_circuit     = V_oc*n_series
    battery.battery_thevenin_voltage = V_Th*n_series
    battery.charge_throughput        = Q 
    battery.battery_temperature      = T_bat 
    battery.voltage_under_load       = voltage_under_load*n_series
    battery.state_of_charge          = new_SOC
    
    return battery
    
def battery_performance_maps():
    battery_data = Data()
    T_bp = np.array([0., 20., 30., 45.])
    SOC_bp = np.array( [0. , 0.03333333, 0.06666667, 0.1 , 0.13333333, 0.16666667,
           0.2 , 0.23333333, 0.26666667, 0.3 , 0.33333333, 0.36666667,
           0.4 , 0.43333333, 0.46666667, 0.5 , 0.53333333, 0.56666667,
           0.6 , 0.63333333, 0.66666667, 0.7 , 0.73333333, 0.76666667,
           0.8 , 0.83333333, 0.86666667, 0.9 , 0.93333333, 0.96666667,
           1. ] )
     
    tV_oc = np.array([ [2.92334783,3.00653623,3.08972464,3.17291304,3.23989855,3.31010145, 3.3803913 ,
           3.44033333,3.49033333,3.52169565,3.54391304,3.58695652, 3.62095652,3.65437681,
           3.68604348,3.72430435,3.75531884,3.79102899, 3.82030435,3.84181159,3.86124638,
           3.88921739,3.91686957,3.96223188, 4.00169565,4.04117391,4.06849275,4.07573913,
           4.08571014,4.10571014, 4.161 ] , 
          [2.99293893,3.05400763,3.11507634,3.17614504, 3.23506616,3.30371247, 3.37521374,
           3.43605852,3.48697455,3.5200229 ,3.54251908,3.58374046, 3.6329313 ,3.67379644,
           3.70287532,3.73784733,3.76526463,3.79174809, 3.81922901,3.84108142,3.87212214,
           3.90738931,3.93615267,3.98113995, 4.02093893,4.04504071,4.07114758,4.07583969,
           4.08371501,4.10560814, 4.161 ] , 
          [2.84084639,2.98428484,3.1050295 ,3.19464496,3.25566531,3.309059 , 3.37185148,
           3.43473652,3.49059613,3.51955239,3.541353 ,3.58558494, 3.62641607,3.6708881 ,
           3.70814547,3.7392177 ,3.76822075,3.79592981, 3.82260427,3.84986368,3.88146592,
           3.91739674,3.94798779,3.98188403, 4.02274568,4.05623296,4.06830824,4.07468871,
           4.08175788,4.10853306, 4.153 ] ,
          [2.81925101,2.97410931,3.09861134,3.18674899,3.24142105,3.29678138, 3.35963563,
           3.42195951,3.47637247,3.51383806,3.54319838,3.59076923, 3.61940891,3.65574089,
           3.7067004 ,3.74153441,3.77023887,3.79773684, 3.82421053,3.85139271,3.88311336,
           3.91906478,3.94918219,3.98310931, 4.02401215,4.05611741,4.07036842,4.07774494,
           4.08190283,4.10867206, 4.153 ] ])
    
    tC_Th = np.array([ [2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000., 2000.,
           2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000., 2000.,2000.,
           2000.,2000.,2000.,2000.,2000.] ,
          [2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000., 2000.,
           2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000., 2000.,2000.,
           2000.,2000.,2000.,2000.,2000.] ,
          [2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000., 2000.,
           2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000., 2000.,2000.,
           2000.,2000.,2000.,2000.,2000.] , 
          [2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000., 2000.,
           2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000.,2000., 2000.,2000.,
           2000.,2000.,2000.,2000.,2000.] ])
    tR_Th = np.array([ [0.09 ,0.09 ,0.09 ,0.09 ,0.07130435 ,0.06 , 0.06 ,0.06 ,0.06 ,0.06 ,0.06 ,0.06 ,
           0.08217391, 0.07492754, 0.07 ,0.07 ,0.07 ,0.07 , 0.07 ,0.05318841,0.04144928,
           0.0573913 ,0.06884058,0.07, 0.07456522,0.075,0.075,0.05586957,0.055 ,0.04021739, 0.04 ] ,
          [0.08534351,0.07516539,0.06498728,0.05480916,0.04838931,0.04589059, 0.045 ,0.045 ,0.04195929,
           0.03937405,0.03642494,0.035 , 0.035 ,0.03848601,0.05430025,0.04534351,0.03624682,0.03115776,
           0.03, 0.03 ,0.03 ,0.03839695,0.04 ,0.04 , 0.04 ,0.03089059,0.03,0.03 ,0.02807125, 0.02505344 , 0.02],
          [0.0677823 ,0.05252289,0.045 ,0.045,0.045,0.045 , 0.04207528,0.04 ,0.03690234,0.035 ,0.0317294 ,
           0.02798576,0.027 ,0.025588 ,0.025 ,0.02129705,0.02   ,0.02 , 0.04377416,0.04190234,0.04,0.04 ,
           0.04 ,0.03121058,0.02820753,0.028,0.02055341,0.02 ,0.02,0.02 ,0.001 ] ,
          [0.06728745,0.04704453,0.04,0.04 ,0.04 ,0.04 ,  0.04 ,0.04  ,0.03267206,0.03 ,0.03 ,0.03 , 0.03,
           0.02603239,0.025 ,0.02091093,0.02,0.02 , 0.04562753,0.04133603,0.04  ,0.04  ,0.04 ,0.0308502 , 
           0.02814575,0.028 ,0.02038866,0.02 ,0.02 ,0.02,  0.001 ]]) 
    
    tR_0 = np.array([ [0.2473913 ,0.20681159,0.16623188,0.12565217,0.09753623,0.08362319, 0.08,0.07666667,0.0715942 ,0.07 ,0.0415942 ,0.05681159,
                       0.067 ,0.067 ,0.067 ,0.067 ,0.067 ,0.06537681, 0.065 ,0.065 ,0.065 ,0.065 ,0.065 ,0.065 , 0.065 ,0.065 ,0.065 ,0.065 ,0.065 ,
                       0.065 , 0.065 ] ,
                      [0.08801527,0.07274809,0.05748092,0.04221374,0.03231552,0.02722646, 0.025 ,0.025 ,0.025 ,0.025 ,0.025 ,0.025 , 0.025 ,0.025 ,0.025 ,
                       0.025 ,0.025 ,0.025 , 0.025 ,0.025 ,0.025 ,0.025 ,0.025 ,0.025 , 0.025 ,0.025 ,0.025 ,0.025 ,0.025 ,0.025 , 0.025 ] ,
                      [0.0677823 ,0.05252289,0.03726348,0.02733469,0.025 ,0.025 , 0.025 ,0.025 ,0.025 ,0.025 ,0.025 ,0.025 , 0.025 ,0.025 ,
                       0.025 ,0.025 ,0.025 ,0.025 , 0.01311292,0.01809766,0.02 ,0.02 ,0.02430824,0.025 ,  0.025 ,0.025 ,0.025 ,0.025 ,0.025 ,0.025 , 0.03] , 
                      [0.06546559,0.0502834 ,0.03510121,0.02663968,0.025 ,0.025 , 0.025 ,0.025 ,0.025 ,0.025 ,0.025 ,0.025 , 0.025 ,0.025 ,0.025 ,0.025 ,
                       0.025 ,0.025 , 0.01218623,0.01,0.01 ,0.01890688,0.02451417,0.025 , 0.025 ,0.025 ,0.025 ,0.025 ,0.025 ,0.025 , 0.03] ])
    
    SMOOTHING = 0.1 # more is more smooth, less true to the data
    battery_data.V_oc_interp = RectBivariateSpline(T_bp, SOC_bp, tV_oc, s=SMOOTHING) # % need Deg C
    battery_data.C_Th_interp = RectBivariateSpline(T_bp, SOC_bp, tC_Th, s=SMOOTHING) # % need Deg C
    battery_data.R_Th_interp = RectBivariateSpline(T_bp, SOC_bp, tR_Th, s=SMOOTHING) # % need Deg C
    battery_data.R_0_interp = RectBivariateSpline(T_bp, SOC_bp, tR_0, s=SMOOTHING)   # % need Deg C
 
    return battery_data