# test_VTOL.py
# 
# Created:  July 2018, M. Clarke
#
""" setup file for eVTOL missions
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units 
import numpy as np
import pylab as plt

import copy, time 
from SUAVE.Core import (
Data, Container,
)

import sys 
sys.path.append('../Vehicles')
# the analysis functions 

sys.path.append('../VTOL')
# the analysis functions

from Lift_Cruise_CRM import vehicle_setup as lift_cruise_vehicle_setup 
from Lift_Cruise_CRM import configs_setup as lift_cruise_configs_setup

import copy

# ----------------------------------------------------------------------
#   Main
# ---------------------------------------------------------------------- 
def main():
    
    lift_cruise_configuration_test()
    
    
    
    return

def lift_cruise_configuration_test():
    # build the vehicle, configs, and analyses
    configs, analyses = lift_cruise_full_setup()
    
    # configs.finalize()
    analyses.finalize()    
    
    # weight analysis
    weights =  analyses.configs.base.weights 
    breakdown = weights.evaluate()          
    
    # mission analysis
    mission = analyses.missions.base
    results = mission.evaluate()

    # RPM check during hover
    RPM            = results.segments.hover_1.conditions.propulsion.rpm[0] 
    RPM_true       = 4685.21033888
    print(RPM) 
    diff_RPM                        = np.abs(RPM - RPM_true)
    print('RPM difference')
    print(diff_RPM)
    assert np.abs((RPM - RPM_true)/RPM_true) < 1e-3  
    
    # battery energy check during transition
    battery_energy_trans_to_hover              = results.segments.transition_to_hover.conditions.propulsion.battery_energy[0]
    battery_energy_trans_to_hover_true         = 92097.82354179
    print(battery_energy_trans_to_hover)
    diff_battery_energy_trans_to_hover                      = np.abs(battery_energy_trans_to_hover  - battery_energy_trans_to_hover_true) 
    print('battery_energy_trans_to_hover difference')
    print(diff_battery_energy_trans_to_hover)
    assert np.abs((battery_energy_trans_to_hover  - battery_energy_trans_to_hover_true)/battery_energy_trans_to_hover) < 1e-3


    # lift coefficient check during cruise
    lift_coefficient              = results.segments.cruise.conditions.aerodynamics.lift_coefficient[0]
    lift_coefficient_true         = 0.3527293
    print(lift_coefficient)
    diff_CL                       = np.abs(lift_coefficient  - lift_coefficient_true) 
    print('CL difference')
    print(diff_CL)
    assert np.abs((lift_coefficient  - lift_coefficient_true)/lift_coefficient_true) < 1e-3    
    
    # plot results
    #plot_mission(results,configs)
    
        
    return

# ----------------------------------------------------------------------
#   Analysis Setup
# ----------------------------------------------------------------------
def lift_cruise_full_setup():
    
    # vehicle data
    vehicle  = lift_cruise_vehicle_setup() 
    configs  = lift_cruise_configs_setup(vehicle)

    # vehicle analyses
    configs_analyses = lift_cruise_analyses_setup(configs)

    # mission analyses
    mission  = lift_cruise_mission_setup(configs_analyses,vehicle)
    missions_analyses = lift_cruise_missions_setup(mission)
    
    analyses = SUAVE.Analyses.Analysis.Container()
    analyses.configs  = configs_analyses
    analyses.missions = missions_analyses    
    
    return configs, analyses



# ----------------------------------------------------------------------
#   Define the Vehicle Analyses
# ----------------------------------------------------------------------
def lift_cruise_analyses_setup(configs):

    analyses = SUAVE.Analyses.Analysis.Container()

    # build a base analysis for each config
    for tag,config in configs.items():
        analysis = lift_cruise_base_analysis(config)
        analyses[tag] = analysis

    return analyses

def lift_cruise_base_analysis(vehicle):
    # ------------------------------------------------------------------
    #   Initialize the Analyses
    # ------------------------------------------------------------------     
    analyses = SUAVE.Analyses.Vehicle()

    # ------------------------------------------------------------------
    #  Basic Geometry Relations
    sizing = SUAVE.Analyses.Sizing.Sizing()
    sizing.features.vehicle = vehicle
    analyses.append(sizing)

    # ------------------------------------------------------------------
    #  Weights
    weights = SUAVE.Analyses.Weights.Weights_Electric_Lift_Cruise()
    weights.vehicle = vehicle
    analyses.append(weights)

    # ------------------------------------------------------------------
    #  Aerodynamics Analysis
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.4*vehicle.excrescence_area_spin / vehicle.reference_area
    analyses.append(aerodynamics)

    # ------------------------------------------------------------------
    #  Energy
    energy= SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.propulsors 
    analyses.append(energy)

    # ------------------------------------------------------------------
    #  Planet Analysis
    planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(planet)

    # ------------------------------------------------------------------
    #  Atmosphere Analysis
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)   

    return analyses    


def lift_cruise_mission_setup(analyses,vehicle):
    # ------------------------------------------------------------------
    #   Initialize the Mission
    # ------------------------------------------------------------------
    mission            = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag        = 'the_mission'

    # airport
    airport            = SUAVE.Attributes.Airports.Airport()
    airport.altitude   =  0.0  * Units.ft
    airport.delta_isa  =  0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()

    mission.airport    = airport    

    # unpack Segments module
    Segments                                                 = SUAVE.Analyses.Mission.Segments
                                                             
    # base segment                                           
    base_segment                                             = Segments.Segment()
    ones_row                                                 = base_segment.state.ones_row
    base_segment.state.numerics.number_control_points        = 10
    base_segment.process.iterate.initials.initialize_battery = SUAVE.Methods.Missions.Segments.Common.Energy.initialize_battery
    base_segment.process.iterate.conditions.planet_position  = SUAVE.Methods.skip
    base_segment.process.iterate.unknowns.network            = vehicle.propulsors.propulsor.unpack_unknowns_transition
    base_segment.process.iterate.residuals.network           = vehicle.propulsors.propulsor.residuals_transition
    base_segment.state.unknowns.battery_voltage_under_load   = vehicle.propulsors.propulsor.battery.max_voltage * ones_row(1)  
    base_segment.state.residuals.network                     = 0. * ones_row(2)    


    # VSTALL Calculation
    m     = vehicle.mass_properties.max_takeoff
    g     = 9.81
    S     = vehicle.reference_area
    atmo  = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    rho   = atmo.compute_values(1000.*Units.feet,0.).density
    CLmax = 1.2

    Vstall = float(np.sqrt(2.*m*g/(rho*S*CLmax)))

    # ------------------------------------------------------------------
    #   First Taxi Segment: Constant Speed
    # ------------------------------------------------------------------
    segment     = Segments.Hover.Climb(base_segment)
    segment.tag = "Ground_Taxi"

    # ------------------------------------------------------------------
    #   First Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------
    segment     = Segments.Hover.Climb(base_segment)
    segment.tag = "climb_1"
    segment.analyses.extend( analyses.base  )                                                            
    segment.altitude_start                                   = 0.0  * Units.ft
    segment.altitude_end                                     = 40.  * Units.ft
    segment.climb_rate                                       = 500. * Units['ft/min']
    segment.battery_energy                                   = vehicle.propulsors.propulsor.battery.max_energy*0.95
                                                             
    segment.state.unknowns.propeller_power_coefficient_lift  = 0.05* ones_row(1)
    segment.state.unknowns.throttle_lift                     = 1.25 * ones_row(1)
    segment.state.unknowns.__delitem__('throttle')

    segment.process.iterate.unknowns.network                 = vehicle.propulsors.propulsor.unpack_unknowns_no_forward
    segment.process.iterate.residuals.network                = vehicle.propulsors.propulsor.residuals_no_forward       
    segment.process.iterate.unknowns.mission                 = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability             = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability          = SUAVE.Methods.skip

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   First Cruise Segment: Transition
    # ------------------------------------------------------------------
    segment     = Segments.Transition.Constant_Acceleration_Constant_Pitchrate_Constant_Altitude(base_segment)
    segment.tag = "transition_1"
    segment.analyses.extend( analyses.base  )
    segment.altitude                                        = 40.  * Units.ft
    segment.air_speed_start                                 = 0.   * Units['ft/min']
    segment.air_speed_end                                   = 1.2 * Vstall
    segment.acceleration                                    = 9.81/5
    segment.pitch_initial                                   = 0.0
    segment.pitch_final                                     = 7.75 * Units.degrees

    segment.state.unknowns.propeller_power_coefficient_lift = 0.05 * ones_row(1)
    segment.state.unknowns.throttle_lift                    = 1.25 * ones_row(1)
    segment.state.unknowns.propeller_power_coefficient      = 0.02 * ones_row(1)
    segment.state.unknowns.throttle                         = .50 * ones_row(1)   
    segment.state.residuals.network                         = 0. * ones_row(3)    

    segment.process.iterate.unknowns.network                = vehicle.propulsors.propulsor.unpack_unknowns_transition
    segment.process.iterate.residuals.network               = vehicle.propulsors.propulsor.residuals_transition    
    segment.process.iterate.unknowns.mission                = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability            = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability         = SUAVE.Methods.skip

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Second Climb Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------
    segment     = Segments.Climb.Constant_Speed_Constant_Rate(base_segment)
    segment.tag = "climb_2"
    segment.analyses.extend( analyses.base  )
    segment.air_speed                                   = np.sqrt((500 * Units['ft/min'])**2 + (1.2*Vstall)**2)
    segment.altitude_start                              = 40.0 * Units.ft
    segment.altitude_end                                = 300. * Units.ft
    segment.climb_rate                                  = 500. * Units['ft/min']

    segment.state.unknowns.propeller_power_coefficient  = 0.01 * ones_row(1)
    segment.state.unknowns.throttle                     = 0.70 * ones_row(1)
    segment.process.iterate.unknowns.network            = vehicle.propulsors.propulsor.unpack_unknowns_no_lift
    segment.process.iterate.residuals.network           = vehicle.propulsors.propulsor.residuals_no_lift     

    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Second Cruise Segment: Constant Speed, Constant Altitude
    # ------------------------------------------------------------------
    segment     = Segments.Cruise.Constant_Speed_Constant_Altitude_Loiter(base_segment)
    segment.tag = "departure_terminal_procedures"
    segment.analyses.extend( analyses.base  )
    segment.altitude                                   = 300.0 * Units.ft
    segment.time                                       = 60.   * Units.second
    segment.air_speed                                  = 1.2*Vstall

    segment.state.unknowns.propeller_power_coefficient = 0.01 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.50 * ones_row(1)

    segment.process.iterate.unknowns.network           = vehicle.propulsors.propulsor.unpack_unknowns_no_lift
    segment.process.iterate.residuals.network          = vehicle.propulsors.propulsor.residuals_no_lift     


    # add to misison
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    #   Third Climb Segment: Constant Acceleration, Constant Rate
    # ------------------------------------------------------------------
    segment     = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
    segment.tag = "accelerated_climb"
    segment.analyses.extend( analyses.base  )
    segment.altitude_start                             = 300.0 * Units.ft
    segment.altitude_end                               = 1000. * Units.ft
    segment.climb_rate                                 = 500.  * Units['ft/min']
    segment.air_speed_start                            = np.sqrt((500 * Units['ft/min'])**2 + (1.2*Vstall)**2)
    segment.air_speed_end                              = 110.  * Units['mph']                                            

    segment.state.unknowns.propeller_power_coefficient = 0.01 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.50 * ones_row(1)

    segment.process.iterate.unknowns.network           = vehicle.propulsors.propulsor.unpack_unknowns_no_lift
    segment.process.iterate.residuals.network          = vehicle.propulsors.propulsor.residuals_no_lift  


    # add to misison
    mission.append_segment(segment)    

    # ------------------------------------------------------------------
    #   Third Cruise Segment: Constant Acceleration, Constant Altitude
    # ------------------------------------------------------------------
    segment     = Segments.Cruise.Constant_Speed_Constant_Altitude(base_segment)
    segment.tag = "cruise"
    segment.analyses.extend( analyses.base  )
    segment.altitude                                   = 1000.0 * Units.ft
    segment.air_speed                                  = 110.   * Units['mph']
    segment.distance                                   = 60.    * Units.miles                       

    segment.state.unknowns.propeller_power_coefficient = 0.01 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.50 * ones_row(1)

    segment.process.iterate.unknowns.network           = vehicle.propulsors.propulsor.unpack_unknowns_no_lift
    segment.process.iterate.residuals.network          = vehicle.propulsors.propulsor.residuals_no_lift    

    # add to misison
    mission.append_segment(segment)     

    # ------------------------------------------------------------------
    #   First Descent Segment: Constant Acceleration, Constant Rate
    # ------------------------------------------------------------------
    segment     = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
    segment.tag = "decelerating_descent"
    segment.analyses.extend(analyses.base )  
    segment.altitude_start                             = 1000.0 * Units.ft
    segment.altitude_end                               = 300. * Units.ft
    segment.climb_rate                                 = -500.  * Units['ft/min']
    segment.air_speed_start                            = 110.  * Units['mph']
    segment.air_speed_end                              = 1.2*Vstall
    
    segment.state.unknowns.propeller_power_coefficient = 0.05 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.75 * ones_row(1)

    segment.process.iterate.unknowns.network           = vehicle.propulsors.propulsor.unpack_unknowns_no_lift
    segment.process.iterate.residuals.network          = vehicle.propulsors.propulsor.residuals_no_lift     

    # add to misison
    mission.append_segment(segment)        

    # ------------------------------------------------------------------
    #   Fourth Cruise Segment: Constant Speed, Constant Altitude
    # ------------------------------------------------------------------
    segment     = Segments.Cruise.Constant_Speed_Constant_Altitude_Loiter(base_segment)
    segment.tag = "arrival_terminal_procedures"
    segment.analyses.extend( analyses.base )
    segment.altitude                                   = 300.   * Units.ft
    segment.air_speed                                  = 1.2*Vstall
    segment.time                                       = 60 * Units.seconds

    segment.state.unknowns.propeller_power_coefficient = 0.01 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.50 * ones_row(1)

    segment.process.iterate.unknowns.network           = vehicle.propulsors.propulsor.unpack_unknowns_no_lift
    segment.process.iterate.residuals.network          = vehicle.propulsors.propulsor.residuals_no_lift   

    # add to misison
    mission.append_segment(segment)    

    # ------------------------------------------------------------------
    #   Second Descent Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------
    segment     = Segments.Climb.Linear_Speed_Constant_Rate(base_segment)
    segment.tag = "descent_2"
    segment.analyses.extend( analyses.base  )
    segment.altitude_start                             = 300.0 * Units.ft
    segment.altitude_end                               = 40. * Units.ft
    segment.climb_rate                                 = -400.  * Units['ft/min']  # Uber has 500->300
    segment.air_speed_start                            = np.sqrt((400 * Units['ft/min'])**2 + (1.2*Vstall)**2)
    segment.air_speed_end                              = 1.2*Vstall 
    
    segment.state.unknowns.propeller_power_coefficient = 0.01 * ones_row(1)
    segment.state.unknowns.throttle                    = 0.50 * ones_row(1)

    segment.process.iterate.unknowns.network           = vehicle.propulsors.propulsor.unpack_unknowns_no_lift
    segment.process.iterate.residuals.network          = vehicle.propulsors.propulsor.residuals_no_lift 

    # add to misison
    mission.append_segment(segment)       

    # ------------------------------------------------------------------
    #   Fifth Cuise Segment: Transition
    # ------------------------------------------------------------------ 
    segment     = Segments.Transition.Constant_Acceleration_Constant_Pitchrate_Constant_Altitude(base_segment)
    segment.tag = "transition_2"
    segment.analyses.extend( analyses.base  )
    segment.altitude                                        = 40. * Units.ft
    segment.air_speed_start                                 = 1.2*Vstall      
    segment.air_speed_end                                   = 0 
    segment.acceleration                                    = -9.81/20
    segment.pitch_initial                                   = 5. * Units.degrees   
    segment.pitch_final                                     = 10. * Units.degrees      
                                                           
    segment.state.unknowns.propeller_power_coefficient_lift = 0.085 * ones_row(1)  
    segment.state.unknowns.throttle_lift                    = 0.9 * ones_row(1)    
    segment.state.unknowns.propeller_power_coefficient      = 0.03 * ones_row(1)  
    segment.state.unknowns.throttle                         = 0.7 * ones_row(1)   
    segment.state.residuals.network                         = 0. * ones_row(3)     
    
    segment.process.iterate.unknowns.network                = vehicle.propulsors.propulsor.unpack_unknowns_transition
    segment.process.iterate.residuals.network               = vehicle.propulsors.propulsor.residuals_transition    
    segment.process.iterate.unknowns.mission                = SUAVE.Methods.skip 
    # add to misison
    mission.append_segment(segment)

      
    # ------------------------------------------------------------------
    #   Third Descent Segment: Constant Speed, Constant Rate
    # ------------------------------------------------------------------
    segment     = Segments.Hover.Descent(base_segment)
    segment.tag = "descent_1"
    segment.analyses.extend( analyses.base  )
    segment.altitude_start                                  = 40.0  * Units.ft
    segment.altitude_end                                    = 0.   * Units.ft
    segment.descent_rate                                    = 300. * Units['ft/min']
    segment.battery_energy                                  = vehicle.propulsors.propulsor.battery.max_energy

    segment.state.unknowns.propeller_power_coefficient_lift = 0.05* ones_row(1)
    segment.state.unknowns.throttle_lift                    = 0.9 * ones_row(1)

    segment.state.unknowns.__delitem__('throttle')
    segment.process.iterate.unknowns.network                = vehicle.propulsors.propulsor.unpack_unknowns_no_forward
    segment.process.iterate.residuals.network               = vehicle.propulsors.propulsor.residuals_no_forward    
    segment.process.iterate.unknowns.mission                = SUAVE.Methods.skip
    segment.process.iterate.conditions.stability            = SUAVE.Methods.skip
    segment.process.finalize.post_process.stability         = SUAVE.Methods.skip

    # add to misison
    mission.append_segment(segment)   
  
    return mission

def lift_cruise_missions_setup(base_mission):

    # the mission container
    missions = SUAVE.Analyses.Mission.Mission.Container()

    # ------------------------------------------------------------------
    #   Base Mission
    # ------------------------------------------------------------------

    missions.base = base_mission


    # done!
    return missions  

# ----------------------------------------------------------------------
#   Define the Configurations
# ---------------------------------------------------------------------

if __name__ == '__main__': 
    main()    
