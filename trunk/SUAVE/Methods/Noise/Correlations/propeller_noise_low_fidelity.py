# propeller_noise_low_fidelty.py
#
# Created:  Feb 2018, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------
import SUAVE
from SUAVE.Core import Units , Data
import numpy as np
from scipy.special import jv 

def propeller_noise_low_fidelity(noise_data,ctrl_pts):
#def propeller_noise_low_fidelity(segment):
    '''    Source:
        1. Herniczek, M., Feszty, D., Meslioui, S., Park, JongApplicability of Early Acoustic Theory for Modern Propeller Design
        2. Schlegel, R., King, R., and Muli, H., Helicopter Rotor Noise Generation and Propagation, Technical Report, 
        US Army Aviation Material Laboratories, Fort Eustis, VA, 1966
        
       Inputs:
           - segment	 - Mission segment
           - Note that the notation is different from the reference, the speed of sound is denotes as a not c
           and airfoil thickness is denoted with "t" and not "h"
       Outputs:
           SPL     (using 3 methods*)   - Overall Sound Pressure Level, [dB] 
           SPL_dBA (using 3 methods*)   - Overall Sound Pressure Level, [dBA] 
       Assumptions:
           - Empirical based procedure.           
           - The three methods used to compute rotational noise SPL are 1) Gutin and Deming, 2) Barry and Magliozzi and 3) Hanson
           - Vortex noise is computed using the method outlined by Schlegel et. al 
    '''
    
    #noise_data = segment.conditions.propulsion.acoustic_outputs
    #ctrl_pts   = segment.state.numerics.number_control_points
    #segment.conditions.freestream.velocity
    #segment.conditions.frames.inertial.position_vector 
 
    SPL_GD_unweighted      = np.zeros((ctrl_pts,1))
    SPL_BM_unweighted      = np.zeros((ctrl_pts,1))
    SPL_H_unweighted       = np.zeros((ctrl_pts,1))
    SPL_v_unweighted       = np.zeros((ctrl_pts,1)) 
    SPL_GDv_dBA            = np.zeros((ctrl_pts,1))
    SPL_BMv_dBA            = np.zeros((ctrl_pts,1))
    SPL_Hv_dBA             = np.zeros((ctrl_pts,1))    
    SPL_v_dBA              = np.zeros((ctrl_pts,1))
                                      
    total_p_pref_r_GD         = []
    total_p_pref_r_BM         = []
    total_p_pref_r_H          = []
    total_p_pref_v            = [] 
    total_p_pref_GDv_dBA      = []
    total_p_pref_BMv_dBA      = []
    total_p_pref_Hv_dBA       = []
    total_p_pref_v_dBA        = []
    
    harmonics    = np.arange(1,19)  # change to 21
    num_h        = len(harmonics)    
    
    for prop  in noise_data.values():            
        SPL_r_GD          = np.zeros((ctrl_pts,num_h))
        SPL_r_BM          = np.zeros_like(SPL_r_GD)
        SPL_r_H           = np.zeros_like(SPL_r_GD)
         
        p_pref_r_GD       = np.zeros_like(SPL_r_GD)
        p_pref_r_BM       = np.zeros_like(SPL_r_GD)
        p_pref_r_H        = np.zeros_like(SPL_r_GD)
         
        SPL_r_GD_dBA      = np.zeros_like(SPL_r_GD)
        SPL_r_BM_dBA      = np.zeros_like(SPL_r_GD)
        SPL_r_H_dBA       = np.zeros_like(SPL_r_GD)
        
        p_pref_r_GD_dBA_simp = np.zeros_like(SPL_r_GD)
        p_pref_r_GD_dBA      = np.zeros_like(SPL_r_GD)
        p_pref_r_BM_dBA      = np.zeros_like(SPL_r_GD)
        p_pref_r_H_dBA       = np.zeros_like(SPL_r_GD)
 
        #m              = harmonics                                             # harmonic number 
        #p_ref          = 2e-5                                                  # referece atmospheric pressure
        #a              = segment.conditions.freestream.speed_of_sound          # speed of sound
        #rho            = segment.conditions.freestream.density                 # air density 
        #x              = 0 #segment.conditions.frames.inertial.position_vector[:,0] # x  relative position from observer
        #y              = 0 #segment.conditions.frames.inertial.position_vector[:,0] # y  relative position from observer currently only computing directly below aircraft
        #z              = segment.conditions.frames.inertial.position_vector[:,0]    # z relative position from observer
        #Vx             = segment.conditions.frames.inertial.velocity_vector[:,0]    # x velocity of propeller  
        #Vy             = segment.conditions.frames.inertial.velocity_vector[:,0]    # y velocity of propeller 
        #Vz             = segment.conditions.frames.inertial.velocity_vector[:,0]    # z velocity of propeller 
        #thrust_angle   = prop.thrust_angle                                     # propeller thrust angle
        #alpha          = segment.conditions.aerodynamics.angle_of_attack       # vehicle angle of attack                                            
        #N              = prop.number_of_engines                                   # numner of engines
        #B              = prop.number_of_blades                                    # number of rotor blades
        #omega          = prop.omega                                            # angular velocity     
        #T              = prop.blade_T                                          # rotor blade thrust     
        #T_distribution = prop.blade_T_distribution                             # rotor blade thrust distribution  
        #dT_dR          = prop.blade_dT_dR                                      # differential thrust distribution
        #dT_dr          = prop.blade_dT_dr                                      # nondimensionalized differential thrust distribution 
        #Q              = prop.blade_Q                                          # rotor blade torque    
        #Q_distribution = prop.blade_Q_distribution                             # rotor blade torque distribution  
        #dQ_dR          = prop.blade_dT_dR                                      # differential torque distribution
        #dQ_dr          = prop.blade_dT_dr                                      # nondimensionalized differential torque distribution
        #R              = prop.radius_distribution                              # radial location     
        #b              = prop.chord_distribution                               # blade chord    
        #beta           = prop.twist_distribution                               # twist distribution  
        #t              = prop.max_thickness_distribution                       # thickness distribution
        #MCA            = prop.mid_chord_aligment                               # Mid Chord Alighment 
        
        
        dim_p = len(prop.radius_distribution)
        observer_angle = np.pi/4 # observer angle = 45 degrees below propeller
        m              = np.repeat(np.tile(np.atleast_2d(harmonics),(ctrl_pts,1))[:, :, np.newaxis], dim_p, axis=2)                                 # harmonic number 
        p_ref          = 2e-5                                                                                                                       # referece atmospheric pressure
        a              = np.repeat(np.tile(np.atleast_2d(prop.speed_of_sound),(1,num_h))[:, :, np.newaxis], dim_p, axis=2)                          # speed of sound
        rho            = np.repeat(np.tile(np.atleast_2d(prop.density),(1,num_h))[:, :, np.newaxis], dim_p, axis=2)                                 # air density 
        x              = np.repeat(np.tile(np.atleast_2d(prop.position[:,0]).T,(1,num_h))[:, :, np.newaxis], dim_p, axis=2)                         # x relative position from observer
        y              = np.repeat(np.tile(np.atleast_2d(prop.position[:,2]).T,(1,num_h))[:, :, np.newaxis], dim_p, axis=2)/np.tan(observer_angle)  # y relative position from observer , currently taken as 45 degrees below the object
        z              = np.repeat(np.tile(np.atleast_2d(prop.position[:,2]).T,(1,num_h))[:, :, np.newaxis], dim_p, axis=2)                         # z relative position from observer
        Vx             = np.repeat(np.tile(np.atleast_2d(prop.velocity[:,0]).T,(1,num_h))[:, :, np.newaxis], dim_p, axis=2)                         # x velocity of propeller  
        Vy             = np.repeat(np.tile(np.atleast_2d(prop.velocity[:,1]).T,(1,num_h))[:, :, np.newaxis], dim_p, axis=2)                         # y velocity of propeller 
        Vz             = np.repeat(np.tile(np.atleast_2d(prop.velocity[:,2]).T,(1,num_h))[:, :, np.newaxis], dim_p, axis=2)                         # z velocity of propeller 
        thrust_angle   = prop.thrust_angle                                                                                                          # propeller thrust angle
        AoA            = prop.AoA                                                                                                                   # vehicle angle of attack                                            
        N              = prop.number_of_engines                                                                                                     # numner of engines
        B              = prop.number_of_blades                                                                                                      # number of rotor blades
        omega          = np.repeat(np.tile(np.atleast_2d(prop.omega),(1,num_h))[:, :, np.newaxis], dim_p, axis=2)                                   # angular velocity            
        dT_dR          = np.repeat(prop.blade_dT_dR[:, np.newaxis, :], num_h, axis=1)                                                               # differential thrust distribution
        dT_dr          = np.repeat(prop.blade_dT_dr[:, np.newaxis, :], num_h, axis=1)                                                               # nondimensionalized differential thrust distribution   
        dQ_dR          = np.repeat(prop.blade_dT_dR[:, np.newaxis, :], num_h, axis=1)                                                               # differential torque distribution
        dQ_dr          = np.repeat(prop.blade_dT_dr[:, np.newaxis, :], num_h, axis=1)                                                               # nondimensionalized differential torque distribution
        R              = np.repeat(np.tile(np.atleast_2d(prop.radius_distribution),(ctrl_pts,1))[:, np.newaxis, :], num_h, axis=1)                  # radial location     
        b              = np.repeat(np.tile(np.atleast_2d(prop.chord_distribution),(ctrl_pts,1))[:, np.newaxis, :], num_h, axis=1)                   # blade chord    
        beta           = np.repeat(np.tile(np.atleast_2d(prop.twist_distribution),(ctrl_pts,1))[:, np.newaxis, :], num_h, axis=1)                   # twist distribution  
        t              = np.repeat(np.tile(np.atleast_2d(prop.max_thickness_distribution),(ctrl_pts,1))[:, np.newaxis, :], num_h, axis=1)           # twist distribution
        MCA            = np.repeat(np.tile(np.atleast_2d(prop.mid_chord_aligment),(ctrl_pts,1))[:, np.newaxis, :], num_h, axis=1)                   # Mid Chord Alighment         
        
        

        # Coordinate geometry 
        # angle between flight path and propeller axis i.e. thrust vector axis
        alpha    =  AoA + thrust_angle
        hover_ascent  = np.where(AoA <= 0)
        hover_descent = np.where(AoA >= np.pi)
        alpha[hover_ascent]  = thrust_angle[hover_ascent]
        alpha[hover_descent] = thrust_angle[hover_descent]
         
        # theta, angle between flight direction and r, the vector from the observer to the noise source
        r_vector = np.array([ -x , -y , z ])
        v_vector = np.array([ Vx , Vy , Vz])  # flight direction vector, Vy should be zero  #v_vector = np.array([np.cos(alpha), 0.,np.sin(alpha)]) 
        theta    = np.arccos(np.dot(r_vector, v_vector)/(np.linalg.norm(r_vector) *np.linalg.norm(v_vector) ))
        
        # observer distance from propeller axis 
        AB_vec = np.array([np.cos(alpha), 0.,np.sin(alpha)])
        AC_vec = np.array([ x , y , -z ]) 
        
        # A-Weighting for Rotational Noise
        
        '''A-weighted sound pressure level can be obtained by applying A(f) to the
        sound pressure level for each harmonic, then adding the results using the
        method in Appendix B.
        '''
        f  = B*omega*m/(2*np.pi)  
        
        #------------------------------------------------------------------------
        # Rotational SPL  by Barry & Magliozzi
        #------------------------------------------------------------------------
        Y            = np.cross(AC_vec,AB_vec)/np.linalg.norm(AB_vec)  # observer distance from propeller axis           
        X            = np.sqrt(x**2 + z**2)                 # distance to observer from propeller plane (z)  # CORRECT 
        V            = np.sqrt(Vx**2 + Vy**2 + Vz**2)       # velocity magnitude
        M            = V/a                                  # Mach number
        S0           = np.sqrt(x**2 + (1 - M**2)*(Y**2))    # amplitude radius    
        A_x          = 0.6853* b*t                           # airfoil cross-sectional area
        # compute phi_t, blade twist angle relative to propeller plane
        phi_t = np.zeros(n) 
        phi_t = np.pi/2 + abs(beta) 
        locations = np.where(beta > 0)
        phi_t[locations] = np.pi/2 - beta[locations]            
                
        # sound pressure for loading noise 
        p_mL_BM  = (1/(np.sqrt(2)*np.pi*S0))[:,:,0]*np.sum(  (R/( b*np.cos(phi_t)))*np.sin((m*B* b*np.cos(phi_t))/(2*R))* \
                                             ((((M + X/S0)*omega)/(a*(1 - M**2)))*(dT_dR) - (1/R**2)*(dQ_dR))* \
                                             ( (jv(m*B,((m*B*omega*R*Y)/(a*S0)))) + (((1 - M**2)*Y*R)/(2*(S0**2)))*\
                                              ((jv((m*B-1),((m*B*omega*R*Y)/(a*S0)))) - (jv((m*B+1),((m*B*omega*R*Y)/(a*S0)))))  )* dR , axis = 2 ) 
        
        # sound pressure for thickness noise 
        p_mT_BM = -((rho*(m**2)*(omega**2)*(B**3) )/(2*np.sqrt(2)*np.pi*((1 - M**2)**2)))[:,:,0]  *  (((S0 + M*X)**2)/(S0**3))[:,:,0]* \
                       np.sum((A_x *((jv(m*B,((m*B*omega*R*Y)/(a*S0)))) + (((1 - M**2)*Y*R)/(2*(S0**2)))*\
                       ((jv((m*B-1),((m*B*omega*R*Y)/(a*S0)))) - (jv((m*B+1),((m*B*omega*R*Y)/(a*S0))))) )) * dR, axis = 2) 
        p_mT_BM[np.isinf(p_mT_BM)] = 0
        p_mT_BM[np.isneginf(p_mT_BM)] = 0
        
        # unweighted rotational sound pressure level          
        SPL_r_BM        = 10*np.log10(N*((p_mL_BM**2 + p_mT_BM**2 )/p_ref**2))
        p_pref_r_BM     = 10**(SPL_r_BM/10)  
        SPL_r_BM_dBA    = A_weighting(SPL_r_BM,f[:,:,0])
        p_pref_r_BM_dBA = 10**(SPL_r_BM_dBA/10)   
        
        
        #------------------------------------------------------------------------
        # Rotational SPL by Hanson
        #------------------------------------------------------------------------
        D             = 2*R_tip                                                                                 # propeller diameter
        V_tip         = R_tip*omega                                                                             # blade_tip_speed 
        M_t           = V_tip/a                                                                                 # tip Mach number 
        M_s           = np.sqrt(M**2 + (r**2)*(M_t**2))                                                         # section relative Mach number 
        r_t           = R_tip                                                                                   # propeller tip radius
        phi           = np.arctan(z/y)                                                                          # tangential angle  
        theta_r       = np.arccos(np.cos(theta)*np.sqrt(1 - (M**2)*(np.sin(theta))**2) + M*(np.sin(theta))**2 ) # theta angle in the retarded reference frame
        theta_r_prime = np.arccos(np.cos(theta_r)*np.cos(alpha) + np.sin(theta_r)*np.sin(phi)*np.sin(alpha) )   #
        phi_prime     = np.arccos((np.sin(theta_r)/np.sin(theta_r_prime))*np.cos(phi))                          # phi angle relative to propeller shaft axis                                                   
        phi_s         = ((2*m*B*M_t)/(M_s*(1 - M*np.cos(theta_r))))*(MCA/D)                                     # phase lag due to sweep
        S_r           = Y/(np.sin(theta_r))                                                                     # distance in retarded reference frame 
        k_x           = ((2*m*B* b*M_t)/(M_s*(1 - M*np.cos(theta_r))))                                          # wave number  
    
        psi_V = (8/(k_x**2))*((2/k_x)*np.sin(0.5*k_x) - np.cos(0.5*k_x))                                        # normalized thickness souce transforms           
        psi_L = (2/k_x)*np.sin(0.5*k_x)                                                                         # normalized loading souce transforms
        locations   = np.where(k_x == 0)
        psi_V[locations] = 2/3
        psi_L[locations] = 1   
        # sound pressure for loading noise 
        p_mL_H = ((  1j*m*B*M_t*np.sin(theta_r)*np.exp(1j*m*B*((omega*S_r/a)+(phi_prime - np.pi/2))) )[:,:,0] / (2*np.sqrt(2)*np.pi*Y*r_t*(1 - M*np.cos(theta_r)))[:,:,0]  ) \
                  *np.sum(( (   (np.cos(theta_r_prime)/(1 - M*np.cos(theta_r)))*(dT_dr) - (1/((r**2)*M_t*r_t))*(dQ_dr)  ) * np.exp(1j*phi_s) *\
                       (jv(m*B,((m*B*r*M_t*np.sin(theta_r_prime))/(1 - M*np.cos(theta_r))))) * psi_L  )* dr , axis = 2)  
        p_mL_H[np.isinf(p_mL_H)] = 0
        p_mL_H = abs(p_mL_H)
        
        # sound pressure for thickness noise 
        p_mT_H = (-(rho*(a**2)*B*np.sin(theta_r)*np.exp(1j*m*B*((omega*S_r/a)+(phi_prime - np.pi/2))))/(4*np.sqrt(2)*np.pi*(Y/D)*(1 - M*np.cos(theta_r))))[:,:,0] \
                *np.sum(((M_s**2)*(t/ b)*np.exp(1j*phi_s)*(jv(m*B,((m*B*r*M_t*np.sin(theta_r_prime))/(1 - M*np.cos(theta_r)))))*(k_x**2)*psi_V ) *dr , axis = 2)  
        p_mT_H[np.isinf(p_mT_H)] = 0  
        p_mT_H  = abs(p_mT_H)
        
        # unweighted rotational sound pressure level
        SPL_r_H        = 10*np.log10(N*((p_mL_H**2 + p_mT_H**2 )/p_ref**2))  
        p_pref_r_H     = 10**(SPL_r_H/10)  
        SPL_r_H_dBA    = A_weighting(SPL_r_H,f[:,:,0])
        p_pref_r_H_dBA = 10**(SPL_r_H_dBA/10)   
        
        # -----------------------------------------------------------------------
        # Vortex Noise (developed by Schlegel et. al in Helicopter Rotor Noise) This is computed in feet 
        # ----------------------------------------------------------------------- 
        V_07      = V_tip*0.70/(Units.feet)                                  # blade velocity at r/R_tip = 0.7 
        St        = 0.28                                                     # Strouhal number             
        t_avg     = np.mean(t)/(Units.feet)                                  # thickness
        c_avg     = np.mean(b)/(Units.feet)                                 # average chord  
        beta_07   = beta[0,0,:][round(n*0.70)]                                      # blade angle of attack at r/R = 0.7
        h_val     = t_avg*np.cos(beta_07) + c_avg*np.sin(beta_07)            # projected blade thickness                   
        f_peak    = (V_07*St)/h_val                                          # V - blade velocity at a radial location of 0.7      
        A_blade   = np.repeat(((np.trapz( b, dx = dR))/(Units.feet**2))[:, :, np.newaxis], dim_p, axis=2)   # area of blade
        CL_07     = 2*np.pi*beta_07
        S_feet    = S/(Units.feet)
        SPL_300ft = 10*np.log10(((6.1e-27)*A_blade*V_07**6)/(10**-16)) + 20*np.log(CL_07/0.4)
        SPL_v     = SPL_300ft - 20*np.log10(S_feet/300)
        
        # estimation of A-Weighting for Vortex Noise   
        f_spectrum  = np.array([0.5*f_peak[:,0,0], 1*f_peak[:,0,0] , 2*f_peak[:,0,0] , 4*f_peak[:,0,0] , 8*f_peak[:,0,0] , 16*f_peak[:,0,0]]).T  # spectrum
        dim_spec    = len(f_spectrum[0,:])
        fr          = f_spectrum/ np.tile(np.atleast_2d(f_peak[:,0,0]).T,(1,dim_spec))                                                  # frequency ratio  
        SPL_weight  = np.array([7.92 , 4.17 , 8.33 , 8.75 ,12.92 , 13.33])                           # SPL weight
        SPL_v       = np.tile(np.atleast_2d(SPL_v[:,0,0]).T,(1,dim_spec)) - np.tile(np.atleast_2d(SPL_weight),(ctrl_pts,1))   # SPL correction
        p_pref_v    = 10**(SPL_v/10)
        
        C             = np.zeros((ctrl_pts,dim_spec)) 
        p_pref_v_dBA  = np.zeros((ctrl_pts,dim_spec-1))
        SPL_v_dbAi    = np.zeros((ctrl_pts,dim_spec))
        SPL_v_dbAi    = A_weighting(SPL_v,f_spectrum ) 
        C[:,:-1]      = (SPL_v_dbAi[:,1:] - SPL_v_dbAi[:,:-1])/(np.log10(fr[:,1:]) - np.log10(fr[:,:-1])) 
        C[:,-1]       = SPL_v_dbAi[:,-1] - C[:,-2]*np.log10(fr[:,-1])   
        p_pref_v_dBA  = (10**(0.1*C[:,1:]))* (  ((fr[:,1:]**(0.1*C[:,:-1] + 1 ))/(0.1*C[:,:-1] + 1 )) - ((fr[:,:-1]**(0.1*C[:,:-1] + 1 ))/(0.1*C[:,:-1] + 1 )) )
         
        prop.SPL_GD_unweighted       = SPL_r_GD
        prop.SPL_BM_unweighted       = SPL_r_BM 
        prop.SPL_H_unweighted        = SPL_r_H  
        prop.SPL_v_unweighted        = SPL_v
        
        # collecting unweighted pressure ratios           
        total_p_pref_r_GD.append(p_pref_r_GD)  
        total_p_pref_r_BM.append(p_pref_r_BM)  
        total_p_pref_r_H.append(p_pref_r_H)    
        total_p_pref_v.append(p_pref_v)
        
        # collecting weighted pressure ratios with vortex noise included 
        total_p_pref_GDv_dBA.append(np.hstack((p_pref_r_GD_dBA,p_pref_v_dBA))) 
        total_p_pref_BMv_dBA.append(np.hstack((p_pref_r_BM_dBA,p_pref_v_dBA))) 
        total_p_pref_Hv_dBA.append(np.hstack((p_pref_r_H_dBA,p_pref_v_dBA)))   
        total_p_pref_v_dBA.append(p_pref_v_dBA)
    
    # Rotational SPL (Unweighted)    
    SPL_GD_unweighted      = np.atleast_2d(decibel_arithmetic(p_pref_r_GD)).T       # Gutin & Deming rotational noise with Schlegel vortex noise
    SPL_BM_unweighted      = np.atleast_2d(decibel_arithmetic(p_pref_r_BM)).T       # Barry & Magliozzi rotational noise with Schlegel vortex noise
    SPL_H_unweighted       = np.atleast_2d(decibel_arithmetic(p_pref_r_H)).T        # Hanson rotational noise with Schlegel vortex noise         
    SPL_v_unweighted       = np.atleast_2d(decibel_arithmetic(p_pref_v)).T
    
    # A- Weighted Rotational and Vortex SPL 
    SPL_GDv_dBA      = np.atleast_2d(decibel_arithmetic(total_p_pref_GDv_dBA)).T
    SPL_BMv_dBA      = np.atleast_2d(decibel_arithmetic(total_p_pref_BMv_dBA)).T
    SPL_Hv_dBA       = np.atleast_2d(decibel_arithmetic(total_p_pref_Hv_dBA)).T
    SPL_v_dBA        = np.atleast_2d(decibel_arithmetic(total_p_pref_v_dBA)).T
    
    noise_data.acoustic_results = Data( 
        SPL_GD_unweighted      =  SPL_GD_unweighted ,
        SPL_BM_unweighted      =  SPL_BM_unweighted ,
        SPL_H_unweighted       =  SPL_H_unweighted,  
        SPL_v_unweighted       =  SPL_v_unweighted ,   
        SPL_GDv_dBA            =  SPL_GDv_dBA,       
        SPL_BMv_dBA            =  SPL_BMv_dBA ,      
        SPL_Hv_dBA             =  SPL_Hv_dBA,            
        SPL_v_dBA              =  SPL_v_dBA
        )
    
    return SPL_GD_unweighted , SPL_BM_unweighted , SPL_H_unweighted , SPL_v_unweighted , SPL_GDv_dBA  , SPL_BMv_dBA , SPL_Hv_dBA , SPL_v_dBA   
    #return
# -----------------------------------------------------------------------
# Decibel Arithmetic
# -----------------------------------------------------------------------
def decibel_arithmetic(p_pref_total):
    SPL_total = 10*np.log10(np.sum(p_pref_total,axis = 1 ))
    return SPL_total

# -----------------------------------------------------------------------
# Rotational Noise A-Weight
# -----------------------------------------------------------------------
def  A_weighting(SPL,f):
    Ra_f       = ((12200**2)*(f**4))/ (((f**2)+(20.6**2)) * ((f**2)+(12200**2)) * (((f**2) + 107.7**2)**0.5)* (((f**2)+ 737.9**2)**0.5)) 
    A_f        =  2.0  + 20*np.log10(Ra_f) 
    SPL_dBA    = SPL + A_f
    return SPL_dBA