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
 
def propeller_noise_low_fidelity(segment):
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
    harmonics    = np.arange(1,21)   
    num_h        = len(harmonics)     
    noise_data = segment.conditions.propulsion.acoustic_outputs
    ctrl_pts   = segment.state.numerics.number_control_points 
    num_pt     = len(noise_data.values())
    
    SPL_BM_unweighted      = np.zeros((ctrl_pts,1))
    SPL_H_unweighted       = np.zeros((ctrl_pts,1))
    SPL_v_unweighted       = np.zeros((ctrl_pts,1)) 
    SPL_BMv_dBA            = np.zeros((ctrl_pts,1))
    SPL_Hv_dBA             = np.zeros((ctrl_pts,1))    
    SPL_v_dBA              = np.zeros((ctrl_pts,1))
                                      
    total_p_pref_r_BM         = np.zeros((ctrl_pts,num_pt*(num_h)))
    total_p_pref_r_H          = np.zeros((ctrl_pts,num_pt*(num_h)))
    total_p_pref_v            = np.zeros((ctrl_pts,num_pt*(6))) 
    total_p_pref_BMv_dBA      = np.zeros((ctrl_pts,num_pt*(num_h + 5)))
    total_p_pref_Hv_dBA       = np.zeros((ctrl_pts,num_pt*(num_h + 5)))
    total_p_pref_v_dBA        = np.zeros((ctrl_pts,num_pt*(num_h + 5)))
    
   
    idx = 0 
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
         
        dim_p          = len(prop.radius_distribution)
                                                                                                                                                    # observer angle = 45 degrees below propeller
        # atmospheric conditions 
        m              = np.repeat(np.tile(np.atleast_2d(harmonics),(ctrl_pts,1))[:, :, np.newaxis], dim_p, axis=2)                                 # harmonic number 
        p_ref          = 2e-5                                                                                                                       # referece atmospheric pressure
        a              = np.repeat(np.tile(np.atleast_2d(segment.conditions.freestream.speed_of_sound),(1,num_h))[:, :, np.newaxis], dim_p, axis=2) # speed of sound
        rho            = np.repeat(np.tile(np.atleast_2d(segment.conditions.freestream.density),(1,num_h))[:, :, np.newaxis], dim_p, axis=2)        # air density 
       
        # position of noise source and observer 
        observer_angle = np.pi/4   
        x_prop         = -segment.conditions.frames.inertial.position_vector[:,0]                                                                         
        y_prop         = -segment.conditions.frames.inertial.position_vector[:,1]                          # currently in suave this is 0 i.e. vertical flight
        z_prop         = -segment.conditions.frames.inertial.position_vector[:,2]                          # noise source at altitide              
        x_obs          = -segment.conditions.frames.inertial.position_vector[:,0]                          # currently observer is moving with noise source 
        y_obs          = -segment.conditions.frames.inertial.position_vector[:,2] /np.tan(observer_angle)  # currently taken at 45 degrees below noise source
        z_obs          = np.zeros_like(segment.conditions.frames.inertial.position_vector[:,2])           # ground = 0 altitide

        # velocity 
        v_vector      = prop.velocity      
        
        # propeller attributes 
        thrust_angle   = prop.thrust_angle                                                                                                          # propeller thrust angle
        AoA            = segment.conditions.aerodynamics.angle_of_attack                                                                            # vehicle angle of attack                                            
        N              = prop.number_of_engines                                                                                                     # number of engines
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
        
        if omega[idx,:].all()==0: # if no rpm, no acoustic measurement is computed
            continue 
        
        # angle between flight path and propeller axis i.e. thrust vector axis
        alpha                = AoA + thrust_angle
        hover_ascent         = np.where(AoA <= 0)         # the AoA of a vertical climb is -90
        hover_descent        = np.where(AoA >= np.pi)     # the AoA of a verticle descent is +90
        alpha[hover_ascent]  = thrust_angle
        alpha[hover_descent] = thrust_angle
        
        # condition if vehicle is not moving 
        for i in range(ctrl_pts):
            if v_vector[i,:].all() == 0.:
                v_vector[i,0] = np.cos(thrust_angle)
                v_vector[i,1] = 0.
                v_vector[i,2] = np.sin(thrust_angle)
                 
        # theta, angle between flight direction and r, the vector from the propeller center axis at the emission point to the observer 
        P          = np.zeros((ctrl_pts,3))
        P[:,0]     = x_prop[:]  # propeller x location    
        P[:,1]     = y_prop[:]  # propeller y location 
        P[:,2]     = z_prop[:]  # propeller z location 
        O          = np.zeros((ctrl_pts,3))
        O[:,0]     = x_obs[:]  # propeller x location    
        O[:,1]     = y_obs[:]  # propeller y location 
        O[:,2]     = z_obs[:]  # propeller z location         
         
        PO_vec     = O - P      # called r in Hanson, the vector from the propeller center axis at the emission point to the observer  
        S          = np.linalg.norm(PO_vec, axis = 1)                          # distance between rotor and         
        theta      = np.arccos(np.sum(PO_vec*v_vector, axis=1)/(np.linalg.norm(PO_vec, axis = 1) *np.linalg.norm(v_vector, axis = 1))) # Source : https://onlinemschool.com/math/library/vector/angl/
        phi           = np.arctan(z_obs/y_obs)  # tangential angle  
        
        # observer distance from propeller axis  
        PA_vec      = np.zeros((ctrl_pts,3))
        PA_vec[:,0] = np.cos(alpha).T[0]           # propeller x location    
        PA_vec[:,1] = 0.                           # propeller y location 
        PA_vec[:,2] = np.sin(alpha).T[0]           # propeller z location          
        
        mag_PA         = np.atleast_2d(np.linalg.norm(PA_vec, axis = 1)).T
        normal_PA      = PA_vec / mag_PA
        mag_normal_PA  = np.atleast_2d(np.linalg.norm(normal_PA , axis = 1)).T
        D1             = np.atleast_2d(np.sum(normal_PA *P, axis=1)).T  # dot product 
        d1             = np.abs(D1/mag_normal_PA)
        D2             = np.atleast_2d(np.sum(normal_PA *O, axis=1)).T  # dot product 
        d2             = np.abs(D2/mag_normal_PA )        
        X              = d1 + d2                                                                    # distance to observer from propeller plane (z_obs):  Source of equation https://www.math.tamu.edu/~glahodny/Math251/Section%2011.4.pdf ,  https://www.youtube.com/watch?v=Noes0nOvLg4  
        Y              = np.atleast_2d(np.linalg.norm(np.cross(PO_vec,PA_vec) , axis = 1)).T/mag_PA # observer distance from propeller axis: Source (method 2) :https://www.qc.edu.hk/math/Advanced%20Level/Point_to_line.htm    
        
        # Coordinate geometry  
        n        = len(R[0,0,:])
        R_tip    = R[0,0,-1]                                                 # Rotor Tip Radius     
        r        = R/R_tip                                                   # non dimensional radius distribution 
        dR       = R[0,0,1] - R[0,0,0]                                      
        dr       = r[0,0,1] - r[0,0,0]                                        
        A        = np.pi*(R_tip**2)                                          # rotor Disc Area 
              
        # Reshape vectors for calculations
        # dimension of vectors = [control point; harmonic ; propeller axis]
        V        = np.atleast_2d(np.linalg.norm(v_vector, axis = 1)).T                  # velocity magnitude 
        V        = np.repeat(np.tile(np.atleast_2d(V),(1,num_h))[:, :, np.newaxis], dim_p, axis=2)                               
        Y        = np.repeat(np.tile(np.atleast_2d(Y),(1,num_h))[:, :, np.newaxis], dim_p, axis=2)                         
        X        = np.repeat(np.tile(np.atleast_2d(X),(1,num_h))[:, :, np.newaxis], dim_p, axis=2)                               
        S        = np.repeat(np.tile(np.atleast_2d(S).T,(1,num_h))[:, :, np.newaxis], dim_p, axis=2)
        x_obs    = np.repeat(np.tile(np.atleast_2d(x_obs).T,(1,num_h))[:, :, np.newaxis], dim_p, axis=2) 
        phi      = np.repeat(np.tile(np.atleast_2d(phi).T,(1,num_h))[:, :, np.newaxis], dim_p, axis=2) 
        theta    = np.repeat(np.tile(np.atleast_2d(theta).T,(1,num_h))[:, :, np.newaxis], dim_p, axis=2)
        alpha    = np.repeat(np.tile(np.atleast_2d(alpha),(1,num_h))[:, :, np.newaxis], dim_p, axis=2)  
        
        # A-Weighting for Rotational Noise
        '''A-weighted sound pressure level can be obtained by applying A(f) to the
        sound pressure level for each harmonic, then adding the results using the
        method in Appendix B.
        '''
        f  = B*omega*m/(2*np.pi) 
        
        #------------------------------------------------------------------------
        # Rotational SPL  by Barry & Magliozzi
        #------------------------------------------------------------------------
        M            = V/a                                             # Mach number
        S0           = np.sqrt(x_obs**2 + (1 - M**2)*(Y**2))           # amplitude radius    
        A_x          = 0.6853* b*t                                     # airfoil cross-sectional area
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
         
    
        # make all nan's and inf's (where observer and propeller are cooincident = 0)
        p_pref_r_BM[p_pref_r_BM==np.inf]           = 0.0
        p_pref_r_H[p_pref_r_H==np.inf]             = 0.0 
        p_pref_v[p_pref_v==np.inf]                 = 0.0 
        p_pref_r_BM_dBA[p_pref_r_BM_dBA==np.inf]   = 0.0       
        p_pref_r_H_dBA[p_pref_r_H_dBA==np.inf]     = 0.0         
        p_pref_v_dBA[p_pref_v_dBA==np.inf]         = 0.0
        p_pref_r_BM[np.isnan(p_pref_r_BM)]         = 0.0
        p_pref_r_H[np.isnan(p_pref_r_H)]           = 0.0 
        p_pref_v[np.isnan(p_pref_v)]               = 0.0 
        p_pref_r_BM_dBA[np.isnan(p_pref_r_BM_dBA)] = 0.0       
        p_pref_r_H_dBA[np.isnan(p_pref_r_H_dBA)]   = 0.0         
        p_pref_v_dBA[np.isnan(p_pref_v_dBA)]       = 0.0           
        
        
        # collecting unweighted pressure ratios     
        total_p_pref_r_BM[:,idx*(num_h):(idx+1)*(num_h)] = p_pref_r_BM
        total_p_pref_r_H[:,idx*(num_h):(idx+1)*(num_h)]  = p_pref_r_H     
        total_p_pref_v[:,idx*(6):(idx+1)*(6)]            = p_pref_v 
        
        # collecting weighted pressure ratios with vortex noise included 
        total_p_pref_BMv_dBA[:,idx*(num_h+5):(idx+1)*(num_h+5)] = np.hstack((p_pref_r_BM_dBA,p_pref_v_dBA))
        total_p_pref_Hv_dBA[:,idx*(num_h+5):(idx+1)*(num_h+5)]  = np.hstack((p_pref_r_H_dBA,p_pref_v_dBA))   
        total_p_pref_v_dBA[:,idx*(5):(idx+1)*(5)]               = p_pref_v_dBA
        
        idx += 1
    
    
    # Rotational SPL (Unweighted)    
    SPL_BM_unweighted      = np.atleast_2d(decibel_arithmetic(total_p_pref_r_BM)).T       # Barry & Magliozzi rotational noise with Schlegel vortex noise
    SPL_H_unweighted       = np.atleast_2d(decibel_arithmetic(total_p_pref_r_H)).T        # Hanson rotational noise with Schlegel vortex noise         
    SPL_v_unweighted       = np.atleast_2d(decibel_arithmetic(total_p_pref_v)).T
    
    # A- Weighted Rotational and Vortex SPL 
    SPL_BMv_dBA      = np.atleast_2d(decibel_arithmetic(total_p_pref_BMv_dBA)).T
    SPL_Hv_dBA       = np.atleast_2d(decibel_arithmetic(total_p_pref_Hv_dBA)).T
    SPL_v_dBA        = np.atleast_2d(decibel_arithmetic(total_p_pref_v_dBA)).T
    
    # make all nan's and inf's (where observer and propeller are cooincident = 0)
    SPL_BM_unweighted[SPL_BM_unweighted==np.inf]    = 0.0
    SPL_H_unweighted[SPL_H_unweighted==np.inf]      = 0.0 
    SPL_v_unweighted[SPL_v_unweighted==np.inf]      = 0.0 
    SPL_BMv_dBA[SPL_BMv_dBA==np.inf]                = 0.0       
    SPL_Hv_dBA[SPL_Hv_dBA==np.inf]                  = 0.0         
    SPL_v_dBA [SPL_v_dBA ==np.inf]                  = 0.0
    SPL_BM_unweighted[np.isnan(SPL_BM_unweighted)]  = 0.0
    SPL_H_unweighted[np.isnan(SPL_H_unweighted)]    = 0.0 
    SPL_v_unweighted[np.isnan(SPL_v_unweighted)]    = 0.0 
    SPL_BMv_dBA[np.isnan(SPL_BMv_dBA)]              = 0.0       
    SPL_Hv_dBA[np.isnan(SPL_Hv_dBA)]                = 0.0         
    SPL_v_dBA [np.isnan(SPL_v_dBA )]                = 0.0      
    
    noise_data.acoustic_results = Data()
    noise_data.acoustic_results.SPL_BM_unweighted      =  SPL_BM_unweighted  
    noise_data.acoustic_results.SPL_H_unweighted       =  SPL_H_unweighted 
    noise_data.acoustic_results.SPL_v_unweighted       =  SPL_v_unweighted  
    noise_data.acoustic_results.SPL_BMv_dBA            =  SPL_BMv_dBA       
    noise_data.acoustic_results.SPL_Hv_dBA             =  SPL_Hv_dBA        
    noise_data.acoustic_results.SPL_v_dBA              =  SPL_v_dBA
          
    return   
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