## @ingroup Components-Energy-Converters
# Axisymmetric_Inlet.py
#
# Created:  July 2019, M. Dethy

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE

# python imports
from warnings import warn

# package imports
import numpy as np

from SUAVE.Components.Energy.Energy_Component import Energy_Component
from SUAVE.Methods.Aerodynamics.Common.Gas_Dynamics import Oblique_Shock, Isentropic, Conical_Shock

# ----------------------------------------------------------------------
#  Axisymmetric Inlet Component
# ----------------------------------------------------------------------
## @ingroup Components-Energy-Converters
class Axisymmetric_Inlet(Energy_Component):
    """This is a two dimensional inlet component intended for use in compression.
    Calling this class calls the compute function.

    Source:
    https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/
    """

    def __defaults__(self):
        """This sets the default values for the component to function.

        Assumptions:
        None

        Source:
        N/A

        Inputs:
        None

        Outputs:
        None

        Properties Used:
        None
        """
        # setting the default values
        self.tag = 'axisymmetric_inlet'
        self.areas                           = Data()
        self.areas.capture                   = 0.0
        self.areas.throat                    = 0.0
        self.areas.inlet_entrance            = 0.0
        self.areas.drag_direct_projection    = 0.0
        self.angles                          = Data()
        self.angles.cone_half_angle          = 0.0
        self.inputs.stagnation_temperature   = np.array([0.0])
        self.inputs.stagnation_pressure      = np.array([0.0])
        self.outputs.stagnation_temperature  = np.array([0.0])
        self.outputs.stagnation_pressure     = np.array([0.0])
        self.outputs.stagnation_enthalpy     = np.array([0.0])

    def compute(self, conditions):
        
        """ This computes the output values from the input values according to
        equations from the source.

        Assumptions:
        Constant polytropic efficiency and pressure ratio
        Adiabatic

        Source:
        https://web.stanford.edu/~cantwell/AA283_Course_Material/AA283_Course_Notes/

        Inputs:
        conditions.freestream.
          isentropic_expansion_factor         [-]
          specific_heat_at_constant_pressure  [J/(kg K)]
          pressure                            [Pa]
          gas_specific_constant               [J/(kg K)]
        self.inputs.
          stagnation_temperature              [K]
          stagnation_pressure                 [Pa]

        Outputs:
        self.outputs.
          stagnation_temperature              [K]
          stagnation_pressure                 [Pa]
          stagnation_enthalpy                 [J/kg]
          mach_number                         [-]
          static_temperature                  [K]
          static_enthalpy                     [J/kg]
          velocity                            [m/s]

        Properties Used:
        self.
          pressure_ratio                      [-]
          polytropic_efficiency               [-]
          pressure_recovery                   [-]
        """

        # unpack from conditions
        gamma = conditions.freestream.isentropic_expansion_factor
        Cp = conditions.freestream.specific_heat_at_constant_pressure
        P0 = conditions.freestream.pressure
        M0 = conditions.freestream.mach_number
        R = conditions.freestream.gas_specific_constant

        # unpack from inputs
        Tt_in = self.inputs.stagnation_temperature
        Pt_in = self.inputs.stagnation_pressure

        # unpack from self
        A0 = conditions.area_initial_freestream
        AE = self.areas.capture # engine face area
        AC = self.areas.throat # narrowest part of inlet
        theta = self.angles.cone_half_angle # incoming angle for the shock in degrees
        
        # Compute the mass flow rate into the engine
        T               = Isentropic.isentropic_relations(M0, gamma)[0]*Tt_in
        v               = np.sqrt(gamma*R*T)*M0
        mass_flow_rate  = conditions.freestream.density * A0 * v
        q_0             = 1/2 * conditions.freestream.density * v**2

        f_M0            = Isentropic.isentropic_relations(M0, gamma)[-1]
        f_ME_isentropic = (f_M0 * A0)/AE
        i_sub           = M0 <= 1.0
        i_sup           = M0 > 1.0
        
        # This 
        if len(Pt_in) == 1:
            Pt_in = Pt_in[0]*np.ones_like(M0)
        if len(Tt_in) == 1:
            Tt_in = Tt_in[0]*np.ones_like(M0)
        
        # initializing the arrays
        Tt_out          = Tt_in
        ht_out          = Cp*Tt_in
        Pt_out          = np.ones_like(Pt_in)
        Mach            = np.ones_like(Pt_in)
        T_out           = np.ones_like(Pt_in)
        f_ME            = np.ones_like(Pt_in)
        MC              = np.ones_like(Pt_in)
        Ms              = np.ones_like(Pt_in)
        beta            = np.ones_like(Pt_in)
        MC_wedge        = np.ones_like(Pt_in)
        Pr_c            = np.ones_like(Pt_in)
        Tr_c            = np.ones_like(Pt_in)
        Pt_th           = np.ones_like(Pt_in)
        f_MC            = np.ones_like(Pt_in)
        Pt_1_ov_Pt_th   = np.ones_like(Pt_in)

        # Conservation of mass properties to evaluate subsonic case
        Pt_out[i_sub]   = Pt_in[i_sub]
        f_ME[i_sub]     = f_ME_isentropic[i_sub]
        Mach[i_sub]     = Isentropic.get_m(f_ME[i_sub], gamma, 1)
        T_out[i_sub]    = Isentropic.isentropic_relations(Mach[i_sub], gamma)[0]*Tt_out[i_sub]
        
        # Analysis of shocks for the supersonic case
        Ms[i_sup]       = Conical_Shock.get_Ms(M0[i_sup], theta/2)
        beta[i_sup]     = Conical_Shock.get_beta(M0[i_sup], theta)
        MC_wedge[i_sup] = Oblique_Shock.oblique_shock_relations(M0[i_sup],gamma,theta,beta)[0]
        MC              = 0.5 * (Ms[i_sup] + MC_wedge[i_sup])
        
        Pt_th[i_sup]         = Conical_Shock.get_Cp(Ms[i_sup], theta)*q0 + P0
        Pt_1_ov_Pt_th[i_sup] = Oblique_Shock.oblique_shock_relations(MC[i_sup],gamma,0,90)[3]
        Pt_out[i_sup]        = Pt_th[i_sup] * Pt_1_ov_Pt_th[i_sup]
        
        f_MC[i_sup] = Isentropic.isentropic_relations(MC[i_sup], gamma)[-1]
        f_ME[i_sup] = f_MC[i_sup]*AC/AE
        
        Mach[i_sup] = Isentropic.get_m(f_ME[i_sup], gamma, 1)
        T_out[i_sup] = Isentropic.isentropic_relations(Mach[i_sup], gamma)[0]*Tt_out[i_sup]
        
        # -- Compute exit velocity and enthalpy
        h_out = Cp * T_out
        u_out = np.sqrt(2. * (ht_out - h_out))

        # pack computed quantities into outputs
        self.outputs.stagnation_temperature = Tt_out
        self.outputs.stagnation_pressure = Pt_out
        self.outputs.stagnation_enthalpy = ht_out
        self.outputs.mach_number = Mach
        self.outputs.static_temperature = T_out
        self.outputs.static_enthalpy = h_out
        self.outputs.velocity = u_out
        conditions.mass_flow_rate = mass_flow_rate
        
    def _compute_drag(self, conditions):

        '''
        Nomenclature/labeling of this section is inconsistent with the above
        but is consistent with Nikolai's methodology as presented in aircraft
        design
        '''
        
        # Unpack constants from freestream conditions
        gamma       = conditions.freestream.isentropic_expansion_factor
        R           = conditions.freestream.gas_specific_constant
        P_inf       = conditions.freestream.pressure
        M_inf       = conditions.freestream.mach_number
        rho_inf     = conditions.freestream.density
        
        # unpack from inputs
        Tt_inf = self.inputs.stagnation_temperature
        Pt_inf = self.inputs.stagnation_pressure
        
        # compute relevant freestream quantities
        T_inf  = Isentropic.isentropic_relations(M_inf, gamma)[0] * Tt_inf
        v_inf  = np.sqrt(gamma*R*T_inf) * M_inf
        q_inf  = 1/2 * rho_inf * v_inf**2
        f_Minf = Isentropic.isentropic_relations(M_inf, gamma)[-1]
        
        # unpack from self
        A_inf = conditions.area_initial_freestream
        AC    = self.areas.capture # engine face area
        A1    = self.areas.inlet_entrance # area of the inlet entrance
        theta = self.angles.cone_angle # cone half angle of the inlet
        AS    = self.areas.drag_direct_projection
        
        # compute A1 quantities
        i_sub           = M_inf <= 1.0
        i_sup           = M_inf > 1.0
        
        # initialize values
        f_M1 = np.ones_like(Tt_inf)
        Pr_1 = np.ones_like(Tt_inf)
        P1   = np.ones_like(Tt_inf)
        M1   = np.ones_like(Tt_inf)
        
        # subsonic case
        f_M1[i_sub]      = (f_Minf[i_sub] * A_inf[i_sub])/A1
        M1[i_sub]        = Isentropic.get_m(f_M1[i_sub], gamma, 1)
        P1[i_sub]        = Isentropic.isentropic_relations(M1[i_sub], gamma)[1] * Pt_inf[i_sub]
        
        # supersonic case
        M1[i_sup], Pr_1[i_sup] = Oblique_Shock.oblique_shock_relations(M_inf[i_sup],gamma,0,theta)[0:2]
        P1[i_sup]              = Pr_1[i_sup]*P_inf[i_sup]
        
        # exposed area related drag
        Ps_ov_Pinf = Conical_Shock.get_invisc_press_recov(theta, M1)
        C_ps       = 2/(gamma*M_inf**2) * (Ps_ov_Pinf-1)
        
        CD_add = (P_inf/q_inf) * (A1/AC) * np.cos(theta/180*np.pi)((P1/P_inf)*(1+gamma*M1**2)-1) - 2*(A_inf/AC) + C_ps+(AS/AC)
        
        if M_inf < 1.4:
            if M_inf >= 0.7 and M_inf <= 0.9:
                	c1_fit = [-10.55390326, 15.71708277, -5.23617066]
                	c2_fit = [16.36281692, -24.54266271, 7.4994281]
                	c3_fit = [-4.86319239, 7.59775242, -1.85372994]
            elif M_inf > 0.9 and M_inf <= 1.1:
                	c1_fit = [2.64544806e-17, 3.60542191e-01]
                	c2_fit = [1.57079398e-16, -1.33508664e+00]
                	c3_fit = [-7.8265315e-16, 1.0450614e+00]
            else:
                c1_fit = [-102.15032982, 403.09453072, -527.81008066, 229.16933773]
                c2_fit = [134.93205478, -539.18500576, 716.8828252, -317.08690229]
                c3_fit = [-29.74762681, 122.74408883, -166.89910445, 75.70782011]
                
            c1 = np.polyval(c1_fit, M_inf)
            c2 = np.polyval(c2_fit, M_inf)
            c3 = np.polyval(c3_fit, M_inf)
                
            # Use coefficients on theta_c to get the pressure recovery
            fit   = [c1, c2, c3]
            K_add = np.polyval(fit, A_inf/AC)
        
        else:
            K_add = 1
        
        D_add  = CD_add * q_inf* AC * K_add
        
        return D_add
    

    
