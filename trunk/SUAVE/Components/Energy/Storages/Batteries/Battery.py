## @ingroup Components-Energy-Storages-Batteries
# Battery.py
# 
# Created:  Nov 2014, M. Vegh
# Modified: Feb 2016, T. MacDonald
# Modified: Feb 2020, M. Clarke

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

# suave imports
import SUAVE

# package imports

from SUAVE.Core                                            import Data
from SUAVE.Components.Energy.Energy_Component              import Energy_Component
from SUAVE.Methods.Power.Battery.Discharge.datta_discharge import datta_discharge
from SUAVE.Methods.Power.Battery.Discharge.chin_discharge  import chin_discharge
from SUAVE.Methods.Power.Battery.Discharge.zhang_discharge import zhang_discharge
from SUAVE.Methods.Power.Battery.Charge.datta_charge       import datta_charge
from SUAVE.Methods.Power.Battery.Charge.chin_charge        import chin_charge
from SUAVE.Methods.Power.Battery.Charge.zhang_charge       import zhang_charge

# ---------------------------------------------------------------- ------
#  Battery
# ----------------------------------------------------------------------    

## @ingroup Components-Energy-Storages-Batteries
class Battery(Energy_Component):
    """
    Energy Component object that stores energy. Contains values
    used to indicate its discharge characterics, including a model
    that calculates discharge losses
    """
    def __defaults__(self):
        self.mass_properties.mass     = 0.0
        self.energy_density           = 0.0
        self.current_energy           = 0.0
        self.initial_temperature      = 20.0
        self.current_capacitor_charge = 0.0
        self.resistance               = 0.07446 # base internal resistance of battery in ohms
        self.max_energy               = 0.0
        self.max_power                = 0.0
        self.max_voltage              = 0.0
        self.datta_discharge_model    = datta_discharge
        self.chin_discharge_model     = chin_discharge
        self.zhang_discharge_model    = zhang_discharge
        self.datta_charge_model       = datta_charge
        self.chin_charge_model        = chin_charge    
        self.zhang_charge_model       = zhang_charge        
        self.ragone                   = Data()
        self.ragone.const_1           = 0.0     # used for ragone functions; 
        self.ragone.const_2           = 0.0     # specific_power=ragone_const_1*10^(specific_energy*ragone_const_2)
        self.ragone.lower_bound       = 0.0     # lower bound specific energy for which ragone curves no longer make sense
        self.ragone.i                 = 0.0
        
    def energy_discharge(self,numerics,dischange_model = 1):
        if dischange_model == 1:
            self.datta_discharge_model(self, numerics)
        elif dischange_model == 2:
            self.chin_discharge_model(self, numerics)
        elif dischange_model == 3:
            self.zhang_discharge_model(self, numerics)            
        else:
            assert AttributeError("Model must be '1' (datta discharge model), '2' (chin discharge model) or '3' (zhang discharge model) ")
        return  
    
    def energy_charge(self,numerics,dischange_model = 1):
        if dischange_model == 1:
            self.datta_charge_model(self, numerics)
        elif dischange_model == 2:
            self.chin_charge_model(self, numerics)
        elif dischange_model == 3:
            self.zhang_charge_model(self, numerics)        
        else:
            assert AttributeError("Fidelity must be '1' (datta charge model), '2' (chin charge model) or '3' (zhang charge model) ")
        return  