## @ingroup Components-Propulsors
# Propulsor.py
# 
# Created:  
# Modified: Feb 2016, T. MacDonald


# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

from SUAVE.Components import Physical_Component
from SUAVE.Core import Data
# ----------------------------------------------------------------------
#  Propulsor
# ----------------------------------------------------------------------

## @ingroup Components-Propulsors
class Propulsor(Physical_Component):

    """ SUAVE.Components.Propulsor()
    
        The Top Level Propulsor Class
            
            Assumptions:
            None
            
            Source:
            N/As
    
    """

    def __defaults__(self):
        
        """ This sets the default attributes for the propulsor.
        
                Assumptions:
                None
                
                Source:
                N/A
                
                Inputs:
                None
                
                Outputs:
                None
                
                Properties Used:
                N/A
        """
        self.tag = 'Propulsor'
        self.number_of_engines = 1.0
        self.nacelle_diameter  = 1.0
        self.engine_length     = 1.0
        
        self.areas             = Data()
        self.areas.wetted      = 0.0
        self.areas.maximum     = 0.0
        self.areas.exit        = 0.0
        self.areas.inflow      = 0.0
        
## @ingroup Components-Propulsors
class Container(Physical_Component.Container):
    """ SUAVE.Components.Propulsor.Container()
        
        The Propulsor Container Class
    
            Assumptions:
            None
            
            Source:
            N/A
    
    """
    pass
    
    def evaluate_thrust(self,state):
        """ This is used to evaluate the thrust produced by the propulsor.
        
                Assumptions:
                Propulsor has "evaluate_thrust" method
                
                Source:
                N/A
                
                Inputs:
                State variables
                
                Outputs:
                Results of the "evaluate_thrust" method
                
                Properties Used:
                N/A
        """

        for propulsor in self.values():
            results = propulsor.evaluate_thrust(state) 
            
        return results

# ----------------------------------------------------------------------
#  Handle Linking
# ----------------------------------------------------------------------

Propulsor.Container = Container