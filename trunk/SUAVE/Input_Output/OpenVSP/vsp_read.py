## @ingroup Input_Output-OpenVSP
# vsp_read.py

# Created:  Jun 2018, T. St Francis
# Modified: Aug 2018, T. St Francis
#           Jan 2020, T. MacDonald

# ----------------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------------

import SUAVE
from SUAVE.Core import Units, Data
from SUAVE.Input_Output.OpenVSP import get_vsp_areas
from SUAVE.Components.Wings.Airfoils.Airfoil import Airfoil 
from SUAVE.Components.Fuselages.Fuselage import Fuselage
import vsp as vsp
import numpy as np


## @ingroup Input_Output-OpenVSP
def vsp_read(tag, units_type='SI'): 	
	"""This reads an OpenVSP vehicle geometry and writes it into a SUAVE vehicle format.
	Includes wings, fuselages, and propellers.

	Assumptions:
	1. OpenVSP vehicle is composed of conventionally shaped fuselages, wings, and propellers. 
	1a. OpenVSP fuselage: generally narrow at nose and tail, wider in center). 
	1b. Fuselage is designed in VSP as it appears in real life. That is, the VSP model does not rely on
	   superficial elements such as canopies, stacks, or additional fuselages to cover up internal lofting oddities.
	1c. This program will NOT account for multiple geometries comprising the fuselage. For example: a wingbox mounted beneath
	   is a separate geometry and will NOT be processed.
	2. Fuselage origin is located at nose. VSP file origin can be located anywhere, preferably at the forward tip
	   of the vehicle or in front (to make all X-coordinates of vehicle positive).
	3. Written for OpenVSP 3.16.1
	
	Source:
	N/A

	Inputs:
	1. A tag for an XML file in format .vsp3.
	2. Units_type set to 'SI' (default) or 'Imperial'

	Outputs:
	Writes SUAVE vehicle with these geometries from VSP:    (All values default to SI. Any other 2nd argument outputs Imperial.)
		Wings.Wing.    (* is all keys)
			origin                                  [m] in all three dimensions
			spans.projected                         [m]
			chords.root                             [m]
			chords.tip                              [m]
			aspect_ratio                            [-]
			sweeps.quarter_chord                    [radians]
			twists.root                             [radians]
			twists.tip                              [radians]
			thickness_to_chord                      [-]
			dihedral                                [radians]
			symmetric                               <boolean>
			tag                                     <string>
			areas.exposed                           [m^2]
			areas.reference                         [m^2]
			areas.wetted                            [m^2]
			Segments.
			  tag                                   <string>
			  twist                                 [radians]
			  percent_span_location                 [-]  .1 is 10%
			  root_chord_percent                    [-]  .1 is 10%
			  dihedral_outboard                     [radians]
			  sweeps.quarter_chord                  [radians]
			  thickness_to_chord                    [-]
			  airfoil                               <NACA 4-series, 6 series, or airfoil file>
			
		Fuselages.Fuselage.			
			origin                                  [m] in all three dimensions
			width                                   [m]
			lengths.
			  total                                 [m]
			  nose                                  [m]
			  tail                                  [m]
			heights.
			  maximum                               [m]
			  at_quarter_length                     [m]
			  at_three_quarters_length              [m]
			effective_diameter                      [m]
			fineness.nose                           [-] ratio of nose section length to fuselage effective diameter
			fineness.tail                           [-] ratio of tail section length to fuselage effective diameter
			areas.wetted                            [m^2]
			tag                                     <string>
			segment[].   (segments are in ordered container and callable by number)
			  vsp.shape                               [point,circle,round_rect,general_fuse,fuse_file]
			  vsp.xsec_id                             <10 digit string>
			  percent_x_location
			  percent_z_location
			  height
			  width
			  length
			  effective_diameter
			  tag
			vsp.xsec_num                              <integer of fuselage segment quantity>
			vsp.xsec_surf_id                          <10 digit string>
	
		Propellers.Propeller.
			location[X,Y,Z]                            [radians]
			rotation[X,Y,Z]                            [radians]
			tip_radius                 [m]
		        hub_radius                 [m]
			thrust_angle                               [radians]
	
	Properties Used:
	N/A
	"""  	
	
	vsp.ClearVSPModel() 
	vsp.ReadVSPFile(tag)	
	
	vsp_fuselages = []
	vsp_wings     = []	
	vsp_props     = []
	
	vsp_geoms     = vsp.FindGeoms()
	geom_names    = []

	vehicle     = SUAVE.Vehicle()
	vehicle.tag = tag

	if units_type == 'SI':
		units_type = 'SI' 
	else:
		units_type = 'Imperial' 

	# The two for-loops below are in anticipation of an OpenVSP API update with a call for GETGEOMTYPE.
	# This print function allows user to enter VSP GeomID manually as first argument in vsp_read functions.
	
	print("VSP geometry IDs: ")	
	
	# Label each geom type by storing its VSP geom ID. (The API call for GETGEOMTYPE was not released as of 8/9/18, v 3.16.1)
	
	for geom in vsp_geoms: 
		geom_name = vsp.GetGeomName(geom)
		geom_names.append(geom_name)
		print(str(geom_name) + ': ' + geom)
	
	# -----------------------------
	# MANUAL VSP ENTRY & PROCESSING
	# -----------------------------		
	
	#fuselage = read_vsp_fuselage(fuselage_id, units_type=units_type) # Replace fuselage_id manually.
	#vehicle.append_component(fuselage)
	
	#wing = read_vsp_wing(wing_id, units_type=units_type)		# Replace wing_id manually.
	#vehicle.append_component(wing)		
	
	#prop = read_vsp_prop(prop_id, units_type=units_type)		# Replace prop_id manually.	
	#vehicle.append_component(prop)
	

	# --------------------------------
	# AUTOMATIC VSP ENTRY & PROCESSING
	# --------------------------------		
		
	#for geom in vsp_geoms:
		#if vsp.GETGEOMTYPE(str(geom)) == 'FUSELAGE':
			#vsp_fuselages.append(geom)
		#if vsp.GETGEOMTYPE(str(geom)) == 'WING':
			#vsp_wings.append(geom)
		#if vsp.GETGEOMTYPE(str(geom)) == 'PROP':
			#vsp_props.append(geom)
	
	# Read VSP geoms and store in SUAVE components.
	
	#for vsp_fuselage in vsp_fuselages:
		#fuselage_id = vsp_fuselages[vsp_fuselage]
		#fuselage = read_vsp_fuselage(fuselage_id, units_type)
		#vehicle.append_component(fuselage)
	
	#for vsp_wing in vsp_wings:
		#wing_id = vsp_wings[vsp_wing]
		#wing = read_vsp_wing(wing_id, units_type)
		#vehicle.append_component(wing)		
	
	#for vsp_prop in vsp_props:
		#prop_id = vsp_props[vsp_prop]
		#prop = read_vsp_prop(prop_id, units_type)		
		#vehicle.append_component(prop)
	
	return vehicle