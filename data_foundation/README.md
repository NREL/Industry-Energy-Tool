# County-Level Industrial Energy Data
One of the most significant impediments to analyzing and building models of
U.S. industrial energy use is the lack of current disaggregated data that
reflect the sector’s heterogeneity and complexity. The IET's foundational data
estimates industrial energy use by county for 2014. The approach for each
subsector is summarized here; for a more detailed explanation please see
[McMillan and Narwade (2018)](https://www.google.com). The data set is
available from the [NREL Data Catalog](https://data.nrel.gov).

## Agriculture
The overall approach to calculate county-level agriculture energy use is
separated into liquid fuels (i.e., diesel, LPG, gasoline, and “other”) and
electricity. Liquid fuel use is estimated by first calculating the liquid fuel
fraction of total fuel and lubricants expenses by region from 2014 USDA
agricultural survey data (USDA National Agricultural Statistics Service 2018).
These fractions are then multiplied by the state-level 2012 total fuel and
lubricants expenses by four- to six-digit NAICS code from the Census of
Agriculture (United States Department of Agriculture 2017) and adjusted based on
the ratio of 2014:2012 fuel expenses by region.

State-level electricity expenses by four- and six-digit NAICS code are
calculated by multiplying 2014 state-level electricity expenses
(USDA Economic Research Service 2018) by the state-level fraction of 2012
utility expenditures by four- to six-digit NAICS code from the 2012 Census of
Agriculture.

Liquid fuel and electricity expenditures are converted into energy units using
2014 EIA state- and regional-level energy prices (EIA 2017d, 2017d).
State energy totals are then apportioned to the county-level using farm counts
by four- to six-digit NAICS code from the 2012 Census of Agriculture.

## Construction
The overall calculation approach is to first estimate state-level construction
energy use derived from 2012 Economic Census data and to then allocate to the
county-level using CBP establishment counts. State-level calculations begin by
using IPF to derive “Cost of materials, parts, supplies, electricity, and fuels
($1,000)” by three-digit NAICS code, employment size class, and state from the
2012 Economic Census. The fuel cost fraction of “Cost of materials, parts,
supplies, electricity, and fuels ($1,000)”  is then calculated for each fuel
type by state and three-digit NAICS. The energy associated with asphalt use is
not captured in the calculation. “Other fuels and lubricants” are assumed to be
LPG.

The fuel cost fraction is multiplied by the results of the IPF to calculate fuel
type expenses by three-digit NAICS code, employment size class, and state.
These fuel type expenses are then multiplied by EIA 2012 state- or Petroleum
Administration for Defense District-level fuel price per million Btu. The
compound annual growth rate from 2012 to 2014 of construction GDP by state is
then used to calculate energy values in 2014. Energy values are then allocated
to the county-level using CBP establishment counts by three-digit NAICS code and
employment size class.

## Manufacturing
The method used by the IET to calculate county-level energy use for
manufacturing industries is based on (1) county counts of manufacturing
establishments by industry and employment size from Census County Business
Patterns (CBP) data (U.S. Census Bureau 2016) and (2) either facility-level
energy data or regional-average energy data. Facility-level estimates of energy
use are calculated for large energy-users using EPA Greenhouse Gas Reporting
Program (GHGRP) emissions data (EPA 2017) and electricity use reported by
industry generators above 1 MW from Form EIA-923 (EIA 2017b). Regional-average
energy intensity values (energy per establishment) are estimated by NAICS code,
fuel type, and employment size class using data derived from
2010 MECS (EIA 2013) and iterative proportional fitting.

In general, the method first identifies the count of GHGRP- and
Form EIA-923-reporting facilities by county and then subtracts their number from
the CBP establishment counts. Energy data for the GHGRP reporters are estimated
using reported emissions data and Form EIA-923 electricity data. Energy use
for the remaining establishments is calculated based on NAICS code, fuel type,
and employment size class using the derived regional-average energy intensity
values.

### Facility-Level Combustion Energy and Net Electricity Calculations
Facilities with annual GHG emissions that exceed 25,000 metric tons carbon
dioxide-equivalent (MMTCO2e) are required to report emissions under the GHGRP
(Mandatory Greenhouse Gas Reporting 2009). GHG emissions reported under Subpart
C General Stationary Combustion Sources and Subpart D Electricity Generation
are used with average GHG emission factors by fuel type (EPA 2018) to
back-calculate facility combustion energy use for relevant manufacturing,
mining, and agriculture facilities.

A crosswalk table of electricity data is created based on facility name and
location for facilities reporting to the GHGRP and Form EIA-923. Note that
Form EIA-923 provides electricity use data only for industrial facilities with
onsite generation—cogenerating or non-cogenerating—exceeding 1 MW. Net
electricity use for these facilities is calculated as the sum of incoming
electricity and generation from noncombustible renewable sources, less the sum
of retail sales, sales for resale (i.e., wholesale sales), tolling agreements,
and all other outgoing electricity. The net electricity use for GHGRP-reporting
facilities that are not listed on Form EIA-923 is calculated using
regional-average net electricity intensities, which are described in the next
section.

### Regional-Average Energy Intensity Approach
Regional-average energy intensities per manufacturing establishment are used to
calculate the combustion energy use of establishments that do not report under
the GHGRP. Likewise, regional-average net electricity use intensities are
calculated to estimate electricity use of establishments that do not report to
EIA Form-923. This section describes the process of first deriving average
energy values by census region, fuel type, NAICS code, and employment size class
using IPF and 2010 MECS data. CBP establishment counts by NAICS code and
employment size class are then used to calculated energy intensity values.
Finally, these per-establishment energy intensities are multiplied by adjusted
CBP establishment counts to estimate the energy use of facilities that do not
report under the GHGRP or Form EIA-923. This final step involves portioning the
number of CBP-reported manufacturing establishments into GHGRP-reporting and
GHGRP non-reporting facilities within a given county to avoid double-counting
energy use.

## Mining
As with the manufacturing energy calculations, mining energy use is calculated
using a facility-level approach and an average intensity approach.
Facility-level detail for combustion fuels is provided by GHGRP data for large
emitters; facility-level detail for net electricity use is provided by the EIA
in Form EIA-923 for covered establishments. National-average energy intensities
are calculated by fuel type and six-digit NAICS code from 2012 Economic Census
data-adjusted CBP establishment counts by county and six-digit NAICS code. The
two approaches are summarized in Figure 4 (page 25).

The 2012 Economic Census data provide detailed national-level delivered cost
data by fuel type by six-digit NAICS code; use of certain fuels in physical
quantities is also provided for several mining industries. Cost data that had
been withheld were estimated based on total fuel costs, where appropriate. As a
result, the fuel cost data by fuel type capture over 95% of the total delivered
fuel costs for most of the mining industries. Fuel use in energy units is then
calculated by multiplying fuel cost data by their 2012 national-average price
and heat of combustion value. Data on electricity purchased for heat and power
(in 1,000 kWh) is provided separately by the 2012 Economic Census but is
likewise available at the six-digit NAICS level for the United States. Fuel and
electricity use in 2014 is calculated by scaling 2012 values using the 2012–2014
change in physical production data from USGS mineral and commodity data
(USGS 2016) by six-digit NAICS code.

National-level 2014 energy use data are allocated to the county-level using 2014
CBP establishment counts data by six-digit NAICS code. These establishment
counts are adjusted by subtracting the number of GHGRP-reporting facilities in
the mining sector, as the fuel use for these facilities is calculated separately
from reported emissions. County-level fuel and electricity estimates by
six-digit NAICS code are then calculated by multiplying the county fraction of
total mining establishment count by the estimated national-level energy use
value. These values are then combined with the results of the GHGRP-based
combustion fuel and EIA Form-923 electricity calculations for mining industries
using the method described above for manufacturing.
