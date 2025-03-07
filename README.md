Author: Ismet Karakan
Contact: karakan@sc.edu

This code models given single channel, or channel network
with meshless RBFCM. The governing equation for the discharge
is variable porameter advection-diffusion. The governing equation
for water levels is dh/dx = S_0 - S_f.

--------------------pre process--------------------------
Create a directory for the case (eg. syntheticNetwork)
For each segment in the network, create directories under the case directory
named 'segmentx' where x is the segment id. Under each segment directory,
create a 'geo' directory to keep geometry related files and boundary files, 
and a 'run' file to feed the model with initial conditions and keep the record
of model results with specified print step.
----geo---
nodes: the position of nodes in the river segment.
slopes: slopes.
mannings_n: mannings.
xsInfo: cross section information for each node, this array directs correct cs for that node.
xs0 (or xs1, xs2): cross section data (y, x1, x2)
boundary_h: downstream water levels for the segment. If the segment has any downstream segment,
          the upstream water level of downstream segment is assigned for the downstream value
          of the upper segment. If not, downstream water levels are interpolated by this
          file. (time h)
boundary_Q: upstream discharge for the segment. If the segment has any upstream segment,
          the downstream values of discharges of the immediate upstream segments are summed,
          and assigned for the upstream discharge for the current segment. If not, the upstream
          discharge is interpolated by this file. (time, Q)
---run---
{time}/h.csv: water levels for given time.
{time}/Q.csv: discharge for given time.

---input---
This file has to stay directly under the case name. Keeps the necessary information for the model.
---network---
This file keeps the information for channel network. connections are defined as the upstream segment and
downstream segment. For example in 'syntheticNetwork' case, network is:
0 2
1 2
which implies there is a junction between segment 0 and 2, and segment 0 is the upstream segment. Similarly,
the second entry implies there is a junction between segment 1 and 2, and segment 1 is the upstream segment.
This example case is a simple Y shaped channel network.
------------------------------------------------------------------

--------------------------process---------------------------------
After setting the case completly with the steps explained above, run process.py. Make sure the caseName is entered
correctly.
-------------------------------------------------------------------

--------------------post process----------------------------------
post_process.m helps the user visualize the downstream and upstream values of each segment versus time.
---------------------------------------------------------------------
