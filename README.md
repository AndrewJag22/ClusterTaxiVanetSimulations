# ClusterTaxiVanetSimulations
Taxi-VANET system level simulator, designed to run on Oracle-Linux cluster

Unfortunately this is highly specilised code for running VANET simulations across a cluster of nodes.
It has been optimised to run using just 2-3GB of memory.

In order to run, code needs to be compiled then submitted as a job. Each job takes the chosen sink_id string to act as the central server which all other sources send messages to.

All code runs in Python3+, we use Pyinstaller to compile code.
