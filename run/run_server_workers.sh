device0='gpu0'
device1='gpu1'
device2='gpu2'
device3='gpu3'
device4='gpu4'
device5='gpu5'
device6='gpu6'
device7='gpu7'

#!/bin/bash
screen -Sdm server sh -c "source ./set4theano.sh; ./run_controller.sh; exec bash"
screen -Sdm worker0 sh -c "source ./set4theano.sh; ./run_worker.sh '$device0'; exec bash"
screen -Sdm worker1 sh -c "source ./set4theano.sh; ./run_worker.sh '$device1'; exec bash"
screen -Sdm worker2 sh -c "source ./set4theano.sh; ./run_worker.sh '$device2'; exec bash"	
screen -Sdm worker3 sh -c "source ./set4theano.sh; ./run_worker.sh '$device3'; exec bash"	
screen -Sdm worker4 sh -c "source ./set4theano.sh; ./run_worker.sh '$device4'; exec bash"
screen -Sdm worker5 sh -c "source ./set4theano.sh; ./run_worker.sh '$device5'; exec bash"
screen -Sdm worker6 sh -c "source ./set4theano.sh; ./run_worker.sh '$device6'; exec bash"
screen -Sdm worker7 sh -c "source ./set4theano.sh; ./run_worker.sh '$device7'; exec bash"
