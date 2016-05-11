#### device 1
if [[ -z $1 ]]; then
	echo 'need a device1 as argument $1, gpu0, gpu1 ...'
	exit 1
else
	device1=$1
fi

if [[ ${device1:0:3} == "gpu" ]]; then
	
	dev1=${device1#gpu}
else
	echo 'device1 starts with *gpu* '
	exit 1
fi

if [[ $dev1 -ge '4' ]]; then
	numa1=1
else
	numa1=0
fi

echo 'numa1:' $numa1 'device1:' $1

numactl -N $numa1 python -u ../exc/convnet_worker.py --d $1

